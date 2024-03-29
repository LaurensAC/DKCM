import argparse
import os

import numpy as np
import math
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import List

IMG_RES = 224
IN_CHANNELS = 25
TL = 80
N_TRAJS = 6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data", type=str, required=True, help="Path to rasterized data"
    )
    parser.add_argument(
        "--dev-data", type=str, required=True, help="Path to rasterized data"
    )
    parser.add_argument(
        "--img-res",
        type=int,
        required=False,
        default=IMG_RES,
        help="Input images resolution",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        required=False,
        default=IN_CHANNELS,
        help="Input raster channels",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        required=False,
        default=TL,
        help="Number time step to predict",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        required=False,
        default=N_TRAJS,
        help="Number of trajectories to predict",
    )
    parser.add_argument(
        "--save", type=str, required=True, help="Path to save model and logs"
    )

    parser.add_argument(
        "--model", type=str, required=False, default="xception71", help="CNN model name"
    )
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--batch-size", type=int, required=False, default=48)
    parser.add_argument("--n-epochs", type=int, required=False, default=60)

    parser.add_argument("--valid-limit", type=int, required=False, default=24 * 100)
    parser.add_argument(
        "--n-monitor-train",
        type=int,
        required=False,
        default=10,
        help="Validate model each `n-validate` steps",
    )
    parser.add_argument(
        "--n-monitor-validate",
        type=int,
        required=False,
        default=1000,
        help="Validate model each `n-validate` steps",
    )

    args = parser.parse_args()

    return args


class Model(nn.Module):
    def __init__(
        self, model_name, in_channels=IN_CHANNELS, time_limit=TL, n_traj=N_TRAJS
    ):
        super().__init__()

        self.n_traj = n_traj
        self.time_limit = time_limit
        self.dt = 0.1  # time step
        self.max_steering_angle = 37 * math.pi / 180  # 0.64577 radians
        self.min_steering_angle = -37 * math.pi / 180  # -0.64577 radians
        self.min_acceleration = -2.001  # forcing float
        self.max_acceleration = 2.001  # forcing float
        self.model = timm.create_model(  # set backbone CNN model
            model_name,
            pretrained=True,
            in_chans=in_channels,
            num_classes=self.n_traj * 2 * self.time_limit + self.n_traj,
        )

    def forward(self, x, current_state: List[torch.Tensor]):
        outputs = self.model(x)
        self.batch_size = len(current_state[2])  # fix for inference 1 sample
        confidences_logits, logits = (
            outputs[:, : self.n_traj],
            outputs[:, self.n_traj :],  #
        )
        logits = logits.view(-1, self.n_traj, self.time_limit, 2)
        logits = logits.cpu()

        # TODO: optimize for torch
        temp_logits = torch.empty(self.batch_size, self.n_traj, self.time_limit, 2)
        for bs in range(self.batch_size):
            for i in range(0, self.n_traj):
                # set initial bicycle parameters
                if self.batch_size > 1:  # if training
                    velocity = current_state[2][bs]
                    self.distance_lr = (current_state[1][bs]) * 0.6 * 0.4
                    self.distance_lf = (current_state[1][bs]) * 0.6 * 0.6
                else:  # inference
                    velocity = current_state[2]
                    self.distance_lr = (current_state[1]) * 0.6 * 0.4
                    self.distance_lf = (current_state[1]) * 0.6 * 0.6

                steering_angle = logits[bs][i][0][0]  # set initial steering_angle
                acceleration = logits[bs][i][0][1]  # set initial acceleration

                # initialize Kinematic Bicycle model
                heading = torch.tensor(0.0)
                x = torch.tensor(0.0)
                y = torch.tensor(0.0)
                xs = []
                ys = []

                # Kinematic layer
                for j in range(0, self.time_limit):

                    # Level 0: don't clip steering_angle
                    # steering_angle = logits[bs][i][j][0]

                    # Level 1: clip steering_angle in feasible range (37,-37)
                    steering_angle = torch.clamp(
                        logits[bs][i][j][0],
                        self.min_steering_angle,
                        self.max_steering_angle,
                    )

                    # Level 2: clip steering_angle in feasible range (37,-37) and limit rate of change (15degree/0.1s)
                    # steering_angle = torch.clamp(
                    #     logits[bs][i][j][0],
                    #     float(max(self.min_steering_angle, steering_angle - 0.262)),
                    #     float(min(self.max_steering_angle, steering_angle + 0.262)),
                    # )

                    # Level 0: don't clip acceleration
                    # acceleration = logits[bs][i][j][1]

                    # Level 1: clip acceleration in feasible range (-1.5,1)
                    acceleration = torch.clamp(
                        logits[bs][i][j][1],
                        self.min_acceleration,
                        self.max_acceleration,
                    )

                    # Level 2: clip acceleration in feasible range (-1.5,1) and limit rate of change (-3/s,1.5/s)
                    # acceleration = torch.clamp(
                    #     logits[bs][i][j][1],
                    #     float(max(self.min_acceleration, acceleration - 0.3)),
                    #     float(min(self.max_acceleration, acceleration + 0.15)),
                    # )

                    # Kinematic Bicycle Model ##################################
                    Beta = math.atan(
                        (self.distance_lr / (self.distance_lr + self.distance_lf))
                        * math.tan(steering_angle)
                    )

                    velocity = velocity + acceleration * self.dt

                    heading = (
                        heading
                        + (velocity / self.distance_lr) * math.sin(Beta) * self.dt
                    )

                    x = x + velocity * math.cos(heading + Beta) * self.dt
                    xs.append(x)
                    y = y + velocity * math.sin(heading + Beta) * self.dt
                    ys.append(y)
                    ############################################################

                xs = torch.stack(xs)
                ys = torch.stack(ys)
                preds = torch.column_stack((xs, ys))
                temp_logits[bs][i] = preds

        logits = temp_logits.reshape(
            self.batch_size, self.n_traj, self.time_limit, 2
        ).cuda()

        return confidences_logits, logits


def pytorch_neg_multi_log_likelihood_batch(gt, logits, confidences, avails):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        logits (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(
        ((gt - logits) * avails) ** 2, dim=-1
    )  # reduce coords and use availability

    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
            error, dim=-1
        )  # reduce time

    # error (batch_size, num_modes)
    error = -torch.logsumexp(error, dim=-1, keepdim=True)

    return torch.mean(error)


# Custom Deep Kinematic Loss (not used)
def pytorch_dkm_loss(gt, logits, confidences, avails):
    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes

    error = torch.sqrt(torch.sum((gt - logits) ** 2, dim=-1))
    error, indices = torch.min(error, dim=-2)
    confidences = nn.functional.log_softmax(confidences, dim=1)
    mode_probability = torch.gather(confidences, -1, indices)
    error = avails * (error - 1 * mode_probability)
    return torch.sum(error) / torch.sum(avails)


class WaymoLoader(Dataset):
    def __init__(self, directory, limit=0, return_vector=False, is_test=False):
        files = os.listdir(directory)
        self.files = [os.path.join(directory, f) for f in files if f.endswith(".npz")]

        if limit > 0:
            self.files = self.files[:limit]
        else:
            self.files = sorted(self.files)

        self.return_vector = return_vector
        self.is_test = is_test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        data = np.load(filename, allow_pickle=True)
        raster = data["raster"].astype("float32")
        raster = raster.transpose(2, 1, 0) / 255

        # inference fix
        try:
            id = int(filename.split("_")[3])
        except:
            id = int(filename.split("_")[2])

        trajectory = data["gt_marginal"]

        is_available = data["future_val_marginal"]

        # fetch current heading,speed and length
        current_dict = [
            data["current_heading"][id][0],
            data["current_length"][id][0],
            data["current_speed"][id][0],
        ]

        if self.is_test:
            center = data["shift"]
            yaw = data["yaw"]
            agent_id = data["object_id"]
            scenario_id = data["scenario_id"]

            return (
                raster,
                center,
                yaw,
                agent_id,
                str(scenario_id),
                data["_gt_marginal"],
                data["gt_marginal"],
            )

        if self.return_vector:
            return raster, trajectory, is_available, data["vector_data"], current_dict

        return raster, trajectory, is_available, current_dict


def main():
    args = parse_args()

    summary_writer = SummaryWriter(os.path.join(args.save, "logs"))

    train_path = args.train_data
    dev_path = args.dev_data
    path_to_save = args.save
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    dataset = WaymoLoader(train_path)

    batch_size = args.batch_size
    num_workers = min(1, batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
    )

    val_dataset = WaymoLoader(dev_path, limit=args.valid_limit)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
    )

    model_name = args.model
    time_limit = args.time_limit
    n_traj = args.n_traj
    model = Model(
        model_name, in_channels=args.in_channels, time_limit=time_limit, n_traj=n_traj
    )
    model.cuda()

    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2 * len(dataloader),
        T_mult=1,
        eta_min=max(1e-2 * lr, 1e-6),
        last_epoch=-1,
    )

    start_iter = 0
    best_loss = float("+inf")
    glosses = []

    tr_it = iter(dataloader)
    n_epochs = args.n_epochs
    progress_bar = tqdm(range(start_iter, len(dataloader) * n_epochs))

    saver = lambda name: torch.save(
        {
            "score": best_loss,
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss.item(),
        },
        os.path.join(path_to_save, name),
    )

    for iteration in progress_bar:
        model.train()
        try:
            x, y, is_available, current_state = next(tr_it)
        except StopIteration:
            tr_it = iter(dataloader)
            x, y, is_available, current_state = next(tr_it)

        x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))
        optimizer.zero_grad()

        confidences_logits, logits = model(x, current_state)

        # add noise
        # logits = logits + torch.mul((0.01**0.5)*torch.randn(8, 6, 80, 2).cuda(),
        # torch.column_stack((torch.arange(1, 10, 0.1125).cuda(),torch.arange(1, 10, 0.1125).cuda())))

        loss = pytorch_neg_multi_log_likelihood_batch(
            y, logits, confidences_logits, is_available
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        glosses.append(loss.item())
        if (iteration + 1) % args.n_monitor_train == 0:
            progress_bar.set_description(
                f"loss: {loss.item():.3}"
                f" avg: {np.mean(glosses[-100:]):.2}"
                f" {scheduler.get_last_lr()[-1]:.3}"
            )
            summary_writer.add_scalar("train/loss", loss.item(), iteration)
            summary_writer.add_scalar("lr", scheduler.get_last_lr()[-1], iteration)

        if (iteration + 1) % args.n_monitor_validate == 0:
            optimizer.zero_grad()
            model.eval()

            with torch.no_grad():
                val_losses = []
                for x, y, is_available, current_state in val_dataloader:
                    x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))

                    confidences_logits, logits = model(x, current_state)
                    loss = pytorch_neg_multi_log_likelihood_batch(
                        y, logits, confidences_logits, is_available
                    )
                    val_losses.append(loss.item())
                print("Validation loss: " + str(np.mean(val_losses)))
                summary_writer.add_scalar("dev/loss", np.mean(val_losses), iteration)

            saver("model_last.pth")

            mean_val_loss = np.mean(val_losses)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                saver("model_best.pth")

                model.eval()
                with torch.no_grad():
                    traced_model = torch.jit.script(
                        model,
                        (
                            torch.rand(
                                1, args.in_channels, args.img_res, args.img_res
                            ).cuda(),
                            current_state,
                        ),
                    )

                traced_model.save(os.path.join(path_to_save, "model_best.pt"))
                del traced_model


if __name__ == "__main__":
    main()
