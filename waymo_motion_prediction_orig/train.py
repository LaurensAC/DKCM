import os

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

IN_CHANNELS = 47  # 91  # 25
TL = 80
N_TRAJS = 8
RESIZE = torchvision.transforms.Resize(224)


def get_model(model_name, in_ch=IN_CHANNELS):
    if model_name.startswith("efficientnet"):
        model = EfficientNet.from_pretrained(
            "efficientnet-b3",
            in_channels=in_ch,
            num_classes=N_TRAJS * 2 * TL + N_TRAJS,
        )
    else:
        model = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=in_ch,
            num_classes=N_TRAJS * 2 * TL + N_TRAJS,
        )

    return model


def pytorch_neg_multi_log_likelihood_batch(gt, pred, confidences, avails):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
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
        ((gt - pred) * avails) ** 2, dim=-1
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


class WaymoLoader(Dataset):
    def __init__(self, directory, limit=False, return_vector=False):
        files = os.listdir(directory)
        self.files = [directory + f for f in files if f.split(".")[-1] == "npz"]
        if limit:
            self.files = self.files[: 24 * 100]
        else:
            self.files = sorted(self.files)

        self.return_vector = return_vector

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        data = np.load(filename, allow_pickle=True)

        raster = data["raster"]
        raster = torch.tensor(raster.transpose(2, 1, 0)) / 255.0
        # raster = RESIZE(raster)

        trajectory = data["gt_marginal"]
        trajectory = torch.tensor(trajectory)

        valid = torch.tensor(data["future_val_marginal"])

        if self.return_vector:
            return raster, trajectory, valid, data["vector_data"]

        return raster, trajectory, valid


def main():
    data_path = "/home/brodt/kaggle/waymo"
    train_path = f"{data_path}/train/"
    dev_path = f"{data_path}/dev/"
    PATH_TO_SAVE = "/home/brodt/kaggle/waymo"

    dataset = WaymoLoader(train_path)

    df = pd.read_csv(f"{train_path}/loss_index.csv")
    df.sort_values("filename", inplace=True)
    # for item, filename in zip(df.itertuples(), dataset.files):
    # assert item.filename == filename.split("/")[-1]

    weights = df.mse.values.astype("float32")
    print(weights.min(), weights.max(), weights.mean())
    weights = np.sqrt(weights)
    print(weights.min(), weights.max(), weights.mean())
    weights = np.sqrt(weights)
    print(weights.min(), weights.max(), weights.mean())
    weights = np.clip(weights, 1.0, 4.0)
    print(weights.min(), weights.max(), weights.mean())
    # import matplotlib.pyplot as plt
    # plt.hist(weights, bins=100)
    # plt.savefig('hist.png')
    # raise
    weights = torch.from_numpy(weights)
    sampler = None
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    batch_size = 16
    num_workers = min(16, batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size * 4,
        shuffle=sampler is None,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
    )

    val_dataset = WaymoLoader(dev_path, limit=True)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size * 8,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model_name = "xception71"  # "seresnext50_32x4d"  #
    model = get_model(model_name).cuda()
    lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2 * len(dataloader),
        T_mult=1,
        eta_min=1e-2 * lr,
        last_epoch=-1,
    )

    start_iter = 0
    # chkp = torch.load("../seresnext50_32x4d_35000_val_460.885.pth")
    # model.load_state_dict(chkp["model_state_dict"])
    # optimizer.load_state_dict(chkp["optimizer_state_dict"])
    # scheduler.load_state_dict(chkp["scheduler_state_dict"])
    # start_iter = chkp["iteration"] + 1

    glosses = []

    tr_it = iter(dataloader)
    n_epochs = 120
    progress_bar = tqdm(range(start_iter, len(dataloader) * n_epochs))

    for iteration in progress_bar:
        model.train()
        try:
            RASTER, TRAJ, VALID = next(tr_it)
        except StopIteration:
            tr_it = iter(dataloader)
            RASTER, TRAJ, VALID = next(tr_it)

        optimizer.zero_grad()

        outputs = model(RASTER.cuda())
        confidences, pred = outputs[:, :N_TRAJS], outputs[:, N_TRAJS:]
        bs = RASTER.shape[0]
        pred = pred.view(bs, N_TRAJS, TL, 2)

        loss = pytorch_neg_multi_log_likelihood_batch(
            TRAJ.cuda(), pred, confidences, VALID.cuda()
        )
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        scheduler.step()

        glosses.append(loss.item())
        if iteration % 10 == 0:
            progress_bar.set_description(
                f"loss: {loss.item():.3}"
                f" avg: {np.mean(glosses[-100:]):.2}"
                f" {scheduler.get_last_lr()[-1]:.3}"
            )
        if iteration % 1000 == 0:
            optimizer.zero_grad()
            model.eval()
            with torch.no_grad():
                val_losses = []
                for RASTER, TRAJ, VALID in val_dataloader:
                    outputs = model(RASTER.cuda())
                    confidences, pred = outputs[:, :N_TRAJS], outputs[:, N_TRAJS:]
                    bs = RASTER.shape[0]
                    pred = pred.view(bs, N_TRAJS, TL, 2)
                    loss = pytorch_neg_multi_log_likelihood_batch(
                        TRAJ.cuda(), pred, confidences, VALID.cuda()
                    )
                    val_losses.append(loss.item())
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss.item(),
                },
                f"{PATH_TO_SAVE}/{model_name}_{iteration}_dev_{round(np.mean(val_losses), 3)}.pth",
            )


if __name__ == "__main__":
    main()
