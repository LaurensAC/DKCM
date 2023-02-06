import argparse
import os
import math
import numpy as np
import torch
import json
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from torch.utils.data import DataLoader
from train_DKCM import WaymoLoader, pytorch_neg_multi_log_likelihood_batch
from scipy.spatial import distance as dis

fig, ax = plt.subplots()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--n-samples", type=int, required=False, default=65)
    parser.add_argument("--use-top1", action="store_true")
    parser.add_argument("--empty_roadmap", action="store_true", required=False, default=False)
    parser.add_argument("--late_detection", action="store_true", required=False, default=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    model = torch.jit.load(args.model).cuda().eval()
    print(model.code)
    # print(model.graph)
    loader = DataLoader(
        WaymoLoader(args.data, return_vector=True),
        batch_size=1,
        num_workers=1,
        shuffle=False,
    )

    # initialize Metrics
    ADE = 0
    mADE = 0

    FDE = []
    mFDE = []

    ADE_1s = 0
    mADE_1s = 0

    FDE_1s = []
    mFDE_1s = []

    total_avails = 0
    total_avails_1s = 0

    dt = 0.1
    result = []

    # set diagonal covariance KF-index
    mean = [0, 0]
    cov = [[250, 0], [0, 1]]

    iii = 0
    with torch.no_grad():
        for x, y, is_available, vector_data, current_state in loader:
            x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))

            if args.empty_roadmap:  # empty roadmap\
                x[0][0] = torch.full((224, 224), 1)
                x[0][1] = torch.full((224, 224), 1)
                x[0][2] = torch.full((224, 224), 1)
            if args.late_detection:  # empty history
                x[0][3] = torch.full((224, 224), 0)
                x[0][4] = torch.full((224, 224), 0)
                x[0][5] = torch.full((224, 224), 0)
                x[0][6] = torch.full((224, 224), 0)
                x[0][7] = torch.full((224, 224), 0)
                x[0][8] = torch.full((224, 224), 0)
                x[0][9] = torch.full((224, 224), 0)
                x[0][10] = torch.full((224, 224), 0)
                x[0][11] = torch.full((224, 224), 0)
                # x[0][12] = torch.full((224,224),0)

            confidences_logits, logits = model(x, current_state)

            argmax = confidences_logits.argmax()
            if args.use_top1:
                confidences_logits = confidences_logits[:, argmax].unsqueeze(1)
                logits = logits[:, argmax].unsqueeze(1)

            loss = pytorch_neg_multi_log_likelihood_batch(
                y, logits, confidences_logits, is_available
            )
            confidences = torch.softmax(confidences_logits, dim=1)
            V = vector_data[0]

            X, idx = V[:, :44], V[:, 44].flatten()

            logits = logits.squeeze(0).cpu().numpy()
            y = y.squeeze(0).cpu().numpy()
            is_available = is_available.squeeze(0).long().cpu().numpy()
            confidences = confidences.squeeze(0).cpu().numpy()

            ########################### KF-index################################
            for traj_id in range(len(logits)):
                last_pos = [0, 0]
                velocity_0 = current_state[2]
                distance_lr = (current_state[1]) * 0.6 * 0.4
                distance_lf = (current_state[1]) * 0.6 * 0.6

                max_mal_distance = -1

                for i in range(80):
                    points = []
                    x_diff = (
                        logits[traj_id][i][0] - last_pos[0]
                    )  # Calculate x difference this step
                    y_diff = (
                        logits[traj_id][i][1] - last_pos[1]
                    )  # Calculate y difference this step

                    # approximate heading based on position difference
                    heading_0 = math.atan2(y_diff, x_diff)

                    # Generate steering_angle/acceleration pairs (Monte Carlo)
                    st_angles, accel_values = np.random.multivariate_normal(
                        mean, cov, 100
                    ).T
                    xs = []
                    ys = []

                    # for each pair calculate the new position according to the Kinematic Bicycle Model:
                    for j in range(100):
                        Beta = math.atan(
                            (distance_lr / (distance_lr + distance_lf))
                            * math.tan(st_angles[j] * math.pi / 180)
                        )
                        velocity = velocity_0 + accel_values[j] * dt
                        heading = (
                            heading_0 + (velocity / distance_lr) * math.sin(Beta) * dt
                        )

                        xs.append(0 + velocity.item() * math.cos(heading + Beta) * dt)
                        ys.append(0 + velocity.item() * math.sin(heading + Beta) * dt)

                    # calculate velocity according to prediction
                    distance = math.dist(last_pos, logits[traj_id][i])
                    velocity_0 = distance / 0.1
                    last_pos = logits[traj_id][i]

                    # calculate mahalanobis distance between prediction and Monte Carlo simulation
                    iv = np.linalg.inv(np.cov(xs, ys))
                    mal_distance = dis.mahalanobis(
                        [np.mean(xs), np.mean(ys)], [x_diff, y_diff], iv
                    )
                    # save max mahalanobis distance (in this trajectory)
                    if mal_distance > max_mal_distance:
                        max_mal_distance = mal_distance

                    # Plot large values (optional)
                    # if dis.mahalanobis([np.mean(xs),np.mean(ys)], [x_diff, y_diff], iv) > 10:
                    #     result.append(True)
                    #     ax.clear()
                    #     ax.scatter(xs,ys, s=0.2, alpha=0.4)
                    #     ax.scatter(0,0)
                    #     ax.scatter(x_diff,y_diff)
                    #     ax.axis('square')
                    #     plt.draw()
                    #     plt.pause(0.1)
                    #     continue
                result.append(max_mal_distance)

            ##############Calculate (min)ADE and (min)FDE#######################
            distance = []
            f_distance = []

            distance_1s = []
            f_distance_1s = []

            for traj_id in range(len(logits)):  # for each trajectory
                traj_distance = []
                for i in range(len(is_available)):  # for each step
                    if is_available[i]:  # if valid
                        traj_distance.append(
                            math.dist(logits[traj_id][i], y[i])
                        )
                        total_avails += 1  # keep track of number of valid steps
                        if i < 10:
                            total_avails_1s += (
                                1  # keep track of number of valid <1s steps
                            )
                    else:
                        traj_distance.append(np.nan)

                if not np.isnan(traj_distance[-1]):  # if final step is not NaN
                    f_distance.append(traj_distance[-1])
                if not np.isnan(traj_distance[9]):  # if 1s step is not NaN
                    f_distance_1s.append(traj_distance[9])

                distance.append(traj_distance)  # save all steps of this trajectory
                distance_1s.append(
                    traj_distance[0:10] # save all <1s steps of this trajectory
                )

            # 8s metrics
            ADE += np.nansum(distance)  # add sum of all trajectories to ADE
            FDE.append(np.mean(f_distance))  # add mean final step to FDE

            try:
                mADE += np.nansum(
                    distance[np.argmin(f_distance)] # add sum of best trajectory to minADE
                )
                mFDE.append(np.min(f_distance))  # add best trajectory to minFDE
            except:
                pass

            # 1s metrics
            ADE_1s += np.nansum(distance_1s)
            FDE_1s.append(np.mean(f_distance_1s))
            try:
                mADE_1s += np.nansum(distance_1s[np.argmin(f_distance_1s)])
                mFDE_1s.append(np.min(f_distance_1s))
            except:
                pass



            ###########################Visualization###########################
            iii += 1
            if iii % 100 == 0:
                print("Progress: " + str(iii))
            if iii > args.n_samples:
                continue

            figure(figsize=(15, 15), dpi=80)
            for i in np.unique(idx):
                _X = X[idx == i]
                if _X[:, 5:12].sum() > 0:
                    plt.plot(_X[:, 0], _X[:, 1], linewidth=4, color="red")
                else:
                    plt.plot(_X[:, 0], _X[:, 1], color="black")
                plt.xlim([-224 // 4, 224 // 4])
                plt.ylim([-224 // 4, 224 // 4])
                plt.xticks([])
                plt.yticks([])

            plt.plot(
                y[is_available > 0][::10, 0],
                y[is_available > 0][::10, 1],
                "-o",
                linewidth=4,
                markersize=14,
                label="gt",
            )

            plt.plot(
                logits[confidences.argmax()][is_available > 0][::10, 0],
                logits[confidences.argmax()][is_available > 0][::10, 1],
                "-o",
                linewidth=4,
                markersize=14,
                label="pred "
                + str(confidences.argmax())
                + " "
                + str(np.max(confidences))[0:5],
            )
            if not args.use_top1:
                for traj_id in range(len(logits)):
                    if traj_id == argmax:
                        continue

                    alpha = confidences[traj_id].item()
                    plt.plot(
                        logits[traj_id][is_available > 0][::10, 0],
                        logits[traj_id][is_available > 0][::10, 1],
                        "-o",
                        linewidth=4,
                        markersize=14,
                        label=f"pred {traj_id} {alpha:.3f}",
                        alpha=0.75,
                    )

            plt.legend(fontsize=30)
            plt.tight_layout()
            plt.savefig(os.path.join(args.save, f"{iii:0>2}_{loss.item():.3f}.png"))
            plt.close()

    with open("KFindex.txt", "w") as f:
        json.dump(result, f, ensure_ascii=False)

    print("\n 8s: \n")

    print(mADE / (total_avails / 6))
    print(ADE / total_avails)
    print(np.nanmean(mFDE))
    print(np.nanmean(FDE))

    print("\n 1s: \n")

    print(mADE_1s / (total_avails_1s / 6))
    print(ADE_1s / total_avails_1s)
    print(np.nanmean(mFDE_1s))
    print(np.nanmean(FDE_1s))


if __name__ == "__main__":
    main()
