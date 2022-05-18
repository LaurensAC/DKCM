import argparse
import os
import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from torch.utils.data import DataLoader

from train_KF_V2 import WaymoLoader, pytorch_neg_multi_log_likelihood_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--n-samples", type=int, required=False, default=50)
    parser.add_argument("--use-top1", action="store_true")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    model = torch.jit.load(args.model).cuda().eval()
    print(model.code)
    #print(model.graph)
    loader = DataLoader(
        WaymoLoader(args.data, return_vector=True),
        batch_size=1,
        num_workers=1,
        shuffle=False,
    )

    # initiate Metrics
    ADE = []
    mADE = []

    FDE = []
    mFDE = []

    iii = 0
    with torch.no_grad():
        for x, y, is_available, vector_data, current_state in loader:
            x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))

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

            # Calculate (m)ADE and (m)FDE
            distance = []
            f_distance = []
            for traj_id in range(len(logits)):
                distances = []
                for i in range(len(is_available)):
                    if is_available[i]:
                        distances.append(math.dist(logits[traj_id][i], y[i]))
                f_distance.append(distances[-1])
                distance.append(np.mean(distances))

            ADE.append(np.mean(distance))
            mADE.append(np.min(distance))

            FDE.append(np.mean(f_distance))
            mFDE.append(np.min(f_distance))

            iii += 1
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

            plt.plot(
                y[is_available > 0][::10, 0],
                y[is_available > 0][::10, 1],
                "-o",
                label="gt",
            )

            plt.plot(
                logits[confidences.argmax()][is_available > 0][::10, 0],
                logits[confidences.argmax()][is_available > 0][::10, 1],
                "-o",
                label="pred top 1",
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
                        label=f"pred {traj_id} {alpha:.3f}",
                        alpha=alpha,
                    )


            plt.title("KF: " + str(loss.item()))
            plt.legend()
            plt.savefig(
                os.path.join(args.save, f"{iii:0>2}_{loss.item():.3f}.png")
            )
            plt.close()


    print(np.mean(mADE))
    print(np.mean(ADE))
    print(np.mean(mFDE))
    print(np.mean(FDE))






if __name__ == "__main__":
    main()
