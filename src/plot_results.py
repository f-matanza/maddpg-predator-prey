import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REWARD_COL     = "adversary_reward_mean"
EPS_COL        = "episode"
DEF_SMOOTH_WIN = 100


def load_rewards(csv_path, reward_col=REWARD_COL):
    episodes = []
    rewards  = []

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty or missing a header row")
        if EPS_COL not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain an '{EPS_COL}' column")
        if reward_col not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a '{reward_col}' column")

        for row in reader:
            episodes.append(int(row[EPS_COL]))
            rewards.append(float(row[reward_col]))

    if not episodes:
        raise ValueError(f"{csv_path} does not contain any data rows")

    return np.array(episodes, dtype=np.int64), np.array(rewards, dtype=np.float64)


def smooth_series(values, window):
    if window <= 1:
        return values

    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def plot_comparison(
    maddpg_csv,
    iddpg_csv,
    output_path,
    reward_col     = REWARD_COL,
    smoothing_win  = DEF_SMOOTH_WIN,
):
    maddpg_eps, maddpg_rew = load_rewards(maddpg_csv, reward_col)
    iddpg_eps,  iddpg_rew  = load_rewards(iddpg_csv,  reward_col)

    plt.figure(figsize=(10, 6))
    plt.plot(
        maddpg_eps,
        smooth_series(maddpg_rew, smoothing_win),
        label     = "MADDPG",
        linewidth = 2,
    )
    plt.plot(
        iddpg_eps,
        smooth_series(iddpg_rew, smoothing_win),
        label     = "Independent DDPG",
        linewidth = 2,
    )
    plt.xlabel("Episode")
    plt.ylabel("Mean adversary reward")
    plt.title("MADDPG vs Independent DDPG")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot MADDPG vs IDDPG reward curves")
    parser.add_argument(
        "--maddpg-csv",
        type    = Path,
        default = Path("results/maddpg_rewards.csv"),
        help    = "CSV log from MADDPG training",
    )
    parser.add_argument(
        "--iddpg-csv",
        type    = Path,
        default = Path("results/iddpg_rewards.csv"),
        help    = "CSV log from Independent DDPG training",
    )
    parser.add_argument(
        "--output",
        type    = Path,
        default = Path("results/reward_comparison.png"),
        help    = "Output image path",
    )
    parser.add_argument(
        "--reward-column",
        default = REWARD_COL,
        help    = "Reward column to plot",
    )
    parser.add_argument(
        "--smoothing-window",
        type    = int,
        default = DEF_SMOOTH_WIN,
        help    = "Rolling-average window for smoothing",
    )
    args = parser.parse_args()

    plot_comparison(
        maddpg_csv    = args.maddpg_csv,
        iddpg_csv     = args.iddpg_csv,
        output_path   = args.output,
        reward_col    = args.reward_column,
        smoothing_win = args.smoothing_window,
    )
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
