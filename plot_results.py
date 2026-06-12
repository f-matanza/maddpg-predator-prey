import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REWARD_COLUMN = "adversary_reward_mean"
EPISODE_COLUMN = "episode"
DEFAULT_SMOOTHING_WINDOW = 100


def load_rewards(csv_path: Path, reward_column: str = REWARD_COLUMN) -> tuple[np.ndarray, np.ndarray]:
    episodes: list[int] = []
    rewards: list[float] = []

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty or missing a header row")
        if EPISODE_COLUMN not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain an '{EPISODE_COLUMN}' column")
        if reward_column not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a '{reward_column}' column")

        for row in reader:
            episodes.append(int(row[EPISODE_COLUMN]))
            rewards.append(float(row[reward_column]))

    if not episodes:
        raise ValueError(f"{csv_path} does not contain any data rows")

    return np.array(episodes, dtype=np.int64), np.array(rewards, dtype=np.float64)


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values

    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def plot_comparison(
    maddpg_csv: Path,
    iddpg_csv: Path,
    output_path: Path,
    reward_column: str = REWARD_COLUMN,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> None:
    maddpg_episodes, maddpg_rewards = load_rewards(maddpg_csv, reward_column)
    iddpg_episodes, iddpg_rewards = load_rewards(iddpg_csv, reward_column)

    plt.figure(figsize=(10, 6))
    plt.plot(
        maddpg_episodes,
        smooth_series(maddpg_rewards, smoothing_window),
        label="MADDPG",
        linewidth=2,
    )
    plt.plot(
        iddpg_episodes,
        smooth_series(iddpg_rewards, smoothing_window),
        label="Independent DDPG",
        linewidth=2,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MADDPG vs IDDPG reward curves")
    parser.add_argument(
        "--maddpg-csv",
        type=Path,
        default=Path("results/maddpg_rewards.csv"),
        help="CSV log from MADDPG training",
    )
    parser.add_argument(
        "--iddpg-csv",
        type=Path,
        default=Path("results/iddpg_rewards.csv"),
        help="CSV log from Independent DDPG training",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/reward_comparison.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--reward-column",
        default=REWARD_COLUMN,
        help="Reward column to plot",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW,
        help="Rolling-average window for smoothing",
    )
    args = parser.parse_args()

    plot_comparison(
        maddpg_csv=args.maddpg_csv,
        iddpg_csv=args.iddpg_csv,
        output_path=args.output,
        reward_column=args.reward_column,
        smoothing_window=args.smoothing_window,
    )
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
