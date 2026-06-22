import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src_v2 import config
from src_v2.env import make_env
from src_v2.independent_ddpg import IndependentDDPG
from src_v2.maddpg import MADDPG
from src_v2.replay_buffer import ReplayBuffer


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_controller(env, algorithm, device):
    if algorithm == "maddpg":
        return MADDPG(env, device=device)
    if algorithm == "iddpg":
        return IndependentDDPG(env, device=device)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def resolve_run_name(run_name, algorithm, mode):
    if run_name and run_name != "auto":
        return run_name
    token = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config.RUN_MARKER}_{algorithm}_{mode}_{token}"


def train(algorithm, run_name="auto"):
    run_name = resolve_run_name(run_name, algorithm, "fast")
    device = resolve_device()
    print(f"Using device: {device}")
    print(f"Run name: {run_name}")

    env = make_env()

    env.reset()
    agent_names = env.possible_agents

    controller = build_controller(env, algorithm, device=device)
    buffer     = ReplayBuffer(config.BUFFER_SIZE, agent_ids=agent_names)

    ckpt_dir = Path(config.CHECKPOINT_DIR) / algorithm / run_name
    csv_path = Path(config.RESULTS_DIR) / f"{algorithm}_{run_name}_{config.RUN_MARKER}_rewards.csv"
    label    = "MADDPG" if algorithm == "maddpg" else "Independent DDPG"

    rewards_history = []
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Starting Training ({label}) - FAST MODE...")
    for ep in range(1, config.NUM_EPS + 1):
        obs, _ = env.reset()

        noise_std = config.NOISE_STD_START - (
            (config.NOISE_STD_START - config.NOISE_STD_END) * (ep / config.NOISE_DECAY_EPS)
        )
        noise_std = max(config.NOISE_STD_END, noise_std)

        ep_reward = {a: 0.0 for a in agent_names}

        while env.agents:
            actions = controller.select_actions(obs, noise_std=noise_std)
            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            padded_next_obs = {a: next_obs.get(a, np.zeros_like(obs[a])) for a in agent_names}
            padded_dones    = {
                a: terminations.get(a, True) or truncations.get(a, True)
                for a in agent_names
            }

            buffer.push(obs, actions, rewards, padded_next_obs, padded_dones)

            for a in env.agents:
                ep_reward[a] += rewards[a]

            obs = next_obs

        # TEST start: 5 updates only at the end of the episode (wrt training time)
        if len(buffer) >= config.BATCH_SIZE:
            for _ in range(5):
                batch = buffer.sample(config.BATCH_SIZE)
                controller.update(batch)
        # TEST end

        adv_rewards = [ep_reward[a] for a in agent_names if "adversary" in a]
        adv_mean    = np.mean(adv_rewards) if adv_rewards else 0.0

        rewards_history.append({
            "episode":               ep,
            "adversary_reward_mean": adv_mean,
            **ep_reward,
        })

        if ep % config.LOG_FREQ == 0:
            avg_rew = {
                a: np.mean([rh[a] for rh in rewards_history[-config.LOG_FREQ:]])
                for a in agent_names
            }
            avg_str = ", ".join([f"{a}: {rew:.2f}" for a, rew in avg_rew.items()])
            print(f"Ep {ep:05d}/{config.NUM_EPS} | Noise: {noise_std:.2f} | Avg Rewards: [{avg_str}]")

        if ep % config.CKPT_FREQ == 0:
            for a_name, agent in controller.agents.items():
                torch.save(
                    agent.actor.state_dict(),
                    ckpt_dir / f"{a_name}_actor_ep{ep}.pt",
                )

    df = pd.DataFrame(rewards_history)
    df.to_csv(csv_path, index=False)
    print(f"Training complete! Logs saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MADDPG or Independent DDPG")
    parser.add_argument(
        "--algorithm",
        choices = ["maddpg", "iddpg"],
        default = "iddpg",
        help    = "Which algorithm to train",
    )
    parser.add_argument(
        "--run-name",
        type    = str,
        default = "auto",
        help    = "Name of the experiment run. Use 'auto' for a timestamp/job-id name.",
    )
    args = parser.parse_args()
    train(args.algorithm, args.run_name)


if __name__ == "__main__":
    main()
