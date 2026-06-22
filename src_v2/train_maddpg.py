import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src_v2 import config
from src_v2.env import make_env
from src_v2.maddpg import MADDPG
from src_v2.replay_buffer import ReplayBuffer


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = make_env()

    env.reset()
    agent_names = env.possible_agents

    controller = MADDPG(env, device=device)
    buffer     = ReplayBuffer(config.BUFFER_SIZE, agent_ids=agent_names)
    token      = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name   = f"{config.RUN_MARKER}_legacy_maddpg_{token}"
    ckpt_dir   = Path(config.CHECKPOINT_DIR) / "maddpg" / run_name
    csv_path   = Path(config.RESULTS_DIR) / f"maddpg_{run_name}_{config.RUN_MARKER}_rewards.csv"

    rewards_history = []
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Starting Training (MADDPG)...")
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

            if len(buffer) >= config.BATCH_SIZE:
                batch = buffer.sample(config.BATCH_SIZE)
                controller.update(batch)

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


if __name__ == "__main__":
    train()
