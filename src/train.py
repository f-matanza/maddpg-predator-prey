import argparse
import os

import numpy as np
import pandas as pd
import torch
from mpe2 import simple_tag_v3

from src import config
from src.independent_ddpg import IndependentDDPG
from src.maddpg import MADDPG
from src.replay_buffer import ReplayBuffer


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


def train(algorithm, run_name="run_0"):
    device = resolve_device()
    print(f"Using device: {device}")

    env = simple_tag_v3.parallel_env(
        num_adversaries    = config.NUM_ADVERSARIES,
        num_good           = config.NUM_GOOD_AGENTS,
        num_obstacles      = config.NUM_OBSTACLES,
        max_cycles         = config.MAX_CYCLES_TRAIN,
        continuous_actions = config.CONTINUOUS_ACTIONS,
    )

    env.reset()
    agent_names = env.possible_agents

    controller = build_controller(env, algorithm, device=device)
    buffer     = ReplayBuffer(config.BUFFER_SIZE, agent_ids=agent_names)

    ckpt_dir  = f"checkpoints/{algorithm}/{run_name}"
    csv_path  = f"results/{algorithm}_{run_name}_rewards.csv"
    label     = "MADDPG" if algorithm == "maddpg" else "Independent DDPG"

    rewards_history = []
    os.makedirs("results", exist_ok=True)
    os.makedirs(ckpt_dir,  exist_ok=True)

    print(f"Starting Training ({label})...")
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
                    f"{ckpt_dir}/{a_name}_actor_ep{ep}.pt",
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
        default = "run_0",
        help    = "Name of the experiment run",
    )
    args = parser.parse_args()
    train(args.algorithm, args.run_name)


if __name__ == "__main__":
    main()
