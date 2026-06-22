import argparse
import os
from pathlib import Path
from datetime import datetime

os.environ["SDL_VIDEODRIVER"] = "dummy"

import imageio
import numpy as np
import torch

from src_v2 import config
from src_v2.env import make_env as make_simple_tag_env
from src_v2.independent_ddpg import IndependentDDPG
from src_v2.maddpg import MADDPG


def make_env(render_mode="rgb_array"):
    return make_simple_tag_env(render_mode=render_mode)


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


def find_checkpoint_episode(ckpt_dir, agent_names):
    ckpt_dir = Path(ckpt_dir)
    episodes = set()
    for agent_name in agent_names:
        for path in ckpt_dir.glob(f"{agent_name}_actor_ep*.pt"):
            ep_str = path.stem.split("_ep")[-1]
            episodes.add(int(ep_str))
    if not episodes:
        raise FileNotFoundError(f"No actor checkpoints found in {ckpt_dir}")
    return max(episodes)


def load_checkpoints(controller, ckpt_dir, episode=None):
    ckpt_dir    = Path(ckpt_dir)
    agent_names = list(controller.agents.keys())
    if episode is None:
        episode = find_checkpoint_episode(ckpt_dir, agent_names)

    for agent_name in agent_names:
        ckpt_path = ckpt_dir / f"{agent_name}_actor_ep{episode}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=controller.device, weights_only=True)
        controller.agents[agent_name].actor.load_state_dict(state_dict)

    return episode


def unique_path(path, overwrite=False):
    path = Path(path)
    if overwrite or not path.exists():
        return path

    for idx in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{idx:03d}{path.suffix}")
        if not candidate.exists():
            return candidate

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def default_output_path(algorithm, run_name, tag=None):
    stem = f"{algorithm}_{run_name}_{config.RUN_MARKER}_trained"
    if tag:
        stem = f"{stem}_{tag}"
    return Path(config.GIF_DIR) / f"{stem}.gif"


def record_gif(controller, output_path, num_episodes=1, fps=15):
    env    = make_env(render_mode="rgb_array")
    frames = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        while env.agents:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            actions = controller.select_actions(obs, noise_std=0.0)
            obs, _, _, _, _ = env.step(actions)

    env.close()

    if not frames:
        raise RuntimeError("No frames captured — check render_mode and environment setup")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Record evaluation GIFs from trained policies")
    parser.add_argument(
        "--algorithm",
        choices     = ["maddpg", "iddpg"],
        default     = "maddpg",
        help        = "Which trained controller to evaluate",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type        = Path,
        default     = None,
        help        = "Directory with per-agent actor checkpoints",
    )
    parser.add_argument(
        "--episode",
        type        = int,
        default     = None,
        help        = "Checkpoint episode to load (default: latest found)",
    )
    parser.add_argument(
        "--output",
        type        = Path,
        default     = None,
        help        = "Output GIF path. Existing files are not overwritten unless --overwrite is set.",
    )
    parser.add_argument(
        "--tag",
        type        = str,
        default     = None,
        help        = "Optional suffix for the GIF filename, e.g. seed_01 or eval_20260620.",
    )
    parser.add_argument(
        "--overwrite",
        action      = "store_true",
        help        = "Overwrite the output GIF if it already exists.",
    )
    parser.add_argument(
        "--num-episodes",
        type        = int,
        default     = 1,
        help        = "Number of episodes to stitch into the GIF",
    )
    parser.add_argument(
        "--fps",
        type        = int,
        default     = 15,
        help        = "GIF frames per second",
    )
    parser.add_argument(
        "--run-name",
        type    = str,
        default = "run_0",
        help    = "Name of the experiment run",
    )
    args = parser.parse_args()

    algorithm  = args.algorithm
    ckpt_dir   = args.checkpoint_dir or Path(config.CHECKPOINT_DIR) / algorithm / args.run_name
    output     = args.output or default_output_path(algorithm, args.run_name, args.tag)
    output     = unique_path(output, overwrite=args.overwrite)
    device     = resolve_device()

    env        = make_env(render_mode=None)
    env.reset()
    controller = build_controller(env, algorithm, device=device)
    env.close()

    episode = load_checkpoints(controller, ckpt_dir, episode=args.episode)
    saved   = record_gif(controller, output, num_episodes=args.num_episodes, fps=args.fps)
    print(f"Loaded episode {episode} from {ckpt_dir}")
    print(f"Saved GIF to {saved}")


if __name__ == "__main__":
    main()
