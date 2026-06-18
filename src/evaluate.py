import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from mpe2 import simple_tag_v3

from src import config
from src.independent_ddpg import IndependentDDPG
from src.maddpg import MADDPG


def make_env(render_mode="rgb_array"):
    return simple_tag_v3.parallel_env(
        num_adversaries    = config.NUM_ADVERSARIES,
        num_good           = config.NUM_GOOD_AGENTS,
        num_obstacles      = config.NUM_OBSTACLES,
        max_cycles         = config.MAX_CYCLES,
        continuous_actions = config.CONTINUOUS_ACTIONS,
        render_mode        = render_mode,
    )


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
        help        = "Output GIF path",
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
    args = parser.parse_args()

    algorithm  = args.algorithm
    ckpt_dir   = args.checkpoint_dir or Path(f"checkpoints/{algorithm}")
    output     = args.output or Path(f"gifs/{algorithm}_trained.gif")
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
