import random
from collections import deque
from typing import Any

import numpy as np


class ReplayBuffer:
    """Joint replay buffer for multi-agent transitions.

    Stores (obs, actions, rewards, next_obs, dones) per transition, with agents
    kept in a fixed order so MADDPG can sample aligned batches.
    """

    def __init__(self, capacity: int, agent_ids: list[str] | None = None):
        self.capacity = capacity
        self.buffer: deque[tuple[Any, ...]] = deque(maxlen=capacity)
        self.agent_ids = list(agent_ids) if agent_ids is not None else None

    def _ordered(self, data: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self.agent_ids is None:
            raise ValueError("agent_ids must be set before pushing transitions")
        return [np.asarray(data[agent_id], dtype=np.float32) for agent_id in self.agent_ids]

    def push(
        self,
        obs_dict: dict[str, np.ndarray],
        action_dict: dict[str, np.ndarray],
        reward_dict: dict[str, float],
        next_obs_dict: dict[str, np.ndarray],
        done_dict: dict[str, bool],
    ) -> None:
        if self.agent_ids is None:
            self.agent_ids = sorted(obs_dict.keys())

        transition = (
            self._ordered(obs_dict),
            self._ordered(action_dict),
            [float(reward_dict[agent_id]) for agent_id in self.agent_ids],
            self._ordered(next_obs_dict),
            [bool(done_dict[agent_id]) for agent_id in self.agent_ids],
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict[str, list[np.ndarray] | np.ndarray]:
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {len(self.buffer)}"
            )

        batch = random.sample(self.buffer, batch_size)
        num_agents = len(self.agent_ids)

        obs = [
            np.stack([transition[0][agent_idx] for transition in batch])
            for agent_idx in range(num_agents)
        ]
        actions = [
            np.stack([transition[1][agent_idx] for transition in batch])
            for agent_idx in range(num_agents)
        ]
        rewards = np.array(
            [[transition[2][agent_idx] for transition in batch] for agent_idx in range(num_agents)],
            dtype=np.float32,
        ).T
        next_obs = [
            np.stack([transition[3][agent_idx] for transition in batch])
            for agent_idx in range(num_agents)
        ]
        dones = np.array(
            [[transition[4][agent_idx] for transition in batch] for agent_idx in range(num_agents)],
            dtype=np.float32,
        ).T

        return {
            "agent_ids": self.agent_ids,
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
        }

    def __len__(self) -> int:
        return len(self.buffer)
