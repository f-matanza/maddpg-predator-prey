from __future__ import annotations

import math
from types import MethodType
from typing import Any

import numpy as np
from mpe2 import simple_tag_v3

from src_v2 import config


def make_env(render_mode: str | None = None):
    env = simple_tag_v3.parallel_env(
        num_adversaries=config.NUM_ADVERSARIES,
        num_good=config.NUM_GOOD_AGENTS,
        num_obstacles=config.NUM_OBSTACLES,
        max_cycles=config.MAX_CYCLES,
        continuous_actions=config.CONTINUOUS_ACTIONS,
        render_mode=render_mode,
        dynamic_rescaling=False,
    )

    if not config.STABLE_ENVIRONMENT:
        return env

    return StableSimpleTagEnv(
        env,
        reset_min_gap=config.RESET_MIN_GAP,
        reset_max_attempts=config.RESET_MAX_ATTEMPTS,
        enforce_obstacle_collisions=config.ENFORCE_OBSTACLE_COLLISIONS,
        fixed_render_camera=config.FIXED_RENDER_CAMERA,
        render_camera_range=config.RENDER_CAMERA_RANGE,
    )


class StableSimpleTagEnv:
    """Small safety wrapper around MPE2 simple_tag.

    MPE2's default simple_tag reset samples all entities independently, so
    obstacles can overlap each other or spawn on top of agents. Its renderer also
    rescales positions with the current outermost entity, which makes fixed
    obstacles appear to move. This wrapper keeps the public ParallelEnv API while
    repairing those two artifacts.
    """

    def __init__(
        self,
        env: Any,
        *,
        reset_min_gap: float,
        reset_max_attempts: int,
        enforce_obstacle_collisions: bool,
        fixed_render_camera: bool,
        render_camera_range: float,
    ) -> None:
        self.env = env
        self.reset_min_gap = reset_min_gap
        self.reset_max_attempts = reset_max_attempts
        self.enforce_obstacle_collisions = enforce_obstacle_collisions
        self.fixed_render_camera = fixed_render_camera
        self.render_camera_range = render_camera_range
        self._base_env = None
        self._static_landmark_positions: dict[str, np.ndarray] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: Any):
        result = self.env.reset(*args, **kwargs)
        self._base_env = _find_base_env(self.env)

        if self.fixed_render_camera:
            _install_fixed_camera_draw(self._base_env, self.render_camera_range)

        self._repair_reset_overlaps()
        self._remember_static_landmarks()

        if self.enforce_obstacle_collisions:
            self._resolve_agent_obstacle_collisions()

        return self._replace_observations(result)

    def step(self, actions: dict[str, np.ndarray]):
        result = self.env.step(actions)

        if self._base_env is None:
            self._base_env = _find_base_env(self.env)

        self._pin_static_landmarks()
        changed = False
        if self.enforce_obstacle_collisions:
            changed = self._resolve_agent_obstacle_collisions()

        if changed:
            return self._replace_observations(result)
        return result

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def _replace_observations(self, result):
        obs = self._current_observations()
        if isinstance(result, tuple) and len(result) == 2:
            _, infos = result
            return obs, infos
        if isinstance(result, tuple) and len(result) == 5:
            _, rewards, terminations, truncations, infos = result
            return obs, rewards, terminations, truncations, infos
        return result

    def _current_observations(self) -> dict[str, np.ndarray]:
        agents = getattr(self.env, "agents", [])
        if not agents:
            return {}
        return {agent: self._base_env.observe(agent) for agent in agents}

    def _repair_reset_overlaps(self) -> None:
        world = self._base_env.world
        rng = getattr(self._base_env, "np_random", np.random.default_rng())

        landmarks = [lm for lm in world.landmarks if not getattr(lm, "boundary", False)]
        placed: list[Any] = []
        for landmark in landmarks:
            landmark.state.p_pos = self._sample_position(
                rng,
                low=-0.9,
                high=0.9,
                entity=landmark,
                existing=placed,
            )
            landmark.state.p_vel = np.zeros(world.dim_p)
            placed.append(landmark)

        placed_entities = list(landmarks)
        for agent in world.agents:
            if not self._has_overlap(agent, placed_entities):
                placed_entities.append(agent)
                continue

            agent.state.p_pos = self._sample_position(
                rng,
                low=-1.0,
                high=1.0,
                entity=agent,
                existing=placed_entities,
            )
            agent.state.p_vel = np.zeros(world.dim_p)
            placed_entities.append(agent)

    def _sample_position(
        self,
        rng: np.random.Generator,
        *,
        low: float,
        high: float,
        entity: Any,
        existing: list[Any],
    ) -> np.ndarray:
        dim_p = self._base_env.world.dim_p
        for _ in range(self.reset_max_attempts):
            candidate = rng.uniform(low, high, dim_p)
            if self._position_clear(candidate, entity, existing):
                return candidate

        # Deterministic fallback for dense setups. The default project uses only
        # two obstacles and four agents, so this should normally never run.
        index = len(existing)
        angle = 2.0 * math.pi * index / max(1, len(existing) + 1)
        radius = 0.8 * high
        return np.array([math.cos(angle), math.sin(angle)]) * radius

    def _position_clear(
        self,
        candidate: np.ndarray,
        entity: Any,
        existing: list[Any],
    ) -> bool:
        for other in existing:
            min_dist = entity.size + other.size + self.reset_min_gap
            dist = np.linalg.norm(candidate - other.state.p_pos)
            if dist < min_dist:
                return False
        return True

    def _has_overlap(self, entity: Any, others: list[Any]) -> bool:
        return not self._position_clear(entity.state.p_pos, entity, others)

    def _remember_static_landmarks(self) -> None:
        self._static_landmark_positions = {
            landmark.name: landmark.state.p_pos.copy()
            for landmark in self._base_env.world.landmarks
            if not getattr(landmark, "boundary", False)
        }

    def _pin_static_landmarks(self) -> None:
        world = self._base_env.world
        for landmark in world.landmarks:
            position = self._static_landmark_positions.get(landmark.name)
            if position is None:
                continue
            landmark.state.p_pos = position.copy()
            landmark.state.p_vel = np.zeros(world.dim_p)

    def _resolve_agent_obstacle_collisions(self) -> bool:
        world = self._base_env.world
        obstacles = [
            landmark
            for landmark in world.landmarks
            if landmark.collide and not getattr(landmark, "boundary", False)
        ]
        changed = False

        for _ in range(4):
            pass_changed = False
            for agent in world.agents:
                if not agent.collide or not agent.movable:
                    continue
                for obstacle in obstacles:
                    delta = agent.state.p_pos - obstacle.state.p_pos
                    dist = float(np.linalg.norm(delta))
                    min_dist = agent.size + obstacle.size
                    if dist >= min_dist:
                        continue

                    if dist < 1e-12:
                        vel_norm = float(np.linalg.norm(agent.state.p_vel))
                        direction = (
                            agent.state.p_vel / vel_norm
                            if vel_norm > 1e-12
                            else np.array([1.0, 0.0])
                        )
                    else:
                        direction = delta / dist

                    agent.state.p_pos = obstacle.state.p_pos + direction * (
                        min_dist + 1e-6
                    )

                    inward_speed = float(np.dot(agent.state.p_vel, direction))
                    if inward_speed < 0.0:
                        agent.state.p_vel = (
                            agent.state.p_vel - inward_speed * direction
                        )

                    changed = True
                    pass_changed = True
            if not pass_changed:
                break

        return changed


def _find_base_env(env: Any) -> Any:
    stack = [env]
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if hasattr(current, "world") and hasattr(current, "scenario"):
            return current

        for attr in ("unwrapped", "aec_env", "env"):
            try:
                child = getattr(current, attr)
            except Exception:
                continue
            if child is None or child is current or callable(child):
                continue
            stack.append(child)

    raise RuntimeError("Could not find the underlying MPE2 SimpleEnv instance")


def _install_fixed_camera_draw(base_env: Any, camera_range: float) -> None:
    base_env._stable_render_camera_range = float(camera_range)
    if getattr(base_env, "_stable_fixed_draw_installed", False):
        return

    base_env.draw = MethodType(_fixed_camera_draw, base_env)
    base_env._stable_fixed_draw_installed = True


def _fixed_camera_draw(self: Any) -> None:
    import pygame
    from mpe2._mpe_utils.core import Agent

    assert self.screen is not None
    self.screen.fill((255, 255, 255))

    cam_range = max(float(self._stable_render_camera_range), 1e-6)
    scale = min(self.width, self.height) * 0.5 * 0.9 / cam_range

    text_line = 0
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for entity in self.world.entities:
        x_world, y_world = entity.state.p_pos
        x = int(round(self.width * 0.5 + x_world * scale))
        y = int(round(self.height * 0.5 - y_world * scale))
        radius = max(2, int(round(entity.size * scale)))
        color = tuple(np.clip(entity.color * 200, 0, 255).astype(int))

        pygame.draw.circle(self.screen, color, (x, y), radius)
        pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 1)

        if not isinstance(entity, Agent) or entity.silent:
            continue

        if np.all(entity.state.c == 0):
            word = "_"
        elif self.continuous_actions:
            word = "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
        else:
            word = alphabet[np.argmax(entity.state.c)]

        message = entity.name + " sends " + word + "   "
        message_x_pos = self.width * 0.05
        message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
        self.game_font.render_to(
            self.screen,
            (message_x_pos, message_y_pos),
            message,
            (0, 0, 0),
        )
        text_line += 1
