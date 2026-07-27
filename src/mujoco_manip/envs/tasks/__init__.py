"""Task environments: reach, push, lift, pick and place.

Home of ``TASK_REGISTRY``, the one place a task string is bound to a class. Both
launchers (``scripts/train.py`` and ``mujoco_manip.cli.main``) read it, so a new
task is registered once here rather than in each entry point -- the failure mode
that mapping being duplicated invites is a task that trains fine and then cannot
be replayed because only one of the two tables knows its name.

``push`` and ``lift`` are stubs with no class yet, so they are deliberately
absent: a registry entry pointing at nothing turns a clear KeyError at startup
into an ImportError halfway through building the vector env.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from ..manipulation_env import ManipulationEnv, RewardConfig
from .obstacle_pick_place import ObstaclePickPlaceEnv, ObstaclePickPlaceRewardConfig
from .pick_place import PickPlaceEnv, PickPlaceRewardConfig
from .reach import ReachEnv, ReachRewardConfig

__all__ = [
    "TASK_REGISTRY",
    "REWARD_CONFIG_REGISTRY",
    "available_tasks",
    "make_task",
    "make_reward_config",
    "ManipulationEnv",
    "ObstaclePickPlaceEnv",
    "ObstaclePickPlaceRewardConfig",
    "PickPlaceEnv",
    "PickPlaceRewardConfig",
    "ReachEnv",
    "ReachRewardConfig",
    "RewardConfig",
]


TASK_REGISTRY: dict[str, type[ManipulationEnv]] = {
    "reach": ReachEnv,
    "pick_place": PickPlaceEnv,
    # Same reward and staging as pick_place, with the spawn and goal boxes split
    # across the static wall in assets/panda_scene.xml and the collision charge
    # live. Registered separately rather than replacing pick_place so the
    # converged baseline stays runnable and comparable.
    "pick_place_obstacle": ObstaclePickPlaceEnv,
    # The base env, exposed under its own name so its generic reach->grasp->
    # lift->place staging stays runnable for comparison against pick_place.
    "manipulation": ManipulationEnv,
}


def available_tasks() -> list[str]:
    """Registered task strings, sorted. Useful for error messages and ``--help``."""
    return sorted(TASK_REGISTRY)


REWARD_CONFIG_REGISTRY: dict[str, type[RewardConfig]] = {
    "reach": ReachRewardConfig,
    "pick_place": PickPlaceRewardConfig,
    "pick_place_obstacle": ObstaclePickPlaceRewardConfig,
    "manipulation": RewardConfig,
}
"""Task string -> the reward dataclass that task's ``__init__`` defaults to.

Kept beside ``TASK_REGISTRY`` for the same reason: the reward weights a task
actually runs with are part of what identifies a run, and a second mapping living
in a launcher is how a config key ends up silently applying the *base*
``RewardConfig`` defaults to ``pick_place`` -- whose ``place`` weight is 6.0, not
the base 4.0. One table, one place to extend.
"""


def make_reward_config(
    task: str, overrides: Mapping[str, Any] | None = None
) -> RewardConfig:
    """The task's default reward config, with ``overrides`` applied by field name.

    Exists so a YAML/CLI reward block reaches the env at all: the task classes
    take a ``reward_config`` instance, not a dict, so without this the weights are
    reachable only by editing the dataclass defaults in source.

    Unknown keys raise rather than being dropped. ``scripts/train.py --set``
    validates a dotted key against the YAML tree, so it catches a typo only if the
    key is absent from the YAML -- a reward block that lists ``action_pen: 0.05``
    passes that check and would otherwise train a full run at the default weight
    while the resolved config claims otherwise.
    """
    if task not in REWARD_CONFIG_REGISTRY:
        raise KeyError(
            f"unknown task {task!r}; have {sorted(REWARD_CONFIG_REGISTRY)}"
        )
    cls = REWARD_CONFIG_REGISTRY[task]
    if not overrides:
        return cls()
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise KeyError(
            f"unknown reward key(s) {unknown} for task {task!r}; "
            f"have {sorted(known)}"
        )
    return dataclasses.replace(cls(), **dict(overrides))


def make_task(task: str, **kwargs: Any) -> ManipulationEnv:
    """Construct a registered task by name.

    Raises ``KeyError`` naming the valid options rather than ``TASK_REGISTRY[task]``'s
    bare miss, so a typo in a config is self-explanatory.
    """
    if task not in TASK_REGISTRY:
        raise KeyError(f"unknown task {task!r}; have {available_tasks()}")
    return TASK_REGISTRY[task](**kwargs)
