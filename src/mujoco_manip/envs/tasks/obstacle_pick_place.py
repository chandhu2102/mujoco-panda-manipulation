"""Obstacle-avoidance pick-and-place: the same chain, now across a barrier.

Everything about the reward is inherited from ``PickPlaceEnv``. What makes this a
harder task is not a new reward term but a *changed geometry*: the object spawn
box and the goal box are moved onto opposite sides of the static ``obstacle``
wall defined in ``assets/panda_scene.xml``, so the transport phase has to clear
it. The collision charge (``RewardConfig.collision_penalty``, scanned by
``ManipulationEnv.obstacle_contacts``) is what prices the alternative of pushing
through.

Why this is a separate class rather than an edit to ``PICK_PLACE_RANDOMIZATION``.
The spawn boxes are not reachable from YAML -- ``scripts/train.py:make_env_fn``
forwards ``reward``, ``curriculum_level`` and the task-space block, but never a
``randomization`` block -- so a split has to live in Python either way. Putting it
here rather than in ``pick_place.py`` keeps the converged 4-D OSC baseline
runnable and directly comparable on this branch: ``task: pick_place`` still
samples the boxes the 400 demonstrations were recorded against.

The one thing this task *does* share with the baseline against its will is the
scene file. The obstacle went into ``assets/panda_scene.xml`` itself, which
``ManipulationEnv`` resolves as ``DEFAULT_SCENE``, so the wall is physically
present for ``task: pick_place`` too. It costs that baseline nothing in reward --
``collision_penalty`` defaults to 0.0 -- but it does mean the baseline's spawn box
(x 0.42-0.58) now straddles a wall at x 0.465-0.495. Two consequences worth
knowing: ``demos/pick_place_osc_400.npz`` was recorded without the wall and its
trajectories drive straight through where it now stands, and a re-run of the
baseline on this branch is no longer the run that converged. If either matters,
give this class its own scene instead -- ``scene_path`` is already a constructor
argument on ``ManipulationEnv``, so it is a one-line default here plus a copy of
the asset with the obstacle in it.

Geometry, all measured against the wall at x 0.465-0.495, top face z 0.50:

* **Object, far side, x 0.56-0.61.** The lower bound is set by the *hand*, not by
  the cube: the hand geom is 10 cm along the jaw axis, so a top-down grasp needs
  ~6.5 cm of clearance from the wall face for any wrist yaw to fit. 0.56 leaves
  6.5 cm. Verified by curriculum seed rate -- 1.7% of 235 seeds rejected at this
  spacing, against 18.7% when the wall's far face was 3 cm closer. The upper bound
  stops short of x 0.62, the edge of the region ``reach.py`` verified reachable.
* **Goal, near side, x 0.37-0.43.** Comfortably inside the OSC ``workspace_low``
  bound of x 0.30 and the reachable floor of x 0.35. The goal is a point in air at
  z >= 0.50, near the top of the wall rather than beside it, so it needs less
  lateral clearance than the grasp does.
* **y is not split** and keeps the baseline's +-0.14. The division is in x alone,
  so the task loses none of its lateral placement diversity -- only the 16 cm x
  band becomes two ~6 cm bands.
* **``goal_z`` unchanged at 0.50-0.62.** The wall top is at z 0.50, so the lowest
  goal sits level with it: the crossing itself needs the cube centre above z 0.52
  and so is higher than that goal, meaning the cube goes up over the wall and the
  final approach then comes back down. That over-and-down move is the point of
  the task, and it is why the goal band was not simply raised above the wall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..manipulation_env import OBSTACLE_GEOM_PREFIX, RandomizationConfig
from .pick_place import PickPlaceEnv, PickPlaceRewardConfig

__all__ = [
    "ObstaclePickPlaceEnv",
    "ObstaclePickPlaceRewardConfig",
    "OBSTACLE_PICK_PLACE_RANDOMIZATION",
]


OBSTACLE_PICK_PLACE_RANDOMIZATION = RandomizationConfig(
    # Far side of the wall. See the module docstring for the clearance arithmetic.
    object_x=(0.56, 0.61),
    object_y=(-0.14, 0.14),
    object_yaw=(-np.pi / 4, np.pi / 4),
    # Near side of the wall.
    goal_x=(0.37, 0.43),
    goal_y=(-0.14, 0.14),
    goal_z=(0.50, 0.62),
    arm_noise=0.05,
    # Kept at the baseline's value even though the split geometry already forces a
    # far larger separation -- the minimum possible object-to-goal distance here is
    # 0.153 m, from 0.13 m of x offset and 0.08 m of z. Left in place so the
    # rejection sampler stays a live guard if the boxes are ever narrowed.
    min_object_goal_distance=0.10,
    randomize_object=True,
    randomize_goal=True,
)


@dataclass
class ObstaclePickPlaceRewardConfig(PickPlaceRewardConfig):
    """Baseline pick-and-place weights, plus a live collision charge.

    Every stage weight is inherited unchanged. That is deliberate: the barrier is
    meant to be a harder *spatial* problem on a reward whose shaping is already
    known to converge, so that a failure here is attributable to the geometry
    rather than to a retuned reward.
    """

    collision_penalty: float = 5.0
    """Charged per step while any arm or finger geom touches the wall.

    5.0 is the harsh end. Read the field's docstring on
    ``manipulation_env.RewardConfig`` before changing it -- in particular the
    failure mode where a charge this size, sitting 6.5 cm from the object spawn
    box, teaches avoidance of the workspace rather than of the wall.
    """


class ObstaclePickPlaceEnv(PickPlaceEnv):
    """Grasp the cube on the far side of the wall, carry it over, place it.

    Reward, observations, the carry grace and the "knocked off the table" failure
    are all inherited from ``PickPlaceEnv``. The differences are the split spawn
    boxes above and a non-zero ``collision_penalty`` by default.

    Note what is *not* overridden: the observation vector. The wall is static and
    identical on every reset, so its geometry is a constant of the task rather
    than state -- a policy has nothing to gain from being told where it is, and
    adding a term would change ``obs_dim`` and invalidate every checkpoint. The
    quantity that does vary, whether the arm is currently in contact, reaches the
    trainer through ``state/in_collision`` in the step info.
    """

    def __init__(
        self,
        *args: Any,
        reward_config: PickPlaceRewardConfig | None = None,
        randomization: RandomizationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            reward_config=reward_config or ObstaclePickPlaceRewardConfig(),
            randomization=randomization or OBSTACLE_PICK_PLACE_RANDOMIZATION,
            **kwargs,
        )
        if not self.has_obstacle:
            # The task is defined by the barrier; without one this is pick_place
            # with a needlessly narrowed spawn box, which would train and
            # evaluate cleanly while measuring nothing. Fail at construction.
            raise ValueError(
                f"ObstaclePickPlaceEnv requires a scene defining at least one geom "
                f"named '{OBSTACLE_GEOM_PREFIX}*'; none of the {self.model.ngeom} "
                f"geoms in this model match"
            )
