"""Reach task: move the end effector to a sampled target pose.

The de-risked step on the way to pick-and-place. Reach needs no contact, no grasp
and no lift, so the reward is a single distance term with no stage gating and
there is useful gradient from the very first environment step. Pick-and-place at
the same step budget is not a comparable proposition: it has to discover a grasp
by exploration before the later reward stages become reachable at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...robots.panda import GRIPPER_MAX_WIDTH, JOINT_VELOCITY_LIMITS
from ..manipulation_env import ManipulationEnv, RandomizationConfig, RewardConfig

__all__ = ["ReachEnv", "ReachRewardConfig", "REACH_RANDOMIZATION"]


# Target box verified reachable: damped-least-squares IK converges to < 1e-4 m
# with no arm/table collision at all 27 corner and midpoint combinations.
REACH_RANDOMIZATION = RandomizationConfig(
    goal_x=(0.35, 0.62),
    goal_y=(-0.22, 0.22),
    goal_z=(0.45, 0.70),
    arm_noise=0.05,
    randomize_object=False,
    randomize_goal=True,
    min_object_goal_distance=0.0,  # no object involved, so no separation to enforce
)


@dataclass
class ReachRewardConfig(RewardConfig):
    """Reach-only weights. The grasp/lift/place stages are unused and zeroed."""

    reach: float = 1.0
    grasp: float = 0.0
    lift: float = 0.0
    place: float = 0.0

    success: float = 2.0
    """Paid on every step spent inside the threshold, not once. Combined with the
    reach term that makes holding the target worth ~3.0/step against ~0.56/step
    for hovering 9 cm out, so there is no incentive to stop short. Kept at 2.0
    rather than 5.0 to limit the size of the value-function discontinuity at the
    threshold boundary."""

    action_penalty: float = 0.005
    velocity_penalty: float = 0.005
    joint_limit_penalty: float = 1.0

    reach_sharpness: float = 5.0
    """Gentler than the grasp task's 10.0: the target can start ~40 cm away, and
    k=10 leaves the gradient nearly flat out there."""

    success_threshold: float = 0.05
    """TCP-to-target distance counted as a reach, in metres."""

    hold_steps: int = 0
    """Consecutive steps required inside the threshold. 0 means instant success;
    raise it to stop a policy from scoring by flying through the target."""


class ReachEnv(ManipulationEnv):
    """Move the tool centre point to a randomized 3D target.

    Differences from ``ManipulationEnv``:

    * Reward is distance-to-target only, with no grasp gating.
    * The cube is parked outside the arm's 0.855 m reach instead of being left on
      the table, so it cannot be knocked into the arm and inject noise into a
      task that has nothing to do with it.
    * Observations drop the object terms (45 -> 31 dims). Here they would be
      frozen constants, and constant inputs still consume first-layer parameters
      and contribute zero-variance columns to the observation normalizer.
    """

    # Far enough out that IK cannot reach it. The cube rests on the floor, so it
    # costs a few contact checks and nothing else.
    OBJECT_PARK_POS = np.array([0.0, 1.5, 0.02])

    def __init__(
        self,
        *args: Any,
        reward_config: ReachRewardConfig | None = None,
        randomization: RandomizationConfig | None = None,
        max_episode_steps: int = 150,
        reset_pose: str = "home",
        **kwargs: Any,
    ) -> None:
        self._steps_at_target = 0
        super().__init__(
            *args,
            reward_config=reward_config or ReachRewardConfig(),
            randomization=randomization or REACH_RANDOMIZATION,
            max_episode_steps=max_episode_steps,
            reset_pose=reset_pose,
            **kwargs,
        )

    # -- randomization ----------------------------------------------------- #

    def _sample_object_pos(self, options: dict[str, Any] | None) -> np.ndarray:
        return self.OBJECT_PARK_POS.copy()

    def _sample_goal(
        self, object_pos: np.ndarray, options: dict[str, Any] | None
    ) -> np.ndarray:
        """Free-space target, independent of the parked cube.

        Signature matches the base class positionally -- the base ``reset`` calls
        ``self._sample_goal(object_pos, options)``.
        """
        del object_pos  # reach targets do not depend on the object
        if options and "goal_pos" in options:
            return np.asarray(options["goal_pos"], dtype=np.float64)
        cfg = self.rand_cfg
        if not cfg.randomize_goal:
            return np.array([0.45, 0.0, 0.55])
        return np.array(
            [
                self.np_random.uniform(*cfg.goal_x),
                self.np_random.uniform(*cfg.goal_y),
                self.np_random.uniform(*cfg.goal_z),
            ]
        )

    # -- task definition --------------------------------------------------- #

    @property
    def target_distance(self) -> float:
        """Current TCP-to-target Euclidean distance."""
        return float(np.linalg.norm(self.robot.eef_pos - self._goal_pos))

    def compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        """Dense reward on TCP-to-target distance, plus smoothness penalties."""
        cfg = self.reward_cfg
        distance = self.target_distance

        r_reach = cfg.reach * (1.0 - np.tanh(cfg.reach_sharpness * distance))
        at_target = distance < cfg.success_threshold
        r_success = cfg.success * float(self._is_success(distance))

        p_action = cfg.action_penalty * float(np.sum(np.square(action)))
        p_velocity = cfg.velocity_penalty * float(
            np.sum(np.square(self.robot.arm_qvel / JOINT_VELOCITY_LIMITS))
        )
        p_limits = cfg.joint_limit_penalty * float(
            np.sum(self.robot.joint_limit_violation())
        )

        reward = r_reach + r_success - p_action - p_velocity - p_limits
        terms = {
            "reward/reach": r_reach,
            "reward/success_bonus": r_success,
            "reward/action_penalty": -p_action,
            "reward/velocity_penalty": -p_velocity,
            "reward/limit_penalty": -p_limits,
            "dist/eef_to_goal": distance,
            "state/at_target": float(at_target),
            "state/steps_at_target": float(self._steps_at_target),
        }
        return float(reward), terms

    def _is_success(self, distance: float, grasped: bool = False) -> bool:
        """Inside the threshold, and held there for ``hold_steps`` if configured."""
        del grasped  # no grasp in this task
        if distance >= self.reward_cfg.success_threshold:
            return False
        return self._steps_at_target >= self.reward_cfg.hold_steps

    def _success_distance(self, terms: dict[str, float]) -> float:
        return terms["dist/eef_to_goal"]

    def _task_failure(self, terms: dict[str, float]) -> bool:
        """No failure state. The base check would fire on the parked cube.

        The cube sits on the floor at z=0.02, far below the base class's
        "dropped off the table" threshold, so inheriting that check would
        terminate every episode on its first step.
        """
        del terms
        return False

    # -- episode plumbing -------------------------------------------------- #

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._steps_at_target = 0
        return super().reset(seed=seed, options=options)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        # Update the dwell counter *after* stepping, using the post-step distance,
        # so hold_steps counts consecutive steps ending inside the threshold.
        if self.target_distance < self.reward_cfg.success_threshold:
            self._steps_at_target += 1
        else:
            self._steps_at_target = 0
        return obs, reward, terminated, truncated, info

    def _observation(self) -> np.ndarray:
        """Proprioception + TCP pose + target. No object terms."""
        eef = self.robot.eef_pos
        goal = self._goal_pos
        parts = [
            self.robot.normalized_arm_qpos(),               # 7
            self.robot.arm_qvel / JOINT_VELOCITY_LIMITS,    # 7
            self.robot.finger_qpos / GRIPPER_MAX_WIDTH,     # 2
            self.robot.finger_qvel,                         # 2
            eef,                                            # 3
            self.robot.eef_quat,                            # 4
            goal,                                           # 3
            goal - eef,                                     # 3, the reach vector
        ]
        return np.concatenate(
            [np.asarray(p, dtype=np.float64).ravel() for p in parts]
        ).astype(np.float32)
