"""Pick-and-place task: grasp an object and carry it to a 3D goal.

The full manipulation chain, and the first task here where the reward the policy
can reach on step one is not the reward that solves the task. Reach has useful
gradient immediately; pick-and-place has to *discover a grasp by exploration*
before the lift and place stages exist at all. Everything below is arranged
around that: four bounded stages, each gated so it cannot be collected out of
order, and each shaped so there is gradient at the distance scale that stage
actually operates on.

Differences from the base ``ManipulationEnv`` staging, all of which are
consequences of the tighter 4 cm tolerance and the longer horizon:

* **Two-scale place potential.** The place stage spans a decade of distance --
  ~40 cm of transport down to a 4 cm tolerance. A single ``tanh`` cannot cover
  both: tuned for the tolerance (k=20) it is numerically flat at 40 cm
  (``1 - tanh(8) = 3e-7``, so no transport gradient); tuned for transport (k=3)
  it barely varies across the whole success region, which is exactly where the
  task is decided. Summing a coarse and a fine term gives a potential that is
  monotone and informative at both ends.
* **Carry grace.** Lift and place are gated on *carrying*, not on the
  instantaneous grasp. MuJoCo pad contacts flicker in and out between substeps
  during transport, and gating 8 of the ~11 available reward on a boolean that
  chatters puts a cliff in the value function at no physical event. An object
  that was grasped and is still clear of the table is still being carried.
* **Gentler reach sharpness** (6.0 vs the base 10.0), for the same reason
  reach.py uses 5.0: from the ``ready_low`` pre-grasp pose the cube starts
  ~15-25 cm away, and k=10 leaves the gradient nearly flat out there.

``terminate_on_success`` defaults to **False**, as it must with a per-step dense
reward -- see the ``ManipulationEnv`` class docstring for the measurement behind
that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...robots.panda import JOINT_VELOCITY_LIMITS
from ..manipulation_env import ManipulationEnv, RandomizationConfig, RewardConfig

__all__ = ["PickPlaceEnv", "PickPlaceRewardConfig", "PICK_PLACE_RANDOMIZATION"]


# Spawn and goal boxes both sit strictly inside the region reach.py verified
# IK-reachable and collision-free (x 0.35-0.62, y +-0.22, z 0.45-0.70), so a
# failure here is a policy failure rather than an unreachable target.
PICK_PLACE_RANDOMIZATION = RandomizationConfig(
    object_x=(0.42, 0.58),
    object_y=(-0.14, 0.14),
    # +-45 deg, not +-180. A square cube's grasp geometry repeats every 90 deg,
    # so the full range presents physically identical states under different
    # quaternions -- aliasing that costs observation variance and buys nothing.
    object_yaw=(-np.pi / 4, np.pi / 4),
    goal_x=(0.42, 0.58),
    goal_y=(-0.14, 0.14),
    # Lower bound is 8 cm above the resting cube centre (z=0.44), so no goal is
    # satisfiable without actually lifting.
    goal_z=(0.50, 0.62),
    arm_noise=0.05,
    # Comfortably above success_threshold=0.04, so a reset can never hand out
    # the success bonus for an episode the policy has not started yet.
    min_object_goal_distance=0.10,
    randomize_object=True,
    randomize_goal=True,
)


@dataclass
class PickPlaceRewardConfig(RewardConfig):
    """Stage weights for pick-and-place.

    Weights are ordered so entering a later stage always dominates perfecting an
    earlier one: hovering with a flawless reach scores at most 1.0, while the
    cheapest grasp scores ~3.0. Without that ordering a policy can settle into
    the local optimum of maximizing stage one forever.
    """

    reach: float = 1.0
    grasp: float = 2.0
    lift: float = 2.0
    place: float = 6.0
    """Highest weight: carrying to the goal is the actual objective."""

    success: float = 10.0
    """Paid on every step the object is inside the threshold, not once. With
    termination off, that puts holding the goal at 14.7/step of place+success
    against 3.4/step for sitting 4.5 cm out -- a 4.3x margin on the only terms
    that differ (19.7 vs 8.5 on the full stage sum), so there is no hover
    optimum. Same reasoning as ReachRewardConfig.success."""

    action_penalty: float = 0.005
    velocity_penalty: float = 0.005
    joint_limit_penalty: float = 1.0

    reach_sharpness: float = 6.0
    """``k`` for the reach potential. See the module docstring."""

    place_sharpness: float = 20.0
    """Fine (terminal-precision) place scale; resolves inside the 4 cm tolerance."""
    place_coarse_sharpness: float = 3.0
    """Coarse (transport) place scale; keeps gradient alive out at ~40 cm."""

    lift_target: float = 0.12
    """Height above the table counted as fully lifted, in metres."""

    success_threshold: float = 0.04
    """Object-to-goal distance counted as a place, in metres."""
    require_grasp_for_success: bool = False
    """False: success is purely "the object is at the goal". The cube cannot
    reach an airborne goal except by being carried there, so demanding contact
    on the success step adds nothing except a way to lose a completed place to a
    single flickering contact."""

    carry_min_height: float = 0.04
    """An object that was grasped and is still this far above the table counts as
    carried even with no pad contact this step. One cube height."""


class PickPlaceEnv(ManipulationEnv):
    """Grasp the cube and bring it to a randomized 3D goal above the table.

    Reward is the four-stage sum described in the module docstring. Observations
    and the "knocked off the table" failure condition are inherited unchanged
    from ``ManipulationEnv``: object and goal state are already in the base
    observation, so there is nothing task-specific to add or remove.
    """

    def __init__(
        self,
        *args: Any,
        reward_config: PickPlaceRewardConfig | None = None,
        randomization: RandomizationConfig | None = None,
        max_episode_steps: int = 250,
        reset_pose: str = "ready_low",
        terminate_on_success: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            reward_config=reward_config or PickPlaceRewardConfig(),
            randomization=randomization or PICK_PLACE_RANDOMIZATION,
            max_episode_steps=max_episode_steps,
            reset_pose=reset_pose,
            terminate_on_success=terminate_on_success,
            **kwargs,
        )

    # -- state ------------------------------------------------------------- #

    @property
    def lift_height(self) -> float:
        """Object height above its resting height on the table, in metres."""
        resting = self.scene.table_top_z + self.object_half_height
        return float(self.object_pos[2] - resting)

    @property
    def place_distance(self) -> float:
        """Object-to-goal Euclidean distance, the quantity success is judged on."""
        return float(np.linalg.norm(self.object_pos - self._goal_pos))

    @property
    def is_success(self) -> bool:
        """Latched: True from the first step the object came within 4 cm of the goal.

        Latched rather than instantaneous because the trainer and the evaluator
        both read success only on the step an episode *ends*. With
        ``terminate_on_success=False`` an unlatched flag would silently score a
        policy that completes the place and then drifts -- or is nudged by the
        cube's own settling contact -- as a failure. The latch is set in
        ``ManipulationEnv.step`` and cleared by ``reset``.
        """
        return bool(self._episode_success)

    def is_carrying(self) -> bool:
        """Grasped now, or grasped earlier and still clear of the table.

        The second clause is the carry grace from the module docstring. Note
        ``_had_grasp`` is updated by the base ``step`` *after* ``compute_reward``
        runs, so during reward computation it means "grasped on an earlier
        step" -- exactly the condition wanted here, since a grasp on the current
        step is already covered by the first clause.
        """
        if self.is_grasped():
            return True
        return self._had_grasp and self.lift_height > self.reward_cfg.carry_min_height

    # -- reward ------------------------------------------------------------ #

    @staticmethod
    def _proximity(distance: float, sharpness: float) -> float:
        """Bounded ``[0, 1]`` closeness potential; 1 at zero distance."""
        return float(1.0 - np.tanh(sharpness * distance))

    def compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        """Four-stage dense reward. Returns ``(reward, per-term breakdown)``."""
        cfg = self.reward_cfg
        d_reach = float(np.linalg.norm(self.robot.eef_pos - self.object_pos))
        d_place = self.place_distance
        height = self.lift_height

        # Stage 1 -- reach: get the tool centre point onto the cube. The only
        # ungated stage, so there is gradient from the very first step.
        r_reach = cfg.reach * self._proximity(d_reach, cfg.reach_sharpness)

        # Stage 2 -- grasp: both pads in contact with the cube *and* the jaw
        # closed narrower than the cube is wide. Contact alone is not a grasp --
        # a finger brushing past a wide-open jaw registers contacts too. The
        # contact scan is mjData.ncon/mjData.contact, wrapped by
        # PandaRobot.contacting_geoms and reached here via _pads_in_contact.
        grasped = self.is_grasped()
        r_grasp = cfg.grasp * float(grasped)

        # Gate for stages 3 and 4. Tolerates the pad-contact flicker that occurs
        # during transport; see the module docstring.
        carrying = self.is_carrying()

        # Stage 3 -- lift: height gained, clipped at lift_target so hurling the
        # cube upward is worth no more than lifting it cleanly.
        lift_progress = float(np.clip(height / cfg.lift_target, 0.0, 1.0))
        r_lift = cfg.lift * lift_progress * float(carrying)

        # Stage 4 -- place: carry to the 3D goal. Coarse and fine scales are
        # averaged, so the term stays bounded by cfg.place. Gated on carrying, so
        # the cube cannot be batted across the table for place reward.
        place_potential = 0.5 * (
            self._proximity(d_place, cfg.place_coarse_sharpness)
            + self._proximity(d_place, cfg.place_sharpness)
        )
        r_place = cfg.place * place_potential * float(carrying)

        success = self._is_success(d_place, grasped)
        r_success = cfg.success * float(success)

        p_action = cfg.action_penalty * float(np.sum(np.square(action)))
        p_velocity = cfg.velocity_penalty * float(
            np.sum(np.square(self.robot.arm_qvel / JOINT_VELOCITY_LIMITS))
        )
        p_limits = cfg.joint_limit_penalty * float(
            np.sum(self.robot.joint_limit_violation())
        )

        reward = (
            r_reach + r_grasp + r_lift + r_place + r_success
            - p_action - p_velocity - p_limits
        )
        terms = {
            "reward/reach": r_reach,
            "reward/grasp": r_grasp,
            "reward/lift": r_lift,
            "reward/place": r_place,
            "reward/success_bonus": r_success,
            "reward/action_penalty": -p_action,
            "reward/velocity_penalty": -p_velocity,
            "reward/limit_penalty": -p_limits,
            # dist/object_to_goal is the key the base _success_distance reads.
            "dist/eef_to_object": d_reach,
            "dist/object_to_goal": d_place,
            "state/lift_height": height,
            "state/gripper_width": self.robot.gripper_width,
            "state/is_grasped": float(grasped),
            "state/carrying": float(carrying),
        }
        return float(reward), terms
