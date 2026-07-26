#!/usr/bin/env python3
"""Record scripted pick-and-place demonstrations for behavioural cloning.

    python scripts/record_demos.py --config configs/train/pick_place_osc.yaml \
        --episodes 60 --out demos/pick_place_osc.npz

    python scripts/record_demos.py --config configs/train/pick_place_10m.yaml \
        --episodes 60 --out demos/pick_place_torque.npz

The env is built through ``scripts/train.py``'s ``make_env_fn``, so the recorded
observations and actions are in *exactly* the spaces the policy will train in --
including the task-space wrapper if the config enables it. Recording against a
hand-built env is the fast way to produce a dataset whose action dimension
silently disagrees with the policy's.

Two experts, selected by the config rather than by a flag, because the action
space determines what a demonstration even *is*:

* **Task space** (``env.task_space.enabled``) -- the expert emits the same 4-D
  tool-centre-point delta the policy will. Trivial to script: the action is a
  clipped direction vector.

* **Torque** (``control_mode: torque``) -- the expert emits normalized joint
  torques, produced by a joint-space PD servo tracking IK waypoints. This is the
  path that makes BC possible *without* abandoning torque control, and it works
  because the mapping is invertible in the direction that matters:
  ``PandaRobot.apply_torque`` scales a normalized action by
  ``JOINT_TORQUE_LIMITS`` and then adds gravity compensation, so the normalized
  action that reproduces a desired pre-compensation torque ``tau`` is exactly
  ``tau / JOINT_TORQUE_LIMITS``. The recorded action is therefore the action the
  policy would have to output, not an approximation of it.

  The servo gains are deliberately softer than
  ``PandaRobot.apply_joint_position``'s. That law is recomputed every substep
  (500 Hz); this one is computed once per control step and held for all 20
  substeps by ``ManipulationEnv.step``, so it is sample-and-hold at 25 Hz and
  ``omega_n * control_dt`` has to stay well under 1 for the discrete loop to be
  stable. See ``TORQUE_EXPERT_NATURAL_FREQ``.

Only episodes that actually pick the object are kept -- see ``--require-lift``.
An episode whose ``is_success`` latched because the cube was already near the
goal, or which reached the goal and then dropped the cube off the table, is not a
demonstration of the task.

Output is a single ``.npz``. Observations are *raw*, i.e. pre-normalization:
``NormalizeObservation`` is a vector-env wrapper applied above ``make_env_fn``,
so it is not in this path, and ``scripts/pretrain_bc.py`` fits the normalizer on
this data itself.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import mujoco  # noqa: E402

from mujoco_manip.envs.manipulation_env import (  # noqa: E402
    ManipulationEnv,
    grasp_quat_candidates,
)
from mujoco_manip.envs.wrappers.task_space import TaskSpaceWrapper  # noqa: E402
from mujoco_manip.robots.controllers.ik import solve_site_ik  # noqa: E402
from mujoco_manip.robots.panda import (  # noqa: E402
    JOINT_POSITION_LIMITS,
    JOINT_TORQUE_LIMITS,
    JOINT_VELOCITY_LIMITS,
)
from train import load_config, make_env_fn  # noqa: E402

LOGGER = logging.getLogger("record_demos")

# --------------------------------------------------------------------------- #
# Waypoint plan
# --------------------------------------------------------------------------- #

HOVER_HEIGHT: float = 0.07
"""Pre-grasp clearance above the object centre, in metres. Far enough that the
descent is vertical -- an approach with a lateral component at contact height
pushes the cube rather than straddling it."""

LIFT_HEIGHT: float = 0.12
"""Clearance above the object's start height for the lift waypoint. Matches
``PickPlaceRewardConfig.lift_target``, so the demonstration reaches the top of
the lift stage's shaping rather than stopping partway up it."""

CLOSE_STEPS: int = 14
"""Control steps spent commanding the jaw shut before the lift.

The jaw is position-actuated so it closes fast, but ``is_grasped`` needs contact
*established* on both pads, and the object settles between the fingers as they
close. Verified on this scene: the predicate first holds around 8 steps in, so
this leaves margin."""

OPEN: float = 1.0
CLOSED: float = -1.0

TORQUE_EXPERT_NATURAL_FREQ: float = 10.0
"""Closed-loop natural frequency of the torque expert, rad/s.

Bounded by sample-and-hold, not by torque: the law is evaluated once per control
step and held for the whole 40 ms interval, so ``omega_n * control_dt`` must stay
comfortably below 1. At 10 rad/s that product is 0.4.

Unlike ``POSITION_NATURAL_FREQ`` this is a genuine closed-loop frequency for
*every* joint, because the expert shapes its command by the mass matrix -- see
``TorqueExpert``. A diagonal PD law cannot make the same claim, and that is not a
refinement: measured on this scene, a critically-damped diagonal law with
per-joint gains from the inertia *diagonal* went unstable in the joint1/joint3
anti-phase mode at every frequency from 2 to 8 rad/s. Velocities alternated sign
and doubled each control step -- +1.36, -3.93, +9.66, -11.59 rad/s on joint1
against -1.57, +2.91, -8.62, +9.16 on joint3 -- from tracking errors of under
0.2 rad and unsaturated commands. Those two joints rotate about nearly parallel
axes in ``ready_low``, so their counter-rotating mode moves very little mass and
its effective inertia is a small fraction of either diagonal entry; gains sized
for the diagonal put that mode's frequency past the Nyquist limit of a 25 Hz
loop."""

TORQUE_EXPERT_DAMPING_RATIO: float = 1.0


@dataclass
class Waypoint:
    """One leg of the scripted trajectory."""

    name: str
    position: np.ndarray
    """Target tool-centre-point position, world frame."""
    gripper: float
    """Normalized jaw command held for this leg: ``OPEN`` or ``CLOSED``."""
    max_steps: int
    tolerance: float = 0.006
    """Position error at which the leg is considered reached, in metres."""
    hold_steps: int = 0
    """Steps to keep commanding this waypoint after reaching it."""
    require_grasp: bool = False
    """Abort the episode if ``is_grasped`` is False when this leg ends."""


def build_plan(env: ManipulationEnv) -> list[Waypoint]:
    """The reach -> grasp -> lift -> place waypoint sequence for this reset."""
    obj = env.object_pos.copy()
    goal = env.goal_pos.copy()
    lift_z = env.scene.table_top_z + env.object_half_height + LIFT_HEIGHT
    return [
        Waypoint("hover", obj + np.array([0.0, 0.0, HOVER_HEIGHT]), OPEN, 60),
        # Tighter tolerance on the descent than on the approach: this is the leg
        # whose error becomes the lateral offset ``is_grasped``'s enclosure test
        # measures.
        Waypoint("descend", obj.copy(), OPEN, 60, tolerance=0.004),
        Waypoint("close", obj.copy(), CLOSED, CLOSE_STEPS,
                 tolerance=0.0, hold_steps=CLOSE_STEPS, require_grasp=True),
        Waypoint("lift", np.array([obj[0], obj[1], lift_z]), CLOSED, 60,
                 require_grasp=True),
        Waypoint("carry", goal.copy(), CLOSED, 100, tolerance=0.015),
        Waypoint("settle", goal.copy(), CLOSED, 12, tolerance=0.0, hold_steps=12),
    ]


# --------------------------------------------------------------------------- #
# Experts
# --------------------------------------------------------------------------- #


class TaskSpaceExpert:
    """Emits the wrapper's 4-D (or 5-D) tool-centre-point delta action."""

    def __init__(self, wrapper: TaskSpaceWrapper, *, approach_speed: float = 0.6) -> None:
        self.wrapper = wrapper
        self.base = wrapper.base
        self.approach_speed = float(approach_speed)
        """Fraction of ``max_delta_pos`` used on the final descent. Full speed
        drives the pads into the cube hard enough to punt it before the jaw
        closes."""

    def action(self, waypoint: Waypoint) -> np.ndarray:
        error = waypoint.position - self.base.robot.eef_pos
        delta = np.clip(error / self.wrapper.max_delta_pos, -1.0, 1.0)
        if waypoint.name in ("descend", "close"):
            delta = delta * self.approach_speed
        action = np.zeros(self.wrapper.action_space.shape, dtype=np.float64)
        action[:3] = delta
        # The wrapper integrates this dimension, so a saturated command is the
        # fastest legal way to reach either jaw extreme and holding it there is
        # what keeps it clamped.
        action[3] = waypoint.gripper
        return action


class TorqueExpert:
    """Emits normalized joint torques from an inertia-shaped (computed-torque) law.

    Four details carry the whole thing, and each was a measured failure first:

    1. **Four yaw candidates, chosen by travel time.** Solving for only the
       nearest face yaw put joint7 at 2.897 rad -- its hard limit -- from a start
       of 0.762, a 2.1 rad error that saturated the wrist torque on step one. See
       ``grasp_quat_candidates``.
    2. **A rate-limited joint path, not a step to the solution.** Commanding the
       IK solution directly is a step input, and a step large enough to saturate
       any joint clips the damping term out of the command -- a PD law whose
       damping has been clipped away is an undamped one. Advancing the *commanded*
       pose at the joints' own velocity limits keeps the tracking error small
       enough that damping is never what saturates.
    3. **Inertia shaping.** The command is a desired *acceleration*, converted to
       torque through the mass matrix:

           tau = M(q) @ (kp * (q_cmd - q) - kd * qvel)

       so the closed loop is ``qddot = kp * e - kd * qdot`` -- identical and
       decoupled on every joint, whatever the configuration. A diagonal law
       instead leaves the closed-loop frequency of each *coupled mode* set by that
       mode's effective inertia, which for the joint1/joint3 anti-phase mode here
       is far below either diagonal entry; see ``TORQUE_EXPERT_NATURAL_FREQ`` for
       the instability that produced. ``mj_mulM`` supplies ``M @ v`` directly, so
       this costs one matrix-vector product per control step and never forms ``M``.
    4. **Torque labels that invert exactly.** ``PandaRobot.apply_torque`` scales a
       normalized action by ``JOINT_TORQUE_LIMITS`` and then adds gravity
       compensation, so ``tau / JOINT_TORQUE_LIMITS`` is precisely the action that
       reproduces ``tau``. The recorded label is the action the policy must output,
       not a proxy for it. Note the consequence for BC: the policy is being taught
       to imitate *this* controller, and it can, because everything the law reads
       -- ``q``, ``qvel``, and the object and goal poses that determine ``q_cmd``
       -- is in the observation.
    """

    def __init__(self, env: ManipulationEnv, *, path_speed: float = 0.35) -> None:
        self.env = env
        self.robot = env.robot
        self.path_speed = float(path_speed)
        """Fraction of each joint's velocity limit the commanded path advances at.
        Below 1.0 so the servo's tracking lag stays inside the joint's remaining
        headroom rather than consuming all of it."""

        # Acceleration-level gains: units 1/s^2 and 1/s, shared by every joint.
        # The mass matrix supplies the per-joint scaling.
        omega = TORQUE_EXPERT_NATURAL_FREQ
        self.kp = omega ** 2
        self.kd = 2.0 * TORQUE_EXPERT_DAMPING_RATIO * omega

        self._step_limit = JOINT_VELOCITY_LIMITS * env.control_dt * self.path_speed
        self._accel = np.zeros(env.model.nv, dtype=np.float64)
        self._torque = np.zeros(env.model.nv, dtype=np.float64)
        self._scratch = mujoco.MjData(env.model)
        self._q_goal = self.robot.arm_qpos.copy()
        """Where the leg ends: the IK solution."""
        self._q_cmd = self.robot.arm_qpos.copy()
        """Where the servo is being told to be *now*, walking toward ``_q_goal``."""
        self._solved_for: str | None = None
        self.ik_failures = 0

    def _solve(self, waypoint: Waypoint) -> bool:
        quat = self.env.object_quat
        object_yaw = 2.0 * float(np.arctan2(quat[3], quat[0]))
        q_now = self.robot.arm_qpos
        q_nominal = self.robot.safe_reset_qpos(self.env.reset_pose)

        best: Any = None
        best_cost = np.inf
        for candidate in grasp_quat_candidates(object_yaw):
            result = solve_site_ik(
                self.env.model,
                self.robot.idx.eef_site_id,
                self.robot.idx.arm_qpos_adr,
                self.robot.idx.arm_dof_adr,
                waypoint.position,
                target_quat=candidate,
                q_init=q_now,
                joint_limits=JOINT_POSITION_LIMITS,
                qpos_full=self.env.data.qpos,
                rot_tolerance=0.15,
                scratch=self._scratch,
                q_nominal=q_nominal,
                posture_gain=0.15,
            )
            # Accept a partial solve whose *position* landed: the rotation
            # tolerance is loose by design (see TaskSpaceWrapper) and what decides
            # a demonstration is whether is_grasped agrees at the end, which the
            # plan checks directly.
            if result.pos_error > 0.005:
                continue
            # Rank by the slowest joint's travel time, not by summed joint angle:
            # what the servo has to survive is the joint that has furthest to go
            # relative to how fast it may go, and that is also what decides whether
            # any joint saturates.
            cost = float(np.max(np.abs(result.qpos - q_now) / JOINT_VELOCITY_LIMITS))
            if cost < best_cost:
                best, best_cost = result, cost

        if best is None:
            self.ik_failures += 1
            return False
        self._q_goal = best.qpos.copy()
        return True

    def begin(self, waypoint: Waypoint) -> bool:
        if waypoint.name != self._solved_for:
            self._solved_for = waypoint.name
            # Start the commanded pose from where the arm actually is, so a leg
            # never begins with a standing error inherited from the previous one.
            self._q_cmd = self.robot.arm_qpos.copy()
            return self._solve(waypoint)
        return True

    def action(self, waypoint: Waypoint) -> np.ndarray:
        self._q_cmd = self._q_cmd + np.clip(
            self._q_goal - self._q_cmd, -self._step_limit, self._step_limit
        )
        dof = self.robot.idx.arm_dof_adr
        # Zero on every non-arm degree of freedom, so `M @ accel` picks out
        # exactly the arm block of the mass matrix and the object's free joint
        # contributes nothing.
        self._accel[:] = 0.0
        self._accel[dof] = (
            self.kp * (self._q_cmd - self.robot.arm_qpos)
            - self.kd * self.robot.arm_qvel
        )
        # Evaluated against the live MjData, so the mass matrix is the one for the
        # current configuration -- already factorized by the preceding step, so
        # this adds no dynamics computation.
        mujoco.mj_mulM(self.env.model, self.env.data, self._torque, self._accel)
        arm = np.clip(self._torque[dof] / JOINT_TORQUE_LIMITS, -1.0, 1.0)
        return np.concatenate([arm, [waypoint.gripper]])


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #


@dataclass
class Trajectory:
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)


def record_episode(
    env: Any,
    expert: TaskSpaceExpert | TorqueExpert,
    *,
    seed: int,
    require_lift: bool,
) -> tuple[Trajectory | None, str]:
    """One scripted attempt. Returns ``(trajectory, outcome)``.

    ``trajectory`` is None for a rejected attempt; ``outcome`` names the reason so
    the caller can report the rejection breakdown rather than only a yield rate. A
    silently low yield is indistinguishable from a plan that is subtly wrong.
    """
    base: ManipulationEnv = env.unwrapped
    obs, _ = env.reset(seed=seed)
    plan = build_plan(base)
    traj = Trajectory()
    grasped_ever = False
    max_height = -np.inf
    info: dict[str, Any] = {}

    for waypoint in plan:
        if isinstance(expert, TorqueExpert) and not expert.begin(waypoint):
            return None, f"ik_failed:{waypoint.name}"

        held = 0
        for _ in range(waypoint.max_steps + waypoint.hold_steps):
            action = expert.action(waypoint)
            traj.observations.append(np.asarray(obs, dtype=np.float32).copy())
            traj.actions.append(np.asarray(action, dtype=np.float32).copy())
            obs, reward, terminated, truncated, info = env.step(action)
            traj.rewards.append(float(reward))

            grasped_ever = grasped_ever or bool(info.get("is_grasped", False))
            max_height = max(max_height, float(info.get("state/lift_height", -np.inf)))

            if terminated or truncated:
                # A termination mid-plan is the object being knocked off the
                # table, which is a failure however the latch reads.
                return None, "terminated_early"

            reached = (
                float(np.linalg.norm(base.robot.eef_pos - waypoint.position))
                < waypoint.tolerance
            )
            if reached or waypoint.tolerance == 0.0:
                held += 1
                if held > waypoint.hold_steps:
                    break

        if waypoint.require_grasp and not base.is_grasped():
            return None, f"no_grasp:{waypoint.name}"

    if not info.get("is_success", False):
        return None, "no_success"
    # The latch makes is_success true for an episode that reached the goal and
    # then dropped the cube, and a curriculum-seeded episode can start already
    # successful. Requiring a real lift is what separates a demonstration of the
    # task from either of those.
    if require_lift and not (grasped_ever and max_height > 0.5 * LIFT_HEIGHT):
        return None, "no_lift"
    return traj, "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--episodes", type=int, default=60, help="successful demos to collect")
    parser.add_argument("--max-attempts", type=int, default=0,
                        help="attempt cap; 0 means 5x --episodes")
    parser.add_argument("--out", type=Path, required=True, help="output .npz path")
    parser.add_argument("--seed", type=int, default=7_000_000)
    parser.add_argument("--require-lift", action="store_true", default=True)
    parser.add_argument("--no-require-lift", dest="require_lift", action="store_false")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S",
    )
    if not args.config.is_file():
        raise SystemExit(f"config not found: {args.config}")
    config = load_config(args.config, args.overrides)

    # Curriculum seeding off: a seeded episode starts mid-task, so the scripted
    # plan's first legs would be demonstrating a reach that already happened, and
    # a stage-3 seed starts inside the success threshold. Demonstrations have to
    # be of the whole task from a stage-0 start.
    env_cfg = dict(config.get("env", {}), curriculum_level=0)
    env = make_env_fn(env_cfg, args.seed, 0)()
    base: ManipulationEnv = env.unwrapped

    if isinstance(env, TaskSpaceWrapper):
        expert: TaskSpaceExpert | TorqueExpert = TaskSpaceExpert(env)
    elif base.control_mode == "torque":
        expert = TorqueExpert(base)
    else:
        raise SystemExit(
            f"no scripted expert for control_mode={base.control_mode!r} without the "
            f"task-space wrapper; enable env.task_space or use control_mode: torque"
        )
    LOGGER.info("expert       %s", type(expert).__name__)
    LOGGER.info("obs/action   %s / %s",
                env.observation_space.shape, env.action_space.shape)

    max_attempts = args.max_attempts or 5 * args.episodes
    trajectories: list[Trajectory] = []
    outcomes: dict[str, int] = {}
    attempt = 0
    while len(trajectories) < args.episodes and attempt < max_attempts:
        traj, outcome = record_episode(
            env, expert, seed=args.seed + attempt, require_lift=args.require_lift
        )
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        attempt += 1
        if traj is not None:
            trajectories.append(traj)
        if attempt % 10 == 0:
            LOGGER.info("attempt %d: %d/%d kept", attempt, len(trajectories), args.episodes)

    env.close()
    if not trajectories:
        raise SystemExit(f"no successful demonstrations in {attempt} attempts: {outcomes}")

    observations = np.concatenate([np.stack(t.observations) for t in trajectories])
    actions = np.concatenate([np.stack(t.actions) for t in trajectories])
    rewards = np.concatenate([np.asarray(t.rewards, dtype=np.float32) for t in trajectories])
    lengths = np.asarray([len(t) for t in trajectories], dtype=np.int64)
    # Episode boundaries, so the value pretraining in pretrain_bc.py can discount
    # within an episode instead of across the concatenation seam.
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        observations=observations,
        actions=actions,
        rewards=rewards,
        episode_starts=starts,
        episode_lengths=lengths,
        control_mode=np.array(base.control_mode),
        task_space=np.array(isinstance(env, TaskSpaceWrapper)),
        obs_dim=np.array(observations.shape[1]),
        action_dim=np.array(actions.shape[1]),
    )
    LOGGER.info(
        "kept %d/%d attempts (%.0f%%), %d transitions -> %s",
        len(trajectories), attempt, 100 * len(trajectories) / attempt,
        len(actions), args.out,
    )
    LOGGER.info("mean episode length %.1f steps", float(lengths.mean()))
    LOGGER.info("outcomes %s", dict(sorted(outcomes.items())))
    if getattr(expert, "ik_failures", 0):
        LOGGER.info("expert IK failures %d", expert.ik_failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
