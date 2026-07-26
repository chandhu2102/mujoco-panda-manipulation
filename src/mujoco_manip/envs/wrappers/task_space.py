"""Task-space (delta-position) action wrapper.

Turns the 8-D joint-level action space of ``ManipulationEnv`` into a 4-D
end-effector one::

    [delta_x, delta_y, delta_z, delta_gripper_width]

The wrapper resolves each delta into joint angles with ``solve_site_ik`` and
hands them through the env's existing ``[-1, 1]^8`` action space, so the reward,
curriculum, observation layout, info dict and vector-env plumbing are all
untouched. The env must be constructed with ``control_mode="joint_position"``;
that is the seam the IK output lands on.

Why this is a different learning problem, not just a different parameterization
-------------------------------------------------------------------------------

Under torque control the policy has to *discover* the map from 7 joint torques to
tool-centre-point motion, and it has to do so through a double integration, so
zero-mean exploration noise produces an unbounded random walk in position. The
grasp phase then asks that random walk to satisfy a three-way conjunction
(``is_grasped``: both pads in contact, jaw width inside a 2.6 cm band, object
enclosed between the pads) within a few centimetres. Nothing about the torque
parameterization makes that conjunction more likely than chance.

Here the map is inverted analytically and once. Exploration noise is a
displacement of the tool centre point in metres, bounded by ``max_delta_pos`` per
step, and "hold still" is the single action ``0`` rather than a
state-dependent torque the policy must learn to compute. The precision the grasp
needs is supplied by the servo, not by the policy.

What it costs
-------------

The controller is now part of the solution rather than part of the problem, so a
policy trained here has learned less about the robot: it cannot exploit dynamics,
it cannot move faster than the servo tracks, and it is bounded by the IK's
reachability rather than the arm's. That is the trade being made deliberately.
Orientation is also no longer the policy's to choose -- see ``wrist_yaw``.
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ...robots.controllers.ik import solve_site_ik
from ...robots.panda import (
    GRIPPER_MAX_WIDTH,
    JOINT_POSITION_LIMITS,
    JOINT_VELOCITY_LIMITS,
)
from ..manipulation_env import ManipulationEnv, nearest_face_yaw, top_down_grasp_quat

__all__ = ["TaskSpaceWrapper", "WristYawMode"]

_LOGGER = logging.getLogger(__name__)

WristYawMode = str
"""One of ``"track_object"``, ``"fixed"``, ``"action"``. See ``TaskSpaceWrapper``."""

_YAW_MODES: tuple[str, ...] = ("track_object", "fixed", "action")


class TaskSpaceWrapper(gym.Wrapper):
    """4-D end-effector deltas in, 8-D joint targets out.

    Args:
        env: A ``ManipulationEnv`` (or subclass) built with
            ``control_mode="joint_position"``.
        max_delta_pos: Metres of commanded tool-centre-point displacement at
            ``|action| = 1``, per control step. The policy's action stays in
            ``[-1, 1]`` -- *not* raw metres -- so the observation normalizer, the
            Gaussian head's initial sigma and ``policy.action_std_ceiling`` all
            keep the meanings they were tuned with. At the default 0.012 m and a
            25 Hz control rate this is a 0.30 m/s tool speed at saturation.
        max_delta_width: Metres of commanded jaw-width change at ``|action| = 1``.
            The default closes the full 8 cm jaw in 8 steps (~0.32 s).
        workspace_low / workspace_high: Clamp on the *target*, in metres. Keeps
            the IK from being asked for poses outside the region the scene was
            verified reachable and collision-free in.
        wrist_yaw: How the jaw axis is chosen.

            * ``"track_object"`` (default) -- yaw follows the object's nearest
              square face. Keeps the action space at 4-D *and* keeps the grasp
              geometrically possible: ``is_grasped`` accepts a jaw width up to
              ``object_width + 0.012 = 0.052`` m, but a 4 cm cube gripped across
              its diagonal measures ``0.04 * sqrt(2) = 0.057`` m. With
              ``PICK_PLACE_RANDOMIZATION``'s +-45 degree yaw range, a fixed jaw
              would therefore be unable to grasp at all near the ends of that
              range, whatever the policy does. Note this reads the object's pose
              in the controller, so part of the alignment problem is solved for
              the policy rather than by it -- the same privileged information is
              already in the observation, but the *credit* for using it is not
              the policy's any more.
            * ``"fixed"`` -- constant ``fixed_yaw``. Honest, and only workable if
              object yaw randomization is narrowed to roughly +-15 degrees.
            * ``"action"`` -- a fifth action dimension commands a yaw delta.
              Keeps the alignment in the policy at the cost of one more
              dimension to explore.
        delta_gripper: True (default) integrates the gripper command, matching
            the ``delta_gripper_width`` action name. False makes dimension 3 an
            *absolute* width, which removes a hidden integrator state and is
            usually the easier of the two to learn -- the policy no longer has to
            track a commanded width it cannot directly observe.
        effort_penalty: Coefficient on ``sum(delta_action ** 2)``, subtracted from
            the env's reward. This is where the action penalty has to live under
            joint-position control; see ``_check_reward_config``.
        ik_max_iterations / ik_pos_tolerance / ik_rot_tolerance: Passed to
            ``solve_site_ik``. The solve is warm-started from the current arm pose
            and the target is at most ``max_delta_pos`` away, so it converges in a
            handful of iterations; the low iteration cap bounds the worst case
            near a singularity rather than being a tuning knob. The rotation
            tolerance matches ``CURRICULUM_IK_ROT_TOLERANCE`` for the reason
            recorded there -- the Panda cannot hit an exact top-down frame at
            every yaw across the workspace, so position converges to well under a
            millimetre while the wrist sits a few degrees off, and a tighter
            tolerance reports converged solves as failures.
        posture_gain: Null-space pull toward ``q_nominal`` inside the IK. Zero
            disables it; see ``solve_site_ik`` for the failure it prevents.
        branch_step_margin: Multiple of one control step's maximum joint travel
            that a solve may command. See ``branch_step_limit``.
    """

    def __init__(
        self,
        env: ManipulationEnv,
        *,
        max_delta_pos: float = 0.012,
        max_delta_width: float = 0.010,
        workspace_low: tuple[float, float, float] = (0.30, -0.28, 0.42),
        workspace_high: tuple[float, float, float] = (0.68, 0.28, 0.75),
        leash: float = 0.05,
        wrist_yaw: WristYawMode = "track_object",
        fixed_yaw: float = 0.0,
        max_delta_yaw: float = 0.10,
        delta_gripper: bool = True,
        effort_penalty: float = 0.01,
        ik_max_iterations: int = 24,
        ik_pos_tolerance: float = 1e-3,
        ik_rot_tolerance: float = 0.15,
        posture_gain: float = 0.15,
        branch_step_margin: float = 3.0,
    ) -> None:
        super().__init__(env)

        base = env.unwrapped
        if not isinstance(base, ManipulationEnv):
            raise TypeError(
                f"TaskSpaceWrapper expects a ManipulationEnv, got {type(base).__name__}"
            )
        if base.control_mode != "joint_position":
            raise ValueError(
                "TaskSpaceWrapper requires control_mode='joint_position' (the IK "
                f"output is a joint *angle* target); env has {base.control_mode!r}"
            )
        if wrist_yaw not in _YAW_MODES:
            raise ValueError(f"wrist_yaw must be one of {list(_YAW_MODES)}, got {wrist_yaw!r}")

        self.base = base
        self.max_delta_pos = float(max_delta_pos)
        self.max_delta_width = float(max_delta_width)
        self.workspace_low = np.asarray(workspace_low, dtype=np.float64)
        self.workspace_high = np.asarray(workspace_high, dtype=np.float64)
        self.leash = float(leash)
        self.wrist_yaw = wrist_yaw
        self.fixed_yaw = float(fixed_yaw)
        self.max_delta_yaw = float(max_delta_yaw)
        self.delta_gripper = bool(delta_gripper)
        self.effort_penalty = float(effort_penalty)
        self.ik_max_iterations = int(ik_max_iterations)
        self.ik_pos_tolerance = float(ik_pos_tolerance)
        self.ik_rot_tolerance = float(ik_rot_tolerance)
        self.posture_gain = float(posture_gain)

        # Nominal posture for the IK's null-space term: the pose the episode
        # starts from, so the secondary objective pulls toward the elbow-up branch
        # the scene was verified collision-free in.
        self.q_nominal = self.base.robot.safe_reset_qpos(self.base.reset_pose).copy()

        # Largest per-joint move a solve may command in one control step, as a
        # multiple of what the joint can physically travel in that time
        # (JOINT_VELOCITY_LIMITS * control_dt). The margin exists because a
        # position target legitimately *leads* the arm; what it catches is a solve
        # that jumped branches, which shows up as a step of order a radian rather
        # than the ~0.09 rad a joint can actually cover in 40 ms.
        self.branch_step_limit = (
            float(branch_step_margin) * JOINT_VELOCITY_LIMITS * self.base.control_dt
        )
        self.branch_rejections = 0

        self._check_reward_config()

        n_actions = 5 if wrist_yaw == "action" else 4
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32
        )

        # An integrating target is controller state the base observation does not
        # contain, so the MDP the policy sees would not be Markov without it: the
        # same joint configuration behaves differently depending on how far the
        # target has run ahead. Three extra numbers, as the *error* rather than the
        # target itself -- the error is what the servo acts on, and it is already
        # scale-free relative to `leash`.
        base_dim = int(np.prod(env.observation_space.shape))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_dim + 3,), dtype=np.float32
        )
        self._target_pos = np.zeros(3, dtype=np.float64)

        # Reused across every solve; see solve_site_ik's `scratch` argument for
        # why this is not a per-call allocation.
        self._ik_scratch = mujoco.MjData(base.model)

        self._commanded_width = GRIPPER_MAX_WIDTH
        self._commanded_yaw = self.fixed_yaw

        # Diagnostics. Public and counted rather than logged per occurrence: a
        # controller that has quietly stopped converging looks exactly like a
        # policy that has stopped improving.
        self.ik_solves = 0
        self.ik_failures = 0
        self._ik_warned = False

    # -- configuration guards ---------------------------------------------- #

    def _check_reward_config(self) -> None:
        """Refuse an ``action_penalty`` that no longer measures what it names.

        ``ManipulationEnv.compute_reward`` is handed the *inner* 8-D action, which
        under ``joint_position`` control is a vector of normalized joint *angles*.
        ``action_penalty * sum(action ** 2)`` is then a penalty on how far the arm
        is commanded from the centre of its joint ranges -- a pose prior, not an
        effort cost, and one that pulls hardest exactly where a top-down grasp
        needs the elbow. It would train, log and plot as an action penalty
        throughout.

        Raising rather than zeroing it silently: the value has to change in the
        config so that ``config.resolved.json`` records what the run actually did.
        """
        penalty = float(getattr(self.base.reward_cfg, "action_penalty", 0.0))
        if penalty != 0.0:
            raise ValueError(
                "TaskSpaceWrapper requires reward.action_penalty = 0: under "
                "control_mode='joint_position' the env's action penalty is applied "
                f"to normalized joint angles, not to effort (got {penalty}). Set "
                "env.reward.action_penalty: 0.0, remove any "
                "anneal.penalties.action_penalty block (the annealer would write "
                "it back mid-run), and use TaskSpaceWrapper(effort_penalty=...) "
                "to penalize the task-space delta instead."
            )

    # -- action translation ------------------------------------------------ #

    def _target_yaw(self, yaw_action: float) -> float:
        if self.wrist_yaw == "fixed":
            return self.fixed_yaw
        if self.wrist_yaw == "track_object":
            # The object's own yaw about z, recovered from its quaternion, wrapped
            # to the nearest square face.
            quat = self.base.object_quat
            yaw = 2.0 * float(np.arctan2(quat[3], quat[0]))
            return nearest_face_yaw(yaw)
        self._commanded_yaw = float(
            np.clip(
                self._commanded_yaw + yaw_action * self.max_delta_yaw,
                -np.pi / 4.0,
                np.pi / 4.0,
            )
        )
        return self._commanded_yaw

    def _inner_action(self, action: np.ndarray) -> np.ndarray:
        """Translate a task-space action into the env's 8-D joint-level action."""
        action = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"action shape {action.shape} != {self.action_space.shape}"
            )

        # The target integrates the deltas; it is not re-derived from the current
        # tool-centre-point position each step. That distinction is the whole
        # controller, and getting it wrong is silent:
        #
        # Setting `target = eef_pos + delta` looks equivalent and is not, because
        # the servo lags its target by roughly `speed / omega_n`. Re-reading
        # `eef_pos` every step makes the target chase the lagging arm, so the loop
        # has *zero positional stiffness at the task level* -- there is nothing
        # for the arm to converge to. Measured on this scene: `delta = 0` held for
        # 10 steps drifted the tool centre point 6.3 cm, and a commanded 12 cm
        # descent produced 3.4 cm of actual motion plus 3 cm of uncommanded
        # lateral drift, because most of each delta was absorbed by the target
        # resetting to wherever the arm had got to.
        #
        # Integrating instead means the target advances by exactly `delta` per
        # step whatever the arm is doing, and the servo's lag becomes a fixed
        # phase delay rather than a loss of gain.
        target = self._target_pos + action[:3] * self.max_delta_pos
        target = np.clip(target, self.workspace_low, self.workspace_high)
        # Leash. An integrating target run against an obstacle -- the table, the
        # object, a joint limit -- would otherwise wind up arbitrarily far away and
        # take as many steps to unwind as it took to accumulate. Bounding the
        # target to `leash` metres of the actual tool centre point caps that recovery
        # at one step. The leash has to exceed the steady-state tracking lag
        # (~speed / omega_n, i.e. ~1.2 cm at full commanded speed) or it would clamp
        # during ordinary fast motion and quietly cap the reachable tool speed.
        eef = self.base.robot.eef_pos
        target = eef + np.clip(target - eef, -self.leash, self.leash)
        self._target_pos = target
        target_pos = target
        yaw = self._target_yaw(action[4] if self.wrist_yaw == "action" else 0.0)

        result = solve_site_ik(
            self.base.model,
            self.base.robot.idx.eef_site_id,
            self.base.robot.idx.arm_qpos_adr,
            self.base.robot.idx.arm_dof_adr,
            target_pos,
            target_quat=top_down_grasp_quat(yaw),
            q_init=self.base.robot.arm_qpos,
            joint_limits=JOINT_POSITION_LIMITS,
            qpos_full=self.base.data.qpos,
            max_iterations=self.ik_max_iterations,
            pos_tolerance=self.ik_pos_tolerance,
            rot_tolerance=self.ik_rot_tolerance,
            scratch=self._ik_scratch,
            q_nominal=self.q_nominal,
            posture_gain=self.posture_gain,
        )
        self.ik_solves += 1
        if not result:
            # Use the partial solve anyway. It is the closest reachable pose the
            # solver found, so commanding it moves the arm the achievable part of
            # the way; freezing instead would make the workspace boundary a wall
            # the policy gets no gradient through, and raising would kill a run
            # over one unreachable target.
            self.ik_failures += 1
            if not self._ik_warned:
                self._ik_warned = True
                _LOGGER.warning(
                    "task-space IK did not converge (pos_error=%.4f m, "
                    "rot_error=%.4f rad); using the partial solve. Further "
                    "failures are counted in ik_failures (now %d/%d) and not "
                    "logged.",
                    result.pos_error, result.rot_error,
                    self.ik_failures, self.ik_solves,
                )

        # Gripper. GRIPPER_MAX_WIDTH is the full jaw opening and apply_gripper
        # maps [-1, 1] onto the per-finger travel, which is half of it -- so the
        # normalized command for a width w is 2w / GRIPPER_MAX_WIDTH - 1.
        if self.delta_gripper:
            self._commanded_width = float(
                np.clip(
                    self._commanded_width + action[3] * self.max_delta_width,
                    0.0,
                    GRIPPER_MAX_WIDTH,
                )
            )
            width = self._commanded_width
        else:
            width = 0.5 * (action[3] + 1.0) * GRIPPER_MAX_WIDTH
        gripper_action = 2.0 * width / GRIPPER_MAX_WIDTH - 1.0

        # Branch-flip guard. The posture term makes this rare, but a solve near a
        # singularity can still return a configuration on the far side of the
        # self-motion manifold. Commanding it would slam the arm across that gap
        # at whatever torque the servo can muster; clamping the *step* keeps the
        # commanded pose on the near side and lets the next solve continue from
        # there.
        q_now = self.base.robot.arm_qpos
        q_target = np.clip(
            result.qpos,
            q_now - self.branch_step_limit,
            q_now + self.branch_step_limit,
        )
        if not np.allclose(q_target, result.qpos):
            self.branch_rejections += 1

        inner = np.empty(self.base.action_space.shape, dtype=np.float64)
        inner[:-1] = self.base.robot.normalize_arm_qpos(q_target)
        inner[-1] = gripper_action
        return inner

    # -- gym API ----------------------------------------------------------- #

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        """Append the servo's position error to the base observation."""
        error = self._target_pos - self.base.robot.eef_pos
        return np.concatenate(
            [np.asarray(obs, dtype=np.float64).ravel(), error]
        ).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        # Start the target *at* the tool centre point, so step 1's error is zero
        # and a zero action is genuinely "stay put" rather than a jump to some
        # carried-over target.
        self._target_pos = self.base.robot.eef_pos.copy()
        # Read the jaw back from the state the env (possibly a curriculum seed)
        # actually left it in, rather than assuming wide open: a clamped stage-2
        # or stage-3 seed starts the episode already gripping, and re-commanding
        # a full-open width on step 1 would drop the object the seed just handed
        # over.
        self._commanded_width = float(self.base.robot.gripper_width)
        self._commanded_yaw = self.fixed_yaw
        return self._augment(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        task_action = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(
            self._inner_action(task_action)
        )
        obs = self._augment(obs)

        # The effort penalty the env can no longer compute for itself, on the
        # action the policy actually chose.
        penalty = self.effort_penalty * float(np.sum(np.square(task_action)))
        reward = float(reward) - penalty
        info = dict(info)
        info["reward/action_penalty"] = -penalty
        info["control/ik_failure_rate"] = self.ik_failures / max(1, self.ik_solves)
        info["control/commanded_width"] = self._commanded_width
        return obs, reward, terminated, truncated, info
