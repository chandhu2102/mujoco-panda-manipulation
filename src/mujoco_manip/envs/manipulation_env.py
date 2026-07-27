"""Shared manipulation env: robot + objects + controller + reward composition."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ..robots.controllers.ik import solve_site_ik
from ..robots.panda import (
    GRIPPER_MAX_WIDTH,
    JOINT_POSITION_LIMITS,
    JOINT_VELOCITY_LIMITS,
    PandaRobot,
)

__all__ = [
    "ManipulationEnv",
    "RewardConfig",
    "RandomizationConfig",
    "DEFAULT_SCENE",
    "ControlMode",
    "CONTROL_MODES",
    "top_down_grasp_quat",
    "grasp_quat_candidates",
    "nearest_face_yaw",
]

_LOGGER = logging.getLogger(__name__)

ControlMode = Literal["torque", "velocity", "joint_position"]

CONTROL_MODES: tuple[str, ...] = ("torque", "velocity", "joint_position")
"""Valid ``control_mode`` values, as one list rather than a repeated literal.

The three differ in what the arm half of the action *means*, which is not a
cosmetic difference:

* ``torque`` -- normalized joint torque. Zero coasts (gravity is compensated
  separately), so the map from actions to tool-centre-point path is a double
  integration and zero-mean exploration noise produces an unbounded random walk
  in position.
* ``velocity`` -- normalized target joint velocity, closed by a P law. Zero
  brakes. One integration, so the same noise produces bounded position jitter.
* ``joint_position`` -- normalized target joint *angle*, closed by a PD law. Zero
  commands the centre of every joint's range, so ``RewardConfig.action_penalty``
  no longer measures effort; see ``PandaRobot.apply_joint_position``. This is the
  mode ``envs.wrappers.task_space.TaskSpaceWrapper`` sits on.
"""

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE = _REPO_ROOT / "assets" / "panda_scene.xml"

# Grasp-predicate tolerances. See ManipulationEnv.is_grasped for why the width
# needs a lower bound and not just an upper one.
GRASP_WIDTH_MIN_RATIO: float = 0.5
"""Jaw must be held at least this fraction of the object's width open."""
GRASP_WIDTH_TOLERANCE: float = 0.012
"""Slack above the object width, for pads that have closed but not compressed."""
GRASP_LATERAL_TOLERANCE: float = 0.01
"""Allowed offset of the object centre from the jaw axis, beyond its half-width."""

OBSTACLE_GEOM_PREFIX: str = "obstacle"
"""Geoms whose name starts with this are barriers the arm is penalized for hitting.

A prefix rather than one exact name, so a scene can add ``obstacle_left`` /
``obstacle_right`` without touching this module. Matching nothing is legal and
means "this scene has no obstacle": ``RewardConfig.collision_penalty`` then has
nothing to multiply and every existing task keeps its reward untouched. See
``ManipulationEnv.obstacle_contacts``.
"""

# --------------------------------------------------------------------------- #
# Start-state (reverse) curriculum
# --------------------------------------------------------------------------- #
#
# Stage 0 is the honest task; stages 1-3 hand the policy a state progressively
# further along the reach -> grasp -> lift -> place chain, so the reward past a
# stage it cannot yet reach is still observable. ``curriculum_level`` sets the
# *highest* stage an env may seed; the stage actually used is drawn per reset.
# See ManipulationEnv._apply_curriculum_seed.

MAX_CURRICULUM_LEVEL: int = 3
_CURRICULUM_LEVELS: frozenset[int] = frozenset(range(MAX_CURRICULUM_LEVEL + 1))

CURRICULUM_PROB_START: float = 0.80
"""Probability of seeding at ``progress = 0``."""
CURRICULUM_PROB_DECAY: float = 0.75
"""Amount the probability decays over the full run."""
CURRICULUM_PROB_FLOOR: float = 0.05
"""Probability floor. Nonzero so late training keeps a trickle of seeded
episodes: they are the only states that exercise the end of the chain once the
policy still cannot reliably grasp, and they keep the value function's estimate
of the later stages from going stale."""

CURRICULUM_HOVER_HEIGHT: float = 0.03
"""Stage 1 pre-grasp clearance above the object centre, in metres."""

CURRICULUM_GOAL_OFFSET: float = 0.075
"""Stage 3 displacement from the goal, in metres.

Stage 3 has to start *near* the goal, not *at* it. Placing the object exactly on
the goal put 198/200 stage-3 resets inside the success threshold on step 0, which
poisons two things at once: the episode latches ``is_success`` before the policy
acts, and the per-step success bonus (10.0/step, ~2500 over a 250-step horizon)
is collected for holding still -- so returns become bimodal by start stage and
advantage normalisation mixes two populations.

7.5 cm clears both thresholds in play by a wide margin: the base
``RewardConfig.success_threshold`` is 0.05 m and ``PickPlaceRewardConfig``
tightens it to 0.04 m, so this leaves a genuine final approach either way.
"""

CURRICULUM_GOAL_OFFSET_ATTEMPTS: int = 6
"""Offset directions tried before a stage-3 seed gives up and rolls back.

A displacement from a goal near the edge of the workspace can land outside it,
and rejecting those with a single draw biased stage 3 away from exactly the
hardest placements. Redrawing the direction -- never the magnitude -- and letting
the IK decide reachability cut the stage-3 rollback rate from 6.5% back to
roughly the un-offset baseline.
"""

START_COLLISION_ATTEMPTS: int = 8
"""Arm-pose redraws allowed before a reset falls back to the un-noised pose.

Generous on purpose. On the shipped geometry the per-draw rejection rate measures
0/2000; at the tightest wall height tried it was 2.15%, where eight redraws leaves
a ~1e-13 chance of reaching the fallback. So the budget is sized for a wall
noticeably closer to the reset pose than the current one, which is the case where
this matters. See ``ManipulationEnv._reject_start_collisions``.
"""

CURRICULUM_SETTLE_STEPS: int = 8
"""Physics substeps run with the jaw commanded shut, to establish contact."""
CURRICULUM_GRASP_WIDTH_RATIO: float = 0.95
"""Initial jaw width as a fraction of the object width, before settling.

Just under 1.0 so the pads start in light contact rather than needing the
settle loop to close a visible gap -- but comfortably above
``GRASP_WIDTH_MIN_RATIO``, so the squeeze has room to compress without
falling out of the band ``is_grasped`` checks.
"""

CURRICULUM_IK_ROT_TOLERANCE: float = 0.15
"""Orientation tolerance for the seeding solve, in radians (~8.6 degrees).

Loose on purpose. The Panda cannot hit an exact top-down frame at every yaw
across the spawn box -- position converges to under a millimetre while the wrist
is left a few degrees off -- and at the solver's default 1e-2 rad that rejected
20% of otherwise perfect seeds. What decides a seed is whether the resulting
grip satisfies ``is_grasped``, which ``_apply_curriculum_seed`` verifies
directly, so this only needs to be tight enough to keep the jaw on a face
rather than a diagonal: at 0.15 rad the jaw spans ``w * (cos + sin) = 0.0455`` m
against a band upper bound of 0.052 m.
"""


def top_down_grasp_quat(yaw: float) -> np.ndarray:
    """Quaternion for a vertical top-down grasp with the jaw axis at ``yaw``.

    Built as an explicit rotation matrix because the axis convention is the part
    that goes wrong silently: local ``z`` is the approach direction (down), local
    ``y`` is the jaw axis, and local ``x`` closes the right-handed set. Verified
    against the ``ready_low`` frame, whose jaw axis is world +y and whose
    ``eef_site`` sits at the finger-tip midpoint.
    """
    local_z = np.array([0.0, 0.0, -1.0])
    local_y = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    local_x = np.cross(local_y, local_z)
    mat = np.column_stack([local_x, local_y, local_z])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.ravel())
    return quat


def grasp_quat_candidates(object_yaw: float) -> list[np.ndarray]:
    """The four wrist orientations that grip a square object face-on.

    Ordered nearest-first: the wrapped yaw needs the least wrist travel from
    ``ready_low``, and the quarter turns are only reached for when that one is
    kinematically out of range.

    At module scope because a scripted controller needs the same four candidates
    for the same reason the curriculum does. Trying only the nearest one puts
    joint7 at its limit for some object yaws -- the curriculum records losing 5/60
    seeds that way -- and for a *servo* the consequence is worse than a rejected
    seed: a 2 rad wrist error saturates the wrist torque from step one, and the
    resulting inertial kick is what tips a critically-damped joint-space PD law
    into bang-bang.
    """
    yaw = nearest_face_yaw(object_yaw)
    offsets = (0.0, np.pi / 2.0, -np.pi / 2.0, np.pi)
    return [top_down_grasp_quat(yaw + off) for off in offsets]


def nearest_face_yaw(yaw: float) -> float:
    """Wrap ``yaw`` to the nearest square face, i.e. into ``[-pi/4, pi/4)``.

    A square face repeats every 90 degrees, so this is the jaw yaw that grips the
    same physical face with the least wrist travel. Factored out of
    ``ManipulationEnv._grasp_quat_candidates`` because a task-space controller
    needs exactly the same wrap to keep the jaw off the cube's diagonal, and
    duplicating the modular arithmetic is how the two silently disagree.
    """
    return ((float(yaw) + np.pi / 4.0) % (np.pi / 2.0)) - np.pi / 4.0


@dataclass(frozen=True)
class _StageSpec:
    """How one curriculum stage differs from the others.

    Three knobs cover all of stages 1-3, which is the point of naming them:
    where the object goes, where the tool centre point goes relative to it, and
    whether the jaw closes on it.
    """

    name: str
    hover: float
    """TCP clearance above the object centre, in metres."""
    at_goal: bool
    """Relocate the object onto the sampled goal before solving."""
    clamp: bool
    """Squeeze the jaw shut and require ``is_grasped`` to agree afterwards."""


_CURRICULUM_STAGE_SPECS: dict[int, _StageSpec] = {
    # Pure grasp. Jaw wide open, hovering clear of the object, so the only thing
    # left to learn is the descent and the close. Deliberately *not* clamped:
    # a stage-1 reset must start un-grasped or there is nothing to discover.
    1: _StageSpec("pre_grasp", hover=CURRICULUM_HOVER_HEIGHT, at_goal=False, clamp=False),
    # Lift and travel. Gripping the object where it spawned, still on the table,
    # so the lift is genuinely ahead of the policy rather than already done.
    2: _StageSpec("grasped_on_table", hover=0.0, at_goal=False, clamp=True),
    # Hover and drop. Gripping the object at the goal, so only the hold (and
    # release, once the reward asks for one) remains.
    3: _StageSpec("grasped_at_goal", hover=0.0, at_goal=True, clamp=True),
}


@dataclass
class RewardConfig:
    """Weights for the dense shaping terms.

    Every distance term is passed through ``1 - tanh(k * d)`` rather than used as
    a raw ``-d``. Two reasons: the term is bounded in ``[0, 1]`` so no single
    stage can dominate the sum once the workspace is large, and its gradient is
    steepest near contact where precision actually matters. Raw negative
    distance has the opposite profile -- flat where you need resolution, and
    unbounded far away, which is what makes naive shaping reward-hack into
    "hover near the object forever".
    """

    reach: float = 1.0
    """Weight on gripper-to-object proximity."""
    grasp: float = 2.0
    """Bonus while both pads contact the object."""
    lift: float = 2.0
    """Weight on height gained above the table."""
    place: float = 4.0
    """Weight on object-to-goal proximity, gated on having grasped."""
    success: float = 10.0
    """One-off bonus on the success condition."""

    action_penalty: float = 0.01
    """L2 penalty on the action, to discourage bang-bang torque chatter."""
    velocity_penalty: float = 0.005
    """L2 penalty on joint velocity, for smoother trajectories."""
    joint_limit_penalty: float = 1.0

    collision_penalty: float = 0.0
    """Charged per step while any arm or finger geom touches an obstacle geom.

    **Zero by default, and that default is load-bearing.** Every task and config
    written before the obstacle scene existed keeps a bit-identical reward: a
    scene with no ``obstacle*`` geom has nothing to charge for, and a scene that
    has one still charges nothing until a config asks. Only
    ``configs/train/pick_place_obstacle.yaml`` turns it on.

    A *flat per-step* charge, not a one-off, and not scaled by contact count or
    penetration depth. Contact count is a solver artefact -- a box-on-box contact
    resolves to between one and four points depending on the incidence angle, so
    scaling by it would make the same physical scrape cost 1x to 4x for reasons
    the policy cannot observe or control.

    On magnitude. This is a per-step charge against a per-step dense reward whose
    full stage sum is ~19 (see ``PickPlaceRewardConfig.success``), so 5.0 makes a
    single step of contact cost more than the entire grasp-plus-lift stack, and
    over a 250-step horizon a stuck-against-the-wall episode accrues -1250. That
    is the intent -- but it is also the shape of a penalty that can teach
    avoidance of the *workspace* rather than of the wall, because the object spawn
    box in the obstacle task sits 6.5 cm from the barrier and the shaping term
    pulling the gripper there is worth at most 1.0/step. If reach reward collapses
    early in a run, that is the trade going the wrong way; anneal this up from
    ~1.0 with the ``anneal.penalties`` block rather than lowering it after the
    fact, since the field is reachable through ``set_reward_weights``.
    """

    reach_sharpness: float = 10.0
    """``k`` in the reach tanh. 10 gives most of the gradient inside ~20 cm."""
    place_sharpness: float = 10.0

    lift_target: float = 0.12
    """Height above the table counted as fully lifted, in metres."""
    success_threshold: float = 0.05
    """Object-to-goal distance counted as success, in metres."""
    require_grasp_for_success: bool = False
    """When True, success also demands both pads still be in contact."""


@dataclass
class RandomizationConfig:
    """Per-reset randomization ranges.

    Randomizing the object and goal each reset is what stops the policy from
    memorizing one trajectory. Note this buys generalization *across object
    placements in this model* -- it is not by itself sim-to-real transfer, which
    additionally needs dynamics/visual randomization and a mesh-accurate model.
    """

    object_x: tuple[float, float] = (0.42, 0.60)
    object_y: tuple[float, float] = (-0.16, 0.16)
    object_yaw: tuple[float, float] = (-np.pi, np.pi)

    goal_x: tuple[float, float] = (0.42, 0.60)
    goal_y: tuple[float, float] = (-0.16, 0.16)
    goal_z: tuple[float, float] = (0.48, 0.62)
    """Goal height range. Above the table top, so the task requires a lift."""

    arm_noise: float = 0.05
    """Uniform per-joint radians added to the reset pose."""
    min_object_goal_distance: float = 0.05
    """Rejection-sample goals at least this far from the object's start."""

    randomize_object: bool = True
    randomize_goal: bool = True


@dataclass
class _SceneIndices:
    object_body_id: int
    object_geom_id: int
    object_joint_qpos_adr: int
    object_joint_dof_adr: int
    goal_site_id: int
    table_geom_id: int
    table_top_z: float = field(default=0.4)
    obstacle_geom_ids: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    """Barrier geoms, empty for scenes without one. See ``OBSTACLE_GEOM_PREFIX``."""


class ManipulationEnv(gym.Env):
    """Gymnasium env wrapping the Panda + table + one graspable object.

    Task: bring the cube to the randomized goal marker above the table. The dense
    reward decomposes into reach -> grasp -> lift -> place stages so that a
    policy gets gradient before it ever completes the task.

    Observations are proprioception plus object and goal state; see
    ``_observation`` for the exact layout. Actions are ``[-1, 1]^8``: seven arm
    commands (torque, velocity or joint angle, per ``control_mode`` -- see
    ``CONTROL_MODES``) plus one gripper width.

    ``terminate_on_success`` defaults to **False**, and changing it needs care.
    Ending the episode on success forfeits every remaining step of a per-step
    dense reward, so unless the success bonus exceeds that forfeited sum, the
    return-maximizing behaviour is to hover just outside the success threshold
    and farm the shaping term for the full horizon. Measured on the reach task
    with a 150-step horizon: hovering 9 cm out scored 5.0x an actual success, and
    a training run duly converged to it -- deterministic eval success fell from
    45% to 0% while the shaped return kept climbing. With termination off,
    reaching and holding is worth ~4.4x hovering and the pathology disappears.
    """

    metadata = {"render_modes": ["rgb_array", "depth_array"], "render_fps": 25}

    def __init__(
        self,
        scene_path: str | Path = DEFAULT_SCENE,
        *,
        control_mode: ControlMode = "torque",
        control_dt: float = 0.04,
        gripper_rate_limit: float | None = None,
        max_episode_steps: int = 250,
        reward_config: RewardConfig | None = None,
        randomization: RandomizationConfig | None = None,
        reset_pose: str = "ready_low",
        gravity_compensation: bool = True,
        terminate_on_success: bool = False,
        curriculum_level: int = 0,
        render_mode: str | None = None,
        camera_name: str = "frontview",
        seed: int | None = None,
    ) -> None:
        super().__init__()

        scene_path = Path(scene_path)
        if not scene_path.is_file():
            raise FileNotFoundError(f"MJCF scene not found: {scene_path}")
        if control_mode not in CONTROL_MODES:
            raise ValueError(
                f"control_mode must be one of {list(CONTROL_MODES)}, got {control_mode!r}"
            )
        if control_mode == "joint_position" and not gravity_compensation:
            # The PD law has finite stiffness, so an uncompensated bias term
            # becomes a standing position error of qfrc_bias / kp on every joint --
            # a silent constant offset between the commanded pose and the reached
            # one, which a task-space controller would then chase forever.
            raise ValueError(
                "control_mode='joint_position' requires gravity_compensation=True; "
                "without it the servo holds a pose only at a standing error of "
                "qfrc_bias / kp"
            )
        if curriculum_level not in _CURRICULUM_LEVELS:
            raise ValueError(
                f"curriculum_level must be one of {sorted(_CURRICULUM_LEVELS)} "
                f"(the highest stage this env may seed), got {curriculum_level!r}"
            )

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        self.control_mode: ControlMode = control_mode
        if gripper_rate_limit is not None and gripper_rate_limit <= 0.0:
            raise ValueError(
                f"gripper_rate_limit must be positive or None, got {gripper_rate_limit}"
            )
        self.gripper_rate_limit = gripper_rate_limit
        """Maximum change in commanded jaw width per control step, in metres.

        ``None`` reproduces the original behaviour: the gripper action is written
        straight to the finger position actuators, so a step from +1 to -1 commands
        the full 8 cm of jaw travel within one control interval. Measured on this
        scene, that closes the jaw from 0.080 m to 0.005 m in a single 40 ms step
        -- a closing speed near 1.9 m/s -- which does not grasp a resting cube, it
        ejects it: pad contact registers on both fingers while the width passes
        straight through the 0.02-0.052 m band ``is_grasped`` requires, and the
        cube is left displaced with the jaw shut on nothing.

        This matters well beyond the scripted expert that exposed it. Under torque
        control the policy's gripper dimension is written through the same path, so
        a policy exploring with sigma near 1.0 slams the jaw at that speed on any
        step it samples a negative gripper action. The grasp is then not merely
        hard to find -- the action that should produce it destroys the state it
        needs. A rate limit of 0.01 m/step spreads the same closure over 8 steps
        (0.25 m/s), which is the regime a scripted grasp succeeds in.
        """
        self._gripper_cmd = 1.0
        """Normalized jaw command actually in force, slewed toward the action."""

        self.reward_cfg = reward_config or RewardConfig()
        self.rand_cfg = randomization or RandomizationConfig()
        self.reset_pose = reset_pose
        self.max_episode_steps = int(max_episode_steps)
        self.terminate_on_success = terminate_on_success
        self.curriculum_level = int(curriculum_level)
        self.render_mode = render_mode
        self.camera_name = camera_name

        # Physics runs at the MJCF timestep; the policy acts every control_dt.
        self.sim_dt = float(self.model.opt.timestep)
        self.n_substeps = max(1, int(round(control_dt / self.sim_dt)))
        self.control_dt = self.n_substeps * self.sim_dt
        if abs(self.control_dt - control_dt) > 1e-9:
            # Silently running at a different rate than asked would quietly
            # invalidate any gamma/horizon tuning done against control_dt.
            raise ValueError(
                f"control_dt={control_dt} is not a multiple of the model timestep "
                f"{self.sim_dt}; nearest is {self.control_dt}"
            )
        self.metadata = dict(self.metadata, render_fps=round(1.0 / self.control_dt))

        self.robot = PandaRobot(
            self.model, self.data, gravity_compensation=gravity_compensation
        )
        self.scene = self._resolve_scene_indices()

        # geom_size holds half-extents, so full extents are twice that. Resolved
        # once here rather than per call: is_grasped reads it on every step, and
        # the model geometry cannot change at runtime.
        self.object_extents = 2.0 * self.model.geom_size[self.scene.object_geom_id].copy()
        self.object_width = float(self.object_extents[0])
        """Jaw-relevant object width in metres. The scene object is a cube, so
        all three extents agree; a non-cubic object would need the width along
        the approach-relative jaw axis instead of the x extent."""

        self._np_random_seed_arg = seed
        self._elapsed_steps = 0
        self._goal_pos = np.zeros(3)
        self._object_start_pos = np.zeros(3)
        self._had_grasp = False
        self._episode_success = False
        self._renderer: mujoco.Renderer | None = None
        # Scratch state for the curriculum's collision-aware IK candidate choice.
        # Allocated on first use and reused, so a scene with no obstacle never
        # pays for it. See _pose_hits_obstacle.
        self._ik_scratch: mujoco.MjData | None = None

        # Curriculum bookkeeping. Failures are counted rather than raised on, so
        # one bad sample cannot kill a long run -- but exposed so a
        # silently-degraded curriculum cannot masquerade as a working one. See
        # _apply_curriculum_seed.
        self.curriculum_seeds_attempted = 0
        self.curriculum_seeds_failed = 0
        self.curriculum_stage_counts: dict[int, int] = {
            stage: 0 for stage in range(MAX_CURRICULUM_LEVEL + 1)
        }
        self._curriculum_progress = 0.0
        self.curriculum_prob = self._curriculum_prob_for(0.0)
        self._curriculum_warned = False
        # Stage the *current* episode started from, echoed in every step's info.
        self._curriculum_stage = 0

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.robot.n_arm_joints + 1,), dtype=np.float32
        )
        # Reset once so the observation has real values to size against.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        obs = self._observation()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    # -- model wiring ------------------------------------------------------ #

    def _resolve_scene_indices(self) -> _SceneIndices:
        def by_name(objtype: mujoco.mjtObj, name: str) -> int:
            oid = mujoco.mj_name2id(self.model, objtype, name)
            if oid < 0:
                raise ValueError(
                    f"{objtype.name} {name!r} missing from the scene; "
                    f"panda_scene.xml must define it"
                )
            return oid

        object_body = by_name(mujoco.mjtObj.mjOBJ_BODY, "object")
        object_joint = by_name(mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
        if self.model.jnt_type[object_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("object_joint must be a free joint for pose randomization")
        table_geom = by_name(mujoco.mjtObj.mjOBJ_GEOM, "table_top")

        # Table top from the geom itself, so moving the table in the MJCF does
        # not silently break the lift/spawn heights.
        table_z = float(
            self.model.geom_pos[table_geom][2] + self.model.geom_size[table_geom][2]
        )
        table_body = self.model.geom_bodyid[table_geom]
        table_z += float(self.model.body_pos[table_body][2])

        # Obstacles are resolved by name *prefix* and are optional -- unlike every
        # other index above, a missing one is not an error. Scanning names rather
        # than requiring a fixed count is what lets one scene file serve both the
        # baseline and the avoidance task.
        obstacle_geoms = np.array(
            [
                gid
                for gid in range(self.model.ngeom)
                if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")
                .startswith(OBSTACLE_GEOM_PREFIX)
            ],
            dtype=np.int32,
        )

        return _SceneIndices(
            object_body_id=object_body,
            object_geom_id=by_name(mujoco.mjtObj.mjOBJ_GEOM, "object_geom"),
            object_joint_qpos_adr=int(self.model.jnt_qposadr[object_joint]),
            object_joint_dof_adr=int(self.model.jnt_dofadr[object_joint]),
            goal_site_id=by_name(mujoco.mjtObj.mjOBJ_SITE, "goal_site"),
            table_geom_id=table_geom,
            table_top_z=table_z,
            obstacle_geom_ids=obstacle_geoms,
        )

    @property
    def object_half_height(self) -> float:
        return float(self.model.geom_size[self.scene.object_geom_id][2])

    # -- state accessors --------------------------------------------------- #

    @property
    def object_pos(self) -> np.ndarray:
        return self.data.xpos[self.scene.object_body_id].copy()

    @property
    def object_quat(self) -> np.ndarray:
        return self.data.xquat[self.scene.object_body_id].copy()

    @property
    def object_velp(self) -> np.ndarray:
        adr = self.scene.object_joint_dof_adr
        return self.data.qvel[adr : adr + 3].copy()

    @property
    def goal_pos(self) -> np.ndarray:
        return self._goal_pos.copy()

    def _set_object_pose(self, pos: np.ndarray, yaw: float) -> None:
        adr = self.scene.object_joint_qpos_adr
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        dof = self.scene.object_joint_dof_adr
        self.data.qvel[dof : dof + 6] = 0.0

    def _set_goal(self, pos: np.ndarray) -> None:
        self._goal_pos = np.asarray(pos, dtype=np.float64).copy()
        # The goal is a bodiless site, so it lives in the model rather than in
        # qpos. Writing model.site_pos keeps it out of the physics state
        # entirely, which is why the arm can never knock the goal around.
        self.model.site_pos[self.scene.goal_site_id] = self._goal_pos

    # -- grasp detection --------------------------------------------------- #

    def _pads_in_contact(self) -> np.ndarray:
        return self.robot.contacting_geoms(np.array([self.scene.object_geom_id]))

    def is_grasped(self) -> bool:
        """Both pads touching, the jaw width consistent with the object, and the
        object actually enclosed between the pads.

        All three conditions are load-bearing. Contact alone is not a grasp -- a
        finger brushing the cube while the jaw is wide open registers contacts
        too. But an *upper* bound on the jaw width is not enough either, and that
        omission is what a 3M-step pick_place run learned to exploit: the policy
        converged on driving the gripper fully shut and pressing the closed
        fingertips against the side of the cube. Both pad geoms report contact,
        and a width of ~0 trivially satisfies ``width < object_width + tol``, so
        the check passed on 93% of steps in episodes where the cube never rose
        more than 1.9 cm off the table. That spoof accounted for 96% of all grasp
        reward the final policy collected, and because the lift and place stages
        gate on this predicate it unlocked those too -- a 4.24/step fixed point
        whose every escape route was strictly downhill.

        The width *band* is the direct fix: a jaw held open by the object it is
        holding cannot be at zero. ``GRASP_WIDTH_MIN_RATIO`` is deliberately
        loose (half the object width) so that soft-contact penetration and pad
        compression on a squeezed grasp still register -- measured widths for
        genuine grips on the 4 cm cube run 0.026-0.038 m.

        The enclosure test rules out the remaining geometry: a cube touched by
        two pads while sitting outside or beside the jaw. It requires the object
        centre to project inside the segment joining the two pads, and to sit
        near that segment rather than merely near the fingers.
        """
        if not bool(self._pads_in_contact().all()):
            return False

        obj_w = self.object_width
        width = self.robot.gripper_width
        if not (GRASP_WIDTH_MIN_RATIO * obj_w < width < obj_w + GRASP_WIDTH_TOLERANCE):
            return False

        tips = self.robot.finger_tip_pos              # (2, 3), world frame
        jaw = tips[1] - tips[0]
        span = float(np.linalg.norm(jaw))
        if span < 1e-9:
            return False
        jaw = jaw / span
        rel = self.object_pos - tips[0]
        along = float(np.dot(rel, jaw))
        lateral = float(np.linalg.norm(rel - along * jaw))
        return 0.0 < along < span and lateral < 0.5 * obj_w + GRASP_LATERAL_TOLERANCE

    # -- obstacle collision ------------------------------------------------ #

    @property
    def has_obstacle(self) -> bool:
        """Whether this scene defines any barrier geom at all."""
        return self.scene.obstacle_geom_ids.size > 0

    def obstacle_contacts(self) -> tuple[int, int]:
        """``(robot_contacts, object_contacts)`` against the obstacle this step.

        Reads ``mjData.contact``, the solver's contact list for the state left by
        the last ``mj_step`` -- so this is "touching *now*", at the end of the
        control interval, not "touched at any point during it". A scrape that
        begins and ends inside one 40 ms interval of 20 substeps is therefore
        invisible here. Sampling per control step rather than per substep is the
        deliberate choice: the penalty has to be a function of the state the
        policy is credited for, and a per-substep count would charge a fast
        traverse more than a slow one for identical geometry.

        Vectorized over ``ncon`` rather than looped like
        ``PandaRobot.contacting_geoms``, because unlike the grasp predicate this
        runs against the *whole* arm -- 11 robot geoms rather than 2 -- on every
        step of every env.

        The two counts are returned separately and only the first is charged for.
        Robot-vs-obstacle is the arm scraping or clipping the barrier, which is
        what the penalty is for; object-vs-obstacle is the *carried cube* clipping
        it, which is reported as ``state/obstacle_contacts_object`` for diagnosis
        but left unpriced -- the cube cannot reach an airborne goal by sliding
        along a wall, so the place term already rules that path out, and pricing
        it twice would charge the policy for the cube's own settling contact.

        Returns ``(0, 0)`` immediately on a scene with no obstacle, which is the
        path every pre-existing task takes.
        """
        if not self.has_obstacle:
            return 0, 0
        ncon = int(self.data.ncon)
        if ncon == 0:
            return 0, 0

        geom1 = np.asarray(self.data.contact.geom1[:ncon], dtype=np.int32)
        geom2 = np.asarray(self.data.contact.geom2[:ncon], dtype=np.int32)
        obstacle = self.scene.obstacle_geom_ids

        # A contact counts when one side is an obstacle and the other is the
        # thing being tested for. Checked in both orders because MuJoCo orders
        # each pair by geom id, not by role.
        hits_obstacle = np.isin(geom1, obstacle), np.isin(geom2, obstacle)
        robot = self.robot.robot_geom_ids
        robot_contacts = int(
            np.count_nonzero(
                (hits_obstacle[0] & np.isin(geom2, robot))
                | (hits_obstacle[1] & np.isin(geom1, robot))
            )
        )
        obj = self.scene.object_geom_id
        object_contacts = int(
            np.count_nonzero(
                (hits_obstacle[0] & (geom2 == obj)) | (hits_obstacle[1] & (geom1 == obj))
            )
        )
        return robot_contacts, object_contacts

    def _collision_terms(self) -> tuple[float, dict[str, float]]:
        """``(penalty, terms)`` for the obstacle charge, shared by every task.

        Factored out rather than inlined because all three ``compute_reward``
        implementations override the base one wholesale, so an inlined penalty
        would apply to whichever task happened to be edited. A config that sets
        ``collision_penalty`` on a task that silently ignored it is precisely the
        class of failure ``make_reward_config``'s unknown-key check exists to
        prevent, and it would slip straight past that check.

        The charge is a flat ``collision_penalty`` while contact is present, not a
        multiple of the contact count -- see the field's docstring.
        """
        robot_hits, object_hits = self.obstacle_contacts()
        in_collision = robot_hits > 0
        penalty = self.reward_cfg.collision_penalty * float(in_collision)
        return penalty, {
            "reward/collision_penalty": -penalty,
            "state/in_collision": float(in_collision),
            "state/obstacle_contacts_robot": float(robot_hits),
            "state/obstacle_contacts_object": float(object_hits),
        }

    # -- reward ------------------------------------------------------------ #

    def set_reward_weights(self, **weights: float) -> dict[str, float]:
        """Overwrite ``reward_cfg`` fields mid-run. Returns what was applied.

        Exists so a training callback can anneal a weight the way
        ``set_curriculum_progress`` anneals the curriculum. ``reward_cfg`` is a
        plain mutable dataclass, so a callback *could* assign to it directly --
        but only on this side of the vector-env boundary. Routing through a named
        method is what lets ``VectorEnv.call()`` reach it, and therefore what
        makes an annealing schedule work under ``AsyncVectorEnv`` too, where the
        sub-envs live in other processes and an attribute written in the parent
        reaches nothing at all.

        Unknown field names raise rather than being dropped, for exactly the
        reason ``make_reward_config`` validates: a schedule that silently anneals
        nothing produces logs indistinguishable from one that is working.

        Note that ``scripts/train.py`` builds one ``RewardConfig`` instance and
        passes it to every training sub-env, so under ``SyncVectorEnv`` all of
        them alias a single object and ``call()`` simply writes the same value N
        times. The evaluator's env is constructed separately and holds its own
        config, so it does *not* follow a schedule applied to the training envs.
        """
        known = {f.name for f in fields(self.reward_cfg)}
        unknown = sorted(set(weights) - known)
        if unknown:
            raise KeyError(
                f"unknown reward field(s) {unknown} for "
                f"{type(self.reward_cfg).__name__}; have {sorted(known)}"
            )
        applied = {name: float(value) for name, value in weights.items()}
        for name, value in applied.items():
            setattr(self.reward_cfg, name, value)
        return applied

    def compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        """Dense staged shaping. Returns ``(reward, per-term breakdown)``."""
        cfg = self.reward_cfg
        eef = self.robot.eef_pos
        obj = self.object_pos
        goal = self._goal_pos

        d_reach = float(np.linalg.norm(eef - obj))
        d_place = float(np.linalg.norm(obj - goal))

        # Stage 1: get the tool centre point to the object.
        r_reach = cfg.reach * (1.0 - np.tanh(cfg.reach_sharpness * d_reach))

        # Stage 2: close on it. Gated on contact so the term cannot be farmed by
        # squeezing the jaw shut in free space.
        grasped = self.is_grasped()
        r_grasp = cfg.grasp * float(grasped)

        # Stage 3: height gained, clipped at lift_target so hurling the cube
        # upward is worth no more than lifting it cleanly.
        height = obj[2] - (self.scene.table_top_z + self.object_half_height)
        lift_progress = float(np.clip(height / cfg.lift_target, 0.0, 1.0))
        r_lift = cfg.lift * lift_progress * float(grasped)

        # Stage 4: carry to the goal. Also gated on grasp, so a policy cannot
        # collect place reward by batting the cube across the table.
        r_place = (
            cfg.place
            * (1.0 - np.tanh(cfg.place_sharpness * d_place))
            * float(grasped)
        )

        success = self._is_success(d_place, grasped)
        r_success = cfg.success * float(success)

        p_action = cfg.action_penalty * float(np.sum(np.square(action)))
        p_velocity = cfg.velocity_penalty * float(
            np.sum(np.square(self.robot.arm_qvel / JOINT_VELOCITY_LIMITS))
        )
        p_limits = cfg.joint_limit_penalty * float(
            np.sum(self.robot.joint_limit_violation())
        )
        p_collision, collision_terms = self._collision_terms()

        reward = (
            r_reach + r_grasp + r_lift + r_place + r_success
            - p_action - p_velocity - p_limits - p_collision
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
            "dist/eef_to_object": d_reach,
            "dist/object_to_goal": d_place,
            "state/lift_height": height,
            "state/gripper_width": self.robot.gripper_width,
            **collision_terms,
        }
        return float(reward), terms

    def _is_success(self, d_place: float, grasped: bool) -> bool:
        if d_place >= self.reward_cfg.success_threshold:
            return False
        if self.reward_cfg.require_grasp_for_success and not grasped:
            return False
        return True

    # -- task hooks -------------------------------------------------------- #
    #
    # Subclasses define a different task by overriding these three plus
    # compute_reward, rather than reimplementing step(). Keeping the hooks
    # explicit means a subclass whose reward reports different keys cannot
    # silently KeyError inside the base step().

    def _success_distance(self, terms: dict[str, float]) -> float:
        """The distance ``_is_success`` should judge, read out of ``terms``."""
        return terms["dist/object_to_goal"]

    def _task_failure(self, terms: dict[str, float]) -> bool:
        """Terminal failure. Here: the object has been knocked off the table."""
        del terms
        return bool(self.object_pos[2] < self.scene.table_top_z - 0.15)

    # -- observation ------------------------------------------------------- #

    def _observation(self) -> np.ndarray:
        eef = self.robot.eef_pos
        obj = self.object_pos
        goal = self._goal_pos
        parts = [
            self.robot.normalized_arm_qpos(),          # 7, scale-free
            self.robot.arm_qvel / JOINT_VELOCITY_LIMITS,  # 7, scale-free
            self.robot.finger_qpos / GRIPPER_MAX_WIDTH,   # 2
            self.robot.finger_qvel,                    # 2
            eef,                                       # 3
            self.robot.eef_quat,                       # 4
            obj,                                       # 3
            self.object_quat,                          # 4
            self.object_velp,                          # 3
            goal,                                      # 3
            obj - eef,                                 # 3, the reach vector
            goal - obj,                                # 3, the place vector
            [float(self.is_grasped())],                # 1
        ]
        return np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts]).astype(
            np.float32
        )

    # -- gym API ----------------------------------------------------------- #

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None and self._np_random_seed_arg is not None:
            seed = self._np_random_seed_arg
            self._np_random_seed_arg = None
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.robot.reset(
            self.reset_pose, gripper_open=True,
            noise=self.rand_cfg.arm_noise, rng=self.np_random,
        )

        object_pos = self._sample_object_pos(options)
        object_yaw = self._sample_object_yaw()
        self._set_object_pose(object_pos, yaw=object_yaw)
        self._set_goal(self._sample_goal(object_pos, options))

        # Refresh the kinematics cache before anything reads a world position.
        # mj_resetData zeroes data.xpos/data.site_xpos, and object_pos/eef_pos
        # read those rather than qpos -- so without this the curriculum's IK
        # target is the world origin, not the object. (Measured: 0/60 seeds.)
        mujoco.mj_forward(self.model, self.data)

        self._reject_start_collisions()

        stage = self._roll_curriculum_stage()
        if stage:
            stage = self._apply_curriculum_seed(stage, object_pos, object_yaw)
        self.curriculum_stage_counts[stage] += 1
        self._curriculum_stage = stage

        mujoco.mj_forward(self.model, self.data)

        self._elapsed_steps = 0
        self._had_grasp = False
        self._episode_success = False
        self._object_start_pos = self.object_pos
        # Seed the slew state from the jaw the reset actually left, not from "open":
        # a clamped curriculum seed starts the episode already gripping, and slewing
        # from +1 would spend the first steps of the episode opening the jaw and
        # dropping the object the seed handed over.
        self._gripper_cmd = float(
            np.clip(2.0 * self.robot.gripper_width / GRIPPER_MAX_WIDTH - 1.0, -1.0, 1.0)
        )
        obs = self._observation()
        _, terms = self.compute_reward(np.zeros(self.action_space.shape, dtype=np.float64))
        # A curriculum reset legitimately starts mid-task, so is_grasped can be
        # True on step 0. _had_grasp stays False: it means "grasped during this
        # episode by the policy", and the base step() will set it on step 1
        # anyway if the grasp holds.
        info = {"is_success": False, "success_now": False,
                "is_grasped": bool(stage and self.is_grasped()),
                "had_grasp": False, "failed": False,
                # ``curriculum/stage`` is the stage actually applied, 0 both when
                # the roll declined to seed and when a seed failed and rolled
                # back. Anything reading success rates should split on it: a
                # stage-3 episode starts at the goal and so starts successful,
                # which is not the same measurement as a stage-0 success.
                "curriculum/stage": float(stage),
                "curriculum/level": float(self.curriculum_level),
                "curriculum/prob": float(self.curriculum_prob),
                "curriculum/seeded": float(bool(stage)), **terms}
        return obs, info

    def _reject_start_collisions(self) -> None:
        """Redraw the reset arm pose until it is not touching an obstacle.

        Only the arm is redrawn. The object and goal boxes are placed clear of the
        barrier by construction (see ``OBSTACLE_PICK_PLACE_RANDOMIZATION``), so
        the only thing ``arm_noise`` can push into a wall is the arm.

        Why this exists rather than only a wall placed clear of the reset pose.
        ``ready_low`` holds the finger tips at x 0.443, z 0.571, hanging down toward
        the near top corner of the barrier, and ``RandomizationConfig.arm_noise``
        of 0.05 rad swings them. At an earlier wall top of z 0.52 that gap was
        3.3 cm and the noise closed it on 2.15% of draws -- measured over 2000
        resets, every one of them a finger contact. Episodes like that start
        already paying ``collision_penalty`` for a state the policy did not choose,
        teaching the value function that the initial state is worth -5.0/step, on
        one episode in fifty.

        The shipped wall top of z 0.50 opens that gap to 5.2 cm and measures 0/2000,
        so on the current geometry this method is a guarantee rather than a code
        path that runs. It is kept because the guarantee is the point: it holds for
        any wall placement and any ``arm_noise``, so neither can silently
        reintroduce the problem the way tuning the wall height alone would.

        Bounded, then falls back to the un-noised pose -- which
        ``PandaRobot._validate_reset_poses`` has already proven collision-free at
        construction, so the fallback cannot itself start in contact. A no-op on
        scenes with no obstacle, which is every task but this one.
        """
        if not self.has_obstacle:
            return
        for _ in range(START_COLLISION_ATTEMPTS):
            if self.obstacle_contacts()[0] == 0:
                return
            self.robot.reset(
                self.reset_pose, gripper_open=True,
                noise=self.rand_cfg.arm_noise, rng=self.np_random,
            )
            mujoco.mj_forward(self.model, self.data)
        if self.obstacle_contacts()[0] == 0:
            return
        self.robot.reset(self.reset_pose, gripper_open=True, noise=0.0)
        mujoco.mj_forward(self.model, self.data)

    # -- start-state curriculum -------------------------------------------- #

    @staticmethod
    def _curriculum_prob_for(progress: float) -> float:
        """Seeding probability at a given fraction of training elapsed."""
        return max(
            CURRICULUM_PROB_FLOOR,
            CURRICULUM_PROB_START - float(progress) * CURRICULUM_PROB_DECAY,
        )

    def set_curriculum_progress(self, progress: float) -> float:
        """Advance the schedule. ``progress`` runs 0.0 (start) to 1.0 (end).

        Called from the training loop rather than inferred inside the env: the
        env counts its own resets but has no idea how long the run is, and
        deriving the schedule from a step budget the env does not own is how
        these silently desynchronise. Returns the new probability.

        Clamped rather than validated, so a caller that overshoots the budget
        (resumed runs, an extended schedule) simply pins to the floor instead of
        raising in the middle of training.
        """
        self._curriculum_progress = float(np.clip(progress, 0.0, 1.0))
        self.curriculum_prob = self._curriculum_prob_for(self._curriculum_progress)
        return self.curriculum_prob

    @property
    def curriculum_progress(self) -> float:
        """Fraction of training elapsed, as last set by the trainer."""
        return self._curriculum_progress

    def _roll_curriculum_stage(self) -> int:
        """Draw the stage to seed this episode, or 0 to start from scratch.

        Uniform over the active stages rather than weighted toward the late
        ones. The stages are not equally hard, but they are equally *unvisited*
        at the start, and weighting is a knob worth adding only once there is a
        measurement saying which stage the policy is actually stuck on.
        """
        if self.curriculum_level < 1:
            return 0
        if self.np_random.random() >= self.curriculum_prob:
            return 0
        return int(self.np_random.integers(1, self.curriculum_level + 1))

    def _apply_curriculum_seed(
        self, stage: int, object_pos: np.ndarray, object_yaw: float
    ) -> int:
        """Seed the requested stage. Returns the stage applied, or 0 on rollback.

        Rationale. With an honest ``is_grasped`` there is no partial credit for
        *nearly* grasping: the predicate is a conjunction of contact, jaw width
        and enclosure, and a torque-controlled policy exploring an 8-D action
        space essentially never satisfies all three by chance. Measured on a 3M
        step run, the grasp rate was 0.0% for the first 2.6M steps, so every
        stage past the reach was unreachable and the value function had nothing
        to propagate. Seeding the later states directly removes the discovery
        problem: the policy practises from states it is handed, and the value
        function propagates backwards into the approach it must still make
        itself.

        Three things have to line up, and only the first is obvious:

        * **Position** -- the TCP goes to the object centre (or ``spec.hover``
          above it). The tool site sits at the jaw midpoint (verified:
          ``eef_pos`` equals the finger-tip midpoint), so aiming the TCP at the
          centre straddles the object.
        * **Orientation** -- the jaw axis has to line up with an object *face*.
          ``ready_low`` holds the jaw along world +y and tilted 18 degrees off
          vertical; closing that jaw on a cube yawed 45 degrees grips the
          diagonal, which measures ``0.04 * sqrt(2) = 0.057`` m and lands
          *outside* the width band ``is_grasped`` accepts. So the solve targets
          a true top-down frame yawed to match the object.
        * **Contact** -- MuJoCo needs the pads actually touching, not merely
          adjacent, before ``is_grasped`` will agree. Hence the settle loop, and
          hence verifying the predicate rather than assuming the geometry worked.
        """
        spec = _CURRICULUM_STAGE_SPECS[stage]
        self.curriculum_seeds_attempted += 1

        # Stage 3 practises the end of the task, so the object starts held near
        # the goal rather than at its spawn -- offset by CURRICULUM_GOAL_OFFSET so
        # a real final approach remains.
        #
        # Retried across offset directions because a 7.5 cm displacement from a
        # goal already near the edge of the workspace can land outside it: the
        # goal box is x 0.42-0.58 while the IK-verified reachable region ends at
        # x 0.62, so a single fixed draw lost 6.5% of stage-3 seeds to
        # unreachable targets (up from 2.5%) -- and lost them *systematically* at
        # the workspace edge, which is precisely where placement is hardest. The
        # IK is the authority on reachability, so it arbitrates: keep drawing
        # directions until one solves. Every draw has the exact specified
        # magnitude; only the direction changes.
        attempts = CURRICULUM_GOAL_OFFSET_ATTEMPTS if spec.at_goal else 1
        result = None
        for _ in range(attempts):
            if spec.at_goal:
                self._set_object_pose(
                    self._goal_pos + self._curriculum_goal_offset(), yaw=object_yaw
                )
                # Relocating the object invalidates the kinematics cache the IK
                # target is about to be read from.
                mujoco.mj_forward(self.model, self.data)
                # A 7.5 cm displacement from a goal near the barrier can put the
                # cube *inside* it. Redraw the direction rather than solving IK to
                # a target embedded in a wall -- this is the same reason the loop
                # redraws on an unreachable target, and it reuses the same budget.
                if self.obstacle_contacts()[1] > 0:
                    result = None
                    continue

            # Read back from the (possibly relocated) object rather than
            # recomputing from the goal, so the IK target tracks wherever the
            # object actually is and the gripper closes on it, not empty space.
            target_pos = self.object_pos + np.array([0.0, 0.0, spec.hover])
            result = self._solve_grasp_ik(target_pos, object_yaw)
            if result:
                break
        if not result:
            return self._curriculum_seed_failed(
                object_pos, object_yaw,
                f"stage {stage} ({spec.name}): IK did not converge in {attempts} "
                f"offset attempt(s) at any of the 4 equivalent grasp yaws "
                f"(best pos_error={result.pos_error:.4f} m, "
                f"rot_error={result.rot_error:.4f} rad)",
            )

        self.robot.reset(result.qpos, gripper_open=True, noise=0.0)
        if not spec.clamp:
            # Stage 1: jaw wide open, clear of the object. Nothing to settle --
            # there is deliberately no contact yet -- so verify the opposite of
            # the clamped stages: that this state does *not* already satisfy the
            # grasp predicate, since the close is the whole exercise.
            self.data.qpos[self.robot.idx.finger_qpos_adr] = 0.5 * GRIPPER_MAX_WIDTH
            self.data.qvel[:] = 0.0
            self.data.ctrl[self.robot.idx.finger_actuator_ids] = 0.5 * GRIPPER_MAX_WIDTH
            mujoco.mj_forward(self.model, self.data)
            if self._pads_in_contact().any() or self.is_grasped():
                return self._curriculum_seed_failed(
                    object_pos, object_yaw,
                    f"stage {stage} ({spec.name}): pre-grasp hover is already "
                    f"touching the object at {spec.hover:.3f} m clearance",
                )
            if self._seed_hits_obstacle():
                return self._curriculum_seed_failed(
                    object_pos, object_yaw,
                    f"stage {stage} ({spec.name}): seeded state is in contact with "
                    f"an obstacle",
                )
            return stage

        # Clamped stages: straddle the object, then squeeze onto it.
        self.data.qpos[self.robot.idx.finger_qpos_adr] = (
            0.5 * CURRICULUM_GRASP_WIDTH_RATIO * self.object_width
        )
        self.data.qvel[self.robot.idx.finger_dof_adr] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # The arm holds its solved pose while the jaw closes, so it neither sags
        # nor fights the contact being established. Which action means "hold" is
        # mode-dependent -- see hold_arm_action. This used to be a hardcoded zero
        # torque, which is correct only under torque control: under a position
        # servo a zero action commands the centre of every joint's range, so the
        # settle loop would have yanked the arm off the IK solution it had just
        # found and every clamped seed would fail its is_grasped check.
        hold = self.hold_arm_action()
        for _ in range(CURRICULUM_SETTLE_STEPS):
            self._apply_arm_action(hold)
            self.robot.apply_gripper(-1.0)
            mujoco.mj_step(self.model, self.data)

        # Hand the policy a still scene: settling leaves small residual
        # velocities that would otherwise read as motion it did not command.
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        if not self.is_grasped():
            return self._curriculum_seed_failed(
                object_pos, object_yaw,
                f"stage {stage} ({spec.name}): settled state failed is_grasped "
                f"(width={self.robot.gripper_width:.4f} m, "
                f"object_width={self.object_width:.4f} m)",
            )
        if self._seed_hits_obstacle():
            return self._curriculum_seed_failed(
                object_pos, object_yaw,
                f"stage {stage} ({spec.name}): settled state is in contact with an "
                f"obstacle",
            )
        return stage

    def _seed_hits_obstacle(self) -> bool:
        """Whether the state just seeded is already touching a barrier.

        Checked because ``_solve_grasp_ik`` is a *kinematic* solve with joint
        limits as its only constraint: nothing in it knows an obstacle exists, so
        on the avoidance scene it can and does return poses that put a link
        through the wall. Seeding one is worse than failing to seed. The episode
        would begin inside a penetration the solver then resolves as an impulse,
        and it would begin already paying ``collision_penalty`` -- teaching the
        value function that the seeded stages are where reward goes to die, which
        is the exact opposite of what a reverse curriculum is for.

        The object side is checked as well, for stage 3 specifically: it relocates
        the cube to the goal plus ``CURRICULUM_GOAL_OFFSET``, and a 7.5 cm
        displacement from a goal near the barrier can land the cube inside it.

        A no-op returning False on any scene without an obstacle, so the baseline
        curriculum's seed accept/reject behaviour is unchanged.
        """
        return any(count > 0 for count in self.obstacle_contacts())

    def _solve_grasp_ik(self, target_pos: np.ndarray, object_yaw: float) -> Any:
        """Top-down grasp IK at ``target_pos``, trying all four face yaws.

        A square face repeats every 90 degrees, so four wrist orientations
        describe the same physical grip. Trying them in turn recovers the solves
        that fail only because joint7 runs out of travel at one of them --
        measured: 5/60 seeds lost to wrist limits at a single candidate, 0/60
        across all four. Returns the last result when none converge, so the
        caller can report the closest miss.

        On an obstacle scene a converged solve is not automatically a *usable*
        one, so the candidates are additionally filtered on collision. The four
        yaws are equivalent as grips but emphatically not as swept volumes: the
        hand geom is 10 cm long along the jaw axis against 6 cm across it, so a
        quarter-turned wrist lays that long axis across the barrier instead of
        parallel to it. Measured against a deliberately tight variant of this scene
        -- the wall 3 cm further from the robot, at x 0.495-0.525, with objects
        spawning from x 0.56 -- taking the first *kinematically* converged candidate
        put a robot geom in contact with the wall on 54 of 235 seeds, 109 of those
        contacts on ``hand_geom`` alone; filtering the same four yaws on collision
        cut it to 44. The wall then moved to buy the remaining clearance (see the
        asset), and on the shipped geometry the two effects together leave the
        seeder at 4 of 235, at or below its obstacle-free baseline.

        The fallback if every candidate collides is the first converged one, so this
        can only improve on the previous behaviour; ``_apply_curriculum_seed`` still
        rejects it downstream.
        """
        last: Any = None
        first_converged: Any = None
        for quat in self._grasp_quat_candidates(object_yaw):
            last = solve_site_ik(
                self.model,
                self.robot.idx.eef_site_id,
                self.robot.idx.arm_qpos_adr,
                self.robot.idx.arm_dof_adr,
                target_pos,
                target_quat=quat,
                q_init=self.robot.arm_qpos,
                joint_limits=JOINT_POSITION_LIMITS,
                qpos_full=self.data.qpos,
                rot_tolerance=CURRICULUM_IK_ROT_TOLERANCE,
            )
            if not last:
                continue
            if not self._pose_hits_obstacle(last.qpos):
                return last
            if first_converged is None:
                first_converged = last
        # Either every converged candidate collides -- hand the best one back and
        # let _apply_curriculum_seed reject it -- or none converged, in which case
        # `last` is the closest miss the caller reports.
        return first_converged if first_converged is not None else last

    def _pose_hits_obstacle(self, arm_qpos: np.ndarray) -> bool:
        """Whether the arm at ``arm_qpos`` touches an obstacle, jaw wide open.

        Evaluated on a scratch ``MjData`` so it cannot disturb the live state --
        the caller is mid-reset and ``self.data`` still holds the pose the IK was
        seeded from. Same technique as ``PandaRobot.pose_collisions``, but scoped
        to obstacle pairs only: the table and the object are *expected* to be in
        contact around a grasp, and rejecting those would reject every seed.

        The scratch buffer is allocated once and reused. Allocating an ``MjData``
        per candidate would mean up to four allocations per reset on every one of
        16 training envs, which is a real cost for a check that is pure kinematics.

        Open jaw rather than the grasp width, because this runs *before* the
        clamped stages squeeze: a wider jaw is the conservative test, since it can
        only add contact, never hide it.
        """
        if not self.has_obstacle:
            return False
        if self._ik_scratch is None:
            self._ik_scratch = mujoco.MjData(self.model)
        scratch = self._ik_scratch
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self.robot.idx.arm_qpos_adr] = arm_qpos
        scratch.qpos[self.robot.idx.finger_qpos_adr] = 0.5 * GRIPPER_MAX_WIDTH
        scratch.qvel[:] = 0.0
        mujoco.mj_forward(self.model, scratch)

        obstacle = self.scene.obstacle_geom_ids
        robot = self.robot.robot_geom_ids
        ncon = int(scratch.ncon)
        if ncon == 0:
            return False
        geom1 = np.asarray(scratch.contact.geom1[:ncon], dtype=np.int32)
        geom2 = np.asarray(scratch.contact.geom2[:ncon], dtype=np.int32)
        return bool(
            np.any(
                (np.isin(geom1, obstacle) & np.isin(geom2, robot))
                | (np.isin(geom2, obstacle) & np.isin(geom1, robot))
            )
        )

    def _curriculum_goal_offset(self) -> np.ndarray:
        """Isotropic displacement of exactly ``CURRICULUM_GOAL_OFFSET`` metres.

        Direction from a normal draw normalised to unit length, which is uniform
        over the sphere -- unlike ``uniform(-r, r, 3)``, which concentrates in the
        corners of a cube and would bias stage 3 toward diagonal approaches.

        No z guard is needed, and adding one would break the exact magnitude the
        offset is specified to have: goal ``z`` is sampled from
        ``RandomizationConfig.goal_z`` (0.50-0.62 for pick-and-place) and the
        projection of a uniform sphere direction is uniform on
        ``[-0.075, +0.075]``, so the object centre lands no lower than 0.425 m
        against a resting height of 0.42 m. Verified over 200 resets: no
        additional rollbacks versus placing the object on the goal exactly.
        """
        direction = self.np_random.normal(size=3)
        norm = float(np.linalg.norm(direction))
        while norm < 1e-9:
            # Measure-zero, but a zero vector would put the object back on the
            # goal and re-introduce exactly the poisoning this offset removes.
            direction = self.np_random.normal(size=3)
            norm = float(np.linalg.norm(direction))
        return direction * (CURRICULUM_GOAL_OFFSET / norm)

    def _grasp_quat_candidates(self, object_yaw: float) -> list[np.ndarray]:
        """The four wrist orientations that grip a square object face-on.

        Ordered nearest-first: the wrapped yaw needs the least wrist travel from
        ``ready_low``, and the quarter turns are only reached for when that one
        is kinematically out of range.
        """
        return grasp_quat_candidates(object_yaw)

    @staticmethod
    def _top_down_grasp_quat(yaw: float) -> np.ndarray:
        """Deprecated alias for the module-level ``top_down_grasp_quat``.

        Kept so the curriculum's existing call sites read unchanged. The
        implementation moved to module scope so a task-space controller can reach
        it without importing the env class, which would be an import cycle:
        ``envs.wrappers.task_space`` is imported by code that builds envs.
        """
        return top_down_grasp_quat(yaw)

    def _curriculum_seed_failed(
        self, object_pos: np.ndarray, object_yaw: float, reason: str
    ) -> int:
        """Roll back to a stage-0 start state. Always returns 0, the stage applied.

        Falling back rather than raising: a single unlucky sample must not kill a
        multi-hour run. Counting rather than swallowing: a curriculum that
        quietly stops seeding looks exactly like one that works, so the counters
        are public and the first failure is logged.

        Both the arm *and* the object are restored, to the pose ``reset``
        sampled. Two ways the object ends up moved by the time we get here: the
        settle loop steps real physics and nudges it, and stage 3 deliberately
        relocated it onto the goal. Restoring only the arm would leave the
        episode starting from a state no stage describes -- and in the stage-3
        case, from one already at the goal and therefore already successful.
        """
        self.curriculum_seeds_failed += 1
        if not self._curriculum_warned:
            self._curriculum_warned = True
            _LOGGER.warning(
                "curriculum seed failed, rolling back to stage 0: %s. Further "
                "failures are counted in curriculum_seeds_failed (now %d/%d) "
                "and not logged.",
                reason,
                self.curriculum_seeds_failed, self.curriculum_seeds_attempted,
            )
        self.robot.reset(
            self.reset_pose, gripper_open=True,
            noise=self.rand_cfg.arm_noise, rng=self.np_random,
        )
        self._set_object_pose(object_pos, yaw=object_yaw)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return 0

    def _sample_object_pos(self, options: dict[str, Any] | None) -> np.ndarray:
        if options and "object_pos" in options:
            return np.asarray(options["object_pos"], dtype=np.float64)
        z = self.scene.table_top_z + self.object_half_height + 1e-3
        if not self.rand_cfg.randomize_object:
            return np.array([0.5, 0.0, z])
        return np.array(
            [
                self.np_random.uniform(*self.rand_cfg.object_x),
                self.np_random.uniform(*self.rand_cfg.object_y),
                z,
            ]
        )

    def _sample_object_yaw(self) -> float:
        if not self.rand_cfg.randomize_object:
            return 0.0
        return float(self.np_random.uniform(*self.rand_cfg.object_yaw))

    def _sample_goal(
        self, object_pos: np.ndarray, options: dict[str, Any] | None
    ) -> np.ndarray:
        if options and "goal_pos" in options:
            return np.asarray(options["goal_pos"], dtype=np.float64)
        if not self.rand_cfg.randomize_goal:
            return np.array([0.5, 0.0, self.rand_cfg.goal_z[1]])
        # Reject goals that start already satisfied, which would hand out the
        # success bonus for doing nothing.
        for _ in range(100):
            goal = np.array(
                [
                    self.np_random.uniform(*self.rand_cfg.goal_x),
                    self.np_random.uniform(*self.rand_cfg.goal_y),
                    self.np_random.uniform(*self.rand_cfg.goal_z),
                ]
            )
            if np.linalg.norm(goal - object_pos) >= self.rand_cfg.min_object_goal_distance:
                return goal
        return goal

    def _slew_gripper(self, gripper_action: float) -> float:
        """Rate-limit the gripper command. Returns the value to actually apply.

        A no-op when ``gripper_rate_limit`` is None, so the default behaviour and
        every existing config are unchanged.
        """
        target = float(gripper_action)
        if self.gripper_rate_limit is None:
            self._gripper_cmd = target
            return target
        # The action spans [-1, 1] over the full jaw width, so a width rate maps to
        # a normalized rate through the same factor.
        rate = 2.0 * self.gripper_rate_limit / GRIPPER_MAX_WIDTH
        step = float(np.clip(target - self._gripper_cmd, -rate, rate))
        self._gripper_cmd = float(np.clip(self._gripper_cmd + step, -1.0, 1.0))
        return self._gripper_cmd

    def _apply_arm_action(self, arm_action: np.ndarray) -> np.ndarray:
        """Dispatch the arm half of an action to the configured control law."""
        if self.control_mode == "torque":
            return self.robot.apply_torque(arm_action)
        if self.control_mode == "velocity":
            return self.robot.apply_velocity(arm_action)
        return self.robot.apply_joint_position(arm_action)

    def hold_arm_action(self) -> np.ndarray:
        """The arm action that means "stay where you are", in the current mode.

        Mode-dependent, which is the point of naming it: zero torque coasts under
        gravity compensation, zero velocity brakes, and holding a *position*
        requires commanding the pose the arm is already in. Used by the curriculum
        settle loop and by the demonstration recorder, both of which need the arm
        to stay put while something else happens.
        """
        if self.control_mode == "joint_position":
            return self.robot.normalized_arm_qpos()
        return np.zeros(self.robot.n_arm_joints, dtype=np.float64)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(
            np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0
        )
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"action shape {action.shape} != {self.action_space.shape}"
            )

        arm_action, gripper_action = action[:-1], action[-1]
        self.robot.apply_gripper(self._slew_gripper(gripper_action))

        for _ in range(self.n_substeps):
            # All three laws depend on the current state, so they are recomputed
            # every substep rather than held for the whole control interval. For
            # the two closed-loop modes this is what makes them 500 Hz servos
            # tracking a 25 Hz target, rather than 25 Hz servos: the target is
            # sampled-and-held, the feedback is not.
            self._apply_arm_action(arm_action)
            mujoco.mj_step(self.model, self.data)

        # mj_step leaves derived quantities current, but site_xpos used by the
        # reward comes from the position stage; refresh to be explicit.
        mujoco.mj_forward(self.model, self.data)

        self._elapsed_steps += 1
        reward, terms = self.compute_reward(action)
        grasped = self.is_grasped()
        self._had_grasp = self._had_grasp or grasped
        success_now = self._is_success(self._success_distance(terms), grasped)
        # Latch: the episode succeeded if the goal was ever met. The trainer reads
        # is_success only on the step an episode ends, so an unlatched flag would
        # make the metric "was at the goal on the final step" -- which, with
        # termination off, silently scores a policy that reaches and then drifts
        # as a failure, and disagrees with the evaluator's definition.
        self._episode_success = self._episode_success or success_now

        failed = self._task_failure(terms)
        terminated = bool(success_now and self.terminate_on_success) or bool(failed)
        truncated = self._elapsed_steps >= self.max_episode_steps and not terminated

        info = {
            "is_success": bool(self._episode_success),
            "success_now": bool(success_now),
            "is_grasped": bool(grasped),
            "had_grasp": bool(self._had_grasp),
            "failed": bool(failed),
            # Repeated on every step, not just at reset, so that whatever reads
            # ``is_success`` on the terminal step can attribute the episode to the
            # state it started from. Success is only meaningful *relative* to that:
            # a stage-3 episode began 7.5 cm from the goal already holding the
            # object, a stage-0 one began across the table with an open jaw.
            # Carrying it per-step also means no cross-step bookkeeping in the
            # trainer, which would otherwise have to reconstruct it from reset
            # infos under whichever autoreset convention the vector env uses.
            "curriculum/stage": float(self._curriculum_stage),
            **terms,
        }
        return self._observation(), reward, terminated, truncated, info

    # -- rendering --------------------------------------------------------- #

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=self.camera_name)
        if self.render_mode == "depth_array":
            self._renderer.enable_depth_rendering()
            depth = self._renderer.render()
            self._renderer.disable_depth_rendering()
            return depth
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
