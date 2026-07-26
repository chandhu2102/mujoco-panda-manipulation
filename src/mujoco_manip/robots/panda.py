"""Franka Emika Panda arm definition.

Joint maps, gripper indices and safe reset states for the MJCF in
``assets/robots/panda/panda.xml``. Limits below are the official Franka values,
so they stay correct if the primitive-geom MJCF is later swapped for the
mujoco_menagerie mesh model.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

__all__ = [
    "PandaRobot",
    "ARM_JOINT_NAMES",
    "FINGER_JOINT_NAMES",
    "JOINT_POSITION_LIMITS",
    "JOINT_VELOCITY_LIMITS",
    "JOINT_TORQUE_LIMITS",
    "RESET_POSES",
    "GRIPPER_JOINT_RANGE",
    "GRIPPER_MAX_WIDTH",
    "VELOCITY_KP",
    "POSITION_NATURAL_FREQ",
    "POSITION_DAMPING_RATIO",
]

# --------------------------------------------------------------------------- #
# Static model description (official Franka Panda values)
# --------------------------------------------------------------------------- #

ARM_JOINT_NAMES: tuple[str, ...] = (
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
)
FINGER_JOINT_NAMES: tuple[str, ...] = ("finger_joint1", "finger_joint2")

ARM_ACTUATOR_NAMES: tuple[str, ...] = (
    "actuator1", "actuator2", "actuator3", "actuator4",
    "actuator5", "actuator6", "actuator7",
)
FINGER_ACTUATOR_NAMES: tuple[str, ...] = ("actuator_finger1", "actuator_finger2")

EEF_SITE_NAME = "eef_site"
FINGER_TIP_SITE_NAMES: tuple[str, ...] = ("left_finger_tip", "right_finger_tip")
FINGER_GEOM_NAMES: tuple[str, ...] = ("left_finger_geom", "right_finger_geom")

# rad. Rows are (lower, upper) per joint.
JOINT_POSITION_LIMITS: np.ndarray = np.array(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],   # joint4 never reaches zero; elbow stays bent
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=np.float64,
)

# rad/s
JOINT_VELOCITY_LIMITS: np.ndarray = np.array(
    [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100], dtype=np.float64
)

# Nm
JOINT_TORQUE_LIMITS: np.ndarray = np.array(
    [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64
)

# Proportional gains for the joint-space velocity law, scaled off the torque
# limits so every joint saturates at a comparable velocity error.
VELOCITY_KP: np.ndarray = JOINT_TORQUE_LIMITS * 0.6

# --------------------------------------------------------------------------- #
# Joint-position servo tuning
#
# Gains are not constants here: they are derived per-joint from the model's own
# mass matrix at a reference pose (see PandaRobot._position_gains), because a
# hand-picked kp that is well damped on joint1 (inertia ~2 kg m^2) is wildly
# overdriven on joint7 (~0.1 kg m^2). Given a desired natural frequency the
# second-order form fixes both gains:
#
#     kp_i = M_ii * omega_n^2        kd_i = 2 * zeta * M_ii * omega_n
#
# omega_n is bounded from above by the torque limits, not by the control rate.
# The servo law is recomputed every *substep* (500 Hz at the 0.002 s MJCF
# timestep), so sample-and-hold is not the binding constraint -- saturation is.
# A typical inter-target joint step at 25 Hz is ~0.05 rad, and keeping
# kp * 0.05 inside the 87 Nm limit on the big joints caps omega_n near 30 rad/s.
# 25 rad/s leaves margin for the larger steps that follow a curriculum seed.
# --------------------------------------------------------------------------- #

POSITION_NATURAL_FREQ: float = 25.0
"""Servo natural frequency in rad/s (~4 Hz). See the note above for the bound."""
POSITION_DAMPING_RATIO: float = 1.0
"""Critically damped at the reference pose. Inertia varies by roughly 2-3x over
the workspace, so the realised ratio drifts either side of 1.0; overshoot is
preferable to the alternative of tuning for the worst case and being sluggish
everywhere else."""
POSITION_GAIN_REFERENCE_POSE: str = "ready_low"
"""Pose whose mass matrix sets the gains. The pre-grasp pose, i.e. the
configuration the precision phase actually happens in."""

GRIPPER_JOINT_RANGE: tuple[float, float] = (0.0, 0.04)
"""Per-finger travel. Total jaw opening is twice this."""

GRIPPER_MAX_WIDTH: float = 2.0 * GRIPPER_JOINT_RANGE[1]

# --------------------------------------------------------------------------- #
# Reset states
#
# All of these are interior to the joint limits by a margin (verified by
# PandaRobot._validate_reset_poses at construction) and collision-free in the
# scene, so a reset can never start the episode already in violation.
# --------------------------------------------------------------------------- #

RESET_POSES: dict[str, np.ndarray] = {
    # Franka's canonical "ready" pose. Elbow up, TCP at ~(0.307, 0, 0.487).
    "home": np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
    # Pre-grasp hover: TCP at ~(0.450, 0, 0.550), i.e. 15 cm above a table top at
    # z=0.4 and centred over the object spawn region, which shortens the reach
    # phase. Solved by damped-least-squares IK, not hand-guessed -- an
    # eyeballed "lower" pose put the hand *inside* the table.
    "ready_low": np.array([0.0, -0.5212, 0.0, -2.1149, 0.0, 1.9151, 0.785]),
    # Folded back over the base and clear of the table: TCP at ~(0.062, 0, 0.467).
    "retracted": np.array([0.0, -1.500, 0.0, -2.600, 0.0, 1.000, 0.785]),
}

DEFAULT_RESET_POSE = "home"

# Fraction of each joint's range kept clear of the hard stops on reset.
JOINT_LIMIT_MARGIN = 0.02


@dataclass
class PandaIndices:
    """Resolved MuJoCo indices. Grouped so callers can pass one object around."""

    arm_joint_ids: np.ndarray
    arm_qpos_adr: np.ndarray
    arm_dof_adr: np.ndarray
    finger_joint_ids: np.ndarray
    finger_qpos_adr: np.ndarray
    finger_dof_adr: np.ndarray
    arm_actuator_ids: np.ndarray
    finger_actuator_ids: np.ndarray
    eef_site_id: int
    finger_tip_site_ids: np.ndarray
    finger_geom_ids: np.ndarray


class PandaRobot:
    """Index maps, state accessors and control mapping for the Panda.

    Holds no simulation state of its own: every method takes or reads the
    ``MjData`` it was constructed with, so the same instance stays valid across
    ``mj_resetData`` calls.
    """

    n_arm_joints = 7
    n_finger_joints = 2

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        gravity_compensation: bool = True,
    ) -> None:
        self.model = model
        self.data = data
        self.gravity_compensation = gravity_compensation
        self.idx = self._resolve_indices(model)
        self._robot_geom_ids: np.ndarray | None = None
        self._validate_reset_poses()
        self.position_kp, self.position_kd = self._position_gains()

    # -- index resolution -------------------------------------------------- #

    @staticmethod
    def _resolve_indices(model: mujoco.MjModel) -> PandaIndices:
        def joint_id(name: str) -> int:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(
                    f"joint {name!r} missing from the model; the MJCF does not "
                    f"look like a Panda"
                )
            return jid

        def actuator_id(name: str) -> int:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"actuator {name!r} missing from the model")
            return aid

        def site_id(name: str) -> int:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(f"site {name!r} missing from the model")
            return sid

        def geom_id(name: str) -> int:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise ValueError(f"geom {name!r} missing from the model")
            return gid

        arm_ids = np.array([joint_id(n) for n in ARM_JOINT_NAMES], dtype=np.int32)
        finger_ids = np.array([joint_id(n) for n in FINGER_JOINT_NAMES], dtype=np.int32)

        return PandaIndices(
            arm_joint_ids=arm_ids,
            arm_qpos_adr=model.jnt_qposadr[arm_ids].copy(),
            arm_dof_adr=model.jnt_dofadr[arm_ids].copy(),
            finger_joint_ids=finger_ids,
            finger_qpos_adr=model.jnt_qposadr[finger_ids].copy(),
            finger_dof_adr=model.jnt_dofadr[finger_ids].copy(),
            arm_actuator_ids=np.array(
                [actuator_id(n) for n in ARM_ACTUATOR_NAMES], dtype=np.int32
            ),
            finger_actuator_ids=np.array(
                [actuator_id(n) for n in FINGER_ACTUATOR_NAMES], dtype=np.int32
            ),
            eef_site_id=site_id(EEF_SITE_NAME),
            finger_tip_site_ids=np.array(
                [site_id(n) for n in FINGER_TIP_SITE_NAMES], dtype=np.int32
            ),
            finger_geom_ids=np.array(
                [geom_id(n) for n in FINGER_GEOM_NAMES], dtype=np.int32
            ),
        )

    def _validate_reset_poses(self) -> None:
        """Fail at construction rather than mid-episode on an unsafe pose.

        Checks joint limits *and* collisions. The limit check alone is not
        enough: a pose can sit comfortably inside every joint range and still
        bury the hand in the table, which is a silent failure that only shows up
        as a policy that cannot learn.
        """
        lower, upper = JOINT_POSITION_LIMITS[:, 0], JOINT_POSITION_LIMITS[:, 1]
        span = upper - lower
        margin = JOINT_LIMIT_MARGIN * span

        for name, qpos in RESET_POSES.items():
            if qpos.shape != (self.n_arm_joints,):
                raise ValueError(
                    f"reset pose {name!r} has shape {qpos.shape}, "
                    f"expected ({self.n_arm_joints},)"
                )
            if np.any(qpos < lower + margin) or np.any(qpos > upper - margin):
                bad = np.flatnonzero((qpos < lower + margin) | (qpos > upper - margin))
                raise ValueError(
                    f"reset pose {name!r} is within {JOINT_LIMIT_MARGIN:.0%} of a "
                    f"joint limit at joint(s) {(bad + 1).tolist()}"
                )
            contacts = self.pose_collisions(qpos)
            if contacts:
                raise ValueError(
                    f"reset pose {name!r} starts in collision: "
                    f"{', '.join(sorted(set(contacts)))}"
                )

    def _position_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-joint ``(kp, kd)`` for the joint-position servo.

        Derived from the diagonal of the mass matrix at
        ``POSITION_GAIN_REFERENCE_POSE`` rather than hand-tuned, so the servo has
        the same closed-loop frequency on every joint. Without that the choice is
        between gains that droop on joint1 and gains that ring on joint7 -- and
        ringing at the wrist is precisely the joint chatter a position servo is
        being introduced to remove.

        The diagonal is used rather than the full matrix because the law is
        decoupled per joint; off-diagonal inertial coupling is left to be
        rejected as a disturbance, which the damping term handles.

        Read out with ``mj_mulM`` on unit basis vectors -- ``M @ e_i`` is column
        ``i``, whose ``i``-th entry is ``M_ii`` -- rather than with ``mj_fullM``,
        whose Python signature has moved between MuJoCo releases. Seven
        matrix-vector products, once, at construction.
        """
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.model.qpos0
        scratch.qpos[self.idx.arm_qpos_adr] = RESET_POSES[POSITION_GAIN_REFERENCE_POSE]
        scratch.qpos[self.idx.finger_qpos_adr] = GRIPPER_JOINT_RANGE[1]
        mujoco.mj_forward(self.model, scratch)

        column = np.zeros(self.model.nv, dtype=np.float64)
        basis = np.zeros(self.model.nv, dtype=np.float64)
        inertia = np.zeros(self.n_arm_joints, dtype=np.float64)
        for i, dof in enumerate(self.idx.arm_dof_adr):
            basis[:] = 0.0
            basis[dof] = 1.0
            mujoco.mj_mulM(self.model, scratch, column, basis)
            inertia[i] = column[dof]
        # A non-positive diagonal would mean a broken model, but clamp rather than
        # raise: a floor keeps the gains finite instead of producing zeros that
        # would silently disable the servo on one joint.
        inertia = np.maximum(inertia, 1e-3)

        omega = POSITION_NATURAL_FREQ
        kp = inertia * (omega ** 2)
        kd = 2.0 * POSITION_DAMPING_RATIO * inertia * omega
        return kp, kd

    def pose_collisions(self, qpos: np.ndarray) -> list[str]:
        """Geom-pair names in contact at ``qpos``, ignoring non-robot pairs.

        Runs on a scratch ``MjData`` so it never disturbs live simulation state.
        """
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.model.qpos0
        scratch.qpos[self.idx.arm_qpos_adr] = qpos
        scratch.qpos[self.idx.finger_qpos_adr] = GRIPPER_JOINT_RANGE[1]
        mujoco.mj_forward(self.model, scratch)

        robot_geoms = set(int(g) for g in self.robot_geom_ids)
        out: list[str] = []
        for c in range(scratch.ncon):
            g1 = int(scratch.contact[c].geom1)
            g2 = int(scratch.contact[c].geom2)
            # Only robot-vs-anything matters; table-vs-object resting contact is
            # expected and not a pose problem.
            if g1 not in robot_geoms and g2 not in robot_geoms:
                continue
            n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)
            out.append(f"{n1}<->{n2}")
        return out

    @property
    def robot_geom_ids(self) -> np.ndarray:
        """Every geom on the arm chain -- all seven links, hand and both fingers.

        Resolved from the body tree rather than from the ``contype == 1`` bitmask
        the MJCF uses for the arm, so a scene that adds another geom in that
        collision class does not silently start counting as part of the robot.

        Cached on first use: the set is fixed by the model, and the callers that
        want it -- ``pose_collisions`` at construction and the obstacle-collision
        penalty in ``envs.manipulation_env`` on *every* step -- would otherwise
        redo the tree walk each time.

        Broader than ``idx.finger_geom_ids`` on purpose. The fingers are what
        grasp, so they are the only geoms a grasp predicate cares about; a
        collision predicate has the opposite requirement, because an elbow driven
        through an obstacle is exactly the failure it exists to catch.
        """
        if self._robot_geom_ids is None:
            self._robot_geom_ids = np.flatnonzero(
                np.isin(self.model.geom_bodyid, self._robot_body_ids())
            ).astype(np.int32)
        return self._robot_geom_ids

    def _robot_body_ids(self) -> np.ndarray:
        """Body ids of the arm chain, found by walking up from the fingers."""
        ids: set[int] = set()
        for jid in list(self.idx.arm_joint_ids) + list(self.idx.finger_joint_ids):
            body = int(self.model.jnt_bodyid[jid])
            ids.add(body)
            # Include intermediate bodies (e.g. the hand carries no joint).
            parent = int(self.model.body_parentid[body])
            while parent > 0:
                ids.add(parent)
                parent = int(self.model.body_parentid[parent])
        # The hand and any other jointless children of arm links.
        for b in range(self.model.nbody):
            if int(self.model.body_parentid[b]) in ids:
                ids.add(b)
        ids.discard(0)
        return np.array(sorted(ids), dtype=np.int32)

    # -- state ------------------------------------------------------------- #

    @property
    def arm_qpos(self) -> np.ndarray:
        return self.data.qpos[self.idx.arm_qpos_adr]

    @property
    def arm_qvel(self) -> np.ndarray:
        return self.data.qvel[self.idx.arm_dof_adr]

    @property
    def finger_qpos(self) -> np.ndarray:
        return self.data.qpos[self.idx.finger_qpos_adr]

    @property
    def finger_qvel(self) -> np.ndarray:
        return self.data.qvel[self.idx.finger_dof_adr]

    @property
    def gripper_width(self) -> float:
        """Jaw opening in metres, i.e. the sum of both finger travels."""
        return float(self.finger_qpos.sum())

    @property
    def eef_pos(self) -> np.ndarray:
        """Tool centre point in world coordinates."""
        return self.data.site_xpos[self.idx.eef_site_id].copy()

    @property
    def eef_mat(self) -> np.ndarray:
        return self.data.site_xmat[self.idx.eef_site_id].reshape(3, 3).copy()

    @property
    def eef_quat(self) -> np.ndarray:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.idx.eef_site_id])
        return quat

    @property
    def finger_tip_pos(self) -> np.ndarray:
        """``(2, 3)`` world positions of the two finger pads."""
        return self.data.site_xpos[self.idx.finger_tip_site_ids].copy()

    def joint_limit_violation(self) -> np.ndarray:
        """Per-joint distance past a limit, zero where inside. Useful for penalties."""
        lower, upper = JOINT_POSITION_LIMITS[:, 0], JOINT_POSITION_LIMITS[:, 1]
        q = self.arm_qpos
        return np.maximum(0.0, np.maximum(lower - q, q - upper))

    def normalized_arm_qpos(self) -> np.ndarray:
        """Arm position mapped to ``[-1, 1]`` across each joint's range."""
        lower, upper = JOINT_POSITION_LIMITS[:, 0], JOINT_POSITION_LIMITS[:, 1]
        return 2.0 * (self.arm_qpos - lower) / (upper - lower) - 1.0

    @staticmethod
    def normalize_arm_qpos(qpos: np.ndarray) -> np.ndarray:
        """Map an arbitrary arm configuration in radians to ``[-1, 1]``.

        ``normalized_arm_qpos`` does this for the *live* state; this does it for a
        pose the caller is holding, such as an IK solution about to be sent
        through the action space.
        """
        lower, upper = JOINT_POSITION_LIMITS[:, 0], JOINT_POSITION_LIMITS[:, 1]
        qpos = np.asarray(qpos, dtype=np.float64)
        return np.clip(2.0 * (qpos - lower) / (upper - lower) - 1.0, -1.0, 1.0)

    @staticmethod
    def denormalize_arm_qpos(normalized: np.ndarray) -> np.ndarray:
        """Exact inverse of ``normalized_arm_qpos``, in radians.

        The round trip has to be exact, not merely close: under
        ``control_mode="joint_position"`` a task-space wrapper solves IK in
        radians, normalizes the result to hand it through the ``[-1, 1]`` action
        space, and this maps it back. Any asymmetry between the two directions
        becomes a constant bias on every commanded joint angle.
        """
        lower, upper = JOINT_POSITION_LIMITS[:, 0], JOINT_POSITION_LIMITS[:, 1]
        normalized = np.clip(np.asarray(normalized, dtype=np.float64), -1.0, 1.0)
        return lower + 0.5 * (normalized + 1.0) * (upper - lower)

    def contacting_geoms(self, geom_ids: np.ndarray) -> np.ndarray:
        """Which of ``self.idx.finger_geom_ids`` currently touch ``geom_ids``.

        Returns a boolean array over the two fingers.
        """
        touching = np.zeros(len(self.idx.finger_geom_ids), dtype=bool)
        targets = set(int(g) for g in np.atleast_1d(geom_ids))
        for c in range(self.data.ncon):
            contact = self.data.contact[c]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            for i, finger in enumerate(self.idx.finger_geom_ids):
                if (g1 == finger and g2 in targets) or (g2 == finger and g1 in targets):
                    touching[i] = True
        return touching

    # -- reset ------------------------------------------------------------- #

    def safe_reset_qpos(
        self,
        pose: str | np.ndarray = DEFAULT_RESET_POSE,
        *,
        noise: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Arm configuration for a reset, guaranteed inside the joint limits.

        ``noise`` adds uniform per-joint perturbation in radians; the result is
        clipped back inside the limits with ``JOINT_LIMIT_MARGIN`` to spare, so
        no amount of noise can produce an out-of-range start.
        """
        if isinstance(pose, str):
            if pose not in RESET_POSES:
                raise KeyError(
                    f"unknown reset pose {pose!r}; have {sorted(RESET_POSES)}"
                )
            qpos = RESET_POSES[pose].copy()
        else:
            qpos = np.asarray(pose, dtype=np.float64).copy()
            if qpos.shape != (self.n_arm_joints,):
                raise ValueError(f"pose must have shape (7,), got {qpos.shape}")

        if noise > 0.0:
            generator = rng if rng is not None else np.random.default_rng()
            qpos += generator.uniform(-noise, noise, size=self.n_arm_joints)

        lower, upper = JOINT_POSITION_LIMITS[:, 0], JOINT_POSITION_LIMITS[:, 1]
        margin = JOINT_LIMIT_MARGIN * (upper - lower)
        return np.clip(qpos, lower + margin, upper - margin)

    def reset(
        self,
        pose: str | np.ndarray = DEFAULT_RESET_POSE,
        *,
        gripper_open: bool = True,
        noise: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Write a safe arm+gripper state into ``data`` and zero its velocities.

        Does not call ``mj_forward``; the caller normally does that once after
        placing objects too.
        """
        qpos = self.safe_reset_qpos(pose, noise=noise, rng=rng)
        self.data.qpos[self.idx.arm_qpos_adr] = qpos
        self.data.qvel[self.idx.arm_dof_adr] = 0.0

        finger = GRIPPER_JOINT_RANGE[1] if gripper_open else GRIPPER_JOINT_RANGE[0]
        self.data.qpos[self.idx.finger_qpos_adr] = finger
        self.data.qvel[self.idx.finger_dof_adr] = 0.0

        # Seed ctrl to match, so the first step does not command a jump.
        self.data.ctrl[self.idx.arm_actuator_ids] = 0.0
        self.data.ctrl[self.idx.finger_actuator_ids] = finger
        return qpos

    # -- control ----------------------------------------------------------- #

    def _write_arm_ctrl(self, torque: np.ndarray) -> np.ndarray:
        """Add gravity compensation, clip to the actuator range, write ``ctrl``."""
        if self.gravity_compensation:
            # qfrc_bias is gravity + Coriolis/centrifugal. Cancelling it means the
            # policy commands accelerations rather than spending most of its
            # output budget holding the arm up.
            torque = torque + self.data.qfrc_bias[self.idx.arm_dof_adr]
        torque = np.clip(torque, -JOINT_TORQUE_LIMITS, JOINT_TORQUE_LIMITS)
        self.data.ctrl[self.idx.arm_actuator_ids] = torque
        return torque

    def apply_torque(self, action: np.ndarray) -> np.ndarray:
        """Normalized ``[-1, 1]`` action -> joint torques, scaled by the limits."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if action.shape != (self.n_arm_joints,):
            raise ValueError(f"expected shape (7,), got {action.shape}")
        return self._write_arm_ctrl(action * JOINT_TORQUE_LIMITS)

    def apply_velocity(self, action: np.ndarray) -> np.ndarray:
        """Normalized ``[-1, 1]`` action -> target joint velocity, tracked by a P law.

        The MJCF only has torque motors, so velocity mode is closed here rather
        than by a second actuator set: ``tau = kp * (v_target - v)``, then the
        same clipping and gravity compensation as torque mode.
        """
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if action.shape != (self.n_arm_joints,):
            raise ValueError(f"expected shape (7,), got {action.shape}")
        target_vel = action * JOINT_VELOCITY_LIMITS
        return self._write_arm_ctrl(VELOCITY_KP * (target_vel - self.arm_qvel))

    def apply_joint_position(self, action: np.ndarray) -> np.ndarray:
        """Normalized ``[-1, 1]`` action -> target joint *angle*, tracked by a PD law.

        ``tau = kp * (q_target - q) - kd * qvel``, then the same gravity
        compensation and clipping as the other two modes. Gains come from
        ``_position_gains``.

        This is the mode a task-space controller sits on top of, and the reason it
        exists as a third law rather than being folded into velocity mode: only a
        *position* servo has non-zero stiffness, so only it holds the tool centre
        point against the reaction force of the pads closing on an object.
        Velocity mode has damping but no stiffness -- pushing on it produces a
        bounded drift rather than a bounded deflection -- and torque mode has
        neither, which is why an open-loop torque command loses the cube on
        contact.

        Note the action semantics differ from the other two modes in a way that
        matters for the reward: ``action = 0`` here is not "do nothing", it
        commands the *centre of every joint's range*. Anything that penalizes
        ``sum(action ** 2)`` therefore penalizes the arm's pose rather than its
        effort. ``RewardConfig.action_penalty`` must be zero under this mode; the
        task-space wrapper owns the effort penalty instead and enforces that.

        Requires ``gravity_compensation=True`` to reach a target without steady
        droop: with the bias term left in, holding a pose costs a standing
        position error of ``qfrc_bias / kp``.
        """
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if action.shape != (self.n_arm_joints,):
            raise ValueError(f"expected shape (7,), got {action.shape}")
        target_qpos = self.denormalize_arm_qpos(action)
        torque = self.position_kp * (target_qpos - self.arm_qpos) - self.position_kd * self.arm_qvel
        return self._write_arm_ctrl(torque)

    def apply_gripper(self, action: float) -> float:
        """Normalized ``[-1, 1]`` action -> finger target. ``-1`` closed, ``+1`` open.

        Both fingers get the same target, which keeps the jaw symmetric without
        needing an equality constraint in the model.
        """
        low, high = GRIPPER_JOINT_RANGE
        target = low + 0.5 * (float(np.clip(action, -1.0, 1.0)) + 1.0) * (high - low)
        self.data.ctrl[self.idx.finger_actuator_ids] = target
        return target
