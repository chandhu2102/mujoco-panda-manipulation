"""Differential IK solver used by task-space controllers.

Damped least squares on the site Jacobian -- the same method the ``RESET_POSES``
in ``robots/panda.py`` were solved with offline, now available at runtime so
callers can target a pose that is only known once the episode is sampled.

Deliberately free of any robot-specific imports: everything the solver needs is
passed in as MuJoCo indices, so it cannot create an import cycle with
``robots.panda``.
"""

from __future__ import annotations

import mujoco
import numpy as np

__all__ = ["IKResult", "solve_site_ik"]


class IKResult:
    """Outcome of a solve. Truthy when both tolerances were met."""

    __slots__ = ("qpos", "converged", "pos_error", "rot_error", "iterations")

    def __init__(
        self,
        qpos: np.ndarray,
        converged: bool,
        pos_error: float,
        rot_error: float,
        iterations: int,
    ) -> None:
        self.qpos = qpos
        self.converged = converged
        self.pos_error = pos_error
        self.rot_error = rot_error
        self.iterations = iterations

    def __bool__(self) -> bool:
        return self.converged

    def __repr__(self) -> str:
        return (
            f"IKResult(converged={self.converged}, pos_error={self.pos_error:.5f}, "
            f"rot_error={self.rot_error:.5f}, iterations={self.iterations})"
        )


def solve_site_ik(
    model: mujoco.MjModel,
    site_id: int,
    qpos_adr: np.ndarray,
    dof_adr: np.ndarray,
    target_pos: np.ndarray,
    *,
    target_quat: np.ndarray | None = None,
    q_init: np.ndarray,
    joint_limits: np.ndarray,
    qpos_full: np.ndarray | None = None,
    max_iterations: int = 200,
    pos_tolerance: float = 1e-3,
    rot_tolerance: float = 1e-2,
    damping: float = 1e-2,
    step_scale: float = 0.8,
    scratch: mujoco.MjData | None = None,
    q_nominal: np.ndarray | None = None,
    posture_gain: float = 0.0,
) -> IKResult:
    """Solve for joint angles putting ``site_id`` at ``target_pos``.

    Runs entirely on a scratch ``MjData``, so live simulation state is never
    touched -- callers write ``result.qpos`` themselves if they want it.

    ``target_quat`` is optional but matters more than it looks for grasping: a
    square object presents a graspable face only every 90 degrees, so a solve
    that nails the position while leaving the wrist yawed 45 degrees off closes
    the jaw across the diagonal instead of across a face. Pass the orientation
    you actually want.

    ``qpos_full`` seeds the non-solved part of the configuration (object pose,
    fingers) so the Jacobian is evaluated in the real scene rather than at
    ``qpos0``. It does not constrain the solve; only ``qpos_adr`` moves.

    Damped least squares rather than a plain pseudo-inverse: near a singularity
    the pseudo-inverse produces enormous joint steps, and this solver runs
    unattended inside ``reset``. ``damping`` trades exactness for that safety.

    ``q_nominal`` with a positive ``posture_gain`` adds a secondary objective
    pulling the solution toward that configuration, projected into the null space
    of the task Jacobian so it cannot disturb the pose being solved for. On a
    7-DoF arm solving a 6-DoF pose there is a one-dimensional self-motion manifold
    and the primary term is indifferent along it, so an unbiased solve drifts
    wherever the iteration happens to take it. That drift is not cosmetic: it
    walks the arm between elbow-up and elbow-down branches, and this solver has no
    collision term, so the branch it lands in can be one where a link rests on the
    table. Measured on a task-space descent to a cube -- with ``posture_gain=0``
    the solver reached ``joint1 = 2.1`` rad with ``link4_geom`` in contact with
    ``table_top``, and the tool centre point stalled 2.6 cm above its target with
    the elbow pinned and the position error unresolvable.

    ``scratch`` lets a caller supply the working ``MjData`` instead of having one
    allocated per call. Irrelevant for the curriculum, which solves once per
    reset, and load-bearing for a task-space controller, which solves once per
    control step per env: at 16 envs and a 4096-step rollout that is 4096
    allocations of a full ``MjData`` per iteration, none of whose contents
    survive the call. The buffer is overwritten, never read on entry (beyond
    ``qpos_full``), so one instance can be reused indefinitely -- but it must not
    be the live ``MjData`` of a running simulation, which this would clobber.
    """
    dof_adr = np.asarray(dof_adr, dtype=np.int32)
    qpos_adr = np.asarray(qpos_adr, dtype=np.int32)
    n = dof_adr.size
    joint_limits = np.asarray(joint_limits, dtype=np.float64)
    if joint_limits.shape != (n, 2):
        raise ValueError(f"joint_limits must be ({n}, 2), got {joint_limits.shape}")

    lower, upper = joint_limits[:, 0], joint_limits[:, 1]
    q = np.clip(np.asarray(q_init, dtype=np.float64).copy(), lower, upper)

    if scratch is None:
        scratch = mujoco.MjData(model)
    if qpos_full is not None:
        scratch.qpos[:] = qpos_full
    target_pos = np.asarray(target_pos, dtype=np.float64)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    err = np.zeros(6)
    site_quat = np.zeros(4)
    neg_site_quat = np.zeros(4)
    delta_quat = np.zeros(4)

    pos_error = rot_error = float("inf")
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        scratch.qpos[qpos_adr] = q
        # Minimal pipeline for site poses plus the Jacobian's COM terms.
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)

        err[:3] = target_pos - scratch.site_xpos[site_id]
        pos_error = float(np.linalg.norm(err[:3]))

        if target_quat is None:
            err[3:] = 0.0
            rot_error = 0.0
        else:
            mujoco.mju_mat2Quat(site_quat, scratch.site_xmat[site_id])
            mujoco.mju_negQuat(neg_site_quat, site_quat)
            mujoco.mju_mulQuat(delta_quat, target_quat, neg_site_quat)
            mujoco.mju_quat2Vel(err[3:], delta_quat, 1.0)
            rot_error = float(np.linalg.norm(err[3:]))

        if pos_error <= pos_tolerance and rot_error <= rot_tolerance:
            return IKResult(q, True, pos_error, rot_error, iterations)

        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        if target_quat is None:
            jac = jacp[:, dof_adr]
            residual = err[:3]
        else:
            jac = np.vstack([jacp[:, dof_adr], jacr[:, dof_adr]])
            residual = err

        # dq = J^# e, with J^# = J^T (J J^T + lambda^2 I)^-1
        jjt = jac @ jac.T + (damping ** 2) * np.eye(jac.shape[0])
        if q_nominal is None or posture_gain <= 0.0:
            # No secondary objective, so the damped solve is enough and the
            # explicit pseudo-inverse the projector needs can be skipped.
            dq = jac.T @ np.linalg.solve(jjt, residual)
        else:
            jac_pinv = jac.T @ np.linalg.inv(jjt)
            # dq = J^# e + (I - J^# J) * k * (q_nominal - q). The projector kills
            # any component of the posture pull that would move the site, so this
            # only slides the solution along the self-motion manifold.
            projector = np.eye(n) - jac_pinv @ jac
            dq = jac_pinv @ residual + projector @ (
                posture_gain * (np.asarray(q_nominal, dtype=np.float64) - q)
            )

        q = np.clip(q + step_scale * dq, lower, upper)

    return IKResult(q, False, pos_error, rot_error, iterations)
