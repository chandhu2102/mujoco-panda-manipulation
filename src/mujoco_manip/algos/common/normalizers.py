"""Running observation / reward normalizers.

``RunningMeanStd`` accumulates mean and variance with Welford's algorithm, in the
batched (Chan et al.) form. The naive alternative -- keeping running sums of
``x`` and ``x^2`` and subtracting -- loses precision catastrophically once the
sums grow large relative to the variance, which is exactly the regime a 1.5M-step
run reaches. Welford never forms that difference: it tracks the mean and the sum
of squared deviations about it, so accuracy does not decay with sample count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # gymnasium is only needed for the vector wrapper
    from gymnasium.vector import VectorEnv, VectorObservationWrapper
except Exception:  # pragma: no cover
    VectorEnv = object  # type: ignore[assignment,misc]
    VectorObservationWrapper = object  # type: ignore[assignment,misc]

__all__ = ["RunningMeanStd", "NormalizeObservation", "NormalizeReward"]


class RunningMeanStd:
    """Online mean/variance via Welford's algorithm, batched.

    Merging a batch of size ``n`` into an aggregate of size ``N``::

        delta  = batch_mean - mean
        mean  += delta * n / (N + n)
        M2    += batch_M2 + delta^2 * N * n / (N + n)

    With ``n = 1`` this reduces exactly to scalar Welford. ``M2`` is the sum of
    squared deviations from the running mean, so ``var = M2 / count``.
    """

    def __init__(
        self,
        shape: tuple[int, ...] = (),
        *,
        epsilon: float = 1e-8,
        var_floor: float = 1e-12,
        dtype: Any = np.float64,
    ) -> None:
        self.shape = tuple(shape)
        self.epsilon = float(epsilon)
        self.var_floor = float(var_floor)
        # Accumulate in float64 whatever the observation dtype: the other half of
        # the precision story, and nearly free.
        self.mean = np.zeros(self.shape, dtype=dtype)
        self.m2 = np.zeros(self.shape, dtype=dtype)
        self.count = 0.0

    # -- accumulation ------------------------------------------------------ #

    def update(self, batch: np.ndarray) -> None:
        """Merge a batch shaped ``(n, *shape)``, or ``(*shape,)`` for one sample."""
        batch = np.asarray(batch, dtype=self.mean.dtype)
        if batch.shape == self.shape:
            batch = batch[np.newaxis, ...]
        if batch.shape[1:] != self.shape:
            raise ValueError(
                f"batch has trailing shape {batch.shape[1:]}, expected {self.shape}"
            )
        n = batch.shape[0]
        if n == 0:
            return
        batch_mean = batch.mean(axis=0)
        # M2 of the batch about its own mean, not about the running mean.
        batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)
        self.update_from_moments(batch_mean, batch_m2, n)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_m2: np.ndarray, batch_count: int
    ) -> None:
        """Merge pre-computed batch moments, so callers can avoid re-reducing data."""
        if batch_count <= 0:
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = (
            self.m2 + batch_m2 + np.square(delta) * (self.count * batch_count / total)
        )
        self.count = total

    # -- readout ----------------------------------------------------------- #

    @property
    def var(self) -> np.ndarray:
        """Population variance. Ones until a second sample makes it meaningful."""
        if self.count < 2:
            return np.ones_like(self.mean)
        return self.m2 / self.count

    @property
    def std(self) -> np.ndarray:
        """Standard deviation, with constant dimensions passed through unscaled.

        Without the ``var_floor`` guard, a dimension that never varies (a padded
        slot, a task-irrelevant field held fixed) would be divided by
        ``sqrt(epsilon)`` -- an amplification of ~1e4 at the default epsilon --
        turning pure numerical noise into a feature that saturates the clip
        bound. Dimensions below the floor keep ``std = 1``, i.e. mean-centred but
        unscaled.
        """
        var = self.var
        return np.where(var < self.var_floor, 1.0, np.sqrt(var + self.epsilon))

    def normalize(
        self, x: np.ndarray, *, clip: float | None = 10.0, dtype: Any = np.float32
    ) -> np.ndarray:
        """Whiten ``x``, optionally clipping to +/- ``clip`` standard deviations.

        Clipping matters on a long run: one physics blow-up or an unset field can
        put a single observation thousands of sigma out, and unclipped that value
        goes straight into the policy's first layer.
        """
        if self.count < 2:
            # Too little data to whiten; passing through beats dividing by a
            # variance estimate that is still a placeholder.
            out = np.asarray(x, dtype=dtype)
            return np.clip(out, -clip, clip) if clip is not None else out
        out = (np.asarray(x, dtype=self.mean.dtype) - self.mean) / self.std
        if clip is not None:
            out = np.clip(out, -clip, clip)
        return out.astype(dtype)

    def denormalize(self, x: np.ndarray, *, dtype: Any = np.float32) -> np.ndarray:
        return (np.asarray(x, dtype=self.mean.dtype) * self.std + self.mean).astype(dtype)

    # -- persistence ------------------------------------------------------- #

    def state_dict(self) -> dict[str, Any]:
        """Plain-python state.

        The trainer's checkpoint captures the policy and optimizer, not env
        wrapper state, so whoever owns this must save it explicitly. Statistics
        lost on resume silently rescale every observation the policy trained on,
        which looks like a mysterious performance cliff rather than a bug.
        """
        return {
            "shape": self.shape,
            "epsilon": self.epsilon,
            "var_floor": self.var_floor,
            "mean": self.mean.copy(),
            "m2": self.m2.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        shape = tuple(state["shape"])
        if shape != self.shape:
            raise ValueError(
                f"normalizer shape mismatch: checkpoint {shape} vs current {self.shape}"
            )
        self.epsilon = float(state["epsilon"])
        self.var_floor = float(state.get("var_floor", self.var_floor))
        self.mean = np.asarray(state["mean"], dtype=self.mean.dtype).copy()
        self.m2 = np.asarray(state["m2"], dtype=self.mean.dtype).copy()
        self.count = float(state["count"])

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RunningMeanStd(shape={self.shape}, count={self.count:.0f}, "
            f"mean_abs={np.abs(self.mean).mean():.4g}, std_mean={self.std.mean():.4g})"
        )


class NormalizeObservation(VectorObservationWrapper):  # type: ignore[misc]
    """Whiten vector-env observations with running statistics.

    Normalizing here rather than inside the policy is deliberate. PPO stores
    observations in the rollout buffer and re-scores them during the update; if
    the statistics moved in between, ``evaluate_actions`` would see differently
    scaled inputs than ``act`` did, and the importance ratio would be wrong even
    for an unchanged policy. Normalizing at the env boundary means the buffer
    holds already-whitened observations, so the ratio stays exact.

    Pass ``training=False`` for evaluation: statistics freeze, so an eval score
    reflects the policy alone rather than a moving input scale.
    """

    def __init__(
        self,
        env: VectorEnv,
        *,
        obs_rms: RunningMeanStd | None = None,
        clip: float = 10.0,
        epsilon: float = 1e-8,
        training: bool = True,
    ) -> None:
        super().__init__(env)
        shape = tuple(self.single_observation_space.shape)
        self.obs_rms = (
            obs_rms if obs_rms is not None else RunningMeanStd(shape, epsilon=epsilon)
        )
        if self.obs_rms.shape != shape:
            raise ValueError(
                f"obs_rms shape {self.obs_rms.shape} != observation shape {shape}"
            )
        self.clip = float(clip)
        self.training = training

    def observations(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations)
        if self.training:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs, clip=self.clip)

    def state_dict(self) -> dict[str, Any]:
        return {"obs_rms": self.obs_rms.state_dict(), "clip": self.clip}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.obs_rms.load_state_dict(state["obs_rms"])
        self.clip = float(state.get("clip", self.clip))


@dataclass
class NormalizeReward:
    """Scale rewards by the running std of the discounted return.

    Off by default in this project: the dense reward in
    ``envs/manipulation_env.py`` is already built from bounded ``1 - tanh(k*d)``
    terms with hand-chosen stage weights, so rescaling it largely discards that
    design and ties the effective learning rate to whatever the return happens to
    look like early on. It is here for reward functions that are not bounded.

    Scale only, never the mean: subtracting a baseline from the reward changes
    the optimal policy rather than just its conditioning.
    """

    num_envs: int
    gamma: float = 0.99
    clip: float = 10.0
    epsilon: float = 1e-8
    return_rms: RunningMeanStd = field(init=False)

    def __post_init__(self) -> None:
        self.return_rms = RunningMeanStd((), epsilon=self.epsilon)
        self._returns = np.zeros(self.num_envs, dtype=np.float64)

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        dones = np.asarray(dones, dtype=bool).reshape(-1)
        self._returns = self._returns * self.gamma + rewards
        self.return_rms.update(self._returns)
        # Reset the discounted accumulator at episode boundaries, else the
        # statistic drifts toward the scale of an infinitely long episode.
        self._returns[dones] = 0.0
        return np.clip(rewards / self.return_rms.std, -self.clip, self.clip)

    def state_dict(self) -> dict[str, Any]:
        return {
            "return_rms": self.return_rms.state_dict(),
            "returns": self._returns.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.return_rms.load_state_dict(state["return_rms"])
        self._returns = np.asarray(state["returns"], dtype=np.float64).copy()
