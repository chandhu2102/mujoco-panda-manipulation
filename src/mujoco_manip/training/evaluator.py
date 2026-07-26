"""Deterministic evaluation and success-rate reporting.

Two things make an eval number comparable across checkpoints, and this module
enforces both:

* **Fixed initial states.** Every call replays the same episode seeds, so a change
  in the score is a change in the policy. Freshly randomized eval episodes make
  small real improvements indistinguishable from sampling noise -- at 20 episodes
  the standard error on a success rate near 50% is over 11 points.
* **Frozen observation statistics.** If the normalizer keeps updating during
  evaluation, the input scale moves while you measure, so the number reflects the
  normalizer as much as the policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch

from ..algos.common.normalizers import RunningMeanStd

LOGGER = logging.getLogger(__name__)

__all__ = ["Evaluator", "EvaluationConfig", "EpisodeResult", "make_evaluator"]


class _Policy(Protocol):
    def act(
        self, obs: torch.Tensor, *, deterministic: bool = ...
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...


@dataclass
class EvaluationConfig:
    """Evaluation protocol settings."""

    n_episodes: int = 20
    deterministic: bool = True
    """Use the policy mean instead of sampling. Sampling would reintroduce the
    variance the fixed seeds exist to remove."""
    base_seed: int = 1_000_000
    """Seeds are ``base_seed + i``. Kept far from training seeds so evaluation
    never lands on states the policy trained against."""
    max_steps: int | None = None
    """Per-episode step cap. ``None`` defers to the env's own horizon."""
    success_key: str = "is_success"
    device: str = "cpu"
    log_per_episode: bool = False


@dataclass
class EpisodeResult:
    """One evaluation episode."""

    seed: int
    success: bool
    ret: float
    length: int
    final_distance: float | None = None
    info: dict[str, float] = field(default_factory=dict)


class Evaluator:
    """Runs a fixed set of deterministic episodes against a single env.

    Satisfies the ``Evaluator`` protocol in ``training.trainer``, so it can be
    handed straight to ``PPOTrainer(evaluator=...)``.

    Deliberately single-env rather than vectorized: explicit per-episode seeding
    is what makes the initial states identical across calls. A vector env
    autoresets from its own RNG stream, and since episode lengths change as the
    policy improves, that stream desynchronizes and the eval set silently drifts.
    Cost is ``n_episodes * horizon`` steps, so budget ``eval_interval`` for it.
    """

    def __init__(
        self,
        env: Any,
        config: EvaluationConfig | None = None,
        *,
        obs_rms: RunningMeanStd | None = None,
        obs_clip: float = 10.0,
        seeds: Sequence[int] | None = None,
    ) -> None:
        self.env = env
        self.cfg = config or EvaluationConfig()
        self.obs_rms = obs_rms
        self.obs_clip = obs_clip
        self.device = torch.device(self.cfg.device)
        self.seeds: list[int] = (
            list(seeds)
            if seeds is not None
            else [self.cfg.base_seed + i for i in range(self.cfg.n_episodes)]
        )
        self.last_results: list[EpisodeResult] = []

    # -- helpers ----------------------------------------------------------- #

    def _prepare_obs(self, obs: np.ndarray) -> torch.Tensor:
        array = np.asarray(obs, dtype=np.float32)
        if self.obs_rms is not None:
            # Frozen: normalize with the training statistics, never update them.
            array = self.obs_rms.normalize(array, clip=self.obs_clip)
        return torch.as_tensor(
            array, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    @staticmethod
    def _distance_from_info(info: dict[str, Any]) -> float | None:
        for key in ("dist/eef_to_goal", "dist/object_to_goal", "dist/eef_to_object"):
            if key in info:
                return float(info[key])
        return None

    # -- protocol ---------------------------------------------------------- #

    @torch.no_grad()
    def evaluate(self, policy: _Policy) -> dict[str, float]:
        """Run every seed once and aggregate. Returns metrics for the trainer."""
        was_training = bool(getattr(policy, "training", False))
        if hasattr(policy, "eval"):
            policy.eval()
        try:
            results = [self._run_episode(policy, seed) for seed in self.seeds]
        finally:
            if was_training and hasattr(policy, "train"):
                policy.train()

        self.last_results = results
        return self._aggregate(results)

    def _run_episode(self, policy: _Policy, seed: int) -> EpisodeResult:
        obs, info = self.env.reset(seed=seed)
        total, steps = 0.0, 0
        success = bool(info.get(self.cfg.success_key, False))
        limit = self.cfg.max_steps or getattr(self.env, "max_episode_steps", 1000)

        while steps < limit:
            action, _, _ = policy.act(
                self._prepare_obs(obs), deterministic=self.cfg.deterministic
            )
            obs, reward, terminated, truncated, info = self.env.step(
                action.squeeze(0).cpu().numpy()
            )
            total += float(reward)
            steps += 1
            # Latch success: an episode that reaches the goal and then drifts out
            # before the horizon still succeeded.
            success = success or bool(info.get(self.cfg.success_key, False))
            if terminated or truncated:
                break

        result = EpisodeResult(
            seed=seed,
            success=success,
            ret=total,
            length=steps,
            final_distance=self._distance_from_info(info),
            info={
                k: float(v)
                for k, v in info.items()
                if isinstance(v, (bool, int, float, np.floating, np.integer))
            },
        )
        if self.cfg.log_per_episode:
            LOGGER.info(
                "eval seed=%d success=%s return=%.3f length=%d final_dist=%s",
                seed, result.success, result.ret, result.length,
                "n/a" if result.final_distance is None else f"{result.final_distance:.4f}",
            )
        return result

    def _aggregate(self, results: list[EpisodeResult]) -> dict[str, float]:
        if not results:
            return {"success_rate": 0.0, "episodes": 0.0}
        successes = np.array([r.success for r in results], dtype=np.float64)
        returns = np.array([r.ret for r in results], dtype=np.float64)
        lengths = np.array([r.length for r in results], dtype=np.float64)
        rate = float(successes.mean())
        n = len(results)

        metrics = {
            "success_rate": rate,
            # Binomial standard error, so a reader can tell a real gain from noise
            # at this episode count.
            "success_rate_stderr": float(np.sqrt(max(rate * (1.0 - rate), 0.0) / n)),
            "episodes": float(n),
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "length_mean": float(lengths.mean()),
        }
        distances = [r.final_distance for r in results if r.final_distance is not None]
        if distances:
            metrics["final_distance_mean"] = float(np.mean(distances))
            metrics["final_distance_min"] = float(np.min(distances))
        return metrics

    # -- convenience ------------------------------------------------------- #

    def summary(self) -> str:
        """One-line human summary of the most recent evaluation."""
        if not self.last_results:
            return "no evaluation run yet"
        m = self._aggregate(self.last_results)
        parts = [
            f"success {m['success_rate']:.1%} +/- {m['success_rate_stderr']:.1%}",
            f"return {m['return_mean']:.2f}",
            f"length {m['length_mean']:.0f}",
        ]
        if "final_distance_mean" in m:
            parts.append(f"final_dist {m['final_distance_mean']:.4f} m")
        return " | ".join(parts)

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()


def make_evaluator(
    env_fn: Callable[[], Any],
    config: EvaluationConfig | None = None,
    **kwargs: Any,
) -> Evaluator:
    """Build an evaluator over a freshly constructed env.

    Use a separate env instance from training: sharing one would let evaluation
    reset the episode the rollout is midway through collecting.
    """
    return Evaluator(env_fn(), config, **kwargs)
