"""Behavioural cloning: demonstration storage, pre-training, and a PPO anchor.

Three pieces, used in that order:

* ``DemoBuffer`` -- the recorded transitions, normalized and on-device.
* ``BCPretrainer`` -- supervised fit of the policy to the demonstrations, plus a
  value-head fit to their discounted returns.
* ``DAPGTrainer`` -- ``PPOTrainer`` with a decaying demonstration likelihood term
  added to the minibatch loss, so the clone is not immediately undone.

Two corrections to the obvious design, both of which matter enough to state up
front.

**The loss is not cross-entropy.** Cross-entropy is the objective for a
categorical head. This policy is a diagonal Gaussian over continuous actions
(``DiagGaussianHead``), so the maximum-likelihood objective is the negative log
density of the demonstrated action, ``-log N(a | mu(s), sigma)``. That is
available directly as ``policy.head.log_prob`` and is what ``BCPretrainer`` uses.
Minimizing it is *not* the same as minimizing MSE on the mean: the log-density
also fits ``sigma``, which is exactly the behaviour that has to be controlled
deliberately (see ``BCPretrainer.fit``).

**Demonstrations cannot be put in the PPO buffer.** ``RolloutBuffer`` stores the
log-probability under the policy that acted, and PPO's ratio
``exp(new_logprob - old_logprob)`` is only an importance weight if that is true.
Inserting scripted transitions makes the denominator a number no policy produced,
so the ratio is meaningless and the clipping does not bound anything. GAE has the
same problem from the other side: it needs the value function's estimates along
the *behaviour* distribution. The two sound ways in are to change the
initialization (``BCPretrainer``) or to add a separate supervised term to the loss
(``DAPGTrainer``) -- never to contaminate the on-policy batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..algos.common.normalizers import RunningMeanStd
from .trainer import PPOConfig, PPOTrainer

LOGGER = logging.getLogger(__name__)

__all__ = ["DemoBuffer", "BCConfig", "BCPretrainer", "DAPGTrainer"]


class DemoBuffer:
    """Demonstration transitions, whitened and resident on the training device.

    Whitening is applied here rather than left to the caller because it is the
    step most easily got wrong in a way that produces no error. In training,
    observations are normalized by ``NormalizeObservation`` *at the vector-env
    boundary*, so the policy never sees a raw observation. ``record_demos.py``
    writes raw observations, since it builds a single env below that wrapper. A
    clone fitted on raw observations and then run behind the normalizer is being
    evaluated on inputs scaled differently by one to two orders of magnitude in
    some dimensions, and presents as BC having done nothing at all.

    Whitening is applied **lazily**, per ``sample``/``epochs`` batch, rather than
    once in ``__init__``. ``obs_rms`` is a live object owned by the caller and
    mutated after this buffer is built: ``NormalizeObservation`` updates it from
    rollouts, and ``scripts/train.py`` loads a checkpoint's statistics into it
    *after* constructing the buffer. Caching a whitened copy froze a snapshot of
    whatever the statistics happened to be at construction, which on the
    ``--resume`` path is an unfitted normalizer -- mean 0, and std 1 via the
    ``var_floor`` guard, i.e. no whitening at all.

    That was not a small error. Measured on a converged clone whose demo-fitted
    statistics were loaded a few lines too late, the DAPG anchor's own NLL came out
    at 122.9 instead of -4.30 (``scripts/pretrain_bc.py`` reports -4.267 on the
    same data and weights), and the gradient it contributed was 35x the intended
    one: 305 against 8.7. ``DAPGTrainer`` adds ``bc_coef * bc_loss`` to a policy
    loss of order 1e-2, so that term dominated the actor and pulled it toward
    reproducing demonstration actions at inputs the policy never sees. A warm start
    worth 87% task success fell to 2% inside 40 iterations, with the episode return
    decreasing monotonically the whole way.

    Per-batch whitening costs one broadcast over ``(batch, obs_dim)`` and makes the
    buffer track the statistics instead of a snapshot of them, which is what the
    paragraph above always intended.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        device: torch.device,
        obs_rms: RunningMeanStd | None = None,
        clip: float = 10.0,
        gamma: float = 0.995,
        return_rms: RunningMeanStd | None = None,
    ) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"demonstration file not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            raw_obs = np.asarray(data["observations"], dtype=np.float64)
            actions = np.asarray(data["actions"], dtype=np.float32)
            rewards = np.asarray(data["rewards"], dtype=np.float64)
            starts = np.asarray(data["episode_starts"], dtype=np.int64)
            lengths = np.asarray(data["episode_lengths"], dtype=np.int64)
            self.control_mode = str(data["control_mode"])
            self.task_space = bool(data["task_space"])

        self.path = path
        self.obs_dim = int(raw_obs.shape[1])
        self.action_dim = int(actions.shape[1])
        self.n_episodes = int(lengths.size)

        # Fit the normalizer on the demonstrations when one is not supplied. This
        # becomes the statistics PPO starts from, written out by the caller.
        if obs_rms is None:
            obs_rms = RunningMeanStd((self.obs_dim,))
            obs_rms.update(raw_obs)
        elif obs_rms.shape != (self.obs_dim,):
            raise ValueError(
                f"obs_rms shape {obs_rms.shape} != demonstration obs dim "
                f"({self.obs_dim},)"
            )
        self.obs_rms = obs_rms
        self.clip = clip

        # Raw, unwhitened. See the class docstring: whitening happens per batch so
        # that a later mutation of obs_rms -- a checkpoint load, or rollout updates
        # -- is reflected instead of silently missed.
        self._raw_observations = torch.as_tensor(
            np.asarray(raw_obs, dtype=np.float32), device=device
        )
        self.actions = torch.as_tensor(actions, device=device)

        raw_returns = self._discounted_returns(rewards, starts, lengths, gamma)
        self._raw_returns = torch.as_tensor(raw_returns, device=device)

        # Return scaling, for runs where PPO normalizes the reward. The value head
        # cloned here has to predict targets on the same scale PPO will produce, or
        # the warm start hands over a critic wrong by a factor of std(return) -- ~425
        # on the 400-episode obstacle set -- with nothing raised. `return_rms` is the
        # statistic both sides share; seeding it from the demonstrations is what lets
        # them agree from step one rather than after PPO's running estimate catches
        # up.
        #
        # Note the two quantities are not identical. NormalizeReward accumulates
        # *backwards-looking* discounted sums during a rollout, while these are the
        # usual forward discounted returns-to-go, so the seed is an order-of-magnitude
        # match rather than an exact one -- for this task 425 against ~744 for a
        # 250-step horizon. That is the point: it removes the initial shock, and the
        # running statistic adapts from there.
        self.return_rms = return_rms
        if return_rms is not None and return_rms.count < 2:
            return_rms.update(raw_returns.astype(np.float64))
        self.device = device

    @staticmethod
    def _discounted_returns(
        rewards: np.ndarray, starts: np.ndarray, lengths: np.ndarray, gamma: float
    ) -> np.ndarray:
        """Per-episode reverse discounted sum.

        Computed per episode rather than over the concatenated array: the episodes
        are stored back to back, so a single reverse sweep would discount the first
        steps of each episode against the tail of the previous one and produce a
        value target for a transition that never happened.

        Truncation is treated as termination -- the demonstrations end when the
        script finishes, not when the env's horizon does, so there is no
        bootstrap value available and none is invented.
        """
        out = np.zeros_like(rewards, dtype=np.float32)
        for start, length in zip(starts, lengths):
            running = 0.0
            for i in range(int(start) + int(length) - 1, int(start) - 1, -1):
                running = float(rewards[i]) + gamma * running
                out[i] = running
        return out

    def __len__(self) -> int:
        return int(self._raw_observations.shape[0])

    def _whiten(self, raw: Tensor) -> Tensor:
        """Apply the *current* ``obs_rms`` to a batch of raw observations.

        Mirrors ``RunningMeanStd.normalize`` exactly -- including the ``var_floor``
        guard that leaves a constant dimension mean-centred but unscaled, and the
        ``count < 2`` pass-through. The two have to agree, because the policy is
        scored on demonstrations whitened here and acts on rollouts whitened there,
        so any divergence is a silent difference between the clone's inputs and the
        policy's. Verified equal to within float32 rounding (9.5e-7) on the fitted
        statistics, and exactly equal at count 0.

        The ``count < 2`` branch is not decoration: without it this method subtracts
        a mean estimated from a single sample while ``normalize`` passes the same
        input straight through, a measured max difference of 0.994. The live
        ``NormalizeObservation`` updates one batch per step (16 envs here, so the
        count goes 0 -> 16) and a resumed run loads a fully fitted normalizer, so
        nothing in this pipeline lands on count 1 -- which is exactly why the
        disagreement would have gone unnoticed.
        """
        rms = self.obs_rms
        if rms.count < 2:
            out = raw
        else:
            # float64 for the subtract-and-divide, then back, because that is what
            # `normalize` does: it casts to `self.mean.dtype`. Dividing by a small
            # std amplifies the difference between doing this in float32 and in
            # float64 by 1/std -- at an early-run std of 2.6e-3 that showed up as a
            # 5.1e-5 disagreement, against 9.5e-7 on fitted statistics.
            work = raw.to(torch.float64)
            mean = torch.as_tensor(rms.mean, dtype=torch.float64, device=raw.device)
            std = torch.as_tensor(rms.std, dtype=torch.float64, device=raw.device)
            out = ((work - mean) / std).to(raw.dtype)
        if self.clip is not None:
            out = torch.clamp(out, -float(self.clip), float(self.clip))
        return out

    @property
    def observations(self) -> Tensor:
        """All transitions, whitened by the current statistics.

        A property rather than a stored tensor so it cannot go stale; it rebuilds
        the full array on each access, so prefer ``sample``/``epochs`` in a loop.
        """
        return self._whiten(self._raw_observations)

    @property
    def returns(self) -> Tensor:
        """Discounted returns, divided by ``return_rms`` when one was supplied.

        Raw and unchanged when it was not, which is every run that does not
        normalize the reward -- so the default path is bit-identical. A property for
        the same reason ``observations`` is one: the statistic is a live object the
        caller may update, and a cached copy would silently go stale.
        """
        if self.return_rms is None:
            return self._raw_returns
        std = float(self.return_rms.std)
        return self._raw_returns / std

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Uniform minibatch of ``(obs, actions, returns)``, sampled with replacement."""
        idx = torch.randint(
            len(self), (batch_size,), device=self.device, generator=generator
        )
        return self._whiten(self._raw_observations[idx]), self.actions[idx], self.returns[idx]

    def epochs(
        self, batch_size: int, n_epochs: int, *, drop_last: bool = True
    ) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        """Shuffled full passes, for supervised pre-training.

        An undersized final batch is dropped by default, which is not the usual
        rounding-error argument -- here it corrupts the weights. ``max_grad_norm``
        is 0.5 while the gradient norm at convergence is ~1200 for a full batch
        (the value loss carries it, on an unnormalized return scale), so *every*
        step is clipped to exactly the ceiling and only its direction varies. A
        tail batch therefore does not take a proportionally smaller step: it takes
        a full-magnitude one in a direction estimated from a handful of
        transitions, as the last thing that happens each epoch.

        With 40961 transitions at ``batch_size`` 256 the tail is a single sample,
        and that one step roughly doubled action MSE (0.0126 -> 0.0261) on the
        400-episode set -- undoing most of the epochs before it. Because the
        permutation covers all ``n`` every pass, the dropped samples differ each
        epoch, so nothing is systematically excluded from the fit.

        ``drop_last=False`` restores the old behaviour. Kept for callers that want
        every sample in a single pass, but it should not be used for training
        while the gradient norm sits orders of magnitude above the clip.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        n = len(self)
        n_full = n // batch_size
        # Fall back to the short batch when there is not even one full batch --
        # dropping it would make the generator yield nothing and the fit silently
        # do no work at all, which is worse than a noisy step.
        limit = n_full * batch_size if (drop_last and n_full >= 1) else n
        for _ in range(n_epochs):
            order = torch.randperm(n, device=self.device)
            for start in range(0, limit, batch_size):
                idx = order[start : start + batch_size]
                yield (
                    self._whiten(self._raw_observations[idx]),
                    self.actions[idx],
                    self.returns[idx],
                )


@dataclass
class BCConfig:
    """Pre-training hyperparameters."""

    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 0.5

    value_coef: float = 0.5
    """Weight on the value-head regression against the demonstrations' returns.

    Worth doing rather than leaving the critic at its initialization. PPO's first
    advantage estimates are ``r + gamma * V(s') - V(s)`` with whatever ``V`` it
    has; an untrained critic makes those estimates noise, and the largest policy
    steps of the whole run are taken against them. Fitting the critic on the
    demonstration returns at least puts it on the right scale along the
    trajectories the policy is about to visit."""

    final_log_std: float | None = 0.3
    """Sigma to install after cloning, or None to keep what BC produced.

    Not optional in practice. The likelihood objective fits ``sigma`` to the
    residual spread of the demonstrations, and a scripted controller is
    deterministic, so the residual is small and ``sigma`` collapses toward
    ``log_std_min``. A collapsed sigma is unrecoverable under PPO from the
    policy-gradient term alone -- the ratio for any action far from the mean
    underflows -- so exploration has to be reinstated explicitly, at a width
    chosen for exploration rather than inherited from a fit."""

    holdout_fraction: float = 0.1
    """Episodes held out to report a generalization gap.

    Reported, not acted on: with a few dozen demonstrations the point is to notice
    that the clone has memorized them, not to early-stop on it."""


class BCPretrainer:
    """Fits an ``ActorCritic`` to demonstrations by maximum likelihood."""

    def __init__(
        self,
        policy: nn.Module,
        demos: DemoBuffer,
        cfg: BCConfig | None = None,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.policy = policy
        self.demos = demos
        self.cfg = cfg or BCConfig()
        self.device = device or demos.device
        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    def _losses(self, obs: Tensor, actions: Tensor, returns: Tensor) -> tuple[Tensor, dict[str, float]]:
        # evaluate_actions gives exactly the three quantities wanted, and gives the
        # log-probability under the same head PPO will score with -- so the clone is
        # fitted in the metric it will later be optimized in.
        log_prob, _entropy, values = self.policy.evaluate_actions(obs, actions)
        nll = -log_prob.mean()
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        loss = nll + self.cfg.value_coef * value_loss
        with torch.no_grad():
            mean, log_std = self.policy._policy_params(obs)
            mse = ((mean - actions) ** 2).mean()
        return loss, {
            "nll": float(nll),
            "value_loss": float(value_loss),
            "action_mse": float(mse),
            "sigma_mean": float(log_std.exp().mean()),
        }

    @torch.no_grad()
    def evaluate(self, obs: Tensor, actions: Tensor, returns: Tensor) -> dict[str, float]:
        _loss, stats = self._losses(obs, actions, returns)
        return stats

    def fit(self) -> dict[str, float]:
        """Run the supervised fit. Returns the final training statistics."""
        cfg = self.cfg
        self.policy.train()
        stats: dict[str, float] = {}

        for epoch in range(cfg.epochs):
            sums: dict[str, float] = {}
            n_batches = 0
            for obs, actions, returns in self.demos.epochs(cfg.batch_size, 1):
                loss, batch_stats = self._losses(obs, actions, returns)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                n_batches += 1
                for key, value in batch_stats.items():
                    sums[key] = sums.get(key, 0.0) + value
            stats = {k: v / max(1, n_batches) for k, v in sums.items()}
            if epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1:
                LOGGER.info(
                    "bc epoch %3d/%d  nll=%8.3f  action_mse=%.5f  value_loss=%8.3f  sigma=%.4f",
                    epoch + 1, cfg.epochs, stats["nll"], stats["action_mse"],
                    stats["value_loss"], stats["sigma_mean"],
                )

        if cfg.final_log_std is not None:
            stats["sigma_before_reset"] = stats.get("sigma_mean", float("nan"))
            self.set_action_std(cfg.final_log_std)
            stats["sigma_mean"] = cfg.final_log_std
        return stats

    def set_action_std(self, sigma: float) -> None:
        """Overwrite the head's sigma in place. See ``BCConfig.final_log_std``.

        Mirrors ``scripts/train.py``'s ``reset_action_std``, including its handling
        of a state-dependent head, so a run that clones and a run that resumes end
        up with the policy width set the same way.
        """
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        head = self.policy.head
        param = head.log_std_param
        if param is None:
            layer = getattr(head, "log_std_layer", None)
            param = None if layer is None else layer.bias
        if param is None:
            raise RuntimeError("policy head exposes no log_std parameter")
        with torch.no_grad():
            param.fill_(float(np.log(sigma)))
        LOGGER.info("set action sigma to %.4g after cloning", sigma)


class DAPGTrainer(PPOTrainer):
    """``PPOTrainer`` plus a decaying demonstration-likelihood term.

    The term is added inside ``_minibatch_loss`` so it participates in the same
    optimizer step as the PPO objective, rather than being interleaved as separate
    updates -- interleaving lets each objective partly undo the other between
    steps, and makes the effective learning rate on the sum depend on the ordering.

    ``compile_update`` must be off: the demonstration minibatch is redrawn every
    call, so the fused graph would see a new tensor each time. The base class
    already falls back to eager on a compile failure, but the config should say so
    rather than rely on that.
    """

    def __init__(
        self,
        env: Any,
        policy: Any,
        cfg: PPOConfig,
        demos: DemoBuffer,
        *,
        bc_coef: float = 0.1,
        bc_decay: float = 0.999,
        bc_batch_size: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(env, policy, cfg, **kwargs)
        self.demos = demos
        self.bc_coef_initial = float(bc_coef)
        self.bc_decay = float(bc_decay)
        self.bc_batch_size = int(bc_batch_size)
        if cfg.compile_update:
            LOGGER.warning(
                "compile_update=True with a BC auxiliary term; the demonstration "
                "minibatch changes every call, so expect recompilation or a "
                "fallback to eager"
            )

    @property
    def bc_coef(self) -> float:
        """Current coefficient. Decayed on ``iteration``, so it is independent of
        batch size and of how many minibatches an update happened to run."""
        return self.bc_coef_initial * (self.bc_decay ** max(0, self.iteration - 1))

    def _minibatch_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        old_logprobs: Tensor,
        old_values: Tensor,
        advantages: Tensor,
        returns: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss, stats = super()._minibatch_loss(
            obs, actions, old_logprobs, old_values, advantages, returns
        )
        coef = self.bc_coef
        if coef <= 0.0:
            return loss, stats
        demo_obs, demo_actions, _demo_returns = self.demos.sample(self.bc_batch_size)
        demo_logprob, _entropy, _values = self.policy.evaluate_actions(
            demo_obs, demo_actions
        )
        bc_loss = -demo_logprob.mean()
        stats = dict(stats)
        stats["bc_loss"] = bc_loss.detach()
        stats["bc_coef"] = torch.as_tensor(coef, device=bc_loss.device)
        return loss + coef * bc_loss, stats
