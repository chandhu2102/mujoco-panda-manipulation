"""Squashed Gaussian and diagonal Gaussian policies.

Everything here is closed-form tensor math rather than ``torch.distributions``.
Verified numerically identical to ``torch.distributions.Normal`` and to
``TransformedDistribution(..., TanhTransform)``.

On why it is written this way: as of torch 2.13 Dynamo *can* trace
``torch.distributions``, so a Normal-based head also compiles with
``fullgraph=True`` — the closed form is not required to avoid graph breaks.
What it does buy is: no per-call Distribution object or arg-validation
overhead in the hot path, a numerically stable tanh correction (see
``squashed_log_prob``), and identical behavior on older torch versions where
Dynamo did break on distribution construction. Do not "simplify" it back to
``Normal`` without re-benchmarking the update step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "gaussian_log_prob",
    "gaussian_entropy",
    "gaussian_sample",
    "squashed_log_prob",
    "DiagGaussianHead",
    "SquashedDiagGaussianHead",
]

# log(sqrt(2*pi)) — the Gaussian normalizing constant.
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
# 0.5*log(2*pi*e) — the per-dimension differential entropy of a unit Gaussian.
_HALF_LOG_2PI_E = 0.5 * math.log(2.0 * math.pi * math.e)

# Keeps atanh finite when re-scoring stored actions that sit on the boundary.
_ATANH_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Functional core — all args are (batch, action_dim), all returns are (batch,).
# --------------------------------------------------------------------------- #


def gaussian_log_prob(actions: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    """Diagonal-Gaussian log density, summed over action dimensions."""
    z = (actions - mean) * torch.exp(-log_std)
    return (-0.5 * z.pow(2) - log_std - _LOG_SQRT_2PI).sum(-1)


def gaussian_entropy(log_std: Tensor) -> Tensor:
    """Exact differential entropy, summed over action dimensions."""
    return (log_std + _HALF_LOG_2PI_E).sum(-1)


def gaussian_sample(mean: Tensor, log_std: Tensor) -> Tensor:
    """Reparameterized sample ``mu + sigma * eps``."""
    return mean + torch.exp(log_std) * torch.randn_like(mean)


def squashed_log_prob(pre_tanh: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    """Log density of ``tanh(u)`` where ``u ~ N(mean, exp(log_std))``.

    Applies the tanh change-of-variables correction
    ``-sum log(1 - tanh(u)^2)`` in its numerically stable form
    ``2 * (log 2 - u - softplus(-2u))``, which avoids the ``log(0)`` that the
    naive expression hits once ``tanh(u)`` saturates.
    """
    base = gaussian_log_prob(pre_tanh, mean, log_std)
    correction = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
    return base - correction.sum(-1)


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #


class DiagGaussianHead(nn.Module):
    """Unbounded diagonal Gaussian over continuous actions.

    ``log_std`` is a free parameter independent of the observation by default,
    which is the standard PPO parameterization: exploration then decays on a
    schedule the optimizer controls, rather than being something the policy can
    collapse state-by-state to sharpen its own likelihood.

    Set ``state_dependent_std=True`` for a heteroscedastic head that predicts
    ``log_std`` from features instead.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        *,
        log_std_init: float = -0.5,
        log_std_bounds: tuple[float, float] = (-5.0, 2.0),
        state_dependent_std: bool = False,
        mean_gain: float = 0.01,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std
        self.log_std_min, self.log_std_max = log_std_bounds

        self.mean_layer = nn.Linear(feature_dim, action_dim)
        # Small gain keeps the initial policy near-deterministic-at-zero, so
        # early updates are not dominated by a wildly off-scale mean.
        nn.init.orthogonal_(self.mean_layer.weight, gain=mean_gain)
        nn.init.zeros_(self.mean_layer.bias)

        if state_dependent_std:
            self.log_std_layer = nn.Linear(feature_dim, action_dim)
            nn.init.orthogonal_(self.log_std_layer.weight, gain=mean_gain)
            nn.init.constant_(self.log_std_layer.bias, log_std_init)
            self.log_std_param = None
        else:
            self.log_std_layer = None
            self.log_std_param = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(mean, log_std)``, both broadcast to ``(batch, action_dim)``."""
        mean = self.mean_layer(features)
        if self.log_std_layer is not None:
            log_std = self.log_std_layer(features)
        else:
            log_std = self.log_std_param.expand_as(mean)
        # Clamping both branches bounds exploration from below (no collapse to a
        # deterministic policy that PPO's ratio cannot recover from) and above.
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, mean: Tensor, log_std: Tensor, *, deterministic: bool = False) -> Tensor:
        if deterministic:
            return mean
        return gaussian_sample(mean, log_std)

    def log_prob(self, actions: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
        return gaussian_log_prob(actions, mean, log_std)

    def entropy(self, mean: Tensor, log_std: Tensor, actions: Tensor) -> Tensor:
        del mean, actions  # exact entropy needs neither
        return gaussian_entropy(log_std)


class SquashedDiagGaussianHead(DiagGaussianHead):
    """Tanh-squashed Gaussian, so actions land inside ``[-1, 1]`` by construction.

    Useful when the controller cannot tolerate out-of-range commands. Two
    caveats versus the unsquashed head:

    * Entropy has no closed form. ``entropy()`` returns the single-sample Monte
      Carlo estimate ``-log_prob``, which is unbiased but noisy — keep
      ``ent_coef`` small when using it.
    * Re-scoring a stored action requires inverting the squash via ``atanh``,
      which loses precision for actions near +/-1. PPO's ratio is a difference
      of log-probs computed the same way on both sides, so the error largely
      cancels, but prefer the unsquashed head plus clipping at the env boundary
      if you can.
    """

    def sample(self, mean: Tensor, log_std: Tensor, *, deterministic: bool = False) -> Tensor:
        pre_tanh = mean if deterministic else gaussian_sample(mean, log_std)
        return torch.tanh(pre_tanh)

    def log_prob(self, actions: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
        clamped = torch.clamp(actions, -1.0 + _ATANH_EPS, 1.0 - _ATANH_EPS)
        return squashed_log_prob(torch.atanh(clamped), mean, log_std)

    def entropy(self, mean: Tensor, log_std: Tensor, actions: Tensor) -> Tensor:
        return -self.log_prob(actions, mean, log_std)
