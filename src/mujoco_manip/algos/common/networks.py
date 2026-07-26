"""MLP and CNN actor-critic torso definitions.

``ActorCritic`` satisfies the ``ActorCritic`` protocol in
``mujoco_manip.training.trainer``: ``act``, ``evaluate_actions``, ``parameters``.

Both hot paths (``act`` and ``evaluate_actions``) compile with
``fullgraph=True`` and zero graph breaks, so the trainer's
``compile_update=True`` fuses the policy forward into the PPO loss as one
graph. Keeping that property means: no Python branching on tensor *values*, no
``.item()`` / ``.cpu()`` calls, and no data-dependent shapes. Config-time
branches (``shared_torso``, ``squash``) are fine — they are static per run.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .distributions import DiagGaussianHead, SquashedDiagGaussianHead

__all__ = ["mlp", "MLPTorso", "ActorCritic"]


def _flat_dim(shape: int | Sequence[int]) -> int:
    if isinstance(shape, int):
        return shape
    return int(math.prod(shape))


def mlp(
    in_dim: int,
    hidden_sizes: Sequence[int],
    activation: Callable[[], nn.Module] = nn.Tanh,
    *,
    gain: float = math.sqrt(2.0),
) -> nn.Sequential:
    """Orthogonally-initialized MLP with an activation after every layer.

    Orthogonal init with gain sqrt(2) is the on-policy-RL default; it keeps
    activation variance stable through depth, which matters more here than in
    supervised training because the data distribution shifts under the policy.
    """
    layers: list[nn.Module] = []
    dim = in_dim
    for size in hidden_sizes:
        layer = nn.Linear(dim, size)
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)
        layers += [layer, activation()]
        dim = size
    return nn.Sequential(*layers)


class MLPTorso(nn.Module):
    """Flatten-then-MLP feature extractor for state observations.

    Any module with an ``out_dim`` attribute mapping ``(batch, *obs_shape) ->
    (batch, out_dim)`` can be substituted — that is the seam a CNN torso for
    pixel observations drops into (see ``observations/encoders.py``).
    """

    def __init__(
        self,
        obs_shape: int | Sequence[int],
        hidden_sizes: Sequence[int] = (256, 256),
        activation: Callable[[], nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")
        self.in_dim = _flat_dim(obs_shape)
        self.out_dim = int(hidden_sizes[-1])
        self.net = mlp(self.in_dim, hidden_sizes, activation)

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs.reshape(obs.shape[0], -1))


class ActorCritic(nn.Module):
    """Gaussian-policy actor-critic for continuous control.

    Args:
        obs_shape: Observation dimension, or shape to be flattened.
        action_dim: Number of continuous action dimensions.
        hidden_sizes: Torso widths.
        activation: Torso activation factory. Tanh is the PPO default; ReLU is
            fine but pairs better with a larger ``num_minibatches``.
        shared_torso: Share one feature trunk between actor and critic. Cheaper,
            but value-loss gradients then perturb the policy features, which is
            usually a net loss on state-based manipulation. Off by default.
        squash: Use a tanh-squashed Gaussian so actions are bounded to
            ``[-1, 1]``. Off by default — see ``SquashedDiagGaussianHead`` for
            why unsquashed-plus-clipping is normally preferable under PPO.
        log_std_init: Initial log sigma. ``-0.5`` is sigma ~= 0.61, a reasonable
            starting exploration scale for normalized action spaces.
        state_dependent_std: Predict ``log_std`` from features instead of
            keeping it a free parameter.
        actor_torso / critic_torso: Inject pre-built torsos (e.g. a CNN). When
            given, ``obs_shape``/``hidden_sizes`` are ignored for that branch.
    """

    def __init__(
        self,
        obs_shape: int | Sequence[int],
        action_dim: int,
        *,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: Callable[[], nn.Module] = nn.Tanh,
        shared_torso: bool = False,
        squash: bool = False,
        log_std_init: float = -0.5,
        log_std_bounds: tuple[float, float] = (-5.0, 2.0),
        state_dependent_std: bool = False,
        actor_torso: nn.Module | None = None,
        critic_torso: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if action_dim < 1:
            raise ValueError("action_dim must be >= 1")

        self.action_dim = int(action_dim)
        self.shared_torso = shared_torso
        self.squash = squash

        self.actor_torso = actor_torso or MLPTorso(obs_shape, hidden_sizes, activation)
        if shared_torso:
            if critic_torso is not None:
                raise ValueError("critic_torso is incompatible with shared_torso=True")
            self.critic_torso = self.actor_torso
        else:
            self.critic_torso = critic_torso or MLPTorso(obs_shape, hidden_sizes, activation)

        head_cls = SquashedDiagGaussianHead if squash else DiagGaussianHead
        self.head = head_cls(
            feature_dim=self.actor_torso.out_dim,
            action_dim=self.action_dim,
            log_std_init=log_std_init,
            log_std_bounds=log_std_bounds,
            state_dependent_std=state_dependent_std,
        )

        self.value_head = nn.Linear(self.critic_torso.out_dim, 1)
        # Unit gain: the value head should start able to represent returns at
        # their natural scale, unlike the deliberately-small policy mean head.
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    # -- internals --------------------------------------------------------- #

    def _policy_params(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        return self.head(self.actor_torso(obs))

    def _value(self, obs: Tensor) -> Tensor:
        return self.value_head(self.critic_torso(obs)).squeeze(-1)

    # -- trainer protocol -------------------------------------------------- #

    def act(self, obs: Tensor, *, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """Sample an action. Returns ``(action, log_prob, value)``.

        ``log_prob`` and ``value`` are ``(batch,)``. ``deterministic=True``
        returns the distribution mean for evaluation rollouts.
        """
        mean, log_std = self._policy_params(obs)
        action = self.head.sample(mean, log_std, deterministic=deterministic)
        log_prob = self.head.log_prob(action, mean, log_std)
        return action, log_prob, self._value(obs)

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Re-score stored actions. Returns ``(log_prob, entropy, value)``, each ``(batch,)``."""
        mean, log_std = self._policy_params(obs)
        log_prob = self.head.log_prob(actions, mean, log_std)
        entropy = self.head.entropy(mean, log_std, actions)
        return log_prob, entropy, self._value(obs)

    # -- conveniences ------------------------------------------------------ #

    def predict_values(self, obs: Tensor) -> Tensor:
        """Critic-only forward, for GAE bootstrapping. Returns ``(batch,)``."""
        return self._value(obs)

    @torch.no_grad()
    def action_std(self) -> Tensor:
        """Current per-dimension sigma. State-dependent heads report the bias term."""
        if self.head.log_std_param is not None:
            log_std = self.head.log_std_param
        else:
            log_std = self.head.log_std_layer.bias
        return torch.clamp(log_std, self.head.log_std_min, self.head.log_std_max).exp()

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Alias for ``act`` so the module is usable as a plain callable."""
        return self.act(obs)
