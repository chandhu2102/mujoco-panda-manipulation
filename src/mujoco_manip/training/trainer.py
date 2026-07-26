"""Train loop: env interaction, updates, eval hooks, checkpointing."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

LOGGER = logging.getLogger(__name__)

__all__ = [
    "PPOConfig",
    "PPOTrainer",
    "RolloutBuffer",
    "TrainerCallback",
    "CallbackList",
    "SuccessRateTracker",
    "extract_successes",
    "extract_success_flags",
    "extract_stage_flags",
    "extract_info_flags",
    "extract_episode_outcomes",
    "extract_episode_grasps",
    "RolloutDiagnostics",
    "VecEnv",
    "ActorCritic",
    "Evaluator",
]


# --------------------------------------------------------------------------- #
# Collaborator interfaces
#
# The env / policy / evaluator modules in this package are still stubs, so the
# trainer depends on these structural protocols rather than concrete classes.
# Anything matching the shapes below can be trained.
# --------------------------------------------------------------------------- #


class VecEnv(Protocol):
    """Gymnasium-style synchronous vector env."""

    num_envs: int

    def reset(self, *, seed: int | None = ...) -> tuple[np.ndarray, dict[str, Any]]:
        ...

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Return ``(obs, rewards, terminations, truncations, infos)``."""
        ...


class ActorCritic(Protocol):
    """Stochastic policy with a value head."""

    def act(self, obs: Tensor, *, deterministic: bool = ...) -> tuple[Tensor, Tensor, Tensor]:
        """Sample an action. Returns ``(action, log_prob, value)``.

        ``log_prob`` and ``value`` are shape ``(batch,)``.
        """
        ...

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Score stored actions. Returns ``(log_prob, entropy, value)``, each ``(batch,)``."""
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        ...


class Evaluator(Protocol):
    """Deterministic evaluation, per ``training/evaluator.py``."""

    def evaluate(self, policy: ActorCritic) -> dict[str, float]:
        """Return metrics including ``success_rate`` in ``[0, 1]``."""
        ...


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class PPOConfig:
    """Hyperparameters for the PPO loop.

    Defaults are the usual continuous-control settings; ``clip_coef`` is the
    requested 0.2.
    """

    # Rollout / batch shape.
    num_steps: int = 512
    """Env steps collected per environment per iteration."""
    num_minibatches: int = 32
    update_epochs: int = 10

    # PPO objective.
    clip_coef: float = 0.2
    """Policy-ratio clip epsilon."""
    clip_vloss: bool = True
    vf_clip_coef: float | None = None
    """Value clip range; falls back to ``clip_coef`` when None."""
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    normalize_advantage: bool = True

    # Returns.
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Optimization.
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    """Stop the epoch loop early once approx-KL exceeds this. None disables."""

    # Minibatch compilation.
    compile_update: bool = False
    """torch.compile the fused (policy forward + PPO loss) minibatch step."""
    compile_mode: str | None = None
    """Passed to torch.compile, e.g. "max-autotune". None uses the default."""
    compile_dynamic: bool | None = False
    """False pins static shapes. Wants batch_size % num_minibatches == 0."""

    # Success tracking.
    success_threshold: float = 0.85
    """Target success rate. Crossing it fires ``on_threshold_reached``."""
    success_window: int = 100
    """Episodes in the rolling train-success window."""

    # Schedule / bookkeeping.
    total_timesteps: int = 10_000_000
    eval_interval: int = 10
    """Iterations between evaluator runs. 0 disables."""
    log_interval: int = 1
    checkpoint_interval: int = 50
    checkpoint_dir: str | None = None
    seed: int | None = None
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def batch_size(self, num_envs: int) -> int:
        return self.num_steps * num_envs

    def minibatch_size(self, num_envs: int) -> int:
        return max(1, self.batch_size(num_envs) // self.num_minibatches)

    def validate(self, num_envs: int) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.num_minibatches < 1:
            raise ValueError("num_minibatches must be >= 1")
        batch = self.batch_size(num_envs)
        if self.num_minibatches > batch:
            raise ValueError(
                f"num_minibatches ({self.num_minibatches}) exceeds batch size ({batch})"
            )
        if batch % self.num_minibatches and self.compile_update:
            LOGGER.warning(
                "batch_size %d is not divisible by num_minibatches %d; the ragged "
                "final minibatch forces a torch.compile recompilation. Pick even "
                "divisors, or set compile_dynamic=None to allow dynamic shapes.",
                batch,
                self.num_minibatches,
            )
        if not 0.0 < self.clip_coef < 1.0:
            raise ValueError("clip_coef must be in (0, 1)")
        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be in [0, 1]")


# --------------------------------------------------------------------------- #
# Rollout storage
# --------------------------------------------------------------------------- #


class RolloutBuffer:
    """Fixed-size on-policy storage with GAE(lambda) advantage computation."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: Sequence[int],
        action_shape: Sequence[int],
        device: torch.device,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs, *obs_shape), device=device)
        self.actions = torch.zeros((num_steps, num_envs, *action_shape), device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        """Terminal flag *entering* the step, i.e. this state began a new episode."""

        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    @property
    def full(self) -> bool:
        return self._step >= self.num_steps

    def add(
        self,
        obs: Tensor,
        action: Tensor,
        logprob: Tensor,
        reward: Tensor,
        value: Tensor,
        done: Tensor,
    ) -> None:
        i = self._step
        if i >= self.num_steps:
            raise RuntimeError("RolloutBuffer is full; call reset() first")
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self._step = i + 1

    @torch.no_grad()
    def compute_returns_and_advantages(
        self,
        last_value: Tensor,
        last_done: Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Backward GAE sweep.

        ``dones[t]`` marks that step ``t`` began a fresh episode, so the
        non-terminal mask for the transition *out of* ``t`` is read from
        ``t + 1`` (and from ``last_done`` for the final step).
        """
        last_gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def flatten(self) -> dict[str, Tensor]:
        """Collapse the ``(step, env)`` axes into a single batch axis."""
        return {
            "obs": self.obs.reshape(-1, *self.obs.shape[2:]),
            "actions": self.actions.reshape(-1, *self.actions.shape[2:]),
            "logprobs": self.logprobs.reshape(-1),
            "values": self.values.reshape(-1),
            "advantages": self.advantages.reshape(-1),
            "returns": self.returns.reshape(-1),
        }


# --------------------------------------------------------------------------- #
# Tracking hooks
# --------------------------------------------------------------------------- #


class TrainerCallback:
    """Hook surface for logging, video capture, early stop, best-model saving.

    Every hook is a no-op by default; override what you need. Returning
    ``False`` from ``on_iteration_end`` requests that training stop.
    """

    def on_training_start(self, trainer: "PPOTrainer") -> None:
        ...

    def on_rollout_end(self, trainer: "PPOTrainer", metrics: dict[str, float]) -> None:
        ...

    def on_update_end(self, trainer: "PPOTrainer", metrics: dict[str, float]) -> None:
        ...

    def on_evaluation(self, trainer: "PPOTrainer", metrics: dict[str, float]) -> None:
        ...

    def on_threshold_reached(
        self, trainer: "PPOTrainer", success_rate: float, source: str
    ) -> None:
        """Fired the first time a success rate crosses ``cfg.success_threshold``.

        ``source`` is ``"train"`` or ``"eval"``.
        """
        ...

    def on_iteration_end(self, trainer: "PPOTrainer", metrics: dict[str, float]) -> bool | None:
        ...

    def on_training_end(self, trainer: "PPOTrainer") -> None:
        ...


class CallbackList(TrainerCallback):
    """Fan each hook out over several callbacks."""

    def __init__(self, callbacks: Sequence[TrainerCallback] = ()) -> None:
        self.callbacks = list(callbacks)

    def append(self, callback: TrainerCallback) -> None:
        self.callbacks.append(callback)

    def on_training_start(self, trainer: "PPOTrainer") -> None:
        for cb in self.callbacks:
            cb.on_training_start(trainer)

    def on_rollout_end(self, trainer: "PPOTrainer", metrics: dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_rollout_end(trainer, metrics)

    def on_update_end(self, trainer: "PPOTrainer", metrics: dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_update_end(trainer, metrics)

    def on_evaluation(self, trainer: "PPOTrainer", metrics: dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_evaluation(trainer, metrics)

    def on_threshold_reached(
        self, trainer: "PPOTrainer", success_rate: float, source: str
    ) -> None:
        for cb in self.callbacks:
            cb.on_threshold_reached(trainer, success_rate, source)

    def on_iteration_end(self, trainer: "PPOTrainer", metrics: dict[str, float]) -> bool:
        keep_going = True
        for cb in self.callbacks:
            # Call every callback even after one votes to stop, so per-iteration
            # loggers still see the final iteration.
            if cb.on_iteration_end(trainer, metrics) is False:
                keep_going = False
        return keep_going

    def on_training_end(self, trainer: "PPOTrainer") -> None:
        for cb in self.callbacks:
            cb.on_training_end(trainer)


@dataclass
class SuccessRateTracker:
    """Rolling success rate over the last ``window`` finished episodes.

    This only measures. Whether a run reaches the threshold is a property of the
    reward, curriculum and hyperparameters, not of this bookkeeping.
    """

    window: int = 100
    episodes: deque = field(default_factory=deque, init=False)
    total_episodes: int = field(default=0, init=False)
    total_successes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.episodes = deque(maxlen=self.window)

    def add(self, successes: Sequence[bool]) -> None:
        for s in successes:
            hit = bool(s)
            self.episodes.append(float(hit))
            self.total_episodes += 1
            self.total_successes += int(hit)

    @property
    def rate(self) -> float | None:
        """None until at least one episode has finished."""
        if not self.episodes:
            return None
        return sum(self.episodes) / len(self.episodes)

    @property
    def is_saturated(self) -> bool:
        """True once the window is full, i.e. the rate spans ``window`` episodes."""
        return len(self.episodes) == self.episodes.maxlen


_SUCCESS_KEYS = ("is_success", "success", "task_success")
_STAGE_KEYS = ("curriculum/stage",)
_GRASP_KEYS = ("had_grasp",)
"""Episode-level: the policy grasped at some point, not "is holding it now"."""

DIAGNOSTIC_PREFIXES = ("reward/", "dist/", "state/")
"""Info keys averaged over the rollout and logged under ``rollout/``.

Prefix-matched rather than enumerated because the reward decomposition differs
per task -- pick_place reports ``state/carrying`` and reach does not -- and a
hardcoded list silently stops covering whichever term a new task adds. Every
key a task puts in its reward breakdown is logged the moment it exists.
"""


def extract_info_flags(
    infos: Any, num_envs: int, keys: Sequence[str], *, cast: Any = float
) -> np.ndarray:
    """Per-env value for the first of ``keys`` present, ``NaN`` where absent.

    Tolerates the shapes the ecosystem actually emits: gymnasium >= 1.0
    dict-of-arrays with a ``_key`` presence mask, the legacy ``final_info``
    array of per-env dicts, and a plain sequence of dicts.
    """
    flags = np.full(num_envs, np.nan, dtype=np.float64)
    if infos is None or len(infos) == 0:
        return flags

    def flag(entry: Any) -> float | None:
        if not isinstance(entry, dict):
            return None
        for key in keys:
            if key in entry:
                return float(cast(entry[key]))
        return None

    if isinstance(infos, dict):
        # Legacy autoreset: terminal dicts tucked under final_info.
        if "final_info" in infos:
            mask = infos.get("_final_info")
            for i, entry in enumerate(infos["final_info"][:num_envs]):
                if mask is not None and not mask[i]:
                    continue
                value = flag(entry)
                if value is not None:
                    flags[i] = value
            return flags

        for key in keys:
            if key not in infos:
                continue
            values = infos[key]
            if np.isscalar(values):
                flags[:] = float(cast(values))
                return flags
            values = np.asarray(values).reshape(-1)
            mask = infos.get(f"_{key}")
            mask = None if mask is None else np.asarray(mask).reshape(-1)
            for i in range(min(num_envs, values.size)):
                if mask is None or mask[i]:
                    flags[i] = float(cast(values[i]))
            return flags
        return flags

    if isinstance(infos, (list, tuple, np.ndarray)):
        for i, entry in enumerate(infos[:num_envs]):
            value = flag(entry)
            if value is not None:
                flags[i] = value
        return flags

    return flags


def extract_success_flags(infos: Any, num_envs: int) -> np.ndarray:
    """Per-env success flag as a float array, ``NaN`` where none was reported."""
    return extract_info_flags(infos, num_envs, _SUCCESS_KEYS, cast=bool)


def extract_stage_flags(infos: Any, num_envs: int) -> np.ndarray:
    """Per-env curriculum start stage, ``NaN`` where the env reports none.

    ``NaN`` rather than 0 matters: a task with no curriculum reports nothing at
    all, which is different from an episode that genuinely started at stage 0.
    Defaulting to 0 would silently file every episode of every non-curriculum
    task under "unassisted", making the distinction meaningless where it is not
    measured.
    """
    return extract_info_flags(infos, num_envs, _STAGE_KEYS, cast=float)


def extract_successes(
    infos: Any, num_envs: int, done: Any | None = None
) -> list[bool]:
    """Success flags for *finished* episodes.

    ``done`` is the ``terminated | truncated`` mask for this step and should
    almost always be supplied. Many envs report ``is_success`` in ``info`` on
    every step, not only at episode end; counting every report would measure the
    fraction of *timesteps* spent in a success state rather than the fraction of
    *episodes* that succeeded, which are very different numbers and would make a
    success-rate threshold meaningless.
    """
    flags = extract_success_flags(infos, num_envs)
    reported = ~np.isnan(flags)
    if done is not None:
        reported &= np.asarray(done, dtype=bool).reshape(-1)[:num_envs]
    return [bool(v) for v in flags[reported]]


def _pair_with_stage(
    infos: Any, num_envs: int, keys: Sequence[str], done: Any | None
) -> list[tuple[bool, int | None]]:
    """``(flag, start_stage)`` per *finished* episode, for the first of ``keys``.

    ``start_stage`` is None when the env reports no curriculum stage. Both are
    read in one pass rather than extracted separately so a flag can never be
    attributed to the wrong env's stage.
    """
    flags = extract_info_flags(infos, num_envs, keys, cast=bool)
    stages = extract_stage_flags(infos, num_envs)
    reported = ~np.isnan(flags)
    if done is not None:
        reported &= np.asarray(done, dtype=bool).reshape(-1)[:num_envs]
    out: list[tuple[bool, int | None]] = []
    for i in np.flatnonzero(reported):
        stage = None if np.isnan(stages[i]) else int(round(float(stages[i])))
        out.append((bool(flags[i]), stage))
    return out


def extract_episode_outcomes(
    infos: Any, num_envs: int, done: Any | None = None
) -> list[tuple[bool, int | None]]:
    """``(success, start_stage)`` per *finished* episode."""
    return _pair_with_stage(infos, num_envs, _SUCCESS_KEYS, done)


def extract_episode_grasps(
    infos: Any, num_envs: int, done: Any | None = None
) -> list[tuple[bool, int | None]]:
    """``(had_grasp, start_stage)`` per *finished* episode.

    Split by stage for the same reason success is: a seeded episode may start
    already holding the object, so its grasp flag says nothing about what the
    policy can do. The stage-0 rate is the one that answers "has exploration
    found a grasp yet", which is the question a stalled pick-and-place run
    actually needs answered -- and the question ``rollout/success_rate`` cannot
    answer, because it reads 0.0 both before a grasp is ever found and after
    grasping works but placement is still imprecise.
    """
    return _pair_with_stage(infos, num_envs, _GRASP_KEYS, done)


class RolloutDiagnostics:
    """Per-step means of the env's reward breakdown over one rollout.

    The reward terms are already in every ``info`` dict; nothing was reading
    them. Averaging them per rollout turns "return is 117 and flat" into a
    statement about *which stage* the return is coming from -- a run collecting
    its whole return from ``reward/reach`` with ``reward/grasp`` pinned at zero
    has a discovery problem, and one collecting grasp and lift but not place has
    a transport problem. Those need opposite fixes and are indistinguishable
    from the scalar return.

    Keys are discovered from the infos rather than declared, so the accumulator
    covers whatever terms the task reports. ``NaN`` entries (an env that did not
    report a key on this step) are skipped per key rather than poisoning the
    mean, so a partially-reporting vector env still yields usable numbers.
    """

    def __init__(self, prefixes: Sequence[str] = DIAGNOSTIC_PREFIXES) -> None:
        self.prefixes = tuple(prefixes)
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()

    def _keys(self, infos: Any) -> list[str]:
        if isinstance(infos, dict):
            # The leading-underscore entries are gymnasium's presence masks, not
            # data; extract_info_flags consults them itself.
            candidates: Any = (k for k in infos if not str(k).startswith("_"))
        elif isinstance(infos, (list, tuple, np.ndarray)):
            candidates = {k for e in infos if isinstance(e, dict) for k in e}
        else:
            return []
        return [k for k in candidates if str(k).startswith(self.prefixes)]

    def update(self, infos: Any, num_envs: int) -> None:
        for key in self._keys(infos):
            values = extract_info_flags(infos, num_envs, (key,), cast=float)
            finite = values[~np.isnan(values)]
            if finite.size:
                self._sums[key] = self._sums.get(key, 0.0) + float(finite.sum())
                self._counts[key] = self._counts.get(key, 0) + int(finite.size)

    def metrics(self, prefix: str = "rollout/") -> dict[str, float]:
        return {
            f"{prefix}{key}": self._sums[key] / count
            for key, count in self._counts.items()
            if count
        }


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class PPOTrainer:
    """Clipped-surrogate PPO over a vectorized env.

    One iteration is: collect ``num_steps`` transitions per env, compute GAE,
    then run ``update_epochs`` passes of minibatch SGD over the flattened batch.
    """

    def __init__(
        self,
        env: VecEnv,
        policy: ActorCritic,
        config: PPOConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        callbacks: Sequence[TrainerCallback] = (),
        evaluator: Evaluator | None = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.cfg = config or PPOConfig()
        self.evaluator = evaluator
        self.callbacks = CallbackList(callbacks)

        self.num_envs = int(env.num_envs)
        self.cfg.validate(self.num_envs)
        self.device = self.cfg.resolve_device()

        if isinstance(policy, nn.Module):
            policy.to(self.device)

        self.optimizer = optimizer or torch.optim.Adam(
            policy.parameters(), lr=self.cfg.learning_rate, eps=1e-5
        )

        self.global_step = 0
        self.iteration = 0
        self.num_iterations = max(
            1, self.cfg.total_timesteps // self.cfg.batch_size(self.num_envs)
        )

        self.success_tracker = SuccessRateTracker(window=self.cfg.success_window)
        # Success split by the curriculum stage each episode *started* from.
        # rollout/success_rate mixes them, which makes it unusable as a progress
        # measure the moment a curriculum is on: a stage-3 episode starts 7.5 cm
        # from the goal already holding the object, so its success says nothing
        # about whether the policy can do the task. unassisted_success_tracker is
        # the one to read, and the one the threshold gate uses.
        self.unassisted_success_tracker = SuccessRateTracker(
            window=self.cfg.success_window
        )
        self.stage_success_trackers: dict[int, SuccessRateTracker] = {}
        # Grasp is the gate every later reward stage sits behind, so it is the
        # first thing to check when success stays at zero. Tracked on the same
        # window as success, and split the same way: the unassisted rate is the
        # one that says whether the *policy* can grasp, rather than whether the
        # curriculum handed it a grasp.
        self.grasp_tracker = SuccessRateTracker(window=self.cfg.success_window)
        self.unassisted_grasp_tracker = SuccessRateTracker(
            window=self.cfg.success_window
        )
        self.diagnostics = RolloutDiagnostics()
        self.best_eval_success: float = -1.0
        self._best_eval_score: tuple[float, float] = (-1.0, float("-inf"))
        self.threshold_reached = False

        self._episode_returns: deque = deque(maxlen=self.cfg.success_window)
        self._episode_lengths: deque = deque(maxlen=self.cfg.success_window)
        self._running_return = np.zeros(self.num_envs, dtype=np.float64)
        self._running_length = np.zeros(self.num_envs, dtype=np.int64)

        self._buffer: RolloutBuffer | None = None
        self._next_obs: Tensor | None = None
        self._next_done: Tensor | None = None

        self._compiled_loss_fn: Callable[..., tuple[Tensor, dict[str, Tensor]]] | None = None
        if self.cfg.compile_update:
            self._compiled_loss_fn = self._build_compiled_loss()

    # -- setup ------------------------------------------------------------- #

    def _build_compiled_loss(self) -> Callable[..., tuple[Tensor, dict[str, Tensor]]] | None:
        kwargs: dict[str, Any] = {"dynamic": self.cfg.compile_dynamic}
        if self.cfg.compile_mode:
            kwargs["mode"] = self.cfg.compile_mode
        try:
            compiled = torch.compile(self._minibatch_loss, **kwargs)
        except Exception as exc:  # pragma: no cover - backend availability
            LOGGER.warning("torch.compile unavailable (%s); using eager updates", exc)
            return None
        LOGGER.info("Minibatch update compiled (mode=%s)", self.cfg.compile_mode or "default")
        return compiled

    def _lazy_init_buffer(self, obs: Tensor, action: Tensor) -> RolloutBuffer:
        if self._buffer is None:
            self._buffer = RolloutBuffer(
                num_steps=self.cfg.num_steps,
                num_envs=self.num_envs,
                obs_shape=tuple(obs.shape[1:]),
                action_shape=tuple(action.shape[1:]),
                device=self.device,
            )
        return self._buffer

    def _to_tensor(self, array: Any) -> Tensor:
        return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=self.device)

    # -- rollout ----------------------------------------------------------- #

    @torch.no_grad()
    def collect_rollout(self) -> dict[str, float]:
        """Step the env ``num_steps`` times per env, filling the buffer."""
        cfg = self.cfg
        if self._next_obs is None:
            obs, _ = self.env.reset(seed=cfg.seed)
            self._next_obs = self._to_tensor(obs)
            self._next_done = torch.zeros(self.num_envs, device=self.device)
            # Nothing is harvested from the reset info: no episode has finished
            # yet, so any is_success it reports is about the initial state.

        obs_t = self._next_obs
        done_t = self._next_done
        assert obs_t is not None and done_t is not None

        started = time.perf_counter()
        episodes_this_rollout = 0
        buffer: RolloutBuffer | None = None
        # Per-rollout, unlike the success trackers' episode window: these are
        # step averages and mixing rollouts would smear the very transition
        # (grasp appears, place starts paying) they exist to make visible.
        self.diagnostics.reset()

        for _ in range(cfg.num_steps):
            action, logprob, value = self.policy.act(obs_t)

            next_obs, reward, terminated, truncated, infos = self.env.step(
                action.detach().cpu().numpy()
            )
            terminated = np.asarray(terminated, dtype=bool).reshape(-1)
            truncated = np.asarray(truncated, dtype=bool).reshape(-1)
            reward_np = np.asarray(reward, dtype=np.float64).reshape(-1)
            episode_over = terminated | truncated

            if buffer is None:
                buffer = self._lazy_init_buffer(obs_t, action)
                buffer.reset()
            buffer.add(
                obs=obs_t,
                action=action,
                logprob=logprob,
                reward=self._to_tensor(reward_np),
                value=value,
                done=done_t,
            )

            self._running_return += reward_np
            self._running_length += 1
            if episode_over.any():
                for i in np.flatnonzero(episode_over):
                    self._episode_returns.append(float(self._running_return[i]))
                    self._episode_lengths.append(int(self._running_length[i]))
                    self._running_return[i] = 0.0
                    self._running_length[i] = 0
                episodes_this_rollout += int(episode_over.sum())

            # Gate on episode_over so the rate is per-episode, not per-timestep.
            for success, stage in extract_episode_outcomes(
                infos, self.num_envs, done=episode_over
            ):
                self.success_tracker.add((success,))
                if stage is None:
                    # No curriculum on this task: every episode is unassisted by
                    # definition, so the two metrics coincide.
                    self.unassisted_success_tracker.add((success,))
                    continue
                self.stage_success_trackers.setdefault(
                    stage, SuccessRateTracker(window=self.cfg.success_window)
                ).add((success,))
                if stage == 0:
                    self.unassisted_success_tracker.add((success,))

            for grasped, stage in extract_episode_grasps(
                infos, self.num_envs, done=episode_over
            ):
                self.grasp_tracker.add((grasped,))
                if stage is None or stage == 0:
                    self.unassisted_grasp_tracker.add((grasped,))

            self.diagnostics.update(infos, self.num_envs)

            obs_t = self._to_tensor(next_obs)
            done_t = self._to_tensor(episode_over.astype(np.float32))
            self.global_step += self.num_envs

        assert buffer is not None
        self._next_obs = obs_t
        self._next_done = done_t

        _, _, last_value = self.policy.act(obs_t)
        buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_done=done_t,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        elapsed = time.perf_counter() - started
        metrics = {
            "rollout/sps": cfg.batch_size(self.num_envs) / max(elapsed, 1e-9),
            "rollout/episodes": float(episodes_this_rollout),
            "rollout/mean_reward": float(buffer.rewards.mean()),
        }
        if self._episode_returns:
            metrics["rollout/episode_return"] = float(np.mean(self._episode_returns))
            metrics["rollout/episode_length"] = float(np.mean(self._episode_lengths))
        rate = self.success_tracker.rate
        if rate is not None:
            metrics["rollout/success_rate"] = rate
        # The headline number: performance on the real task, ignoring every
        # episode the curriculum handed a head start.
        unassisted = self.unassisted_success_tracker.rate
        if unassisted is not None:
            metrics["rollout/true_unassisted_success_rate"] = unassisted
            metrics["rollout/unassisted_episodes"] = float(
                self.unassisted_success_tracker.total_episodes
            )
        for stage, tracker in sorted(self.stage_success_trackers.items()):
            stage_rate = tracker.rate
            if stage_rate is not None:
                metrics[f"rollout/success_stage_{stage}"] = stage_rate
                metrics[f"rollout/episodes_stage_{stage}"] = float(
                    tracker.total_episodes
                )
        # Grasp rates alongside success, so a run that is failing at the grasp
        # and a run that is failing at the placement are distinguishable from
        # the logs alone.
        grasp_rate = self.grasp_tracker.rate
        if grasp_rate is not None:
            metrics["rollout/grasp_rate"] = grasp_rate
        unassisted_grasp = self.unassisted_grasp_tracker.rate
        if unassisted_grasp is not None:
            metrics["rollout/true_unassisted_grasp_rate"] = unassisted_grasp
        metrics.update(self.diagnostics.metrics())
        self.callbacks.on_rollout_end(self, metrics)
        return metrics

    # -- update ------------------------------------------------------------ #

    def _minibatch_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        old_logprobs: Tensor,
        old_values: Tensor,
        advantages: Tensor,
        returns: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Fused policy forward + clipped PPO objective.

        Deliberately free of Python-side branching on tensor *values* so it
        stays torch.compile-friendly (config branches are static per run).
        """
        cfg = self.cfg
        new_logprobs, entropy, new_values = self.policy.evaluate_actions(obs, actions)

        log_ratio = new_logprobs - old_logprobs
        ratio = log_ratio.exp()

        if cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Clipped surrogate: pessimistic bound on the policy improvement.
        pg_loss = torch.max(
            -advantages * ratio,
            -advantages * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
        ).mean()

        if cfg.clip_vloss:
            v_clip = cfg.vf_clip_coef if cfg.vf_clip_coef is not None else cfg.clip_coef
            v_clipped = old_values + torch.clamp(new_values - old_values, -v_clip, v_clip)
            v_loss = 0.5 * torch.max(
                (new_values - returns) ** 2, (v_clipped - returns) ** 2
            ).mean()
        else:
            v_loss = 0.5 * ((new_values - returns) ** 2).mean()

        entropy_mean = entropy.mean()
        loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy_mean

        with torch.no_grad():
            stats = {
                "policy_loss": pg_loss.detach(),
                "value_loss": v_loss.detach(),
                "entropy": entropy_mean.detach(),
                # Schulman's low-variance approx-KL estimator.
                "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
                "clip_fraction": ((ratio - 1.0).abs() > cfg.clip_coef).float().mean(),
            }
        return loss, stats

    def _run_loss(self, *args: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if self._compiled_loss_fn is not None:
            try:
                return self._compiled_loss_fn(*args)
            except Exception as exc:  # pragma: no cover - backend fallback
                LOGGER.warning(
                    "Compiled minibatch step failed (%s); using eager for the rest "
                    "of the run",
                    exc,
                )
                self._compiled_loss_fn = None
        return self._minibatch_loss(*args)

    def update(self) -> dict[str, float]:
        """Run ``update_epochs`` of minibatch SGD over the current rollout."""
        cfg = self.cfg
        buffer = self._buffer
        if buffer is None:
            raise RuntimeError("collect_rollout() must run before update()")

        batch = buffer.flatten()
        batch_size = batch["obs"].shape[0]
        minibatch_size = cfg.minibatch_size(self.num_envs)
        indices = np.arange(batch_size)

        sums: dict[str, float] = {}
        n_minibatches = 0
        epochs_ran = 0
        stop_early = False
        last_kl = 0.0

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(indices)
            epochs_ran = epoch + 1
            for start in range(0, batch_size, minibatch_size):
                mb = torch.as_tensor(
                    indices[start : start + minibatch_size], device=self.device
                )

                loss, stats = self._run_loss(
                    batch["obs"][mb],
                    batch["actions"][mb],
                    batch["logprobs"][mb],
                    batch["values"][mb],
                    batch["advantages"][mb],
                    batch["returns"][mb],
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                n_minibatches += 1
                sums["train/loss"] = sums.get("train/loss", 0.0) + float(loss.detach())
                sums["train/grad_norm"] = sums.get("train/grad_norm", 0.0) + float(grad_norm)
                for key, value in stats.items():
                    sums[f"train/{key}"] = sums.get(f"train/{key}", 0.0) + float(value)

                last_kl = float(stats["approx_kl"])

            if cfg.target_kl is not None and last_kl > cfg.target_kl:
                LOGGER.debug(
                    "Early stop after epoch %d/%d: approx_kl %.4f > %.4f",
                    epochs_ran,
                    cfg.update_epochs,
                    last_kl,
                    cfg.target_kl,
                )
                stop_early = True
                break

        metrics = {k: v / max(n_minibatches, 1) for k, v in sums.items()}
        metrics["train/explained_variance"] = self._explained_variance(
            batch["values"], batch["returns"]
        )
        metrics["train/epochs_ran"] = float(epochs_ran)
        metrics["train/early_stopped"] = float(stop_early)
        metrics["train/learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        self.callbacks.on_update_end(self, metrics)
        return metrics

    @staticmethod
    def _explained_variance(values: Tensor, returns: Tensor) -> float:
        var_returns = float(returns.var())
        if var_returns == 0.0:
            return float("nan")
        return float(1.0 - returns.sub(values).var() / var_returns)

    # -- schedule / eval / checkpoints ------------------------------------- #

    def _anneal_lr(self) -> None:
        if not self.cfg.anneal_lr:
            return
        frac = max(0.0, 1.0 - (self.iteration - 1) / self.num_iterations)
        lr = frac * self.cfg.learning_rate
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def evaluate(self) -> dict[str, float]:
        """Run the evaluator, tracking best-so-far success."""
        if self.evaluator is None:
            return {}
        is_module = isinstance(self.policy, nn.Module)
        was_training = bool(getattr(self.policy, "training", False))
        if is_module:
            self.policy.eval()
        try:
            with torch.no_grad():
                raw = self.evaluator.evaluate(self.policy)
        finally:
            if is_module and was_training:
                self.policy.train()

        metrics = {
            (k if "/" in k else f"eval/{k}"): float(v)
            for k, v in (raw or {}).items()
            if isinstance(v, (bool, int, float, np.floating, np.integer))
        }
        self.callbacks.on_evaluation(self, metrics)

        success = metrics.get("eval/success_rate")
        if success is not None:
            # Rank on (success_rate, return_mean), not success_rate alone. Success
            # rate saturates at 1.0, so a strict > on it freezes best.pt at the
            # first checkpoint to touch the ceiling while the policy keeps
            # refining. Measured on the 1.5M reach run: the step-246k best.pt
            # scored mean final distance 0.030 m against 0.020 m for the step-1.5M
            # policy, both at 100% success.
            score = (success, metrics.get("eval/return_mean", float("-inf")))
            if score > self._best_eval_score:
                self._best_eval_score = score
                self.best_eval_success = success
                self.save_checkpoint("best.pt")
            self._check_threshold(success, source="eval")
        return metrics

    def _check_threshold(self, success_rate: float, source: str) -> None:
        if self.threshold_reached or success_rate < self.cfg.success_threshold:
            return
        self.threshold_reached = True
        LOGGER.info(
            "Success threshold %.1f%% crossed at step %d (%s: %.1f%%)",
            self.cfg.success_threshold * 100,
            self.global_step,
            source,
            success_rate * 100,
        )
        self.callbacks.on_threshold_reached(self, success_rate, source)

    def save_checkpoint(self, filename: str) -> Path | None:
        """Write policy + optimizer state. No-op without ``cfg.checkpoint_dir``."""
        if not self.cfg.checkpoint_dir:
            return None
        directory = Path(self.cfg.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        payload = {
            "policy": self.policy.state_dict() if isinstance(self.policy, nn.Module) else None,
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "iteration": self.iteration,
            "best_eval_success": self.best_eval_success,
            "config": vars(self.cfg),
        }
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str | Path, *, load_optimizer: bool = True) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("policy") is not None and isinstance(self.policy, nn.Module):
            self.policy.load_state_dict(payload["policy"])
        if load_optimizer and payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload.get("global_step", 0))
        self.iteration = int(payload.get("iteration", 0))
        self.best_eval_success = float(payload.get("best_eval_success", -1.0))
        # Reset the return tie-breaker: the resumed run has not measured one yet,
        # so the first evaluation after resume is free to claim best.pt.
        self._best_eval_score = (self.best_eval_success, float("-inf"))

    # -- main loop --------------------------------------------------------- #

    def train(self, total_timesteps: int | None = None) -> dict[str, float]:
        """Run the loop until ``total_timesteps`` or a callback stops it."""
        cfg = self.cfg
        if total_timesteps is not None:
            cfg.total_timesteps = total_timesteps
        self.num_iterations = max(1, cfg.total_timesteps // cfg.batch_size(self.num_envs))

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
        if isinstance(self.policy, nn.Module):
            self.policy.train()

        LOGGER.info(
            "PPO: %d iterations x %d steps x %d envs = %d timesteps on %s",
            self.num_iterations,
            cfg.num_steps,
            self.num_envs,
            self.num_iterations * cfg.batch_size(self.num_envs),
            self.device,
        )
        self.callbacks.on_training_start(self)

        metrics: dict[str, float] = {}
        try:
            while self.global_step < cfg.total_timesteps:
                self.iteration += 1
                self._anneal_lr()

                metrics = {"iteration": float(self.iteration)}
                metrics.update(self.collect_rollout())
                metrics.update(self.update())

                # Gate on the unassisted rate, not the mixed one. Curriculum
                # seeding inflates rollout/success_rate by construction -- at
                # curriculum_level=3 roughly a quarter of episodes start already
                # holding the object near the goal -- so thresholding the mixed
                # rate would declare the task solved off the back of the head
                # starts. Falls back to the mixed tracker only when nothing
                # reports a stage, i.e. tasks with no curriculum, where the two
                # are the same measurement anyway.
                gate = self.unassisted_success_tracker
                if not gate.total_episodes:
                    gate = self.success_tracker
                train_rate = gate.rate
                if train_rate is not None and gate.is_saturated:
                    self._check_threshold(train_rate, source="train")

                if cfg.eval_interval and self.iteration % cfg.eval_interval == 0:
                    metrics.update(self.evaluate())

                if cfg.checkpoint_interval and self.iteration % cfg.checkpoint_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}.pt")

                if cfg.log_interval and self.iteration % cfg.log_interval == 0:
                    self._log(metrics)

                if self.callbacks.on_iteration_end(self, metrics) is False:
                    LOGGER.info("Training stopped by callback at iteration %d", self.iteration)
                    break
        except KeyboardInterrupt:
            LOGGER.warning("Interrupted at step %d; saving checkpoint", self.global_step)
            self.save_checkpoint("interrupted.pt")
            raise
        finally:
            self.callbacks.on_training_end(self)

        self.save_checkpoint("final.pt")
        return metrics

    # Metric key -> short log label. Labels are explicit rather than derived from
    # the key suffix, since rollout/ and eval/ both carry a success_rate.
    _LOG_FIELDS = (
        ("rollout/success_rate", "success"),
        # Next to the mixed rate rather than replacing it, so the gap between
        # them is visible on the terminal line: that gap *is* the curriculum's
        # contribution, and watching it close is how you know seeding can be
        # annealed out.
        ("rollout/true_unassisted_success_rate", "true_success"),
        # On the console line because it is the leading indicator: grasp rate
        # moves off zero long before success does, and a run where it never
        # does is a run to kill early rather than at 3M steps.
        ("rollout/true_unassisted_grasp_rate", "grasp"),
        ("rollout/episode_return", "return"),
        ("rollout/sps", "sps"),
        ("train/policy_loss", "pg_loss"),
        ("train/value_loss", "v_loss"),
        ("train/approx_kl", "kl"),
        ("train/clip_fraction", "clipfrac"),
        ("eval/success_rate", "eval_success"),
    )

    def _log(self, metrics: dict[str, float]) -> None:
        parts = [f"iter={self.iteration}", f"step={self.global_step}"]
        for key, label in self._LOG_FIELDS:
            if key in metrics:
                parts.append(f"{label}={metrics[key]:.4g}")
        LOGGER.info(" | ".join(parts))
