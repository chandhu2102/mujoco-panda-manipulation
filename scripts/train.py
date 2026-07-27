#!/usr/bin/env python3
"""Master training launcher: config -> envs -> policy -> PPO -> checkpoints.

    python scripts/train.py --config configs/train/reach.yaml
    python scripts/train.py --config configs/train/reach.yaml --set ppo.learning_rate=1e-4
    python scripts/train.py --config configs/train/reach.yaml --smoke   # 8k steps
    python scripts/train.py --config configs/train/reach.yaml --resume outputs/reach_1p5m

Everything needed to reproduce a run lands in ``<output_dir>/<run_name>/``:
resolved config, TensorBoard events, metrics JSONL, checkpoints, and the
observation-normalizer statistics.

``--smoke`` redirects that whole directory to ``<run_name>.smoke/``. A smoke test
borrows the config -- and therefore the ``run_name`` -- of the real run it is
validating, so without the redirect it writes an 8k-step policy over that run's
``best.pt``/``final.pt`` and 8k-step statistics over its ``obs_normalizer.npz``,
neither of which is recoverable from the run directory. ``--smoke --resume`` still
reads the original directory's checkpoint; only writes are redirected.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

# Allow running as a plain script from a clone, with no install step.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mujoco_manip.algos.common.networks import ActorCritic  # noqa: E402
from mujoco_manip.algos.common.normalizers import (  # noqa: E402
    NormalizeObservation,
    NormalizeReward,
    RunningMeanStd,
)
from mujoco_manip.envs.manipulation_env import ManipulationEnv  # noqa: E402
from mujoco_manip.envs.tasks import TASK_REGISTRY, make_reward_config  # noqa: E402
from mujoco_manip.envs.wrappers.task_space import TaskSpaceWrapper  # noqa: E402
from mujoco_manip.training.callbacks import (  # noqa: E402
    ActionStdCeiling,
    CurriculumSchedule,
    EarlyStopOnThreshold,
    EntropyPenaltySchedule,
    LinearSchedule,
    MetricHistory,
    NormalizerCheckpoint,
    ProgressCallback,
    TensorBoardCallback,
)
from mujoco_manip.training.bc import DAPGTrainer, DemoBuffer  # noqa: E402
from mujoco_manip.training.evaluator import EvaluationConfig, Evaluator  # noqa: E402
from mujoco_manip.training.trainer import PPOConfig, PPOTrainer  # noqa: E402

LOGGER = logging.getLogger("train")

# Sourced from the package registry rather than redeclared, so a task added in
# mujoco_manip.envs.tasks is visible here and to scripts/enjoy.py (which imports
# this name) without a second edit.
TASKS: dict[str, type[ManipulationEnv]] = TASK_REGISTRY
ACTIVATIONS = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU, "elu": torch.nn.ELU}

SMOKE_SUFFIX = ".smoke"
"""Appended to the run directory under ``--smoke``; see ``resolve_run_dir``."""

_TASK_SPACE_KEYS: frozenset[str] = frozenset(
    {"enabled"} | set(TaskSpaceWrapper.__init__.__annotations__) - {"env", "return"}
)
"""Accepted keys under ``env.task_space``, read off the wrapper's own signature.

Validated for the same reason ``build_ppo_config`` rejects unknown PPO keys: a
misspelled ``max_delta_pos`` would otherwise train a full run at the default
delta while the resolved config claims a different one.
"""


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def load_config(path: Path, overrides: list[str]) -> dict[str, Any]:
    """Load YAML and apply ``dotted.key=value`` overrides."""
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    for override in overrides:
        if "=" not in override:
            raise SystemExit(f"--set expects key=value, got {override!r}")
        key, raw = override.split("=", 1)
        node = config
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise SystemExit(f"--set {key}: no config section {part!r}")
            node = node[part]
        if parts[-1] not in node:
            raise SystemExit(f"--set {key}: unknown key (typo?)")
        # Route through the YAML parser so 1e-4/true/[1,2] behave as written.
        node[parts[-1]] = yaml.safe_load(raw)
    return config


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_run_dir(base: Path, *, smoke: bool, repo_root: Path = _REPO_ROOT) -> Path:
    """Absolute directory every artifact of this run is written under.

    ``smoke`` appends ``SMOKE_SUFFIX``, and that is the entire isolation
    mechanism: the checkpoint directory, ``obs_normalizer.npz``, ``tb/``,
    ``metrics.jsonl`` and ``config.resolved.json`` are all derived from this one
    path, so redirecting it here redirects every one of them without each call
    site needing to know about ``--smoke``.

    Idempotent -- a directory already ending in ``.smoke`` is not suffixed twice,
    so ``--run-name foo.smoke --smoke`` stays ``foo.smoke``.
    """
    path = base if base.is_absolute() else repo_root / base
    if smoke and not path.name.endswith(SMOKE_SUFFIX):
        path = path.parent / f"{path.name}{SMOKE_SUFFIX}"
    return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def make_env_fn(env_cfg: dict[str, Any], seed: int, index: int) -> Callable[[], Any]:
    """Build a thunk for one env instance, seeded distinctly."""
    task = env_cfg.get("task", "reach")
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r}; have {sorted(TASKS)}")
    cls = TASKS[task]
    kwargs: dict[str, Any] = {
        "control_mode": env_cfg.get("control_mode", "torque"),
        "control_dt": float(env_cfg.get("control_dt", 0.04)),
        "gravity_compensation": bool(env_cfg.get("gravity_compensation", True)),
    }
    # Horizon, reset pose and the termination rule are only forwarded when the
    # config actually sets them, so an omitted key falls through to the task
    # class's own default. Hardcoding fallbacks here would silently impose
    # reach's 150-step horizon and `home` pose on every other task -- and, worse,
    # default terminate_on_success to True, which the ManipulationEnv docstring
    # records driving eval success from 45% to 0% under a dense reward.
    for key, cast in (
        ("max_episode_steps", int),
        ("reset_pose", str),
        ("terminate_on_success", bool),
        ("curriculum_level", int),
        # Omitted-key-falls-through applies here too, and pointedly: the env's
        # default is None (no limit), which is the behaviour every config written
        # before this key existed was trained under.
        ("gripper_rate_limit", float),
    ):
        if key in env_cfg:
            kwargs[key] = cast(env_cfg[key])

    # Reward weights, same omitted-key-falls-through rule as above: no `reward`
    # block means the task class picks its own dataclass. Nested under `env`
    # rather than given a top-level section so it lands in
    # config.resolved.json's "env" automatically -- which is also the block
    # scripts/enjoy.py rebuilds from, so a replay uses the weights that trained.
    if env_cfg.get("reward"):
        kwargs["reward_config"] = make_reward_config(task, env_cfg["reward"])

    # Task-space control is a wrapper, not an env flag, so it has to be applied
    # here rather than inside the task class -- and applied *here* specifically,
    # because make_env_fn is the single construction path shared by the training
    # vector env, the evaluator's env and scripts/enjoy.py's replay env. A wrapper
    # added at any one of those call sites would silently give the other two a
    # different action space than the policy was trained on.
    task_space_cfg = env_cfg.get("task_space") or {}
    if task_space_cfg.get("enabled", False):
        unknown = sorted(set(task_space_cfg) - _TASK_SPACE_KEYS)
        if unknown:
            raise SystemExit(
                f"unknown env.task_space keys: {unknown}; have "
                f"{sorted(_TASK_SPACE_KEYS - {'enabled'})}"
            )
        wrapper_kwargs = {k: v for k, v in task_space_cfg.items() if k != "enabled"}
    else:
        wrapper_kwargs = None

    def thunk() -> Any:
        # Distinct seed per worker, else all envs randomize identically and the
        # effective batch is one trajectory wide.
        env = cls(seed=seed + 1000 * index, **kwargs)
        if wrapper_kwargs is not None:
            return TaskSpaceWrapper(env, **wrapper_kwargs)
        return env

    return thunk


def build_vector_env(env_cfg: dict[str, Any], seed: int) -> Any:
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

    num_envs = int(env_cfg.get("num_envs", 8))
    thunks = [make_env_fn(env_cfg, seed, i) for i in range(num_envs)]
    if env_cfg.get("async_envs", False):
        # MuJoCo releases the GIL during mj_step, so Sync is already parallel-ish
        # and cheaper; Async only wins for heavier per-env work such as rendering.
        return AsyncVectorEnv(thunks)
    return SyncVectorEnv(thunks)


def build_policy(
    policy_cfg: dict[str, Any], obs_dim: int, action_dim: int
) -> ActorCritic:
    activation = policy_cfg.get("activation", "tanh")
    if activation not in ACTIVATIONS:
        raise SystemExit(f"unknown activation {activation!r}; have {sorted(ACTIVATIONS)}")
    return ActorCritic(
        obs_shape=obs_dim,
        action_dim=action_dim,
        hidden_sizes=tuple(policy_cfg.get("hidden_sizes", (256, 256))),
        activation=ACTIVATIONS[activation],
        shared_torso=bool(policy_cfg.get("shared_torso", False)),
        squash=bool(policy_cfg.get("squash", False)),
        log_std_init=float(policy_cfg.get("log_std_init", -0.5)),
        state_dependent_std=bool(policy_cfg.get("state_dependent_std", False)),
    )


_BC_KEYS: frozenset[str] = frozenset(
    {"enabled", "demos", "coef", "decay", "batch_size",
     # Consumed by scripts/pretrain_bc.py rather than here, but accepted so one
     # `bc` block can configure both stages of the pipeline.
     "epochs", "learning_rate", "value_coef", "final_sigma"}
)

_SCHEDULE_KEYS = frozenset(LinearSchedule.__dataclass_fields__)
_ANNEAL_KEYS = frozenset({"enabled", "rebase_on_resume", "ent_coef", "penalties"})


def build_schedule(label: str, spec: Any) -> LinearSchedule:
    """One ``{start, end, ramp_start?, ramp_end?}`` block -> a ``LinearSchedule``.

    Validated key by key for the same reason ``build_ppo_config`` rejects unknown
    PPO keys: a misspelled ``ramp`` silently produces the un-ramped schedule, and
    a schedule that anneals on the wrong window is indistinguishable in the logs
    from one that does not anneal at all until the run is over.
    """
    if not isinstance(spec, dict):
        raise SystemExit(f"{label}: expected a mapping of {sorted(_SCHEDULE_KEYS)}")
    unknown = sorted(set(spec) - _SCHEDULE_KEYS)
    if unknown:
        raise SystemExit(f"{label}: unknown key(s) {unknown}; have {sorted(_SCHEDULE_KEYS)}")
    for required in ("start", "end"):
        if required not in spec:
            raise SystemExit(f"{label}: missing required key {required!r}")
    try:
        return LinearSchedule(**{k: float(v) for k, v in spec.items()})
    except ValueError as exc:
        raise SystemExit(f"{label}: {exc}") from exc


def build_anneal_callback(
    anneal_cfg: dict[str, Any], *, log_interval: int
) -> EntropyPenaltySchedule | None:
    """The entropy/penalty annealer described by the config's ``anneal`` block.

    Returns None when the block is absent or disabled, which is what keeps a
    config that predates this feature training at its fixed coefficients.
    """
    if not anneal_cfg or not anneal_cfg.get("enabled", False):
        return None
    unknown = sorted(set(anneal_cfg) - _ANNEAL_KEYS)
    if unknown:
        raise SystemExit(f"unknown anneal config keys: {unknown}")

    entropy = None
    if anneal_cfg.get("ent_coef"):
        entropy = build_schedule("anneal.ent_coef", anneal_cfg["ent_coef"])

    penalties: dict[str, LinearSchedule] = {}
    for name, spec in (anneal_cfg.get("penalties") or {}).items():
        penalties[str(name)] = build_schedule(f"anneal.penalties.{name}", spec)

    if entropy is None and not penalties:
        raise SystemExit(
            "anneal.enabled is true but neither ent_coef nor penalties is set"
        )
    return EntropyPenaltySchedule(
        entropy=entropy,
        penalties=penalties,
        rebase_on_resume=bool(anneal_cfg.get("rebase_on_resume", False)),
        log_interval=log_interval,
    )


def reset_action_std(policy: ActorCritic, sigma: float) -> None:
    """Re-initialize the policy's ``log_std`` to ``sigma``, in place.

    For resuming a run whose sigma ran away: the entropy bonus is the only
    consistent gradient on ``log_std`` while advantages are uninformative, so a
    stalled run arrives at its checkpoint with a much wider policy than it
    started with, and inherits that width across the resume. Nothing else in the
    checkpoint needs discarding -- the torso and critic are worth keeping -- so
    this rewinds one parameter rather than the whole policy.
    """
    if sigma <= 0.0:
        raise SystemExit(f"--reset-action-std expects a positive sigma, got {sigma}")
    head = policy.head
    param = head.log_std_param
    if param is None:
        param = getattr(head, "log_std_layer", None)
        param = None if param is None else param.bias
    if param is None:
        raise SystemExit("--reset-action-std: policy head exposes no log_std parameter")
    with torch.no_grad():
        before = param.exp().tolist()
        param.fill_(float(np.log(sigma)))
    LOGGER.info(
        "reset action sigma to %.3g (was %s)",
        sigma, "[" + ", ".join(f"{s:.2f}" for s in before) + "]",
    )


def build_ppo_config(
    ppo_cfg: dict[str, Any], *, device: torch.device, seed: int, run_dir: Path
) -> PPOConfig:
    known = {f for f in PPOConfig.__dataclass_fields__}
    unknown = set(ppo_cfg) - known
    if unknown:
        # Silently ignoring a misspelled hyperparameter is how a 1.5M-step run
        # gets thrown away, so refuse to start instead.
        raise SystemExit(f"unknown ppo config keys: {sorted(unknown)}")
    return PPOConfig(
        **{k: v for k, v in ppo_cfg.items()},
        device=str(device),
        seed=seed,
        checkpoint_dir=str(run_dir / "checkpoints"),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a dotted config key")
    parser.add_argument("--run-name", default=None, help="override run_name")
    parser.add_argument("--resume", type=Path, default=None,
                        metavar="RUN_DIR", help="resume from a run directory")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run (8k steps) to validate the pipeline")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the minibatch update")
    parser.add_argument("--reset-action-std", type=float, default=None, metavar="SIGMA",
                        help="after --resume, re-initialize the policy's action "
                             "sigma to SIGMA (rewinds a runaway log_std; the "
                             "torso and critic are kept)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build everything, print the plan, then exit")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.config.is_file():
        raise SystemExit(f"config not found: {args.config}")
    config = load_config(args.config, args.overrides)

    env_cfg = config.get("env", {})
    ppo_cfg = dict(config.get("ppo", {}))
    policy_cfg = config.get("policy", {})
    norm_cfg = config.get("normalize_obs", {})
    eval_cfg = config.get("eval", {})
    stop_cfg = config.get("early_stop", {})
    anneal_cfg = config.get("anneal", {})

    if args.smoke:
        ppo_cfg.update(total_timesteps=8192, eval_interval=2, log_interval=1,
                       checkpoint_interval=4, num_steps=64, num_minibatches=4)
        env_cfg = dict(env_cfg, num_envs=4, max_episode_steps=60)
        eval_cfg = dict(eval_cfg, n_episodes=4)
    if args.compile:
        ppo_cfg["compile_update"] = True

    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(config.get("device", "auto"))

    run_name = args.run_name or config.get("run_name", "run")

    # Where --resume *reads* from, deliberately separate from where this run
    # writes. Under --smoke the two differ, which is what lets a smoke test
    # exercise the resume path against a real checkpoint without writing anything
    # back beside it.
    resume_dir = (
        None if args.resume is None else resolve_run_dir(args.resume, smoke=False)
    )
    run_dir = resolve_run_dir(
        resume_dir or Path(config.get("output_dir", "outputs")) / run_name,
        smoke=args.smoke,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        LOGGER.info("--smoke: writing all artifacts to %s", run_dir)

    # -- envs ------------------------------------------------------------- #
    venv = build_vector_env(env_cfg, seed)
    obs_rms: RunningMeanStd | None = None
    # `freeze` stops the running statistics from being updated by rollouts, so the
    # observation scaling stays exactly what it was when loaded. It exists for
    # warm-started runs, where a drifting normalizer silently rewrites the inputs
    # the cloned policy was fitted against -- see the key's documentation in
    # configs/train/pick_place_obstacle.yaml for the measurement that motivated it
    # (same BC weights: 87% success under the demo-fitted statistics, 0% under the
    # same run's statistics 90 iterations later).
    freeze_obs_norm = bool(norm_cfg.get("freeze", False))
    if norm_cfg.get("enabled", True):
        venv = NormalizeObservation(
            venv, clip=float(norm_cfg.get("clip", 10.0)), training=not freeze_obs_norm
        )
        obs_rms = venv.obs_rms

    obs_dim = int(np.prod(venv.single_observation_space.shape))
    action_dim = int(np.prod(venv.single_action_space.shape))

    # -- policy ----------------------------------------------------------- #
    policy = build_policy(policy_cfg, obs_dim, action_dim).to(device)
    ppo_config = build_ppo_config(ppo_cfg, device=device, seed=seed, run_dir=run_dir)

    batch = ppo_config.batch_size(int(env_cfg.get("num_envs", 8)))
    if batch % ppo_config.num_minibatches:
        LOGGER.warning(
            "batch_size %d is not divisible by num_minibatches %d; the final "
            "minibatch will be ragged", batch, ppo_config.num_minibatches,
        )

    # -- evaluator (its own env, deterministic seeds, frozen statistics) ---- #
    # Curriculum seeding is forced off for evaluation. Eval has to measure the
    # task, and a seeded eval env measures something else entirely: at
    # curriculum_level=3 the object starts *at the goal*, so those episodes are
    # already inside the success threshold on step 0 and score as successes the
    # policy did not earn. Sharing env_cfg with training would silently do that.
    eval_env_cfg = dict(env_cfg, curriculum_level=0)
    eval_env = make_env_fn(eval_env_cfg, seed + 99_991, 0)()
    evaluator = Evaluator(
        eval_env,
        EvaluationConfig(
            n_episodes=int(eval_cfg.get("n_episodes", 20)),
            deterministic=bool(eval_cfg.get("deterministic", True)),
            base_seed=int(eval_cfg.get("base_seed", 1_000_000)),
            device=str(device),
        ),
        obs_rms=obs_rms,
        obs_clip=float(norm_cfg.get("clip", 10.0)),
    )

    # -- callbacks --------------------------------------------------------- #
    # No smoke-specific filename here: run_dir is already the isolated .smoke
    # directory, so metrics.jsonl inside it cannot collide with the real run's.
    # Keeping the standard name also means plot_results.py can plot a smoke run,
    # which is how the plotting path itself gets exercised.
    callbacks: list[Any] = [
        ProgressCallback(every_n_iterations=max(1, ppo_config.log_interval)),
        # Advances the reverse curriculum's seeding probability. Installed
        # unconditionally: it no-ops on tasks without a curriculum, and at
        # curriculum_level=0 the env ignores the probability entirely.
        CurriculumSchedule(log_interval=max(1, ppo_config.log_interval)),
    ]

    # Both of the next two *add* scalars to the metrics dict the trainer is
    # assembling, and TensorBoardCallback/MetricHistory below only see what is in
    # that dict when their own hook fires. CallbackList runs in list order, so
    # they have to be registered ahead of the consumers or their series are
    # missing from both TensorBoard and metrics.jsonl.
    anneal_callback = build_anneal_callback(
        anneal_cfg, log_interval=max(1, ppo_config.log_interval)
    )
    if anneal_callback is not None:
        callbacks.append(anneal_callback)
    # Installed even with no ceiling configured, for the logging alone:
    # train/action_std_max is what makes a runaway policy sigma visible, and its
    # absence is why the last run's collapse read as a reward problem.
    # `or None` so an omitted key, null, false and 0 all mean "log only, do not
    # clamp" -- ActionStdCeiling rejects a non-positive ceiling, and a config
    # disabling the clamp with `action_std_ceiling: false` should not be an error.
    callbacks.append(ActionStdCeiling(policy_cfg.get("action_std_ceiling") or None))

    callbacks += [
        TensorBoardCallback(run_dir / "tb"),
        # A resume continues one series, so it appends; a fresh run rotates any
        # existing history aside instead of truncating it. MetricHistory's "auto"
        # would infer the resume from the loaded global_step anyway -- passing it
        # explicitly means the intent does not depend on that inference holding.
        MetricHistory(
            run_dir / "metrics.jsonl",
            mode="append" if resume_dir is not None and not args.smoke else "auto",
        ),
    ]
    normalizer_ckpt: NormalizerCheckpoint | None = None
    if obs_rms is not None:
        normalizer_ckpt = NormalizerCheckpoint(obs_rms, run_dir / "checkpoints")
        callbacks.append(normalizer_ckpt)
    if stop_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopOnThreshold(consecutive=int(stop_cfg.get("consecutive", 2)))
        )

    # Optional DAPG-style anchor. Absent or disabled `bc` block -> the plain
    # PPOTrainer, byte-for-byte the previous behaviour. The demonstrations are
    # loaded with the *live* obs_rms, not a fresh one: NormalizeObservation keeps
    # updating those statistics during training, so re-fitting here would whiten
    # the demonstrations by a different transform than the rollouts get, and the
    # auxiliary term would be pulling the policy toward a distribution it never
    # sees.
    # Optional return-scale normalization of the reward PPO learns from. Off unless
    # asked for, so every existing config is bit-identical.
    #
    # What it is for: episode returns on the pick-and-place tasks run ~2500
    # unnormalized, which puts the value loss near 1.7e4 and the global gradient
    # norm near 1000 against a max_grad_norm of 10 -- so clipping scales *both*
    # networks down by ~100x and the critic never catches its target
    # (explained_variance ~0.4). Dividing the reward by the running standard
    # deviation of the discounted return brings the value target to order 1.
    #
    # What it costs: NormalizeReward's own docstring makes the argument against it,
    # and that argument is real -- this reward is deliberately built from bounded
    # 1 - tanh(k*d) terms with hand-chosen stage weights. The scaling is a single
    # scalar applied to the whole reward, so it preserves the *ratios* between those
    # weights, but it does tie the effective learning rate to whatever the return
    # happens to look like early in a run. Treat it as an experiment, not a default.
    norm_reward_cfg = config.get("normalize_reward", {}) or {}
    reward_normalizer: NormalizeReward | None = None
    if norm_reward_cfg.get("enabled", False):
        unknown = sorted(set(norm_reward_cfg) - {"enabled", "clip"})
        if unknown:
            raise SystemExit(
                f"unknown normalize_reward keys: {unknown}; have ['clip', 'enabled']"
            )
        reward_normalizer = NormalizeReward(
            num_envs=int(env_cfg.get("num_envs", 8)),
            gamma=ppo_config.gamma,
            clip=float(norm_reward_cfg.get("clip", 10.0)),
        )
        LOGGER.info(
            "reward norm  on (clip %.3g) -- rollout/episode_return stays raw; "
            "train/reward_scale logs the divisor",
            float(norm_reward_cfg.get("clip", 10.0)),
        )
        # Refuse the combination that fails silently. scripts/pretrain_bc.py fits
        # the value head on *raw* discounted demonstration returns -- mean ~425 on
        # the 400-episode obstacle set -- while a normalized run drives the value
        # target to order 1. Resuming one into the other hands PPO a critic
        # mis-scaled by a factor of hundreds, which does not raise: it presents as
        # advantages that are noise for the first few hundred iterations, which is
        # exactly how an 87% warm start got walked down to 2% once already.
        if resume_dir is not None:
            raise SystemExit(
                "normalize_reward.enabled with --resume is refused: a checkpoint's "
                "value head was fitted on unnormalized returns (pretrain_bc.py uses "
                "raw discounted demo returns, mean ~425), and normalizing the reward "
                "rescales the value target by ~1/std(return). The critic would be "
                "wrong by orders of magnitude with no error raised.\n"
                "  Either train from scratch with normalization on, or leave it off "
                "for warm-started runs. Making the two compatible means normalizing "
                "the demonstration returns on the same statistic -- a change to "
                "DemoBuffer, not a config key."
            )

    bc_cfg = config.get("bc", {}) or {}
    if bc_cfg.get("enabled", False):
        unknown = sorted(set(bc_cfg) - _BC_KEYS)
        if unknown:
            raise SystemExit(f"unknown bc config keys: {unknown}; have {sorted(_BC_KEYS)}")
        demos_path = Path(bc_cfg.get("demos", ""))
        if not demos_path.is_absolute():
            demos_path = _REPO_ROOT / demos_path
        demos = DemoBuffer(
            demos_path, device=device, obs_rms=obs_rms,
            clip=float(norm_cfg.get("clip", 10.0)), gamma=ppo_config.gamma,
        )
        if (demos.obs_dim, demos.action_dim) != (obs_dim, action_dim):
            raise SystemExit(
                f"bc.demos shapes {(demos.obs_dim, demos.action_dim)} != env "
                f"{(obs_dim, action_dim)}; re-record against this config"
            )
        LOGGER.info("bc anchor    %s (%d transitions, coef %.4g decay %.5g)",
                    demos_path.name, len(demos),
                    float(bc_cfg.get("coef", 0.1)), float(bc_cfg.get("decay", 0.999)))
        trainer = DAPGTrainer(
            venv, policy, ppo_config, demos,
            reward_normalizer=reward_normalizer,
            bc_coef=float(bc_cfg.get("coef", 0.1)),
            bc_decay=float(bc_cfg.get("decay", 0.999)),
            bc_batch_size=int(bc_cfg.get("batch_size", 256)),
            callbacks=callbacks, evaluator=evaluator,
        )
    else:
        trainer = PPOTrainer(
            venv, policy, ppo_config, callbacks=callbacks, evaluator=evaluator,
            reward_normalizer=reward_normalizer,
        )

    # -- resume ------------------------------------------------------------ #
    if resume_dir is not None:
        # Read from resume_dir, not run_dir: under --smoke the latter is the
        # isolated scratch copy and holds no checkpoint to resume from.
        source = resume_dir / "checkpoints"
        latest = _latest_checkpoint(source)
        if latest is None:
            raise SystemExit(f"--resume: no checkpoint found under {source}")
        trainer.load_checkpoint(latest)
        # Normalizer statistics sit beside the checkpoint being loaded, so they
        # come from source too. NormalizerCheckpoint.load() writes into the
        # obs_rms it was handed, which is the live one, so a throwaway loader
        # pointed at source is enough.
        if obs_rms is not None and not NormalizerCheckpoint(obs_rms, source).load():
            if freeze_obs_norm:
                # Frozen *and* unloaded means every observation is scaled by an
                # all-zeros mean and unit variance for the whole run, with nothing
                # to correct it. That trains and logs cleanly while feeding the
                # policy garbage, so it has to be fatal rather than a warning.
                raise SystemExit(
                    f"normalize_obs.freeze is set but no normalizer statistics were "
                    f"found under {source}; a frozen normalizer has no way to "
                    f"recover from that. Run scripts/pretrain_bc.py first, or unset "
                    f"the key."
                )
            LOGGER.warning(
                "resumed policy from %s but found no saved normalizer statistics; "
                "observation scaling will differ from the original run", latest.name,
            )
        elif obs_rms is not None and freeze_obs_norm:
            LOGGER.info(
                "normalizer  frozen at count=%.0f (rollouts will not update it)",
                obs_rms.count,
            )
        LOGGER.info("Resumed from %s at step %s", latest.name, f"{trainer.global_step:,}")
        if args.reset_action_std is not None:
            # After load_checkpoint, so it overwrites the checkpoint's log_std
            # rather than being overwritten by it.
            reset_action_std(policy, args.reset_action_std)
    elif args.reset_action_std is not None:
        # A fresh policy's sigma is policy.log_std_init, which is already in the
        # config and lands in config.resolved.json. Accepting the flag here would
        # set the same quantity from two places, with only one of them recorded.
        raise SystemExit(
            "--reset-action-std applies to a resumed policy; for a fresh run set "
            "policy.log_std_init instead (log_std_init = ln sigma)"
        )

    resolved = {
        "run_name": run_name, "seed": seed, "device": str(device),
        # Recorded so the artifact identifies itself: scripts/enjoy.py rebuilds the
        # env and policy from this file, and an 8k-step smoke policy evaluated as
        # if it were a finished run is a misleading result, not an error.
        "smoke": bool(args.smoke),
        "resumed_from": None if resume_dir is None else str(resume_dir),
        "env": env_cfg, "policy": policy_cfg, "normalize_obs": norm_cfg,
        "eval": eval_cfg, "early_stop": stop_cfg, "anneal": anneal_cfg,
        # ppo.ent_coef below is only the *starting* coefficient when the annealer
        # is active -- it is rewritten every iteration, so the resolved file would
        # otherwise imply a fixed value the run never used.
        "reset_action_std": args.reset_action_std,
        "ppo": asdict(ppo_config) if hasattr(ppo_config, "__dataclass_fields__") else {},
        "obs_dim": obs_dim, "action_dim": action_dim,
        "n_policy_params": sum(p.numel() for p in policy.parameters()),
    }
    (run_dir / "config.resolved.json").write_text(json.dumps(resolved, indent=2, default=str))

    LOGGER.info("run dir      %s", run_dir)
    LOGGER.info("task         %s (%d envs, %s control)", env_cfg.get("task"),
                env_cfg.get("num_envs"), env_cfg.get("control_mode"))
    LOGGER.info("obs/action   %d / %d", obs_dim, action_dim)
    LOGGER.info("policy       %s params", f"{resolved['n_policy_params']:,}")
    LOGGER.info("budget       %s steps, %d iters x %d batch",
                f"{ppo_config.total_timesteps:,}", trainer.num_iterations, batch)
    LOGGER.info("device       %s", device)

    if args.dry_run:
        LOGGER.info("--dry-run: exiting before training")
        venv.close()
        evaluator.close()
        return 0

    try:
        trainer.train()
    except KeyboardInterrupt:
        LOGGER.warning("interrupted; checkpoint written to %s", run_dir / "checkpoints")
        return 130
    finally:
        venv.close()
        evaluator.close()

    LOGGER.info("final eval   %s", evaluator.summary())
    LOGGER.info("best eval success %.1f%%", trainer.best_eval_success * 100)
    LOGGER.info("threshold %s reached", "WAS" if trainer.threshold_reached else "was NOT")
    return 0


def _latest_checkpoint(directory: Path) -> Path | None:
    """Furthest-along checkpoint: newest ``step_*.pt`` or a later ``interrupted.pt``.

    ``interrupted.pt`` is considered alongside the stepped files rather than only
    as a fallback. ``checkpoint_interval`` is 50 iterations, so a Ctrl-C lands on
    average 25 iterations past the last ``step_*.pt`` -- resuming from the stepped
    file silently replays that gap, which is ~200k env steps at this batch size.
    Its ``global_step`` has to be read from the payload because, unlike the
    stepped files, the name does not carry one.

    ``final.pt``/``best.pt`` stay pure fallbacks: ``best.pt`` is whichever
    iteration evaluated best and is usually far behind, so preferring it by step
    count would be wrong.
    """
    if not directory.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []
    for path in directory.glob("step_*.pt"):
        suffix = path.stem.split("_")[1]
        if suffix.isdigit():
            candidates.append((int(suffix), path))

    interrupted = directory / "interrupted.pt"
    if interrupted.is_file():
        try:
            payload = torch.load(interrupted, map_location="cpu", weights_only=False)
            candidates.append((int(payload.get("global_step", 0)), interrupted))
        except Exception:  # noqa: BLE001 - a corrupt file must not block a resume
            LOGGER.warning("ignoring unreadable %s", interrupted)

    if candidates:
        # max() on (step, path) would tie-break on the path string; sort by step
        # only and let the last-added candidate win a tie, which is interrupted.pt
        # -- the one written later.
        return sorted(candidates, key=lambda item: item[0])[-1][1]

    for name in ("final.pt", "best.pt"):
        if (directory / name).is_file():
            return directory / name
    return None


if __name__ == "__main__":
    raise SystemExit(main())
