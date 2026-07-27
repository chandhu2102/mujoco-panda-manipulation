#!/usr/bin/env python3
"""Pre-train a policy on recorded demonstrations, then hand it to PPO.

    python scripts/pretrain_bc.py --config configs/train/pick_place_bc_torque.yaml \
        --demos demos/pick_place_torque.npz --out outputs/pick_place_bc_torque

    python scripts/train.py --config configs/train/pick_place_bc_torque.yaml \
        --resume outputs/pick_place_bc_torque

Writes a run directory that ``train.py --resume`` can consume directly:

    <out>/checkpoints/best.pt             policy + optimizer, global_step 0
    <out>/checkpoints/bc_init.pt          immutable copy of the above
    <out>/checkpoints/obs_normalizer.npz  statistics fitted on the demonstrations
    <out>/bc_summary.json                 what was fitted, and on what

``best.pt`` rather than a new filename because ``train.py``'s
``_latest_checkpoint`` already looks for it as a fallback, so no launcher change is
needed. ``bc_init.pt`` exists because the resumed run writes its own ``best.pt``
into the same directory the moment an evaluation beats the initial score -- the
run directory *is* ``--resume``'s directory, by design -- and the clone is worth
keeping for comparison.

``global_step`` is written as 0 deliberately, so the resumed run gets its full
budget and the entropy/curriculum/penalty schedules all start at progress 0 rather
than partway along.

The normalizer is the part most easily got wrong. Training whitens observations at
the vector-env boundary with ``NormalizeObservation``, so the policy never sees a
raw observation; ``record_demos.py`` records raw ones. Fitting the statistics here
and saving them where ``NormalizerCheckpoint`` looks is what makes the cloned
weights mean the same thing on both sides of the handoff.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from mujoco_manip.training.bc import BCConfig, BCPretrainer, DemoBuffer  # noqa: E402
from mujoco_manip.algos.common.normalizers import RunningMeanStd  # noqa: E402
from mujoco_manip.training.callbacks import (  # noqa: E402
    RETURN_NORMALIZER_FILENAME,
    NormalizerCheckpoint,
)
from train import (  # noqa: E402
    build_policy,
    load_config,
    make_env_fn,
    resolve_device,
    resolve_run_dir,
    seed_everything,
)

LOGGER = logging.getLogger("pretrain_bc")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--demos", type=Path, default=None,
                        help="demonstration .npz; defaults to config bc.demos")
    parser.add_argument("--out", type=Path, required=True, help="run directory to write")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--final-sigma", type=float, default=None,
                        help="action sigma installed after cloning; see BCConfig")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S",
    )
    if not args.config.is_file():
        raise SystemExit(f"config not found: {args.config}")
    config = load_config(args.config, args.overrides)

    demos_path = args.demos or config.get("bc", {}).get("demos")
    if demos_path is None:
        raise SystemExit("no demonstrations: pass --demos or set bc.demos in the config")
    demos_path = Path(demos_path)
    if not demos_path.is_absolute():
        demos_path = _REPO_ROOT / demos_path

    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(config.get("device", "auto"))
    run_dir = resolve_run_dir(args.out, smoke=False)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    # Build one env purely to read the spaces the policy must match. Cheaper and
    # safer than trusting the shapes in the .npz: those are what the *recorder*
    # saw, and the check below is what catches a demonstration set recorded before
    # a config change to control_mode or task_space.
    env = make_env_fn(dict(config.get("env", {})), seed, 0)()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    control_mode = env.unwrapped.control_mode
    env.close()

    bc_cfg_block = config.get("bc", {}) or {}
    gamma = float(config.get("ppo", {}).get("gamma", 0.995))
    clip = float(config.get("normalize_obs", {}).get("clip", 10.0))

    # When the PPO stage normalizes the reward, the value head has to be cloned
    # against the same return scale, or the warm start hands over a critic wrong by
    # a factor of std(return). DemoBuffer seeds this statistic from the
    # demonstrations and divides its returns by it; train.py loads the same file
    # into NormalizeReward so both stages agree from step one.
    norm_reward_cfg = config.get("normalize_reward", {}) or {}
    return_rms = RunningMeanStd(()) if norm_reward_cfg.get("enabled", False) else None

    demos = DemoBuffer(
        demos_path, device=device, clip=clip, gamma=gamma, return_rms=return_rms
    )
    if return_rms is not None:
        LOGGER.info(
            "return norm  value targets scaled by std(demo return)=%.2f "
            "(raw mean %.1f); PPO must run with normalize_reward.enabled",
            float(return_rms.std), float(demos._raw_returns.mean()),
        )
    if (demos.obs_dim, demos.action_dim) != (obs_dim, action_dim):
        raise SystemExit(
            f"demonstration shapes {(demos.obs_dim, demos.action_dim)} do not match "
            f"the env's {(obs_dim, action_dim)}. Re-record with this config: the "
            f"task-space wrapper changes both dimensions, so a dataset from a "
            f"different control setup silently encodes a different action."
        )
    if demos.control_mode != control_mode:
        raise SystemExit(
            f"demonstrations were recorded under control_mode={demos.control_mode!r} "
            f"but this config uses {control_mode!r}. The action dimension matches by "
            f"coincidence -- 7 torques and 7 joint angles are both 7 numbers -- but "
            f"the labels mean entirely different things."
        )
    LOGGER.info(
        "demos        %s: %d transitions, %d episodes, control_mode=%s, task_space=%s",
        demos_path.name, len(demos), demos.n_episodes, demos.control_mode,
        demos.task_space,
    )
    LOGGER.info("obs/action   %d / %d", obs_dim, action_dim)

    bc_cfg = BCConfig(
        epochs=int(args.epochs if args.epochs is not None else bc_cfg_block.get("epochs", 100)),
        batch_size=int(args.batch_size if args.batch_size is not None else bc_cfg_block.get("batch_size", 256)),
        learning_rate=float(
            args.learning_rate if args.learning_rate is not None
            else bc_cfg_block.get("learning_rate", 1e-3)
        ),
        value_coef=float(bc_cfg_block.get("value_coef", 0.5)),
        final_log_std=(
            args.final_sigma if args.final_sigma is not None
            else bc_cfg_block.get("final_sigma", 0.3)
        ),
    )

    policy = build_policy(config.get("policy", {}), obs_dim, action_dim).to(device)
    pretrainer = BCPretrainer(policy, demos, bc_cfg, device=device)

    before = pretrainer.evaluate(demos.observations, demos.actions, demos.returns)
    LOGGER.info("before       nll=%.3f action_mse=%.5f", before["nll"], before["action_mse"])
    stats = pretrainer.fit()
    after = pretrainer.evaluate(demos.observations, demos.actions, demos.returns)
    LOGGER.info("after        nll=%.3f action_mse=%.5f", after["nll"], after["action_mse"])

    # Statistics first: a checkpoint present without them is the failure mode
    # train.py can only warn about.
    NormalizerCheckpoint(demos.obs_rms, checkpoints).save()
    if return_rms is not None:
        NormalizerCheckpoint(
            return_rms, checkpoints,
            filename=RETURN_NORMALIZER_FILENAME, label="return",
        ).save()

    payload = {
        "policy": policy.state_dict(),
        # None rather than a fresh Adam state. train.py's load_checkpoint skips a
        # None optimizer and keeps the one it just built at the config's learning
        # rate, which is what a run starting PPO from scratch wants; a BC
        # optimizer's moments are for a different objective.
        "optimizer": None,
        "global_step": 0,
        "iteration": 0,
        "best_eval_success": -1.0,
        "bc": {"demos": str(demos_path), "config": asdict(bc_cfg), "stats": after},
    }
    torch.save(payload, checkpoints / "best.pt")
    torch.save(payload, checkpoints / "bc_init.pt")

    summary = {
        "demos": str(demos_path),
        "n_transitions": len(demos),
        "n_episodes": demos.n_episodes,
        "control_mode": demos.control_mode,
        "task_space": demos.task_space,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "gamma": gamma,
        "obs_normalizer_count": float(demos.obs_rms.count),
        "bc_config": asdict(bc_cfg),
        "before": before,
        "after": after,
        "final_stats": stats,
    }
    (run_dir / "bc_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    LOGGER.info("wrote %s", checkpoints / "best.pt")
    LOGGER.info("resume with: python scripts/train.py --config %s --resume %s",
                args.config, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
