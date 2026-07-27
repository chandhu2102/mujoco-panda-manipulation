"""Entry point: launch training or replay by task name.

    PYTHONPATH=src python -m mujoco_manip.cli.main train pick_place
    PYTHONPATH=src python -m mujoco_manip.cli.main train pick_place --smoke
    PYTHONPATH=src python -m mujoco_manip.cli.main train pick_place --set ppo.gamma=0.99
    PYTHONPATH=src python -m mujoco_manip.cli.main enjoy outputs/pick_place_3m --no-viewer
    PYTHONPATH=src python -m mujoco_manip.cli.main tasks

``PYTHONPATH=src`` is needed because the project has no ``pyproject.toml`` yet, so
``mujoco_manip`` is not on the path by default. The ``scripts/`` launchers insert
it themselves and need no such prefix; drop it here once the package is
installable.

This wraps ``scripts/train.py`` and ``scripts/enjoy.py`` rather than
reimplementing them: it resolves a task string to its default config and
forwards every remaining argument through untouched, so ``--set``, ``--smoke``,
``--resume`` and friends behave identically whichever way a run is started.

Two mappings live here, and ``tasks`` prints them joined:

* task -> env class, from ``mujoco_manip.envs.tasks.TASK_REGISTRY``
* task -> default training config, in ``TASK_CONFIGS`` below

``check`` validates both without building a simulation, which is the cheap way
to catch a task registered in one table and missing from the other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..envs.tasks import TASK_REGISTRY, available_tasks

__all__ = ["TASK_CONFIGS", "config_for_task", "main"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts"
CONFIG_ROOT = _REPO_ROOT / "configs" / "train"


# Task string -> default training config. Every registered task that has a
# validated config gets an entry; `manipulation` has none yet, so it is reachable
# only via an explicit --config path rather than being silently pointed at
# another task's hyperparameters.
TASK_CONFIGS: dict[str, Path] = {
    "reach": CONFIG_ROOT / "reach.yaml",
    "pick_place": CONFIG_ROOT / "pick_place.yaml",
    # Note this is the only pick_place variant whose default config is the one
    # that actually converges. `pick_place` maps to the torque config, which is
    # the honest default for that task string but reached 0% over 6.4M steps --
    # configs/train/pick_place_osc_bc.yaml is the one to reach for, and it has no
    # task string of its own because it trains the same `pick_place` env.
    "pick_place_obstacle": CONFIG_ROOT / "pick_place_obstacle.yaml",
}


def config_for_task(task: str) -> Path:
    """Default config path for ``task``.

    Raises ``SystemExit`` with the valid options on an unknown or config-less
    task, and on an entry whose file has gone missing -- a stale mapping should
    fail before a run starts, not after it has allocated 16 environments.
    """
    if task not in TASK_REGISTRY:
        raise SystemExit(f"unknown task {task!r}; registered: {available_tasks()}")
    if task not in TASK_CONFIGS:
        raise SystemExit(
            f"task {task!r} has no default config; pass one explicitly with "
            f"--config. Tasks with configs: {sorted(TASK_CONFIGS)}"
        )
    path = TASK_CONFIGS[task]
    if not path.is_file():
        raise SystemExit(f"config for task {task!r} is missing: {path}")
    return path


def _load_script(name: str):
    """Import ``scripts/<name>.py``.

    The scripts are standalone launchers rather than package modules, so they are
    reached by path. Both of them also insert ``src`` on ``sys.path`` themselves,
    which is harmless when it is already importable.
    """
    for entry in (_REPO_ROOT / "src", _SCRIPTS):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    if not (_SCRIPTS / f"{name}.py").is_file():
        raise SystemExit(f"launcher not found: {_SCRIPTS / f'{name}.py'}")
    return __import__(name)


def _cmd_tasks() -> int:
    width = max(len(t) for t in available_tasks())
    print(f"{'task'.ljust(width)}  {'env class'.ljust(22)}  default config")
    for task in available_tasks():
        cfg = TASK_CONFIGS.get(task)
        shown = "-" if cfg is None else str(cfg.relative_to(_REPO_ROOT))
        print(f"{task.ljust(width)}  {TASK_REGISTRY[task].__name__.ljust(22)}  {shown}")
    return 0


def _cmd_check() -> int:
    problems: list[str] = []
    for task, path in TASK_CONFIGS.items():
        if task not in TASK_REGISTRY:
            problems.append(f"{task}: has a config but is not in TASK_REGISTRY")
        if not path.is_file():
            problems.append(f"{task}: config missing at {path}")
    for problem in problems:
        print(f"FAIL  {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"OK    {len(TASK_CONFIGS)} task->config mappings resolve")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mujoco-manip",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a task (wraps scripts/train.py)")
    # No argparse `choices`: config_for_task already reports unknown tasks with
    # the registered list, and choices interacts badly with nargs="?" defaulting
    # to None for the --config-only form.
    p_train.add_argument(
        "task", nargs="?", default=None,
        help=f"task to train, one of {available_tasks()}; resolves to its default config",
    )
    p_train.add_argument(
        "--config", type=Path, default=None,
        help="explicit config path, overriding the task's default",
    )

    sub.add_parser("enjoy", help="replay a run (wraps scripts/enjoy.py)")
    sub.add_parser("tasks", help="list registered tasks and their configs")
    sub.add_parser("check", help="validate the task->config mappings")

    args, extra = parser.parse_known_args(argv)

    if args.command == "tasks":
        return _cmd_tasks()
    if args.command == "check":
        return _cmd_check()
    if args.command == "enjoy":
        return int(_load_script("enjoy").main(extra))

    if args.config is not None:
        config = args.config
    elif args.task is not None:
        config = config_for_task(args.task)
    else:
        raise SystemExit(
            f"train needs a task or --config; tasks: {sorted(TASK_CONFIGS)}"
        )
    return int(_load_script("train").main(["--config", str(config), *extra]))


if __name__ == "__main__":
    raise SystemExit(main())
