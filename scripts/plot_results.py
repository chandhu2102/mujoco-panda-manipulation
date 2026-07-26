#!/usr/bin/env python3
"""Plot a run's training curve from its ``metrics.jsonl``.

    python scripts/plot_results.py                       # newest non-empty run under outputs/
    python scripts/plot_results.py outputs/reach_1p5m    # a specific run
    python scripts/plot_results.py outputs/reach_1p5m -o /tmp/curve.png --theme dark

Two stacked panels share one x-axis (environment steps): success rate on top,
episode return below. They are deliberately *not* overlaid on twin y-axes -- a
rate in [0, 1] and a return in the hundreds share no scale, and aligning them on
one plot would invent a correlation the data does not contain.

Rollout metrics are logged every ``log_interval`` iterations and are noisy, so
each is drawn twice: the raw trace held back at low opacity, and an
exponential moving average carrying the readable line. Eval metrics are logged
only every ``eval_interval`` iterations, so they are drawn as their own marked
series rather than smoothed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write a file; never try to open a window

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]

LOGGER = logging.getLogger("plot_results")

#: A run's metrics file, newest spelling first. Older runs wrote the singular.
METRICS_FILENAMES = ("metrics.jsonl", "metric.jsonl")

X_KEY = "global_step"


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Theme:
    """Chart ink for one surface.

    Series 1/2 are the first two categorical slots, stepped for the surface they
    render on -- the dark column is the same two hues re-stepped, not an
    automatic inversion of the light one.
    """

    surface: str
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    axis: str
    series_1: str
    series_2: str


THEMES = {
    "light": Theme(
        surface="#fcfcfb",
        text_primary="#0b0b0b",
        text_secondary="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series_1="#2a78d6",
        series_2="#eb6834",
    ),
    "dark": Theme(
        surface="#1a1a19",
        text_primary="#ffffff",
        text_secondary="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series_1="#3987e5",
        series_2="#d95926",
    ),
}

FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

RAW_ALPHA = 0.22


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def find_metrics_file(run_dir: Path) -> Path:
    """Return the metrics file inside ``run_dir``."""
    for name in METRICS_FILENAMES:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no {' or '.join(METRICS_FILENAMES)} in {run_dir}"
    )


def find_latest_run(outputs_dir: Path) -> Path:
    """Return the most recently written run directory that logged anything.

    Runs that died before their first log interval leave a zero-byte metrics
    file behind; those are skipped rather than reported as the active run.
    """
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"no outputs directory at {outputs_dir}")

    candidates = []
    for run_dir in outputs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        try:
            metrics = find_metrics_file(run_dir)
        except FileNotFoundError:
            continue
        if metrics.stat().st_size == 0:
            LOGGER.debug("skipping %s: metrics file is empty", run_dir.name)
            continue
        candidates.append((metrics.stat().st_mtime, run_dir))

    if not candidates:
        raise FileNotFoundError(
            f"no run under {outputs_dir} has a non-empty metrics file"
        )

    candidates.sort()
    return candidates[-1][1]


def load_records(metrics_path: Path) -> list[dict]:
    """Parse ``metrics.jsonl`` into records, tolerating a truncated tail.

    A run killed mid-write can leave a half-written final line; that is worth a
    warning, not a crash.
    """
    records: list[dict] = []
    skipped = 0
    with metrics_path.open() as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                LOGGER.warning("%s:%d is not valid JSON, skipping", metrics_path, lineno)
                continue
            if X_KEY not in record:
                skipped += 1
                LOGGER.warning("%s:%d has no %r, skipping", metrics_path, lineno, X_KEY)
                continue
            records.append(record)

    if not records:
        raise ValueError(f"{metrics_path} contained no usable records")
    if skipped:
        LOGGER.warning("skipped %d unusable line(s) in %s", skipped, metrics_path)
    records.sort(key=lambda r: r[X_KEY])
    return records


def series(records: list[dict], key: str) -> tuple[list[float], list[float]]:
    """Extract the ``(steps, values)`` pairs where ``key`` is present and finite.

    Keys are logged at different intervals -- and eval keys only on eval
    iterations -- so every series carries its own x values rather than sharing
    one index.
    """
    steps: list[float] = []
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN
            continue
        steps.append(float(record[X_KEY]))
        values.append(value)
    return steps, values


def ema(values: list[float], span: int) -> list[float]:
    """Exponential moving average, seeded on the first sample (no warm-up dip)."""
    if span <= 1 or not values:
        return list(values)
    alpha = 2.0 / (span + 1.0)
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1.0 - alpha) * out[-1])
    return out


def run_title(run_dir: Path) -> str:
    """Human label for the run: its configured name and task if available."""
    config_path = run_dir / "config.resolved.json"
    name, task = run_dir.name, None
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            config = {}
        name = config.get("run_name") or name
        env = config.get("env")
        if isinstance(env, dict):
            task = env.get("task")
    return f"{name} — {task} task" if task else name


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Panel:
    """One subplot: a rollout series, and the matching eval series if logged."""

    title: str
    rollout_key: str
    rollout_label: str
    eval_key: str
    eval_label: str
    as_percent: bool = False


PANELS = (
    Panel(
        title="Success rate",
        rollout_key="rollout/success_rate",
        rollout_label="rollout (smoothed)",
        eval_key="eval/success_rate",
        eval_label="eval (deterministic)",
        as_percent=True,
    ),
    Panel(
        title="Episode return",
        rollout_key="rollout/episode_return",
        rollout_label="rollout (smoothed)",
        eval_key="eval/return_mean",
        eval_label="eval (deterministic)",
    ),
)


def _format_value(value: float, as_percent: bool) -> str:
    if as_percent:
        return f"{value * 100:.0f}%"
    return f"{value:,.0f}" if abs(value) >= 10 else f"{value:.2f}"


def _style_axes(ax, theme: Theme) -> None:
    """Recessive chrome: hairline horizontal grid, no top/right spines."""
    ax.set_facecolor(theme.surface)
    ax.grid(True, axis="y", color=theme.grid, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.axis)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme.muted, labelsize=9, length=0)


def _draw_panel(ax, panel: Panel, records: list[dict], theme: Theme, span: int,
                with_legend: bool) -> bool:
    """Draw one panel. Returns False if neither of its series was logged."""
    drawn = 0

    roll_steps, roll_values = series(records, panel.rollout_key)
    if roll_values:
        # Raw trace recedes; the EMA carries the readable line.
        ax.plot(
            roll_steps,
            roll_values,
            color=theme.series_1,
            linewidth=1.0,
            alpha=RAW_ALPHA,
            zorder=2,
        )
        ax.plot(
            roll_steps,
            ema(roll_values, span),
            color=theme.series_1,
            linewidth=1.8,
            label=panel.rollout_label,
            zorder=3,
        )
        drawn += 1

    eval_steps, eval_values = series(records, panel.eval_key)
    if eval_values:
        ax.plot(
            eval_steps,
            eval_values,
            color=theme.series_2,
            linewidth=1.8,
            marker="o",
            markersize=5.0,
            # A surface-colored ring separates markers where they overlap the
            # rollout line, instead of outlining every mark in ink.
            markeredgecolor=theme.surface,
            markeredgewidth=1.6,
            label=panel.eval_label,
            zorder=4,
        )
        drawn += 1

        # Direct-label the endpoint only -- the axis carries the rest. Anchor it
        # past the rightmost sample of either series so the label never lands on
        # the rollout line, which usually runs a few intervals beyond the last eval.
        label_x = max(eval_steps[-1], roll_steps[-1] if roll_steps else eval_steps[-1])
        ax.annotate(
            _format_value(eval_values[-1], panel.as_percent),
            xy=(label_x, eval_values[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=theme.text_secondary,
            fontsize=9,
            va="center",
            ha="left",
            zorder=5,
        )

    if not drawn:
        return False

    ax.set_title(
        panel.title,
        color=theme.text_primary,
        fontsize=11,
        loc="left",
        pad=8,
    )
    if panel.as_percent:
        ax.set_ylim(-0.04, 1.06)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))

    # One series needs no legend box: the panel title already names it. Every
    # panel shows the same two series, so only the first one carries the legend.
    if drawn >= 2 and with_legend:
        legend = ax.legend(
            loc="lower right",
            frameon=False,
            fontsize=9,
            handlelength=1.8,
            borderpad=0.2,
        )
        for text in legend.get_texts():
            text.set_color(theme.text_secondary)

    _style_axes(ax, theme)
    return True


def plot(records: list[dict], run_dir: Path, out_path: Path, theme: Theme,
         span: int, dpi: int) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = FONT_STACK

    panels = [
        p for p in PANELS
        if series(records, p.rollout_key)[1] or series(records, p.eval_key)[1]
    ]
    if not panels:
        raise ValueError(
            "none of the expected metrics were logged: "
            + ", ".join(k for p in PANELS for k in (p.rollout_key, p.eval_key))
        )

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(8.5, 3.1 * len(panels) + 0.8),
        sharex=True,
        constrained_layout=True,
    )
    if len(panels) == 1:
        axes = [axes]
    fig.patch.set_facecolor(theme.surface)

    legend_drawn = False
    for ax, panel in zip(axes, panels):
        _draw_panel(ax, panel, records, theme, span, with_legend=not legend_drawn)
        legend_drawn = legend_drawn or bool(ax.get_legend())

    steps = [r[X_KEY] for r in records]
    scale, unit = (1e6, "M") if max(steps) >= 1e6 else (1e3, "k")
    axes[-1].xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / scale:g}{unit}")
    )
    axes[-1].set_xlabel(
        "Environment steps", color=theme.text_secondary, fontsize=10, labelpad=6
    )
    # Leave room on the right for the endpoint labels, which sit outside the data.
    axes[-1].set_xlim(0, max(steps) * 1.06)

    fig.suptitle(
        run_title(run_dir),
        color=theme.text_primary,
        fontsize=13,
        fontweight="bold",
        x=0.008,
        ha="left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=theme.surface)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="run output directory (default: newest non-empty run under outputs/)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="output PNG (default: <run_dir>/training_curve.png)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=9,
        metavar="N",
        help="EMA span for rollout series in log intervals; 1 disables (default: 9)",
    )
    parser.add_argument(
        "--theme",
        choices=sorted(THEMES),
        default="light",
        help="chart surface (default: light)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="output DPI (default: 200)")
    parser.add_argument(
        "--table",
        action="store_true",
        help="also print the eval series as a table (the plot's WCAG-clean twin)",
    )
    return parser.parse_args(argv)


def print_eval_table(records: list[dict]) -> None:
    """The table view: every eval value the chart encodes, as text."""
    steps, success = series(records, "eval/success_rate")
    _, returns = series(records, "eval/return_mean")
    if not steps:
        LOGGER.warning("no eval records to tabulate")
        return
    print(f"{'steps':>12}  {'success':>8}  {'return':>10}")
    for i, step in enumerate(steps):
        ret = f"{returns[i]:10.2f}" if i < len(returns) else " " * 10
        print(f"{step:>12,.0f}  {success[i] * 100:7.1f}%  {ret}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)

    try:
        run_dir = args.run_dir or find_latest_run(_REPO_ROOT / "outputs")
        run_dir = run_dir.resolve()
        metrics_path = find_metrics_file(run_dir)
        records = load_records(metrics_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 1

    if args.run_dir is None:
        LOGGER.info("using most recent run: %s", run_dir)

    out_path = (args.out or run_dir / "training_curve.png").resolve()
    try:
        plot(records, run_dir, out_path, THEMES[args.theme], args.smooth, args.dpi)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("read %d records from %s", len(records), metrics_path)
    LOGGER.info("wrote %s", out_path)

    if args.table:
        print_eval_table(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
