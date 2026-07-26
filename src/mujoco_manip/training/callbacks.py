"""Callbacks: logging, video capture, early stop, best-model saving.

Video capture is not implemented here: it needs the ``envs/wrappers/video_record.py``
wrapper (still a stub) and an encoder such as imageio-ffmpeg, which is not among
this project's dependencies. ``ManipulationEnv.render()`` already returns frames,
so the missing piece is only the writer.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ..algos.common.normalizers import NormalizeObservation, RunningMeanStd
from .trainer import PPOTrainer, TrainerCallback

LOGGER = logging.getLogger(__name__)

__all__ = [
    "TensorBoardCallback",
    "NormalizerCheckpoint",
    "EarlyStopOnThreshold",
    "ProgressCallback",
    "MetricHistory",
    "CurriculumSchedule",
    "LinearSchedule",
    "EntropyPenaltySchedule",
    "ActionStdCeiling",
]

_MAX_UNWRAP = 16


def vector_call_target(env: Any, *, max_unwrap: int = _MAX_UNWRAP) -> Any | None:
    """Innermost wrapper exposing ``call``, or None if there isn't one.

    Gymnasium 1.x vector wrappers do not forward arbitrary attributes, and
    ``NormalizeObservation`` (a ``VectorObservationWrapper``) does not expose
    ``call`` at all, so ``trainer.env.call(...)`` raises ``AttributeError`` even
    when every sub-env implements the method being dispatched. Unwrap until
    something can dispatch.
    """
    for _ in range(max_unwrap):
        if env is None:
            return None
        if hasattr(env, "call"):
            return env
        env = getattr(env, "env", None)
    return None


@dataclass(frozen=True)
class LinearSchedule:
    """``start`` -> ``end``, linear in training progress, held flat outside the ramp.

    ``ramp_start``/``ramp_end`` delay or compress the move: a penalty that should
    stay at its baseline until the policy has found the behaviour worth cleaning
    up is ``LinearSchedule(0.005, 0.015, ramp_start=0.5)``, which is flat at
    0.005 over the first half of the run and only then walks up.

    Values outside ``[ramp_start, ramp_end]`` clamp to ``start``/``end`` rather
    than extrapolating, so a resumed run that overshoots its step budget pins to
    the endpoint instead of running the coefficient off the end of the schedule.
    """

    start: float
    end: float
    ramp_start: float = 0.0
    ramp_end: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.ramp_start < self.ramp_end <= 1.0:
            raise ValueError(
                "need 0 <= ramp_start < ramp_end <= 1, got "
                f"ramp_start={self.ramp_start}, ramp_end={self.ramp_end}"
            )

    def __call__(self, progress: float) -> float:
        span = self.ramp_end - self.ramp_start
        t = (float(progress) - self.ramp_start) / span
        t = min(1.0, max(0.0, t))
        return self.start + t * (self.end - self.start)

    def describe(self) -> str:
        window = (
            ""
            if (self.ramp_start, self.ramp_end) == (0.0, 1.0)
            else f" over progress {self.ramp_start:g}-{self.ramp_end:g}"
        )
        return f"{self.start:g} -> {self.end:g}{window}"


class EntropyPenaltySchedule(TrainerCallback):
    """Anneal ``ent_coef`` and reward penalties from training progress.

    Two coupled knobs that a fixed config gets wrong in opposite directions at
    opposite ends of a long run:

    * **Entropy.** The grasp is a discovery problem -- ``is_grasped`` is a
      conjunction that torque-space exploration does not satisfy by chance -- so
      early training wants a wide policy. Precision placement at 4 cm wants a
      narrow one. A single ``ent_coef`` either starves the search or blocks the
      convergence.
    * **Penalties.** ``action_penalty``/``velocity_penalty`` bill every joule the
      arm spends. Charged full price before the reach-to-grasp trajectory exists,
      the cheapest way to stop paying is to stop moving, and holding still is a
      local optimum the policy can find far more easily than a grasp. Charged
      only once the behaviour exists, they do what they are for: clean up torque
      chatter around a trajectory already worth keeping.

    Both are pushed once per iteration. The timing differs, and only one of them
    is exact:

    * ``ent_coef`` is read by ``PPOTrainer._minibatch_loss`` at update time, and
      this fires from ``on_rollout_end`` -- i.e. after collection, before the
      update -- so the update uses the value for the progress it is at.
    * The penalties live inside the envs and shape reward as it is generated, so
      a value pushed after a rollout applies to the *next* one. One iteration of
      lag out of hundreds, the same approximation ``CurriculumSchedule`` accepts.

    ``rebase_on_resume`` decides what "progress" means for a resumed run. Left
    False (the default), progress is ``global_step / total_timesteps`` -- the
    same measure ``CurriculumSchedule`` uses, so the two schedules stay in phase.
    That also means resuming at 6.2M of 10M starts this schedule 62% annealed,
    which is correct for continuing a run and wrong for restarting a collapsed
    one. True instead maps ``[resume_step, total]`` onto ``[0, 1]``, giving the
    resumed policy the full entropy schedule again -- at the cost of running the
    curriculum's seeding probability out of phase with it, which is why it is
    opt-in and logged loudly.
    """

    def __init__(
        self,
        *,
        entropy: LinearSchedule | None = None,
        penalties: dict[str, LinearSchedule] | None = None,
        rebase_on_resume: bool = False,
        log_interval: int = 0,
    ) -> None:
        self.entropy = entropy
        self.penalties = dict(penalties or {})
        self.rebase_on_resume = bool(rebase_on_resume)
        self._log_interval = int(log_interval)
        self._start_step = 0

    # -- progress ----------------------------------------------------------- #

    def _progress(self, trainer: PPOTrainer) -> float:
        total = max(1, int(trainer.cfg.total_timesteps))
        step = int(trainer.global_step)
        if self.rebase_on_resume and self._start_step > 0:
            span = max(1, total - self._start_step)
            fraction = (step - self._start_step) / span
        else:
            fraction = step / total
        return float(np.clip(fraction, 0.0, 1.0))

    # -- hooks -------------------------------------------------------------- #

    def on_training_start(self, trainer: PPOTrainer) -> None:
        # Non-zero only when a checkpoint was loaded before train(), so this
        # identifies a resume -- and where it resumes from -- without a flag.
        self._start_step = int(trainer.global_step)

        if trainer.cfg.compile_update and self.entropy is not None:
            # ent_coef enters the fused loss as a Python float, so torch.compile
            # guards on its value: a coefficient that changes every iteration
            # recompiles every iteration, which costs far more than the fused
            # kernel saves.
            LOGGER.warning(
                "compile_update=True with a live entropy schedule: ent_coef is a "
                "compile-time constant in the fused loss, so every scheduled "
                "change forces a recompilation. Drop --compile or pin ent_coef."
            )

        if self.rebase_on_resume and self._start_step > 0:
            LOGGER.warning(
                "rebase_on_resume: annealing over the remaining %s steps "
                "(resumed at %s of %s), so this schedule is deliberately out of "
                "phase with the curriculum's seeding probability, which stays on "
                "absolute progress.",
                f"{max(0, int(trainer.cfg.total_timesteps) - self._start_step):,}",
                f"{self._start_step:,}",
                f"{int(trainer.cfg.total_timesteps):,}",
            )

        if self.entropy is not None:
            LOGGER.info("entropy schedule   ent_coef %s", self.entropy.describe())
        for name, schedule in sorted(self.penalties.items()):
            LOGGER.info("penalty schedule   %-16s %s", name, schedule.describe())

        # Apply the start-of-schedule values before iteration 1, so the first
        # update and the first rollout already run under the schedule rather than
        # under whatever the YAML happened to set.
        self._push(trainer, self._progress(trainer), metrics=None)

    def on_rollout_end(self, trainer: PPOTrainer, metrics: dict[str, float]) -> None:
        self._push(trainer, self._progress(trainer), metrics=metrics)

    # -- application -------------------------------------------------------- #

    def _push(
        self, trainer: PPOTrainer, progress: float, *, metrics: dict[str, float] | None
    ) -> None:
        applied: dict[str, float] = {}

        if self.entropy is not None:
            trainer.cfg.ent_coef = self.entropy(progress)
            applied["ent_coef"] = trainer.cfg.ent_coef

        if self.penalties:
            values = {name: sched(progress) for name, sched in self.penalties.items()}
            self._push_penalties(trainer, values)
            applied.update(values)

        if metrics is not None:
            metrics["sched/progress"] = progress
            for name, value in applied.items():
                metrics[f"sched/{name}"] = value

        if (
            self._log_interval
            and applied
            and trainer.iteration % self._log_interval == 0
        ):
            LOGGER.info(
                "schedule progress %.3f -> %s",
                progress,
                ", ".join(f"{k}={v:.4g}" for k, v in sorted(applied.items())),
            )

    def _push_penalties(self, trainer: PPOTrainer, values: dict[str, float]) -> None:
        target = vector_call_target(trainer.env)
        if target is None:
            raise RuntimeError(
                "EntropyPenaltySchedule found no vector env exposing call() under "
                f"{type(trainer.env).__name__}, so the penalty schedule cannot "
                "reach the envs. Refusing to train: the penalties would silently "
                "stay at their config values for the whole run."
            )
        try:
            target.call("set_reward_weights", **values)
        except (AttributeError, KeyError) as exc:
            # Deliberately fatal, and fatal on the *first* push (fired from
            # on_training_start) so it lands before any compute is spent. A task
            # whose reward config lacks these fields is a misconfigured run, not
            # a run to quietly continue with a schedule that anneals nothing.
            raise RuntimeError(
                f"EntropyPenaltySchedule could not set {sorted(values)} on the "
                f"training envs: {exc!r}. Check the field names against the "
                "task's reward config."
            ) from exc


class ActionStdCeiling(TrainerCallback):
    """Cap the policy's action sigma after every update, and log where it sits.

    A guard against the failure this project has already paid for once. With a
    positive ``ent_coef`` and no advantage signal past the reach stage -- which is
    the state of any run whose grasp rate is pinned near zero -- the entropy bonus
    is the only consistent gradient on ``log_std``, and Adam walks it upward
    essentially without opposition. Measured on the 3M -> 6.4M resume run, mean
    policy entropy rose monotonically 10.5 -> 14.6 nats (sigma ~0.78 -> ~1.5,
    with three joints past 2.0) while episode return fell 213 -> 94 and the
    unassisted grasp rate decayed 0.086 -> 0.008. At that sigma the sampled
    torques are close to uniform over the action range, which both drowns the fine
    positioning ``is_grasped`` requires and rings up the action/velocity penalties
    that scale with sigma squared.

    Clamping ``log_std`` after the optimizer step bounds that runaway without
    touching the objective: exploration stays as wide as the ceiling allows and
    the entropy bonus keeps rewarding a wide *mean* policy, but sigma cannot
    ratchet past the point where the arm can no longer place the object.

    ``max_std=None`` disables the clamp and leaves only the logging, which is
    still worth installing -- ``train/action_std_max`` is the metric that makes
    this failure visible in the first place.
    """

    def __init__(self, max_std: float | None = 1.0) -> None:
        if max_std is not None and max_std <= 0.0:
            raise ValueError(f"max_std must be positive or None, got {max_std!r}")
        self.max_std = None if max_std is None else float(max_std)
        self._warned = False

    @staticmethod
    def _log_std_param(trainer: PPOTrainer) -> Any | None:
        """The free ``log_std`` tensor, or the std layer's bias for state-dependent heads."""
        head = getattr(getattr(trainer, "policy", None), "head", None)
        if head is None:
            return None
        param = getattr(head, "log_std_param", None)
        if param is not None:
            return param
        layer = getattr(head, "log_std_layer", None)
        return None if layer is None else layer.bias

    def on_training_start(self, trainer: PPOTrainer) -> None:
        if self.max_std is not None:
            LOGGER.info("action sigma ceiling %.3g", self.max_std)

    def on_update_end(self, trainer: PPOTrainer, metrics: dict[str, float]) -> None:
        param = self._log_std_param(trainer)
        if param is None:
            if not self._warned:
                self._warned = True
                LOGGER.warning(
                    "ActionStdCeiling found no log_std parameter on %s; sigma is "
                    "neither capped nor logged.",
                    type(getattr(trainer, "policy", None)).__name__,
                )
            return

        with torch.no_grad():
            if self.max_std is not None:
                ceiling = math.log(self.max_std)
                clamped = int((param > ceiling).sum())
                if clamped:
                    param.clamp_(max=ceiling)
                metrics["train/action_std_clamped"] = float(clamped)
            std = param.exp()
            metrics["train/action_std_mean"] = float(std.mean())
            metrics["train/action_std_max"] = float(std.max())


class CurriculumSchedule(TrainerCallback):
    """Drive ``ManipulationEnv.set_curriculum_progress`` from training progress.

    The env owns the probability *curve* but cannot know where in the run it is:
    it counts its own resets and has no view of the step budget. This callback
    supplies the one number it is missing, once per iteration.

    Without it the schedule never advances -- ``progress`` stays at 0.0 and the
    seeding probability sits at its starting value for the whole run, which looks
    like a working curriculum right up until you notice eval never improves
    because the policy was never asked to start from scratch.

    A no-op when the envs do not implement the method, so it is safe to install
    unconditionally -- but a *loud* no-op when the method exists and cannot be
    reached, which is the failure this class is easiest to get wrong. Gymnasium
    1.x vector wrappers do not forward arbitrary attributes, and
    ``NormalizeObservation`` (a ``VectorObservationWrapper``) does not expose
    ``call`` at all, so ``trainer.env.call(...)`` raises ``AttributeError`` even
    though every sub-env implements the method. Hence ``_vector_target``: unwrap
    until something can dispatch, and distinguish "no curriculum on this task"
    (quiet) from "could not reach the envs" (warn).
    """

    def __init__(self, *, log_interval: int = 0) -> None:
        self._log_interval = int(log_interval)
        self._supported = True

    def on_training_start(self, trainer: PPOTrainer) -> None:
        self._push(trainer, 0.0)

    def on_rollout_end(self, trainer: PPOTrainer, metrics: dict[str, float]) -> None:
        # Before the rollout would be more natural, but there is no such hook;
        # on_rollout_end still lands before the next collection, and the error is
        # one iteration out of hundreds.
        total = max(1, int(trainer.cfg.total_timesteps))
        self._push(trainer, trainer.global_step / total)

    @staticmethod
    def _vector_target(env: Any) -> Any | None:
        """Innermost wrapper exposing ``call``, or None if there isn't one."""
        return vector_call_target(env)

    def _push(self, trainer: PPOTrainer, progress: float) -> None:
        if not self._supported:
            return
        progress = float(np.clip(progress, 0.0, 1.0))

        target = self._vector_target(trainer.env)
        if target is None:
            self._supported = False
            LOGGER.warning(
                "CurriculumSchedule could not find a vector env exposing call() "
                "under %s; the curriculum will NOT anneal and seeding stays at "
                "its starting probability for the whole run.",
                type(trainer.env).__name__,
            )
            return

        try:
            probs = target.call("set_curriculum_progress", progress)
        except AttributeError:
            # The envs themselves have no curriculum. Expected for tasks that do
            # not define one, so stop trying rather than paying for the
            # exception every iteration.
            self._supported = False
            LOGGER.debug(
                "envs expose no set_curriculum_progress; curriculum schedule inactive"
            )
            return

        if self._log_interval and trainer.iteration % self._log_interval == 0:
            values = [p for p in np.atleast_1d(probs) if p is not None]
            if values:
                LOGGER.info(
                    "curriculum progress %.3f -> seed probability %.3f",
                    progress, float(np.mean(values)),
                )


class TensorBoardCallback(TrainerCallback):
    """Write scalars to TensorBoard.

    Everything is keyed by ``global_step`` (environment steps), not iteration
    count, so runs with different ``num_steps``/``num_envs`` remain directly
    comparable on the same axis.

    The ``SummaryWriter`` is built on first use rather than in ``__init__``, for
    the same reason ``MetricHistory`` defers its file: a launcher constructs its
    whole callback list while it is still deciding whether to train, and merely
    constructing a writer creates ``tb/`` and an event file on the spot.
    ``scripts/train.py --dry-run`` consequently left an orphan 88-byte event file
    in every run directory it inspected, and TensorBoard lists those alongside
    real runs -- an empty series that reads as a crashed run rather than as
    something that never started. Event files only accumulate, so this was litter
    rather than data loss, but litter indistinguishable from a failure is still
    worth not creating.
    """

    def __init__(self, log_dir: str | Path, *, flush_secs: int = 30) -> None:
        self.log_dir = Path(log_dir)
        self.flush_secs = int(flush_secs)
        self.writer: Any | None = None
        # Deliberately no mkdir and no SummaryWriter here: both touch the disk.

    def _ensure_writer(self) -> Any:
        """Build the writer on first use. Idempotent.

        Every write path goes through this, so a caller that drives the hooks
        directly without ``on_training_start`` still logs. Only training calls
        those hooks, which is what keeps the ``--dry-run`` guarantee intact
        without depending on a flag.
        """
        if self.writer is None:
            # Imported here so torch.utils.tensorboard is only needed by runs that
            # actually log, matching where the import lived before.
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir), flush_secs=self.flush_secs)
        return self.writer

    def on_training_start(self, trainer: PPOTrainer) -> None:
        writer = self._ensure_writer()
        cfg = {k: v for k, v in vars(trainer.cfg).items() if isinstance(v, (int, float, str, bool))}
        writer.add_text(
            "config", "```json\n" + json.dumps(cfg, indent=2, default=str) + "\n```"
        )

    def _write(self, metrics: dict[str, float], step: int) -> None:
        writer = self._ensure_writer()
        for key, value in metrics.items():
            if isinstance(value, (bool, int, float, np.floating, np.integer)) and np.isfinite(
                float(value)
            ):
                writer.add_scalar(key, float(value), step)

    def on_rollout_end(self, trainer: PPOTrainer, metrics: dict[str, float]) -> None:
        self._write(metrics, trainer.global_step)

    def on_update_end(self, trainer: PPOTrainer, metrics: dict[str, float]) -> None:
        self._write(metrics, trainer.global_step)

    def on_evaluation(self, trainer: PPOTrainer, metrics: dict[str, float]) -> None:
        self._write(metrics, trainer.global_step)

    def on_threshold_reached(
        self, trainer: PPOTrainer, success_rate: float, source: str
    ) -> None:
        self._ensure_writer().add_text(
            "milestones",
            f"threshold {trainer.cfg.success_threshold:.0%} crossed at step "
            f"{trainer.global_step} ({source}: {success_rate:.1%})",
            trainer.global_step,
        )

    def on_training_end(self, trainer: PPOTrainer) -> None:
        # Nothing was logged, so there is nothing to flush. Creating a writer here
        # purely to close it would recreate the empty file this defers to avoid.
        if self.writer is None:
            return
        self.writer.flush()
        self.writer.close()


class NormalizerCheckpoint(TrainerCallback):
    """Persist observation-normalizer statistics alongside policy checkpoints.

    The trainer's checkpoint holds the policy and optimizer; env-wrapper state is
    invisible to it. Resuming without these statistics rescales every observation
    the policy was trained on, which presents as an unexplained performance cliff
    rather than as an error.
    """

    FILENAME = "obs_normalizer.npz"

    def __init__(
        self,
        target: NormalizeObservation | RunningMeanStd,
        directory: str | Path,
        *,
        every_n_iterations: int = 25,
    ) -> None:
        self.rms = target.obs_rms if isinstance(target, NormalizeObservation) else target
        self.directory = Path(directory)
        self.every = max(1, int(every_n_iterations))

    def path(self) -> Path:
        return self.directory / self.FILENAME

    def save(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        state = self.rms.state_dict()
        path = self.path()
        np.savez(
            path,
            mean=state["mean"],
            m2=state["m2"],
            count=np.array(state["count"]),
            epsilon=np.array(state["epsilon"]),
            var_floor=np.array(state["var_floor"]),
            shape=np.array(state["shape"]),
        )
        return path

    def load(self) -> bool:
        path = self.path()
        if not path.is_file():
            return False
        with np.load(path) as data:
            self.rms.load_state_dict(
                {
                    "shape": tuple(int(x) for x in data["shape"]),
                    "epsilon": float(data["epsilon"]),
                    "var_floor": float(data["var_floor"]),
                    "mean": data["mean"],
                    "m2": data["m2"],
                    "count": float(data["count"]),
                }
            )
        LOGGER.info("Loaded observation normalizer (count=%.0f) from %s", self.rms.count, path)
        return True

    def on_iteration_end(self, trainer: PPOTrainer, metrics: dict[str, float]) -> None:
        if trainer.iteration % self.every == 0:
            self.save()

    def on_evaluation(self, trainer: PPOTrainer, metrics: dict[str, float]) -> None:
        # Best-model checkpoints are written during evaluation, so the statistics
        # that produced that score must be saved at the same moment.
        self.save()

    def on_training_end(self, trainer: PPOTrainer) -> None:
        LOGGER.info("Saved observation normalizer to %s", self.save())


class EarlyStopOnThreshold(TrainerCallback):
    """Stop once evaluated success clears the threshold, optionally with patience.

    ``consecutive`` guards against stopping on one lucky evaluation. Requiring two
    consecutive clears roughly squares the odds of a false positive, which matters
    when the eval set is only ~20 episodes wide.
    """

    def __init__(self, *, consecutive: int = 2, use_eval: bool = True) -> None:
        self.consecutive = max(1, int(consecutive))
        self.use_eval = use_eval
        self.streak = 0
        self.should_stop = False
        self.stopped_at_step: int | None = None

    def on_evaluation(self, trainer: PPOTrainer, metrics: dict[str, float]) -> None:
        if not self.use_eval:
            return
        rate = metrics.get("eval/success_rate")
        if rate is None:
            return
        if rate >= trainer.cfg.success_threshold:
            self.streak += 1
            LOGGER.info(
                "Eval success %.1f%% >= threshold %.1f%% (%d/%d consecutive)",
                rate * 100, trainer.cfg.success_threshold * 100,
                self.streak, self.consecutive,
            )
        else:
            self.streak = 0
        if self.streak >= self.consecutive:
            self.should_stop = True
            self.stopped_at_step = trainer.global_step

    def on_iteration_end(self, trainer: PPOTrainer, metrics: dict[str, float]) -> bool:
        return not self.should_stop


class ProgressCallback(TrainerCallback):
    """Console progress with throughput and a wall-clock ETA."""

    def __init__(self, *, every_n_iterations: int = 10) -> None:
        self.every = max(1, int(every_n_iterations))
        self.started = 0.0

    def on_training_start(self, trainer: PPOTrainer) -> None:
        self.started = time.perf_counter()
        LOGGER.info(
            "Target %s steps in %d iterations of %d steps",
            f"{trainer.cfg.total_timesteps:,}",
            trainer.num_iterations,
            trainer.cfg.batch_size(trainer.num_envs),
        )

    def on_iteration_end(self, trainer: PPOTrainer, metrics: dict[str, float]) -> None:
        if trainer.iteration % self.every:
            return
        elapsed = time.perf_counter() - self.started
        sps = trainer.global_step / max(elapsed, 1e-9)
        remaining = max(0, trainer.cfg.total_timesteps - trainer.global_step)
        eta_min = remaining / max(sps, 1e-9) / 60.0
        success = metrics.get("rollout/success_rate")
        LOGGER.info(
            "[%5.1f%%] step %s/%s | %.0f sps | elapsed %.1f min | eta %.1f min%s",
            100.0 * trainer.global_step / max(trainer.cfg.total_timesteps, 1),
            f"{trainer.global_step:,}",
            f"{trainer.cfg.total_timesteps:,}",
            sps,
            elapsed / 60.0,
            eta_min,
            "" if success is None else f" | train success {success:.1%}",
        )

    def on_training_end(self, trainer: PPOTrainer) -> None:
        elapsed = time.perf_counter() - self.started
        LOGGER.info(
            "Finished %s steps in %.1f min (%.0f sps)",
            f"{trainer.global_step:,}", elapsed / 60.0,
            trainer.global_step / max(elapsed, 1e-9),
        )


class MetricHistory(TrainerCallback):
    """Keep metrics in memory and optionally dump them to JSONL.

    Useful for plotting a finished run without parsing TensorBoard event files.

    **The file is prepared in ``on_training_start``, never in ``__init__``.** That
    ordering is the entire safety property, and it is worth stating plainly
    because the obvious placement is the wrong one: a launcher builds its whole
    callback list while it is still deciding whether to train at all, so
    truncation done at construction lands on runs that never produce a
    replacement. ``scripts/train.py --dry-run`` constructs these callbacks and
    then exits before ``train()``; the previous ``write_text("")`` here emptied a
    finished 1.5M-step run's history on the way past, and the loss was only
    recoverable because TensorBoard happened to hold the same scalars. Deferring
    makes ``--dry-run`` safe structurally, rather than by a flag that each new
    caller has to remember to thread through.

    ``mode`` decides what happens when the target file already holds records:

    * ``"auto"`` (default) -- append when the trainer starts with steps already on
      the clock (a resume), rotate an existing history aside otherwise. Never
      truncates and never discards.
    * ``"append"`` -- always append. Correct for ``--resume``, whose steps extend
      the existing series monotonically.
    * ``"rotate"`` -- always move an existing file aside first.
    * ``"overwrite"`` -- truncate. The old behaviour, now opt-in only.

    Rotation rather than blind appending for fresh runs because
    ``scripts/plot_results.py`` sorts records by ``global_step``: appending an 8k
    step ``--smoke`` run onto a finished 1.5M one leaves a perfectly valid file
    whose 32 early rows sort into the front of the real curve, which renders as an
    unexplained collapse instead of as an obvious mistake. Preserving history and
    keeping the active file single-series are both requirements, and only rotation
    satisfies both.
    """

    MODES = ("auto", "append", "rotate", "overwrite")

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        mode: Literal["auto", "append", "rotate", "overwrite"] = "auto",
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.path = Path(path) if path else None
        self.mode = mode
        self.records: list[dict[str, Any]] = []
        self.archived_to: Path | None = None
        self._prepared = False
        # Deliberately no filesystem access here -- not even mkdir. See above.

    # -- file preparation, deferred until training actually begins ---------- #

    def _iter_existing(self) -> Any:
        """Yield parsed records already in the target file, skipping junk lines."""
        if self.path is None or not self.path.is_file():
            return
        with self.path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A run killed mid-write leaves a half line. It still proves
                    # the file holds history worth keeping, which is all that
                    # matters here, so count it rather than crash on it.
                    yield {}

    def _existing_summary(self) -> tuple[int, int | None]:
        """``(record count, highest global_step)`` for the current file."""
        count = 0
        last: int | None = None
        for record in self._iter_existing():
            count += 1
            step = record.get("global_step")
            if isinstance(step, (int, float)):
                last = max(last or 0, int(step))
        return count, last

    def _rotate(self, last_step: int | None) -> Path:
        """Move the existing history aside; return where it landed.

        Named by the last step it contains rather than by a timestamp, so the
        archive is self-describing and two rotations of different runs cannot
        collide on a same-second name.
        """
        assert self.path is not None
        base = (
            f"{self.path.stem}.upto_{last_step}"
            if last_step is not None
            else f"{self.path.stem}.previous"
        )
        target = self.path.with_name(f"{base}{self.path.suffix}")
        attempt = 2
        while target.exists():
            target = self.path.with_name(f"{base}-{attempt}{self.path.suffix}")
            attempt += 1
        self.path.rename(target)
        return target

    def _prepare(self, *, start_step: int | None) -> None:
        """Resolve ``mode`` against what is on disk. Idempotent.

        ``start_step`` is the trainer's step count at the moment training begins:
        ``0`` for a fresh run, the checkpoint's step for a resume, and ``None``
        when the caller cannot tell (see ``on_iteration_end``).
        """
        if self.path is None or self._prepared:
            return
        self._prepared = True
        self.path.parent.mkdir(parents=True, exist_ok=True)

        mode = self.mode
        if mode == "auto":
            # Only a genuinely fresh run rotates. Both a resume and an unknown
            # provenance append, that being the one outcome which neither
            # truncates nor moves an existing file.
            mode = "rotate" if start_step == 0 else "append"

        count, last_step = self._existing_summary()

        if mode == "overwrite":
            if count:
                LOGGER.warning(
                    "Truncating %d existing record(s) in %s (mode='overwrite')",
                    count, self.path,
                )
            self.path.write_text("")
            return

        if not count:
            return  # nothing on disk to protect; plain append starts it

        # Appending is only coherent when the incoming run extends the series. A
        # resume from an early checkpoint of a longer run restarts *behind* the
        # recorded history, and appending there sorts the replayed steps into the
        # middle of the old curve -- the same silent corruption rotation exists to
        # avoid, so fall back to rotating.
        if (
            mode == "append"
            and start_step is not None
            and last_step is not None
            and start_step < last_step
        ):
            LOGGER.warning(
                "Resuming at step %s but %s already records up to step %s; "
                "appending would interleave two series, so the existing history "
                "is being preserved aside instead",
                f"{start_step:,}", self.path.name, f"{last_step:,}",
            )
            mode = "rotate"

        if mode == "rotate":
            self.archived_to = self._rotate(last_step)
            LOGGER.info(
                "Preserved %d existing metric record(s) as %s; starting a fresh %s",
                count, self.archived_to.name, self.path.name,
            )
        else:
            LOGGER.info(
                "Appending to %d existing metric record(s) in %s (last step %s)",
                count, self.path.name, f"{last_step:,}" if last_step else "unknown",
            )

    def on_training_start(self, trainer: PPOTrainer) -> None:
        # global_step is non-zero only if a checkpoint was loaded before train(),
        # so it identifies a resume -- and where it resumes from -- without the
        # callback needing a launcher flag at all.
        self._prepare(start_step=int(trainer.global_step))

    def on_iteration_end(self, trainer: PPOTrainer, metrics: dict[str, float]) -> None:
        # Defensive: CallbackList always fires on_training_start first, but a
        # direct user of this callback might not. global_step has already advanced
        # by now, so it cannot identify a resume; None keeps this path append-only.
        self._prepare(start_step=None)

        record = {"global_step": trainer.global_step, "iteration": trainer.iteration}
        record.update(
            {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (bool, int, float, np.floating, np.integer))
            }
        )
        self.records.append(record)
        if self.path:
            with self.path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
