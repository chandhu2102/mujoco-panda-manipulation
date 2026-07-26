#!/usr/bin/env python3
"""Passive viewer: watch a trained policy act, and report the metrics it earns.

On macOS the interactive viewer must run under ``mjpython``, not ``python``:

    mjpython scripts/enjoy.py outputs/reach_1p5m
    mjpython scripts/enjoy.py outputs/reach_1p5m --checkpoint best.pt --episodes 5
    mjpython scripts/enjoy.py outputs/reach_1p5m --stochastic --speed 0.5

Headless (no window, metrics only) runs under plain python:

    python scripts/enjoy.py outputs/reach_1p5m --no-viewer --episodes 20

Headless with an mp4 written to disk -- also plain python, no window server needed:

    python scripts/enjoy.py outputs/reach_1p5m --video rollout.mp4 --episodes 5

The env and policy are rebuilt from the run's ``config.resolved.json``, so the
architecture always matches the checkpoint rather than relying on flags being
re-typed consistently. Observation-normalizer statistics are loaded from the same
run and held frozen -- a policy evaluated against different input scaling than it
trained on is not the policy you trained.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the launcher's builders so the env/policy construction cannot drift from
# how they were built during training.
from train import TASKS, build_policy, make_env_fn, resolve_device  # noqa: E402

from mujoco_manip.algos.common.normalizers import RunningMeanStd  # noqa: E402
from mujoco_manip.training.callbacks import NormalizerCheckpoint  # noqa: E402

LOGGER = logging.getLogger("enjoy")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass
class EpisodeRecord:
    """One finished episode."""

    index: int
    ret: float
    length: int
    success: bool
    final_distance: float | None


@dataclass
class MetricAccumulator:
    """Running totals for the current episode, plus every finished one."""

    episodes: list[EpisodeRecord] = field(default_factory=list)
    ep_return: float = 0.0
    ep_length: int = 0
    ep_success: bool = False
    ep_distance: float | None = None
    total_steps: int = 0

    _DISTANCE_KEYS = ("dist/eef_to_goal", "dist/object_to_goal", "dist/eef_to_object")

    def accumulate(self, reward: float, info: dict[str, Any]) -> None:
        self.ep_return += float(reward)
        self.ep_length += 1
        self.total_steps += 1
        # is_success latches inside the env, but OR-ing here keeps this correct
        # even against an env that reports only the instantaneous flag.
        self.ep_success = self.ep_success or bool(info.get("is_success", False))
        for key in self._DISTANCE_KEYS:
            if key in info:
                self.ep_distance = float(info[key])
                break

    def close_episode(self) -> EpisodeRecord:
        record = EpisodeRecord(
            index=len(self.episodes) + 1,
            ret=self.ep_return,
            length=self.ep_length,
            success=self.ep_success,
            final_distance=self.ep_distance,
        )
        self.episodes.append(record)
        self.ep_return, self.ep_length = 0.0, 0
        self.ep_success, self.ep_distance = False, None
        return record

    def summary(self) -> dict[str, float]:
        if not self.episodes:
            return {}
        returns = np.array([e.ret for e in self.episodes], dtype=np.float64)
        lengths = np.array([e.length for e in self.episodes], dtype=np.float64)
        successes = np.array([e.success for e in self.episodes], dtype=np.float64)
        rate = float(successes.mean())
        out = {
            "episodes": float(len(self.episodes)),
            "success_rate": rate,
            "success_rate_stderr": float(
                np.sqrt(max(rate * (1.0 - rate), 0.0) / len(self.episodes))
            ),
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "length_mean": float(lengths.mean()),
        }
        distances = [e.final_distance for e in self.episodes if e.final_distance is not None]
        if distances:
            out["final_distance_mean"] = float(np.mean(distances))
            out["final_distance_max"] = float(np.max(distances))
        return out


# --------------------------------------------------------------------------- #
# Offscreen recording
# --------------------------------------------------------------------------- #


class Recorder:
    """Offscreen frame capture to mp4/gif, independent of the interactive viewer.

    Owns its own ``mujoco.Renderer`` rather than calling ``env.render()``: the env's
    renderer is fixed at 480x640 and reads ``env.camera_name``, so recording through
    it would mean mutating the env the policy is being evaluated in. Offscreen
    rendering uses the CGL backend on macOS and needs no window server, so this path
    works under plain ``python`` -- no ``mjpython``, no display.
    """

    def __init__(
        self,
        model: Any,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        camera: str | None = None,
    ) -> None:
        import mujoco

        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
            raise SystemExit(
                "--video needs an encoder. Install it into this venv:\n"
                "    ./venv/bin/pip install 'imageio[ffmpeg]'"
            ) from exc

        # Resolved to an id here so a typo fails before the rollout runs rather
        # than after N episodes of physics.
        self.camera: int | str = -1  # -1 is the free camera
        if camera:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if cam_id < 0:
                raise SystemExit(f"no camera named {camera!r} in the model")
            self.camera = cam_id

        self.path = path
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(path), fps=fps)
        self.fps = fps
        self.frames = 0

    def capture(self, data: Any) -> None:
        self.renderer.update_scene(data, camera=self.camera)
        self.writer.append_data(self.renderer.render())
        self.frames += 1

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()
        LOGGER.info(
            "video       %s (%d frames, %.1f fps, %.1fs)",
            self.path, self.frames, self.fps, self.frames / max(self.fps, 1e-6),
        )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def resolve_checkpoint(run_dir: Path, requested: str | None) -> Path:
    """Pick a checkpoint file, preferring the best-scoring one."""
    ckpt_dir = run_dir / "checkpoints"
    if requested:
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = ckpt_dir / requested
        if not candidate.is_file():
            raise SystemExit(f"checkpoint not found: {candidate}")
        return candidate
    for name in ("best.pt", "final.pt", "interrupted.pt"):
        if (ckpt_dir / name).is_file():
            return ckpt_dir / name
    stepped = sorted(
        ckpt_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else -1,
    )
    if stepped:
        return stepped[-1]
    raise SystemExit(f"no checkpoints under {ckpt_dir}")


def load_run(run_dir: Path, args: argparse.Namespace) -> tuple[Any, Any, RunningMeanStd | None, Path]:
    """Rebuild env + policy + normalizer from a run directory."""
    resolved = run_dir / "config.resolved.json"
    if not resolved.is_file():
        raise SystemExit(
            f"{resolved} not found -- pass a run directory produced by scripts/train.py"
        )
    config = json.loads(resolved.read_text())

    env_cfg = dict(config.get("env", {}))
    policy_cfg = dict(config.get("policy", {}))
    norm_cfg = dict(config.get("normalize_obs", {}))
    task = env_cfg.get("task", "reach")
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r} in config; have {sorted(TASKS)}")

    if args.max_steps is not None:
        env_cfg["max_episode_steps"] = args.max_steps
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    env = make_env_fn(env_cfg, seed, 0)()

    device = resolve_device(args.device or config.get("device", "cpu"))
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    policy = build_policy(policy_cfg, obs_dim, action_dim).to(device)

    checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("policy") is None:
        raise SystemExit(f"{checkpoint} holds no policy weights")
    policy.load_state_dict(payload["policy"])
    policy.eval()

    obs_rms: RunningMeanStd | None = None
    if norm_cfg.get("enabled", True) and not args.no_normalizer:
        obs_rms = RunningMeanStd((obs_dim,))
        if NormalizerCheckpoint(obs_rms, run_dir / "checkpoints").load():
            LOGGER.info("normalizer  count=%.0f (frozen)", obs_rms.count)
        else:
            # Silently continuing with identity scaling would make the policy look
            # broken for reasons unrelated to the policy.
            LOGGER.warning(
                "normalizer statistics missing from %s; the run trained WITH "
                "normalization, so behaviour will not match. Pass "
                "--no-normalizer to acknowledge this.", run_dir / "checkpoints",
            )
            obs_rms = None

    LOGGER.info("run         %s", run_dir)
    LOGGER.info("checkpoint  %s (step %s)", checkpoint.name,
                f"{int(payload.get('global_step', 0)):,}")
    LOGGER.info("task        %s | obs %d | action %d | device %s",
                task, obs_dim, action_dim, device)
    LOGGER.info("policy      %s deterministic",
                "stochastic" if args.stochastic else "")
    return env, policy, obs_rms, device


def prepare_obs(
    obs: np.ndarray, obs_rms: RunningMeanStd | None, clip: float, device: torch.device
) -> torch.Tensor:
    array = np.asarray(obs, dtype=np.float32)
    if obs_rms is not None:
        array = obs_rms.normalize(array, clip=clip)
    return torch.as_tensor(array, dtype=torch.float32, device=device).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #


def run(
    env: Any,
    policy: Any,
    obs_rms: RunningMeanStd | None,
    device: torch.device,
    args: argparse.Namespace,
    viewer: Any | None,
    recorder: Recorder | None = None,
) -> MetricAccumulator:
    """Drive the policy, syncing the viewer and auto-resetting on episode end."""
    metrics = MetricAccumulator()
    deterministic = not args.stochastic
    clip = float(args.obs_clip)
    # Through `.unwrapped`: `make_env_fn` may return a wrapper (task-space control
    # wraps the env), and gymnasium's Wrapper no longer forwards unknown attribute
    # lookups to the wrapped env, so `env.control_dt` raises rather than resolving.
    frame_dt = env.unwrapped.control_dt / max(args.speed, 1e-6)

    obs, info = env.reset(seed=args.seed)
    if viewer is not None:
        viewer.sync()
    if recorder is not None:
        recorder.capture(env.unwrapped.data)
    next_frame = time.perf_counter()

    with torch.no_grad():
        while True:
            if viewer is not None and not viewer.is_running():
                LOGGER.info("viewer closed")
                break
            if args.episodes and len(metrics.episodes) >= args.episodes:
                break
            if args.max_total_steps and metrics.total_steps >= args.max_total_steps:
                LOGGER.info("reached --max-total-steps")
                break

            obs_tensor = prepare_obs(obs, obs_rms, clip, device)
            action, _, _ = policy.act(obs_tensor, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(
                action.squeeze(0).cpu().numpy()
            )

            metrics.accumulate(reward, info)

            if recorder is not None:
                recorder.capture(env.unwrapped.data)

            if viewer is not None:
                viewer.sync()
                # Pace to wall clock so the motion is watchable rather than a blur.
                next_frame += frame_dt
                delay = next_frame - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_frame = time.perf_counter()

            if terminated or truncated:
                record = metrics.close_episode()
                LOGGER.info(
                    "episode %3d | %-7s | return %8.2f | %3d steps | %s | final_dist %s",
                    record.index,
                    "SUCCESS" if record.success else "failure",
                    record.ret,
                    record.length,
                    "terminated" if terminated else "truncated",
                    "n/a" if record.final_distance is None
                    else f"{record.final_distance:.4f} m",
                )
                obs, info = env.reset()
                if recorder is not None:
                    recorder.capture(env.unwrapped.data)
                if viewer is not None:
                    viewer.sync()
                    next_frame = time.perf_counter()

    return metrics


def report(metrics: MetricAccumulator) -> None:
    summary = metrics.summary()
    if not summary:
        LOGGER.info("no episodes completed")
        return
    LOGGER.info("-" * 62)
    LOGGER.info(
        "%d episodes | success %.1f%% +/- %.1f%%",
        int(summary["episodes"]), summary["success_rate"] * 100,
        summary["success_rate_stderr"] * 100,
    )
    LOGGER.info(
        "return %.2f +/- %.2f | mean length %.1f steps",
        summary["return_mean"], summary["return_std"], summary["length_mean"],
    )
    if "final_distance_mean" in summary:
        LOGGER.info(
            "final distance mean %.4f m | worst %.4f m",
            summary["final_distance_mean"], summary["final_distance_max"],
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Accepted either positionally or as --run-dir; both spellings are natural to
    # reach for, and a usage error here costs a viewer launch to discover.
    parser.add_argument("run_dir", type=Path, nargs="?", default=None,
                        help="run directory from scripts/train.py")
    parser.add_argument("--run-dir", dest="run_dir_flag", type=Path, default=None,
                        help="same as the positional run_dir")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint filename or path (default: best.pt)")
    parser.add_argument("--episodes", type=int, default=0,
                        help="stop after N episodes (0 = until the window closes)")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions instead of using the policy mean")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback rate; 0.5 is half speed")
    parser.add_argument("--seed", type=int, default=None, help="seed for the first reset")
    parser.add_argument("--camera", default=None,
                        help="camera name to start the viewer on")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override the env episode horizon")
    parser.add_argument("--max-total-steps", type=int, default=0,
                        help="hard cap on total steps (0 = unlimited)")
    parser.add_argument("--obs-clip", type=float, default=10.0)
    parser.add_argument("--no-normalizer", action="store_true",
                        help="skip loading normalizer statistics")
    parser.add_argument("--no-viewer", action="store_true",
                        help="headless: collect metrics with no window")
    parser.add_argument("--video", type=Path, default=None,
                        help="render the rollout to this .mp4/.gif; implies --no-viewer")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=float, default=None,
                        help="default: the env's control rate scaled by --speed")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(message)s",
    )

    if args.run_dir and args.run_dir_flag:
        raise SystemExit("pass the run directory once, positionally or via --run-dir")
    target = args.run_dir or args.run_dir_flag
    if target is None:
        parser.error("a run directory is required (positionally or via --run-dir)")
    args.run_dir = target

    run_dir = target if target.is_absolute() else _REPO_ROOT / target
    if not run_dir.is_dir():
        raise SystemExit(f"run directory not found: {run_dir}")

    if args.video:
        # Recording is offscreen, so a window would only add an mjpython
        # requirement to a path whose whole point is not needing one.
        if not args.no_viewer:
            LOGGER.info("--video renders offscreen; running headless")
            args.no_viewer = True
        if not args.episodes:
            # Without a stop condition the writer would grow the file until the
            # process is killed, and a truncated mp4 has no moov atom to play.
            args.episodes = 5
            LOGGER.info("--video with no --episodes: recording %d episodes", args.episodes)

    if not args.no_viewer and not args.episodes:
        LOGGER.info("running until the viewer window is closed (Ctrl-C also works)")

    env, policy, obs_rms, device = load_run(run_dir, args)

    recorder = None
    if args.video:
        fps = args.video_fps or (1.0 / env.unwrapped.control_dt) * max(args.speed, 1e-6)
        recorder = Recorder(
            env.unwrapped.model,
            args.video if args.video.is_absolute() else Path.cwd() / args.video,
            width=args.video_width,
            height=args.video_height,
            fps=fps,
            camera=args.camera,
        )

    viewer_handle = None
    try:
        if args.no_viewer:
            metrics = run(env, policy, obs_rms, device, args, None, recorder)
        else:
            import mujoco.viewer

            try:
                viewer_handle = mujoco.viewer.launch_passive(
                    env.unwrapped.model, env.unwrapped.data,
                    show_left_ui=False, show_right_ui=False,
                )
            except RuntimeError as exc:
                if "mjpython" in str(exc):
                    raise SystemExit(
                        "The passive viewer needs mjpython on macOS. Re-run:\n"
                        f"    mjpython {' '.join(sys.argv)}\n"
                        "Or collect metrics without a window: add --no-viewer"
                    ) from exc
                raise
            with viewer_handle as viewer:
                if args.camera:
                    import mujoco

                    cam_id = mujoco.mj_name2id(
                        env.unwrapped.model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera
                    )
                    if cam_id < 0:
                        raise SystemExit(f"no camera named {args.camera!r} in the model")
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = cam_id
                metrics = run(env, policy, obs_rms, device, args, viewer)
    except KeyboardInterrupt:
        LOGGER.info("interrupted")
        return 130
    finally:
        # Closed in `finally` so a Ctrl-C still finalizes the container -- an mp4
        # whose writer never closed is missing its moov atom and will not play.
        if recorder is not None:
            recorder.close()
        env.close()

    report(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
