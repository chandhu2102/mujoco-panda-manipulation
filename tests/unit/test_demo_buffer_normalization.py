"""Regression tests for ``DemoBuffer``'s observation whitening.

These exist because of a silent defect that cost a full training run. ``obs_rms``
is a live object the caller mutates *after* the buffer is constructed --
``NormalizeObservation`` updates it from rollouts, and ``scripts/train.py`` loads
a checkpoint's statistics into it 37 lines after building the buffer. The buffer
used to whiten eagerly in ``__init__``, so it froze a snapshot of whatever the
statistics happened to be at that moment: on the ``--resume`` path, an unfitted
normalizer, i.e. no whitening at all.

Nothing raised. The DAPG anchor simply scored the policy against demonstrations
in the wrong space, contributing 41x the intended gradient and dominating an
actor whose policy loss is of order 1e-2. An 87%-success warm start fell to 2%
inside 40 iterations. The only visible symptom was ``train/bc_loss`` sitting at
~120 where ``scripts/pretrain_bc.py`` had just reported -4.27 on the same data --
which is exactly the kind of thing a test asserts and a human skims past.

Runnable two ways, because this repo has no test runner installed yet::

    python -m pytest tests/unit/test_demo_buffer_normalization.py
    python tests/unit/test_demo_buffer_normalization.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mujoco_manip.algos.common.normalizers import RunningMeanStd  # noqa: E402
from mujoco_manip.training.bc import DemoBuffer  # noqa: E402

OBS_DIM = 6
ACTION_DIM = 3
CLIP = 10.0


def _write_demos(path: Path, n_episodes: int = 4, length: int = 25) -> np.ndarray:
    """A tiny synthetic demonstration file. Returns the raw observations.

    Synthetic rather than a fixture copied from ``demos/``: the real sets are
    megabytes and git-ignored, and nothing here depends on their content -- only
    on the whitening applied to whatever is in them. Per-dimension scales are
    deliberately spread over three orders of magnitude, since the bug's damage
    was proportional to how far the statistics were from identity.

    Written as float32, which is what ``scripts/record_demos.py`` produces and
    therefore what the whitening contract is exercised against in practice. The
    distinction matters for the exactness assertions below: ``DemoBuffer`` holds
    observations at float32 to match the policy's inputs, so a float64 source
    would be rounded once on load and then whitened in float64, while
    ``RunningMeanStd.normalize`` would whiten the unrounded original -- a ~1e-7
    relative disagreement that is real but has no bearing on any real demo file.
    """
    rng = np.random.default_rng(0)
    n = n_episodes * length
    scales = np.array([1e-2, 1e-1, 1.0, 5.0, 2e-2, 3.0])
    offsets = np.array([0.5, -2.0, 0.0, 10.0, -0.03, 1.0])
    raw_obs = (rng.normal(size=(n, OBS_DIM)) * scales + offsets).astype(np.float32)
    np.savez_compressed(
        path,
        observations=raw_obs,
        actions=rng.uniform(-1, 1, (n, ACTION_DIM)).astype(np.float32),
        rewards=rng.normal(size=n).astype(np.float64),
        episode_starts=(np.arange(n_episodes) * length).astype(np.int64),
        episode_lengths=np.full(n_episodes, length, dtype=np.int64),
        control_mode=np.array("joint_position"),
        task_space=np.array(True),
    )
    return raw_obs.astype(np.float64)


def _buffer(path: Path, obs_rms: RunningMeanStd | None) -> DemoBuffer:
    return DemoBuffer(
        path, device=torch.device("cpu"), obs_rms=obs_rms, clip=CLIP, gamma=0.995
    )


def test_whitening_tracks_obs_rms_mutated_after_construction() -> None:
    """The regression itself: build with empty statistics, fit them afterwards.

    This is ``scripts/train.py``'s ordering on ``--resume``. The buffer must
    reflect the statistics as they are *when read*, not as they were when it was
    built.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demos.npz"
        raw_obs = _write_demos(path)

        obs_rms = RunningMeanStd((OBS_DIM,))
        buf = _buffer(path, obs_rms)

        # Unfitted: normalize passes through (mean 0, std 1 via var_floor).
        before = buf.observations.numpy()
        assert np.allclose(before, np.clip(raw_obs, -CLIP, CLIP), atol=1e-6), (
            "an unfitted normalizer should leave observations essentially unchanged"
        )

        obs_rms.update(raw_obs)
        after = buf.observations.numpy()

        assert not np.allclose(before, after), (
            "buffer ignored statistics fitted after construction -- this is the "
            "eager-caching bug that silently unwhitened the DAPG anchor"
        )
        assert np.array_equal(after, obs_rms.normalize(raw_obs, clip=CLIP)), (
            "whitening disagrees with RunningMeanStd.normalize after the update"
        )
        # Whitened data should be roughly standardized; the raw data is not.
        assert abs(float(after.mean())) < 0.05
        assert 0.8 < float(after.std()) < 1.25


def test_whiten_matches_running_mean_std_at_every_count() -> None:
    """Parity with ``RunningMeanStd.normalize``, including its ``count < 2`` branch.

    The two must agree exactly: the policy is scored on demonstrations whitened by
    the buffer and acts on rollouts whitened by the wrapper, so any divergence is
    a silent difference between training inputs and acting inputs. ``count == 1``
    is unreachable in the live pipeline -- the vector env updates a whole batch at
    once, so the count steps 0 -> num_envs -- which is precisely why a mismatch
    there would never have surfaced on its own.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demos.npz"
        raw_obs = _write_demos(path)
        buf = _buffer(path, RunningMeanStd((OBS_DIM,)))

        for count in (0, 1, 2, 16, len(raw_obs)):
            rms = RunningMeanStd((OBS_DIM,))
            if count:
                rms.update(raw_obs[:count])
            buf.obs_rms = rms
            expected = rms.normalize(raw_obs, clip=CLIP)
            assert np.array_equal(buf.observations.numpy(), expected), (
                f"whitening diverges from RunningMeanStd.normalize at count={count}"
            )


def test_sample_and_epochs_return_whitened_rows() -> None:
    """Both batch paths must whiten, not just the ``observations`` property.

    ``sample`` feeds the DAPG anchor and ``epochs`` feeds ``BCPretrainer``; the bug
    would have been equally invisible in either.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demos.npz"
        raw_obs = _write_demos(path)
        obs_rms = RunningMeanStd((OBS_DIM,))
        obs_rms.update(raw_obs)
        buf = _buffer(path, obs_rms)

        whitened = buf.observations.numpy()

        obs, actions, returns = buf.sample(32)
        assert obs.shape == (32, OBS_DIM)
        assert actions.shape == (32, ACTION_DIM)
        assert returns.shape == (32,)
        # Every sampled row must be some row of the whitened set.
        for row in obs.numpy():
            assert np.isclose(whitened, row).all(axis=1).any(), (
                "sample() returned a row that is not in the whitened observations"
            )

        seen = 0
        for batch_obs, _, _ in buf.epochs(16, 1):
            for row in batch_obs.numpy():
                assert np.isclose(whitened, row).all(axis=1).any(), (
                    "epochs() yielded a row that is not in the whitened observations"
                )
            seen += batch_obs.shape[0]
        assert seen > 0


def test_clip_is_applied() -> None:
    """Whitened values stay inside +/- clip, which bounds a blown-up observation."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demos.npz"
        raw_obs = _write_demos(path)
        rms = RunningMeanStd((OBS_DIM,))
        rms.update(raw_obs[:8])  # deliberately narrow, so later rows sit far out
        buf = _buffer(path, rms)
        out = buf.observations.numpy()
        assert np.isfinite(out).all()
        assert float(np.abs(out).max()) <= CLIP + 1e-6


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}\n     {exc}")
        else:
            print(f"ok   {name}")
    print("\n" + ("all passed" if not failures else f"{failures} failed"))
    raise SystemExit(1 if failures else 0)
