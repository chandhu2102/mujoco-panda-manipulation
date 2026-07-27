# mujoco-panda-manipulation

Reinforcement learning for a Franka Panda arm in MuJoCo — reaching, pick-and-place, and
pick-and-place across a barrier the policy is penalized for touching.

The interesting part of this repo is not the PPO implementation. It is the record of what
*didn't* work: the same task, the same reward, and the same 10M-step budget produce a 0%
policy under joint-torque control and a 95% one under operational-space control warm-started
from demonstrations. Most of the design notes in the source explain a measurement, not a
preference.

**[▶ Watch the obstacle policy](docs/rollouts.html)** — open locally, or
`open docs/rollouts.html`.

---

## Results

| Task | Control | Steps | Train success | Eval |
|---|---|---:|---:|---:|
| `reach` | torque | 1.50 M | **100%** | 100% |
| `pick_place` | torque | 6.39 M | 0% | 5% |
| `pick_place` | velocity | 10.0 M | 16% | 15% |
| `pick_place` | OSC + 400 demos | 10.0 M | **95%** | 100% |
| `pick_place_obstacle` | OSC + 400 demos | 3.68 M | **95%** | 100% |

**Read the eval column with care.** Every number in it comes from the training loop's own
evaluation, which runs **20 episodes** — enough to report 20/20 while the true rate is 95%.
The one task measured properly, over 100 deterministic episodes on held-out seeds, is
`pick_place_obstacle`:

```
success 95.0% ± 2.2    grasp 99%    touched the barrier 3%    median final error 20 mm
```

Behavioural cloning alone reaches 87% there; PPO carries it the rest of the way. Treat the
`pick_place` OSC row as "somewhere in the low-to-mid nineties" rather than solved, and re-run
it at `eval.n_episodes: 100` if the exact figure matters.

## What made the difference

Three findings, each of which cost a full training run to establish.

**Torque control cannot find the grasp.** Zero torque coasts under gravity compensation, so
the map from actions to tool-path is a double integration and zero-mean exploration noise is
an unbounded random walk in position. A 6.39 M-step run finished at 0% success; on a separate
3 M-step run the grasp rate was 0.0% for the first 2.6 M steps. With an honest grasp
predicate there is no partial credit for nearly grasping, so until that first grasp lands
every later reward stage is unreachable and the value function has nothing to propagate.
Velocity control fixes the integration order and reaches 16%, still short.

**Operational-space control plus demonstrations does.** `TaskSpaceWrapper` turns the action
into a 4-D tool-centre-point delta and closes an IK loop underneath, which is a space a
scripted expert can also act in — so the same 400 episodes serve as both a BC warm start and
a DAPG anchor during PPO.

**A reward is only as honest as its predicate.** An earlier 3 M-step run converged on
pressing the closed fingertips against the side of the cube: both pads reported contact and a
jaw width near zero trivially satisfied `width < object_width + tol`, so the grasp check
passed on 93% of steps in episodes where the cube never rose more than 1.9 cm off the table.
That spoof accounted for 96% of all grasp reward the final policy collected.
`ManipulationEnv.is_grasped` now requires a width *band* and geometric enclosure, not just
contact.

## Quickstart

Requires Python 3.14, MuJoCo 3.10, PyTorch 2.13, Gymnasium 1.3. There is no
`pyproject.toml` yet — the `scripts/` launchers put `src/` on the path themselves.

```bash
python -m venv venv && source venv/bin/activate
pip install mujoco torch numpy gymnasium imageio imageio-ffmpeg

# Train the task that works, from scratch (~1.5 M steps, no demonstrations needed)
python scripts/train.py --config configs/train/reach.yaml

# Watch a trained policy
python scripts/enjoy.py outputs/reach_1p5m --episodes 5
python scripts/enjoy.py outputs/reach_1p5m --episodes 5 --video rollout.mp4  # headless
```

Any config key can be overridden without editing the file, and the override is recorded in
the run's `config.resolved.json`:

```bash
python scripts/train.py --config configs/train/reach.yaml --set ppo.gamma=0.99
python scripts/train.py --config configs/train/reach.yaml --smoke   # 8k steps, isolated dir
```

## The demonstration pipeline

Pick-and-place needs all three stages. Each reads the same config, so the action space the
expert records in is the one the policy trains in.

```bash
# 1. Scripted expert. Only episodes that actually pick the object are kept.
python scripts/record_demos.py --config configs/train/pick_place_obstacle.yaml \
    --episodes 400 --out demos/pick_place_obstacle_osc_400.npz

# 2. Clone it. Also fits the observation normalizer and pre-trains the value head.
python scripts/pretrain_bc.py --config configs/train/pick_place_obstacle.yaml \
    --demos demos/pick_place_obstacle_osc_400.npz --out outputs/pick_place_obstacle

# 3. PPO from the clone, with a decaying BC term anchoring it (DAPG-style).
python scripts/train.py --config configs/train/pick_place_obstacle.yaml \
    --resume outputs/pick_place_obstacle
```

Demonstrations are *not* interchangeable between tasks. `record_demos.py` reads the scene
geometry, so the obstacle set crosses the barrier at altitude in both directions and discards
any episode that touches it; the obstacle-free set drives straight through where the wall now
stands. Both are 48/4, so nothing downstream catches the mix-up — see the `bc:` block in
`configs/train/pick_place_obstacle.yaml`.

## Tasks

| Task | Goal | Reward stages |
|---|---|---|
| `reach` | tool centre point to a 3-D target | reach → hold |
| `pick_place` | cube to a randomized airborne goal | reach → grasp → lift → place |
| `pick_place_obstacle` | same, across a static barrier | the above, plus a collision charge |
| `manipulation` | the base env under its own name, for comparison | generic four-stage |

`pick_place_obstacle` splits the spawn boxes across a 10 cm wall — object beyond it, goal
before it — so every episode carries the cube over. The barrier lives in its own scene file,
included from the base one, so `pick_place` compiles without it (14 geoms against 15) and
stays the task its demonstrations were recorded against. The barrier's top face sits at z 0.50 and
the lowest goal at z 0.50, which means the crossing is *higher* than the target: the cube goes
up over the wall and back down, rather than rising monotonically.

## Layout

```
assets/panda_scene.xml         scene: arm + table + cube + goal marker
assets/panda_scene_obstacle.xml  the same, +1 barrier geom, by <include>
assets/robots/panda/           the arm, with its own actuators and contact classes
src/mujoco_manip/
  envs/manipulation_env.py     base env: reward stages, grasp predicate, curriculum
  envs/tasks/                  one file per task; TASK_REGISTRY binds the strings
  envs/wrappers/task_space.py  OSC: 4-D tool-space delta -> joint targets via IK
  robots/panda.py              index maps, control laws, reset-pose validation
  training/trainer.py          PPO loop, callbacks, checkpointing
  training/bc.py               DemoBuffer, BC pre-training, DAPG anchor
  algos/common/normalizers.py  running mean/std, observation and reward scaling
configs/train/*.yaml           one file per experiment, heavily commented
scripts/                       train, enjoy, record_demos, pretrain_bc, plot_results
tests/unit/                    regression tests
```

Reading order, if you want the substance: `envs/manipulation_env.py` for the reward and the
grasp predicate, `envs/wrappers/task_space.py` for the controller, then the config that
matches the run you care about. The configs carry the experimental record — why a weight is
what it is, and which measurement set it.

## Things worth knowing before changing something

**Success does not terminate the episode, deliberately.** With a per-step dense reward,
ending on success forfeits the remaining steps of it — so unless the success bonus exceeds
that forfeited sum, the return-maximizing behaviour is to hover just outside the threshold
and farm the shaping term. Measured on `reach`: hovering 9 cm out scored 5× an actual
success, and a run duly converged to it while its shaped return kept climbing.

**Returns are unnormalized by default, and it shows.** Episode returns run ~2500 with a value
loss around 1.7e4, so the gradient norm sits near 1000 against a `max_grad_norm` of 10 — both
networks are effectively throttled by clipping, and the critic tracks its target poorly
(`explained_variance` ~0.4). `normalize_reward.enabled` wires `NormalizeReward` in to fix
that; it is off by default, and `rollout/episode_return` stays in raw units either way so runs
remain comparable.

It is not yet usable where it would help most, which is worth knowing before reaching for it.
Warm-started runs are refused, because `pretrain_bc.py` fits the value head on raw discounted
demonstration returns (mean ~425) while normalizing rescales the value target by
~1/std(return) — a critic wrong by orders of magnitude, silently. And from-scratch runs, where
it *is* allowed, do not exhibit the pathology: 40 iterations from scratch have returns near 20
and a value loss near 2, so there is nothing for it to fix (measured: `explained_variance`
+0.49 → +0.71, `grad_norm` 10.5 → 10.3, both already at the clip). Closing that gap means
normalizing the demonstration returns on the same statistic — a change to `DemoBuffer`.

**Curriculum seeding can hurt a cloned policy.** The reverse curriculum hands the policy
states further along the chain, which helps when exploration cannot reach them. But those
states come from an IK solve that knows nothing about the demonstrations: on
`pick_place_obstacle` the clone scores 82% on honest starts and 37% on seeded ones, because a
stage-1 seed is a pre-grasp pose the demonstrations never contain. Hence
`curriculum_level: 0` there and `1` on the obstacle-free baseline.

**Exploration noise is task-specific.** `max_delta_pos` is 12 mm, so sigma 0.3 injects
~3.6 mm per axis per step — a centimetres-wide random walk over an episode. On an open table
that hits nothing; with a 3 cm wall and a 4 cm cube it halves success and quadruples barrier
contact. The obstacle config runs sigma 0.1.

## Known gaps

- No `pyproject.toml`, so `mujoco_manip` is not installable; the CLI needs `PYTHONPATH=src`
  while the `scripts/` launchers do not.
- No test runner is installed. `tests/unit/test_demo_buffer_normalization.py` runs either as
  pytest or directly: `python tests/unit/test_demo_buffer_normalization.py`.
- `eval.n_episodes` is 20 everywhere, which is too few to distinguish 95% from 100%.
- `manipulation` has no default config, so it is reachable only via
  `scripts/train.py --config`, not the task-name CLI.
- `docker/` and `notebooks/` are empty placeholders.
- Reward normalization exists but cannot be combined with a BC warm start (see above).

## Command reference

| Command | Notes |
|---|---|
| `scripts/train.py --config C` | train; `--resume DIR`, `--set K=V`, `--smoke` |
| `scripts/enjoy.py DIR` | replay; `--episodes N`, `--video P`, `--camera C`, `--no-viewer` |
| `scripts/record_demos.py --config C --out P` | scripted demonstrations; `--episodes N` |
| `scripts/pretrain_bc.py --config C --demos P --out DIR` | clone + fit normalizer |
| `scripts/plot_results.py` | curves from `metrics.jsonl` |
| `PYTHONPATH=src python -m mujoco_manip.cli.main train TASK` | train by task name, using its default config |
| `PYTHONPATH=src python -m mujoco_manip.cli.main tasks` | list tasks and their default configs |
| `PYTHONPATH=src python -m mujoco_manip.cli.main check` | validate the task tables without building a sim |

Runs write to `outputs/<run_name>/`: `metrics.jsonl`, `config.resolved.json`, `tb/`, and
`checkpoints/` (`best.pt`, periodic `step_*.pt`, `obs_normalizer.npz`). Note that `best.pt` is
selected on those 20-episode evaluations, so it is not reliably the best policy in a run.

Cameras available for `--camera`: `frontview`, `sideview`, `topview`, `cornerview`, and
`wristview` (eye-in-hand, defined on the hand body in `assets/robots/panda/panda.xml`).
