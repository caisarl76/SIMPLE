# Generate and certify a PSI0 batch

The batch command runs SIMPLE motion-planning generation, validates all four raw
camera streams, converts the head-left camera and state/action columns, and runs
two independent certifications through a pinned PSI0 loader. The default task is
`simple/G1WholebodyBendPickMP-v0`, with 10 successful episodes at each requested
DR level (0, 1, 2 by default).

## Storage

New output defaults use `/mnt/data/jihun/datasets/SIMPLE`. Set the environment
variable **before starting the CLI** to choose another location:

```bash
export SIMPLE_OUTPUT_ROOT=/mnt/data/jihun/datasets/SIMPLE
```

Generation uses `$SIMPLE_OUTPUT_ROOT/datagen`, evaluation uses
`$SIMPLE_OUTPUT_ROOT/evals` (or `evals_decoupled_wbc`), and the batch command creates
an exclusive timestamp/UUID directory under `$SIMPLE_OUTPUT_ROOT/batches`.
Planning, rendering, and teleoperation outputs use corresponding subdirectories.
Existing `--save-dir`, `--eval-dir`, and batch `--output-root` options override these
defaults. Input `--data-dir` defaults are unchanged: explicitly point evaluation
at the intended source dataset. Setting this variable in `.env` alone does not
load it into a running shell.

Raw episodes, processed data, and evidence stay outside Git. The direct converter
still requires an explicit `--out-dir`; create its parent before invoking it.
The batch command creates that parent automatically. Existing datasets are never
overwritten by the batch command. The older direct generation/debug workflows
retain their original behavior; use fresh destinations for those commands.

## Prerequisites

- A working SIMPLE simulation environment with Isaac/MuJoCo, CuRobo, assets, and
  materialized policy weights. The original 30-episode run used
  `.worktrees/psi0-normal-benchmark` at `fd6972d`; the corrected DR verification
  uses `.worktrees/dr-material-verification` at `45f16a8`.
- A Python environment with NumPy, PyArrow, Torch, and the pinned PSI0 loader's
  dependencies for validation/conversion; locally this is `.venv/bin/python`.
- `ffmpeg`, `ffprobe`, and `nvidia-smi` on `PATH`.
- A clean PSI0 checkout and its full commit SHA. Certification verifies that
  identity and uses the loader offline.
- For generation, an idle selected GPU and an existing Isaac EULA acceptance
  (`OMNI_KIT_ACCEPT_EULA=YES`). The command does not accept the EULA itself.

## Preview and run

From the SIMPLE repository, preview the exact commands without writing files or
starting simulation:

```bash
python3 scripts/generate_psi0_batch.py \
  --generator-root .worktrees/dr-material-verification \
  --psi0-root /home/jihun/work/Psi0/.worktrees/simple-dataset-cert-885538e \
  --psi0-commit 885538e0bbee05caa1a89d382653c860596eee95 \
  --gpu 1 --episodes-per-level 10 --levels 0 1 2 \
  --dry-run
```

Remove `--dry-run` to execute using the same options. Use `.venv/bin/python` as
the orchestration interpreter when executing (it imports the final certificate
validator). The generator defaults to `<generator-root>/.venv/bin/python`;
`--generator-python` and `--converter-python` can override the subprocess
interpreters. The default generator checkout is the current repository; the
explicit checkout above includes the verified DR material fix and uses the
established simulation environment. Avoid the older benchmark checkout when
you need fixed materials at DR1/DR2.

Each level runs sequentially with 50 Hz rendering, headless mode, WebRTC disabled,
and a fresh output directory. Only the selected GPU's compute occupancy blocks
startup. The default one-hour timeout applies per command; override it with
`--timeout-seconds`. On interruption or timeout, the command signals and reaps the
child process group. Failures stop subsequent stages and preserve logs and partial
artifacts for inspection. A new invocation allocates a new run; it does not
resume or delete an earlier run.

The prior generation took about 24–25 minutes per ten episodes on the local
RTX 3060. This is an observation from one batch, not a throughput guarantee.

## Validate and convert existing raw episodes

`--raw-root` skips simulation. It expects `drN/simple/ENV/level-N` directories
beneath that root. Selected levels must each contain exactly
`--episodes-per-level` episodes. Other levels are excluded from conversion.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/generate_psi0_batch.py \
  --raw-root /mnt/data/jihun/datasets/SIMPLE/bendpick-batch-30/20260907T061840Z-fd6972d/raw \
  --psi0-root /home/jihun/work/Psi0/.worktrees/simple-dataset-cert-885538e \
  --psi0-commit 885538e0bbee05caa1a89d382653c860596eee95
```

This creates a new converted dataset and evidence. Source raw files remain in
place and their hashes must match before and after conversion/certification.
The thread variables limit CPU parallelism for this small offline traversal.

Each successful run contains:

- `plan.json`: selected paths, counts, and exact commands.
- `provenance.json`: source commits, tracked-diff digests, and workflow hashes.
- `logs/`: command/exit records, raw reports, preflight, conversion, and certification.
- `processed/bendpick-psi0/`: 32-D float32 states, 36-D float32 actions, H.264/yuv420p
  head-left video at 640×360 and 50 FPS.
- Two adjacent certification bundles, each with `PASS.json`.
- `result.json`: counts, hashes, dataset path, and both certificates.

The first 60 frames of every episode are trimmed; there is no downsampling.
Certification traverses every retained index and all episode boundaries, checks
finite image/state/action tensors, and must leave the dataset tree unchanged.
A failed run writes `failure.json` instead of `result.json`.

## Review before scaling

The September 7 batch has 30 episodes, 6,419 raw frames, and 4,619 retained frames.
The sampled visual review covers all episodes and the trim boundary; its contact
sheets and findings live beside the original batch in `review/REVIEW.md`.

All episodes use the same cracker-box target and room. They are useful for a
pipeline/training smoke test, but do not establish broad object or scene coverage.
In those original September 7 recordings, table materials vary at **all three
levels**; level 2 removes distractors. Do not relabel that batch as fixed-material
benchmark data. Preserve its raw recordings and certificates.

Commit `45f16a8` fixes material sampling to read the current configuration and makes
fixed object shader values deterministic, including fixed ground material when
an explicit table material is supplied. Five material regression tests pass
(the old code fails three of them); the combined focused suite passes 22 tests.

The September 8 verification generated three fresh episodes per level in
`/mnt/data/jihun/datasets/SIMPLE/dr-material-verification/20260908T022158Z-622a2d4b`:

| DR level | Material states | Lighting states | Distractors per episode | Retained frames |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3 | 3 | 3 | 464 |
| 1 | 1 | 1 | 3 | 462 |
| 2 | 1 | 1 | 0 | 461 |

Both PSI0 certifications passed all 1,387 retained frames (1,927 raw frames).
`dr-conditions.json` records the parameter checks; `review/REVIEW.md` and its
contact sheets record sampled visual inspection of all nine episodes and four
camera streams. These are verification samples, not evidence of broad object or
scene generalization.
