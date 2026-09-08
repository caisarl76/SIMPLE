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

## Prepare training and run the H100 smoke test

`scripts/prepare_psi0_training_split.py` copies raw episodes into disjoint
`raw/train/` and `raw/val/` trees. Within each source DR directory, the last
episode is held out and the remaining episodes are used for training. It
preserves source episode IDs and environment configurations, checks copied
file hashes, and writes the assignments and source batch provenance to
`split.json`. These sparse raw trees are converter inputs, not training repos;
convert each tree separately to obtain contiguous episode indices and freshly
computed statistics. Existing output directories are rejected.

```bash
python3 scripts/prepare_psi0_training_split.py \
  --batch-root /mnt/data/jihun/datasets/SIMPLE/dr-material-verification/20260908T022158Z-622a2d4b \
  --output-root /mnt/data/jihun/datasets/SIMPLE/training/corrected9-20260908
```

That destination already exists from the September 8 run. Choose a new path to
repeat preparation. Its `processed/train` contains six episodes / 925 frames;
`processed/val` contains three episodes / 462 frames. Both splits passed two
certifications through PSI0 commit `885538e0bbee05caa1a89d382653c860596eee95`.
Training and validation source Parquet hashes do not overlap. Use only
`processed/train/meta/stats_psi0.json` for normalization in both splits.
Stored actions remain 36-D and states 32-D; the model pads states to 36-D.

The full PSI0 smoke test ran in Docker on **H100 physical GPU 7**, not on the
workstation. Container `jihun_psi0_simple_corrected9_gpu7_20260908` uses an
isolated source snapshot, read-only data and pretrained weights, and separate
writable results. The existing `jihun_psi0_sonic_train_gpu23_20260805` container
was inspected only; its GPUs were occupied by other workloads.

The run completed three optimizer steps, three held-out validation passes,
and checkpoint saves with batch size one, bf16, frozen VLM, 36-D actions,
30-frame chunks, and RTC training. All losses were finite:

| Step | Training loss | Validation loss |
| --- | ---: | ---: |
| 1 | 43.4030 | 44.4903 |
| 2 | 42.2987 | 39.5637 |
| 3 | 29.9186 | 45.9084 |

Checkpoints 2 and 3 remain on H100 under
`/mnt/data01/jhkim/model_weight/Psi0/simple-corrected9-smoke-20260908/results/runs/finetune/`.
Checkpoint 3 passed a fresh strict VLM/action-head reload. This verifies the
training pipeline; three steps do not establish a learned picking policy.

Local evidence is under the training destination in `training-result.json`
and `h100-evidence/`, including the exact source/launch package, arguments,
run configuration, metrics, and logs. The first training attempt lacked FFmpeg;
installing it inside the new container resolved video decoding. The successful
environment used Python 3.11, Torch 2.7.0+cu126, Transformers 4.57.1, and
FlashAttention 2.7.4.post1. Metrics were recorded locally without W&B uploads.

The 13 required pretrained files were downloaded and checksum-verified from
`USC-PSI-Lab/psi-model` revision `a4c69a6e4af7eee5beb1aee2597c4184fc508683`
into `/mnt/data/jihun/model_weights/psi0-training`. H100 used its existing
pretrained assets under `/mnt/data01/jhkim/model_weight/Psi0`.

When using a shared simulation virtual environment, explicitly set
`PYTHONPATH=<corrected-generator-worktree>/src`. Otherwise its editable install
can import a different worktree even when the working directory is correct.

Checkpoint 3 also completed one held-out simulation episode per DR level,
with a 300-step cap: **0/3 task successes**. All 12 camera videos contain
301 frames at 640×360 and 50 Hz. Sampled front-left review shows the robot
remaining upright but knocking the target off the table. This is a completed
runtime smoke test, not successful task learning. Results and contact sheets
are in `simulation-smoke-retry1/`; the earlier failed import attempt remains
in `simulation-smoke/`. The H100 smoke container was stopped after evaluation.

The next generation batch was started at
`/mnt/data/jihun/datasets/SIMPLE/batches/20260908T073639Z-6cb35317`, requesting
100 corrected episodes per DR level on workstation GPU 1. Its `result.json`
will establish completion and certification; a started run alone is not a
completed dataset. It retains the same target and room as the verification
batch, so broader object/scene coverage remains separate work.
