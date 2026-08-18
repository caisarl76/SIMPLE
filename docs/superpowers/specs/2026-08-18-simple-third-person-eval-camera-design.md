# SIMPLE Third-Person Evaluation Camera Design

## Status and scope

This design adds an opt-in third-person camera to SIMPLE policy evaluation and
verifies it with one official PSI0 simulation episode. It does not retrain a
policy, diagnose OpenFaucet, reproduce the complete level-0 benchmark, or
authorize any real-robot interface.

The supported production path is the official G1 teleoperation task family
through `eval` or `eval-decoupled-wbc`, in `mujoco`, `isaac`, or
`mujoco_isaac` mode. The public `simple.evals.EnvRunner` receives the same
configuration so library and CLI evaluation cannot silently disagree. Motion
planning tasks that already define external cameras are not part of the
acceptance run; they may opt in only if they satisfy the clipping compatibility
check described below.

This revision deliberately brings two supporting repairs into scope:

1. environment construction must use a fresh task object rather than a cached
   task singleton; and
2. video finalization and evaluation cleanup must be checked and bounded.

Without those repairs, an optional camera can mutate another live environment
or leave an evaluation worker and raw video in an indeterminate state.

## User-facing contract

1. Both evaluation CLIs expose
   `--third-person-video/--no-third-person-video`, defaulting to disabled.
2. Both configuration types gain `third_person_video: bool = False`:
   the frozen worker configuration in `simple.evals.api.EvalConfig` and the
   public `simple.evals.EvalConfig` exported from `env_runner`.
3. `--third-person-video --no-save-video` is rejected before policy health
   checks, dataset loading, task construction, or simulator startup. The same
   combination supplied through either Python API raises `ValueError` during
   configuration validation.
4. With the flag disabled, sensor keys, observation spaces, policy requests,
   and video names are unchanged.
5. With the flag enabled, the environment owns exactly one additional RGB
   observation named `third_person`. It is present before either simulation
   engine is constructed.
6. A task that already defines `third_person` is rejected before any Isaac or
   MuJoCo resource is created. Existing sensors are never overwritten.
7. `third_person` is recording-only. `Psi0Agent` and
   `Psi0DecoupledWbcAgent` continue to send only `head_stereo_left` and their
   existing state/history fields to the server.

## Canonical camera contract

A focused module under `simple.evals` owns `THIRD_PERSON_SENSOR_KEY` and a
factory that returns a fresh `CameraCfg` on every call. The exact configuration
is:

```python
CameraCfg(
    uid="simple_eval_third_person_v1",
    mount="eye_in_world",
    width=640,
    height=360,
    focal_length=1.88,
    fov=np.deg2rad(90.0),
    near=0.2,
    far=5.0,
    pose={
        "distance": 2.5,
        "polar": np.deg2rad(60.0),
        "azimuth": np.deg2rad(0.0),
    },
)
```

`fov`, `polar`, and `azimuth` are radians. `fov` is the horizontal field of
view used by `CameraCfg.fx`; the vertical field follows from 640 by 360 and
square pixels. `focal_length` is the physical focal-length parameter used to
derive the Isaac aperture and is not a replacement for `fov`.

`eye_in_world` is a new, explicit mount. It is intentionally not
`eye_on_base`, whose current implementations disagree between backends. Both
engines attach `eye_in_world` to the workspace/world root, not to the robot.
The spherical pose is converted once by `CameraEntity`: the camera center is

```text
[r cos(azimuth) sin(polar),
 r sin(azimuth) sin(polar),
 r cos(polar)]
```

in workspace coordinates, and its optical axis points at workspace origin with
zero roll. Thus the canonical center is approximately
`[2.1650635, 0.0, 1.25]` metres. The camera remains world-fixed as the robot
moves.

### Backend rendering semantics

- **Isaac:** `eye_in_world` is created directly beneath the workspace prim.
  Resolution, pose, focal length, aperture, and clipping range come from the
  `CameraEntity`; the existing hard-coded `0.01, 10.0` clipping call is
  replaced by `cam_cfg.near, cam_cfg.far` for every Isaac camera.
- **MuJoCo:** `eye_in_world` is attached directly to `mjSpec.worldbody` with
  the existing Isaac-to-MuJoCo optical-frame rotation. MuJoCo clipping is
  model-global rather than per-camera. When the evaluation camera is enabled,
  every configured RGB/stereo camera must request the same `(near, far)` pair
  as the evaluation camera. This compatibility check happens before engine
  startup. After compilation, the implementation sets
  `model.vis.map.znear * model.stat.extent == 0.2` and
  `model.vis.map.zfar * model.stat.extent == 5.0`, within floating-point
  tolerance, before creating renderers. A mismatched task is rejected instead
  of silently changing another camera's optical contract.
- **Mixed mode:** SIMPLE continues to place Isaac images in observations when
  Isaac is present. Therefore a mixed-mode `third_person` video is the Isaac
  rendering. A separate MuJoCo-only render test is still required; passing a
  mixed-mode test is not evidence that the MuJoCo camera is correct.

The two backend tests use the same asymmetric scene markers at known workspace
coordinates. They assert image shape, non-empty pixels, left/right marker
ordering, the projected center target, and effective clipping. Pixel-for-pixel
equality is not required because the renderers and materials differ.

## Environment ownership and construction order

### Fresh task instances

`TaskRegistry` gains an uncached construction method, named
`make_fresh(uid, *args, **kwargs)`, that validates the UID and directly invokes
the registered task class. The existing cached `make` method remains available
for callers that explicitly depend on its legacy behavior, but `BaseDualSim`
must use `make_fresh` whenever it receives a task UID string. It must not call
`TaskRegistry.make` a second time at the end of construction.

For a newly created task, `BaseDualSim` deep-copies the task's sensor mapping
and sensor configurations onto the task instance before applying any
environment-only sensors. It never mutates the registered task class or a
mapping/configuration owned by another task instance. Evaluation uses task UID
strings; passing a preconstructed `Task` leaves ownership with the caller and
is not the supported injection path.

Two simultaneously live MuJoCo-only environments are a required regression
case. An opt-in environment and an opt-out environment for the same task UID
must have different task objects, sensor mappings, and layouts. Resetting or
closing either one must not change the other. Isaac's process-global
`SimulationApp` is tested separately and is not used to fake two concurrently
supported Isaac applications.

### Fail-before-resource ordering

`BaseDualSim.__init__` performs these phases in this exact order:

1. resolve a fresh task instance;
2. deep-copy its sensor configuration;
3. validate the extra mapping, key collisions, camera fields/mount, and
   MuJoCo clipping compatibility;
4. merge the validated mapping and derive task observation/action spaces;
5. initialize the Isaac application if requested;
6. construct `IsaacSimSimulator` if requested; and
7. construct `MujocoSimulator`.

The collision/contract tests replace `_init_isaac`, `IsaacSimSimulator`, and
`MujocoSimulator` with call-counting fakes and require all three counts to stay
zero. Post-validation constructor failure is also rollback-safe: already
constructed engines are closed in reverse order, and an Isaac application
started by this constructor is closed and its module globals restored before
the exception is re-raised.

## Evaluation plumbing

Both CLI parent functions, worker kwargs, and worker entry points carry the
boolean. The parent rejects the invalid save-video combination before spawning
workers. Each worker calls the canonical factory and passes
`extra_sensor_cfgs={"third_person": fresh_cfg}` to `gym.make` only when the
flag is true.

The public `simple.evals.EnvRunner` performs the same validation and passes the
same mapping in `_make_env`. Its `EvalConfig` is updated independently of the
frozen CLI/API configuration; neither class is treated as an alias of the
other.

`VideoRecorder` already discovers every HWC RGB observation-space entry. It
therefore needs no third-person conditional: the new observation naturally
creates a writer named `third_person`.

## Artifact and finalization contract

### Exact paths

Inside every episode directory, the successful final artifact is exactly:

```text
episode_<N>/third_person_success.mp4
```

or, for a failed/incomplete episode:

```text
episode_<N>/third_person_failed.mp4
```

The complete standard-eval path is:

```text
<eval_dir>/<policy>/<environment-id-without-namespace>/<split>/
  episode_<N>/third_person_<verdict>.mp4
```

The current decoupled-WBC artifact root is intentionally retained:

```text
data/evals/<policy>/
  [<server-policy>-<server-timestamp>.]<environment-id-without-namespace>/
  <basename(data_dir)>/episode_<N>/third_person_<verdict>.mp4
```

The bracketed server prefix is present only when `/info` supplies both fields.
`eval_dir` remains the dataset/log root for that CLI; this design does not
silently reinterpret it as the decoupled video root.

### Checked bounded finalization

`VideoWriter.release` no longer invokes shell strings. Its contract is:

1. close the OpenCV writer exactly once;
2. if FFmpeg is available, run an argv-form `subprocess.run` with `-nostdin`,
   `check=True`, captured stderr, and a 60-second timeout;
3. transcode to a temporary `.mp4` beside the destination;
4. atomically replace the verdict path only after successful transcoding; and
5. delete the raw `<camera>.mp4` only after that replacement succeeds.

If FFmpeg is unavailable, the closed raw MP4 is atomically renamed to the
verdict path. On FFmpeg error or timeout, the temporary output is removed, the
raw MP4 is preserved, and `VideoFinalizationError` reports both the raw path
and the failure. Finalization is idempotent.

`VideoRecorder.release` uses one 60-second absolute deadline for all of its
writers, attempts every writer even if one fails, and raises an aggregate error
only after all attempts. This prevents three cameras from accumulating three
independent 60-second waits.

### Worker lifecycle

Both evaluation workers use `try/finally` ownership at two levels:

- each episode always finalizes an opened `VideoRecorder`, marking it failed
  if the episode did not reach a verdict; and
- each worker always closes the agent when it exposes `close`, the rollout
  wrapper, and the raw environment, even after reset, policy, render,
  finalization, or result-persistence failure.

Cleanup errors are collected rather than preventing later resources from being
closed. The original rollout error remains primary, with cleanup errors
attached or logged; a finalization failure makes the worker result non-success
and includes the preserved raw-video path.

For multi-worker cancellation, the parent sends an interrupt to live workers,
allows a 65-second cleanup grace (the shared 60-second transcode deadline plus
five seconds), then terminates any worker that still has not exited. It joins
every child and closes every pipe in a parent `finally` block. The end-to-end
command uses an interrupt-first outer timeout with a longer kill grace so it
does not preempt this cleanup window.

## Policy isolation

Regression coverage is required for both `Psi0Agent` and
`Psi0DecoupledWbcAgent`. For each agent, the test builds two otherwise identical
observations, with the second containing an arbitrary `third_person` image.
It freezes:

- `HttpActionClient.timestamp`;
- session ID/PID-derived fields;
- global step and reset-history state; and
- all state, instruction, history, and condition inputs.

The captured requests are passed through the repository's numpy serializer and
canonical JSON encoding (`sort_keys=True`, compact separators). Their UTF-8
bytes must be identical, and their image dictionary must contain exactly
`rgb_head_stereo_left`. Merely comparing action shapes or Python dictionary
subsets is insufficient.

## Automated acceptance tests

Tests cover all of the following:

1. the camera factory's key, UID, mount, resolution, focal length, radians,
   clipping planes, pose, and fresh-object behavior;
2. both `EvalConfig` classes and both CLI parent-to-worker paths;
3. early rejection of `--third-person-video --no-save-video`;
4. two simultaneously live environments for one task UID, one opt-in and one
   opt-out, with independent task/sensor/layout state;
5. default-path parity with the original task sensor keys;
6. collision and invalid-camera rejection with zero calls to `_init_isaac`,
   `IsaacSimSimulator`, and `MujocoSimulator`;
7. reverse-order rollback after each possible post-validation construction
   failure;
8. Isaac `eye_in_world` parent, pose, intrinsics, resolution, and configured
   near/far behavior;
9. MuJoCo worldbody parent, pose, intrinsics, resolution, effective near/far,
   and rejection of incompatible existing camera clipping;
10. mixed mode selecting Isaac's `third_person` output;
11. unchanged canonical PSI0 request bytes for both PSI0 agents;
12. checked FFmpeg success, nonzero exit, timeout, missing-FFmpeg, raw
    preservation, idempotency, and shared-deadline behavior;
13. episode and worker cleanup after reset, policy, render, transcode, and
    interruption failures; and
14. exact episode-relative success/failure paths.

The focused test command is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_third_person_eval_camera.py \
  tests/test_third_person_eval_backends.py \
  tests/test_third_person_policy_isolation.py \
  tests/test_eval_video_finalization.py \
  tests/test_eval_worker_cleanup.py
```

The static and full regression gates are:

```bash
ruff check src/simple tests
ruff format --check src/simple tests
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider tests
git diff --check
```

## End-to-end verification

Start or reuse the official XMovePick step-40000 PSI0 server on the H100 and an
SSH tunnel exposing it at `127.0.0.1:22085`. Record the remote checkpoint path,
checkpoint hash, server command/log, and tunnel PID in the run directory. Then
run exactly one official episode locally:

```bash
SIMPLE_DISABLE_TUI=1 CUDA_VISIBLE_DEVICES=1 \
timeout --signal=INT --kill-after=75s 1800s \
  .venv/bin/eval-decoupled-wbc \
  simple/G1WholebodyXMovePickTeleop-v0 psi0_decoupled_wbc train \
  --data-format lerobot \
  --data-dir data/evals/simple-eval/G1WholebodyXMovePickTeleop-v0/dr-level-0 \
  --host 127.0.0.1 --port 22085 \
  --sim-mode mujoco_isaac --headless \
  --num-episodes 1 --episode-start 0 --num-workers 1 \
  --third-person-video --save-video
```

The run passes only if:

- the episode reaches a normal success/failure verdict;
- the exact episode-relative third-person verdict path exists and no raw
  `third_person.mp4` remains after successful finalization;
- FFprobe reports 640 by 360, positive frame count, and positive duration;
- decoded first/middle/last frames are non-empty and a reviewed middle frame
  contains the G1 and task scene;
- the ordinary head-stereo video still exists;
- a separate MuJoCo-only render integration test passed;
- both frozen PSI0 request-byte tests passed; and
- the local evaluator, worker, FFmpeg process, SSH tunnel, and remote PSI0
  server are stopped, with their exit states recorded.

If transcoding fails or times out, the run fails but the raw MP4 must remain for
diagnosis. The verification certifies rendering, policy-input isolation, and
recording lifecycle only. The episode verdict is not a benchmark result or a
deployment approval.
