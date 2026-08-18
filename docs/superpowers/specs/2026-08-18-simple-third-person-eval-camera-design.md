# SIMPLE Third-Person Evaluation Camera Design

## Status and scope

This design adds an opt-in third-person camera to SIMPLE policy evaluation and
verifies it with one official PSI0 simulation episode. It does not change the
default task camera set, retrain a policy, diagnose OpenFaucet, run the complete
level-0 benchmark, or authorize a real-robot interface.

The feature applies to both `eval` and `eval-decoupled-wbc`. The motivating use
case is the eight official teleoperation candidates that currently record only
the robot's head-stereo view. Existing motion-planning tasks may also use the
flag, although some already provide external `front_stereo` cameras.

## Requirements

1. Both evaluation CLIs expose
   `--third-person-video/--no-third-person-video`, defaulting to disabled.
2. With the flag disabled, task sensor configuration, observation spaces,
   policy requests, and output filenames remain unchanged.
3. With the flag enabled, the environment adds exactly one RGB observation
   named `third_person` before either simulation engine is constructed.
4. The camera configuration is 640 by 360 RGB, uses the repository's proven
   `eye_on_base` external-camera mount, has a 2.5 metre orbit distance, a
   60-degree polar angle, a zero-degree azimuth, a 90-degree field of view, a
   0.2 metre near plane, and a 5 metre far plane.
5. Camera injection is instance-local. It must not mutate a task class's
   `sensor_cfgs` dictionary or leak into a later environment created without the
   flag.
6. Injection fails before simulator construction if `third_person` is already a
   task sensor key.
7. The existing `VideoRecorder` records the additional observation without a
   task-specific branch. Final artifacts follow its existing convention:
   `third_person_success.mp4` or `third_person_failed.mp4`.
8. PSI0 continues to serialize only its configured head image and state. The
   `third_person` image is evaluation evidence and is never sent to the policy
   server.

## Architecture

### Camera factory

A focused module under `simple.evals` owns the canonical evaluation camera key
and factory. Keeping numeric camera parameters in one place prevents the two
CLI implementations from drifting. Each call returns a fresh `CameraCfg`.

### Environment injection

`BaseDualSim` accepts an optional mapping of evaluation-only sensor
configurations. It resolves the registered task, copies the task's base sensor
mapping onto the task instance, validates that no extra key collides, and then
merges the extra mapping. This happens before `IsaacSimSimulator` and
`MujocoSimulator` are created, so both engines build the same named camera and
the environment observation space includes it.

Every environment construction explicitly applies either the supplied mapping
or an empty mapping. This is necessary because `TaskRegistry` caches task
instances: a later environment in the same process must restore the base sensor
set when the flag is disabled.

The evaluation camera is passed as an environment-level argument and consumed
by `BaseDualSim`; it is not forwarded through individual task or robot
constructors.

### Evaluation plumbing

`EvalConfig` gains a frozen boolean `third_person_video` field with a false
default. Both CLIs expose the matching Typer option, copy the field through
worker kwargs, and add the canonical camera mapping to `gym.make` only when the
field is true. The field remains compatible with single- and multi-worker
evaluation because it is a primitive boolean and camera construction occurs in
each worker.

### Recording and policy isolation

`VideoRecorder` already discovers all HWC RGB observation-space entries and
creates one writer per entry. Therefore the new camera needs no recorder
special case. PSI0 agents select `head_stereo_left` explicitly, so the request
payload must remain byte-for-byte independent of the additional observation
key; a regression test will enforce this behavior.

The existing FFmpeg finalization implementation is outside this change. The
end-to-end verification runs under an outer timeout and treats a decodable,
non-empty finalized third-person MP4 as the artifact contract.

## Error handling and cleanup

- A `third_person` key collision raises `ValueError` before simulator resources
  are created.
- Invalid camera construction fails evaluation startup rather than falling back
  to an ego view.
- Existing evaluation cleanup remains responsible for video writers,
  simulators, policy clients, and worker processes.
- Verification must stop the H100 PSI0 server and SSH tunnel and confirm that
  no local evaluation process, listener, or GPU job remains.

## Tests

Automated tests cover:

1. The factory returns the exact key, mount, resolution, clipping planes, field
   of view, and orbit pose.
2. The default environment path retains the original sensor keys.
3. Opt-in injection adds exactly `third_person` and does not mutate class-level
   task configuration.
4. A subsequent opt-out construction in the same process removes the injected
   camera from the cached task instance.
5. A task-defined `third_person` key is rejected.
6. `EvalConfig` and both CLI-to-worker paths preserve the boolean flag.
7. Adding `third_person` to an observation does not change the image/state
   dictionaries serialized by `Psi0DecoupledWbcAgent`.
8. Existing evaluation and video-recorder tests remain green.

## End-to-end verification

Run official level-0 episode 0 of
`G1WholebodyXMovePickTeleop-v0` with its official step-40000 PSI0 checkpoint,
the decoupled-WBC evaluation runner, headless local simulation, and the new
flag. Verification requires:

- the policy episode reaches a normal success or failure verdict;
- `third_person_<verdict>.mp4` exists and is larger than an empty container;
- FFprobe reports 640 by 360 video and a positive frame count/duration;
- a decoded middle frame visibly contains the G1 robot and task scene;
- the ordinary head-stereo video is still produced;
- policy request-contract tests pass; and
- all local and H100 evaluation infrastructure is stopped afterward.

The verification certifies rendering and recording only. Its task verdict does
not constitute a benchmark result or deployment approval.
