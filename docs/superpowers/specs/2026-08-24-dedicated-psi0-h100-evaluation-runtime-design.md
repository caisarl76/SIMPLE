# Dedicated PSI0 H100 Evaluation Runtime Design

## Status and scope

This design replaces use of the shared H100 training container for official
SIMPLE PSI0 simulation evaluation. It creates one persistent but normally
stopped evaluation container on physical H100 GPU 7, starts the official PSI0
server only for an owned evaluation run, tunnels the private server to PC2,
and proves bounded cleanup afterward.

The initial acceptance workload is episodes 2 and 3 of
`simple/G1WholebodyXMovePickTeleop-v0` with the official step-40000 checkpoint,
`psi0_decoupled_wbc`, and the existing `mujoco_isaac` headless evaluator. The
ordinary head-stereo recordings remain the visual artifacts. The deferred
third-person camera is not part of this work.

This design does not:

- modify or train PSI0;
- certify the step-40000 checkpoint for the PC2 real-time bridge;
- add `/contract` or `/act-rtc-v1` to the official server;
- enable DDS, a real network interface, or a real robot; or
- stop, restart, or otherwise modify the existing shared
  `jihun_psi0_sonic_train_gpu23_20260805` container.

The checkpoint remains `UNCERTIFIED` for bridge control. Standard `/act`
simulation success is evidence only for the official SIMPLE evaluation path.

## Fixed identities

The first implementation supports one explicit runtime profile. These values
are configuration constants, not inferred from whichever container happens to
be running:

| Field | Value |
| --- | --- |
| SSH target | `h100` |
| Container | `jihun_psi0_simple_eval_gpu7` |
| Image tag | `pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel` |
| Physical GPU | host index `7` |
| Container CUDA device | index `0` |
| Container server port | `22185` |
| Default PC2 loopback port | `22085` |
| Task | `simple/G1WholebodyXMovePickTeleop-v0` |
| Policy | `psi0_decoupled_wbc` |
| Checkpoint step | `40000` |
| Action execution horizon | `24` |
| Simulation mode | `mujoco_isaac` |
| PC2 simulator GPU | index `1` |
| WBC environment/interface/domain | `sim` / `lo` / `0` |

The checkpoint host path is:

```text
/mnt/data01/jhkim/model_weight/Psi0/simple-checkpoints/
  g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205
```

and is visible in the container beneath
`/hfm/cache/checkpoints/psi0/simple-checkpoints`. Its required model size and
SHA-256 are:

```text
6253648840 bytes
27df2e24c5efd176b962d2b219565056fc5081b69e050821a313249e677dd0f9
```

The mounted server sources must contain these exact files and hashes:

```text
src/psi/deploy/psi0_serve_simple.py
  e03c4ee61dd29d95292ec1ce24efe48d6c0d5b6b177a485b0b9e16c44d79cba4

scripts/launch_psi0_server.sh
  0202137143460c305cee39adb1901c7c480caaa7c199e4b4cd77b81cec374579
```

The remote source tree currently has no Git metadata, so these content hashes,
the resolved Docker image ID, and the complete container configuration are the
runtime provenance. Container creation records the image ID in both a Docker
label and
`outputs/psi0-h100-runtime/container-identity.json`. Reuse requires exact
agreement between that record, the label, the current container image ID, and
the expected creation manifest; a moved image tag is not accepted as
equivalent. A missing identity record blocks reuse rather than silently
adopting an existing container.

## Container contract

### Persistent object, ephemeral workload

`jihun_psi0_simple_eval_gpu7` is created once and reused, but it is stopped
outside an evaluation. Its command is an idle process supervised by Docker
`--init`; the policy server never starts from the container command, entrypoint,
restart policy, or a background service.

The required Docker configuration is:

- NVIDIA runtime with only host GPU 7 exposed;
- `--init` so orphaned server children are reaped;
- `--restart=no`;
- the ordinary private Docker bridge network;
- no published or host-network ports;
- private 64 GiB shared memory rather than host IPC;
- a read-only root filesystem with explicit writable runtime locations;
- fixed ownership labels for SIMPLE, the selected host GPU UUID, image ID, and
  runtime-contract version; and
- the mounts below.

| Host path | Container path | Access |
| --- | --- | --- |
| `/home/kube/psi0-train/repo` | `/workspace/Psi0` | read-only |
| `/mnt/data01/jhkim/model_weight/Psi0` | `/hfm/cache/checkpoints/psi0` | read-only |
| `/mnt/data01/huggingface` | `/root/.cache/huggingface` | read-only |
| `/mnt/data01/jhkim/datasets` | `/hfm/data` | read-only |
| `/mnt/data01/jhkim/psi0-simple-eval` | `/runtime` | read-write |

`/tmp` is a bounded writable `tmpfs`. `XDG_CACHE_HOME`, `TORCH_HOME`, and
`TRITON_CACHE_DIR` point beneath `/runtime/cache`; `HF_HOME` remains on the
read-only Hugging Face mount. `PYTHONDONTWRITEBYTECODE=1` prevents writes to the
mounted source tree. A run writes only beneath
`/runtime/runs/<run-id>` and the shared compilation caches. It cannot edit the
server source, checkpoint, dataset, or Hugging Face inputs.

If a container with the fixed name exists but any required field differs,
`create` and `evaluate` fail closed. The tool reports the complete field-level
diff and does not rename, remove, recreate, start, or mutate that container.
Recreation is a separate explicit operator action outside `evaluate`.

### GPU selection and exclusion

Existing stopped or idle containers configured for GPU 7 do not block this
runtime. An active compute process does.

Before starting the container, the manager records host `nvidia-smi -L` and
the structured compute-process table. It resolves host GPU index 7 to a UUID
and requires no compute process with that UUID. After container start, an
in-container probe must see exactly one GPU and its UUID must exactly equal the
selected host UUID. CUDA ordinal 0 alone is never accepted as identity proof.

Failure to query either host or container GPU state is an unknown state and
fails preflight. The manager never starts or stops another container to make
GPU 7 available.

## Host-side interface

A Python CLI under `scripts/` owns the complete lifecycle. It uses argument
vector subprocesses, monotonic deadlines, and a replaceable command runner so
unit tests do not require SSH or Docker. It exposes four operations:

```text
create    attest inputs and create the stopped container if it is absent
status    report container, GPU, server, tunnel, and artifact state read-only
evaluate  run the complete preflight, server, tunnel, episodes, and cleanup
stop      stop only resources whose ownership can be proven
```

`evaluate` accepts a unique output root and episode list; the production
profile restricts the first acceptance run to episodes `2,3`. It refuses an
output directory that already exists. It also refuses a local port that is
already bound or an episode whose canonical evaluator artifact would be
overwritten. It refuses an absent container and directs the operator to run
`create` first. There is no force flag in this scope.

No local command uses `shell=True`. Remote commands are built from fixed
tokens and strictly validated values. A run ID is generated locally from UTC
time, the SIMPLE short SHA, and a random suffix, and must match
`[A-Za-z0-9._-]+` before it is included in a remote path.

## Lifecycle state machine

The manager persists every state transition before performing the next
external action:

```text
NEW -> PREFLIGHTED -> CONTAINER_STARTED -> SERVER_STARTED -> TUNNEL_READY
    -> SERVER_READY -> EVALUATING -> CLEANING -> CLEAN | FAILED
```

Any state may transition to `CLEANING`. A PASS verdict requires the terminal
state `CLEAN`; `FAILED` is used for evaluation failure, timeout, unknown
liveness, cleanup failure, or evidence failure. A policy episode may report
task failure without making the infrastructure invalid, but the manifest
records that episode as failed and the overall evaluation verdict is not
PASS.

### Preflight

Preflight is read-only and completes before the container is started. It must:

1. confirm bounded SSH connectivity;
2. collect the remote host identity and wall clock;
3. attest the checkpoint file, size, and hash;
4. attest both server source hashes;
5. resolve and record the Docker image ID;
6. attest an existing container or determine that `create` is required;
7. resolve GPU 7 to its UUID and prove no active compute process uses it;
8. prove the local loopback port is free;
9. prove no live owned server or stale owned PID record exists; and
10. confirm all local episode inputs exist and no target artifact would be
    overwritten.

A preflight failure starts nothing. Merely finding other containers configured
for GPU 7 is diagnostic evidence, not a failure.

### Server startup and readiness

The server command is equivalent to:

```bash
PYTHONPATH=/workspace/Psi0/src \
  /workspace/Psi0/.venv/bin/python -m psi.deploy.psi0_serve_simple \
  --host 0.0.0.0 --port 22185 --device cuda:0 --policy=psi0 \
  --run-dir=<fixed-container-checkpoint-path> --ckpt-step=40000 \
  --action-exec-horizon=24 --rtc
```

It is launched with a run-specific log, PID record, exact command digest, and
Linux process start time beneath `/runtime/runs/<run-id>`. A PID is considered
owned only when the container identity, PID record, process start time, and
normalized command all agree. A port occupant without that complete ownership
proof blocks startup and is never signalled.

Readiness has three gates:

1. the owned process remains alive and container port 22185 is listening;
2. the SSH tunnel is alive and the PC2 loopback endpoint accepts connections;
3. one schema-valid `/act` warm-up returns HTTP 200 with a finite NumPy action
   of exact shape `(24, 36)`.

The deterministic warm-up uses one black `(360, 640, 3)` `uint8` RGB image
under `rgb_head_stereo_left`. Its `(1, 32)` `float32` state is all zero except
`state[0, 31] = 0.74`, the command-height field. It uses
`history={"reset": true}`, empty condition and ground-truth action,
`dataset_name="simple"`, and the existing SIMPLE NumPy JSON serialization.
Its predicted values are not executed, scored, or treated as checkpoint
certification. The request and response structural summary and latency are
recorded. The response record hashes the returned dtype, shape, and contiguous
action bytes without embedding the large tensor in the manifest.

The official server's missing `/info`, `/contract`, and `/act-rtc-v1` remain
expected limitations. Readiness must not claim those endpoints exist.

### Private tunnel

After every container start, the manager resolves the current private bridge
IP. It then starts:

```text
ssh -N -L 127.0.0.1:<local-port>:<container-ip>:22185 h100
```

The tunnel binds loopback only. No Docker port is published. Readiness requires
the SSH child to remain alive and a connection through the loopback endpoint
to succeed. The manager owns the exact local SSH PID and process start time;
it never uses a pattern-wide `pkill`.

### Episode evaluation

Episodes 2 and 3 run separately in ascending order, giving each an independent
20-minute deadline and exit record. The command is based on the already proven
episode-1 path:

```bash
SIMPLE_DISABLE_TUI=1 CUDA_VISIBLE_DEVICES=1 \
MPLCONFIGDIR=<run-dir>/matplotlib-cache \
timeout --signal=INT --kill-after=75s 1200s \
  .venv/bin/eval-decoupled-wbc \
  simple/G1WholebodyXMovePickTeleop-v0 psi0_decoupled_wbc train \
  --data-format lerobot \
  --data-dir data/evals/simple-eval/G1WholebodyXMovePickTeleop-v0/dr-level-0 \
  --host 127.0.0.1 --port <local-port> \
  --sim-mode mujoco_isaac --headless \
  --eval-dir <run-dir>/episode_<N>/eval-logs \
  --num-episodes 1 --episode-start <N> --num-workers 1 --save-video
```

There is no `--third-person-video` flag. The evaluator must retain
`sim`/`lo`/domain 0 isolation and must not open a real Unitree interface.
Immediately before `gym.make`, the worker atomically emits the actual
`sonic_config` fields it will consume to the run's interface-evidence file.
The manager requires `ENV_TYPE="sim"`, `INTERFACE="lo"`, and `DOMAIN_ID=0`
from that worker-produced record; a separately reconstructed default or a
hard-coded zero counter is not acceptable evidence. The worker refuses any
other values before environment or channel construction.

Before the next episode starts, the prior evaluator must have exited and its
local worker/WBC children must be gone. If cleanup after one episode cannot be
proven, the second episode is not started. A normal task-level failure may
proceed to the next episode; an infrastructure or cleanup failure may not.

Each evaluator is started as a new local process group. The recorded leader
PID, leader process start time, process-group ID, and exact argv define local
ownership. Worker and WBC descendants are enumerated from that group and the
process tree; cleanup never matches processes by name alone.

## Deadlines and shutdown

All deadlines use `time.monotonic()` and are passed as a shared absolute
deadline through nested cleanup operations.

| Operation | Deadline |
| --- | ---: |
| Individual SSH/Docker inspection | 15 s |
| Container start or stop | 30 s |
| Model process and TCP readiness | 300 s total |
| Warm-up `/act` response | 120 s |
| Each evaluator episode | 1200 s plus its 75 s forced-exit allowance |
| Tunnel signal stage | 2 s each for INT, TERM, and KILL |
| Server signal stage | 5 s each for INT, TERM, and KILL |
| Final cleanup verification | 30 s |

Cleanup is registered before each resource is started and executes in reverse
order. It is attempted in this order:

1. interrupt and reap the active evaluator and its owned children;
2. close the SSH tunnel;
3. stop the owned server inside the container;
4. copy final remote evidence;
5. stop the dedicated container; and
6. verify local and remote postconditions.

INT, TERM, and KILL are used only after ownership is revalidated at that
stage. After KILL, a fresh bounded wait and liveness check are mandatory.
Failure in one cleanup action is recorded but does not skip later actions.
Terminal restoration, open log closure, and manifest finalization are
unconditional even after the shared cleanup budget is exceeded.

The persistent container object remains for reuse, but its required terminal
state is `exited`. A run cannot pass if the container is running, the owned
server or tunnel is alive, either relevant port is listening, an evaluator or
owned WBC child remains, or liveness is unknown.

## Evidence contract

Local evidence is immutable beneath:

```text
outputs/official-psi0-standard-eval/<run-id>/
```

Remote transient evidence is beneath:

```text
/mnt/data01/jhkim/psi0-simple-eval/runs/<run-id>/
```

The local run directory contains at least:

```text
manifest.json
run-manifest.md
preflight/
  ssh-host.json
  gpu-before.json
  configured-gpu7-containers.json
  container-inspect.json
  source-hashes.json
  checkpoint.json
server/
  command.json
  server.log
  readiness.json
  warmup.json
tunnel/
  process.json
  tunnel.log
episode_2/
  command.json
  evaluator.log
  result.json
  artifacts.json
episode_3/
  command.json
  evaluator.log
  result.json
  artifacts.json
cleanup/
  actions.json
  gpu-after.json
  container-final.json
  processes-final.json
  ports-final.json
```

JSON documents have a schema version and use atomic write-then-rename. The
final manifest records every external command in redacted argv form, start and
finish timestamps, monotonic durations, return code or timeout, identities,
hashes, state transitions, episode verdicts, cleanup attempts, and unresolved
errors. No token, SSH private material, environment secret, or full image/action
payload is stored.

For every resulting head-stereo MP4, the artifact record includes the absolute
source path, copied immutable evidence path, byte count, SHA-256, codec,
resolution, frame rate, frame count, and duration. First, middle, and final
frames must decode and be nonblank. This is an automated integrity check, not
a new third-person or semantic-success classifier.

The Markdown manifest is a human-readable rendering of `manifest.json`; it is
not an independent source of truth. Existing failed or successful run
directories are never overwritten or deleted.

## Verdict rules

Infrastructure PASS requires all of the following:

- every preflight and identity assertion passed;
- container CUDA device 0 UUID equalled host GPU 7 UUID;
- the warm-up returned a finite `(24, 36)` action;
- both episode evaluator processes reached a normal recorded exit;
- each episode produced a recorded SIMPLE verdict and valid stereo artifacts;
- worker-emitted runtime evidence proved the actual WBC configuration was
  `sim` / `lo` / domain 0 before each environment was constructed;
- no real-robot control process or non-loopback Unitree interface owned by the
  run was observed;
- every owned process and port was absent after cleanup;
- the dedicated container was stopped; and
- every mandatory evidence file was finalized.

The overall evaluation result separately reports each task verdict. An episode
that executes normally but returns task failure is valid infrastructure
evidence but makes the evaluation result FAIL. A timeout, VPN loss, malformed
response, output collision, hash mismatch, active GPU process, unknown remote
state, or incomplete cleanup makes both the affected run and infrastructure
verdict FAIL.

The manager never changes a failed verdict to PASS on retry. A retry receives a
new run ID and preserves all earlier evidence.

## Error handling and recovery

- **VPN or SSH loss before startup:** fail without starting anything.
- **VPN loss after startup:** continue local cleanup, repeatedly attempt only
  owned remote cleanup within the shared deadline, then record remote state as
  unknown if it cannot be attested. Unknown is failure, not stopped.
- **GPU becomes busy between probes:** recheck immediately before server launch;
  fail and stop the dedicated container without launching the server.
- **Unexpected server or port occupant:** preserve diagnostics and do not
  signal it.
- **Server crash or invalid warm-up:** preserve the complete server log and
  clean up without launching an evaluator.
- **Tunnel failure during evaluation:** interrupt the evaluator, then perform
  normal cleanup; do not retry within the same run directory.
- **Evaluator timeout:** use its bounded INT/KILL behavior, prove descendants
  are gone, and do not start another episode unless cleanup passed.
- **Artifact or FFprobe failure:** preserve the original files and fail; do not
  transcode or delete evidence in this scope.
- **Keyboard interrupt:** enter the same cleanup state machine and finish the
  manifest before propagating a nonzero exit.

## Verification strategy

### Unit and contract tests

Tests use fake monotonic clocks and a scripted command runner. They cover:

- exact create/inspect configuration, labels, mounts, and no published ports;
- image, source, checkpoint, and GPU UUID mismatch rejection;
- idle configured GPU-7 containers being allowed and active compute processes
  being rejected;
- strict run ID/path validation and existing-output refusal;
- deterministic warm-up serialization and exact `(24, 36)` finite validation;
- lifecycle transitions and persistence before external actions;
- PID reuse, command mismatch, process-start-time mismatch, unknown liveness,
  and foreign-port refusal;
- readiness, SSH, warm-up, evaluator, and artifact timeout paths;
- reverse-order cleanup with one cleanup action failing;
- bounded post-KILL waits and survival detection;
- per-episode routing for exactly episodes 2 and 3;
- refusal to start episode 3 after episode-2 infrastructure cleanup failure;
- task failure versus infrastructure failure classification;
- atomic evidence writes, secret redaction, immutable reruns, and incomplete
  evidence rejection; and
- Ctrl-C during preflight, server load, tunnel readiness, inference, and final
  cleanup.

No unit test imports a simulator, opens SSH, creates a Docker container, or
connects to a real interface.

### Staged integration gates

The runtime is promoted through these gates in order:

1. run focused unit tests, Ruff, formatting, compilation, and whitespace checks;
2. run `status` and preflight read-only against H100;
3. create the container, attest it, and leave it stopped without loading a model;
4. start the server, complete one warm-up, exercise bounded cleanup, and prove
   the container is stopped with no owned resources;
5. inject one tunnel interruption during a non-control probe and prove the
   failure manifest and cleanup behavior; and
6. run official episodes 2 and 3 and validate their standard stereo artifacts.

Each gate writes a new immutable run directory. A later gate does not waive an
earlier failure.

## Acceptance boundary

Completion of this work means a dedicated, reusable H100 evaluation container
and its lifecycle manager passed the staged gates and official episodes 2 and
3 produced fully attributed standard-evaluation evidence. It does not make the
checkpoint safe for the real-time PC2 bridge.

Bridge certification remains separate work requiring same-episode raw and
processed data provenance, proof of the corrected 32-D observation contract,
an attested `/contract` plus `/act-rtc-v1` implementation, and 100 warmed
requests with zero failures and p99 latency at or below 0.10 seconds for
`d=6`. Real-control remains prohibited until that independent gate is approved.
