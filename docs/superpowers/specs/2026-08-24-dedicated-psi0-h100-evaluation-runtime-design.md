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
| Image family (informational) | `pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel` |
| Executable image reference | exact digest from the committed runtime profile |
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

The server snapshot must contain these exact spot-check files and hashes:

```text
src/psi/deploy/psi0_serve_simple.py
  e03c4ee61dd29d95292ec1ce24efe48d6c0d5b6b177a485b0b9e16c44d79cba4

scripts/launch_psi0_server.sh
  0202137143460c305cee39adb1901c7c480caaa7c199e4b4cd77b81cec374579
```

Those two hashes are diagnostic spot checks, not sufficient provenance. The
authoritative input is a committed
`configs/psi0_h100_eval_runtime_v1.json` profile. Before implementation can
create the container, a separate `freeze-provenance` operation must produce and
verify all of the following:

1. an exact repository-digest image reference of the form
   `pytorch/pytorch@sha256:<64 lowercase hex characters>`;
2. the corresponding local Docker image ID;
3. a complete canonical manifest of the PSI0 source tree, its `.venv`, and a
   dedicated offline Hugging Face cache containing every resolved server asset;
4. an immutable remote snapshot named by that manifest's root SHA-256; and
5. Python, PyTorch, CUDA-runtime, package-freeze, server-file, and checkpoint
   identities derived from that snapshot.

The tree manifest includes every regular file's relative path, mode, byte
count, and SHA-256; every symlink's relative path and exact target; and every
directory's relative path and mode. Entries are UTF-8 JSON lines sorted by raw
relative-path bytes, and the root digest is the SHA-256 of those canonical
lines including their terminating newlines. Device nodes, sockets, absolute
symlinks outside the digest-pinned image's read-only `/usr`, `/opt/conda`,
`/lib`, or `/lib64` roots, relative symlinks escaping both the snapshot and
those image roots, and hard links outside the tree are rejected. Absolute
virtual-environment interpreter links into those four image roots are retained
and attested by exact target; the pinned image digest covers their targets.
Only declared generated caches such as `__pycache__`, `.pytest_cache`, and
runtime logs are excluded, and Python is configured not to generate or import
from those excluded locations.

The verified snapshot is atomically installed at:

```text
/mnt/data01/jhkim/psi0-simple-eval-inputs/<tree-root-sha256>/
```

It includes the complete `.venv` used to execute the server and a dedicated
`.hf-cache`. `freeze-provenance` resolves that cache before sealing the snapshot
and then runs the Python/PyTorch/package-freeze and offline dependency-load
probes inside the exact digest-pinned image with the candidate snapshot mounted
read-only. Runtime sets
`HF_HOME=/workspace/Psi0/.hf-cache`, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`; the broad mutable host Hugging Face cache is not
mounted. The container mounts this snapshot, never the mutable
`/home/kube/psi0-train/repo` working tree. The profile records the exact image
digest and ID, tree digest, entry count/bytes, package-freeze hash, checkpoint
hash/size, and both spot hashes. `freeze-provenance` writes a
candidate profile for explicit Git review; it does not update an existing
profile or create a container.

The committed profile has this exact key set and JSON types:

```text
schema_version: integer exactly 1
profile_id: nonempty string
image_reference: digest-qualified string
image_id: "sha256:" plus 64 lowercase hexadecimal characters
source_snapshot_host_path: absolute string ending in the tree-root digest
source_tree_root_sha256: 64-character lowercase hexadecimal string
source_entry_count: positive integer
source_regular_file_bytes: positive integer
package_freeze_sha256: 64-character lowercase hexadecimal string
python_version: nonempty string
torch_version: nonempty string
torch_cuda_version: nonempty string
checkpoint_path: absolute host path string
checkpoint_size: positive integer
checkpoint_sha256: 64-character lowercase hexadecimal string
server_source_sha256: 64-character lowercase hexadecimal string
launcher_source_sha256: 64-character lowercase hexadecimal string
h100_roots_identity_sha256: 64-character lowercase hexadecimal string
simple_source_commit: 40-character lowercase Git SHA
simple_root_tree: 40-character lowercase Git tree SHA
recursive_gitlinks_sha256: 64-character lowercase hexadecimal string
pc2_source_tree_sha256: 64-character lowercase hexadecimal string
pc2_venv_tree_sha256: 64-character lowercase hexadecimal string
pc2_base_python_tree_sha256: 64-character lowercase hexadecimal string
pc2_package_freeze_sha256: 64-character lowercase hexadecimal string
pc2_import_origins_sha256: 64-character lowercase hexadecimal string
pc2_native_closure_sha256: 64-character lowercase hexadecimal string
pc2_episode_data_tree_sha256: 64-character lowercase hexadecimal string
pc2_runtime_identity_sha256: 64-character lowercase hexadecimal string
pc2_python_version: nonempty string
pc2_torch_version: nonempty string
pc2_torch_cuda_version: nonempty string
pc2_mujoco_version: nonempty string
pc2_isaac_version: nonempty string
pc2_nvidia_driver_version: nonempty string
pc2_cuda_driver_version: nonempty string
```

No additional keys, type coercion, mutable tag, relative path, or Boolean as an
integer is accepted. The candidate profile is not active merely because it was
generated. To avoid a self-referential source hash, implementation is first
committed as source commit `I`; the profile names `I`, and a follow-up approval
commit `P` may change only the single profile path. Runtime verifies `P` has
first parent `I`, `git diff --name-only I..P` is exactly the profile path, and
that path's blob and SHA-256 match the reviewed profile. It executes the sealed
source from `I` while reading the profile blob approved by `P`.

`create` and `evaluate` accept only the already committed profile and use the
digest-qualified image reference. They verify the complete snapshot manifest
before container startup and again after cleanup. They never learn or adopt an
expected digest from a mutable tag. A profile, snapshot, or image mismatch is
`PROVENANCE_BLOCKED` and starts no workload.

Container creation records the committed profile hash in a Docker label and in
the manager-only
`/mnt/data01/jhkim/psi0-simple-eval-control/container-identity.json` record.
That record is never container-visible; each run copies its verified redacted
form into local evidence. Reuse requires exact agreement between the control
record, the label, the container image ID, and the committed profile. A missing
identity record blocks reuse rather than silently adopting an existing
container. Creating or replacing the control record is an atomic, fsynced
operation covered by the current lease token and generation.

## Container contract

### Persistent object, ephemeral workload

`jihun_psi0_simple_eval_gpu7` is created once and reused, but it is stopped
outside an evaluation. Its command is an idle process supervised by Docker
`--init`; the policy server never starts from the container command, entrypoint,
restart policy, or a background service.

On first creation, `create` may start the idle container only long enough to
attest the in-container GPU UUID, offline dependency probe, mounts, and
read-only behavior. It then stops the container and succeeds only after Docker
reports exact `exited` state. It never launches the policy server or leaves a
new container in `created` state.

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
| `/mnt/data01/jhkim/psi0-simple-eval-inputs/<tree-root-sha256>` | `/workspace/Psi0` | read-only |
| `/mnt/data01/jhkim/model_weight/Psi0` | `/hfm/cache/checkpoints/psi0` | read-only |
| `/mnt/data01/jhkim/psi0-simple-eval-workloads` | `/runtime` | read-write |

Manager control state is stored separately at
`/mnt/data01/jhkim/psi0-simple-eval-control` and is never mounted into the
container. The sealed input, manager-control, and container-workload roots are
pairwise sibling roots: after resolving existing parents, none may equal or be
an ancestor/descendant of another. The container has no bind mount of a parent
that contains either the input or control root.

The three H100 root directories are an operator-provisioned prerequisite with
fixed owner, group, mode, device, and inode identities recorded in the runtime
profile. The lifecycle manager does not bootstrap or repair the control root
before lease acquisition. A missing, replaced, symlinked, unexpectedly
writable, or permission-mismatched root fails without creating a lease or
starting a workload.

Creation and every reuse audit all Docker bind sources and destinations, their
resolved host paths, propagation, and read-only flags. Every container-visible
alias of the sealed snapshot and checkpoint must be read-only; no alias of the
manager-control root may exist. An in-container mount probe must prove writes
under `/workspace/Psi0` and `/hfm/cache/checkpoints/psi0` fail with a read-only
filesystem error, `/runtime` is writable, and control/input aliases such as
`/runtime/lease.d`, `/runtime/control`, and `/runtime/inputs` do not exist.
Unexpected mounts, overlapping host roots, symlinked aliases, or a successful
input/control write are `PROVENANCE_BLOCKED` before server launch.

`/tmp` is a bounded writable `tmpfs`. `XDG_CACHE_HOME`, `TORCH_HOME`, and
`TRITON_CACHE_DIR` point beneath the current
`/runtime/runs/<run-id>/generated-cache`; caches are never shared across runs.
`HF_HOME` points to the sealed cache inside the read-only snapshot.
`PYTHONDONTWRITEBYTECODE=1` prevents writes to the mounted source tree. A run
writes only beneath `/runtime/runs/<run-id>`. It cannot edit the server source,
dependency environment, checkpoint, or Hugging Face inputs.

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

GPU exclusion continues after startup. The manager records and validates the
H100 process table immediately before server launch, after server readiness,
before and after each episode, immediately before server shutdown, and after
container shutdown. During server warm-up and episode execution it polls at
least every five seconds with a bounded remote helper; a missed or unknown poll
is an infrastructure failure.

Once the server starts, each allowed GPU process must map from the host
`nvidia-smi` PID to an exact PID/start-time/argv in the server's process tree,
carry the current lease token in its owned record, and belong to the dedicated
container cgroup. Container-namespace and host PIDs are correlated through
recorded namespace PID data; process name alone is never sufficient. The
allowed set is the attested server PID and its fully recorded descendants.

A process outside that set is foreign. The manager interrupts only its own
evaluator, tunnel, and server resources and never signals the foreign GPU PID.
It may stop the dedicated container only after proving the foreign process is
outside that container and every process inside the container is owned by the
current lease. A foreign or unknown process inside the dedicated container
triggers the existing `FOREIGN_BLOCKED` rule and leaves the container running.

Failure to query either host or container GPU state is an unknown state and
fails preflight. The manager never starts or stops another container to make
GPU 7 available.

### PC2 GPU selection and exclusion

The local simulator GPU has an equivalent fail-closed gate. After acquiring
the remote lease, the manager resolves PC2 GPU index 1 to its physical UUID and
records the complete local compute-process table. No active compute process may
use that UUID before the allocation probe or immediately before either
episode.

The allocation probe runs in a tracked, bounded subprocess with
`CUDA_VISIBLE_DEVICES=1`. Through the CUDA runtime used by the sealed PC2
environment's PyTorch, it must observe exactly one CUDA device, report its UUID
equal to the selected physical UUID, allocate a small tensor on `cuda:0`, and
synchronize successfully. The probe has a 30-second internal deadline plus a
tracked PID and bounded INT/TERM/KILL cleanup; killing only a wrapper process
is insufficient.

After the probe exits, the manager requires its GPU process to disappear. It
rechecks GPU-1 process ownership before episode 2, after episode-2 cleanup,
before episode 3, and after final cleanup. During an episode, only PIDs in the
recorded evaluator process group/tree may appear on the selected UUID. A
foreign process appearing at any gate is `FOREIGN_BLOCKED`: the manager may
interrupt its own evaluator and clean up its own remote resources, but it never
signals the foreign GPU process. GPU inventory, UUID, probe result, and
before/after process tables are mandatory evidence rather than values inferred
from `CUDA_VISIBLE_DEVICES`.

### PC2 execution provenance

Evaluation does not execute from the editable development worktree or its
shared virtual-environment symlink. The provenance-freeze gate creates a
dedicated PC2 input closure under:

```text
/mnt/data/jihun/psi0-simple-eval-inputs/<pc2-tree-sha256>/source
/mnt/data/jihun/psi0-simple-eval-inputs/<pc2-tree-sha256>/venv
/mnt/data/jihun/psi0-simple-eval-inputs/<pc2-tree-sha256>/episode-data
```

The source snapshot contains only files tracked by the exact SIMPLE commit and
every recursively initialized submodule at the exact gitlink recorded by that
commit. Preparation requires byte-empty output from root
`git status --porcelain=v1 --untracked-files=all` and from the equivalent
recursive submodule checks. It rejects any staged, unstaged, untracked,
missing, prefixed (`+`, `-`, or `U`), dirty, or uninitialized root/submodule
state and records the ordered path/commit/gitlink table. Preserved development
artifacts must be moved outside the worktree before this gate; they are never
silently allowlisted. The snapshot is then copied from Git object identities,
not worktree paths, and verified with a complete canonical file manifest.

The dedicated venv is prepared at its final path before hashing. Editable
`.pth`/installation links are rewritten only during preparation to the sealed
source and exact submodule roots; any path escaping the PC2 input root is
rejected. A clean import-origin probe requires `simple`, `gear_sonic`,
`decoupled_wbc`, `unitree_sdk2py`, `xrobotoolkit`, and other repo-local packages
to resolve beneath the sealed source, and all third-party packages beneath the
sealed venv or an explicitly hashed base-Python/standard-library root. The
complete venv, base interpreter, package freeze, native-extension/shared-library
closure, Isaac/MuJoCo versions, and NVIDIA driver/CUDA identities are recorded
both individually and in the canonical `pc2_runtime_identity_sha256` digest.

The ignored development path
`data/evals/simple-eval/G1WholebodyXMovePickTeleop-v0/dr-level-0` is not assumed
to be part of the Git snapshot. `freeze-provenance` copies it into the sealed
`episode-data` directory using no-follow traversal, rejects special files and
escaping links, and records the same path/mode/size/content canonical manifest
used for other input trees. The profile records that manifest root as
`pc2_episode_data_tree_sha256`. Evaluation reads only this sealed copy; both
the mutable development dataset and any global `data/evals` output remain off
its import and input paths.

The evaluator is invoked as the sealed venv's Python module entry point with
the sealed source as working directory; it does not use the development
worktree's console script. `PYTHONDONTWRITEBYTECODE=1` and run-scoped cache
variables prevent writes to the input closure. The sealed source and venv are
made non-writable and their complete manifests, import origins, package freeze,
git commit/gitlinks, interpreter/shared-library hashes, and driver identities
must match both before the remote lease is acquired and after all evaluation
cleanup.

The committed runtime profile contains the expected PC2 closure identities.
Any dirty root/submodule, gitlink mismatch, escaping import, changed local
environment, source/venv write, or pre/post manifest difference is
`PROVENANCE_BLOCKED`. Run outputs and caches live only under the disjoint
`/mnt/data/jihun/psi0-simple-eval-workloads/<run-id>` root and cannot make the
sealed Git snapshot dirty.

## Remote lease and concurrency

Every mutating operation acquires one host-wide lease before inspecting or
changing the fixed container, GPU, server port, or evaluation targets. The
manager-only control paths are:

```text
/mnt/data01/jhkim/psi0-simple-eval-control/lease.lock
/mnt/data01/jhkim/psi0-simple-eval-control/lease.json
/mnt/data01/jhkim/psi0-simple-eval-control/transactions/
/mnt/data01/jhkim/psi0-simple-eval-control/mutation.lock
```

Every acquire, heartbeat, recovery-claim, cleanup mark, and release is a short
transaction executed by one bounded remote helper. The helper opens
`lease.lock`, takes `fcntl.flock(LOCK_EX)`, rereads `lease.json`, performs its
compare-and-swap, atomically replaces and fsyncs the JSON plus parent directory,
then releases the OS lock. The long-lived ownership is the persisted lease, not
the SSH connection or `flock`. Unknown keys or types invalidate the record.

```text
schema_version: integer exactly 1
lease_token: 32-character lowercase hexadecimal string
generation: positive integer
mode: exactly "normal" or "recovery"
recovery_of_token_sha256: null in normal mode, otherwise 64 lowercase hex
run_id: nonempty validated string
operation: exactly "freeze-provenance", "create", "evaluate", or "stop"
manager_host: nonempty string
manager_pid: positive integer
manager_start_time_ticks: positive integer from PC2 /proc/<pid>/stat
remote_boot_id: canonical UUID string
created_at_utc: UTC RFC3339 string
created_monotonic_ns: positive integer from the H100 host
heartbeat_monotonic_ns: positive integer from the H100 host
cleanup_required: Boolean
```

The holder refreshes the token-checked heartbeat every 10 seconds. The lease
expires after 45 seconds without a valid heartbeat. Loss of the heartbeat is
an infrastructure failure, but expiry alone never authorizes another manager
to stop a process or container. A partially written temporary transaction is
ignored only after its recorded owner helper is proven dead; it is never
treated as a valid lease.

The holder revalidates token, generation, mode, and nonexpired heartbeat
immediately before every Docker start/stop, server launch/signal, tunnel
launch, evaluator launch, and remote-helper signal. A manager whose lease was
replaced cannot perform another mutable action, even if an earlier local
preflight passed.

Every remote helper that changes Docker, processes, snapshots, or control
records also takes `mutation.lock`, revalidates its lease tuple after taking the
lock, and holds the lock through that single bounded mutation and its daemon
postcondition. It revalidates again before starting any later mutation. A
recovery claim may replace the expired lease while an old mutation is already
in flight, but it performs no cleanup signal until the old helper has reached a
terminal state, the recoverer has acquired `mutation.lock`, and all resulting
host/container/process state has been reconciled. Thus the old generation can
finish at most its already-started bounded mutation and cannot race recovery
cleanup or begin another one.

Normal stale reclamation runs under `lease.lock` and replaces the expired owner
with a new normal token and incremented generation only when all of these are
true in the same transaction:

- the heartbeat is expired;
- the recorded remote boot identity and timestamps satisfy the stale rules;
- the dedicated container is absent or exactly `exited`;
- container port 22185 has no listener;
- no server PID record names a live process; and
- the H100 GPU has no process owned by the stale run; and
- no remote helper or helper descendant owned by the stale lease remains.

If the dedicated container is running, `evaluate` never reclaims the lease and
never stops it. An expired lease with a fully attested old PID/container record
is `STALE_OWNED_BLOCKED` and requires the explicit
`stop --recover-stale <run-id>` operation.

That operation first performs a distinct recovery-claim transaction under
`lease.lock`. It requires an expired exact old run ID, exact old token digest
and generation, `cleanup_required=true`, matching container/profile labels,
and complete PID/start-time/argv/cgroup/port/helper records. A running
container is allowed only when every live workload in it is attributable to
the expired lease. The transaction atomically replaces the old owner with one
new `mode="recovery"` token, incremented generation, and
`recovery_of_token_sha256`. This fences the old manager before any signal.
Exactly one concurrent recoverer can win; every loser observes a token or
generation mismatch and performs no mutation.

After claiming, the recoverer revalidates its recovery token immediately
before every signal. It may signal the exact old helper first, wait through the
bounded post-KILL liveness check, acquire `mutation.lock`, reconcile any
daemon-side effect of its last mutation, and only then stop the exact old
server and container resources proven by the claim. Any mismatch, missing
fact, surviving helper, foreign process, foreign port occupant, or unknown
liveness prevents further cleanup or changes the result to `FOREIGN_BLOCKED`;
the container is left running and no foreign process is signalled.

`evaluate` requires the dedicated container to exist and be `exited` at its
initial leased check. It does not accept `created`, `running`, `paused`,
`restarting`, `removing`, `dead`, or unknown states. Cleanup stops the container
only if the current lease token started it and every live process in it is
owned by that same run. Detection of a foreign or unknown workload changes the
terminal state to `FOREIGN_BLOCKED`, closes only resources proven to belong to
the current manager, and deliberately does not stop the container.

The lease is released under `lease.lock` by exact token, generation, and mode
comparison only after the container is verified `exited` and no owned remote
helper remains. If owned cleanup is incomplete, the owner record is atomically
marked `cleanup_required=true`, heartbeat stops, and explicit stale recovery is
required. Read-only `status` never acquires, refreshes, claims, or removes a
lease.

### Remote helper ownership and deadlines

No remote operation relies on the lifetime of its local SSH client. The
manager passes the reviewed remote-helper source directly to a fixed system
Python interpreter and records its SHA-256; it does not install or replace an
executable before acquiring a lease. Every Docker, hashing, snapshot, lease,
and process-control invocation runs through that content-attested helper with
an internal monotonic deadline shorter than the local SSH deadline.

Before doing work, the helper exclusively creates a transaction record under
`/mnt/data01/jhkim/psi0-simple-eval-control/transactions/<helper-id>.json`
containing the helper-source digest, exact lease token
digest/generation/mode, helper ID, run ID, operation, remote boot ID, host PID,
process-group ID, process start ticks, normalized argv and digest, internal
deadline, start time, and
`state="running"`. It fsyncs the record before spawning a child. Every child
runs in the helper's owned process group and is recorded with PID, start time,
argv digest, and parent PID before its result can be accepted.

For acquisition and recovery-claim transactions, those identity fields are
the proposed token, generation, and mode that the same locked compare-and-swap
will install; for all other operations they must equal the current lease.
Transaction creation, compare-and-swap, and terminal update use the
operator-provisioned control root and never a container-visible path.

On its internal deadline, the remote helper applies bounded INT/TERM/KILL to
its own recorded child group, performs a fresh post-KILL wait and liveness
check, and atomically records `completed`, `failed`, `timed_out`, or
`cleanup_required`. Docker operations additionally record and verify the
resulting daemon-side container state; ending a Docker client is not proof that
the daemon operation ended.

The PC2 manager gives SSH an outer deadline at least 15 seconds beyond the
helper's internal deadline. If the SSH process times out or disconnects, the
manager records remote state as unknown, reconnects read-only, and resolves the
transaction by exact helper PID/start-time/argv/token plus daemon postcondition.
It never equates a dead SSH client with a dead remote helper. A live helper
blocks normal lease reclamation. A stale recovery-claim may signal that helper
only after the recovery token fences the old manager and all helper ownership
fields match.

`freeze-provenance` uses the same transaction protocol for tree hashing,
snapshot installation, and offline probes. A crashed freeze therefore leaves
an owned, inspectable helper record; no later create/evaluate or normal stale
reclamation proceeds until the helper is proven gone or recovered explicitly.

## Host-side interface

A Python CLI under `scripts/` owns the complete lifecycle. It uses argument
vector subprocesses, monotonic deadlines, and a replaceable command runner so
unit tests do not require SSH or Docker. It exposes five operations plus one
explicit recovery form:

```text
freeze-provenance
          create a candidate immutable input snapshot/profile for Git review
create    attest inputs and create the stopped container if it is absent
status    report container, GPU, server, tunnel, and artifact state read-only
evaluate  run the complete preflight, server, tunnel, episodes, and cleanup
stop      stop only resources whose current ownership can be proven
stop --recover-stale <run-id>
          recover only a fully attested expired owned workload
```

`evaluate` accepts a unique output root and episode list; the production
profile restricts the first acceptance run to episodes `2,3`. It refuses an
output directory that already exists. It also refuses a local port that is
already bound or any preexisting run-scoped log, video, runtime-evidence, raw,
temporary, or verdict path. It refuses an absent container and directs the
operator to run `create` first. It never inspects or changes prior global
`data/evals` videos. There is no force flag in this scope.

No local command uses `shell=True`. Remote commands are built from fixed
tokens and strictly validated values. A run ID is generated locally from UTC
time, the SIMPLE short SHA, and a random suffix. Generated IDs, profile IDs,
helper IDs, and user-supplied `stop --recover-stale` IDs must match
`[A-Za-z0-9][A-Za-z0-9._-]{0,127}` and must not equal `.` or `..`; separators,
NULs, empty values, Unicode normalization aliases, and dot segments are
rejected before any remote command.

For every input, control, workload, transaction, and evidence path, the manager
resolves the configured root itself with `realpath`, rejects a symlinked root,
constructs exactly one validated child segment, and verifies the existing
parent is the configured root and the candidate is strictly below—not equal to
or outside—that root. Existing recovery paths are resolved without following a
final symlink and must have the same parent. Remote helpers use directory file
descriptors plus no-follow/exclusive operations; string-prefix checks alone are
not accepted. Path-validation failure performs no lease or workload mutation.

## Lifecycle state machine

The manager persists every state transition before performing the next
external action:

```text
NEW -> LOCAL_ATTESTED -> LEASED -> PREFLIGHTED -> CONTAINER_STARTED
    -> SERVER_STARTED -> TUNNEL_READY -> SERVER_READY -> EVALUATING
    -> CLEANING -> CLEAN | FAILED | PROVENANCE_BLOCKED
    | STALE_OWNED_BLOCKED | FOREIGN_BLOCKED
```

Any state may transition to `CLEANING`. A PASS verdict requires the terminal
state `CLEAN`; `FAILED` is used for owned evaluation failure, timeout, cleanup
failure, or evidence failure. Unknown ownership/liveness is
`FOREIGN_BLOCKED`, never an assertion that the resource was stopped. A policy
episode may report task failure without making the infrastructure invalid, but
the manifest records that episode as failed and the overall evaluation verdict
is not PASS.

### Preflight

Before acquiring the remote lease, a read-only local phase verifies the
approved profile commit/blob, sealed SIMPLE source and exact recursive
gitlinks, PC2 venv/base interpreter/import/native closure, configured path
roots, and initial PC2 GPU inventory. Failure reaches `PROVENANCE_BLOCKED`
without touching H100 control state.

After the atomic lease is acquired, the remaining preflight is read-only except
for the tracked PC2 CUDA allocation probe and completes before the container is
started. Across the two phases it must:

1. revalidate the current lease token and heartbeat;
2. confirm bounded SSH connectivity and collect the remote host identity,
   boot ID, and wall clock;
3. attest the complete sealed PC2 source, exact clean recursive gitlinks,
   venv/base-Python/native environment, and import origins against the profile;
4. attest the H100 immutable source, `.venv`, and offline HF snapshot manifest;
5. attest the checkpoint file, size, and hash plus both source spot hashes;
6. verify the exact digest-qualified image reference and image ID from the
   committed profile;
7. require an existing profile-matching container in exact `exited` state for
   `evaluate`, or an absent container for first `create`;
8. resolve H100 GPU 7 to its UUID and prove no active compute process uses it;
9. resolve PC2 GPU 1 to its UUID, prove it has no active compute process, and
   pass the bounded CUDA allocation/UUID probe;
10. prove the local loopback port is free and container port 22185 has no
   listener;
11. prove no live server/helper or stale ownership record conflicts with the
    current lease;
12. confirm and hash all local episode inputs; and
13. confirm both run-scoped video roots, runtime-evidence paths, and all target
    artifacts are absent.

A preflight failure starts no workload. It releases the lease only after the
container is still verified exited and no owned cleanup remains. Merely finding
other containers configured for GPU 7 is diagnostic evidence, not a failure.

### Server startup and readiness

The server command is equivalent to:

```bash
PYTHONPATH=/workspace/Psi0/src \
  /workspace/Psi0/.venv/bin/python -m psi.deploy.psi0_serve_simple \
  --host 0.0.0.0 --port 22185 --device cuda:0 --policy=psi0 \
  --run-dir=<fixed-container-checkpoint-path> --ckpt-step=40000 \
  --action-exec-horizon=24 --rtc
```

It writes its run-specific log beneath `/runtime/runs/<run-id>`, but all
ownership records remain outside the container at
`/mnt/data01/jhkim/psi0-simple-eval-control/runs/<run-id>`. The remote launch
helper records host PID, namespace PID, cgroup/container ID, process start time,
parentage, exact command digest, and lease token/generation before readiness.
A PID is considered owned only when the container identity, manager-only
record, process start time, cgroup, namespace mapping, and normalized command
all agree. Container-writable PID files are diagnostic only and never establish
ownership. A port occupant without complete manager-control proof blocks
startup and is never signalled.

Readiness has three gates:

1. the owned process remains alive and container port 22185 is listening;
2. the SSH tunnel is alive and the PC2 loopback endpoint accepts connections;
3. one schema-valid `/act` warm-up returns HTTP 200 with a finite NumPy action
   of exact shape `(24, 36)`.

The deterministic warm-up uses one black `(360, 640, 3)` `uint8` RGB image
under `rgb_head_stereo_left`. Its `(1, 32)` `float32` state is all zero except
`state[0, 31] = 0.74`, the command-height field. It uses
`instruction="move forward to the table and pick up the object"`,
`history={"reset": true}`, empty condition and ground-truth action,
`dataset_name="simple"`, and
`timestamp="1970-01-01_00-00-00"`. The manager constructs a production
`RequestMessage` and calls its existing `serialize()` method; it does not
maintain a second serializer.

For evidence, the serialized object is encoded as UTF-8 canonical JSON with
sorted keys, compact separators, and no NaN values. `warmup.json` records its
SHA-256 as `canonical_request_sha256` together with the exact instruction,
timestamp, shapes, dtypes, and serializer source hash. Tests freeze all values
and compare the complete serialized object and digest byte-for-byte. The
predicted values are not executed, scored, or treated as checkpoint
certification. The response record contains the structural summary and
latency, and hashes the returned dtype, shape, and contiguous action bytes
without embedding the large tensor in the manifest.

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

### Run-scoped standard videos

`--eval-dir` controls evaluator logs but does not currently control video
placement. The managed path therefore adds an explicit
`--video-output-dir PATH` option to `eval-decoupled-wbc` and the corresponding
frozen `simple.evals.api.EvalConfig` field. Parent kwargs and worker kwargs must
carry the exact path. When set, it replaces the derived global
`data/evals/<policy>/...` video root; it does not change `--eval-dir`.

The lifecycle manager always passes a new absolute path:

```text
<run-dir>/episode_<N>/videos
```

The worker creates the episode subdirectory beneath it, producing these exact
per-camera paths:

```text
<run-dir>/episode_<N>/videos/episode_<N>/<camera>.raw.mp4
<run-dir>/episode_<N>/videos/episode_<N>/<camera>_<verdict>.mp4
```

The output root, episode directory, raw path, temporary transcode path, and
verdict path must all be absent before their owning step. Existing paths are a
collision error; neither `VideoRecorder` nor `VideoWriter` deletes, truncates,
rotates, or overwrites them. An unmanaged caller that omits
`--video-output-dir` retains the legacy derived root but receives the same
no-overwrite and checked-finalization behavior.

The managed evaluator constructs `VideoRecorder` with deferred initialization.
Its reset does not open a provisional writer. After stabilization completes,
the worker calls one explicit `start(observation)` operation that exclusively
creates each raw file and seeds frame zero. The existing second `_init_writers`
delete/reopen pattern is removed from this path. Calling `start` twice, stepping
before `start`, or resetting an active recorder is an error; no stabilization
file is created and then deleted.

`VideoWriter.release` becomes a bounded checked operation using subprocess
argv, never shell strings. It first releases OpenCV and verifies the raw MP4 is
nonempty. FFmpeg runs with `-nostdin`, a 120-second monotonic deadline, captured
logs, and a tracked process subject to bounded INT/TERM/KILL cleanup. It writes
an absent `<camera>_<verdict>.tmp.mp4`, and the final path appears by atomic
rename only after FFmpeg exits zero and FFprobe validates codec, dimensions,
positive frame count, and positive duration.

The raw MP4 is retained after both successful and failed finalization. On
timeout, nonzero exit, malformed output, write failure, or missing FFmpeg, the
temporary output is preserved with a diagnostic suffix, the raw MP4 remains,
and the episode reports infrastructure failure. `VideoRecorder.release`
attempts every camera and returns structured results; one camera failure does
not skip closure of the others. Worker evaluation wraps recording, result
persistence, and environment closure in `try/finally`, so render, write,
transcode, persistence, and interruption failures all close writers and
preserve raw evidence.

Because every retry has a new immutable run directory, retry never touches an
earlier raw or verdict video. This hardening applies only to the ordinary
evaluation cameras; it does not implement the deferred third-person camera.

### Episode evaluation

Episodes 2 and 3 run separately in ascending order, giving each an independent
20-minute deadline and exit record. The command is based on the already proven
episode-1 path:

```bash
SIMPLE_DISABLE_TUI=1 CUDA_VISIBLE_DEVICES=1 \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=<run-dir>/generated-cache/matplotlib \
PYTHONPATH=<sealed-source-and-submodule-paths> \
  <sealed-pc2-venv>/bin/python -m simple.cli.eval_decoupled_wbc \
  simple/G1WholebodyXMovePickTeleop-v0 psi0_decoupled_wbc train \
  --data-format lerobot \
  --data-dir <sealed-pc2-input>/episode-data \
  --host 127.0.0.1 --port <local-port> \
  --sim-mode mujoco_isaac --headless \
  --eval-dir <run-dir>/episode_<N>/eval-logs \
  --video-output-dir <run-dir>/episode_<N>/videos \
  --runtime-evidence-path \
    <run-dir>/episode_<N>/wbc-runtime-contract.json \
  --runtime-evidence-run-id <run-id> \
  --runtime-evidence-nonce <episode-nonce> \
  --num-episodes 1 --episode-start <N> --num-workers 1 --save-video
```

The displayed environment assignments describe the manager's explicit
subprocess environment; production passes an argv plus environment mapping and
does not invoke a shell or an external `timeout` wrapper. The lifecycle manager
owns the evaluator process-group deadline and escalation directly.

There is no `--third-person-video` flag. The evaluator must retain
`sim`/`lo`/domain 0 isolation and must not open a real Unitree interface.
Immediately before `gym.make`, the worker atomically emits the actual
`sonic_config` fields it will consume to the episode's runtime-evidence file.
The manager requires `ENV_TYPE="sim"`, `INTERFACE="lo"`, and `DOMAIN_ID=0`
from that worker-produced record; a separately reconstructed default or a
hard-coded zero counter is not acceptable evidence. The worker refuses any
other values before environment or channel construction.

The three runtime-evidence options are supplied together or rejected by the
parent before spawning. `run-id` must equal the manager's run ID, and the
episode nonce is a fresh 128-bit lowercase hexadecimal value used only once.
The exact JSON schema is:

```text
schema_version: integer exactly 1
run_id: string
episode_index: integer
nonce: 32-character lowercase hexadecimal string
worker_id: integer exactly 0
worker_pid: positive integer
created_at_utc: UTC RFC3339 string
created_at_monotonic_ns: positive integer
simple_commit: 40-character lowercase Git SHA
config: exact object {ENV_TYPE: string, INTERFACE: string, DOMAIN_ID: integer}
status: exactly "validated" or "rejected"
error: null for validated, nonempty string for rejected
```

No additional keys or Boolean-as-integer values are accepted. The worker
creates a nonce-named temporary file with exclusive creation, flushes and
fsyncs it, links it to the absent final path without replacement, fsyncs the
directory, and removes the temporary name. It completes this operation before
calling `gym.make`. Unsafe values write `status="rejected"` and raise before
`gym.make`; a preexisting final file, missing option, write failure, or invalid
nonce also raises before `gym.make`.

The manager accepts only a `validated` record whose run ID, episode index,
nonce, worker ID/PID, commit, exact config, file identity, and creation interval
match the current evaluator. The file must be created after that evaluator's
recorded process start and before its first environment-construction progress
event. The current `worker_init(status="creating_env")` report is moved: the
worker must durably create the evidence file, emit
`runtime_contract(status="validated")`, then emit
`worker_init(status="creating_env")`, and only then call `gym.make`.

For an unsafe contract, the worker durably creates the rejected record, emits
`runtime_contract(status="rejected")`, and raises to the existing worker-error
path. It emits no `creating_env` event and makes zero `gym.make` calls. Missing,
stale, replayed, replaced, or cross-episode evidence fails the run. Episodes 2
and 3 have different paths and nonces; neither record can satisfy the other
episode.

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
deadline through nested cleanup operations. A remote row has both an internal
H100-helper deadline and a longer PC2 SSH observation deadline; expiry of the
latter triggers helper reconciliation, not an assumption that remote work
stopped.

| Operation | Internal/owned deadline | PC2 outer deadline |
| --- | ---: | ---: |
| Lease or short remote inspection helper | 10 s | 25 s |
| H100 GPU ownership poll | 10 s | 25 s |
| Complete remote snapshot verification | 600 s | 615 s |
| Remote provenance freeze/copy/probe | 1800 s | 1815 s |
| Remote Docker start or stop helper | 30 s | 45 s |
| PC2 CUDA allocation/UUID probe | 30 s | 45 s process reconciliation |
| Model process and TCP readiness | 300 s total | owned server reconciliation |
| Warm-up `/act` response | 120 s | owned server reconciliation |
| Each evaluator episode | 1200 s | 75 s owned process-group escalation |
| Each FFmpeg finalization | 120 s | bounded owned signal stages |
| Tunnel signal stage | 2 s each for INT, TERM, and KILL | fresh liveness check |
| Server signal stage | 5 s each for INT, TERM, and KILL | fresh liveness check |
| Final remote cleanup verification | 30 s | 45 s |

Cleanup is registered before each resource is started and executes in reverse
order. It is attempted in this order:

1. interrupt and reap the active evaluator and its owned children;
2. close the SSH tunnel;
3. stop the owned server inside the container, unless ownership has become
   foreign or unknown;
4. copy final remote evidence;
5. stop the dedicated container only when the current lease started it and all
   remaining workloads are owned by the current run; and
6. verify local and remote postconditions.

INT, TERM, and KILL are used only after ownership is revalidated at that
stage. After KILL, a fresh bounded wait and liveness check are mandatory.
Failure in one cleanup action is recorded but does not skip later actions.
Terminal restoration, open log closure, and manifest finalization are
unconditional even after the shared cleanup budget is exceeded.

If either ownership check in steps 3 or 5 is foreign or unknown, the manager
records `FOREIGN_BLOCKED` and does not send a signal or stop the container. It
still closes its proven local evaluator/tunnel resources and finalizes
evidence. This exception to the desired exited postcondition is always a
non-PASS terminal state and leaves the lease marked for operator inspection.

The persistent container object remains for reuse, but its required terminal
state for PASS is `exited`. A run cannot pass if the container is running, the
owned server or tunnel is alive, either relevant port is listening, an
evaluator or owned WBC child remains, either selected GPU retains an owned
process, the complete source snapshot changed, or liveness is unknown.

## Evidence contract

Local evidence is immutable beneath:

```text
/mnt/data/jihun/psi0-simple-eval-workloads/<run-id>/
```

Remote transient evidence is beneath:

```text
/mnt/data01/jhkim/psi0-simple-eval-workloads/runs/<run-id>/
```

The local run directory contains at least:

```text
manifest.json
run-manifest.md
lease/
  acquisition.json
  heartbeats.jsonl
  recovery-claim.json
  final.json
preflight/
  ssh-host.json
  runtime-profile.json
  profile-hash.json
  h100-roots-identity.json
  container-identity.json
  pc2-source-before.json
  pc2-gitlinks-before.json
  pc2-venv-before.json
  pc2-import-origins.json
  pc2-native-closure.json
  pc2-episode-data-before.json
  episode-inputs.json
  source-tree-manifest.json
  source-tree-verification-before.json
  h100-gpu-before.json
  pc2-gpu-before.json
  pc2-cuda-probe.json
  configured-gpu7-containers.json
  container-inspect.json
  source-hashes.json
  checkpoint.json
server/
  command.json
  pid-namespace-cgroup-map.json
  server.log
  readiness.json
  warmup.json
  h100-gpu-monitor.jsonl
remote-helpers/
  transactions.json
  final-liveness.json
tunnel/
  process.json
  tunnel.log
episode_2/
  command.json
  evaluator.log
  wbc-runtime-contract.json
  h100-gpu-before.json
  h100-gpu-after.json
  pc2-gpu-before.json
  pc2-gpu-after.json
  result.json
  artifacts.json
  videos/episode_2/<camera>.raw.mp4
  videos/episode_2/<camera>_<verdict>.mp4
episode_3/
  command.json
  evaluator.log
  wbc-runtime-contract.json
  h100-gpu-before.json
  h100-gpu-after.json
  pc2-gpu-before.json
  pc2-gpu-after.json
  result.json
  artifacts.json
  videos/episode_3/<camera>.raw.mp4
  videos/episode_3/<camera>_<verdict>.mp4
cleanup/
  actions.json
  h100-gpu-after.json
  pc2-gpu-after.json
  pc2-source-after.json
  pc2-gitlinks-after.json
  pc2-venv-after.json
  pc2-episode-data-after.json
  source-tree-verification-after.json
  remote-helpers-final.json
  container-final.json
  processes-final.json
  ports-final.json
```

JSON documents have a schema version and use atomic write-then-rename. The
final manifest records every external command in redacted argv form, start and
finish timestamps, monotonic durations, return code or timeout, identities,
hashes, state transitions, episode verdicts, cleanup attempts, and unresolved
errors. The evidence retains only the lease-token SHA-256 after release, never
an authentication token, SSH private material, environment secret, raw lease
token, or full image/action payload.

For every raw and finalized head-stereo MP4, the artifact record includes its
run-scoped absolute path, role, byte count, SHA-256, codec, resolution, frame
rate, frame count, duration, FFmpeg/FFprobe logs, and finalization status.
First, middle, and final frames of each successful verdict artifact must decode
and be nonblank. This is an automated integrity check, not a new third-person
or semantic-success classifier.

The Markdown manifest is a human-readable rendering of `manifest.json`; it is
not an independent source of truth. Existing failed or successful run
directories are never overwritten or deleted.

## Verdict rules

Infrastructure PASS requires all of the following:

- one exclusive remote lease covered every mutable action and was released by
  matching token after clean shutdown;
- every preflight and identity assertion passed;
- all H100 input/control/workload and PC2 input/workload roots were disjoint,
  with no writable container alias to an input or control root;
- the digest-pinned image and complete immutable PSI0 source plus `.venv`
  snapshot matched the committed profile before and after execution;
- the sealed SIMPLE source, exact recursive gitlinks, PC2 venv/base Python,
  package/import/native closure, and driver identities matched the committed
  profile before and after execution;
- container CUDA device 0 UUID equalled host GPU 7 UUID;
- every H100 GPU poll contained only the exact attested server process tree;
- PC2 CUDA device 0 under `CUDA_VISIBLE_DEVICES=1` equalled physical GPU 1's
  UUID, passed allocation, and had no foreign compute process at any gate;
- the canonical warm-up digest matched and returned a finite `(24, 36)` action;
- both episode evaluator processes reached a normal recorded exit;
- each episode produced a recorded SIMPLE verdict, retained raw stereo MP4s,
  and valid checked verdict MP4s under its unique run directory;
- worker-emitted runtime evidence proved the actual WBC configuration was
  `sim` / `lo` / domain 0 before each environment was constructed, with a
  distinct current nonce for each episode;
- no real-robot control process or non-loopback Unitree interface owned by the
  run was observed;
- every owned process and port was absent after cleanup;
- every remote helper and helper descendant reached a recorded terminal state;
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

- **Lease contention:** return the current lease metadata and
  `FOREIGN_BLOCKED`; do not run preflight mutations or cleanup another manager.
- **VPN or SSH loss before startup:** fail without starting anything.
- **VPN loss after startup:** continue local cleanup, repeatedly attempt only
  owned remote cleanup within the shared deadline, then record remote state as
  unknown if it cannot be attested. Unknown is `FOREIGN_BLOCKED`, not stopped.
- **Either GPU becomes busy between probes:** stop or avoid starting only the
  current run's resources. Never signal a foreign GPU process. Do not start the
  next episode until both GPUs pass their required gates.
- **Provenance changes:** reject before startup when found initially; if the
  post-run complete tree verification differs, preserve both manifests and fail
  even when evaluation otherwise succeeded.
- **Unexpected server or port occupant:** preserve diagnostics and do not
  signal it or stop a container holding it.
- **Server crash or invalid warm-up:** preserve the complete server log and
  clean up without launching an evaluator.
- **Tunnel failure during evaluation:** interrupt the evaluator, then perform
  normal cleanup; do not retry within the same run directory.
- **Evaluator timeout:** use its bounded INT/KILL behavior, prove descendants
  are gone, and do not start another episode unless cleanup passed.
- **Artifact, FFmpeg, or FFprobe failure:** preserve the raw and diagnostic
  temporary files, close every writer, and fail without deleting or replacing
  evidence.
- **Keyboard interrupt:** enter the same cleanup state machine and finish the
  manifest before propagating a nonzero exit.

## Verification strategy

### Unit and contract tests

Tests use fake monotonic clocks and a scripted command runner. They cover:

- atomic lease acquisition by two concurrent managers with exactly one winner;
- transaction-lock serialization, heartbeat token/generation checks,
  interrupted transaction cleanup, expired/exited stale recovery, active-lease
  refusal, and compare-and-swap loss;
- two concurrent stale recoverers with exactly one recovery-claim winner, the
  old manager fenced before signals, and every loser issuing zero signals;
- running-container stale leases producing `STALE_OWNED_BLOCKED`, and foreign
  or unknown process/port cases producing `FOREIGN_BLOCKED` without any stop or
  signal command;
- pairwise-disjoint H100 roots, every Docker bind alias, read-only input writes,
  inaccessible control paths, writable workload path, exact labels/mounts, and
  no published ports;
- predetermined image-digest enforcement, complete source plus `.venv` manifest
  generation, immutable snapshot installation, and rejection of a changed
  source file, dependency file, symlink, image, or checkpoint before startup;
- idle configured GPU-7 containers being allowed and active compute processes
  being rejected;
- H100 host/container PID namespace and cgroup mapping, pre-server and
  before/during/after-episode polls, server-descendant allowlisting, missed-poll
  failure, and foreign-process handling that stops only owned resources;
- PC2 GPU-1 UUID mapping, CUDA allocation success, UUID mismatch, probe timeout
  cleanup, active foreign-process refusal, and every before/between/after
  recheck without signalling foreign users;
- sealed PC2 SIMPLE source construction from Git objects, exact recursive clean
  gitlinks, dirty/untracked submodule refusal, venv/base-Python/native manifest,
  editable-path escape rejection, import-origin validation, ignored episode-data
  sealing, and source/environment/dataset pre/post mutation detection;
- strict run/helper/profile/recovery ID validation, `.`/`..` and symlink escape
  rejection, resolved-root containment, and existing-output refusal;
- byte-exact production warm-up serialization including instruction and fixed
  timestamp, canonical request digest, and exact `(24, 36)` finite validation;
- `--video-output-dir` parent-to-worker routing, run-scoped episode paths,
  deferred single initialization after stabilization, no-overwrite behavior,
  raw retention after success and failure, checked transcode success,
  timeout/nonzero/malformed-output handling, and independent opposite-verdict
  retries;
- exact runtime-evidence option pairing and schema/type validation;
- unsafe `ENV_TYPE`, interface, and domain values each failing before
  `gym.make`, with the rejected record present and no `creating_env` event;
- validated event order of durable evidence, `runtime_contract`,
  `worker_init(creating_env)`, then `gym.make`, plus rejected event order of
  durable evidence, `runtime_contract(rejected)`, then worker error;
- missing, stale, replayed, wrong-PID, wrong-commit, wrong-nonce, and replaced
  runtime evidence rejection, plus independent valid records for episodes 2
  and 3;
- lifecycle transitions and persistence before external actions;
- PID reuse, command mismatch, process-start-time mismatch, unknown liveness,
  and foreign-port refusal;
- readiness, SSH, warm-up, evaluator, FFmpeg, and artifact timeout paths;
- remote helper internal timeout, SSH outer timeout, reconnection/liveness
  reconciliation, Docker daemon postcondition, and crashed
  `freeze-provenance` blocking stale reclamation until exact recovery;
- reverse-order cleanup with one cleanup action failing;
- bounded post-KILL waits and survival detection;
- per-episode routing for exactly episodes 2 and 3;
- refusal to start episode 3 after episode-2 infrastructure cleanup failure;
- task failure versus infrastructure failure classification;
- atomic evidence writes, secret redaction, immutable reruns, and incomplete
  evidence rejection;
- render, video-write, result-persistence, and environment-close failures still
  closing all writers and preserving raw videos; and
- Ctrl-C during lease acquisition, preflight, server load, tunnel readiness,
  inference, video finalization, and final cleanup.

No unit test imports a simulator, opens SSH, creates a Docker container, or
connects to a real interface.

### Staged integration gates

The runtime is promoted through these gates in order:

1. run focused unit tests, Ruff, formatting, compilation, and whitespace checks;
2. seal the exact SIMPLE source commit plus recursive gitlinks and dedicated
   PC2 environment, run remote `freeze-provenance`, verify both complete
   closures and the digest-qualified image, then commit only the candidate
   profile as approval commit `P` without starting the dedicated container;
3. run two concurrent normal lease probes and two concurrent recovery-claim
   probes, require exactly one winner in each case, prove the old token is
   fenced, and release cleanly;
4. run `status` and local/leased preflight, including path containment, mount
   alias, GPU, remote-helper, and pre/post provenance checks;
5. create the container, attest every mount alias and the in-container
   read-only/inaccessible probes, and leave it exited without loading a model;
6. start the server, complete one warm-up, exercise H100 GPU monitoring and
   bounded cleanup, and prove
   the container is stopped with no owned resources;
7. inject one tunnel interruption and one internally timed-out remote helper,
   then prove remote liveness reconciliation, failure evidence, and cleanup;
   and
8. run official episodes 2 and 3 and validate their run-scoped raw and checked
   standard stereo artifacts, independent WBC evidence, GPU evidence, lease
   release, zero live remote helpers, and both post-run execution-closure
   manifests.

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
