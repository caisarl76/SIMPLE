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

- modify or train PSI0; the fixed official server sources and their spot hashes
  remain byte-identical, and checkpoint-load evidence is collected externally;
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

The mutable checkpoint intake path used only by `freeze-provenance` is:

```text
/mnt/data01/jhkim/model_weight/Psi0/simple-checkpoints/
  g1wholebodyxmovepick-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2604022205
```

It is never visible to the runtime container. The reviewed profile selects one
exact weight relative path within the protected content-addressed copy. That
named weight's required size and SHA-256 are:

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
4. an immutable remote server snapshot named by that manifest's root SHA-256;
5. a separate, complete checkpoint snapshot named by its canonical tree
   SHA-256 and containing the exact profiled weight relative path; and
6. Python, PyTorch, CUDA-runtime, package-freeze, server-file, and checkpoint
   identities derived only from those protected snapshots.

The authoritative tree manifest is computed only from the final normalized
payload metadata. It includes every regular file's relative path, final mode,
byte count, and SHA-256; every symlink's relative path and exact target; and
every directory's relative path and final mode. Entries are UTF-8 JSON lines
sorted by raw relative-path bytes, and the root digest is the SHA-256 of those
canonical lines including their terminating newlines. Installer completion
metadata is outside the payload namespace and explicitly excluded from the
payload manifest/root digest; its bytes and hash are attested separately.
H100 completion metadata has one exact manager-only location per final digest:

```text
/mnt/data01/jhkim/psi0-simple-eval-control/input-completions/
  server/<source-tree-root-sha256>.json
/mnt/data01/jhkim/psi0-simple-eval-control/input-completions/
  checkpoint/<checkpoint-tree-root-sha256>.json
```

It is root-owned, never part of or below a payload snapshot, and never mounted
into the container. Device nodes, sockets, absolute symlinks outside the
digest-pinned image's read-only `/usr`, `/opt/conda`,
`/lib`, or `/lib64` roots, relative symlinks escaping both the snapshot and
those image roots, and hard links outside the tree are rejected. Absolute
virtual-environment interpreter links into those four image roots are retained
and attested by exact target; the pinned image digest covers their targets.
Only declared generated caches such as `__pycache__`, `.pytest_cache`, and
runtime logs are excluded, and Python is configured not to generate or import
from those excluded locations.

The verified server snapshot is atomically installed at:

```text
/mnt/data01/jhkim/psi0-simple-eval-inputs/server/<tree-root-sha256>/
```

It includes the complete `.venv` used to execute the server and a dedicated
`.hf-cache`. `freeze-provenance` resolves that cache before sealing the snapshot
in staging. After the installer returns the authoritative final receipt, freeze
runs the Python/PyTorch/package-freeze and offline dependency-load probes inside
the exact digest-pinned image with the protected final snapshot mounted
read-only. Runtime sets
`HF_HOME=/workspace/Psi0/.hf-cache`, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`; the broad mutable host Hugging Face cache is not
mounted. The container mounts this snapshot, never the mutable
`/home/kube/psi0-train/repo` working tree.

The reviewed seccomp JSON is materialized during that same freeze. The manager
copies the exact committed `configs/psi0_h100_eval_seccomp_v1.json` bytes into
server staging at `runtime-tools/psi0_h100_eval_seccomp_v1.json`; the
privileged H100 installer publishes it as part of the protected source
payload. The committed profile requires:

```text
container_seccomp_profile_host_path =
  <source_snapshot_host_path>/runtime-tools/psi0_h100_eval_seccomp_v1.json
```

`container_seccomp_profile_sha256` is its file digest, while
`source_installer_receipt_sha256` and `source_completion_sha256` bind its path,
mode, bytes, complete containing tree, and protected publication. There is no
mutable workspace copy, upload, or run-selected seccomp file in `create`.

The checkpoint is frozen separately under:

```text
/mnt/data01/jhkim/psi0-simple-eval-inputs/checkpoint/<checkpoint-tree-root-sha256>/
```

The freeze operation no-follow copies the complete selected checkpoint run,
including configuration, metadata, and every weight shard, from the mutable
model-weight tree into a token-named staging directory. It creates a
content-only intake manifest and requires one normalized, nonempty, traversal-free
`checkpoint_weight_relative_path` to identify the 6.25 GB weight file within
that tree. The named file's type, relative path, size, and digest must agree
with that intake manifest; the authoritative mode-inclusive manifest is
created later by the installer. The mutable source checkpoint tree is never mounted or
read during server startup or evaluation.

Both server and checkpoint snapshots are promoted by an operator-owned input
installer after intake verification. Their payloads are owned by a dedicated
input UID; directories are normalized to `0555`, checkpoint regular files to
`0444`, and server regular files to `0444` or descriptor-allowlisted `0555`.
Their parents cannot be renamed by the lifecycle account, and ACL/mode write
probes by both the lifecycle account and container must fail. Installation uses
a same-device token-named staging tree and an absent content-addressed target.
The installer capability
is absent from `create`, `evaluate`, and the server container. An existing
target is adopted only if its completion metadata and complete manifest are
byte-identical; otherwise provenance freezes fail closed. Consequently, a
second ordinary process running as the lifecycle account cannot change a
checkpoint during model loading and restore it later.

The installer interface is the operator-provisioned, root-owned executable
`/usr/local/sbin/psi0-eval-install-input`, invoked through one exact
noninteractive allowlisted argv rather than a shell. The profile records its
regular-file SHA-256. It accepts only a current freeze helper ID/token,
`server` or `checkpoint`, a resolved candidate beneath that run's remote
workload staging root, the expected content-only intake-manifest digest, and
the configured input kind root. It independently revalidates the
manager-control transaction, no-follow intake manifest, same-device
destination, and path containment. It then applies the canonical ownership and
mode policy and fsyncs the payload before computing the authoritative
mode-inclusive manifest and its root digest. That digest selects the absent
destination. Only afterward does it make the snapshot root non-writable,
atomically rename and fsync both parents, then exclusively create/fsync the
separately hashed manager-only completion metadata as the final mutation. Its
durable receipt contains the final tree digest, manifest hash,
completion-metadata hash, root identity, and mode-policy version; only this
receipt may populate the candidate profile. The root-owned,
lifecycle-readable but non-writable receipts are exclusively created at:

```text
/mnt/data01/jhkim/psi0-simple-eval-control/input-installations/
  server/<source-tree-root-sha256>.json
/mnt/data01/jhkim/psi0-simple-eval-control/input-installations/
  checkpoint/<checkpoint-tree-root-sha256>.json
```

Their hashes are `source_installer_receipt_sha256` and
`checkpoint_installer_receipt_sha256`; neither file is mounted in the
container. The installer's fixed policy cannot install arbitrary
source/destination paths or
mutate an existing target. The lifecycle account has no general sudo or input-
root write permission.

The profile records the exact image digest and ID, server-tree digest, entry
count/bytes, package-freeze hash, protected checkpoint snapshot and named
weight identities, final mode-policy versions, external completion-metadata
hashes, H100 installer-receipt hashes, and both spot hashes. `freeze-provenance` writes a
candidate profile for explicit Git review; it does not update an existing
profile or create a container.

The committed profile has this exact key set and JSON types:

```text
schema_version: integer exactly 1
profile_id: nonempty string
image_reference: digest-qualified string
image_id: "sha256:" plus 64 lowercase hexadecimal characters
source_snapshot_host_path: absolute string ending in /server/<tree-root digest>
source_tree_root_sha256: 64-character lowercase hexadecimal string
source_entry_count: positive integer
source_regular_file_bytes: positive integer
source_completion_sha256: 64-character lowercase hexadecimal string
source_installer_receipt_sha256: 64-character lowercase hexadecimal string
source_mode_policy_version: positive integer
package_freeze_sha256: 64-character lowercase hexadecimal string
python_version: nonempty string
torch_version: nonempty string
torch_cuda_version: nonempty string
checkpoint_snapshot_host_path: absolute string ending in /checkpoint/<checkpoint-tree digest>
checkpoint_weight_relative_path: normalized nonempty relative path with no dot segments
checkpoint_entry_count: positive integer
checkpoint_regular_file_bytes: positive integer
checkpoint_size: positive integer
checkpoint_sha256: 64-character lowercase hexadecimal string
checkpoint_tree_root_sha256: 64-character lowercase hexadecimal string
checkpoint_completion_sha256: 64-character lowercase hexadecimal string
checkpoint_installer_receipt_sha256: 64-character lowercase hexadecimal string
checkpoint_mode_policy_version: positive integer
checkpoint_tracer_path: absolute in-container path string
checkpoint_tracer_sha256: 64-character lowercase hexadecimal string
checkpoint_tracer_version: nonempty string
checkpoint_tracer_argv_sha256: 64-character lowercase hexadecimal string
checkpoint_tracer_probe_sha256: 64-character lowercase hexadecimal string
checkpoint_tracer_probe_sentinel_sha256: 64-character lowercase hexadecimal string
checkpoint_tracer_probe_argv_sha256: 64-character lowercase hexadecimal string
container_seccomp_profile_host_path: absolute string beneath source_snapshot_host_path
container_seccomp_profile_sha256: 64-character lowercase hexadecimal string
container_security_contract_sha256: 64-character lowercase hexadecimal string
server_source_sha256: 64-character lowercase hexadecimal string
launcher_source_sha256: 64-character lowercase hexadecimal string
h100_roots_identity_sha256: 64-character lowercase hexadecimal string
h100_input_installer_sha256: 64-character lowercase hexadecimal string
simple_source_commit: 40-character lowercase Git SHA
simple_root_tree: 40-character lowercase Git tree SHA
recursive_gitlinks_sha256: 64-character lowercase hexadecimal string
pc2_closure_id: 64-character lowercase hexadecimal string
pc2_input_host_path: absolute string ending in pc2_closure_id
pc2_source_tree_sha256: 64-character lowercase hexadecimal string
pc2_venv_tree_sha256: 64-character lowercase hexadecimal string
pc2_base_python_host_path: absolute string ending in pc2_base_python_tree_sha256
pc2_base_python_tree_sha256: 64-character lowercase hexadecimal string
pc2_base_python_loader_relative_path: normalized nonempty relative path with no dot segments
pc2_base_python_loader_sha256: 64-character lowercase hexadecimal string
pc2_base_python_completion_sha256: 64-character lowercase hexadecimal string
pc2_base_python_root_identity_sha256: 64-character lowercase hexadecimal string
pc2_base_python_installer_receipt_sha256: 64-character lowercase hexadecimal string
pc2_package_freeze_sha256: 64-character lowercase hexadecimal string
pc2_import_origins_sha256: 64-character lowercase hexadecimal string
pc2_native_closure_sha256: 64-character lowercase hexadecimal string
pc2_episode_data_tree_sha256: 64-character lowercase hexadecimal string
pc2_task_assets_tree_sha256: 64-character lowercase hexadecimal string
pc2_asset_requirements_sha256: 64-character lowercase hexadecimal string
pc2_asset_normalization_results_sha256: 64-character lowercase hexadecimal string
pc2_runtime_identity_sha256: 64-character lowercase hexadecimal string
pc2_runtime_identity_sidecar_sha256: 64-character lowercase hexadecimal string
pc2_closure_completion_sha256: 64-character lowercase hexadecimal string
pc2_closure_root_identity_sha256: 64-character lowercase hexadecimal string
pc2_roots_identity_sha256: 64-character lowercase hexadecimal string
pc2_input_installer_sha256: 64-character lowercase hexadecimal string
pc2_installer_config_sha256: 64-character lowercase hexadecimal string
pc2_installer_service_unit_sha256: 64-character lowercase hexadecimal string
pc2_installer_socket_unit_sha256: 64-character lowercase hexadecimal string
pc2_installer_receipt_sha256: 64-character lowercase hexadecimal string
pc2_runner_launcher_sha256: 64-character lowercase hexadecimal string
pc2_runner_config_sha256: 64-character lowercase hexadecimal string
pc2_runner_service_unit_sha256: 64-character lowercase hexadecimal string
pc2_runner_socket_unit_sha256: 64-character lowercase hexadecimal string
pc2_runner_sandbox_contract_sha256: 64-character lowercase hexadecimal string
pc2_evaluator_uid: positive integer
pc2_evaluator_gid: positive integer
pc2_mode_policy_version: positive integer
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
and separately hashed external completion metadata before container startup
and again after cleanup. They never learn or adopt an
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
read-only behavior, plus the model-free interruptible tracer detach probe. It
then stops the container and succeeds only after Docker
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
- `--cap-drop=ALL --cap-add=SYS_PTRACE`, no `--privileged`, no host PID
  namespace, and `--security-opt=no-new-privileges:true`;
- the committed `configs/psi0_h100_eval_seccomp_v1.json` supplied through
  `--security-opt=seccomp=<container_seccomp_profile_host_path>` from the
  protected H100 server snapshot, with its exact bytes hashed as
  `container_seccomp_profile_sha256`;
- fixed ownership labels for SIMPLE, the selected host GPU UUID, image ID, and
  runtime-contract version, seccomp digest, and canonical security-contract
  digest; and
- the mounts below.

The seccomp file is the reviewed Docker-default-compatible allowlist for the
digest-pinned image with only the `ptrace` rule needed by the direct-child
checkpoint tracer added. It is never `unconfined`. The canonical security
contract records the exact capability add/drop sets, seccomp digest,
`no-new-privileges`, PID/IPC/network modes, read-only-root flag, and every
tmpfs/mount setting; its digest is `container_security_contract_sha256`.
The remote Docker CLI no-follow opens and hashes the exact protected host file
immediately before `docker create`; Docker reads that H100-client path and
sends its contents to the daemon. The helper reads/hashes it again after create
and requires the source installer receipt/completion and full source tree to
remain exact. Create and reuse compare the daemon-side inspect result and
digest label field-for-field with that contract. Docker's default seccomp policy is an allowlist and may block
`ptrace`, so merely adding `SYS_PTRACE` without the reviewed seccomp profile is
not accepted; the security assumptions follow Docker's
[seccomp documentation](https://docs.docker.com/engine/security/seccomp/).

| Host path | Container path | Access |
| --- | --- | --- |
| exact profiled `/mnt/data01/jhkim/psi0-simple-eval-inputs/server/<tree-root-sha256>` | `/workspace/Psi0` | read-only |
| exact profiled `/mnt/data01/jhkim/psi0-simple-eval-inputs/checkpoint/<checkpoint-tree-root-sha256>` | `/checkpoint` | read-only |
| `/mnt/data01/jhkim/psi0-simple-eval-workloads` | `/runtime` | read-write |

Manager control state is stored separately at
`/mnt/data01/jhkim/psi0-simple-eval-control` and is never mounted into the
container. The sealed input, manager-control, and container-workload roots are
pairwise sibling roots: after resolving existing parents, none may equal or be
an ancestor/descendant of another. The container has no bind mount of a parent
that contains either the input or control root.

The three H100 root directories are an operator-provisioned prerequisite with
fixed owner, group, mode, device, and inode identities recorded in the runtime
profile. The input and workload roots are required to reside on the same
filesystem for installer publication; the manager-control root remains
disjoint and never needs to share that device. The lifecycle manager does not
bootstrap or repair the control root
before lease acquisition. A missing, replaced, symlinked, unexpectedly
writable, or permission-mismatched root fails without creating a lease or
starting a workload.

Creation and every reuse audit all Docker bind sources and destinations, their
resolved host paths, propagation, and read-only flags. Every container-visible
alias of the sealed snapshot and checkpoint must be read-only; no alias of the
manager-control root may exist. An in-container mount probe must prove writes
under `/workspace/Psi0` and `/checkpoint` fail with a read-only filesystem
error, `/runtime` is writable, and control/input aliases such as
`/runtime/lease.d`, `/runtime/control`, and `/runtime/inputs` do not exist.
Unexpected mounts, overlapping host roots, symlinked aliases, or a successful
input/control write are `PROVENANCE_BLOCKED` before server launch.

The checkpoint mount exposes only the protected content-addressed snapshot,
not the mutable source model tree or the input-root parent. The lifecycle
manager opens the named weight using no-follow component traversal beneath the
snapshot and requires the resolved relative path, inode/type, size, and digest
to equal the profile before launch. The complete canonical tree, completion
marker, root ownership/mode/ACL, and failed lifecycle-account write probe are
verified before container start and again after container and helper cleanup.
Any path, inode/type, size, ownership, mode, symlink target, entry-set, or
content change is `PROVENANCE_BLOCKED`, even if the server and episodes
otherwise succeeded. Mutating or deleting the original model-weight tree after
freeze has no effect on these checks or on the server's mounted bytes.

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

Before Docker create or start, the write-ahead journal records the exact absent
create target or existing container ID, expected labels/image/mount digest,
prior state, lease tuple, and requested post-state. First creation embeds the
creating run token digest and generation in ownership labels. A start of the
persistent object is attributable only when its preexisting immutable identity
and the pending journal both match. Recovery re-reads Docker daemon state from
that target; it never depends on whether the original Docker client returned.

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

A process outside that set is foreign. The manager requests interruption of
only its runner-supervised evaluator and directly cleans only its owned tunnel
and server resources; it never signals the foreign GPU PID.
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

The allocation probe runs as the runner supervisor's fixed `cuda_probe`
operation in the same dedicated-UID/private-root sandbox and direct-loader
path used for evaluation, with `CUDA_VISIBLE_DEVICES=1`. Through the CUDA
runtime used by the sealed PC2 environment's PyTorch, it must observe exactly
one CUDA device, report its UUID equal to the selected physical UUID, allocate
a small tensor on `cuda:0`, and synchronize successfully. The probe has a
30-second internal deadline plus an attested child PID and bounded
supervisor-owned INT/KILL/reap; killing only a client process is insufficient.

After the probe exits, the manager requires its GPU process to disappear. It
rechecks GPU-1 process ownership before episode 2, after episode-2 cleanup,
before episode 3, and after final cleanup. During an episode, only PIDs in the
runner's attested evaluator process group/tree may appear on the selected UUID.

From immediately before evaluator launch through its final descendant-reaping
check, an independent manager task polls the structured PC2 GPU process table
at least every five seconds. Each poll has a 10-second subprocess deadline and
is appended to that episode's `pc2-gpu-monitor.jsonl`. An allowed PID must map
to the exact evaluator leader or a currently enumerated descendant by PID,
`/proc` start ticks, parentage, process-group ID, and normalized argv; process
name or PGID alone is insufficient. The monitor runs independently of the
evaluator and its TUI/event pipe. A missed deadline, malformed/unknown GPU
state, disappeared identity mapping, PID reuse, or process outside the attested
tree immediately makes the episode an infrastructure failure and initiates
cleanup of only the owned evaluator/remote resources.

A foreign process appearing at any gate is `FOREIGN_BLOCKED`: the manager may
request interruption of its own supervised evaluator and clean up its own
remote resources, but it never signals the foreign GPU process. GPU inventory,
UUID, probe result, and
before/during/after process tables are mandatory evidence rather than values
inferred from `CUDA_VISIBLE_DEVICES`.

### PC2 execution provenance

Evaluation does not execute from the editable development worktree or its
shared virtual-environment symlink. The provenance-freeze gate creates a
dedicated PC2 input closure under:

```text
/mnt/data/psi0-simple-eval-inputs/<pc2-closure-id>/source
/mnt/data/psi0-simple-eval-inputs/<pc2-closure-id>/venv
/mnt/data/psi0-simple-eval-inputs/<pc2-closure-id>/episode-data
/mnt/data/psi0-simple-eval-inputs/<pc2-closure-id>/task-data
```

PC2 closure construction has its own local ownership protocol; the H100 lease
does not substitute for it. Its manager-only paths are:

```text
/mnt/data/jihun/psi0-simple-eval-control/pc2-construction.lock
/mnt/data/jihun/psi0-simple-eval-control/pc2-constructions/<pc2-closure-id>.json
/mnt/data/jihun/psi0-simple-eval-workloads/input-staging/
  <pc2-closure-id>.<owner-token>/
```

The protected input root is operator-provisioned beneath root-owned
`/mnt/data`, not beneath the lifecycle account's writable
`/mnt/data/jihun` directory. The configured PC2 input, control, and workload
roots are same-host, pairwise-disjoint resolved roots. A separately protected
base-Python input root is also operator-provisioned at:

```text
/mnt/data/psi0-simple-eval-base-python/<base-python-tree-sha256>/
```

The lifecycle account cannot write or rename either protected parent. The
closure input root, base-Python input root, and workload staging root must be
on one device so both protected publications can use atomic rename. The exact
four-root identity—closure input, base-Python input, manager control, and
workload—is `pc2_roots_identity_sha256`. Before creating a staging entry, the
constructor obtains an exclusive `flock` and atomically
writes an exact owner record containing schema version, closure ID, descriptor
hash, run ID, random 256-bit owner-token hash, host name, boot ID, owner PID and
`/proc` start ticks, state, phase, staging and final resolved paths,
`cleanup_required`, pending action, heartbeat monotonic/wall times, and last
error. The lock is held through construction; the owner record is the
write-ahead crash record and is fsynced before every filesystem mutation. The
stable `pc2-construction.lock` inode is never replaced; only the owner-record
path is atomically replaced on updates.
Neither record is stored in the input closure or exposed to an evaluator.
The fixed resolved paths, mount IDs, device/inode, owner/group, and modes of
all four roots are committed as `pc2_roots_identity_sha256`; a replaced,
symlinked, cross-device, or unexpectedly permissive root fails before lock
acquisition.

Publication uses an operator-provisioned root service, not ambient `sudo`, a
setuid executable, or a lifecycle-owned daemon. Systemd owns the root-only
executable `/usr/local/libexec/psi0-eval-install-pc2-input`, the socket unit
`psi0-eval-pc2-installer.socket`, and the service unit
`psi0-eval-pc2-installer@.service`. Their exact hashes are recorded as
`pc2_input_installer_sha256`, `pc2_installer_config_sha256`,
`pc2_installer_socket_unit_sha256`, and
`pc2_installer_service_unit_sha256`. The socket is fixed at
`/run/psi0-simple-eval/pc2-installer.sock`, owned by root and writable only by
the dedicated `psi0-eval-installer` group. The lifecycle account may connect
to that socket but cannot start, stop, replace, reconfigure, or execute the
service with another argv.

The root-owned, mode-`0444` configuration contains only the four already
profiled root identities, lifecycle UID/group, dedicated input UID/group,
mode-policy version, request size/deadline limits, and journal/socket paths; its
exact bytes are `pc2_installer_config_sha256`. Unit drop-ins, environment
files, and runtime overrides are forbidden by preflight.

The reviewed service unit fixes `User=root`, `NoNewPrivileges=yes`,
`PrivateNetwork=yes`, `RestrictAddressFamilies=AF_UNIX`,
`ProtectSystem=strict`, an empty executable search path beyond the single
attested installer, a capability bounding set limited to the ownership/mode
operations it requires, and `ReadWritePaths` limited to the two protected
input roots plus the token-derived workload staging root. It has no shell,
Docker socket, SSH key, device access, network listener, or arbitrary command
field. The socket unit uses `Accept=yes`, one request per connection, bounded
message/idle timeouts, and no filesystem path supplied by the peer.

Each connection sends exactly one length-bounded canonical JSON request plus
the already locked stable construction-lock FD via `SCM_RIGHTS`. The service obtains the
peer UID/PID with `SO_PEERCRED`, requires the operator-configured lifecycle UID
and a live matching PID/start-ticks owner record, `fstat`s the received
descriptor, and proves that it is the configured stable construction-lock
inode. It independently opens that lock path and requires
`flock(LOCK_EX|LOCK_NB)` to fail with `EWOULDBLOCK` while the received
same-open-description FD remains live. Only then does it independently
`openat2(..., RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` the current owner-record
path, verify its regular-file inode/content/schema/token/phase and manager
PID/start ticks, and bind its digest into the root journal. The manager is
forbidden to advance its construction phase or change any token, identity,
path, pending action, or cleanup field while the service transaction is live;
it may atomically replace only the heartbeat timestamps while holding the same
stable lock. The service's continued authority derives from that stable
same-open-description lock FD and token-bound journal, not from retaining a
replaceable owner-record inode.
Closure operations accept only `pc2-construction.lock`; base-Python operations
accept only `base-python-construction.lock`. Their device/inode identities are
fixed in the installer configuration, and neither request may select a lock or
owner-record path.
The strict request
operation is `install_closure`, `recover_closure`, `install_base_python`, or
`recover_base_python`; it contains only schema version,
operation, expected source/destination kind and operation-specific IDs,
descriptor/intake digest,
owner-token hash, and recovery-token hash or null. No arbitrary path appears
on the wire. The service derives both roots and exact token-named staging path
from its root-owned configuration. Closure operations require the exact
precomputed closure ID; `install_base_python` requires an intake ID and a null
destination ID because the service's final mode-inclusive digest selects the
absent target. Its strict response contains operation,
token hashes, final tree/component hashes, completion hash, receipt hash, root
identity, and terminal installer phase. Both messages reject additional keys,
coercion, oversize frames, missing/extra FDs, and peer mismatch.

The service no-follow validates all roots and may rename only from the
configured staging root to the corresponding protected closure or base-Python
root. It cannot replace an existing target, launch a workload, or confer
general root/input-root access. The final snapshots and parents are owned by a
dedicated input UID, so the lifecycle/evaluator account cannot chmod, chown,
unlink, rename, replace, or create any entry in them. The locked FD survives
the privilege boundary as the same open file description; there is no
shell/`sudo` FD-preservation ambiguity.

The installer's durable receipt is exclusively created at
`/mnt/data/psi0-simple-eval-inputs/.installations/<pc2-closure-id>.json` under
a preprovisioned root-owned directory. It is outside the closure/component
hashes, separately hashed in the profile, and cannot be removed or replaced by
the lifecycle account. The mutable construction owner record is concurrency
evidence only and can never override this protected receipt or `COMPLETE.json`.
Base-Python receipts use the equally protected
`/mnt/data/psi0-simple-eval-base-python/.installations/<tree-sha256>.json`.
Its separately hashed completion metadata is
`/mnt/data/psi0-simple-eval-base-python/.completions/<tree-sha256>.json`; both
metadata files are outside the base payload/root digest and are required by
the inherited-base-root preflight.
The service's append-only, fsynced operation journals are likewise outside all
payload hashes. They are root-owned and lifecycle-readable only through their
exact token-derived paths after a terminal response; the lifecycle account
cannot create, append, truncate, rename, or delete them, and the evaluator
namespace cannot see them:

```text
/mnt/data/psi0-simple-eval-inputs/.installer-journal/
  <pc2-closure-id>.<operation-token-sha256>.jsonl
/mnt/data/psi0-simple-eval-base-python/.installer-journal/
  <tree-sha256>.<operation-token-sha256>.jsonl
```

Journal records have exact sequence numbers and phases and are hash-chained to
the request, owner/recovery tokens, source/destination identities, rename
postcondition, and canonical receipt bytes. Every receipt field, including any
recorded timestamp, is fixed in the pre-rename journal record; recovery never
generates a new value. The service fsyncs a pending record before each
privileged mutation and a result record afterward.

Construction occurs only in the token-named staging directory. An
`INCOMPLETE.json` marker is created and fsynced before its first child. The
phase enum is exact and monotonic:

```text
ALLOCATED -> SOURCE -> EPISODE_DATA -> VENV -> TASK_DATA_COPY ->
HSSD_NORMALIZED -> PAYLOAD_READY -> INSTALLING -> FINAL_RENAMED ->
RECEIPT_CREATED -> COMPLETE
```

The ordering is contractual: the dedicated venv is complete and probed before
`HSSD_NORMALIZED` invokes that venv's USD/Sdf APIs. No ambient worktree Python
or shared venv may normalize assets. Before each construction phase the owner
record sets `pending_action` and
`cleanup_required=true`; after the phase's manifest and directories are
fsynced it records the completed phase and clears only `pending_action`.
Heartbeat writes use a separate short owner-record transaction and cannot be
starved by hashing or venv installation.

`PAYLOAD_READY` hashes are intake/content diagnostics only. During
`INSTALLING`, the privileged installer first applies the canonical final
ownership and mode policy to every component payload: directories `0555`,
ordinary files `0444`, and only descriptor-allowlisted executable files
`0555`; it then fsyncs them. Only after this metadata normalization does it
compute the authoritative mode-inclusive manifests for `source`, `venv`,
`episode-data`, and normalized `task-data`. It validates, normalizes to `0444`,
and hashes the already complete `hssd-normalization-results.json`, then
computes `pc2_runtime_identity_sha256`, whose canonical inputs include the
protected base-Python tree/root/receipt, loader hash, normalization-results
digest, runner sandbox digest, and evaluator UID/GID. It writes the exact
`runtime-identity.json` as owned mode `0444` and hashes that sidecar before
writing `COMPLETE.json` as the completion metadata. The normalization-results
file, identity sidecar, and completion marker are explicitly excluded from
every component payload-tree hash; the first two have their own profile hashes,
and the marker records the owner-token hash, descriptor, final component
hashes, normalization-results hash, sidecar hash, closure-root
device/inode/mount identity, canonical mode-policy version, and schema. Its own SHA-256 is
`pc2_closure_completion_sha256` in the profile.

The installer fsyncs all payloads and markers, removes `INCOMPLETE.json`, makes
`COMPLETE.json` immutable to the lifecycle account, and makes the staging root
`0555` last. It then atomically renames the now-final tree to the previously
absent protected destination and fsyncs both parents. A durable installer
service journal entry binds that rename to the operation token as
`FINAL_RENAMED`. The service then exclusively creates/fsyncs the receipt and
appends `RECEIPT_CREATED` before sending its one terminal response. The root
journal is authoritative for these two privileged subphases; there is no
intermediate manager acknowledgement or impossible mid-request owner-record
update.

After the terminal response, the manager independently opens the protected
journal and receipt, validates their sequence/hash chain and response digest,
then mirrors `FINAL_RENAMED` and `RECEIPT_CREATED` into two separately fsynced
owner-record transitions. A crash before either mirror leaves the manager phase
at `INSTALLING`; recovery derives the privileged progress only from the
token-bound root journal. The candidate profile may use only the protected
receipt, never earlier intake hashes. Only after the lifecycle manager reopens
and revalidates the published root does its owner record enter `COMPLETE` with
`cleanup_required=false`.

Recovery is exact and never guesses ownership. A constructor seeing an active
matching PID/start-ticks/boot-ID owner or held lock returns
`LOCAL_CONSTRUCTION_BLOCKED`. Once the lock is acquirable and that exact owner
is proven dead, one recoverer atomically claims the owner record with a fresh
token. A token-matching staging directory whose phase precedes `INSTALLING` is
re-manifested for evidence and then removed before a fresh build; no glob or
user-supplied path is accepted. At `INSTALLING`, `FINAL_RENAMED`, or
`RECEIPT_CREATED`, recovery calls only the installer service's
token/record-bound `recover_closure` operation, sending the claimed stable
construction-lock FD, prior operation-token hash, and fresh recovery-token
hash. If the final path is absent
and the exact staging path contains a valid owner-token-bound `COMPLETE.json`
whose final metadata and full manifests agree, the installer finishes the
atomic rename rather than rebuilding; otherwise it removes only that exact
owned staging tree and restarts from `ALLOCATED`. If a valid `COMPLETE.json`
final closure exists but the protected receipt is absent after a crash between
rename and receipt creation, the service alone may reconstruct the receipt. It
requires the exact prior operation token in both `COMPLETE.json` and its
root-owned journal, the fresh recovery token/locked claim, an absent receipt
opened with `O_CREAT|O_EXCL`, and complete revalidation of the descriptor,
authoritative manifests, permissions, root identity, and sidecar. It creates
exactly the receipt bytes implied by that closure, fsyncs the receipt and
directory, and records `RECEIPT_CREATED`. An existing receipt is adopted only
when byte-identical; a mismatch blocks. Two concurrent recoverers cannot both
hold the constructor lock, and exclusive receipt creation provides the second
fence. If a valid `COMPLETE.json` final closure and valid receipt exist after a
crash between receipt creation and owner-record completion, recovery adopts
them only after all checks, then marks the record complete. A final path
without valid completion metadata, invalid installing-stage metadata, a
staging/final token mismatch, live unknown child, or any contradictory state is
`PROVENANCE_BLOCKED` and is never automatically deleted. Construction tests
inject a crash after allocation, every named manager and installer subphase,
metadata normalization, every authoritative hash, sidecar/marker fsync, root
mode change, rename, `FINAL_RENAMED`, exclusive receipt creation/fsync,
`RECEIPT_CREATED`, and published revalidation; they require either no final path or one
fully valid final closure, deterministic stale recovery, and exactly one
winner from concurrent constructors and recoverers.

`pc2-closure-id` is computed before the destination exists. It is the SHA-256
of a canonical, path-independent descriptor containing schema version, source
commit/tree/gitlinks, ordered Git-object manifest, episode-data manifest,
immutable pre-normalization task-asset requirements, task-data source manifest,
the path-independent base-Python payload-tree digest, canonical
mode/HSSD-normalization policy versions, the attested path-independent PC2
installer executable hash, and hashes of the locked wheel/install inputs used to construct
the venv. Absolute installation paths and the resulting venv tree hash are not
descriptor inputs. The descriptor selects the absent final directory; all
components are built in staging, then the complete installed source/venv/data
trees are hashed and recorded separately in the candidate profile. Runtime
requires `pc2_input_host_path` to equal the configured input root joined with exactly
`pc2_closure_id`. This avoids a path/hash fixed point while keeping the closure
name stable and reviewable.

Location- and installation-bound fields are deliberately excluded from the
closure descriptor: base/closure host paths, mount IDs, devices/inodes,
root-identity digests, completion/receipt bytes, installer configuration and
service/socket units, lifecycle/evaluator UIDs, and timestamps. They remain
mandatory reviewed runtime provenance and may block adoption, but moving an
otherwise byte-identical protected base tree or regenerating location-bound
receipt metadata cannot change `pc2_closure_id`.

The descriptor never contains a document that normalization later edits. Its
`pc2_asset_requirements_sha256` names a frozen input document created before
the closure ID: logical resource names, original source identities, recursive
dependencies/content hashes, and the required normalizer policy/version. It
contains no generated output hash or old-to-new mapping and becomes read-only
before staging is named. `HSSD_NORMALIZED` instead creates a distinct
`hssd-normalization-results.json` containing the requirements digest, closure
ID, exact USD/Sdf tool identity, source/normalized layer hashes, and ordered
old-to-new mappings. Its digest is
`pc2_asset_normalization_results_sha256`; it is not an input to
`pc2_closure_id`. The results file sits beside the runtime-identity sidecar,
is excluded from component tree hashes, and is separately normalized,
read-only, hashed, and bound into `runtime-identity.json`, `COMPLETE.json`, the
installer receipt, profile, and pre/post evidence. Thus a requirement or
policy change selects a new closure ID, while deterministic result bytes are
verified as outputs without a digest cycle. The canonical results schema has
no timestamp, host/staging path, inode, or unordered mapping; repeated
normalization of identical requirements/tool identities must produce
byte-identical results.

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

Before computing a closure descriptor, the constructor prepares or adopts one
separate protected base-Python snapshot. The privileged installer service
publishes the fully resolved CPython executable, standard library, and native
runtime dependency closure beneath
`/mnt/data/psi0-simple-eval-base-python/<pc2_base_python_tree_sha256>` using
the same final-metadata-first manifest algorithm and a distinct completion
marker/receipt. The final mode-inclusive tree digest selects its directory;
the profile records that path/tree as `pc2_base_python_host_path` and
`pc2_base_python_tree_sha256`, the external completion metadata as
`pc2_base_python_completion_sha256`, its resolved protected root as
`pc2_base_python_root_identity_sha256`, and the receipt as
`pc2_base_python_installer_receipt_sha256`. Only the path-independent
`pc2_base_python_tree_sha256` is a closure-descriptor input; the other four are
location-bound runtime provenance. The
shared development interpreter path recorded by the current `.venv/pyvenv.cfg`
is an intake source only and is never an allowed runtime dependency.

Base-Python publication has its own write-ahead record and exact monotonic
phases:

```text
/mnt/data/jihun/psi0-simple-eval-control/base-python-construction.lock
/mnt/data/jihun/psi0-simple-eval-control/base-python-constructions/
  <base-python-intake-sha256>.json
/mnt/data/jihun/psi0-simple-eval-workloads/input-staging/
  base-python.<base-python-intake-sha256>.<owner-token>/
```

The intake identity is path-independent and known before staging allocation.
The owner/heartbeat/claim schema and locking rules are identical to closure
construction, with intake ID replacing closure ID. Its exact monotonic phases
are:

```text
BASE_ALLOCATED -> BASE_COPIED -> BASE_METADATA_NORMALIZED ->
BASE_FINAL_RENAMED -> BASE_RECEIPT_CREATED -> BASE_COMPLETE
```

The protected base digest, completion metadata, root identity, and receipt are
validated before the closure descriptor is computed, but only the payload-tree
digest contributes to that descriptor. Crash recovery uses the same
locked-FD service protocol and token-bound exclusive receipt reconstruction as
closure publication. An incomplete or contradictory base snapshot blocks
closure construction; it is never adopted from an ambient interpreter path.

The dedicated venv is prepared in closure staging with no embedded staging or
final closure path. Repo-local `.pth` entries use normalized relative paths from
`site-packages` to sibling sealed source/submodule roots; evaluation invokes
`venv/bin/python` only as the sealed loader's program argument and does not
depend on its external `PT_INTERP` or generated absolute-shebang console
scripts. `pyvenv.cfg` and interpreter links may reference only the
exact protected base-Python snapshot. A byte scan and structured inspection of
`.pth`, `.egg-link`, scripts, configuration, and symlinks rejects either the
staging prefix, the future final prefix, or any path outside the closure and
attested base-Python roots. A clean import-origin probe after atomic
publication requires `simple`, `gear_sonic`,
`decoupled_wbc`, `unitree_sdk2py`, `xrobotoolkit`, and other repo-local packages
to resolve beneath the sealed source, and all third-party packages beneath the
sealed venv or an explicitly hashed base-Python/standard-library root. The
complete venv, base interpreter, package freeze, native-extension/shared-library
closure—including the copied dynamic loader and every user-space GPU/graphics
library—Isaac/MuJoCo versions, and NVIDIA driver/CUDA identities are recorded
both individually and in the canonical `pc2_runtime_identity_sha256` digest.
Import-origin records store normalized paths relative to the closure root or
attested base-Python root, never a staging/final absolute prefix. The staging
probe runs before `PAYLOAD_READY`; the installer validates and includes its
path-independent digest in `pc2_runtime_identity_sha256`, and the post-publish
probe must reproduce the same document byte-for-byte.

Because the Git-object source closure intentionally contains no `.git`, the
installer creates `<closure>/runtime-identity.json` as a read-only identity
sidecar after authoritative payload hashing.
It has the following exact schema and no additional or coerced fields:

```text
schema_version: integer exactly 1
simple_commit: 40-character lowercase Git SHA
simple_root_tree: 40-character lowercase Git tree SHA
recursive_gitlinks_sha256: 64-character lowercase hexadecimal string
pc2_closure_id: 64-character lowercase hexadecimal string
pc2_source_tree_sha256: 64-character lowercase hexadecimal string
pc2_base_python_tree_sha256: 64-character lowercase hexadecimal string
pc2_base_python_loader_sha256: 64-character lowercase hexadecimal string
pc2_base_python_root_identity_sha256: 64-character lowercase hexadecimal string
pc2_asset_requirements_sha256: 64-character lowercase hexadecimal string
pc2_asset_normalization_results_sha256: 64-character lowercase hexadecimal string
pc2_runtime_identity_sha256: 64-character lowercase hexadecimal string
pc2_closure_root_identity_sha256: 64-character lowercase hexadecimal string
pc2_runner_sandbox_contract_sha256: 64-character lowercase hexadecimal string
```

The sidecar sits outside the component trees whose digests it carries, so it
does not create a hash fixed point. Its own SHA-256 is stored in
`COMPLETE.json` and `pc2_runtime_identity_sidecar_sha256` in the reviewed
profile. `pc2_closure_root_identity_sha256` covers the final resolved protected
path, mount ID, device/inode, dedicated owner/group, and final `0555` root mode;
the installer verifies that expected identity after applying the root mode and
atomic rename. Runtime never derives commit identity from `.git`, the current
worktree, an environment variable, or a CLI string.

Before evaluator launch, the manager opens both the protected closure root and
the exact profiled base-Python snapshot root with
`O_PATH|O_DIRECTORY|O_NOFOLLOW`, obtains each mount ID/device/inode, and opens
the sidecar relative to the closure root with
`openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)`. It verifies all three
descriptors, both completion markers/receipts, and exact profile identities,
then transfers them to the runner supervisor as the closure-root,
base-Python-root, and identity-sidecar FDs. The runner exposes the first two at
the fixed private-root mountpoints and passes all three to the evaluator as
`--runtime-root-fd`, `--runtime-base-python-root-fd`, and
`--runtime-identity-fd`. The closure's `venv/bin/python` path must itself be
beneath the closure FD. Its executable bytes, `sys.base_prefix`, stdlib, and
allowed shared-library origins must remain beneath the protected closure or
base-Python FDs after private-root translation. Working directory, repo-local
imports, venv site-packages, episode data, and task data must remain beneath
the closure FD. No interpreter, mapping, or import origin may resolve beneath
a third root.

The runner and evaluator require each protected pathname's current
mount/device/inode to equal the corresponding `fstat()` identity before
namespace construction, before exec, before worker spawn, before
`runtime_contract`, and after environment close. Every checked path is opened
relative to the appropriate retained root descriptor. The venv's intake
base-Python reference may name only the profiled base payload, but runtime
launch uses the sealed loader and private-root paths below; an arbitrary
external `.venv/pyvenv.cfg` target is rejected.

The lifecycle account cannot rename either protected parent or snapshot,
restore write permission, or replace an entry. A missing/non-regular descriptor,
mount/device/inode mismatch, path-to-FD mismatch, seek/replacement, premature
EOF, extra sidecar byte, or identity mismatch fails before worker spawn or
`gym.make`. A required adversarial test opens all three descriptors, then from
a second same-account process attempts chmod, child replacement, closure or
base-Python rename, either parent rename, symlink substitution, and lookalike
closure/base-Python swaps before exec; every mutation must fail and the
evaluator must report the original two root identities. A fixture that
deliberately permits any one swap must be rejected, not silently execute
closure/base snapshot B with identity A.

#### Dedicated evaluator identity and sealed loader root

The lifecycle manager never executes the evaluator as its own UID. An
operator-provisioned root supervisor consists of the exact static executable
`/usr/local/libexec/psi0-eval-run-pc2-evaluator` (root-owned mode `0555`),
root-owned mode-`0444` configuration
`/etc/psi0-simple-eval/pc2-runner-v1.json`, and systemd units
`psi0-eval-pc2-runner.socket` and `psi0-eval-pc2-runner@.service`, recorded as
`pc2_runner_launcher_sha256`, `pc2_runner_config_sha256`,
`pc2_runner_socket_unit_sha256`, and `pc2_runner_service_unit_sha256`. Its
fixed socket is `/run/psi0-simple-eval/evaluator-launcher.sock`. The lifecycle
manager may open one bounded control connection; the dedicated evaluator UID/
GID recorded as `pc2_evaluator_uid`/`pc2_evaluator_gid` is not a member of the
installer, lifecycle, control, or staging groups and cannot connect to either
privileged socket.

The supervisor executable is a statically linked, no-`PT_INTERP`, no-
`DT_NEEDED` binary. Its service namespace does not mount the installer socket
or construction roots and its path policy permits only the fixed runner
socket, reviewed profile, and read-only GPU/device metadata. Protected input
and workload roots enter only as validated directory FDs and may be used only by
the fixed child-mount operations below; the service cannot traverse their host
parents or sibling paths. This keeps installer authority out of the runner
service itself as well as out of its evaluator child.

The runner service's canonical capability bounding/ambient sets are exact and
contain only the reviewed namespace/mount, UID/GID drop, run-root ownership,
and model-free probe tracing capabilities; all are cleared before evaluator
exec. Its systemd filesystem allowlist and seccomp policy are part of
`pc2_runner_sandbox_contract_sha256`. An added capability, writable host path,
unit override, or executable dependency fails preflight.

The manager sends one strict launch request plus exactly six FDs with
`SCM_RIGHTS`: event-write, acknowledgement-read, closure-root,
base-Python-root, identity-sidecar, and the configured workload-parent-root FD.
The supervisor requires that sixth FD to match the profiled workload-parent
inode, no-follow opens the exact `<run-id>` directory and its manager-created
exclusive nonce marker, then exclusively creates
`<run-id>/evaluator-output`, changes only that new child to the evaluator UID
plus lifecycle evidence group, and mounts it at `/run-output`. The request
contains only schema version, reviewed profile
hash, operation (`loader_probe`, `cuda_probe`, or `evaluate_episode`), run ID,
episode index/nonce/loopback port with exact operation-specific nullability,
and the six expected FD-role labels. Only `evaluate_episode` accepts episode
index `2` or `3`, a fresh nonce, and a loopback port; both probes require those
three fields to be null. The supervisor validates `SO_PEERCRED`, the
manager PID/start ticks, all FD identities, the reviewed profile, and every
enumerated value. It constructs the fixed argv/environment from those values;
it never accepts an arbitrary command, path, argument, or environment
extension. It then supervises one child in a fresh process group, reports its
PID/start ticks/process-group ID over the same connection, and accepts only
bounded `INT`, `KILL`, status, and close messages for that child. The evaluator
inherits only the five existing runtime/event FDs; it receives the run output
as a mounted path, not an authority FD. The launcher control socket and every
unrelated FD are close-on-exec. Losing the manager connection triggers the
same bounded owned-child cleanup and terminal evidence.

Before dropping privilege, the supervisor constructs a fresh mount namespace
and pivots into a new empty tmpfs root. FD-backed, no-follow mounts expose only:

| Sandbox path | Source | Access |
| --- | --- | --- |
| `/sealed/closure` | inherited closure-root FD | read-only, nodev, nosuid |
| `/sealed/base-python` | inherited base-root FD | read-only, nodev, nosuid |
| `/run-output` | supervisor-created evaluator-output child FD | read-write, noexec, nodev, nosuid |
| `/tmp` | new bounded tmpfs | read-write, noexec, nodev, nosuid |
| `/proc` | new proc mount | read-only except required process-self interfaces |
| `/dev` and `/sys` | exact profiled GPU/device and read-only hardware allowlists | no executable regular files |
| allowlisted `/etc` and `/usr/share` aliases | fixed paths below inherited base-root FD | read-only, nodev, nosuid |

The host network namespace is retained only for the existing PC2 loopback
tunnel. Host `/`, `/home`, `/mnt/data`, `/usr`, `/lib*`, `/etc/ld.so.cache`,
manager control/staging, and `/run/psi0-simple-eval` are absent. The small
private-root `/etc` and `/usr/share` alias manifest is fixed in the sandbox
contract; every alias is a no-follow read-only bind from a named path beneath
the protected base FD and can never resolve to the host tree. Required
configuration/ICD files and every user-space GPU/graphics library are copied
into that protected base-Python tree. After mount setup the child drops
all supplementary groups and capabilities, sets the exact evaluator UID/GID,
sets `PR_SET_NO_NEW_PRIVS`, applies the reviewed runner seccomp/device policy,
and can write only `/run-output` and `/tmp`. The canonical namespace, UID/GID,
mount, device, capability, seccomp, and environment document is
`pc2_runner_sandbox_contract_sha256`.

The base snapshot is a complete executable closure, not merely copied Python
bytes. It contains the exact glibc loader at
`pc2_base_python_loader_relative_path`, whose bytes are
`pc2_base_python_loader_sha256`, plus every recursively resolved `DT_NEEDED`,
RPATH/RUNPATH, Python-extension, Isaac/MuJoCo, CUDA/driver, Vulkan/GL, and
explicit `dlopen` dependency. Intake rejects absolute or escaping dependency
targets unless the sandbox maps that exact alias back into the protected base
tree. The manager and supervisor each open the loader relative to their
retained base FD with `openat2`, require both opens to identify the same
profiled inode, verify it is an executable regular ELF with no `PT_INTERP`, and
the supervisor invokes that already-open FD with
`execveat(..., AT_EMPTY_PATH)`.
The exact loader argv prefix is:

```text
<sealed-loader-fd>
--inhibit-cache
--library-path
/sealed/closure/venv/lib:/sealed/base-python/lib:/sealed/base-python/lib64:/sealed/base-python/usr/lib:/sealed/base-python/usr/lib64
--argv0
/sealed/closure/venv/bin/python
/sealed/closure/venv/bin/python
```

The kernel therefore never follows Python's intake `/lib64/ld-linux...`
`PT_INTERP`; the sealed loader resolves the venv program and all libraries
inside the private root. This uses the dynamic loader's documented direct
invocation, `--inhibit-cache`, and `--library-path` behavior
([`ld.so(8)`](https://man7.org/linux/man-pages/man8/ld.so.8.html)). Construction
records the loader's glibc version and rejects a version without the profiled
`--argv0` behavior. The runner's fixed model-free `loader_probe` operation runs
the exact loader with `--verify` and `--list`, then starts `python -I -S` with
the loader's `LD_DEBUG=files,libs` diagnostic captured on a dedicated bounded
pipe. The static supervisor traces that probe's `openat`/`openat2` and mapping
syscalls, resolves dirfds through its retained root FDs, and reconciles the
trace, loader diagnostic, and `/proc/<pid>/maps`. Every regular-file-backed
executable mapping and regular-file loader access must resolve beneath
`/sealed/closure` or `/sealed/base-python`; the exact kernel pseudo-mappings
and profiled anonymous/JIT mapping classes are recorded separately and may not
name a host path. The run/tmp/proc/dev/sys mounts may contain no executable
regular mapping. The same audit remains active through environment close.
Missing dependencies, host-library/cache access, an
external executable mapping, or a namespace/UID drift fails before
`gym.make` and stops only the supervised evaluator.

The ignored development path
`data/evals/simple-eval/G1WholebodyXMovePickTeleop-v0/dr-level-0` is not assumed
to be part of the Git snapshot. `freeze-provenance` copies it into the sealed
`episode-data` directory using no-follow traversal, rejects special files and
escaping links, and records the same path/mode/size/content canonical manifest
used for other input trees. The profile records that manifest root as
`pc2_episode_data_tree_sha256`. Evaluation reads only this sealed copy; both
the mutable development dataset and any global `data/evals` output remain off
its import and input paths.

The simulator also consumes ignored repository-relative `data/` resources.
The managed path adds one explicit process configuration seam:
`SIMPLE_DATA_ROOT=/sealed/closure/task-data`. `simple.utils.get_data_dir()`
uses that absolute root when supplied and rejects a missing, relative,
symlinked, writable, or out-of-closure value. The manager also sets
`SIMPLE_ASSET_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`. In offline mode, `resolve_data_path()` performs only
no-follow containment and existence checks: `auto_download=True` cannot call
`snapshot_download`, create directories, extract archives, remove archives, or
write any path. A missing resource fails before simulator construction.
`simple.core.simulator.Simulator.get_data_dir()` and `resolve_data_path()` are
changed to delegate to this same validated utility; no backend retains a
repository-relative bypass. Tests enumerate and exercise both public resolver
paths.

`freeze-provenance` computes a task-specific asset requirement manifest for
episodes 2 and 3 using the exact sealed episode configurations, task ID,
`g1_sonic` robot, `hssd:scene0`, `mujoco_isaac` backends, and material choices.
The manifest enumerates every logical resource and every transitive file it
references: robot MJCF/USD/configuration/CuRobo files and their meshes/textures,
all target and distractor GraspNet stable/grasp/collision/model/USD files, the
selected HSSD scene and its USD dependencies, and every selected MDL/material
dependency. MJCF includes/assets, USD dependencies, YAML references, and MDL
textures are expanded recursively with cycle and escape rejection. The exact
files are copied with no-follow traversal into `task-data`, whose complete
canonical tree hash is `pc2_task_assets_tree_sha256`; the ordered logical-path
and content-hash pre-normalization requirement document is
`pc2_asset_requirements_sha256`. That document is immutable after closure-ID
selection. Nothing else from mutable development `data/` is copied or used.

HSSD needs one additional freeze-time normalization step. The existing
`HssdSceneManager._hack_fix_tmp_paths()` shell pipeline and global `/tmp` copy
are removed from the managed production path, not merely skipped by the
simulator-free probe. During `HSSD_NORMALIZED`, after `VENV` and
`TASK_DATA_COPY` have completed, a dedicated normalizer invokes only the
staging closure's attested venv to open the copied HSSD layer with its USD/Sdf
APIs, enumerates sublayers,
references, payloads, and every scalar/array asset path, and maps only the
expected absolute or relative `tmp/<content-id>/{props,textures}/...`
dependencies to normalized relative paths beneath that scene's sealed
directory. It rejects shell metacharacters, unknown external schemes,
unmanifested dependencies, path traversal, cycles, missing targets, and any
resolved path outside `task-data`. It writes a new closure-local layer by
exclusive temporary file plus fsync/atomic rename; the source and normalized
layer hashes, USD tool version, requirements digest, closure ID, and ordered
old-to-new mapping are recorded only in the separate
`hssd-normalization-results.json`. The pre-normalization requirements are never
edited. The original mutable source is never edited.

At runtime `HssdSceneManager.load("hssd:scene0")` resolves the already
normalized sealed layer and performs no subprocess, shell, string scan,
directory creation, or asset copy. `HssdSuite.data_dir` and both backends
receive that same closure-local scene directory. A required production-path
test calls the real registered manager's `load("hssd:scene0")`, opens the
returned normalized layer through USD, and resolves every dependency beneath
the sealed root while network/download calls, `subprocess.run`/`Popen`, shell
entry points, `shutil.copy*`, and writes outside a test run root are set to
fail. It proves a sentinel snapshot of global `/tmp` is unchanged. This test is
in addition to, not replaced by, the general simulator-free asset probe.

Before either episode, a simulator-free production asset probe loads that
episode's sealed reset configuration, resolves its complete requirement subset
through the actual `resolve_data_path()` offline seam, parses every transitive
reference, opens every required regular file read-only, and compares type,
size, and SHA-256 with the manifest. It runs once for episode 2 and once for
episode 3 and must cover the union frozen above. The probe asserts that no
network/download function, write syscall, Isaac application, MuJoCo simulator,
or environment constructor is invoked. Missing, extra, changed, unprobed, or
out-of-root resources fail before `gym.make` and before that episode's GPU
workload.

The evaluator is invoked as the sealed venv's Python module entry point with
the sealed source as working directory; it does not use the development
worktree's console script. Its `HOME`, `TMPDIR`, `TMP`, `TEMP`,
`XDG_CACHE_HOME`, `MPLCONFIGDIR`, `TORCH_HOME`, `TRITON_CACHE_DIR`,
`OV_CACHE_DIR`, `CUDA_CACHE_PATH`, `NUMBA_CACHE_DIR`, and Omni/Isaac log and
user-config roots all point beneath the current run's `generated-cache` or
`logs` directories.
`PYTHONDONTWRITEBYTECODE=1` and these run-scoped variables prevent writes to
the input closure or global `/tmp`; production-path tests audit created paths
and reject any evaluator write outside its run root. The sealed source, venv,
episode-data, and task-data are non-writable, and their complete manifests,
asset requirements/normalization/probe results, import origins, package freeze, Git
commit/gitlinks, interpreter/shared-library hashes, and driver identities must
match both before the remote lease is acquired and after all evaluation
cleanup.

The protected closure and base-Python pathnames are never substituted into an
environment by string alone: the manager derives every displayed sealed path
relative to the appropriate verified root FD, compares it with the protected
canonical pathname, and retains both descriptors until the evaluator and all
descendants exit. The distinct evaluator UID has no path, group, socket,
control-record, construction-lock, staging-root, or installer-journal access;
none is mounted in its namespace. Only the lifecycle manager retains the
separate installer and evaluator-supervisor connections, neither FD is passed
through exec, and no evaluator descendant can publish or replace an input.

The committed runtime profile contains the expected PC2 closure identities.
Any dirty root/submodule, gitlink mismatch, escaping import, attempted asset
download/write, changed local environment/input, or pre/post manifest
difference is
`PROVENANCE_BLOCKED`. Run outputs and caches live only under the disjoint
`/mnt/data/jihun/psi0-simple-eval-workloads/<run-id>` root and cannot make the
sealed Git snapshot dirty. The lifecycle manager exclusively creates that run
root and nonce marker beneath the profiled workload parent. The runner
supervisor validates the parent FD, run directory, and marker, exclusively
creates only `<run-id>/evaluator-output`, assigns that child to the evaluator
UID plus lifecycle evidence group, and mounts the child at `/run-output`. The
evaluator cannot traverse the workload parent, manager evidence, or any sibling
staging/run directory, while the manager retains read-only access to the
evaluator-output evidence.

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

Every acquire, heartbeat, recovery-claim, write-ahead cleanup mark, and release
is a short lease transaction executed by one bounded remote helper. The helper
opens `lease.lock`, takes `fcntl.flock(LOCK_EX)`, rereads `lease.json`, performs its
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
pending_mutation: null or exact object containing mutation_id, kind,
  target_identity_sha256, and started_monotonic_ns
```

Every new `freeze-provenance`, `create`, or `evaluate` lease starts with
`cleanup_required=true`; no crash window exists in which the operation owns a
resource but recovery is disallowed. Before each resource-creating mutation,
the holder atomically persists the exact `pending_mutation` under `lease.lock`
and fsyncs the operation-specific journal. After the mutation it records the
observed resource identity and clears only `pending_mutation`.
`cleanup_required` remains true for the whole operation and is cleared only
after its operation-specific terminal postconditions are durably proven. A
clean release then removes the exact lease by compare-and-swap. If the manager
crashes before release with the flag already false, normal stale reclamation
may remove only that no-resource lease.

The holder refreshes the token-checked heartbeat every 10 seconds using an
independent heartbeat task. The lease expires after 45 seconds without a valid
heartbeat. Loss of the heartbeat is
an infrastructure failure, but expiry alone never authorizes another manager
to stop a process or container. A partially written temporary transaction is
ignored only after its recorded owner helper is proven dead; it is never
treated as a valid lease.

The holder revalidates token, generation, mode, and nonexpired heartbeat
immediately before every Docker start/stop, server launch/signal, tunnel
launch, evaluator launch, and remote-helper signal. A manager whose lease was
replaced cannot perform another mutable action, even if an earlier local
preflight passed.

Lease acquisition, heartbeat, write-ahead marking, recovery-claim, terminal
clearing, and release take only `lease.lock`; they never wait for or take
`mutation.lock`. Helpers that change Docker, workload processes, or snapshots
take `mutation.lock`, then briefly take `lease.lock` to revalidate their tuple,
and hold only `mutation.lock` through that single bounded mutation and daemon
postcondition. No code may hold `lease.lock` while acquiring `mutation.lock`;
the sole nested lock order is `mutation.lock` then `lease.lock`. The independent
heartbeat therefore continues while a provenance helper holds
`mutation.lock` for its full 1,800-second allowance. Each mutation helper
revalidates again before starting any later mutation. A
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
- `cleanup_required=false` and `pending_mutation=null`;
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
and generation, `cleanup_required=true`, the exact operation journal, and all
ownership fields required for that operation. It does not require unrelated
records. The transaction atomically replaces the old owner with one new
`mode="recovery"` token, incremented generation, and
`recovery_of_token_sha256`. This fences the old manager before any signal.
Exactly one concurrent recoverer can win; every loser observes a token or
generation mismatch and performs no mutation.

Recovery then applies one exact predicate:

- **freeze-provenance:** reconcile the recorded helper plus distinct
  token-named server-snapshot, checkpoint-snapshot, and candidate-profile
  staging paths. It may remove only incomplete staging paths named in the
  journal, preserves a protected snapshot only when its operator-installer
  completion metadata and complete canonical manifest agree, and requires no
  container, PID, cgroup, port, or server record. A crash after either atomic
  promotion but before external completion-metadata creation may recreate that
  metadata only after the installer revalidates the exact journal/token,
  protected final root, final mode-inclusive manifest, and absent metadata
  path. Recovery then adopts that one exact complete snapshot and resumes the
  remaining freeze phases; it never recopies from or substitutes a changed
  mutable checkpoint source.
- **create:** reconcile the fixed container through its expected image/profile
  labels and write-ahead run token. If creation or the short attestation start
  happened, stop only that attributable container, require exact `exited`
  state, reconcile the manager-only identity record, and require no server or
  helper.
- **evaluate:** require the exact container labels plus every available
  local evaluator/tunnel and remote server/helper PID, start-time, argv,
  namespace, process-group, cgroup, port, and GPU mapping, including the
  checkpoint tracer and its server child. Evaluate recovery
  must run on the recorded PC2 manager host so it can reconcile local process
  identities and event-pipe EOF; another host is `FOREIGN_BLOCKED`. It stops
  only those attributed workloads and reaches exited/no-helper/no-port
  postconditions, then completes the post-cleanup source and checkpoint
  manifests before clearing the cleanup flag.

A record missing a fact required by its own operation becomes
`FOREIGN_BLOCKED`; missing records that the operation could never create are
not demanded. A running container is recoverable only for `create` or
`evaluate` and only when every live workload in it is attributable to the
expired lease.

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
comparison only after its operation-specific terminal predicate passes,
`pending_mutation=null`, and no owned remote helper remains. Normal completion
atomically clears `cleanup_required` only after that proof; incomplete cleanup
leaves its already-true value unchanged, stops the heartbeat, and requires
explicit stale recovery. Read-only `status` never acquires, refreshes, claims,
or removes a lease.

### Remote helper ownership and deadlines

No remote operation relies on the lifetime of its local SSH client. The
manager passes the reviewed remote-helper source and one-use helper ID to a
fixed system-Python bootstrap and records its SHA-256; it does not install or
replace a shared executable before acquiring a lease. The bootstrap exclusively
creates a helper-ID directory beneath `transactions`, writes and fsyncs the
source plus initial record, then forks. The child calls `setsid()`, places each
work child in a tracked process group, closes every inherited descriptor except
its manager-control files, redirects stdin to `/dev/null` and stdout/stderr to
transaction-owned logs, installs its internal monotonic deadline, and only then
writes a durable `detached_ready` acknowledgement containing helper PID/start
ticks/PGID/source digest/record identity. The SSH-facing bootstrap parent emits
that exact acknowledgement and exits. No helper or work child retains an SSH
pipe, PTY, socket, stdin, stdout, or stderr descriptor.

The manager accepts launch only after reading the acknowledgement and
reconciling it through a fresh read-only SSH query. If SSH disappears before
acknowledgement, launch state is unknown and the manager reconciles the helper
ID record and exact PID/start/argv before any retry; it never launches a second
helper with that ID. If SSH disappears afterward, the detached helper continues
to its internal deadline and terminal record without the original session.
Every Docker, hashing, snapshot, lease, and process-control invocation uses
this content-attested protocol.

Before doing work, the helper exclusively creates a transaction record under
`/mnt/data01/jhkim/psi0-simple-eval-control/transactions/<helper-id>.json`
containing the helper-source digest, exact lease token
digest/generation/mode, helper ID, run ID, operation, remote boot ID, host PID,
process-group ID, process start ticks, normalized argv and digest, internal
deadline, start time, detached acknowledgement identity, log paths, and state.
The state sequence is exactly `launching`, `detached_ready`, then one terminal
state. It fsyncs `detached_ready` before spawning a work child. Every child
waits on a private start barrier until its PID, start time, argv digest, parent
PID, and process group have been fsynced; only then can it run.

For acquisition and recovery-claim transactions, those identity fields are
the proposed token, generation, and mode that the same locked compare-and-swap
will install; for all other operations they must equal the current lease.
Transaction creation, compare-and-swap, and terminal update use the
operator-provisioned control root and never a container-visible path.

On its internal deadline, the remote helper applies bounded INT/TERM/KILL to
its own recorded child group, performs a fresh post-KILL wait and liveness
check, and atomically records `completed`, `failed`, `timed_out`, or
`cleanup_failed`. A failed/timed-out helper leaves the lease's independent
`cleanup_required` flag true. Docker operations additionally record and verify
the resulting daemon-side container state; ending a Docker client is not proof that
the daemon operation ended.

The PC2 manager gives SSH an outer deadline at least 15 seconds beyond the
helper's internal deadline. If the SSH process times out or disconnects, the
manager records remote state as unknown, reconnects read-only, and resolves the
transaction by exact helper PID/start-time/argv/token plus daemon postcondition.
It never equates a dead SSH client with a dead remote helper. A live helper
blocks normal lease reclamation. A stale recovery-claim may signal that helper
only after the recovery token fences the old manager and all helper ownership
fields match.

An integration fault test launches a live, bounded snapshot or Docker mutation,
waits for `detached_ready` and the child-start record, forcibly severs the
originating SSH transport, and proves through a new SSH session that the same
helper enforces its internal timeout, reaps its exact child group, records a
terminal state, and leaves the expected daemon postcondition. Killing only the
SSH client or testing only a helper-requested timeout does not satisfy this
gate.

`freeze-provenance` uses the same transaction protocol for tree hashing,
server/checkpoint snapshot staging and operator-owned promotion, and offline
probes. Its write-ahead journal names both mutable intake identities, both
staging paths, the exact weight relative path, expected content-only intake
manifest digests, canonical mode policies, and each installer receipt/
promotion postcondition before the corresponding helper is launched. Final
mode-inclusive tree digests are accepted only from the verified privileged
installer receipts. A crashed
freeze therefore leaves an owned, inspectable helper record; no later
create/evaluate or normal stale reclamation proceeds until the helper is proven
gone or recovered explicitly.

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
gitlinks, PC2 installer and runner socket/service identities, protected closure and
base-Python completion/receipts, construction owner record, closure/base/
identity descriptors, venv/base loader/interpreter/import/native closure,
runner UID/GID/private-root contract, configured path roots, and initial PC2
GPU inventory. An incomplete or
recoverable local construction is handled
under the local construction lock before this read-only gate; unknown ownership
fails closed. Failure reaches `PROVENANCE_BLOCKED` without touching H100
control state.

After the atomic lease is acquired, the remaining preflight is read-only except
for the two tracked runner-supervised PC2 loader/CUDA probe children and
completes before the container is started. Across the two phases it must:

1. revalidate the current lease token and heartbeat;
2. confirm bounded SSH connectivity and collect the remote host identity,
   boot ID, and wall clock;
3. attest the complete sealed PC2 source, exact clean recursive gitlinks,
   venv/protected-base-Python/native environment, episode data, task assets,
   immutable requirements, normalization results, import origins, and offline
   data-root behavior against the profile; attest the runner binary/config/
   units, dedicated evaluator UID/GID and private-root contract, and pass the
   model-free direct-loader mapping/access probe;
4. attest the H100 input-installer executable, protected source, `.venv`,
   offline HF snapshot, profiled checkpoint tracer and exact security/argv
   contract, protected seccomp host path/digest, source installer receipt and
   completion metadata, and failed lifecycle-account write probe;
5. attest the protected checkpoint snapshot, installer completion metadata,
   complete tree, exact weight relative path and file size/hash, and failed
   lifecycle-account write probe, plus both source spot hashes;
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
12. confirm and hash all local episode inputs and pass the simulator-free
    episode-2 and episode-3 asset probes; and
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
  --run-dir=/checkpoint --ckpt-step=40000 \
  --action-exec-horizon=24 --rtc
```

This is the unchanged official server CLI; no new PSI0 option, resolver hook,
source patch, or server-written attestation is introduced. Before continuation
of the stopped bootstrap, the manager independently opens the protected
`checkpoint_weight_relative_path` beneath `/checkpoint` with no-follow
component traversal and validates its path, inode, size, SHA-256, whole-tree
manifest, and read-only mount against the profile. No mutable checkpoint or
broad model cache is mounted, so the official `--run-dir=/checkpoint
--ckpt-step=40000` resolver has no alternate run tree.

To bind that unchanged loader to the protected path, freeze requires a
`strace` executable already present in the digest-pinned image and records its
path, file SHA-256, and version in the profile; runtime never installs a tool.
The fixed argv template, in this exact token order and with only the validated
run-specific output path substituted, is:

```text
<checkpoint_tracer_path>
--interruptible=anywhere
--follow-forks
--decode-fds=path
--trace=%file
--absolute-timestamps=format:unix,precision:ns
--output=/runtime/runs/<run-id>/server/checkpoint-access.raw
--
/workspace/Psi0/.venv/bin/python
-m
psi.deploy.psi0_serve_simple
--host
0.0.0.0
--port
22185
--device
cuda:0
--policy=psi0
--run-dir=/checkpoint
--ckpt-step=40000
--action-exec-horizon=24
--rtc
```

Canonical JSON encoding of that token template is
`checkpoint_tracer_argv_sha256`; runtime also records the instantiated argv
digest. `--kill-on-exit`, `--seccomp-bpf`, every `--daemonize`/`-D` form,
attach mode, an output pipe (`-o |...` or `-o !...`), and all unprofiled
options are forbidden. `--interruptible=anywhere` is mandatory because
`-o FILE PROG` otherwise defaults to non-interruptible tracing, as specified by
the [`strace(1)` interruptibility contract](https://man7.org/linux/man-pages/man1/strace.1.html).
The manager
sends `SIGINT` by pidfd to the tracer PID only, never its process group; the
required terminal observation is that the tracer is reaped as
`WIFSIGNALED && WTERMSIG == SIGINT` after detaching every tracee, while the
recorded server PID/start-ticks remains alive and unchanged. No TERM/KILL is a
successful detach path.

After its ownership record is durable, the stopped bootstrap continues by
`execve`ing that exact tracer. The tracer starts exactly one official server
child, follows that child and its owned descendants, records decoded `%file`
syscalls, and writes to the exclusively created regular output file beneath
the run directory. The stopped bootstrap sets and verifies
`RLIMIT_FSIZE={268435456,268435456}` before `execve`; no descendant may raise
it. A five-second monitor rejects a missed poll or a file reaching the limit;
`SIGXFSZ`, truncation, replacement, non-regular output, or an unexpected writer
is a hard failure and triggers owned cleanup. The tracer and server
PID/start-ticks/argv/parentage/cgroup plus the first trace-header identity are
fsynced before readiness can advance. The parser retains and validates every
`/checkpoint` access. After model readiness and before warm-up, the manager
interrupts only that tracer, requires a clean bounded detach that leaves its
recorded server child alive, copies and seals the raw trace, and parses it into
`checkpoint-access.json`. The record must prove the exact owned server process
opened the profiled relative weight as a regular file without write flags and
opened no alternative checkpoint payload. The fixed official loader source
hash, exact run-dir/step argv, protected tree, observed open, and successful
model-backed warm-up jointly attest the loaded input without changing PSI0.
Tracer startup/detach failure, missing expected open, unexpected checkpoint path, PID
mismatch, trace truncation, or a surviving tracer fails readiness and triggers
owned cleanup.

`create` must pass a model-free trace-and-detach probe before accepting the
container. The profiled read-only helper
`/workspace/Psi0/runtime-tools/checkpoint_trace_probe.py` and a fixed sentinel
are part of the protected server snapshot; the helper digest is
`checkpoint_tracer_probe_sha256` and the sentinel digest is
`checkpoint_tracer_probe_sentinel_sha256`. `freeze-provenance` copies these reviewed
manager-owned auxiliary files into the otherwise sealed snapshot after
verifying the official PSI0 intake; `runtime-tools` is outside `src/psi`, is
absent from `PYTHONPATH`, and cannot alter the official server module or
checkpoint resolver. The probe argv is the exact tracer prefix
above through `--`, with only output replaced by
`/runtime/runs/<run-id>/create/checkpoint-trace-probe.raw` and the command
replaced by these exact tokens:

```text
/workspace/Psi0/.venv/bin/python
/workspace/Psi0/runtime-tools/checkpoint_trace_probe.py
--sentinel=/workspace/Psi0/runtime-tools/checkpoint_trace_sentinel
--ready-path=/runtime/runs/<run-id>/create/checkpoint-trace-probe.ready
--wait-seconds=30
```

Its canonical placeholder-template digest is
`checkpoint_tracer_probe_argv_sha256`. Under the exact container capability,
seccomp, namespace, and no-new-privileges contract, the manager invokes the
same tracer prefix/options against that helper, waits for the helper to open
the sentinel and enter a bounded wait, sends tracer-only `SIGINT`, and requires
the exact tracer terminal state above, a complete sentinel open record, and a
still-live unchanged helper. It then signals and reaps only that owned helper
within five seconds and proves no tracer/helper remains. Probe failure leaves
the container stopped and blocks server launch. No model, CUDA allocation,
network listener, or PSI0 server is started by this probe.

It writes its run-specific log beneath `/runtime/runs/<run-id>`, but all
ownership records remain outside the container at
`/mnt/data01/jhkim/psi0-simple-eval-control/runs/<run-id>`. The remote launch
helper first persists the write-ahead command/container/cgroup/token target,
then uses `docker exec` to start a minimal in-container bootstrap carrying a
one-use launch nonce. Before loading Python/model code, that bootstrap verifies
its fixed argv and stops itself with `SIGSTOP`. The host helper locates exactly
that stopped process through container init descendants, nonce, namespace PID,
host PID/start ticks, and cgroup; it records and fsyncs those identities,
parentage, exact final traced command digest, and lease tuple in manager control state.
Only after that record is durable does it send `SIGCONT`, after which the
bootstrap `execve`s the profiled tracer and that tracer launches exactly one
unchanged server child. The manager adds the tracer/server parent-child and
process identities to the record before accepting readiness. A crash before
the control record leaves only an inert, write-ahead-attributable stopped
bootstrap. A crash after continuation but before the child record is complete
is reconciled from the exact bootstrap/tracer ancestry, nonce, cgroup, and
write-ahead command; an unexpected descendant is `FOREIGN_BLOCKED`. Recovery
can therefore identify the stopped, tracing, and fully running states without
assuming a child record already exists.
A PID is considered owned only when the container identity, manager-only
record, process start time, cgroup, namespace mapping, and normalized command
all agree. Container-writable PID files are diagnostic only and never establish
ownership. A port occupant without complete manager-control proof blocks
startup and is never signalled.

Readiness has four gates:

1. the manager-owned checkpoint-access trace proves the unchanged official
   loader opened the protected exact relative weight for step 40000;
2. the owned process remains alive and container port 22185 is listening;
3. the SSH tunnel is alive and the PC2 loopback endpoint accepts connections;
4. one schema-valid `/act` warm-up returns HTTP 200 with a finite NumPy action
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

The supervisor passes `/run-output/episode_<N>/videos`; its new host-side
evidence path is:

```text
<run-dir>/evaluator-output/episode_<N>/videos
```

The worker creates the episode subdirectory beneath it, producing these exact
per-camera paths:

```text
<run-dir>/evaluator-output/episode_<N>/videos/episode_<N>/<camera>.raw.mp4
<run-dir>/evaluator-output/episode_<N>/videos/episode_<N>/<camera>_<verdict>.mp4
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
20-minute deadline and exit record. The supervisor, not the lifecycle account,
constructs the following effective private-root environment from the reviewed
runner configuration. `<N>`, `<run-id>`, `<episode-nonce>`, and `<local-port>`
are the only enumerated launch-request substitutions; `PYTHONPATH` is the exact
ordered sandbox-path list whose relative origins are in the profiled import
manifest:

```text
SIMPLE_DISABLE_TUI=1
CUDA_VISIBLE_DEVICES=1
PYTHONDONTWRITEBYTECODE=1
SIMPLE_DATA_ROOT=/sealed/closure/task-data
SIMPLE_ASSET_OFFLINE=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HOME=/run-output/generated-cache/home
TMPDIR=/run-output/generated-cache/tmp
TMP=/run-output/generated-cache/tmp
TEMP=/run-output/generated-cache/tmp
XDG_CACHE_HOME=/run-output/generated-cache/xdg
MPLCONFIGDIR=/run-output/generated-cache/matplotlib
TORCH_HOME=/run-output/generated-cache/torch
TRITON_CACHE_DIR=/run-output/generated-cache/triton
OV_CACHE_DIR=/run-output/generated-cache/ov
CUDA_CACHE_PATH=/run-output/generated-cache/cuda
NUMBA_CACHE_DIR=/run-output/generated-cache/numba
PYTHONPATH=<profiled sandbox source/submodule paths>
```

The supervisor duplicates event-write, acknowledgement-read, closure-root,
base-Python-root, and identity-sidecar FDs to fixed child descriptors `3`
through `7`, opens the loader as launcher-only FD `8`, and executes this exact
argv without a shell or external timeout wrapper:

```text
8
--inhibit-cache
--library-path
/sealed/closure/venv/lib:/sealed/base-python/lib:/sealed/base-python/lib64:/sealed/base-python/usr/lib:/sealed/base-python/usr/lib64
--argv0
/sealed/closure/venv/bin/python
/sealed/closure/venv/bin/python
-m
simple.cli.eval_decoupled_wbc
simple/G1WholebodyXMovePickTeleop-v0
psi0_decoupled_wbc
train
--data-format
lerobot
--data-dir
/sealed/closure/episode-data
--host
127.0.0.1
--port
<local-port>
--sim-mode
mujoco_isaac
--headless
--eval-dir
/run-output/episode_<N>/eval-logs
--video-output-dir
/run-output/episode_<N>/videos
--runtime-evidence-path
/run-output/episode_<N>/wbc-runtime-contract.json
--runtime-evidence-run-id
<run-id>
--runtime-evidence-nonce
<episode-nonce>
--runtime-event-fd
3
--runtime-ack-fd
4
--runtime-root-fd
5
--runtime-base-python-root-fd
6
--runtime-identity-fd
7
--num-episodes
1
--episode-start
<N>
--num-workers
1
--save-video
```

The manager owns the episode deadline through the supervisor control protocol;
only the supervisor may signal or reap the evaluator process group.

There is no `--third-person-video` flag. The evaluator must retain
`sim`/`lo`/domain 0 isolation and must not open a real Unitree interface.
Immediately before `gym.make`, the worker atomically emits the actual
`sonic_config` fields it will consume to the episode's runtime-evidence file.
The manager requires `ENV_TYPE="sim"`, `INTERFACE="lo"`, and `DOMAIN_ID=0`
from that worker-produced record; a separately reconstructed default or a
hard-coded zero counter is not acceptable evidence. The worker refuses any
other values before environment or channel construction.

The three runtime-evidence options and all five runtime channel/root/identity
FDs are supplied as one indivisible contract or rejected by the parent before
spawning. `run-id` must equal the manager's run ID, and the episode nonce is a
fresh 128-bit lowercase hexadecimal value used only once.
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
simple_root_tree: 40-character lowercase Git tree SHA
recursive_gitlinks_sha256: 64-character lowercase hexadecimal string
pc2_closure_id: 64-character lowercase hexadecimal string
pc2_source_tree_sha256: 64-character lowercase hexadecimal string
pc2_base_python_tree_sha256: 64-character lowercase hexadecimal string
pc2_base_python_loader_sha256: 64-character lowercase hexadecimal string
pc2_base_python_root_identity_sha256: 64-character lowercase hexadecimal string
pc2_runtime_identity_sha256: 64-character lowercase hexadecimal string
pc2_closure_root_identity_sha256: 64-character lowercase hexadecimal string
pc2_runner_sandbox_contract_sha256: 64-character lowercase hexadecimal string
runtime_identity_sidecar_sha256: 64-character lowercase hexadecimal string
evaluator_uid: integer exactly pc2_evaluator_uid
evaluator_gid: integer exactly pc2_evaluator_gid
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

The source, closure, base-loader, and runner identity values in this record
come only from the validated inherited identity FD and supervisor launch
record. The evaluator computes and retains the exact sidecar digest before
spawning its worker, the worker carries the already parsed immutable object,
and the manager binds that digest plus the supervisor-attested UID/GID and
sandbox digest to the reviewed profile through the contract event and
acknowledgement; no layer calls Git.
Closing, replacing, or mutating the identity descriptor before the durable
record and contract event are emitted produces a rejected contract and zero
environment construction.

The lifecycle manager creates a fresh event pipe and acknowledgement pipe and
opens the verified protected closure root, protected base-Python root,
identity sidecar, and new empty run-output root before each evaluator. It
continuously drains the event read end and retains the acknowledgement write
end. It transfers only the opposite pipe ends plus the three read-only input
FDs and configured workload-parent FD to the runner supervisor in the strict
six-FD launch request. The supervisor creates/mounts the output child,
duplicates only the five
runtime/event FDs to descriptors 3--7, and performs the evaluator exec.

All five evaluator FD options are required together and accepted only with
`--num-workers 1`; the evaluator validates their direction/type, sandbox path
bindings, and exact sidecar bytes, closes them on every exit path, and does not
pass them to the simulator, policy client, or unrelated descendants. The
existing in-process `progress_reporter` remains a TUI consumer but is not
safety evidence. Every worker `report()` first emits one canonical JSON line
to the manager event pipe with a single `os.write` no larger than `PIPE_BUF`,
then invokes the optional TUI callback.

Each event has the exact schema below and no additional keys or coerced types:

```text
schema_version: integer exactly 1
run_id: exact current run ID
episode_index: integer exactly 2 or 3
sequence: positive integer starting at 1 and increasing by one
event: nonempty enumerated string
evaluator_pid: exact positive supervisor-reported child PID
worker_pid: exact positive PID, equal to evaluator_pid for num_workers=1
created_at_monotonic_ns: positive integer
payload: exact event-specific object
```

Allowed event payloads are exact: `runtime_contract` carries `status`,
`record_sha256`, `file_device`, `file_inode`, `file_size`, `simple_commit`,
`simple_root_tree`, `recursive_gitlinks_sha256`, `pc2_closure_id`,
`pc2_source_tree_sha256`, `pc2_base_python_tree_sha256`,
`pc2_base_python_loader_sha256`,
`pc2_base_python_root_identity_sha256`, `pc2_runtime_identity_sha256`,
`pc2_closure_root_identity_sha256`, `pc2_runner_sandbox_contract_sha256`,
`runtime_identity_sidecar_sha256`, `evaluator_uid`, and `evaluator_gid`;
`worker_init`
with `status="creating_env"` carries `total_episodes=1`, while
`status="ready"` also carries finite `setup_seconds`, `max_episode_steps`, and
`video_path`; `episode_start` carries `episode`; `episode_step` carries
`episode` and positive `step`; `episode_end` carries those plus
`completed_episodes`, `successes`, finite `episode_seconds`, and finite
`steps_per_second`; `worker_status` carries `status="closing"`; `worker_done`
carries `completed_episodes` and `successes`; and `worker_error` carries an
enumerated `phase` and nonempty `error_code`. A rejected runtime contract uses
the evidence file's hash/identity in the same payload. Tracebacks remain in the
evaluator log and are not accepted as unbounded pipe payloads.

The single acknowledgement line has exact keys `schema_version=1`, `run_id`,
`episode_index`, `accepted_sequence`, `record_sha256`, `simple_commit`,
`simple_root_tree`, `recursive_gitlinks_sha256`, `pc2_closure_id`,
`pc2_source_tree_sha256`, `pc2_base_python_tree_sha256`,
`pc2_base_python_loader_sha256`,
`pc2_base_python_root_identity_sha256`, `pc2_runtime_identity_sha256`,
`pc2_closure_root_identity_sha256`, `pc2_runner_sandbox_contract_sha256`,
`runtime_identity_sidecar_sha256`, `evaluator_uid`, and `evaluator_gid`; all
identities must equal the profile, inherited sidecar, supervisor launch record,
contract event, and durable file. No other acknowledgement bytes or keys are
accepted.

The manager appends validated lines to
`episode_<N>/runtime-events.jsonl`, records receive time, and rejects a line
larger than `PIPE_BUF`, malformed UTF-8/JSON, unknown event/payload keys,
wrong identity, duplicate/gapped/out-of-order sequence, silence past the
event-specific deadline, or EOF before the terminal `worker_done`/`worker_error`
event. A full pipe, partial write, or event-channel write error makes the worker
fail before proceeding; safety events are never dropped.

The manager accepts only a `validated` record whose run ID, episode index,
nonce, worker ID/PID, commit/tree/gitlinks, closure/runtime identity, exact
config, file identity, and creation interval match the current evaluator,
profile, and inherited sidecar. The file must be created after that evaluator's
recorded process start and before its first environment-construction progress
event. The current `worker_init(status="creating_env")` report is moved: the
worker must durably create the evidence file, emit
`runtime_contract(status="validated", record_sha256=..., file_identity=...)`,
then wait for the manager acknowledgement. The manager independently opens and
hashes the absent-then-created record, verifies the event identity, and writes
one exact canonical acknowledgement containing every field listed above. The
worker validates it within five seconds,
emits `worker_init(status="creating_env")`, and only then calls `gym.make`.
Missing, malformed, mismatched, duplicate, or late acknowledgement fails with
zero `creating_env` event and zero `gym.make` call.

For an unsafe contract, the worker durably creates the rejected record, emits
`runtime_contract(status="rejected")`, and raises to the existing worker-error
path. It emits no `creating_env` event and makes zero `gym.make` calls. Missing,
stale, replayed, replaced, or cross-episode evidence fails the run. Episodes 2
and 3 have different paths and nonces; neither record can satisfy the other
episode.

The manager accepts environment construction only after it has observed the
contract event, validated the file, acknowledged it, and then observed
`creating_env` at the next sequence. For rejection it
requires `runtime_contract(rejected)` followed by `worker_error`, EOF, and zero
`creating_env` event. Process-boundary tests launch the actual evaluator CLI
through the runner supervisor with both inherited pipes and a fake `gym.make`,
proving validated and rejected ordering, acknowledgement failures,
malformed-event/early-EOF handling, exact UID/private-root state, and that no
in-process callback substitution can satisfy the manager.

Before the next episode starts, the prior evaluator must have exited and its
local worker/WBC children must be gone. If cleanup after one episode cannot be
proven, the second episode is not started. A normal task-level failure may
proceed to the next episode; an infrastructure or cleanup failure may not.

The runner supervisor starts each evaluator as a new local process group. Its
attested launch response and `/proc` reconciliation record the leader PID,
leader process start time, process-group ID, evaluator UID/GID, sandbox
namespace identities, and exact argv. These define local ownership. Worker and
WBC descendants are enumerated from that group and process tree; cleanup never
matches processes by name alone, and the lifecycle manager requests signals
through the supervisor rather than signaling the group itself.

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
| PC2 GPU ownership poll | 10 s | next five-second poll is not skipped |
| Each offline episode asset probe | 120 s | tracked local-process cleanup |
| Complete remote snapshot verification | 600 s | 615 s |
| Remote provenance freeze/copy/probe | 1800 s | 1815 s |
| Remote Docker start or stop helper | 30 s | 45 s |
| PC2 CUDA allocation/UUID probe | 30 s | 45 s process reconciliation |
| Model process and TCP readiness | 300 s total | owned server reconciliation |
| Warm-up `/act` response | 120 s | owned server reconciliation |
| Each evaluator episode | 1200 s | 75 s runner-supervised INT/KILL/reap |
| Each FFmpeg finalization | 120 s | bounded owned signal stages |
| Tunnel signal stage | 2 s each for INT, TERM, and KILL | fresh liveness check |
| Checkpoint tracer detach/signal | 5 s each for INT, TERM, and KILL | preserve trace and verify server identity |
| Server signal stage | 5 s each for INT, TERM, and KILL | fresh liveness check |
| Final remote cleanup verification | 30 s | 45 s |

Cleanup is registered before each resource is started and executes in reverse
order. It is attempted in this order:

1. request bounded interrupt/KILL/reap of the active evaluator and its owned
   children through the runner supervisor, then close its control connection;
2. close the SSH tunnel;
3. detach or stop the exact owned checkpoint tracer if it remains, preserving
   its bounded trace and revalidating the server child;
4. stop the owned server inside the container, unless ownership has become
   foreign or unknown;
5. copy final remote evidence;
6. stop the dedicated container only when the current lease started it and all
   remaining workloads are owned by the current run; and
7. verify local and remote postconditions.

Remote INT, TERM, and KILL, and local runner INT/KILL, are used only after
ownership is revalidated at that stage. After KILL, a fresh bounded wait and
liveness check are mandatory.
Failure in one cleanup action is recorded but does not skip later actions.
Terminal restoration, open log closure, and manifest finalization are
unconditional even after the shared cleanup budget is exceeded.

If an ownership check in steps 3, 4, or 6 is foreign or unknown, the manager
records `FOREIGN_BLOCKED` and does not send a signal or stop the container. It
still closes its proven local evaluator/tunnel resources and finalizes
evidence. This exception to the desired exited postcondition is always a
non-PASS terminal state and leaves the lease marked for operator inspection.

The persistent container object remains for reuse, but its required terminal
state for PASS is `exited`. A run cannot pass if the container is running, the
owned server, checkpoint tracer, or tunnel is alive, either relevant port is listening, an
evaluator or owned WBC child remains, either selected GPU retains an owned
process, any sealed PC2/H100 input or the checkpoint tree changed, or liveness
is unknown.

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
  mutation-journal.jsonl
  recovery-claim.json
  final.json
preflight/
  ssh-host.json
  runtime-profile.json
  profile-hash.json
  h100-roots-identity.json
  h100-input-installer.json
  source-installer-receipt.json
  checkpoint-installer-receipt.json
  seccomp-profile.json
  container-identity.json
  pc2-construction-final.json
  pc2-input-installer.json
  pc2-installer-service.json
  pc2-installer-receipt.json
  pc2-runner-service.json
  pc2-runner-sandbox.json
  pc2-base-python-completion.json
  pc2-base-python-manifest.json
  pc2-loader-closure.json
  pc2-loader-probe.json
  pc2-base-python-root-before.json
  pc2-base-python-installer-receipt.json
  pc2-closure-descriptor.json
  pc2-closure-completion.json
  pc2-closure-root-before.json
  pc2-runtime-identity.json
  pc2-runtime-identity-file.json
  pc2-source-before.json
  pc2-gitlinks-before.json
  pc2-venv-before.json
  pc2-import-origins.json
  pc2-native-closure.json
  pc2-episode-data-before.json
  pc2-task-assets-before.json
  pc2-asset-requirements.json
  hssd-normalization-results.json
  episode-inputs.json
  source-tree-manifest.json
  source-tree-verification-before.json
  h100-gpu-before.json
  pc2-gpu-before.json
  pc2-cuda-probe.json
  configured-gpu7-containers.json
  container-inspect.json
  source-hashes.json
  checkpoint-tree-manifest.json
  checkpoint-completion.json
  checkpoint-weight-identity.json
  checkpoint-tracer-identity.json
  checkpoint-tracer-probe.json
  container-security-contract.json
  checkpoint-before.json
server/
  command.json
  pid-namespace-cgroup-map.json
  checkpoint-access.raw
  checkpoint-access.json
  checkpoint-tracer-process.json
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
  runner-process.json
  executable-map-audit.json
  filesystem-write-audit.json
  runtime-events.jsonl
  asset-probe.json
  h100-gpu-before.json
  h100-gpu-after.json
  pc2-gpu-before.json
  pc2-gpu-after.json
  pc2-gpu-monitor.jsonl
  result.json
  artifacts.json
episode_3/
  command.json
  runner-process.json
  executable-map-audit.json
  filesystem-write-audit.json
  runtime-events.jsonl
  asset-probe.json
  h100-gpu-before.json
  h100-gpu-after.json
  pc2-gpu-before.json
  pc2-gpu-after.json
  pc2-gpu-monitor.jsonl
  result.json
  artifacts.json
evaluator-output/episode_2/
  evaluator.log
  wbc-runtime-contract.json
  videos/episode_2/<camera>.raw.mp4
  videos/episode_2/<camera>_<verdict>.mp4
evaluator-output/episode_3/
  evaluator.log
  wbc-runtime-contract.json
  videos/episode_3/<camera>.raw.mp4
  videos/episode_3/<camera>_<verdict>.mp4
cleanup/
  actions.json
  h100-gpu-after.json
  pc2-gpu-after.json
  pc2-source-after.json
  pc2-gitlinks-after.json
  pc2-venv-after.json
  pc2-runtime-identity-after.json
  pc2-construction-after.json
  pc2-closure-root-after.json
  pc2-base-python-root-after.json
  pc2-base-python-manifest-after.json
  pc2-runner-final.json
  pc2-executable-map-audit-final.json
  pc2-episode-data-after.json
  pc2-task-assets-after.json
  checkpoint-after.json
  source-tree-verification-after.json
  remote-helpers-final.json
  container-final.json
  processes-final.json
  ports-final.json
```

Only `evaluator-output/` is mounted in the child namespace and writable by the
evaluator UID. The manager owns and writes every other evidence path, drains
events directly into the top-level `episode_<N>/runtime-events.jsonl`, and
copies no manager-authored evidence into the child-writable tree. After the
supervised child exits, it no-follow hashes and validates the explicitly
allowlisted evaluator-output files before referencing them from manager-owned
`artifacts.json` and the final manifest.

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
- the digest-pinned image, attested input installer, and complete protected PSI0
  source plus `.venv` snapshot matched the committed profile before and after
  execution;
- the protected H100 source installer receipt bound the exact reviewed seccomp
  host path and bytes, Docker consumed that protected path, and its inspect
  contract matched the same digest;
- the protected checkpoint snapshot, external completion metadata, complete tree, and
  exact named weight relative path matched before startup and after all H100
  cleanup, and neither lifecycle account nor container could write it;
- the sealed SIMPLE source, exact recursive gitlinks, PC2 venv/base Python,
  protected base-Python completion/root/receipt, package/import/native closure,
  episode dataset, task-asset requirements plus normalization results, and
  driver identities matched the committed profile before and after execution;
- the attested PC2 installer normalized metadata before authoritative hashes,
  its root-owned socket/service and stable-construction-lock-FD plus current-
  owner-record peer/token contract matched, the local construction record,
  authoritative root journal, mirrored rename/receipt phases, and protected
  completion/receipt metadata were valid, and lifecycle/evaluator accounts
  could not mutate or rename either protected snapshot;
- the inherited protected closure-root, base-Python-root, and identity
  descriptors remained bound to the executed interpreter/import/data paths,
  and their identities matched every
  durable WBC record, contract event, and acknowledgement without consulting
  `.git`;
- the attested runner executed under the dedicated evaluator UID/GID in the
  exact private-root sandbox, exposed no installer/control/staging path, used
  the protected loader by FD, and recorded no executable mapping or loader
  access outside the closure/base roots;
- both episode-specific offline asset probes covered their exact manifest
  subsets with zero download, write, network, or simulator construction;
- the real `hssd:scene0` manager path used only normalized closure-local USD
  dependencies with no subprocess, shell, global `/tmp` copy, or write outside
  the episode run directory;
- container CUDA device 0 UUID equalled host GPU 7 UUID;
- every H100 GPU poll contained only the exact attested server process tree;
- the exact seccomp/capability/no-new-privileges contract and model-free tracer
  probe passed; the external checkpoint-access trace attributed the profiled
  protected weight open to the unchanged owned official server, and the
  interruptible tracer detached cleanly without kill-on-exit;
- PC2 CUDA device 0 under `CUDA_VISIBLE_DEVICES=1` equalled physical GPU 1's
  UUID, passed allocation, and had no foreign compute process at any gate;
- every PC2 episode GPU poll contained only exact evaluator descendants and no
  required poll was missed;
- the canonical warm-up digest matched and returned a finite `(24, 36)` action;
- both runner-supervised episode evaluator processes reached a normal recorded
  exit and their supervisor control sessions closed cleanly;
- each episode produced a recorded SIMPLE verdict, retained raw stereo MP4s,
  and valid checked verdict MP4s under its unique run directory;
- worker-emitted runtime evidence proved the actual WBC configuration was
  `sim` / `lo` / domain 0 before each environment was constructed, with a
  distinct current nonce for each episode;
- the manager-owned event channel proved the durable WBC record,
  identity-bound `runtime_contract`, acknowledgement, `creating_env`, and
  construction order without malformed data or early EOF;
- no real-robot control process or non-loopback Unitree interface owned by the
  run was observed;
- every owned process and port was absent after cleanup;
- every owned checkpoint tracer was absent after cleanup;
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
- **PC2 protected closure/base-root/FD mismatch, loader escape, or attempted
  swap:** fail before worker or environment construction when possible;
  otherwise request stop of only the owned evaluator through the runner.
  Preserve descriptor/path/mapping identities and never adopt the new path.
- **Runner identity, namespace, or authority failure:** reject launch before
  exec, or stop/reap only the already supervised child. A missing runner
  terminal record, surviving child, evaluator access to either privileged
  socket/control root, or host-library mapping is cleanup failure.
- **Checkpoint trace failure:** stop only the owned unchanged server/tracer,
  preserve the bounded raw trace, and fail readiness; never infer a successful
  load from the HTTP listener alone.
- **Provenance changes:** reject before startup when found initially; if the
  post-run complete tree verification differs, preserve both manifests and fail
  even when evaluation otherwise succeeded.
- **Unexpected server or port occupant:** preserve diagnostics and do not
  signal it or stop a container holding it.
- **Server crash or invalid warm-up:** preserve the complete server log and
  clean up without launching an evaluator.
- **Tunnel failure during evaluation:** request runner-supervised evaluator
  interrupt, then perform normal cleanup; do not retry within the same run
  directory.
- **Evaluator timeout:** use the runner's bounded INT/KILL behavior, prove the
  child and descendants are gone, and do not start another episode unless
  cleanup passed.
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
- write-ahead `cleanup_required` and pending-mutation persistence before every
  resource-creating action, crash injection immediately after each action, and
  operation-specific freeze/create/evaluate recovery without unrelated record
  requirements;
- the global lock order, heartbeat operations never taking `mutation.lock`, and
  heartbeat continuity/nonexpiry throughout a fake 1,800-second mutation;
- two concurrent stale recoverers with exactly one recovery-claim winner, the
  old manager fenced before signals, and every loser issuing zero signals;
- running-container stale leases producing `STALE_OWNED_BLOCKED`, and foreign
  or unknown process/port cases producing `FOREIGN_BLOCKED` without any stop or
  signal command;
- pairwise-disjoint H100 roots, every Docker bind alias, read-only input writes,
  inaccessible control paths, writable workload path, exact labels/mounts, and
  no published ports;
- protected seccomp materialization beneath the source snapshot, exact
  H100-host path/receipt/completion binding, no-follow pre/post create hashes,
  Docker inspect digest equality, and refusal of a missing, mutable,
  path-mismatched, or run-uploaded profile before `docker create`;
- predetermined image-digest enforcement, complete source plus `.venv` manifest
  generation, protected snapshot installation, and rejection of a changed
  source file, dependency file, symlink, image, or protected checkpoint before
  startup or a checkpoint-tree change after cleanup;
- checkpoint snapshot copy and atomic promotion, exact normalized weight
  relative-path selection, missing/duplicate/wrong-file rejection, source-tree
  mutation after freeze having no effect on mounted bytes, lifecycle/container
  write-probe refusal, unchanged official step-40000 command/source hashes,
  stopped-bootstrap tracer launch, exact protected-weight open attribution,
  exact interruptible argv-template hashing, forbidden kill-on-exit/daemonize/
  seccomp-bpf rejection, exact capability/seccomp/no-new-privileges inspect,
  model-free create-time trace/detach probe, unexpected-path/truncated-trace/
  failed-detach rejection, bounded tracer cleanup, and startup using only
  `/checkpoint`;
- installer executable attestation, current freeze-token enforcement,
  server/checkpoint kind restriction, arbitrary source/destination and existing
  target refusal, same-device atomic publication, and absence of general input-
  root write privilege from the lifecycle account;
- H100 server/checkpoint final ownership/mode normalization preceding every
  authoritative manifest hash, completion metadata excluded from payload
  hashes and hashed separately, snapshot root/metadata made non-writable last,
  and rejection when any post-hash metadata differs;
- idle configured GPU-7 containers being allowed and active compute processes
  being rejected;
- H100 host/container PID namespace and cgroup mapping, pre-server and
  before/during/after-episode polls, server-descendant allowlisting, missed-poll
  failure, and foreign-process handling that stops only owned resources;
- PC2 GPU-1 UUID mapping, CUDA allocation success, UUID mismatch, probe timeout
  cleanup, active foreign-process refusal, five-second during-episode polling,
  exact evaluator-descendant attribution, missed/malformed poll failure, and
  every before/between/after recheck without signalling foreign users;
- sealed PC2 SIMPLE source construction from Git objects, exact recursive clean
  gitlinks, dirty/untracked submodule refusal, venv/base-Python/native manifest,
  editable-path escape rejection, import-origin validation, ignored
  episode-data and complete task-asset sealing, offline data-root/download-write
  refusal, per-episode recursive asset probes, and all-input pre/post mutation
  detection;
- local PC2 constructor ownership/heartbeat, exactly one concurrent constructor
  and stale recoverer, write-ahead phase updates, crash injection after staging
  allocation, every source/data/venv/HSSD/installer phase, final metadata
  normalization, authoritative hashes, sidecar/completion metadata, atomic
  rename, explicit `FINAL_RENAMED` and `RECEIPT_CREATED`, crash recovery
  between rename and receipt, and published revalidation, with no partial final
  closure and exact adoption only after a post-rename crash;
- privileged PC2 installer transport through the root-owned Unix socket,
  `SO_PEERCRED` plus one `SCM_RIGHTS` stable construction-lock FD, independent
  no-follow opening and token/PID/phase validation of the current owner record,
  rejection of an owner-record FD masquerading as the lock, owner-record
  invariant mutation during a live transaction, acceptance of heartbeat-only
  atomic replacement while the same stable lock remains held, wrong lock
  inode, arbitrary path, or extra FD, strict request/response schemas,
  lifecycle refusal to
  reconfigure/start the service, and exactly one token-bound receipt from two
  concurrent post-rename recoverers;
- authoritative root-journal sequencing of rename then exclusive receipt
  creation before the one terminal response, manager mirroring of
  `FINAL_RENAMED` and `RECEIPT_CREATED` only after journal/response validation,
  and crash recovery before response plus before/after each mirror without an
  intermediate protocol frame;
- protected base-Python publication and recovery through every named phase,
  exact completion/root/receipt identities before closure-ID computation,
  venv interpreter-link resolution beneath the inherited base root FD, and
  rejection of the development `.venv/pyvenv.cfg` external target;
- complete base executable closure discovery for the loader, Python, native
  extensions, Isaac/MuJoCo, CUDA/driver, Vulkan/GL, and declared `dlopen`
  dependencies; direct no-`PT_INTERP` loader `execveat`, exact
  `--inhibit-cache`/`--library-path`/`--argv0` argv, model-free verify/list
  probe, and rejection of any host `/lib*`, loader-cache, external mapping, or
  regular-file access;
- staging-independent relative venv links, rejection of embedded staging/final
  prefixes, successful imports after atomic publication, invalid final marker
  fail-closed behavior, and no deletion of unknown or live-owned paths;
- PC2 final ownership/mode normalization before authoritative component hashes,
  explicit sidecar/completion-metadata exclusion, separately attested sidecar,
  root made non-writable last, and profile generation only from the privileged
  installer's final receipt;
- runtime identity sidecar exact schema/hash/completion binding, source closure
  with no `.git`, missing/wrong/non-regular/replaced identity FD rejection, and
  exact commit/tree/gitlink/closure/runtime identity propagation through the
  durable WBC record, contract event, and acknowledgement;
- freeze-time HSSD USD dependency normalization through USD/Sdf APIs, rejection
  of unexpected external/traversing assets, and the real registered
  `HssdSceneManager.load("hssd:scene0")` resolving every dependency inside the
  sealed task root with subprocess/network/copy calls forbidden and global
  `/tmp` unchanged;
- constructor-order proof that the dedicated staging venv passes its USD/Sdf
  probe before HSSD normalization, with ambient/shared Python imports forced to
  fail and recovery covering both `TASK_DATA_COPY` and `HSSD_NORMALIZED`;
- exact run-scoped HOME/temp/cache/Omni/Isaac/log environment and filesystem
  write audits rejecting every evaluator-created path outside its run root;
- path-independent `pc2_closure_id` reproducibility, exact profile/path mapping,
  proof that changing base/closure paths, mount/device/inode identities,
  completion/receipt bytes, installer service/config/socket units, or
  lifecycle/evaluator UIDs cannot affect the ID while their runtime provenance
  still blocks mismatches,
  immutable pre-normalization requirements changing the ID, normalization
  result/mapping changes leaving the ID fixed but failing result provenance,
  and an exact requirements-hash/closure-ID link in every normalization result;
- strict run/helper/profile/recovery ID validation, `.`/`..` and symlink escape
  rejection, resolved-root containment, and existing-output refusal;
- byte-exact production warm-up serialization including instruction and fixed
  timestamp, canonical request digest, and exact `(24, 36)` finite validation;
- `--video-output-dir` parent-to-worker routing, run-scoped episode paths,
  deferred single initialization after stabilization, no-overwrite behavior,
  raw retention after success and failure, checked transcode success,
  timeout/nonzero/malformed-output handling, and independent opposite-verdict
  retries;
- runner supervisor attestation and strict six-FD launch transport, exact
  operation enums/nullability and episode substitutions, SO_PEERCRED manager
  binding, fixed private-root mount table, dedicated evaluator UID/GID with no
  installer/control/staging access, exact workload-parent inode plus run-marker
  validation, exclusive evaluator-output creation and arbitrary output-FD/path
  refusal,
  fixed five-FD evaluator inheritance, evaluator-output-only writes, launcher socket
  closure on exec, manager-disconnect cleanup, and arbitrary argv/environment/
  FD refusal;
- exact runtime-evidence option/five-FD evaluator pairing and schema/type
  validation, including loader/sandbox/UID/GID identities;
- unsafe `ENV_TYPE`, interface, and domain values each failing before
  `gym.make`, with the rejected record present and no `creating_env` event;
- validated event order of durable evidence, `runtime_contract`,
  `worker_init(creating_env)`, then `gym.make`, plus rejected event order of
  durable evidence, `runtime_contract(rejected)`, then worker error;
- actual manager/runner/evaluator event-pipe, acknowledgement-pipe,
  closure-root-FD, base-root-FD, identity-FD, and workload-parent-FD transport with
  exact supervisor PID/start/process-group response, sequence, private-root
  identity validation, plus malformed UTF-8/JSON,
  partial/oversize write, silence, early EOF, and callback-only rejection;
- same-account closure/base-Python swap attempts after descriptor open and
  before evaluator exec, requiring chmod/rename/unlink/symlink/lookalike
  substitutions to fail; a deliberately swappable fixture must fail the
  corresponding root-FD/path/loader identity validation, while the evaluator
  UID independently proves it cannot invoke either privileged service or
  traverse construction state;
- missing, stale, replayed, wrong-PID, wrong-commit, wrong-nonce, and replaced
  runtime evidence rejection, plus independent valid records for episodes 2
  and 3;
- lifecycle transitions and persistence before external actions;
- PID reuse, command mismatch, process-start-time mismatch, unknown liveness,
  and foreign-port refusal;
- readiness, SSH, warm-up, evaluator, FFmpeg, and artifact timeout paths;
- remote helper internal timeout, SSH outer timeout, reconnection/liveness
  reconciliation, detached-ready descriptor audit, severed originating SSH
  during a live mutation, Docker daemon postcondition, and crashed
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
2. seal the exact SIMPLE source commit, recursive gitlinks, dedicated PC2
   base-Python snapshot, environment, episode dataset, and complete episode-2/3
   asset closure through the local write-ahead constructor, with protected
   base-Python completion preceding closure-ID computation and venv creation
   preceding HSSD normalization; freeze requirements before the closure ID and
   emit normalization results separately; use the attested PC2 installer
   socket service to normalize final metadata,
   compute authoritative manifests, create the identity sidecar/completion
   metadata, and publish beneath the protected root. Run both offline asset
   probes and the real `hssd:scene0` load test, then run remote
   `freeze-provenance` to install both the protected
   server and protected checkpoint snapshots. Verify the PC2 closure ID and
   completion record, both complete H100 closures, exact checkpoint weight
   relative path, protected seccomp path/receipt, failed write probes,
   attested runner service/UID/GID/sandbox, complete loader closure, and
   digest-qualified image, then commit only the candidate profile as approval
   commit `P` without starting the dedicated container;
3. inject and recover every local base-Python/closure-construction crash point,
   including final rename before receipt creation, then run
   two concurrent local constructors/recoverers and two concurrent normal
   remote lease probes, requiring exactly one winner in each race. Attempt the
   same-account post-descriptor-open closure/base-Python swap and require every
   mutation to fail while closure-root/base-root/identity FD bindings remain
   exact. Exercise the strict six-FD runner launch with a model-free loader
   verify/list and `python -I -S` mapping audit; require the evaluator UID to be
   unable to reach installer/control/staging state. Exercise a simulated
   1,800-second remote mutation while heartbeats remain current, and release cleanly;
4. run `status` and local/leased preflight, including path containment, mount
   alias, GPU, runner-service, loader/private-root, remote-helper, and pre/post
   provenance checks;
5. create the container using only the protected seccomp host path, attest its
   receipt plus every mount alias and exact seccomp/capability contract, run
   in-container read-only/inaccessible probes and the harmless
   interruptible tracer detach probe, then leave it exited without loading a
   model;
6. create owned stale fixtures for each operation: an incomplete freeze staging
   tree, an interrupted create, and a running evaluate-labelled container with
   an inert `SIGSTOP` server-bootstrap fixture (the model is not loaded). For
   each, run two concurrent recovery claimers, require exactly one winner,
   prove old-token fencing and the operation-specific terminal predicate, and
   preserve independent evidence;
7. launch the profiled exact-argv interruptible checkpoint tracer from the stopped bootstrap,
   start its unchanged official server child, prove the exact protected weight open, detach the
   tracer cleanly, complete one warm-up, exercise H100 GPU monitoring and
   bounded cleanup, and prove the container is stopped with no owned resources;
8. inject one tunnel interruption, one internally timed-out remote helper, and
   one forcibly severed originating SSH session during a live owned mutation;
   then prove detached execution, remote liveness reconciliation, failure
   evidence, and cleanup; and
9. run official episodes 2 and 3 and validate their run-scoped raw and checked
   standard stereo artifacts, independent WBC evidence, GPU evidence, lease
   release, manager event streams, continuous PC2/H100 GPU monitors, offline
   asset/HSSD probes, runner-supervised dedicated-UID/private-root launches,
   inherited protected-closure/base-root/identity-FD bindings, sealed-loader
   mapping audits, filesystem write audits,
   zero live remote helpers, and all post-run execution/protected-checkpoint
   closure manifests.

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
