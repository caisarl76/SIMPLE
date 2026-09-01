# PSI0 Converter V2 Certification Design

Date: 2026-09-01
Status: approved conversational design; implementation pending written-spec review

## Objective

Harden `scripts/postprocess_psi0.py` so that it can regenerate a PSI0-compatible SIMPLE dataset whose rows, media, statistics, schemas, and provenance are internally consistent and independently certifiable.

The first target is the existing official SIMPLE level-0 bend-pick episode. The converter must be correct for multiple episodes as well: it cannot publish mixed video profiles, misaligned Arrow columns, statistics over discarded frames, or unverifiable converter provenance.

This milestone stops after converter implementation, regeneration into a fresh output, and offline-loader certification. It does not start simulation, training, an inference server, Docker, SSH, or H100 work. A later training smoke test is permitted only after a separate GPU-ownership gate.

## Fixed isolation and data boundaries

Development uses a new branch and worktree based on the converter-only commit:

```text
base commit: 91daf5d3a0994774c6996843326c615ff17119e6
branch:      fix/psi0-converter-v2
worktree:    /home/jihun/work/SIMPLE/.worktrees/psi0-converter-v2
```

The existing feature worktree contains unrelated dedicated-runtime changes and is not an implementation base. It remains untouched.

The regeneration input is the existing untracked dataset at this absolute path, opened read-only:

```text
/home/jihun/work/SIMPLE/.worktrees/psi0-simple-pc2-bridge/outputs/official-psi0-breadth/20260901T021451Z-fd6972d/datasets/datagen-bendpick-l0/simple/G1WholebodyBendPickMP-v0/level-0
```

The only canonical regeneration destination is a fresh sibling of the existing processed dataset:

```text
/home/jihun/work/SIMPLE/.worktrees/psi0-simple-pc2-bridge/outputs/official-psi0-breadth/20260901T021451Z-fd6972d/datasets/processed-psi0-bendpick-l0-v2
```

The existing raw dataset and `processed-psi0-bendpick-l0` output are never modified. The exact regeneration command is deferred to the implementation plan, after the converter change itself is committed.

The first regeneration explicitly selects:

```text
video_key:     observation.rgb_head_stereo_left
skip:          60
downsample:    1
output_fps:    50
episode limit: 1
```

These remain normal validated converter arguments rather than hard-coded values.

## Considered approaches

### Recommended: targeted converter hardening

Refactor the existing converter into small validation, selection, media, schema, statistics, provenance, and publication helpers. Keep the existing 32-D state and 36-D action mappings from commit `91daf5d`, while making all downstream artifacts derive from one retained-row selection and one dataset-wide media decision.

This is the smallest approach that fixes the observed defects and preserves compatibility with the existing workflow.

### Rejected: patch only the generated dataset

Editing Parquet schemas, statistics, metadata, or video files after conversion would leave the converter broken and make regeneration non-reproducible. It would also weaken provenance by separating the published data from the program that produced it.

### Deferred: replace the converter with a generic dataset framework

A new generic pipeline could provide broader extensibility, but it would expand scope beyond the bend-pick certification and increase implementation risk. The focused helpers in this design can later become reusable boundaries if more converters need them.

## Pipeline architecture

The converter is a fail-closed, staged pipeline:

```text
converter/script attestation
          |
stable kernel-released conversion lock
          |
source discovery and full preflight
          |
canonical retained-index construction
          |
dataset-wide media-mode decision
          |
unique sibling staging directory
          |
Parquet + video + metadata + provenance generation
          |
independent staged-output certification
          |
atomic no-replace publication
```

No canonical output or staging directory is created until every source episode passes preflight. All generated data and media use the same immutable preflight records; processing does not rediscover or reinterpret sources mid-run.

## Converter source attestation

The current `git log` lookup identifies a commit but cannot detect uncommitted changes to the executed script. Before source discovery, the converter must:

1. resolve `Path(__file__)` and require it to be the repository-relative `scripts/postprocess_psi0.py`;
2. resolve the most recent commit containing that path;
3. read the exact committed blob with Git;
4. read the executed script bytes from disk;
5. require the two byte sequences to match exactly;
6. compute SHA-256 over the executed bytes.

An invalid commit identifier, missing blob, unexpected script path, Git failure, or byte mismatch terminates before output creation. Successful provenance records both `converter_commit` and `converter_script_sha256`.

This rule means regeneration happens only after the converter implementation is committed. Uncommitted changes to unrelated files do not change the converter byte identity, although final PSI0 certification has its own clean-environment rule.

## Source discovery and preflight

Preflight resolves the exact set of episodes bounded by `--total_episodes` and validates all of them before staging begins.

Global arguments must satisfy:

- `skip >= 0`;
- `downsample > 0`;
- requested output FPS is finite and positive;
- chunk size and episode limit are positive;
- the canonical destination does not exist;
- every requested SIMPLE root and required metadata file exists.

For each episode, preflight must:

- read the Parquet schema and row count;
- require every input column used by `build_vectors` and `build_proprio_obs`;
- require `observation.joint_qpos` and `action` to be fixed-size 43-element
  float vectors;
- require `observation.amo_policy_command` to be a fixed-size 9-element float
  vector;
- require `observation.amo_policy_target_yaw` and
  `observation.amo_policy_turning_flag` to be float scalars;
- require `task_index` to be an `int64` scalar column with the same row count as
  the episode and one constant value throughout that episode;
- reject `frame_count <= skip`;
- construct `retained_indices = np.arange(skip, frame_count, downsample, dtype=np.int64)`;
- require `retained_indices` to be nonempty and strictly increasing;
- resolve exactly one video for the requested `video_key`;
- hard-fail if that video is missing;
- hash the source Parquet and source video;
- probe the source video without modification;
- require one video stream and an unambiguous finite positive FPS;
- require the probed video frame count to equal the Parquet row count;
- validate the codec, pixel format, dimensions, duration, frame rates, and complete audio-stream profile.

Preflight also validates the metadata used after vector construction:

- `meta/tasks.jsonl` contains unique nonnegative integer `task_index` values and
  nonempty string `task` values;
- every Parquet `task_index` has exactly one matching task row;
- `meta/episodes.jsonl` contains unique nonnegative integer `episode_index`
  values, positive integer `length` values, nonempty string-task lists, and a
  string `environment_config` containing valid JSON;
- every selected Parquet filename has exactly one metadata lookup by its source
  episode index; positional list indexing is prohibited;
- the metadata length equals the Parquet row count and its task list contains
  the task text selected by the Parquet task index;
- task remapping across roots is deterministic by ordered first occurrence of
  the task text, and every emitted episode lookup is resolved during preflight.

The probed media identity is a canonical JSON object containing at least:

- video codec name;
- pixel format;
- width and height;
- average and nominal frame rates as reduced rational strings;
- duration;
- frame count;
- ordered audio-stream identities, including codec, sample format, sample rate, channel count, and channel layout.

Malformed or missing probe fields fail preflight. The converter does not silently omit a video while incrementing `total_videos`.

## Canonical retained-row contract

Each episode has exactly one retained-index array. Every emitted vector, scalar, index, statistic, and video frame derives from it.

For an episode with `n = len(retained_indices)`:

```text
frame_index  = 0 .. n-1
episode_index = the new contiguous output episode index repeated n times
index         = global_offset .. global_offset+n-1
task_index    = the remapped task index repeated n times
timestamp     = frame_index / output_fps, stored as float32
next.done     = false for rows 0 .. n-2 and true only for row n-1
```

`global_offset` begins at zero and advances by `n`, so the full dataset index is exactly `0 .. total_frames-1`. The converter never derives output indices by adding offsets to unfiltered source indices.

Python slicing is not used independently to infer cardinality. Non-divisible cases such as 214 source rows, `skip=60`, and `downsample=4` retain exactly the rows named by the canonical index array and therefore cannot produce unequal Arrow column lengths.

## Numeric and Arrow schema contract

All retained values in every emitted floating-point column must be finite before Arrow construction. This includes the seven vector columns and `timestamp`. Any NaN or infinity aborts staged generation; no statistics or metadata are published from invalid values.

The Parquet schema is explicit rather than inferred from Python lists:

| Column | Arrow type |
|---|---|
| `states` | fixed-size `list<float32>[32]` |
| `action` | fixed-size `list<float32>[36]` |
| `observation.hand_joints` | fixed-size `list<float32>[14]` |
| `observation.arm_joints` | fixed-size `list<float32>[14]` |
| `observation.leg_joints` | fixed-size `list<float32>[15]` |
| `observation.prev_torso_rpy` | fixed-size `list<float32>[3]` |
| `observation.prev_height` | fixed-size `list<float32>[1]` |
| `timestamp` | `float32` |
| `frame_index` | `int64` |
| `episode_index` | `int64` |
| `index` | `int64` |
| `task_index` | `int64` |
| `next.done` | `bool` |

Every column has exactly `n` rows. `info.json` uses the same exact vector shapes, including `[32]` for `states` and `[36]` for `action`; `[-1]` is prohibited.

## Dataset-wide media contract

`info.json` exposes one dataset-wide `video_info`, so the pipeline makes one media-mode decision after probing every selected input.

### Copy-all mode

Byte-for-byte copying is allowed only if all conditions hold:

- `skip == 0` and `downsample == 1`;
- every input has the same codec, pixel format, dimensions, average and nominal FPS, and complete audio-stream profile;
- the common source FPS exactly equals the requested output FPS;
- every source video frame count equals its retained-row count.

If copy-all is selected, every episode is copied. Copying only a subset is prohibited.

### Transcode-all mode

If any copy-all condition fails, every episode is transcoded to this canonical profile:

```text
codec:       H.264
pixel format: yuv420p
width:       640
height:      360
fps:         requested output FPS
audio:       absent
frame count: retained row count for that episode
```

The video filter selects exactly `retained_indices` and resets presentation timestamps to `N / (output_fps * timebase)`. Output paths use non-overwriting FFmpeg behavior inside the fresh staging directory.

After copying or transcoding, every output is probed. It must have one video stream, the selected dataset profile, and exactly its episode's retained-row count. Copy-all output hashes must equal their source hashes. Any mismatch aborts publication.

`info.json.features["observation.images.egocentric"].video_info` is generated from the verified dataset output profile. It is never copied from the last source episode. The advertised image shape is generated from that same profile.

The top-level provenance records `media_mode` as `copy_all` or `transcode_all`, plus the canonical output profile. Per-episode provenance records both source and output media identities and hashes.

## Statistics contract

Per-episode and global statistics are computed only from emitted rows, after canonical selection and dtype conversion. Discarded source frames are never included.

All accumulated arrays must have lengths equal to the emitted episode or dataset cardinality. Statistics are independently recomputable from final Parquet files and include the established mean, standard deviation, minimum, maximum, first percentile, and ninety-ninth percentile fields.

Every statistics block emitted for an episode contains `count: [n]`. Every
global statistics block contains `count: [total_frames]`. This applies to all
blocks in `episodes_stats.jsonl`, `stats.json`, and `stats_psi0.json` regardless
of scalar or vector width. `stats.json` and `stats_psi0.json` are written from
the same canonical object and must be byte-for-byte identical.

The JSON writer uses `allow_nan=False` for statistics. The same fail-closed setting is used for all JSON and JSONL output so invalid floating-point values cannot appear as nonstandard JSON tokens.

## Provenance contract

Each episode record includes a `conversion_provenance` object with:

- source episode index;
- source Parquet SHA-256;
- source video SHA-256;
- requested video key;
- requested output FPS;
- skip and downsample;
- exact retained-row count;
- canonical source media identity;
- output video SHA-256;
- canonical output media identity;
- converter commit;
- converter script SHA-256.

Dataset-level provenance includes:

- converter identity;
- normalized invocation parameters;
- ordered input-root identities;
- dataset media mode and canonical output profile;
- ordered episode provenance digests;
- output schema identity.

All SHA-256 values are lowercase 64-character hexadecimal strings. Canonical provenance JSON uses sorted keys, compact separators, UTF-8, and `allow_nan=False` when a digest is computed.

## Staging, manifest, and atomic publication

### Crash-recoverable conversion lock

Before preflight or any staging mutation, the converter opens the stable sibling
path `<parent>/.<final-name>.conversion.lock` with
`O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW` and mode `0600`. It validates a regular
file owned by the current UID with mode `0600` and link count one, then acquires
a nonblocking exclusive `flock` on that open file description. Contention exits
nonzero without creating a staging directory.

The lock file may persist, but ownership is represented only by the kernel lock,
not by file existence or file contents. Normal close, exception unwinding,
process death, and `_exit` release it. The converter holds the locked file
descriptor through publication and the terminal post-publication evidence
write. A replacement process can acquire the same stable lock after a crash,
inspect existing final and staging paths, and fail closed without deleting or
adopting them.

After lock acquisition, the converter rechecks that the canonical destination
is absent and then creates a unique hidden staging directory beside it. It
atomically writes and fsyncs `CONVERSION_STATUS.json` with state `in_progress`
and the staging identity.

### Payload manifest

The staged payload manifest is `meta/conversion_manifest.json`. Its ordered
entries cover every regular dataset file below `data/`, `videos/`, and `meta/`
except the manifest itself. This includes every Parquet file, video, standard
metadata file, statistics file, and dataset/episode provenance file. Each entry
contains the POSIX relative path, byte size, and lowercase SHA-256. Entries are
sorted bytewise by relative path.

Exactly two canonical files are excluded from manifest membership:

- `meta/conversion_manifest.json`, to avoid hashing itself;
- root `CONVERSION_STATUS.json`, because it records the manifest digest and
  publication state.

The complete canonical dataset tree must equal the manifest entry paths plus
those two reserved paths. Extra files, symlinks, devices, sockets, directories
outside the declared layout, and multiply linked files are prohibited. The
manifest is canonical UTF-8 JSON with sorted keys, compact separators, and
`allow_nan=False`; `manifest_sha256` is the SHA-256 of those exact manifest
bytes. The complete status record contains that digest, manifest byte size,
entry count, aggregate covered-byte count, converter identity, and state
`complete`.

Post-publication certification evidence is never stored below the canonical
dataset and is therefore not a manifest member.

### Durable publication ordering

Before publication, while retaining the conversion lock, the converter:

1. flushes and closes every payload writer;
2. validates the complete staged dataset;
3. enumerates the exact payload membership and writes the canonical manifest;
4. atomically replaces `CONVERSION_STATUS.json` with the complete record by
   fsyncing its temporary regular file, renaming it, and fsyncing the staging
   root;
5. normalizes every canonical regular file to mode `0444` and every canonical
   directory to mode `0555`;
6. reopens every manifest-covered file, the manifest, and the complete status
   with no-follow semantics and successfully calls `fsync` on each one;
7. fsyncs every canonical directory bottom-up, ending with the staging root;
8. revalidates exact membership, hashes, modes, file types, link counts, and the
   complete record;
9. rechecks that the canonical destination is absent;
10. publishes with atomic `renameat2(RENAME_NOREPLACE)` on the same filesystem;
11. fsyncs the destination parent directory.

Failure to open or fsync any generated Parquet, video, metadata, provenance,
manifest, or status file aborts publication. Directory fsync alone is not
sufficient. The implementation fails closed if atomic no-replace rename cannot
be guaranteed.

On a caught pre-publication failure, the canonical destination remains absent.
When the staging root remains writable, `CONVERSION_STATUS.json` is atomically
replaced with state `failed`, the phase, and error using `allow_nan=False`. A
process crash can leave a staging directory in `in_progress` or even `complete`
state; any such directory remains noncanonical and is never adopted or removed
automatically. A later run creates a new staging directory only after acquiring
the stable lock. The implementation plan defines an explicit inspection and
authorized-removal command for preserved staging artifacts.

### Immutable dataset and sibling certification evidence

After the parent fsync, the canonical dataset is immutable: converter and
certifier code open it read-only, never chmod or replace its contents, and
revalidate its `0444` files, `0555` directories, exact membership, and hashes.

Each post-publication certification attempt exclusively creates this sibling
root while the conversion lock is still held:

```text
<parent>/.<final-name>.certification-<manifest-sha256>-<certificate-uuid>/
```

The evidence root records the manifest digest, complete-status digest, exact
dataset root identity, certification environment, per-check results, and a
terminal `PASS` or `FAIL` verdict. All evidence files and directories are
fsynced before the evidence parent is fsynced. A process crash can leave a
nonterminal evidence root; it is preserved and never treated as certification.
A new attempt uses a new UUID and never overwrites prior evidence.

A standalone certification retry must acquire and hold the same stable
conversion lock before inspecting the canonical root or creating a new evidence
root.

If post-publication verification fails, the published dataset remains
immutable and is classified `PUBLISHED_UNCERTIFIED`. The certifier writes and
fsyncs a `FAIL` result when possible, returns nonzero, and does not delete,
rename, repair, or certify the dataset. An environmental failure may be retried
in a new evidence root. A data-integrity failure requires diagnosis and a new
canonical output name; the existing output remains preserved evidence. The
conversion lock is released only after the terminal evidence write and parent
fsync, or automatically by the kernel if the process exits.

## Output certification

Certification runs against the staged dataset before publication and again
read-only after publication. Only a terminal sibling `PASS` evidence root bound
to the exact manifest digest certifies the canonical output.

It verifies:

- exact Parquet column names and Arrow types for every episode;
- equal row counts across every column;
- finite values in every float column;
- local and global index formulas;
- timestamp formulas and float32 storage;
- exactly one terminal flag per nonempty episode;
- exact episode and global statistics recomputed from Parquet;
- `count: [n]` in every episode block and `count: [total_frames]` in every
  global block;
- byte-for-byte equality and independent validation of `stats.json` and
  `stats_psi0.json`;
- exact video count;
- per-episode video frame counts and media profiles;
- one truthful dataset-wide `video_info`;
- provenance hashes against source and output bytes;
- converter script bytes against its recorded Git blob;
- complete metadata cardinalities and paths.

The first target is expected to retain 154 rows from the observed 214-row episode with `skip=60`, `downsample=1`, and `output_fps=50`. This is an expectation to verify, not a hard-coded general converter rule.

## PSI0 offline-loader certification

Dataset certification is also bound to an exact PSI0 checkout and Python environment. The certificate records:

- absolute PSI0 repository root;
- exact PSI0 commit;
- clean tracked-worktree result;
- resolved Python executable and Python version;
- platform identity;
- versions of loader-relevant packages, including PyTorch, PyArrow, NumPy, Hugging Face Datasets, TorchVision, and PyAV;
- SHA-256 of a sorted environment package manifest;
- exact loader command and normalized arguments;
- loader exit status and result digest.

The production PSI0 offline loader must open the dataset without network access
and call its production `dataset[i]` retrieval path for every integer index from
zero through `total_frames - 1`. For every returned sample it validates the
declared image, 32-D state, and 36-D action shapes and dtypes; every floating
tensor or array must be finite. The ordered visited-index record must be exactly
`0 .. total_frames-1`, with no gaps, duplicates, sampling shortcut, direct
Parquet substitute, or first/last-only optimization. It also enumerates every
episode and validates episode boundaries. The exact checkout, environment, and
shell commands belong in the implementation plan.

A certificate from a different PSI0 commit or environment does not certify this output.

## Test strategy

Implementation follows test-driven development. New tests are written and observed failing before production changes.

### Selection and row tests

- reject `frame_count < skip` before creating output or staging;
- reject `frame_count == skip` before creating output or staging;
- retain the exact ceiling-like cardinality for non-divisible downsampling;
- require aligned column lengths;
- require continuous frame and global indices;
- require float32 timestamp values equal to `frame_index / output_fps`;
- require one final terminal flag;
- reject missing, duplicate, malformed, length-mismatched, task-mismatched, or
  position-only task/episode metadata lookups before staging;

### Schema, finite-value, and statistics tests

- assert all seven vector columns are fixed-size float32 lists with exact dimensions;
- assert timestamp is float32 and index columns have exact scalar types;
- mutate each emitted float column with NaN and infinity and require rejection;
- assert JSON serialization rejects nonfinite statistics;
- recompute episode and global statistics from retained rows;
- distinguish retained-only statistics from the legacy all-source-row result;
- require `count: [n]` in every episode block and
  `count: [total_frames]` in every global block;
- require canonical byte equality plus independent semantic validation of
  `stats.json` and `stats_psi0.json`.

### Media tests

- require a missing source video to fail before staging;
- select copy-all only for homogeneous identity selections at matching FPS;
- select transcode-all for any skip or downsampling;
- select transcode-all for requested/source FPS mismatch;
- use a two-episode fixture with heterogeneous codec, FPS, and dimensions and require two identical canonical outputs;
- reject source video/Parquet frame-count mismatch;
- reject any output frame-count or profile mismatch;
- assert `info.json` is generated from and equals the verified dataset output profile;
- assert copy-all source and output hashes match.

### Provenance and publication tests

- accept committed script bytes and reject one-byte dirty changes;
- validate every provenance field and digest;
- prove that preflight failures create no output or staging path;
- use real processes to prove lock contention creates no staging path and that
  a holder terminated with `_exit` releases the stable lock for a new process;
- inject failures during Parquet, video, metadata, and validation phases and
  require a `failed` staging-status record with no canonical output;
- reject an existing final destination;
- prove every manifest-covered file and both reserved files are fsynced before
  bottom-up directory fsync, durable complete state, rename, and parent fsync;
- prove exact manifest membership and reject missing, extra, symlinked,
  hard-linked, mutated, or self-referential entries;
- prove successful no-replace atomic publication through injected filesystem
  adapters;
- inject post-publication verification failures and require an immutable
  `PUBLISHED_UNCERTIFIED` dataset plus a durable sibling `FAIL` root;
- prove a new certification UUID never overwrites a prior failed or incomplete
  evidence root.

### Integration certification tests

- regenerate a multi-episode synthetic dataset and run the full validator;
- run the exact official raw episode into the fresh `-v2` output only after converter commit;
- run the exact pinned PSI0 offline loader through `dataset[i]` for every index,
  assert the visited sequence is exactly `0 .. total_frames-1`, and preserve its
  certificate.

## Files in implementation scope

- `scripts/postprocess_psi0.py`
- `tests/test_postprocess_psi0.py`
- focused new converter test modules if splitting them improves isolation
- a focused offline certification script or module if keeping it outside the converter makes the validation boundary clearer
- generated certification evidence under the fresh output or its sibling evidence directory

No simulator, robot-control, PC2 bridge, dedicated-runtime, PSI0 training, or third-person-camera source is modified by this work.

## Execution sequence and gates

1. Write RED tests for preflight, canonical selection, schemas, finite values, statistics, media profiles, provenance, and atomic publication.
2. Implement the smallest converter and validator changes that satisfy them.
3. Run focused and relevant repository static/unit checks.
4. Commit the converter and tests.
5. Verify executed converter bytes match that commit's blob.
6. Regenerate from the fixed absolute read-only input into the fresh `-v2` destination.
7. Certify Parquet, media, metadata, statistics, provenance, publication, and the pinned PSI0 offline loader.
8. Preserve hashes and the certification report.
9. Stop. Do not start training automatically.

A later H100 training smoke requires a fresh ownership check for the requested GPU immediately before launch. Existing evidence showed foreign GPU-6 processes, so GPU 6 is not presumed available. A foreign or ambiguous workload is a hard stop; this work never terminates it.

## Acceptance criteria

The milestone is complete only when:

- converter implementation starts from `91daf5d` in the isolated branch;
- focused tests and repository checks pass;
- the converter and tests are committed before regeneration;
- the raw input and existing processed output remain byte-for-byte untouched;
- only the fresh `processed-psi0-bendpick-l0-v2` canonical destination is published;
- every episode satisfies the exact row, schema, finite-value, media, and provenance contracts;
- task/episode metadata is completely resolved before staging and remains
  consistent with every emitted record;
- global metadata and statistics, including every required count, are
  recomputable from published artifacts;
- the immutable canonical tree exactly matches its payload manifest and two
  reserved files;
- the exact pinned PSI0 offline loader accepts the complete output;
- the loader retrieves and validates every dataset index exactly once;
- the certificate records exact source, converter, output, PSI0, and environment identities;
- one exclusively created sibling evidence root records a terminal `PASS`
  bound to the exact manifest digest;
- no conversion failure leaves a partial canonical output;
- no simulation, training, inference server, H100 workload, or real-robot process is started.
