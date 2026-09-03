# PSI0 Converter V2 Certification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Use superpowers:test-driven-development for every code-changing task and superpowers:verification-before-completion before each success claim.

**Goal:** Make scripts/postprocess_psi0.py publish one immutable, internally consistent PSI0 dataset and certify every published sample through PSI0's production offline loader.

**Architecture:** Keep the reviewed 32-D state and 36-D action mappings in the existing converter. Refactor the script into explicit attestation, preflight, retained-row, media, generation, validation, durability, and certification boundaries. Resolve all source inputs before staging, drive all outputs from one retained-index array and one dataset-wide media decision, publish through an fsynced no-replace rename, and write PSI0 evidence only into a manifest-bound sibling directory.

**Tech Stack:** Python 3.10, NumPy, PyArrow/Parquet, FFmpeg/FFprobe, POSIX flock/fsync, Linux renameat2, pytest, Ruff, LeRobot, PSI0.

---

## Scope and hard stops

Implementation base and approved design:

    repository: /home/jihun/work/SIMPLE/.worktrees/psi0-converter-v2
    branch: fix/psi0-converter-v2
    base: 91daf5d3a0994774c6996843326c615ff17119e6
    approved design: b7aa36273c47ead2d9a002336722a781d70896f2
    design file: docs/superpowers/specs/2026-09-01-psi0-converter-v2-certification-design.md

This plan stops after converter implementation, official conversion, and offline-loader certification. It must not launch SIMPLE simulation, training, an inference server, Docker, SSH, H100 work, or a robot process. GPU 6 is not presumed available and is irrelevant to this milestone.

The converter implementation commit must exist before official conversion. No uncommitted converter is allowed to generate official output.

Every multi-command shell block in this plan begins with `set -euo pipefail`. Each block is a fresh shell: it must redeclare and validate every path or identity it consumes. No block inherits variables from an earlier block.

Implementation is additionally gated by the annotated tag `psi0-converter-v2-plan-approved`. That tag is created only after plan approval and must point exactly to the reviewed plan commit. This avoids embedding a commit hash in the commit that defines it.

## Planned files

Modify:

- pyproject.toml
- scripts/postprocess_psi0.py
- tests/test_postprocess_psi0.py

Create:

- scripts/certify_psi0_dataset.py
- tests/psi0_converter_fixtures.py
- tests/test_postprocess_psi0_preflight.py
- tests/test_postprocess_psi0_rows.py
- tests/test_postprocess_psi0_media.py
- tests/test_postprocess_psi0_publication.py
- tests/test_certify_psi0_dataset.py
- tests/test_postprocess_psi0_e2e.py

Do not add a converter implementation module outside scripts/postprocess_psi0.py. The approved provenance contract binds that executed script to its committed blob.

## Stable implementation interfaces

Later tasks use these names. Do not rename or duplicate them:

    @dataclass(frozen=True)
    class ConverterIdentity:
        commit: str
        script_sha256: str

    @dataclass(frozen=True)
    class MediaIdentity:
        codec_name: str
        pixel_format: str
        width: int
        height: int
        average_frame_rate: str
        nominal_frame_rate: str
        duration: str
        frame_count: int
        audio_streams: tuple[dict[str, object], ...]

    @dataclass(frozen=True)
    class MediaProfile:
        codec_name: str
        pixel_format: str
        width: int
        height: int
        average_frame_rate: str
        nominal_frame_rate: str
        audio_streams: tuple[dict[str, object], ...]

    @dataclass(frozen=True)
    class EpisodePlan:
        source_root: Path
        source_episode_index: int
        output_episode_index: int
        parquet_path: Path
        video_path: Path
        parquet_sha256: str
        video_sha256: str
        frame_count: int
        retained_indices: np.ndarray
        source_task_index: int
        output_task_index: int
        task_text: str
        environment_config: str
        source_media: MediaIdentity

    @dataclass(frozen=True)
    class ConversionPlan:
        output_path: Path
        output_fps: Fraction
        skip: int
        downsample: int
        chunks_size: int
        video_key: str
        media_mode: Literal["copy_all", "transcode_all"]
        output_media: MediaProfile
        tasks: tuple[dict[str, object], ...]
        episodes: tuple[EpisodePlan, ...]
        converter: ConverterIdentity

    @dataclass(frozen=True)
    class PublicationResult:
        dataset_root: Path
        manifest_sha256: str
        complete_status_sha256: str
        state: Literal["published", "publication_uncertain"]

    class PublicationUncertainError(RuntimeError):
        def __init__(self, publication: PublicationResult) -> None:
            super().__init__("publication rename completed but parent durability is uncertain")
            self.publication = publication

    class PublishedBoundaryError(RuntimeError):
        def __init__(self, publication: PublicationResult) -> None:
            super().__init__("failure after durable publication")
            self.publication = publication

One canonical_json_bytes() implementation encodes all JSON with sorted keys, compact separators, UTF-8, newline termination, and allow_nan=False.

---

### Task 0: Re-establish the approved baseline and guardrails

**Files:**

- Read: docs/superpowers/specs/2026-09-01-psi0-converter-v2-certification-design.md
- Read: scripts/postprocess_psi0.py
- Read: tests/test_postprocess_psi0.py

- [ ] Verify branch ancestry and the tracked starting state.

Run:

    set -euo pipefail
    cd /home/jihun/work/SIMPLE/.worktrees/psi0-converter-v2
    PLAN=docs/superpowers/plans/2026-09-03-psi0-converter-v2-certification.md
    BASE=b7aa36273c47ead2d9a002336722a781d70896f2
    test "$(git cat-file -t refs/tags/psi0-converter-v2-plan-approved)" = "tag"
    REVIEWED=$(git rev-parse 'refs/tags/psi0-converter-v2-plan-approved^{commit}')
    test "$(git branch --show-current)" = "fix/psi0-converter-v2"
    test "$(git rev-parse HEAD)" = "$REVIEWED"
    test "$(git rev-parse HEAD^)" = "$BASE"
    test "$(git diff --name-only "$BASE"..HEAD)" = "$PLAN"
    test "$(git show -s --format=%P HEAD)" = "$BASE"
    test -z "$(git status --porcelain --untracked-files=no)"
    git diff --quiet
    git diff --cached --quiet
    git diff --check "$BASE"..HEAD

Expected: only the reviewed plan commit may follow b7aa362; no converter or test implementation is changed.

- [ ] Record the focused baseline without changing files.

Run:

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_postprocess_psi0.py

Expected: 3 passed.

- [ ] Record local tool identities for later comparison.

Run:

    set -euo pipefail
    /home/jihun/work/SIMPLE/.venv/bin/python --version
    /home/jihun/work/SIMPLE/.venv/bin/python -c 'import av,datasets,numpy,pyarrow,torch,torchvision; print(av.__version__,datasets.__version__,numpy.__version__,pyarrow.__version__,torch.__version__,torchvision.__version__)'
    ffmpeg -version | head -1
    ffprobe -version | head -1

Expected: all imports and media tools succeed. This is diagnostic, not certification.

---

### Task 1: Add canonical encoders, source attestation, selection, schema, and statistics primitives

**Files:**

- Modify: scripts/postprocess_psi0.py
- Modify: tests/test_postprocess_psi0.py
- Create: tests/test_postprocess_psi0_rows.py

- [ ] Write RED tests for dirty converter bytes, non-divisible selection, zero retained rows, exact Arrow types, finiteness, exact indices, and retained-only counts.

Create tests/test_postprocess_psi0_rows.py with these literal tests:

    import json
    from fractions import Fraction

    import numpy as np
    import pyarrow as pa
    import pytest

    from scripts import postprocess_psi0 as converter


    def test_retained_indices_are_the_only_cardinality_source():
        selected = converter.make_retained_indices(214, skip=60, downsample=4)
        np.testing.assert_array_equal(selected, np.arange(60, 214, 4, dtype=np.int64))
        assert len(selected) == 39


    @pytest.mark.parametrize("frame_count", [59, 60])
    def test_zero_retained_rows_fail(frame_count):
        with pytest.raises(ValueError, match="frame_count must be greater than skip"):
            converter.make_retained_indices(frame_count, skip=60, downsample=1)


    def test_output_schema_is_exact():
        schema = converter.output_schema()
        expected = {
            "states": pa.list_(pa.float32(), 32),
            "action": pa.list_(pa.float32(), 36),
            "observation.hand_joints": pa.list_(pa.float32(), 14),
            "observation.arm_joints": pa.list_(pa.float32(), 14),
            "observation.leg_joints": pa.list_(pa.float32(), 15),
            "observation.prev_torso_rpy": pa.list_(pa.float32(), 3),
            "observation.prev_height": pa.list_(pa.float32(), 1),
            "timestamp": pa.float32(),
            "frame_index": pa.int64(),
            "episode_index": pa.int64(),
            "index": pa.int64(),
            "task_index": pa.int64(),
            "next.done": pa.bool_(),
        }
        assert schema.names == list(expected)
        assert {field.name: field.type for field in schema} == expected


    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_finite_array_rejects_nonfinite_values(bad):
        values = np.zeros((2, 32), dtype=np.float32)
        values[1, 3] = bad
        with pytest.raises(ValueError, match="states contains nonfinite values"):
            converter.require_finite("states", values)


    def test_stats_are_retained_only_and_include_count():
        source = np.arange(8, dtype=np.float32).reshape(4, 2)
        retained = source[[1, 3]]
        block = converter.stats_block(retained)
        assert block["count"] == [2]
        np.testing.assert_allclose(block["mean"], [4.0, 5.0])
        assert block["mean"] != converter.stats_block(source)["mean"]
        json.dumps(block, allow_nan=False)


    def test_row_table_uses_exact_indices_timestamps_and_terminal_flag():
        vectors = {
            "states": np.zeros((3, 32), np.float32),
            "action": np.zeros((3, 36), np.float32),
            "observation.hand_joints": np.zeros((3, 14), np.float32),
            "observation.arm_joints": np.zeros((3, 14), np.float32),
            "observation.leg_joints": np.zeros((3, 15), np.float32),
            "observation.prev_torso_rpy": np.zeros((3, 3), np.float32),
            "observation.prev_height": np.full((3, 1), 0.74, np.float32),
        }
        table = converter.build_output_table(
            vectors=vectors,
            output_episode_index=2,
            global_offset=5,
            output_task_index=1,
            output_fps=Fraction(50, 1),
        )
        assert table.schema == converter.output_schema()
        assert table["frame_index"].to_pylist() == [0, 1, 2]
        assert table["episode_index"].to_pylist() == [2, 2, 2]
        assert table["index"].to_pylist() == [5, 6, 7]
        assert table["task_index"].to_pylist() == [1, 1, 1]
        assert table["next.done"].to_pylist() == [False, False, True]
        assert table["timestamp"].type == pa.float32()
        np.testing.assert_array_equal(
            np.asarray(table["timestamp"]),
            np.asarray([0.0, 0.02, 0.04], dtype=np.float32),
        )

Replace the legacy provenance test in tests/test_postprocess_psi0.py with a test that creates a temporary Git repository, commits scripts/postprocess_psi0.py, confirms resolve_converter_identity() succeeds, changes one byte without committing, and confirms it raises "executed converter differs from its recorded Git blob". Use real git subprocesses; do not mock away blob comparison.

- [ ] Run RED.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_postprocess_psi0.py tests/test_postprocess_psi0_rows.py

Expected: missing-function and legacy schema/statistics failures.

- [ ] Implement the primitives in scripts/postprocess_psi0.py.

Add these complete functions:

    def canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
        separators = None if pretty else (",", ":")
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=separators,
            sort_keys=True,
        )
        return (text + "\n").encode("utf-8")


    def sha256_bytes(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()


    def resolve_converter_identity(
        repository_root: Path, script_path: Path
    ) -> ConverterIdentity:
        repository_root = repository_root.resolve(strict=True)
        script_path = script_path.resolve(strict=True)
        expected = repository_root / "scripts" / "postprocess_psi0.py"
        if script_path != expected:
            raise RuntimeError(f"unexpected converter path: {script_path}")
        commit = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", "scripts/postprocess_psi0.py"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise RuntimeError("could not resolve converter source commit")
        committed = subprocess.run(
            ["git", "show", f"{commit}:scripts/postprocess_psi0.py"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        executed = script_path.read_bytes()
        if executed != committed:
            raise RuntimeError("executed converter differs from its recorded Git blob")
        return ConverterIdentity(commit=commit, script_sha256=sha256_bytes(executed))


    def make_retained_indices(
        frame_count: int, skip: int, downsample: int
    ) -> np.ndarray:
        if skip < 0:
            raise ValueError("skip must be nonnegative")
        if downsample <= 0:
            raise ValueError("downsample must be positive")
        if frame_count <= skip:
            raise ValueError("frame_count must be greater than skip")
        result = np.arange(skip, frame_count, downsample, dtype=np.int64)
        if result.size == 0 or not np.all(result[1:] > result[:-1]):
            raise ValueError("retained indices must be nonempty and strictly increasing")
        return result


    def require_finite(name: str, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values)
        if not np.issubdtype(result.dtype, np.floating):
            raise ValueError(f"{name} must be floating point")
        if not np.isfinite(result).all():
            raise ValueError(f"{name} contains nonfinite values")
        return result


    def output_schema() -> pa.Schema:
        return pa.schema(
            [
                pa.field("states", pa.list_(pa.float32(), 32)),
                pa.field("action", pa.list_(pa.float32(), 36)),
                pa.field("observation.hand_joints", pa.list_(pa.float32(), 14)),
                pa.field("observation.arm_joints", pa.list_(pa.float32(), 14)),
                pa.field("observation.leg_joints", pa.list_(pa.float32(), 15)),
                pa.field("observation.prev_torso_rpy", pa.list_(pa.float32(), 3)),
                pa.field("observation.prev_height", pa.list_(pa.float32(), 1)),
                pa.field("timestamp", pa.float32()),
                pa.field("frame_index", pa.int64()),
                pa.field("episode_index", pa.int64()),
                pa.field("index", pa.int64()),
                pa.field("task_index", pa.int64()),
                pa.field("next.done", pa.bool_()),
            ]
        )


    def stats_block(values: np.ndarray) -> dict[str, list[float] | list[int]]:
        array = require_finite(
            "statistics input", np.asarray(values, dtype=np.float32)
        )
        if array.shape[0] == 0:
            raise ValueError("statistics input is empty")
        if array.ndim == 1:
            array = array[:, None]
        block = {
            "mean": array.mean(0, dtype=np.float64).astype(np.float32).tolist(),
            "std": array.std(0, dtype=np.float64).astype(np.float32).tolist(),
            "min": array.min(0).astype(np.float32).tolist(),
            "max": array.max(0).astype(np.float32).tolist(),
            "q01": np.quantile(array, 0.01, axis=0).astype(np.float32).tolist(),
            "q99": np.quantile(array, 0.99, axis=0).astype(np.float32).tolist(),
            "count": [int(array.shape[0])],
        }
        canonical_json_bytes(block)
        return block

Implement build_output_table() by converting each vector to contiguous np.float32, checking exact (n, width), calling require_finite(), constructing fixed lists with pa.FixedSizeListArray.from_arrays(), building all scalar columns from n, and passing schema=output_schema() to pa.Table.from_arrays(). It must not accept or read a source index.

- [ ] Run GREEN checks.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_postprocess_psi0.py tests/test_postprocess_psi0_rows.py
    ruff check scripts/postprocess_psi0.py tests/test_postprocess_psi0.py tests/test_postprocess_psi0_rows.py
    ruff format --check scripts/postprocess_psi0.py tests/test_postprocess_psi0.py tests/test_postprocess_psi0_rows.py

- [ ] Commit Task 1.

    set -euo pipefail
    git add scripts/postprocess_psi0.py tests/test_postprocess_psi0.py tests/test_postprocess_psi0_rows.py
    git commit -m "refactor: define PSI0 conversion row contracts"

---

### Task 2: Build source fixtures and fail-closed preflight

**Files:**

- Create: tests/psi0_converter_fixtures.py
- Create: tests/test_postprocess_psi0_preflight.py
- Modify: scripts/postprocess_psi0.py

- [ ] Add a fixture that writes one valid SIMPLE episode with fixed-size float32 Parquet columns, exact task/episode metadata, and a real small H.264 video made with ffmpeg testsrc2.

The helper signature is:

    def make_source_episode(
        root: Path,
        *,
        episode_index: int = 0,
        frames: int = 8,
        fps: int = 4,
        size: str = "64x48",
        video_codec: str = "libx264",
        task_index: int = 0,
        task_text: str = "test task",
    ) -> Path:

It must write:

    data/chunk-000/episode_NNNNNN.parquet
    videos/chunk-000/observation.rgb_head_stereo_left/episode_NNNNNN.mp4
    meta/info.json
    meta/tasks.jsonl
    meta/episodes.jsonl

Use pa.FixedSizeListArray.from_arrays for widths 43, 9, and 43. Use observation.joint_qpos, observation.amo_policy_command, observation.amo_policy_target_yaw, observation.amo_policy_turning_flag, action, and task_index with the exact approved source types.

- [ ] Add RED one-mutation-at-a-time preflight tests.

The test matrix must independently reject, before output or staging creation:

| Group | Cases |
|---|---|
| Global | skip<0; downsample<=0; FPS zero/negative/nonfinite; chunks-size<=0; total-episodes<=0; missing root; missing metadata; existing destination |
| Parquet | each required column missing; scalar/list mismatch; vector width mismatch; dtype mismatch; unequal row count; task_index non-int64/nonconstant |
| Selection | frame_count<skip; frame_count==skip |
| Tasks | duplicate/negative/noninteger index; empty/nonstring task; missing lookup |
| Episodes | duplicate/negative/noninteger index; nonpositive/wrong length; empty/nonstring tasks; invalid/nonstring environment_config; task mismatch; missing index lookup |
| Video | missing/duplicate candidate; no video stream; ambiguous/nonfinite/zero FPS; frame-count mismatch; malformed audio identity |

Use this assertion helper in every case:

    def assert_preflight_rejected_without_output(call, output: Path) -> None:
        parent = output.parent
        before = sorted(
            (path.relative_to(parent), path.lstat().st_mode, path.lstat().st_size)
            for path in parent.rglob("*")
        )
        with pytest.raises((ValueError, RuntimeError, FileExistsError)):
            call()
        after = sorted(
            (path.relative_to(parent), path.lstat().st_mode, path.lstat().st_size)
            for path in parent.rglob("*")
        )
        assert after == before
        assert not output.exists()
        assert not list(parent.glob(f".{output.name}.staging-*"))

Add a positive two-root test that orders roots and episodes deterministically, resolves episode metadata by episode_index rather than list position, and remaps task text by first occurrence.

- [ ] Run RED.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_postprocess_psi0_preflight.py

- [ ] Implement parse_rate(), probe_media(), and preflight_conversion() as read-only operations.

Required source schema:

    REQUIRED_SOURCE_TYPES = {
        "observation.joint_qpos": pa.list_(pa.float32(), 43),
        "observation.amo_policy_command": pa.list_(pa.float32(), 9),
        "observation.amo_policy_target_yaw": pa.float32(),
        "observation.amo_policy_turning_flag": pa.float32(),
        "action": pa.list_(pa.float32(), 43),
        "task_index": pa.int64(),
    }

Rate parser:

    def parse_rate(value: str) -> Fraction:
        try:
            result = Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"invalid media rate: {value!r}") from exc
        if result <= 0:
            raise ValueError(f"media rate must be positive: {value!r}")
        return result

probe_media() must invoke ffprobe with -count_frames, -show_streams, -show_format, and JSON output as an argv list with check=True. It parses exactly one video stream, canonicalizes avg_frame_rate and r_frame_rate as reduced Fraction strings, requires positive finite duration, requires positive nb_read_frames (or an equal positive nb_frames when the read counter is unavailable), and captures every audio stream's codec_name, sample_fmt, integer sample_rate, channels, and channel_layout in stream order.

preflight_conversion() must:

1. validate all global arguments;
2. expand/sort roots and data files;
3. load tasks and episodes into duplicate-rejecting keyed mappings;
4. inspect schema with pq.read_schema() and row count with ParquetFile metadata;
5. read only task_index to prove its type, cardinality, and constancy;
6. call make_retained_indices();
7. resolve exactly one video, hash it, and probe it;
8. prove source media frame count equals Parquet rows;
9. resolve task and episode metadata by keys;
10. assign contiguous output episode/task indices;
11. return immutable EpisodePlan records.

It must not call mkdir(), copy, transcode, chmod, rename, or write.

- [ ] Run GREEN and commit.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_postprocess_psi0_preflight.py tests/test_postprocess_psi0_rows.py
    ruff check scripts/postprocess_psi0.py tests/psi0_converter_fixtures.py tests/test_postprocess_psi0_preflight.py
    ruff format --check scripts/postprocess_psi0.py tests/psi0_converter_fixtures.py tests/test_postprocess_psi0_preflight.py
    git add scripts/postprocess_psi0.py tests/psi0_converter_fixtures.py tests/test_postprocess_psi0_preflight.py
    git commit -m "feat: preflight PSI0 conversion inputs"

---

### Task 3: Enforce one dataset-wide media decision

**Files:**

- Create: tests/test_postprocess_psi0_media.py
- Modify: scripts/postprocess_psi0.py

- [ ] Add RED media-decision and real ffmpeg tests.

Cover these cases with actual MediaIdentity/EpisodePlan fixtures:

1. homogeneous, unsampled, matching FPS selects copy_all;
2. nonzero skip selects transcode_all;
3. downsample>1 selects transcode_all;
4. requested/source FPS mismatch selects transcode_all;
5. any codec/pixel-format/dimension/rate/audio mismatch selects transcode_all;
6. two heterogeneous real inputs both produce identical H.264/yuv420p/640x360/requested-FPS/no-audio profiles;
7. copy-all output bytes and SHA equal the source;
8. copy destination collision and transcode publication-boundary collision fail without overwrite or prompt;
9. ffmpeg nonzero exit fails;
10. output frame-count/profile drift fails certification.

Run RED:

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_postprocess_psi0_media.py

- [ ] Implement media_selection_key(), decide_media_mode(), and write_episode_video().

The dataset-homogeneity selection key is exactly:

    def media_selection_key(value: MediaIdentity) -> tuple[object, ...]:
        return (
            value.codec_name,
            value.pixel_format,
            value.width,
            value.height,
            value.average_frame_rate,
            value.nominal_frame_rate,
            value.audio_streams,
        )

    def media_profile(value: MediaIdentity) -> MediaProfile:
        return MediaProfile(
            codec_name=value.codec_name,
            pixel_format=value.pixel_format,
            width=value.width,
            height=value.height,
            average_frame_rate=value.average_frame_rate,
            nominal_frame_rate=value.nominal_frame_rate,
            audio_streams=value.audio_streams,
        )

Copy-all requires skip==0, downsample==1, every selection key identical, both source rates equal requested FPS, and every source frame count equal retained count. Its dataset-wide result is media_profile(first_source). Otherwise every episode is transcoded to a MediaProfile containing H.264, yuv420p, 640x360, requested FPS, and no audio. Duration and frame count remain per-episode MediaIdentity fields and are never placed in the dataset-wide profile.

write_episode_video() must:

- never rely on a separate destination-absence check for correctness;
- use copy_file_exclusive() in copy_all, opening the destination with O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW, streaming from a no-follow source descriptor, fsyncing the destination, and validating source/destination identities and SHA-256;
- otherwise exclusively create a mode-0700 private directory beneath the staged destination parent and make FFmpeg write `artifact.mp4` only inside that directory, never directly to the intended staged pathname;
- invoke FFmpeg with an argv list and never shell=True, passing both -n and -nostdin before the input;
- generate the select expression from the exact retained_indices values;
- use scale=640:360:flags=lanczos and setpts=N/(FPS*TB);
- use the locally supported FFmpeg 4.4 form -vsync cfr, -r requested FPS, -frames:v retained count, -c:v libx264, -pix_fmt yuv420p, and -an;
- probe and validate the private artifact, reopen it no-follow, fsync it, and verify its descriptor identity is unchanged;
- call the shared directory-FD `rename_noreplace(private_artifact, destination)` implementation based on `renameat2(RENAME_NOREPLACE)`, then fsync the staged destination parent;
- remove and fsync away the now-empty private directory only after successful no-replace publication; preserve it as diagnostic staging evidence on failure;
- require copied SHA equality;
- never delete source or failed output.

Implement `rename_noreplace()` in this task so both media publication and final dataset publication use one tested syscall wrapper. It opens and validates the source and destination parent directories, passes validated basenames to libc `renameat2` with `RENAME_NOREPLACE`, distinguishes EEXIST, and fails closed on an unavailable syscall or EXDEV. Task 6 reuses this helper and must not redefine it.

The RED test first executes ffmpeg -version and ffmpeg -encoders, requires major version 4 or newer and the libx264 encoder, then runs the exact production argv. It pre-creates the copy destination with sentinel bytes. For transcoding, an injected `before_media_publish` callback creates a sentinel at the intended destination only after the private artifact has been fully encoded, probed, and fsynced. The no-replace rename must fail with EEXIST, leave the sentinel byte-identical, preserve the validated private artifact, and never read stdin. A successful case proves the private artifact is atomically renamed, the parent is fsynced, and no private directory remains. A future migration to -fps_mode requires a separate compatibility change; do not pass that unsupported option on this host.

- [ ] Run GREEN and commit.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_postprocess_psi0_media.py tests/test_postprocess_psi0_preflight.py
    ruff check scripts/postprocess_psi0.py tests/test_postprocess_psi0_media.py
    ruff format --check scripts/postprocess_psi0.py tests/test_postprocess_psi0_media.py
    git add scripts/postprocess_psi0.py tests/test_postprocess_psi0_media.py
    git commit -m "feat: enforce one PSI0 dataset media profile"


---

### Task 4: Generate retained rows, metadata, statistics, and provenance in staging

**Files:**

- Modify: scripts/postprocess_psi0.py
- Modify: tests/test_postprocess_psi0_rows.py
- Create: tests/test_postprocess_psi0_e2e.py

- [ ] Add RED synthetic staged-generation tests.

Call generate_staged_dataset(plan, staging) and assert before publication:

| Artifact | Exact assertion |
|---|---|
| Parquet | Every column length equals len(retained_indices); schema equals output_schema(); all floats finite |
| Rows | frame_index 0..n-1; global index 0..total_frames-1; exact episode/task indices |
| Time/done | timestamp is float32(frame_index/output_fps); only final row is done |
| Vectors | build_vectors/build_proprio_obs output indexed by the same retained_indices |
| Episode stats | Recomputed from retained Parquet; every block count=[n] |
| Global stats | Recomputed from all final Parquet; every block count=[total_frames] |
| Stats files | stats.json bytes equal stats_psi0.json bytes |
| info.json | state/action shapes [32]/[36]; media fields from plan.output_media |
| Videos | one per episode; count/profile/frame count match plan |
| Provenance | every approved episode/dataset field and digest present |
| JSON | parsing rejects NaN/Infinity via parse_constant |
| Manifest | absent until the publication task |

The generated tree is exactly:

    CONVERSION_STATUS.json
    data/chunk-NNN/episode_NNNNNN.parquet
    videos/chunk-NNN/egocentric/episode_NNNNNN.mp4
    meta/info.json
    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/episodes_stats.jsonl
    meta/stats.json
    meta/stats_psi0.json
    meta/relative_stats.json
    meta/lang_map.json
    meta/modality.json
    meta/conversion_provenance.json

Task 6 adds only meta/conversion_manifest.json before sealing. Episode provenance is embedded in each episodes.jsonl row; dataset provenance is meta/conversion_provenance.json. No log, certificate, temporary, cache, or diagnostic file may remain at manifest time.

Emit task rows as {task_index, task, category:"", description:task}. Emit episode rows with exact keys episode_index, tasks, length, dataset_from_index, dataset_to_index, robot_type, instruction, environment_config, and conversion_provenance. tasks is [output_task_index], instruction is the complete matching emitted task row, and dataset bounds are inclusive. The validator requires exact key sets and proves every Parquet task_index maps to both fields.

Parameterize each emitted float column over np.nan, np.inf, and -np.inf. Mutate the selected value before Arrow construction and require failure with no manifest.

- [ ] Run RED.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_postprocess_psi0_rows.py tests/test_postprocess_psi0_e2e.py

- [ ] Implement staged generation around one selected mapping per episode.

Use this exact data flow:

    table = pq.read_table(episode.parquet_path)
    qpos = np.asarray(table["observation.joint_qpos"].to_pylist(), dtype=np.float32)
    command = np.asarray(
        table["observation.amo_policy_command"].to_pylist(), dtype=np.float32
    )
    target_yaw = np.asarray(
        table["observation.amo_policy_target_yaw"], dtype=np.float32
    )
    turning = np.asarray(
        table["observation.amo_policy_turning_flag"], dtype=np.float32
    )
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    history = np.concatenate([initial_command[None], command[:-1]], axis=0)
    states, actions = build_vectors(
        qpos, command, history, action, target_yaw, turning
    )
    hand, arm, leg, torso, height = build_proprio_obs(qpos, history)
    indices = episode.retained_indices
    selected = {
        "states": states[indices],
        "action": actions[indices],
        "observation.hand_joints": hand[indices],
        "observation.arm_joints": arm[indices],
        "observation.leg_joints": leg[indices],
        "observation.prev_torso_rpy": torso[indices],
        "observation.prev_height": height[indices],
    }

Pass selected to build_output_table(), episode statistics, global accumulators, and provenance cardinality. Never append unsliced states, actions, or source indices.

Write fresh staged canonical files atomically with one implementation:

    def atomic_write_new_bytes(path: Path, payload: bytes, mode: int = 0o644) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            mode,
        )
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path, follow_symlinks=False)
            fsync_directory(path.parent)
        finally:
            try:
                temporary.unlink()
                fsync_directory(path.parent)
            except FileNotFoundError:
                pass
            os.close(fd)

The hard-link publication fails with FileExistsError if the final staged path exists. A crash-surviving temporary is an extra manifest member and makes publication fail closed. Use one byte object for stats.json and stats_psi0.json.

Keep the existing statistics-key surface exact: each episodes_stats.jsonl row contains only action and timestamp; global stats.json/stats_psi0.json contain states, action, timestamp, frame_index, episode_index, index, task_index, and next.done. Every block contains count. Do not silently add auxiliary-vector statistics in this milestone.

Construct the image feature from the verified MediaProfile with this exact key layout:

    {
        "dtype": "video",
        "shape": [profile.height, profile.width, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "has_audio": bool(profile.audio_streams),
            "video.channels": 3,
            "video.codec": profile.codec_name,
            "video.fps": float(Fraction(profile.average_frame_rate)),
            "video.height": profile.height,
            "video.is_depth_map": False,
            "video.pix_fmt": profile.pixel_format,
            "video.width": profile.width,
        },
    }

The validator requires nominal and average rates to equal output_fps even though video_info exposes average FPS as a JSON number.

info.json retains the current LeRobot v2.1 top-level keys with recomputed values: codebase_version, robot_type, total_episodes, total_frames, total_tasks, total_videos, total_chunks, chunks_size, fps, data_path, video_path, and features. data_path and video_path remain the existing chunk templates. Its feature key set is exactly the image feature plus the seven vector columns and six scalar/index columns from output_schema(); state/action shapes are [32]/[36], not [-1].

Per-episode provenance must contain source episode index, source Parquet/video SHA-256, requested video key/FPS, skip/downsample, retained count, source/output media identities and hashes, converter commit, and script SHA-256. Dataset provenance must contain normalized invocation, ordered input roots, media mode/output profile, ordered episode provenance digests, and output-schema digest.

- [ ] Run GREEN and commit.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider \
      tests/test_postprocess_psi0.py \
      tests/test_postprocess_psi0_rows.py \
      tests/test_postprocess_psi0_preflight.py \
      tests/test_postprocess_psi0_media.py \
      tests/test_postprocess_psi0_e2e.py
    ruff check scripts/postprocess_psi0.py tests/psi0_converter_fixtures.py tests/test_postprocess_psi0*.py
    ruff format --check scripts/postprocess_psi0.py tests/psi0_converter_fixtures.py tests/test_postprocess_psi0*.py
    git add scripts/postprocess_psi0.py tests/test_postprocess_psi0_rows.py tests/test_postprocess_psi0_e2e.py
    git commit -m "feat: generate retained PSI0 dataset artifacts"

---

### Task 5: Add the stable conversion lock and exact manifest validator

**Files:**

- Create: tests/test_postprocess_psi0_publication.py
- Modify: scripts/postprocess_psi0.py

- [ ] Add RED real-process lock tests.

Use multiprocessing.get_context("fork") and the production conversion_lock() context manager:

1. a holder acquires; a contender gets BlockingIOError; no staging exists;
2. the holder calls os._exit(71); the parent observes 71; a replacement acquires the same persistent lock file;
3. symlink, directory, wrong mode, and hard-linked lock paths fail before staging;
4. wrong owner is tested only when a privilege-isolated fixture can safely create it.

- [ ] Add RED manifest validation tests.

Build a minimal valid staged tree through the staged generator. Independently mutate and reject:

    missing manifest member
    extra regular member
    symlink
    FIFO/device/socket
    hard link
    wrong byte size
    wrong SHA-256
    manifest self-entry
    status entry
    unsorted or duplicate path
    absolute path
    "." or ".." component
    uppercase/malformed digest
    file outside data/videos/meta
    missing or extra reserved path

Positive assertion:

    enumerated_regular_files == (
        set(manifest_entry_paths)
        | {"meta/conversion_manifest.json", "CONVERSION_STATUS.json"}
    )

- [ ] Run RED.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider \
      tests/test_postprocess_psi0_publication.py -k 'lock or manifest'

- [ ] Implement conversion_lock().

    @contextlib.contextmanager
    def conversion_lock(output: Path):
        lock_path = output.parent / f".{output.name}.conversion.lock"
        fd = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError("conversion lock is not a regular file")
            if metadata.st_uid != os.getuid():
                raise RuntimeError("conversion lock has wrong owner")
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                raise RuntimeError("conversion lock has wrong mode")
            if metadata.st_nlink != 1:
                raise RuntimeError("conversion lock has wrong link count")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield fd
        finally:
            os.close(fd)

The stable file may persist; its contents never represent ownership.

- [ ] Implement manifest enumeration and validation.

build_payload_manifest(staging) must use lstat/no-follow traversal, prohibit symlinks and non-regular members, reject st_nlink != 1, exclude only meta/conversion_manifest.json and CONVERSION_STATUS.json, require all members beneath data, videos, or meta, sort POSIX paths by UTF-8 bytes, and include path/size/lowercase SHA-256.

validate_payload_manifest() must independently re-enumerate and rehash the tree. It must not accept a caller-supplied file map. The manifest itself is canonical JSON and cannot hash itself.

- [ ] Run GREEN and commit.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider \
      tests/test_postprocess_psi0_publication.py -k 'lock or manifest'
    ruff check scripts/postprocess_psi0.py tests/test_postprocess_psi0_publication.py
    ruff format --check scripts/postprocess_psi0.py tests/test_postprocess_psi0_publication.py
    git add scripts/postprocess_psi0.py tests/test_postprocess_psi0_publication.py
    git commit -m "feat: lock and manifest PSI0 conversion output"

---

### Task 6: Implement and crash-test the durable publication state machine

**Files:**

- Modify: scripts/postprocess_psi0.py
- Modify: tests/test_postprocess_psi0_publication.py

- [ ] Add an injectable production filesystem adapter and RED event-order tests.

The adapter interface is concrete:

    class PublicationFilesystem:
        def fsync_file(self, path: Path) -> None:
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            fd = os.open(path, flags)
            try:
                metadata = os.fstat(fd)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise RuntimeError(f"unsafe file for fsync: {path}")
                os.fsync(fd)
            finally:
                os.close(fd)

        def fsync_directory(self, path: Path) -> None:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

        def chmod(self, path: Path, mode: int) -> None:
            fd = os.open(
                path,
                os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            try:
                before = os.fstat(fd)
                if not (stat.S_ISREG(before.st_mode) or stat.S_ISDIR(before.st_mode)):
                    raise RuntimeError(f"unsafe chmod target: {path}")
                if before.st_nlink != 1 and stat.S_ISREG(before.st_mode):
                    raise RuntimeError(f"multiply linked chmod target: {path}")
                os.fchmod(fd, mode)
                after = os.fstat(fd)
                if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
                    raise RuntimeError(f"chmod target identity changed: {path}")
                reopened = os.open(
                    path,
                    os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
                )
                try:
                    visible = os.fstat(reopened)
                    if (visible.st_dev, visible.st_ino) != (before.st_dev, before.st_ino):
                        raise RuntimeError(f"chmod pathname was replaced: {path}")
                    if stat.S_IMODE(visible.st_mode) != mode:
                        raise RuntimeError(f"chmod mode did not persist: {path}")
                finally:
                    os.close(reopened)
            finally:
                os.close(fd)

        def atomic_write_new(self, path: Path, payload: bytes, mode: int) -> None:
            atomic_write_new_bytes(path, payload, mode)

        def atomic_replace_status(
            self,
            path: Path,
            payload: bytes,
            mode: int,
            fault: Callable[[str], None],
        ) -> None:
            atomic_replace_status(path, payload, mode, fault)

        def rename_noreplace(self, source: Path, destination: Path) -> None:
            rename_noreplace(source, destination)

Task 6 reuses Task 3's `rename_noreplace()` for final dataset publication. `atomic_replace_status()` alone may replace the known regular CONVERSION_STATUS.json and must expose callbacks after temporary-file creation, immediately before temporary-file fsync, after temporary-file fsync, and after status rename. It then fsyncs the staging root and returns only if that fsync succeeds. On unavailable syscall, EXDEV, or any non-success other than the separately reported EEXIST collision, fail closed. Never fall back to overwrite-capable publication rename.

Add a host-backed adapter test that asserts `os.chmod in os.supports_follow_symlinks` is false on this interpreter, then seals a regular file and directory through PublicationFilesystem.chmod(). It must prove fchmod receives the opened descriptor, reject a symlink/FIFO, and use an injected post-open pathname swap to prove the final no-follow reopen detects identity drift.

- [ ] Instrument every boundary with fault(point).

Exact point names:

    after_payload_close
    after_staged_validation
    after_manifest_file_fsync
    after_manifest_rename
    after_meta_fsync
    after_each_payload_fsync:<relative-path>
    after_each_precomplete_directory_fsync:<relative-path>
    after_precomplete_revalidation
    before_complete_temp_fsync
    after_complete_temp_fsync
    after_complete_status_rename
    after_complete_root_fsync
    after_each_chmod:<relative-path>
    after_each_final_file_fsync:<relative-path>
    after_each_final_directory_fsync:<relative-path>
    after_final_revalidation
    before_publication_rename
    after_publication_rename
    after_destination_parent_fsync

For every point, tests assert state on disk rather than trusting an in-memory label.

- [ ] Test all failure classes.

| Boundary | Required classification/state |
|---|---|
| Before complete temp write | pre_completion_failed; durable failed; writable staging preserved |
| Complete temp fsync/status rename/root fsync uncertainty | completion_uncertain_unpublished; never assumed complete |
| `after_complete_root_fsync`, after the caller first sets durable_complete | complete_unpublished; complete preserved; no failed rewrite |
| After rename, before parent fsync | publication_uncertain; immutable visible tree |
| After parent fsync | published |

Also assert:

- every manifest file, manifest, and in_progress status is fsynced before bottom-up pre-completion directory fsync;
- pre-completion revalidation precedes complete;
- complete temp is fsynced, status renamed, and staging root fsynced in that order;
- every canonical file is 0444 and directory 0555;
- every file is fsynced after chmod, then directories bottom-up;
- final revalidation precedes no-replace rename;
- destination collision preserves both paths;
- no code rewrites complete to failed.

- [ ] Add real fork/os._exit tests at each complete-status and publication boundary.

A child runs production publication and exits inside fault(). The parent restarts inspection under conversion_lock(), infers the allowed state from disk, and never removes/adopts preserved staging.

Add two dedicated `after_destination_parent_fsync` cases:

- `test_after_destination_parent_fsync_exception_is_durably_published` calls `publish_staged_dataset()` directly, raises a normal exception from the hook, and requires `PublishedBoundaryError` carrying `state="published"`, a visible canonical root with valid manifest/complete status/final modes, and no `PublicationUncertainError`.
- `test_after_destination_parent_fsync_exit_restarts_as_published` calls `os._exit(73)` from that hook in a real child. After `waitpid`, the parent acquires the stable conversion lock, no-follow reopens the canonical root, validates its manifest and complete status, proves the staging pathname is absent, and classifies it as published. The restart path must not create certification evidence.

- [ ] Run RED.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider \
      tests/test_postprocess_psi0_publication.py -k 'durable or crash or publication'

- [ ] Implement publish_staged_dataset() in the exact 13 approved steps.

Keep four explicit booleans:

    completion_write_started = False
    durable_complete = False
    publication_renamed = False
    publication_durable = False

Set `completion_write_started` immediately before creating/fsyncing the complete temporary file. `atomic_replace_status()` performs the status replacement and staging-root fsync but does not call `fault("after_complete_root_fsync")`. Immediately after that helper returns, the caller must execute these statements in this order:

    durable_complete = True
    fault("after_complete_root_fsync")

Caught failures write failed only when both booleans are false. A failure with completion_write_started and not durable_complete is completion_uncertain_unpublished. An exception from `after_complete_root_fsync`, or any later failure before rename, is complete_unpublished. A failure after rename and before parent fsync is publication_uncertain. Add an event-order test and fork/_exit restart test proving the Boolean transition precedes the named hook and that this hook can never produce completion_uncertain_unpublished.

`publish_staged_dataset()` may return only a `PublicationResult(state="published")`. If the no-replace rename succeeds but the destination-parent fsync raises or is interrupted, construct `PublicationResult(state="publication_uncertain")` and raise `PublicationUncertainError` carrying it. Do not return that result through the success path. The original conversion command must therefore exit nonzero without invoking a certifier. Only a later standalone certifier, while holding `conversion_lock()`, may revalidate the visible immutable tree and establish parent durability.

Final publication uses this exact ordering after final validation:

    filesystem.rename_noreplace(staging, output)
    publication_renamed = True
    try:
        fault("after_publication_rename")
        filesystem.fsync_directory(output.parent)
    except BaseException as exc:
        uncertain = PublicationResult(
            dataset_root=output,
            manifest_sha256=manifest_sha256,
            complete_status_sha256=complete_status_sha256,
            state="publication_uncertain",
        )
        raise PublicationUncertainError(uncertain) from exc
    publication_durable = True
    result = PublicationResult(
        dataset_root=output,
        manifest_sha256=manifest_sha256,
        complete_status_sha256=complete_status_sha256,
        state="published",
    )
    try:
        fault("after_destination_parent_fsync")
    except BaseException as exc:
        raise PublishedBoundaryError(result) from exc
    return result

An exception from `after_publication_rename` is also wrapped as `PublicationUncertainError` because it precedes the parent fsync. Once `publication_durable` is true, no branch may construct `PublicationUncertainError`, call `report_conversion_failure()`, or rewrite any canonical path. Add event-order assertions proving parent fsync precedes `publication_durable`, result construction precedes `after_destination_parent_fsync`, and the normal return follows that hook.

- [ ] Run GREEN and commit.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_postprocess_psi0_publication.py
    ruff check scripts/postprocess_psi0.py tests/test_postprocess_psi0_publication.py
    ruff format --check scripts/postprocess_psi0.py tests/test_postprocess_psi0_publication.py
    git add scripts/postprocess_psi0.py tests/test_postprocess_psi0_publication.py
    git commit -m "feat: publish PSI0 datasets durably"

---

### Task 7: Add independent staged and published dataset validation

**Files:**

- Create: scripts/certify_psi0_dataset.py
- Create: tests/test_certify_psi0_dataset.py
- Modify: scripts/postprocess_psi0.py

- [ ] Add RED semantic-validator tests.

Build a valid two-episode dataset through production generation. Independently mutate every acceptance category:

| Category | Mutations |
|---|---|
| Arrow | name, order, vector width, value type, scalar type, unequal count |
| Numeric | NaN/positive infinity/negative infinity in each float column |
| Indices | local/global gap, duplicate, wrong episode/task, timestamp, done |
| Statistics | each statistic value, each count, wrong episode/global scope, unequal stats file bytes |
| Video | missing/extra, frame count, codec, pixel format, dimensions, FPS, audio |
| Metadata | totals, chunks, paths, vector shapes, video_info, task/episode cardinality |
| Provenance | every source/output hash and identity, converter commit/blob digest, episode digest order |
| Tree | membership, hash, size, type, link count, mode, reserved paths |

Each mutation must fail independently with a stable error code.

- [ ] Add RED sibling-evidence lifecycle tests.

Assert:

    evidence path == parent / f".{name}.certification-{manifest_sha}-{uuid}/"
    evidence mkdir is exclusive and mode 0700
    no evidence file is ever below dataset_root
    PASS.json and FAIL.json are mutually exclusive terminal records
    incomplete evidence is preserved and never reused
    retry uses a different UUID
    post-publication failure leaves dataset bytes, modes, and inodes unchanged
    publication_uncertain is fully revalidated under conversion_lock
    every evidence file and directory plus parent is fsynced before lock release

- [ ] Run RED.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider \
      -m 'not production_loader' \
      tests/test_certify_psi0_dataset.py -k 'semantic or evidence'

- [ ] Implement reusable validator and evidence writer.

scripts/certify_psi0_dataset.py must expose:

    def validate_dataset(
        dataset_root: Path,
        *,
        expected: DatasetExpectations | None,
        require_final_modes: bool,
    ) -> DatasetValidationResult:

    def certify_published_dataset(
        publication: PublicationResult,
        *,
        psi0_root: Path,
        psi0_commit: str,
        python: Path,
        certificate_uuid: uuid.UUID | None = None,
    ) -> Path:

Staged validation receives expectations from immutable preflight. Published validation derives expected facts from canonical metadata/provenance, reopens the source paths recorded there, and independently hashes them.

`certify_published_dataset()` rejects `publication.state != "published"` before creating an evidence root. The standalone CLI path for a previously `publication_uncertain` canonical pathname must acquire `conversion_lock()`, validate the immutable root, fsync its destination parent, and only then construct a `PublicationResult(state="published")` for certification. The original converter process never calls this recovery path.

The evidence root is exclusively created, all evidence is canonical and fsynced, terminal PASS/FAIL is written last, files are sealed 0444, directories 0555, and the sibling parent is fsynced. A post-publication error returns nonzero as PUBLISHED_UNCERTIFIED and never mutates the dataset. The CLI also implements --print-tree-digest as a read-only manifest/status/tree validation that prints only the canonical digest and creates no evidence root.

Use this exact evidence layout; files that cannot be produced after a failure are absent and the terminal FAIL lists only the files that were durably completed:

    certificate-request.json
    dataset-root-identity.json
    dataset-validation.json
    source-validation.json
    psi0-environment.json
    psi0-loader-command.json
    psi0-loader-cache.json
    psi0-loader-result.json
    PASS.json or FAIL.json

dataset-root-identity.json records absolute path, st_dev, st_ino, st_uid, st_gid, st_mode, st_nlink, manifest SHA-256, and complete-status SHA-256 from a no-follow descriptor. PASS.json/FAIL.json contains schema_version=1, certificate UUID, terminal verdict, those three identities, and sorted path/size/SHA-256 entries for every preceding evidence file. The terminal file is excluded from its own list. FAIL.json additionally records a stable stage/error code and message. A terminal record is valid only after independently rehashing every listed file and proving there are no unlisted regular files in the evidence root.

- [ ] Run GREEN and commit.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider \
      -m 'not production_loader' \
      tests/test_certify_psi0_dataset.py tests/test_postprocess_psi0_publication.py
    ruff check scripts/postprocess_psi0.py scripts/certify_psi0_dataset.py tests/test_certify_psi0_dataset.py
    ruff format --check scripts/postprocess_psi0.py scripts/certify_psi0_dataset.py tests/test_certify_psi0_dataset.py
    git add scripts/postprocess_psi0.py scripts/certify_psi0_dataset.py tests/test_certify_psi0_dataset.py
    git commit -m "feat: certify immutable PSI0 dataset output"

---

### Task 8: Certify every sample through PSI0's pinned production loader

**Files:**

- Modify: pyproject.toml
- Modify: scripts/certify_psi0_dataset.py
- Modify: tests/test_certify_psi0_dataset.py

- [ ] Add RED loader subprocess tests with a fake pinned PSI0 source tree.

The fake exports psi.data.lerobot.compat.LeRobotDataset, records each __getitem__ index, and returns real torch.float32 tensors. Tests require:

- visited indices exactly list(range(total_frames));
- image shape (3,height,width), state shape (32,), action shape (36,);
- every floating tensor finite;
- every episode boundary covered;
- wrong shape, dtype, nonfinite, gap, duplicate, or missing index fails;
- dirty tracked PSI0 worktree and wrong commit fail before import;
- offline environment variables are present;
- HOME, HF_HOME, HF_DATASETS_CACHE, XDG_CACHE_HOME, TORCH_HOME, and TMPDIR point only into one private certificate-scoped cache;
- the loader may create actual Hugging Face/Datasets cache files, their pre-cleanup manifest is recorded, and the cache is absent before a terminal evidence record;
- cache cleanup runs after loader success and loader validation failure;
- a cache cleanup failure leaves a nonterminal evidence root and returns nonzero;
- environment and result manifests are canonical and digested.

The fake must fail if the certifier reads Parquet instead of calling dataset[index].

Also add `test_real_pinned_psi0_loader_traverses_every_synthetic_row`, marked `production_loader`. It must build and publish a two-episode, seven-row dataset through the production converter, invoke the real pinned `psi.data.lerobot.compat.LeRobotDataset`, and exercise its real `datasets.load_dataset()` path. Read `PSI0_PRODUCTION_ROOT` and `PSI0_PRODUCTION_COMMIT` inside the marked test body, never during module import or collection. When this test is selected, missing variables are a test failure, never a skip. The test asserts exact visited indices, image/state/action shape and dtype, finiteness, episode boundaries, a manifest of cache files created beneath the private evidence-scoped cache, successful cache removal, and no network access.

Register the marker in `pyproject.toml`. Every ordinary test command in this plan that can collect `tests/test_certify_psi0_dataset.py` must pass `-m 'not production_loader'`; only Task 11 passes `-m production_loader` together with the pinned environment variables and exact node ID.

Add this exact configuration:

    [tool.pytest.ini_options]
    markers = [
      "production_loader: requires the pinned PSI0 checkout and runs its real LeRobot loader",
    ]

- [ ] Run RED.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider \
      -m 'not production_loader' \
      tests/test_certify_psi0_dataset.py -k psi0_loader

- [ ] Implement a private --loader-worker mode.

The parent:

1. validates exact PSI0 HEAD and an empty tracked status;
2. resolves the Python executable;
3. records Python/platform plus PyTorch, PyArrow, NumPy, Datasets, TorchVision, PyAV, and a sorted importlib.metadata distribution manifest;
4. hashes canonical environment bytes;
5. exclusively creates evidence_root/.loader-cache and its home, hf, datasets, xdg, torch, and tmp children with mode 0700;
6. exclusively allocates a temporary result path inside the sibling evidence root;
7. invokes the worker with a scrubbed offline environment;
8. validates canonical result bytes and digest;
9. records a no-follow, sorted path/type/size/SHA-256 manifest of every cache file as psi0-loader-cache.json;
10. removes only the exact cache tree with the tested FD-relative no-follow remover, fsyncs the evidence root, and proves absence before writing PASS or FAIL.

Use this exact subprocess form:

    set -euo pipefail
    env -i \
      PATH=/usr/bin:/bin \
      HOME=<EVIDENCE_ROOT>/.loader-cache/home \
      HF_HOME=<EVIDENCE_ROOT>/.loader-cache/hf \
      HF_DATASETS_CACHE=<EVIDENCE_ROOT>/.loader-cache/datasets \
      XDG_CACHE_HOME=<EVIDENCE_ROOT>/.loader-cache/xdg \
      TORCH_HOME=<EVIDENCE_ROOT>/.loader-cache/torch \
      TMPDIR=<EVIDENCE_ROOT>/.loader-cache/tmp \
      HF_HUB_OFFLINE=1 \
      HF_DATASETS_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      PYTHONNOUSERSITE=1 \
      <PYTHON> -I scripts/certify_psi0_dataset.py --loader-worker \
      --psi0-src <PSI0_ROOT>/src \
      --dataset-root <DATASET_ROOT> \
      --result <EXCLUSIVE_TEMP_RESULT>

Python -I ignores PYTHONPATH. The worker must resolve --psi0-src, insert only that path at sys.path[0], and record it before importing:

    from psi.data.lerobot.compat import LeRobotDataset

Use the installed constructor exactly:

    dataset = LeRobotDataset(repo_id="simple-certified", root=dataset_root)

Then execute:

    visited = []
    for index in range(total_frames):
        sample = dataset[index]
        visited.append(index)
        validate_tensor(
            "observation.images.egocentric",
            sample["observation.images.egocentric"],
            (3, height, width),
        )
        validate_tensor("states", sample["states"], (32,))
        validate_tensor("action", sample["action"], (36,))
    if visited != list(range(total_frames)):
        raise RuntimeError("loader traversal was incomplete")

Validate each episode's inclusive dataset_from_index/dataset_to_index range against visited. Do not add a fallback loader or direct-Parquet shortcut.

Cache handling is a try/finally around subprocess launch, result decoding, and tensor validation. A terminal PASS or FAIL is forbidden until cleanup succeeds. If cleanup itself fails, preserve the private nonterminal evidence root, print its exact path and CACHE_CLEANUP_FAILED code, and return nonzero; do not claim a certification verdict.

- [ ] Run GREEN and commit.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider \
      -m 'not production_loader' tests/test_certify_psi0_dataset.py
    ruff check scripts/certify_psi0_dataset.py tests/test_certify_psi0_dataset.py
    ruff format --check scripts/certify_psi0_dataset.py tests/test_certify_psi0_dataset.py
    git add pyproject.toml scripts/certify_psi0_dataset.py tests/test_certify_psi0_dataset.py
    git commit -m "feat: traverse certified data through PSI0 loader"


---

### Task 9: Wire the CLI and exercise the complete synthetic pipeline

**Files:**

- Modify: scripts/postprocess_psi0.py
- Modify: tests/test_postprocess_psi0_e2e.py

- [ ] Add RED CLI tests.

Test exact options:

    --sim-root ROOT_OR_GLOB
    --out-dir PATH
    --skip INTEGER
    --downsample INTEGER
    --total-episodes INTEGER (with backward-compatible alias --total_episodes)
    --fps POSITIVE_RATIONAL
    --video-key STRING
    --chunks-size POSITIVE_INTEGER
    --preflight-only
    --certify-psi0-root PATH
    --certify-psi0-commit 40_HEX
    --certify-python PATH
    --inspect-preserved-staging PATH
    --remove-preserved-staging PATH
    --expected-status-sha256 64_HEX
    --confirm-remove EXACT_ABSOLUTE_PATH

Reject partial certification-option sets. No default may point to a developer checkout. Inspection and removal are mutually exclusive maintenance modes and cannot be combined with conversion/certification options except --out-dir, which identifies the associated stable lock.

Unit tests may inject a ConverterIdentity. At least one subprocess test must create a temporary Git repository, commit the real script, and invoke it so source attestation remains active.

Add this literal command-level regression so certificate discovery is tested by Bash rather than only described:

    def test_certificate_array_expansion_is_executable(tmp_path):
        import subprocess

        parent = tmp_path / "parent"
        parent.mkdir()
        certificate = parent / (
            ".output.certification-"
            + "a" * 64
            + "-00000000-0000-0000-0000-000000000001"
        )
        certificate.mkdir()
        script = r'''
    set -euo pipefail
    PARENT=$1
    mapfile -t CERTS < <(find "$PARENT" -maxdepth 1 -type d -name '.output.certification-*' -print | sort)
    test "${#CERTS[@]}" -eq 1
    CERT_ROOT=${CERTS[0]}
    test "$CERT_ROOT" = "$PARENT/.output.certification-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-00000000-0000-0000-0000-000000000001"
    '''
        subprocess.run(["bash", "-c", script, "bash", str(parent)], check=True)

Add this same-run uncertainty regression. It uses the production converter and filesystem adapter; only source identity and the certifier call are intercepted:

    def test_original_run_never_certifies_publication_uncertain(tmp_path, monkeypatch):
        import sys

        import pytest

        from scripts import postprocess_psi0 as converter
        from psi0_converter_fixtures import make_source_episode

        source = make_source_episode(tmp_path / "source", frames=8, fps=4)
        output = tmp_path / "processed"
        psi0_root = tmp_path / "psi0"
        psi0_root.mkdir()
        certifier_calls = []

        class FailDestinationParentFsync(converter.PublicationFilesystem):
            def fsync_directory(self, path):
                if path == output.parent and output.exists():
                    raise OSError("injected destination-parent fsync failure")
                super().fsync_directory(path)

        monkeypatch.setattr(
            converter,
            "resolve_converter_identity",
            lambda *_: converter.ConverterIdentity("a" * 40, "b" * 64),
        )
        monkeypatch.setattr(
            converter,
            "PublicationFilesystem",
            FailDestinationParentFsync,
        )
        monkeypatch.setattr(
            converter,
            "certify_published_dataset",
            lambda *args, **kwargs: certifier_calls.append((args, kwargs)),
        )
        args = converter.build_parser().parse_args(
            [
                "--sim-root", str(source),
                "--out-dir", str(output),
                "--skip", "0",
                "--downsample", "1",
                "--fps", "4",
                "--total-episodes", "1",
                "--video-key", "observation.rgb_head_stereo_left",
                "--chunks-size", "1000",
                "--certify-psi0-root", str(psi0_root),
                "--certify-psi0-commit", "c" * 40,
                "--certify-python", sys.executable,
            ]
        )

        with pytest.raises(converter.PublicationUncertainError):
            converter.run_conversion(args)

        assert output.is_dir()
        assert certifier_calls == []
        assert list(tmp_path.glob(".processed.certification-*")) == []

    def test_post_parent_fsync_hook_bypasses_failure_reporting_and_certification(
        tmp_path,
        monkeypatch,
    ):
        import sys

        import pytest

        from scripts import postprocess_psi0 as converter
        from psi0_converter_fixtures import make_source_episode

        source = make_source_episode(tmp_path / "source", frames=8, fps=4)
        output = tmp_path / "processed"
        psi0_root = tmp_path / "psi0"
        psi0_root.mkdir()
        failure_reports = []
        certifier_calls = []

        def fail_after_parent_fsync(point):
            if point == "after_destination_parent_fsync":
                raise RuntimeError("injected post-parent-fsync hook failure")

        monkeypatch.setattr(
            converter,
            "resolve_converter_identity",
            lambda *_: converter.ConverterIdentity("a" * 40, "b" * 64),
        )
        monkeypatch.setattr(converter, "no_fault", fail_after_parent_fsync)
        monkeypatch.setattr(
            converter,
            "report_conversion_failure",
            lambda *args: failure_reports.append(args),
        )
        monkeypatch.setattr(
            converter,
            "certify_published_dataset",
            lambda *args, **kwargs: certifier_calls.append((args, kwargs)),
        )
        args = converter.build_parser().parse_args(
            [
                "--sim-root", str(source),
                "--out-dir", str(output),
                "--skip", "0",
                "--downsample", "1",
                "--fps", "4",
                "--total-episodes", "1",
                "--video-key", "observation.rgb_head_stereo_left",
                "--chunks-size", "1000",
                "--certify-psi0-root", str(psi0_root),
                "--certify-psi0-commit", "c" * 40,
                "--certify-python", sys.executable,
            ]
        )

        with pytest.raises(converter.PublishedBoundaryError) as caught:
            converter.run_conversion(args)

        assert caught.value.publication.state == "published"
        assert output.is_dir()
        assert failure_reports == []
        assert certifier_calls == []
        assert list(tmp_path.glob(".processed.certification-*")) == []

- [ ] Add RED preserved-staging inspection/removal tests.

Inspection acquires the stable conversion lock and reports the no-follow tree identity, status bytes/hash, manifest state if present, modes, and failure classification without mutation. Authorized removal additionally requires:

- the path is an existing direct sibling named exactly .<output-name>.staging-<uuid>;
- parent/output/staging paths are absolute and contain no dot segments after lexical validation;
- the path is not a symlink and remains below the output parent;
- --expected-status-sha256 matches the no-follow status bytes;
- --confirm-remove exactly equals the absolute staging path;
- the canonical output is absent;
- the same stable lock remains held through deletion and parent fsync.

Use an FD-relative, no-follow recursive remover. Test nested files, symlinks as leaf entries, hard links, a sibling attack, path swap, wrong digest/confirmation, a live lock holder, and preservation of unrelated siblings. The converter never calls removal automatically.

- [ ] Run RED.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider \
      tests/test_postprocess_psi0_e2e.py -k cli

- [ ] Replace monolithic main() with this exact ordering.

    def run_conversion(args: argparse.Namespace) -> PublicationResult | ConversionPlan:
        repository = Path(__file__).resolve().parents[1]
        identity = resolve_converter_identity(repository, Path(__file__))
        output = Path(args.out_dir).resolve(strict=False)
        with conversion_lock(output):
            plan = preflight_conversion(args, identity)
            if args.preflight_only:
                print(canonical_json_bytes(preflight_report(plan)).decode(), end="")
                return plan
            staging = create_unique_staging(plan)
            write_in_progress_status(staging, plan)
            try:
                generate_staged_dataset(plan, staging)
                validate_staged_dataset(staging, plan)
                result = publish_staged_dataset(
                    staging,
                    output,
                    identity,
                    PublicationFilesystem(),
                    no_fault,
                )
            except (PublicationUncertainError, PublishedBoundaryError):
                # The staging pathname was already renamed. This original
                # operation exits nonzero without failure reporting or evidence.
                raise
            except BaseException as exc:
                report_conversion_failure(staging, exc)
                raise
            if result.state != "published":
                raise AssertionError("publish_staged_dataset returned a non-success state")
            if args.certify_psi0_root is not None:
                certify_published_dataset(
                    result,
                    psi0_root=Path(args.certify_psi0_root),
                    psi0_commit=args.certify_psi0_commit,
                    python=Path(args.certify_python),
                )
            return result

Preflight must finish before create_unique_staging(). Certification stays inside conversion_lock(). report_conversion_failure() may emit diagnostics but must not replace durable complete or mutate published output.

preflight_report(plan) returns this exact canonical JSON schema so Task 10 can machine-assert it:

    {
        "schema_version": 1,
        "output_path": str(plan.output_path),
        "selected_episodes": len(plan.episodes),
        "source_frames": sum(item.frame_count for item in plan.episodes),
        "retained_frames": sum(len(item.retained_indices) for item in plan.episodes),
        "media_mode": plan.media_mode,
        "output_profile": dataclasses.asdict(plan.output_media),
        "episodes": [
            {
                "source_episode_index": item.source_episode_index,
                "output_episode_index": item.output_episode_index,
                "source_frames": item.frame_count,
                "retained_frames": len(item.retained_indices),
                "source_media": dataclasses.asdict(item.source_media),
            }
            for item in plan.episodes
        ],
    }

Maintenance modes branch before converter-source attestation and preflight, acquire conversion_lock(output), perform only their named action, and exit. Removal is destructive and may be run only after explicit user approval for the exact inspected path.

- [ ] Run full synthetic E2E.

The positive test uses two heterogeneous episodes, exercises transcode_all, validates the published tree, runs all-index fake PSI0 traversal, and finds one terminal sibling PASS. It records snapshots proving source inputs did not change and no child process remains.

The negative E2E matrix covers preflight, generation, each durability class, destination collision, post-publication certification failure, and a second certification UUID. Its publication-uncertain case invokes `run_conversion()` with certification options, proves `PublicationUncertainError` exits the original operation nonzero, and snapshots the output parent to prove that no sibling evidence root was created. Its post-parent-fsync-hook case proves `PublishedBoundaryError(state="published")`, zero staging-failure reports, zero certifier calls, no sibling evidence root, and a restart-valid canonical output.

Run:

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider \
      -m 'not production_loader' \
      tests/test_postprocess_psi0_e2e.py tests/test_certify_psi0_dataset.py

- [ ] Run all focused tests and static checks.

    set -euo pipefail
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider \
      -m 'not production_loader' \
      tests/test_postprocess_psi0.py \
      tests/test_postprocess_psi0_preflight.py \
      tests/test_postprocess_psi0_rows.py \
      tests/test_postprocess_psi0_media.py \
      tests/test_postprocess_psi0_publication.py \
      tests/test_certify_psi0_dataset.py \
      tests/test_postprocess_psi0_e2e.py
    ruff check \
      scripts/postprocess_psi0.py scripts/certify_psi0_dataset.py \
      tests/psi0_converter_fixtures.py tests/test_postprocess_psi0*.py \
      tests/test_certify_psi0_dataset.py
    ruff format --check \
      scripts/postprocess_psi0.py scripts/certify_psi0_dataset.py \
      tests/psi0_converter_fixtures.py tests/test_postprocess_psi0*.py \
      tests/test_certify_psi0_dataset.py
    /home/jihun/work/SIMPLE/.venv/bin/python -m py_compile scripts/postprocess_psi0.py scripts/certify_psi0_dataset.py

- [ ] Commit all implementation before touching official data.

    set -euo pipefail
    git add scripts/postprocess_psi0.py scripts/certify_psi0_dataset.py tests
    git commit -m "feat: complete certified PSI0 converter v2"

- [ ] Re-run source attestation against committed bytes.

    /home/jihun/work/SIMPLE/.venv/bin/python - <<'PY'
    from pathlib import Path
    import subprocess
    from scripts.postprocess_psi0 import resolve_converter_identity

    root = Path.cwd().resolve()
    identity = resolve_converter_identity(root, root / "scripts/postprocess_psi0.py")
    expected = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", "scripts/postprocess_psi0.py"],
        text=True,
    ).strip()
    assert identity.commit == expected
    assert len(identity.script_sha256) == 64
    print(identity)
    PY

Expected: current committed converter identity, no mismatch.

---

### Task 10: Preflight official source read-only

**Files:**

- Read only: /home/jihun/work/SIMPLE/.worktrees/psi0-simple-pc2-bridge/outputs/official-psi0-breadth/20260901T021451Z-fd6972d/datasets/datagen-bendpick-l0/simple/G1WholebodyBendPickMP-v0/level-0

- [ ] Require destination and staging absence.

    set -euo pipefail
    RAW=/home/jihun/work/SIMPLE/.worktrees/psi0-simple-pc2-bridge/outputs/official-psi0-breadth/20260901T021451Z-fd6972d/datasets/datagen-bendpick-l0/simple/G1WholebodyBendPickMP-v0/level-0
    OUT=/home/jihun/work/SIMPLE/.worktrees/psi0-simple-pc2-bridge/outputs/official-psi0-breadth/20260901T021451Z-fd6972d/datasets/processed-psi0-bendpick-l0-v2
    test -d "$RAW"
    test -d "$(dirname "$OUT")"
    test ! -e "$OUT"
    test -z "$(find "$(dirname "$OUT")" -maxdepth 1 -name ".$(basename "$OUT").staging-*" -print -quit)"

If either exists, stop for inspection. Do not remove it automatically.

- [ ] Verify fixed source artifact identities.

    set -euo pipefail
    RAW=/home/jihun/work/SIMPLE/.worktrees/psi0-simple-pc2-bridge/outputs/official-psi0-breadth/20260901T021451Z-fd6972d/datasets/datagen-bendpick-l0/simple/G1WholebodyBendPickMP-v0/level-0
    test -d "$RAW"
    sha256sum --check --strict <<SHA256
    9ff41be14a579df9e389aebb9a1d716105ccc790f479a59915b7d8932231f522  $RAW/data/chunk-000/episode_000000.parquet
    c9140e38c584a127874ca6431298d6ab05356dc2f39443efbb1325b2c123f648  $RAW/videos/chunk-000/observation.rgb_head_stereo_left/episode_000000.mp4
    a35c847b459d2250a899ac2cf960fde8b598ae2d1984316826b83d6180b49073  $RAW/meta/info.json
    1bfd373fb67d266daaa3e0b18eb604d188509460515d0f9069bf073e5a1b755f  $RAW/meta/tasks.jsonl
    d95176ba41bfdc839be5d5ca64196da646c302cb5e304895d17985315a97035e  $RAW/meta/episodes.jsonl
    SHA256

Expected: all five lines report OK; any byte mismatch exits nonzero.

- [ ] Run the converter's read-only preflight entry point.

    set -euo pipefail
    cd /home/jihun/work/SIMPLE/.worktrees/psi0-converter-v2
    RAW=/home/jihun/work/SIMPLE/.worktrees/psi0-simple-pc2-bridge/outputs/official-psi0-breadth/20260901T021451Z-fd6972d/datasets/datagen-bendpick-l0/simple/G1WholebodyBendPickMP-v0/level-0
    OUT=/home/jihun/work/SIMPLE/.worktrees/psi0-simple-pc2-bridge/outputs/official-psi0-breadth/20260901T021451Z-fd6972d/datasets/processed-psi0-bendpick-l0-v2
    REPORT=$(mktemp /tmp/psi0-converter-v2-preflight.XXXXXX.json)
    trap 'rm -f -- "$REPORT"' EXIT
    /home/jihun/work/SIMPLE/.venv/bin/python scripts/postprocess_psi0.py \
      --preflight-only \
      --sim-root "$RAW" \
      --out-dir "$OUT" \
      --skip 60 \
      --downsample 1 \
      --fps 50 \
      --total-episodes 1 \
      --video-key observation.rgb_head_stereo_left \
      --chunks-size 1000 >"$REPORT"
    test ! -e "$OUT"
    test -z "$(find "$(dirname "$OUT")" -maxdepth 1 -name ".$(basename "$OUT").staging-*" -print -quit)"
    /home/jihun/work/SIMPLE/.venv/bin/python - "$REPORT" "$OUT" <<'PY'
    import json
    import sys

    report = json.load(open(sys.argv[1], encoding="utf-8"))
    assert set(report) == {
        "schema_version", "output_path", "selected_episodes", "source_frames",
        "retained_frames", "media_mode", "output_profile", "episodes",
    }
    assert report["schema_version"] == 1
    assert report["output_path"] == sys.argv[2]
    assert report["selected_episodes"] == 1
    assert report["source_frames"] == 214
    assert report["retained_frames"] == 154
    assert report["media_mode"] == "transcode_all"
    assert len(report["episodes"]) == 1
    episode = report["episodes"][0]
    assert set(episode) == {
        "source_episode_index", "output_episode_index", "source_frames",
        "retained_frames", "source_media",
    }
    assert episode["source_episode_index"] == 0
    assert episode["output_episode_index"] == 0
    assert episode["source_frames"] == 214
    assert episode["retained_frames"] == 154
    source = episode["source_media"]
    assert set(source) == {
        "codec_name", "pixel_format", "width", "height",
        "average_frame_rate", "nominal_frame_rate", "duration",
        "frame_count", "audio_streams",
    }
    assert source["codec_name"] == "av1"
    assert source["pixel_format"] == "yuv420p"
    assert [source["width"], source["height"]] == [640, 360]
    assert source["average_frame_rate"] == source["nominal_frame_rate"] == "50"
    from fractions import Fraction
    assert Fraction(source["duration"]) > 0
    assert source["frame_count"] == 214
    assert source["audio_streams"] == []
    output = report["output_profile"]
    assert output == {
        "codec_name": "h264", "pixel_format": "yuv420p", "width": 640,
        "height": 360, "average_frame_rate": "50", "nominal_frame_rate": "50",
        "audio_streams": [],
    }
    PY

No output or staging path exists. Stop on any mismatch.

---

### Task 11: Prepare the clean pinned PSI0 loader checkout

**Files:**

- Create only if absent: /home/jihun/work/Psi0/.worktrees/simple-dataset-cert-885538e

Pinned values:

    repository: /home/jihun/work/Psi0
    worktree: /home/jihun/work/Psi0/.worktrees/simple-dataset-cert-885538e
    commit: 885538e0bbee05caa1a89d382653c860596eee95
    Python: /home/jihun/work/SIMPLE/.venv/bin/python

- [ ] Validate the commit and create a detached worktree only if the path is absent.

    set -euo pipefail
    PSI0_REPO=/home/jihun/work/Psi0
    PSI0_WORKTREE=/home/jihun/work/Psi0/.worktrees/simple-dataset-cert-885538e
    PSI0_COMMIT=885538e0bbee05caa1a89d382653c860596eee95
    test -d "$PSI0_REPO/.git" -o -f "$PSI0_REPO/.git"
    git -C "$PSI0_REPO" cat-file -e "$PSI0_COMMIT^{commit}"
    if test -e "$PSI0_WORKTREE"; then
      test "$(git -C "$PSI0_WORKTREE" rev-parse HEAD)" = "$PSI0_COMMIT"
      test -z "$(git -C "$PSI0_WORKTREE" status --porcelain --untracked-files=no)"
    else
      git -C "$PSI0_REPO" worktree add --detach "$PSI0_WORKTREE" "$PSI0_COMMIT"
    fi
    test "$(git -C "$PSI0_WORKTREE" rev-parse HEAD)" = "$PSI0_COMMIT"
    test -z "$(git -C "$PSI0_WORKTREE" status --porcelain --untracked-files=no)"

Never remove or overwrite a pre-existing mismatched path.

- [ ] Probe import only; do not open official data.

    set -euo pipefail
    PSI0_WORKTREE=/home/jihun/work/Psi0/.worktrees/simple-dataset-cert-885538e
    PSI0_COMMIT=885538e0bbee05caa1a89d382653c860596eee95
    CERT_PYTHON=/home/jihun/work/SIMPLE/.venv/bin/python
    test "$(git -C "$PSI0_WORKTREE" rev-parse HEAD)" = "$PSI0_COMMIT"
    test -z "$(git -C "$PSI0_WORKTREE" status --porcelain --untracked-files=no)"
    test -x "$CERT_PYTHON"
    env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 \
      "$CERT_PYTHON" -I - "$PSI0_WORKTREE/src" <<'PY'
    import sys
    from pathlib import Path

    source = Path(sys.argv[1]).resolve(strict=True)
    sys.path.insert(0, str(source))
    from psi.data.lerobot.compat import LEROBOT_LAYOUT, LeRobotDataset

    print(LEROBOT_LAYOUT, LeRobotDataset)
    PY

Expected: production compatibility import succeeds offline. This is not certification.

- [ ] Run the real production-loader synthetic integration test.

The test `test_real_pinned_psi0_loader_traverses_every_synthetic_row` is added to tests/test_certify_psi0_dataset.py in Task 8 with marker production_loader. It uses the production staged generator and publisher to create a two-episode, 7-row synthetic dataset, invokes the real pinned `psi.data.lerobot.compat.LeRobotDataset`, and asserts visited_indices==list(range(7)), exact tensor shapes/dtypes/finiteness, both episode boundaries, terminal PASS, a nonempty Hugging Face/Datasets cache manifest, and absence of .loader-cache before terminalization. It does not monkeypatch the loader or datasets.load_dataset.

Run in a fresh shell:

    set -euo pipefail
    cd /home/jihun/work/SIMPLE/.worktrees/psi0-converter-v2
    PSI0_PRODUCTION_ROOT=/home/jihun/work/Psi0/.worktrees/simple-dataset-cert-885538e
    PSI0_PRODUCTION_COMMIT=885538e0bbee05caa1a89d382653c860596eee95
    test "$(git -C "$PSI0_PRODUCTION_ROOT" rev-parse HEAD)" = "$PSI0_PRODUCTION_COMMIT"
    test -z "$(git -C "$PSI0_PRODUCTION_ROOT" status --porcelain --untracked-files=no)"
    PSI0_PRODUCTION_ROOT="$PSI0_PRODUCTION_ROOT" \
    PSI0_PRODUCTION_COMMIT="$PSI0_PRODUCTION_COMMIT" \
    PYTHONDONTWRITEBYTECODE=1 \
      /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider \
      -m production_loader \
      tests/test_certify_psi0_dataset.py::test_real_pinned_psi0_loader_traverses_every_synthetic_row

Expected: `1 passed`; the test's temporary output and cache are cleaned, and no network is used.

---

### Task 12: Convert official data once and certify every sample

This task is gated on Tasks 0-11 and a clean committed implementation. It does no simulation or training.

- [ ] Reconfirm implementation cleanliness and converter identity.

    set -euo pipefail
    cd /home/jihun/work/SIMPLE/.worktrees/psi0-converter-v2
    git diff --check b7aa36273c47ead2d9a002336722a781d70896f2..HEAD
    test -z "$(git status --porcelain --untracked-files=no)"
    /home/jihun/work/SIMPLE/.venv/bin/python -c 'from pathlib import Path; from scripts.postprocess_psi0 import resolve_converter_identity; print(resolve_converter_identity(Path.cwd(), Path("scripts/postprocess_psi0.py")))'

- [ ] Snapshot all raw source hashes into an exclusive mode-0600 handoff.

    set -euo pipefail
    umask 077
    RAW=/home/jihun/work/SIMPLE/.worktrees/psi0-simple-pc2-bridge/outputs/official-psi0-breadth/20260901T021451Z-fd6972d/datasets/datagen-bendpick-l0/simple/G1WholebodyBendPickMP-v0/level-0
    SNAPSHOT=/tmp/psi0-converter-v2-raw-before.sha256
    test -d "$RAW"
    test ! -e "$SNAPSHOT"
    set -o noclobber
    find "$RAW" -type f -print0 | sort -z | xargs -0 sha256sum >"$SNAPSHOT"
    set +o noclobber
    test "$(stat -c '%a' "$SNAPSHOT")" = "600"
    test -s "$SNAPSHOT"

- [ ] Execute the exact conversion and certification command.

    set -euo pipefail
    cd /home/jihun/work/SIMPLE/.worktrees/psi0-converter-v2
    RAW=/home/jihun/work/SIMPLE/.worktrees/psi0-simple-pc2-bridge/outputs/official-psi0-breadth/20260901T021451Z-fd6972d/datasets/datagen-bendpick-l0/simple/G1WholebodyBendPickMP-v0/level-0
    OUT=/home/jihun/work/SIMPLE/.worktrees/psi0-simple-pc2-bridge/outputs/official-psi0-breadth/20260901T021451Z-fd6972d/datasets/processed-psi0-bendpick-l0-v2
    PSI0_WORKTREE=/home/jihun/work/Psi0/.worktrees/simple-dataset-cert-885538e
    PSI0_COMMIT=885538e0bbee05caa1a89d382653c860596eee95
    CERT_PYTHON=/home/jihun/work/SIMPLE/.venv/bin/python
    SNAPSHOT=/tmp/psi0-converter-v2-raw-before.sha256
    test -d "$RAW"
    test -f "$SNAPSHOT"
    test "$(stat -c '%a' "$SNAPSHOT")" = "600"
    test ! -e "$OUT"
    test "$(git -C "$PSI0_WORKTREE" rev-parse HEAD)" = "$PSI0_COMMIT"
    test -z "$(git -C "$PSI0_WORKTREE" status --porcelain --untracked-files=no)"
    "$CERT_PYTHON" scripts/postprocess_psi0.py \
      --sim-root "$RAW" \
      --out-dir "$OUT" \
      --skip 60 \
      --downsample 1 \
      --fps 50 \
      --total-episodes 1 \
      --video-key observation.rgb_head_stereo_left \
      --chunks-size 1000 \
      --certify-psi0-root "$PSI0_WORKTREE" \
      --certify-psi0-commit "$PSI0_COMMIT" \
      --certify-python "$CERT_PYTHON"
    test -d "$OUT"

Expected: exit 0, one canonical output, 154 rows, one terminal sibling PASS root, and no post-publication file beneath the dataset.

- [ ] Prove source bytes did not change and retire the temporary handoff.

    set -euo pipefail
    umask 077
    RAW=/home/jihun/work/SIMPLE/.worktrees/psi0-simple-pc2-bridge/outputs/official-psi0-breadth/20260901T021451Z-fd6972d/datasets/datagen-bendpick-l0/simple/G1WholebodyBendPickMP-v0/level-0
    SNAPSHOT=/tmp/psi0-converter-v2-raw-before.sha256
    AFTER=/tmp/psi0-converter-v2-raw-after.sha256
    test -f "$SNAPSHOT"
    test "$(stat -c '%a' "$SNAPSHOT")" = "600"
    test ! -e "$AFTER"
    set -o noclobber
    find "$RAW" -type f -print0 | sort -z | xargs -0 sha256sum >"$AFTER"
    set +o noclobber
    cmp "$SNAPSHOT" "$AFTER"
    rm -f -- "$SNAPSHOT" "$AFTER"
    test ! -e "$SNAPSHOT"
    test ! -e "$AFTER"

- [ ] Resolve and inspect the manifest-bound certificate without inherited variables.

    set -euo pipefail
    cd /home/jihun/work/SIMPLE/.worktrees/psi0-converter-v2
    OUT=/home/jihun/work/SIMPLE/.worktrees/psi0-simple-pc2-bridge/outputs/official-psi0-breadth/20260901T021451Z-fd6972d/datasets/processed-psi0-bendpick-l0-v2
    test -d "$OUT"
    MANIFEST_SHA=$(/home/jihun/work/SIMPLE/.venv/bin/python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$OUT/meta/conversion_manifest.json")
    mapfile -t CERTS < <(find "$(dirname "$OUT")" -maxdepth 1 -type d -name ".$(basename "$OUT").certification-$MANIFEST_SHA-*" -print | sort)
    test "${#CERTS[@]}" -eq 1
    CERT_ROOT=${CERTS[0]}
    test -f "$CERT_ROOT/PASS.json"
    test ! -e "$CERT_ROOT/FAIL.json"

    /home/jihun/work/SIMPLE/.venv/bin/python - "$CERT_ROOT" "$OUT" <<'PY'
    import json
    import sys
    from pathlib import Path

    certificate = Path(sys.argv[1])
    dataset = Path(sys.argv[2])
    info = json.loads((dataset / "meta/info.json").read_text())
    loader = json.loads((certificate / "psi0-loader-result.json").read_text())
    assert info["total_frames"] == 154
    assert loader["visited_indices"] == list(range(154))
    assert loader["episode_boundaries"] == [
        {"episode_index": 0, "from": 0, "to": 153}
    ]
    assert loader["sample_count"] == 154
    assert loader["all_finite"] is True
    PY

- [ ] Run one standalone certification retry and prove dataset immutability.

Use the certifier's own manifest verification to obtain BEFORE_TREE and AFTER_TREE rather than a home-grown hash. The command must print a canonical tree digest without changing the dataset:

    set -euo pipefail
    cd /home/jihun/work/SIMPLE/.worktrees/psi0-converter-v2
    OUT=/home/jihun/work/SIMPLE/.worktrees/psi0-simple-pc2-bridge/outputs/official-psi0-breadth/20260901T021451Z-fd6972d/datasets/processed-psi0-bendpick-l0-v2
    PSI0_WORKTREE=/home/jihun/work/Psi0/.worktrees/simple-dataset-cert-885538e
    PSI0_COMMIT=885538e0bbee05caa1a89d382653c860596eee95
    CERT_PYTHON=/home/jihun/work/SIMPLE/.venv/bin/python
    test -d "$OUT"
    test "$(git -C "$PSI0_WORKTREE" rev-parse HEAD)" = "$PSI0_COMMIT"
    test -z "$(git -C "$PSI0_WORKTREE" status --porcelain --untracked-files=no)"
    BEFORE_TREE=$("$CERT_PYTHON" scripts/certify_psi0_dataset.py "$OUT" --print-tree-digest)
    "$CERT_PYTHON" scripts/certify_psi0_dataset.py "$OUT" \
      --psi0-root "$PSI0_WORKTREE" \
      --psi0-commit "$PSI0_COMMIT" \
      --python "$CERT_PYTHON" \
      --expected-video-key observation.images.egocentric
    AFTER_TREE=$("$CERT_PYTHON" scripts/certify_psi0_dataset.py "$OUT" --print-tree-digest)
    test "$BEFORE_TREE" = "$AFTER_TREE"
    MANIFEST_SHA=$("$CERT_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$OUT/meta/conversion_manifest.json")
    mapfile -t CERTS < <(find "$(dirname "$OUT")" -maxdepth 1 -type d -name ".$(basename "$OUT").certification-$MANIFEST_SHA-*" -print | sort)
    test "${#CERTS[@]}" -eq 2
    for certificate in "${CERTS[@]}"; do
      test -f "$certificate/PASS.json"
      test ! -e "$certificate/FAIL.json"
    done

Expected: a second different certificate UUID and terminal PASS; canonical tree digest unchanged.

If publication succeeds but certification fails, classify the output PUBLISHED_UNCERTIFIED, preserve it and all evidence, and stop. Never rerun conversion at the same destination or repair the canonical tree.

---

### Task 13: Final verification and handoff, then stop

- [ ] Run all focused tests and static gates after official certification.

    set -euo pipefail
    cd /home/jihun/work/SIMPLE/.worktrees/psi0-converter-v2
    PYTHONDONTWRITEBYTECODE=1 /home/jihun/work/SIMPLE/.venv/bin/python -m pytest -q -p no:cacheprovider \
      -m 'not production_loader' \
      tests/test_postprocess_psi0.py \
      tests/test_postprocess_psi0_preflight.py \
      tests/test_postprocess_psi0_rows.py \
      tests/test_postprocess_psi0_media.py \
      tests/test_postprocess_psi0_publication.py \
      tests/test_certify_psi0_dataset.py \
      tests/test_postprocess_psi0_e2e.py
    ruff check \
      scripts/postprocess_psi0.py scripts/certify_psi0_dataset.py \
      tests/psi0_converter_fixtures.py tests/test_postprocess_psi0*.py \
      tests/test_certify_psi0_dataset.py
    ruff format --check \
      scripts/postprocess_psi0.py scripts/certify_psi0_dataset.py \
      tests/psi0_converter_fixtures.py tests/test_postprocess_psi0*.py \
      tests/test_certify_psi0_dataset.py
    /home/jihun/work/SIMPLE/.venv/bin/python -m py_compile scripts/postprocess_psi0.py scripts/certify_psi0_dataset.py
    git diff --check b7aa36273c47ead2d9a002336722a781d70896f2..HEAD
    test -z "$(git status --porcelain --untracked-files=no)"
    git diff --quiet
    git diff --cached --quiet

- [ ] Produce a read-only handoff summary with:

    implementation commit
    converter commit and script SHA-256
    source artifact SHA-256 values
    canonical output path and root identity
    manifest and complete-status SHA-256
    all output file hashes
    media mode and verified profile
    episode/global row and statistics validation
    PSI0 commit, interpreter, environment-manifest digest
    loader result digest and exact visited-index range
    every sibling certificate root and terminal verdict
    raw before/after comparison result
    pytest/Ruff/format/compile results
    explicit statement that no simulation, training, inference server, H100, or robot process started

Do not append the summary to a sealed dataset or sealed evidence root. Print it and link the already-created immutable evidence.

- [ ] Stop. Do not run a training smoke, H100 ownership probe, simulation evaluation, inference launch, or deployment step under this plan.

---

## Plan-review checklist

After—and only after—approval of the current plan commit, create the immutable execution handoff in a fresh shell. This command is part of the gate and is not authorized by this unapproved revision:

    set -euo pipefail
    cd /home/jihun/work/SIMPLE/.worktrees/psi0-converter-v2
    BASE=b7aa36273c47ead2d9a002336722a781d70896f2
    TAG=psi0-converter-v2-plan-approved
    test "$(git rev-parse HEAD^)" = "$BASE"
    test "$(git diff --name-only "$BASE"..HEAD)" = docs/superpowers/plans/2026-09-03-psi0-converter-v2-certification.md
    test -z "$(git status --porcelain --untracked-files=no)"
    git diff --quiet
    git diff --cached --quiet
    test -z "$(git tag --list "$TAG")"
    git tag -a "$TAG" -m "approve PSI0 converter v2 implementation plan" HEAD
    test "$(git rev-parse "$TAG^{commit}")" = "$(git rev-parse HEAD)"

Before implementation approval, verify:

- [ ] Every source schema and metadata lookup is validated before staging.
- [ ] frame_count <= skip fails before output/staging creation.
- [ ] One retained-index array controls rows, statistics, indices, and video selection.
- [ ] Seven vector columns are exact fixed-size float32; every float is finite.
- [ ] Every episode/global statistics block contains the exact count.
- [ ] Media is copy-all or canonical transcode-all; heterogeneous output is impossible.
- [ ] Converter bytes equal a committed blob before official conversion.
- [ ] Stable flock precedes staging and ownership is kernel-released.
- [ ] Manifest membership and reserved exclusions are exact and non-self-referential.
- [ ] Every generated file and directory is fsynced before durable completion.
- [ ] complete is persisted only after payload fsync and revalidation.
- [ ] All four failure classes have disk-backed and crash-restart tests.
- [ ] Output uses no-replace rename, is sealed/immutable, and parent-fsynced.
- [ ] Certification evidence exists only in unique manifest-bound sibling roots.
- [ ] PSI0 calls production dataset[i] for every index and records the exact sequence.
- [ ] Official conversion follows the implementation commit and uses only the fresh -v2 path.
- [ ] The plan ends before simulation, training, inference, H100, or robot work.
