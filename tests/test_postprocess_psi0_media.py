import argparse
import dataclasses
import errno
import hashlib
import os
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from psi0_converter_fixtures import make_source_episode
from scripts import postprocess_psi0 as converter


IDENTITY = converter.ConverterIdentity(commit="0" * 40, script_sha256="1" * 64)
VIDEO_KEY = "observation.rgb_head_stereo_left"


def _identity(**overrides: object) -> converter.MediaIdentity:
    values = {
        "codec_name": "h264",
        "pixel_format": "yuv420p",
        "width": 64,
        "height": 48,
        "average_frame_rate": "4",
        "nominal_frame_rate": "4",
        "duration": "2",
        "frame_count": 8,
        "audio_streams": (),
    }
    values.update(overrides)
    return converter.MediaIdentity(**values)


def _episode(
    tmp_path: Path,
    *,
    name: str = "source",
    identity: converter.MediaIdentity | None = None,
    retained_indices: np.ndarray | None = None,
) -> converter.EpisodePlan:
    source_media = identity or _identity()
    if retained_indices is None:
        retained_indices = np.arange(source_media.frame_count, dtype=np.int64)
    retained_indices.setflags(write=False)
    source_root = tmp_path / name
    return converter.EpisodePlan(
        source_root=source_root,
        source_episode_index=0,
        output_episode_index=0,
        parquet_path=source_root / "episode.parquet",
        video_path=source_root / "episode.mp4",
        parquet_sha256="2" * 64,
        video_sha256="3" * 64,
        frame_count=source_media.frame_count,
        retained_indices=retained_indices,
        source_task_index=0,
        output_task_index=0,
        task_text="test task",
        environment_config="{}",
        source_media=source_media,
    )


def _args(source: Path | str, output: Path, **overrides: object) -> argparse.Namespace:
    values = {
        "sim_root": str(source),
        "out_dir": str(output),
        "skip": 0,
        "downsample": 1,
        "total_episodes": 100,
        "fps": "4",
        "video_key": VIDEO_KEY,
        "chunks_size": 1000,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _preflight(source: Path | str, output: Path, **overrides: object):
    return converter.preflight_conversion(_args(source, output, **overrides), IDENTITY)


def _video(root: Path, episode_index: int = 0) -> Path:
    return (
        root / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{episode_index:06d}.mp4"
    )


def _private_media_directories(destination: Path) -> list[Path]:
    return sorted(
        path
        for path in destination.parent.iterdir()
        if path.is_dir() and path.name.startswith(f".{destination.name}.media-")
    )


def _overwrite_with_red_video(path: Path, *, frames: int, size: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=red:size={size}:rate=4",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _decoded_frame_digests(path: Path) -> str:
    return subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "framemd5",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_required_ffmpeg_and_libx264_are_available():
    version = subprocess.run(
        ["ffmpeg", "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    first_line = version.stdout.splitlines()[0]
    major = int(first_line.split()[2].split(".", maxsplit=1)[0])
    assert major >= 4

    encoders = subprocess.run(
        ["ffmpeg", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "libx264" in encoders.stdout


def test_homogeneous_unsampled_matching_fps_selects_copy_all(tmp_path):
    first = _episode(tmp_path, name="first")
    second = _episode(
        tmp_path,
        name="second",
        identity=dataclasses.replace(first.source_media, duration="99/4"),
    )

    mode, profile = converter.decide_media_mode(
        (first, second), skip=0, downsample=1, output_fps=Fraction(4)
    )

    assert mode == "copy_all"
    assert profile == converter.media_profile(first.source_media)
    assert not hasattr(profile, "duration")
    assert not hasattr(profile, "frame_count")


@pytest.mark.parametrize(
    ("skip", "downsample", "output_fps"),
    [
        (1, 1, Fraction(4)),
        (0, 2, Fraction(4)),
        (0, 1, Fraction(5)),
    ],
)
def test_sampling_or_fps_mismatch_selects_transcode_all(
    tmp_path, skip, downsample, output_fps
):
    episode = _episode(
        tmp_path,
        retained_indices=np.arange(skip, 8, downsample, dtype=np.int64),
    )

    mode, profile = converter.decide_media_mode(
        (episode,),
        skip=skip,
        downsample=downsample,
        output_fps=output_fps,
    )

    assert mode == "transcode_all"
    assert profile == converter.MediaProfile(
        codec_name="h264",
        pixel_format="yuv420p",
        width=640,
        height=360,
        average_frame_rate=str(output_fps),
        nominal_frame_rate=str(output_fps),
        audio_streams=(),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("codec_name", "av1"),
        ("pixel_format", "yuv444p"),
        ("width", 80),
        ("height", 60),
        ("average_frame_rate", "5"),
        ("nominal_frame_rate", "5"),
        (
            "audio_streams",
            (
                {
                    "codec_name": "aac",
                    "sample_fmt": "fltp",
                    "sample_rate": 48000,
                    "channels": 2,
                    "channel_layout": "stereo",
                },
            ),
        ),
    ],
)
def test_any_dataset_media_mismatch_selects_transcode_all(tmp_path, field, value):
    first = _episode(tmp_path, name="first")
    changed_identity = dataclasses.replace(first.source_media, **{field: value})
    second = _episode(tmp_path, name="second", identity=changed_identity)

    mode, profile = converter.decide_media_mode(
        (first, second), skip=0, downsample=1, output_fps=Fraction(4)
    )

    assert mode == "transcode_all"
    assert profile.codec_name == "h264"
    assert profile.pixel_format == "yuv420p"
    assert (profile.width, profile.height) == (640, 360)
    assert profile.average_frame_rate == "4"
    assert profile.nominal_frame_rate == "4"
    assert profile.audio_streams == ()


def test_source_frame_count_must_equal_retained_count_for_copy_all(tmp_path):
    episode = _episode(
        tmp_path,
        retained_indices=np.arange(7, dtype=np.int64),
    )

    mode, _ = converter.decide_media_mode(
        (episode,), skip=0, downsample=1, output_fps=Fraction(4)
    )

    assert mode == "transcode_all"


def test_copy_all_writes_byte_identical_output_with_matching_sha(tmp_path):
    source = make_source_episode(tmp_path / "source")
    plan = _preflight(source, tmp_path / "unused-output")
    destination = tmp_path / "staging" / "episode.mp4"
    destination.parent.mkdir()

    output_identity = converter.write_episode_video(
        episode=plan.episodes[0],
        destination=destination,
        media_mode=plan.media_mode,
        output_fps=plan.output_fps,
        output_profile=plan.output_media,
    )

    source_bytes = _video(source).read_bytes()
    assert destination.read_bytes() == source_bytes
    assert (
        hashlib.sha256(destination.read_bytes()).hexdigest()
        == plan.episodes[0].video_sha256
    )
    assert output_identity == plan.episodes[0].source_media


def test_copy_all_collision_never_overwrites_destination(tmp_path):
    source = make_source_episode(tmp_path / "source")
    plan = _preflight(source, tmp_path / "unused-output")
    destination = tmp_path / "staging" / "episode.mp4"
    destination.parent.mkdir()
    destination.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError):
        converter.write_episode_video(
            episode=plan.episodes[0],
            destination=destination,
            media_mode=plan.media_mode,
            output_fps=plan.output_fps,
            output_profile=plan.output_media,
        )

    assert destination.read_bytes() == b"sentinel"
    assert _video(source).exists()


def test_copy_all_rejects_probe_time_destination_swap(tmp_path, monkeypatch):
    source = make_source_episode(tmp_path / "source")
    replacement_source = make_source_episode(tmp_path / "replacement-source")
    replacement = _video(replacement_source)
    _overwrite_with_red_video(replacement, frames=8, size="64x48")
    plan = _preflight(source, tmp_path / "unused-output")
    destination = tmp_path / "staging" / "episode.mp4"
    destination.parent.mkdir()
    source_bytes = _video(source).read_bytes()
    replacement_bytes = replacement.read_bytes()
    assert replacement_bytes != source_bytes
    actual_probe = converter.probe_media

    def swapping_probe(path: Path, **kwargs):
        identity = actual_probe(path, **kwargs)
        if path.name == destination.name:
            destination.unlink()
            replacement.rename(destination)
        return identity

    monkeypatch.setattr(converter, "probe_media", swapping_probe)
    with pytest.raises(RuntimeError, match="identity changed"):
        converter.write_episode_video(
            episode=plan.episodes[0],
            destination=destination,
            media_mode=plan.media_mode,
            output_fps=plan.output_fps,
            output_profile=plan.output_media,
        )

    assert destination.read_bytes() == replacement_bytes
    assert destination.read_bytes() != source_bytes
    assert _video(source).read_bytes() == source_bytes


def test_copy_all_rejects_parent_replacement_after_durability_fsync(
    tmp_path, monkeypatch
):
    source = make_source_episode(tmp_path / "source")
    plan = _preflight(source, tmp_path / "unused-output")
    destination = tmp_path / "staging" / "episode.mp4"
    destination.parent.mkdir()
    saved_parent = tmp_path / "saved-staging"
    actual_fsync_directory = converter.fsync_directory
    attacked = False

    def replace_parent_after_fsync(
        path: Path, *, directory_fd: int | None = None
    ) -> None:
        nonlocal attacked
        actual_fsync_directory(path, directory_fd=directory_fd)
        if path == destination.parent and not attacked:
            attacked = True
            destination.parent.rename(saved_parent)
            destination.parent.mkdir()
            os.link(saved_parent / destination.name, destination)
            (saved_parent / destination.name).unlink()

    monkeypatch.setattr(converter, "fsync_directory", replace_parent_after_fsync)
    with pytest.raises(RuntimeError, match="identity changed|link count"):
        converter.write_episode_video(
            episode=plan.episodes[0],
            destination=destination,
            media_mode=plan.media_mode,
            output_fps=plan.output_fps,
            output_profile=plan.output_media,
        )

    assert attacked
    assert destination.is_file()
    assert not (saved_parent / destination.name).exists()


def test_copy_all_rejects_byte_identical_replacement_before_verification_reopen(
    tmp_path, monkeypatch
):
    source = make_source_episode(tmp_path / "source")
    plan = _preflight(source, tmp_path / "unused-output")
    destination = tmp_path / "staging" / "episode.mp4"
    destination.parent.mkdir()
    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(_video(source).read_bytes())
    preserved_created = destination.parent / "exclusively-created.mp4"
    actual_open = converter.os.open
    actual_close = converter.os.close
    write_fd = None
    swapped = False

    def monitored_open(path, flags, *args, **kwargs):
        nonlocal write_fd
        fd = actual_open(path, flags, *args, **kwargs)
        if (
            path == os.fsencode(destination.name)
            and flags & os.O_WRONLY
            and flags & os.O_EXCL
        ):
            write_fd = fd
        return fd

    def swap_after_write_close(fd: int) -> None:
        nonlocal swapped
        actual_close(fd)
        if fd == write_fd and not swapped:
            swapped = True
            destination.rename(preserved_created)
            replacement.rename(destination)

    monkeypatch.setattr(converter.os, "open", monitored_open)
    monkeypatch.setattr(converter.os, "close", swap_after_write_close)
    with pytest.raises(RuntimeError, match="identity changed"):
        converter.write_episode_video(
            episode=plan.episodes[0],
            destination=destination,
            media_mode=plan.media_mode,
            output_fps=plan.output_fps,
            output_profile=plan.output_media,
        )

    assert swapped
    assert destination.read_bytes() == preserved_created.read_bytes()
    assert destination.stat().st_ino != preserved_created.stat().st_ino
    assert destination.stat().st_nlink == 1
    assert preserved_created.stat().st_nlink == 1


@pytest.mark.parametrize("failure", ["unavailable", "cross-device"])
def test_rename_noreplace_fails_closed_when_kernel_support_is_unusable(
    tmp_path, monkeypatch, failure
):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"source")

    if failure == "unavailable":

        class FakeLibc:
            pass

    else:

        class FakeRename:
            argtypes = None
            restype = None

            def __call__(self, *args):
                converter.ctypes.set_errno(errno.EXDEV)
                return -1

        class FakeLibc:
            renameat2 = FakeRename()

    monkeypatch.setattr(converter.ctypes, "CDLL", lambda *args, **kwargs: FakeLibc())
    expected_error = RuntimeError if failure == "unavailable" else OSError
    with pytest.raises(expected_error):
        converter.rename_noreplace(source, destination)

    assert source.read_bytes() == b"source"
    assert not destination.exists()


def test_two_heterogeneous_inputs_produce_one_canonical_profile(tmp_path):
    first = make_source_episode(
        tmp_path / "source-a", fps=4, size="64x48", video_codec="libx264"
    )
    second = make_source_episode(
        tmp_path / "source-b", fps=5, size="80x60", video_codec="mpeg4"
    )
    plan = _preflight(str(tmp_path / "source-*"), tmp_path / "unused-output")
    destination_parent = tmp_path / "staging"
    destination_parent.mkdir()

    identities = [
        converter.write_episode_video(
            episode=episode,
            destination=destination_parent / f"episode-{index}.mp4",
            media_mode=plan.media_mode,
            output_fps=plan.output_fps,
            output_profile=plan.output_media,
        )
        for index, episode in enumerate(plan.episodes)
    ]

    assert plan.media_mode == "transcode_all"
    assert all(
        converter.media_profile(value) == plan.output_media for value in identities
    )
    assert all(value.frame_count == 8 for value in identities)
    assert not [path for path in destination_parent.iterdir() if path.is_dir()]
    assert _video(first).exists()
    assert _video(second).exists()


def test_transcode_collision_preserves_sentinel_and_private_artifact(
    tmp_path, monkeypatch
):
    source = make_source_episode(tmp_path / "source")
    plan = _preflight(source, tmp_path / "unused-output", skip=1, downsample=2)
    destination = tmp_path / "staging" / "episode.mp4"
    destination.parent.mkdir()
    actual_run = converter.subprocess.run
    ffmpeg_calls = []

    def recording_run(argv, **kwargs):
        if argv[0] == "ffmpeg":
            ffmpeg_calls.append((argv, kwargs))
        return actual_run(argv, **kwargs)

    def collide() -> None:
        destination.write_bytes(b"sentinel")

    monkeypatch.setattr(converter.subprocess, "run", recording_run)
    with pytest.raises(FileExistsError):
        converter.write_episode_video(
            episode=plan.episodes[0],
            destination=destination,
            media_mode=plan.media_mode,
            output_fps=plan.output_fps,
            output_profile=plan.output_media,
            before_media_publish=collide,
        )

    assert destination.read_bytes() == b"sentinel"
    private_directories = _private_media_directories(destination)
    assert len(private_directories) == 1
    private_artifact = private_directories[0] / "artifact.mp4"
    assert private_artifact.is_file()
    assert converter.probe_media(private_artifact).frame_count == len(
        plan.episodes[0].retained_indices
    )
    assert len(ffmpeg_calls) == 1
    argv, kwargs = ffmpeg_calls[0]
    assert argv[:3] == ["ffmpeg", "-n", "-nostdin"]
    assert argv.index("-n") < argv.index("-i")
    assert argv.index("-nostdin") < argv.index("-i")
    assert "-fps_mode" not in argv
    assert (
        "select=eq(n\\,1)+eq(n\\,3)+eq(n\\,5)+eq(n\\,7)" in argv[argv.index("-vf") + 1]
    )
    assert "scale=640:360:flags=lanczos" in argv[argv.index("-vf") + 1]
    assert "setpts=N/(4*TB)" in argv[argv.index("-vf") + 1]
    assert argv[argv.index("-vsync") + 1] == "cfr"
    assert argv[argv.index("-r") + 1] == "4"
    assert argv[argv.index("-frames:v") + 1] == "4"
    assert argv[argv.index("-c:v") + 1] == "libx264"
    assert argv[argv.index("-pix_fmt") + 1] == "yuv420p"
    assert "-an" in argv
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs.get("shell") is not True
    source_argument = Path(argv[argv.index("-i") + 1])
    assert source_argument.parent == Path("/proc/self/fd")
    assert int(source_argument.name) in kwargs["pass_fds"]
    assert len(kwargs["pass_fds"]) == 2


def test_transcode_rejects_post_preflight_source_replacement(tmp_path):
    source = make_source_episode(tmp_path / "source")
    plan = _preflight(source, tmp_path / "unused-output", skip=1)
    original = _video(source)
    original_bytes = original.read_bytes()
    replacement_source = make_source_episode(tmp_path / "replacement-source")
    replacement = _video(replacement_source)
    _overwrite_with_red_video(replacement, frames=8, size="64x48")
    replacement_bytes = replacement.read_bytes()
    assert replacement_bytes != original_bytes
    preserved_original = original.with_name("preflight-original.mp4")
    original.rename(preserved_original)
    replacement.rename(original)
    destination = tmp_path / "staging" / "episode.mp4"
    destination.parent.mkdir()

    with pytest.raises(RuntimeError, match="source video"):
        converter.write_episode_video(
            episode=plan.episodes[0],
            destination=destination,
            media_mode=plan.media_mode,
            output_fps=plan.output_fps,
            output_profile=plan.output_media,
        )

    assert not destination.exists()
    assert original.read_bytes() == replacement_bytes
    assert preserved_original.read_bytes() == original_bytes


def test_transcode_uses_clean_snapshot_during_transient_source_mutation(
    tmp_path, monkeypatch
):
    source = make_source_episode(tmp_path / "source")
    plan = _preflight(source, tmp_path / "unused-output", skip=1)
    original = _video(source)
    original_bytes = original.read_bytes()
    replacement_source = make_source_episode(tmp_path / "replacement-source")
    replacement = _video(replacement_source)
    _overwrite_with_red_video(replacement, frames=8, size="64x48")
    replacement_bytes = replacement.read_bytes()
    assert replacement_bytes != original_bytes

    clean_destination = tmp_path / "clean-staging" / "episode.mp4"
    clean_destination.parent.mkdir()
    converter.write_episode_video(
        episode=plan.episodes[0],
        destination=clean_destination,
        media_mode=plan.media_mode,
        output_fps=plan.output_fps,
        output_profile=plan.output_media,
    )
    clean_digests = _decoded_frame_digests(clean_destination)
    attacked_destination = tmp_path / "attacked-staging" / "episode.mp4"
    attacked_destination.parent.mkdir()
    actual_run = converter.subprocess.run
    mutation_observed = False

    def mutate_during_ffmpeg(argv, **kwargs):
        nonlocal mutation_observed
        if argv[0] != "ffmpeg":
            return actual_run(argv, **kwargs)
        mutation_observed = True
        with original.open("r+b") as stream:
            stream.write(replacement_bytes)
            stream.truncate()
        try:
            return actual_run(argv, **kwargs)
        finally:
            with original.open("r+b") as stream:
                stream.write(original_bytes)
                stream.truncate()

    monkeypatch.setattr(converter.subprocess, "run", mutate_during_ffmpeg)
    converter.write_episode_video(
        episode=plan.episodes[0],
        destination=attacked_destination,
        media_mode=plan.media_mode,
        output_fps=plan.output_fps,
        output_profile=plan.output_media,
    )

    assert mutation_observed
    assert original.read_bytes() == original_bytes
    assert _decoded_frame_digests(attacked_destination) == clean_digests


def test_transcode_fails_closed_when_sealed_memfd_is_unavailable(tmp_path, monkeypatch):
    source = make_source_episode(tmp_path / "source")
    plan = _preflight(source, tmp_path / "unused-output", skip=1)
    destination = tmp_path / "staging" / "episode.mp4"
    destination.parent.mkdir()

    class LibcWithoutMemfd:
        pass

    monkeypatch.setattr(
        converter.ctypes,
        "CDLL",
        lambda *args, **kwargs: LibcWithoutMemfd(),
    )

    with pytest.raises(RuntimeError, match="sealed memfd"):
        converter.write_episode_video(
            episode=plan.episodes[0],
            destination=destination,
            media_mode=plan.media_mode,
            output_fps=plan.output_fps,
            output_profile=plan.output_media,
        )

    assert not destination.exists()
    assert _private_media_directories(destination) == []


def test_transcode_rejects_private_artifact_swap_at_publication_boundary(tmp_path):
    source = make_source_episode(tmp_path / "source")
    replacement_source = make_source_episode(
        tmp_path / "replacement-source", frames=4, size="640x360"
    )
    replacement = _video(replacement_source)
    _overwrite_with_red_video(replacement, frames=4, size="640x360")
    plan = _preflight(source, tmp_path / "unused-output", skip=1, downsample=2)
    destination = tmp_path / "staging" / "episode.mp4"
    destination.parent.mkdir()
    replacement_bytes = replacement.read_bytes()

    def swap_private_artifact() -> None:
        private_directories = _private_media_directories(destination)
        assert len(private_directories) == 1
        artifact = private_directories[0] / "artifact.mp4"
        artifact.rename(private_directories[0] / "validated-original.mp4")
        replacement.rename(artifact)

    with pytest.raises(RuntimeError, match="identity changed"):
        converter.write_episode_video(
            episode=plan.episodes[0],
            destination=destination,
            media_mode=plan.media_mode,
            output_fps=plan.output_fps,
            output_profile=plan.output_media,
            before_media_publish=swap_private_artifact,
        )

    assert not destination.exists()
    private_directories = _private_media_directories(destination)
    assert len(private_directories) == 1
    assert (private_directories[0] / "artifact.mp4").read_bytes() == replacement_bytes
    assert (private_directories[0] / "validated-original.mp4").is_file()
    assert _video(source).exists()


def test_transcode_rejects_destination_parent_swap_with_hard_link(tmp_path):
    source = make_source_episode(tmp_path / "source")
    plan = _preflight(source, tmp_path / "unused-output", skip=1)
    destination = tmp_path / "staging" / "episode.mp4"
    destination.parent.mkdir()
    saved_parent = tmp_path / "saved-staging"

    def swap_parent_and_link_artifact() -> None:
        private_directories = _private_media_directories(destination)
        assert len(private_directories) == 1
        private_name = private_directories[0].name
        destination.parent.rename(saved_parent)
        destination.parent.mkdir()
        attacker_private = destination.parent / private_name
        attacker_private.mkdir(mode=0o700)
        os.link(
            saved_parent / private_name / "artifact.mp4",
            attacker_private / "artifact.mp4",
        )

    with pytest.raises(RuntimeError, match="identity changed|link count"):
        converter.write_episode_video(
            episode=plan.episodes[0],
            destination=destination,
            media_mode=plan.media_mode,
            output_fps=plan.output_fps,
            output_profile=plan.output_media,
            before_media_publish=swap_parent_and_link_artifact,
        )

    assert not destination.exists()
    saved_private = next(path for path in saved_parent.iterdir() if path.is_dir())
    assert (saved_private / "artifact.mp4").is_file()
    attacker_private = next(
        path for path in destination.parent.iterdir() if path.is_dir()
    )
    assert (attacker_private / "artifact.mp4").is_file()


@pytest.mark.parametrize("post_publication_fsync", [1, 2])
def test_transcode_rejects_in_place_output_mutation_at_each_durability_boundary(
    tmp_path, monkeypatch, post_publication_fsync
):
    source = make_source_episode(tmp_path / "source")
    plan = _preflight(source, tmp_path / "unused-output", skip=1)
    replacement_source = make_source_episode(
        tmp_path / "replacement-source", frames=7, size="640x360"
    )
    replacement = _video(replacement_source)
    _overwrite_with_red_video(replacement, frames=7, size="640x360")
    replacement_bytes = replacement.read_bytes()
    destination = tmp_path / "staging" / "episode.mp4"
    destination.parent.mkdir()
    actual_fsync_directory = converter.fsync_directory
    parent_fsync_count = 0

    def mutate_after_selected_fsync(
        path: Path, *, directory_fd: int | None = None
    ) -> None:
        nonlocal parent_fsync_count
        actual_fsync_directory(path, directory_fd=directory_fd)
        if path == destination.parent:
            parent_fsync_count += 1
            if parent_fsync_count == post_publication_fsync + 1:
                with destination.open("r+b") as stream:
                    stream.write(replacement_bytes)
                    stream.truncate()

    monkeypatch.setattr(converter, "fsync_directory", mutate_after_selected_fsync)
    with pytest.raises(RuntimeError, match="changed|differs"):
        converter.write_episode_video(
            episode=plan.episodes[0],
            destination=destination,
            media_mode=plan.media_mode,
            output_fps=plan.output_fps,
            output_profile=plan.output_media,
        )

    assert parent_fsync_count == post_publication_fsync + 1
    assert destination.read_bytes() == replacement_bytes
    assert _video(source).exists()


def test_transcode_success_atomically_publishes_and_removes_private_directory(
    tmp_path, monkeypatch
):
    source = make_source_episode(tmp_path / "source")
    plan = _preflight(source, tmp_path / "unused-output", skip=1)
    destination = tmp_path / "staging" / "episode.mp4"
    destination.parent.mkdir()
    actual_fsync_directory = converter.fsync_directory
    fsynced = []

    def recording_fsync(path: Path, *, directory_fd: int | None = None) -> None:
        fsynced.append(path)
        actual_fsync_directory(path, directory_fd=directory_fd)

    monkeypatch.setattr(converter, "fsync_directory", recording_fsync)
    identity = converter.write_episode_video(
        episode=plan.episodes[0],
        destination=destination,
        media_mode=plan.media_mode,
        output_fps=plan.output_fps,
        output_profile=plan.output_media,
    )

    assert destination.is_file()
    assert identity.frame_count == len(plan.episodes[0].retained_indices)
    assert converter.media_profile(identity) == plan.output_media
    assert _private_media_directories(destination) == []
    assert fsynced.count(destination.parent) >= 2
    assert any(path.name.startswith(f".{destination.name}.media-") for path in fsynced)


def test_ffmpeg_nonzero_exit_fails_and_preserves_private_directory(
    tmp_path, monkeypatch
):
    source = make_source_episode(tmp_path / "source")
    plan = _preflight(source, tmp_path / "unused-output", skip=1)
    destination = tmp_path / "staging" / "episode.mp4"
    destination.parent.mkdir()

    actual_run = converter.subprocess.run

    def fail_ffmpeg(argv, **kwargs):
        if argv[0] == "ffmpeg":
            return subprocess.CompletedProcess(argv, 19)
        return actual_run(argv, **kwargs)

    monkeypatch.setattr(converter.subprocess, "run", fail_ffmpeg)
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        converter.write_episode_video(
            episode=plan.episodes[0],
            destination=destination,
            media_mode=plan.media_mode,
            output_fps=plan.output_fps,
            output_profile=plan.output_media,
        )

    assert not destination.exists()
    assert len(_private_media_directories(destination)) == 1
    assert _video(source).exists()


@pytest.mark.parametrize(
    "drift",
    [
        {"frame_count": 1},
        {"pixel_format": "yuv444p"},
    ],
)
def test_output_frame_count_or_profile_drift_fails_before_publication(
    tmp_path, monkeypatch, drift
):
    source = make_source_episode(tmp_path / "source")
    plan = _preflight(source, tmp_path / "unused-output", skip=1)
    destination = tmp_path / "staging" / "episode.mp4"
    destination.parent.mkdir()
    actual_probe = converter.probe_media

    def drifting_probe(path: Path, **kwargs):
        identity = actual_probe(path, **kwargs)
        if path.name == "artifact.mp4":
            return dataclasses.replace(identity, **drift)
        return identity

    monkeypatch.setattr(converter, "probe_media", drifting_probe)
    with pytest.raises(RuntimeError, match="output media"):
        converter.write_episode_video(
            episode=plan.episodes[0],
            destination=destination,
            media_mode=plan.media_mode,
            output_fps=plan.output_fps,
            output_profile=plan.output_media,
        )

    assert not destination.exists()
    private_directories = _private_media_directories(destination)
    assert len(private_directories) == 1
    assert (private_directories[0] / "artifact.mp4").is_file()
