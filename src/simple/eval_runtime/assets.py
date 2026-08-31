from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path

from .canonical import canonical_json_bytes, sha256_bytes


class AssetProbeError(RuntimeError):
    """Raised when a sealed evaluation asset does not match its manifest."""


@dataclass(frozen=True, slots=True)
class AssetRequirement:
    kind: str
    logical_path: str
    sha256: str

    def __post_init__(self) -> None:
        relative = Path(self.logical_path)
        if not self.kind or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("ASSET_REQUIREMENT_PATH")
        if len(self.sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.sha256
        ):
            raise ValueError("ASSET_REQUIREMENT_SHA256")


@dataclass(frozen=True, slots=True)
class AssetProbeResult:
    schema_version: int
    episode_index: int
    requirement_count: int
    requirements_sha256: str


def requirements_sha256(requirements: tuple[AssetRequirement, ...]) -> str:
    ordered = sorted(
        (asdict(item) for item in requirements),
        key=lambda item: item["logical_path"].encode(),
    )
    return sha256_bytes(canonical_json_bytes(ordered))


def _probe_regular_file(root: Path, requirement: AssetRequirement) -> None:
    target = root.joinpath(requirement.logical_path)
    try:
        root_identity = root.resolve(strict=True)
        target_identity = target.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise AssetProbeError(
            f"ASSET_PROBE_MISSING:{requirement.logical_path}"
        ) from exc
    if root_identity not in target_identity.parents:
        raise AssetProbeError(f"ASSET_PROBE_ESCAPE:{requirement.logical_path}")

    current = root_identity
    for component in Path(requirement.logical_path).parts:
        current = current / component
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise AssetProbeError(
                f"ASSET_PROBE_MISSING:{requirement.logical_path}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise AssetProbeError(f"ASSET_PROBE_SYMLINK:{requirement.logical_path}")
    if not stat.S_ISREG(os.lstat(target_identity).st_mode):
        raise AssetProbeError(f"ASSET_PROBE_NOT_REGULAR:{requirement.logical_path}")

    digest = hashlib.sha256()
    with target_identity.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != requirement.sha256:
        raise AssetProbeError(f"ASSET_PROBE_HASH:{requirement.logical_path}")


def probe_episode_assets(
    root: Path, requirements: tuple[AssetRequirement, ...], *, episode_index: int
) -> AssetProbeResult:
    if type(episode_index) is not int or episode_index < 0:
        raise AssetProbeError("ASSET_PROBE_EPISODE")
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise AssetProbeError("ASSET_PROBE_ROOT")
    seen: set[str] = set()
    for requirement in requirements:
        if requirement.logical_path in seen:
            raise AssetProbeError(f"ASSET_PROBE_DUPLICATE:{requirement.logical_path}")
        seen.add(requirement.logical_path)
        _probe_regular_file(root, requirement)
    return AssetProbeResult(
        schema_version=1,
        episode_index=episode_index,
        requirement_count=len(requirements),
        requirements_sha256=requirements_sha256(requirements),
    )
