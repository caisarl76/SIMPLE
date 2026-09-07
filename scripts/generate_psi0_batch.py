#!/usr/bin/env python3
"""Run the validated BendPick generation → raw checks → PSI0 certification workflow."""

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from simple.output_paths import output_path  # noqa: E402


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def run(command, cwd, logs, name, *, env=None, timeout=3600):
    """Record each command and stop its whole process group on timeout/interruption."""
    write_json(logs / f"{name}-command.json", {"command": command, "cwd": str(cwd)})
    print(f"[{name}] running; log: {logs / (name + '.log')}", flush=True)
    started = time.monotonic()
    with (logs / f"{name}.log").open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            finally:
                # A descendant may ignore SIGINT after the direct child exits.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            raise
        finally:
            write_json(
                logs / f"{name}-exit.json",
                {
                    "exit_code": child.returncode,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
    if code:
        raise RuntimeError(f"{name} exited {code}; inspect {logs / (name + '.log')}")
    return (logs / f"{name}.log").read_text()


def check_gpu(gpu):
    selected = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    busy = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).splitlines()
    if not selected or selected in {line.strip() for line in busy}:
        raise RuntimeError(
            f"GPU {gpu} is unavailable or already has a compute workload"
        )
    return selected


def source_hashes(datasets):
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for dataset in datasets
        for path in sorted(dataset.rglob("*"))
        if path.is_file()
    }


def build_plan(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path(output_path("batches"))
    )
    parser.add_argument("--episodes-per-level", type=int, default=10)
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--task", default="simple/G1WholebodyBendPickMP-v0")
    parser.add_argument("--generator-root", type=Path, default=ROOT)
    parser.add_argument("--generator-python", type=Path)
    parser.add_argument(
        "--converter-python", type=Path, default=ROOT / ".venv/bin/python"
    )
    parser.add_argument("--psi0-root", type=Path, required=True)
    parser.add_argument("--psi0-commit", required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--raw-root", type=Path, help="Existing parent of dr0/dr1/dr2; skips generation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without writing or starting processes",
    )
    args = parser.parse_args(argv)
    if args.episodes_per_level <= 0 or args.timeout_seconds <= 0:
        parser.error("episode count and timeout must be positive")
    if len(set(args.levels)) != len(args.levels) or not set(args.levels) <= {0, 1, 2}:
        parser.error("levels must be unique values from 0, 1, 2")
    if not re.fullmatch(r"simple/[A-Za-z0-9_-]+", args.task):
        parser.error(
            "task must be a simple/ENV identifier without path or glob characters"
        )
    for key in ("output_root", "generator_root", "psi0_root", "raw_root"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    # Keep the virtualenv executable path: resolving its symlink loses the venv.
    args.generator_python = (
        (args.generator_python or args.generator_root / ".venv/bin/python")
        .expanduser()
        .absolute()
    )
    args.converter_python = args.converter_python.expanduser().absolute()
    if args.raw_root is not None and not args.raw_root.is_dir():
        parser.error("--raw-root must be an existing directory")
    levels = sorted(args.levels)
    run_root = args.output_root / (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    )
    raw = args.raw_root or run_root / "raw"
    output = run_root / "processed" / "bendpick-psi0"
    datasets = [raw / f"dr{level}" / args.task / f"level-{level}" for level in levels]
    pattern = str(
        raw / ("dr[" + "".join(map(str, levels)) + "]") / args.task / "level-*"
    )
    total = args.episodes_per_level * len(levels)
    base = [
        str(args.converter_python),
        "scripts/postprocess_psi0.py",
        "--sim-root",
        pattern,
        "--out-dir",
        str(output),
        "--skip",
        "60",
        "--downsample",
        "1",
        "--fps",
        "50",
        "--total-episodes",
        str(total),
        "--video-key",
        "observation.rgb_head_stereo_left",
        "--chunks-size",
        "1000",
    ]
    cert_base = [
        str(args.converter_python),
        "scripts/certify_psi0_dataset.py",
        str(output),
    ]
    commands = {}
    for level, dataset in zip(levels, datasets):
        if args.raw_root is None:
            commands[f"dr{level}-datagen"] = [
                str(args.generator_python),
                "-m",
                "simple.cli.datagen",
                args.task,
                "--sim-mode=mujoco_isaac",
                "--headless",
                "--no-webrtc",
                "--render-hz=50",
                f"--num-episodes={args.episodes_per_level}",
                f"--shard-size={args.episodes_per_level}",
                f"--dr-level={level}",
                f"--save-dir={raw / f'dr{level}'}",
            ]
        commands[f"dr{level}-validate"] = [
            str(args.converter_python),
            str(ROOT / "scripts/validate_simple_raw.py"),
            str(dataset),
            "--expected-episodes",
            str(args.episodes_per_level),
            "--report",
            str(run_root / "logs" / f"dr{level}-raw-verification.json"),
        ]
    commands["preflight"] = base + ["--preflight-only"]
    commands["conversion"] = base + [
        "--certify-psi0-root",
        str(args.psi0_root),
        "--certify-psi0-commit",
        args.psi0_commit,
        "--certify-python",
        str(args.converter_python),
    ]
    commands["tree-before"] = cert_base + ["--print-tree-digest"]
    commands["certification"] = cert_base + [
        "--psi0-root",
        str(args.psi0_root),
        "--psi0-commit",
        args.psi0_commit,
        "--python",
        str(args.converter_python),
        "--expected-video-key",
        "observation.images.egocentric",
    ]
    commands["tree-after"] = commands["tree-before"]
    return args, {
        "run_root": str(run_root),
        "raw_root": str(raw),
        "dataset": str(output),
        "source_datasets": list(map(str, datasets)),
        "source_pattern": pattern,
        "levels": levels,
        "episodes": total,
        "episodes_per_level": args.episodes_per_level,
        "task": args.task,
        "gpu": args.gpu,
        "raw_reused": args.raw_root is not None,
        "generator_root": str(args.generator_root),
        "converter_root": str(ROOT),
        "psi0_root": str(args.psi0_root),
        "psi0_commit": args.psi0_commit,
        "commands": commands,
    }


def execute(args, plan):
    if args.raw_root is None and os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        raise RuntimeError(
            "Generation requires your existing OMNI_KIT_ACCEPT_EULA=YES acceptance"
        )
    run_root = Path(plan["run_root"])
    run_root.mkdir(parents=True, exist_ok=False)
    args.created_run = True
    logs = run_root / "logs"
    logs.mkdir()
    output = Path(plan["dataset"])
    output.parent.mkdir()
    write_json(run_root / "plan.json", plan)
    print(f"Batch: {run_root}", flush=True)
    commands = plan["commands"]
    datasets = list(map(Path, plan["source_datasets"]))
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    provenance = {}
    for name, repo in (("converter", ROOT), ("generator", args.generator_root)):
        provenance[name] = {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "tracked_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
            ).hexdigest(),
        }
    provenance["workflow_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            ROOT / "scripts/validate_simple_raw.py",
            ROOT / "scripts/postprocess_psi0.py",
            ROOT / "scripts/certify_psi0_dataset.py",
        )
    }
    write_json(run_root / "provenance.json", provenance)
    reports = []
    for level in plan["levels"]:
        if args.raw_root is None:
            gpu_uuid = check_gpu(args.gpu)
            generation_env = {
                **env,
                "CUDA_VISIBLE_DEVICES": args.gpu,
                "PYTHONPATH": str(args.generator_root / "src"),
                "PYTHONUNBUFFERED": "1",
                "SIMPLE_DISABLE_TUI": "1",
            }
            write_json(
                logs / f"dr{level}-environment.json",
                {
                    "gpu_uuid": gpu_uuid,
                    **{
                        k: generation_env[k]
                        for k in (
                            "CUDA_VISIBLE_DEVICES",
                            "PYTHONPATH",
                            "OMNI_KIT_ACCEPT_EULA",
                        )
                    },
                },
            )
            run(
                commands[f"dr{level}-datagen"],
                args.generator_root,
                logs,
                f"dr{level}-datagen",
                env=generation_env,
                timeout=args.timeout_seconds,
            )
        run(
            commands[f"dr{level}-validate"],
            ROOT,
            logs,
            f"dr{level}-validate",
            env=env,
            timeout=args.timeout_seconds,
        )
        report = json.loads((logs / f"dr{level}-raw-verification.json").read_text())
        if not report["ok"] or report["actual_episodes"] != args.episodes_per_level:
            raise RuntimeError(f"Unexpected raw validation result for level {level}")
        reports.append(report)
    if {Path(p).resolve() for p in glob.glob(plan["source_pattern"])} != set(datasets):
        raise RuntimeError(
            "Source glob includes missing or unexpected level directories"
        )
    before = source_hashes(datasets)
    validated_hashes = {
        str(Path(report["raw_root"]) / relative): digest
        for report in reports
        for relative, digest in report["source_sha256"].items()
    }
    if before != validated_hashes:
        raise RuntimeError("Raw data changed after validation")
    write_json(logs / "raw-before-conversion.json", before)
    expected_frames = sum(report["retained_frames"] for report in reports)
    preflight = json.loads(
        run(
            commands["preflight"],
            ROOT,
            logs,
            "preflight",
            env=env,
            timeout=args.timeout_seconds,
        )
    )
    if (
        preflight["selected_episodes"] != plan["episodes"]
        or preflight["retained_frames"] != expected_frames
    ):
        raise RuntimeError("Preflight counts differ from validated raw data")
    if preflight["media_mode"] != "transcode_all":
        raise RuntimeError("Expected uniform H.264 transcode profile")
    run(
        commands["conversion"],
        ROOT,
        logs,
        "conversion",
        env=env,
        timeout=args.timeout_seconds,
    )
    tree_before = run(
        commands["tree-before"],
        ROOT,
        logs,
        "tree-before",
        env=env,
        timeout=args.timeout_seconds,
    ).strip()
    run(
        commands["certification"],
        ROOT,
        logs,
        "certification",
        env=env,
        timeout=args.timeout_seconds,
    )
    tree_after = run(
        commands["tree-after"],
        ROOT,
        logs,
        "tree-after",
        env=env,
        timeout=args.timeout_seconds,
    ).strip()
    if tree_before != tree_after or before != source_hashes(datasets):
        raise RuntimeError("Raw data or converted dataset changed during certification")
    from scripts.certify_psi0_dataset import validate_evidence_terminal

    manifest = hashlib.sha256(
        (output / "meta/conversion_manifest.json").read_bytes()
    ).hexdigest()
    certificates = sorted(
        output.parent.glob(f".{output.name}.certification-{manifest}-*")
    )
    if len(certificates) != 2:
        raise RuntimeError("Expected two certification bundles")
    for certificate in certificates:
        evidence = validate_evidence_terminal(certificate)
        loader = json.loads((certificate / "psi0-loader-result.json").read_text())
        if (
            evidence["verdict"] != "PASS"
            or not loader["all_finite"]
            or loader["visited_indices"] != list(range(expected_frames))
            or len(loader["episode_boundaries"]) != plan["episodes"]
        ):
            raise RuntimeError(f"Incomplete loader certification: {certificate}")
    info = json.loads((output / "meta/info.json").read_text())
    if (
        info["total_episodes"] != plan["episodes"]
        or info["total_frames"] != expected_frames
    ):
        raise RuntimeError("Published metadata counts differ")
    result = {
        "dataset": str(output),
        "episodes": plan["episodes"],
        "frames": expected_frames,
        "raw_frames": sum(report["total_frames"] for report in reports),
        "manifest_sha256": manifest,
        "tree_digest": tree_after,
        "raw_unchanged": True,
        "certificates": list(map(str, certificates)),
        "verdicts": ["PASS", "PASS"],
    }
    write_json(run_root / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


def main(argv=None):
    args, plan = build_plan(argv)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    try:
        return execute(args, plan)
    except (Exception, KeyboardInterrupt) as exc:
        root = Path(plan["run_root"])
        if getattr(args, "created_run", False) and not (root / "failure.json").exists():
            write_json(
                root / "failure.json", {"error": str(exc), "type": type(exc).__name__}
            )
        print(f"Batch failed: {exc}. Preserve artifacts at {root}.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
