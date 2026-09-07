import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/generate_psi0_batch.py"
spec = importlib.util.spec_from_file_location("batch_command", SCRIPT)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def options(tmp_path):
    return [
        "--output-root",
        str(tmp_path / "batches"),
        "--psi0-root",
        str(tmp_path),
        "--psi0-commit",
        "abc",
    ]


def test_dry_run_is_write_free_and_records_exact_commands(tmp_path, capsys):
    assert batch.main([*options(tmp_path), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["episodes"] == 30
    assert not (tmp_path / "batches").exists()
    for level in (0, 1, 2):
        assert (
            f"--save-dir={plan['raw_root']}/dr{level}"
            in plan["commands"][f"dr{level}-datagen"]
        )
    assert "--certify-psi0-root" in plan["commands"]["conversion"]
    assert "--print-tree-digest" in plan["commands"]["tree-after"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--levels", "1", "1"],
        ["--levels", "-1"],
        ["--levels", "3"],
        ["--episodes-per-level", "0"],
        ["--task", "../escape"],
    ],
)
def test_rejects_invalid_arguments(tmp_path, extra):
    with pytest.raises(SystemExit):
        batch.build_plan([*options(tmp_path), *extra])
    assert not (tmp_path / "batches").exists()


def test_reused_raw_subset_excludes_other_levels(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    args, plan = batch.build_plan(
        [*options(tmp_path), "--raw-root", str(raw), "--levels", "2"]
    )
    assert args.raw_root == raw
    assert plan["episodes"] == 10
    assert not any(name.endswith("datagen") for name in plan["commands"])
    assert "dr[2]" in plan["source_pattern"]
    assert len(plan["source_datasets"]) == 1


def test_generation_does_not_accept_eula_implicitly(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    args, plan = batch.build_plan(options(tmp_path))
    with pytest.raises(RuntimeError, match="acceptance"):
        batch.execute(args, plan)
    assert not (tmp_path / "batches").exists()


def test_gpu_check_allows_other_gpu_workload(monkeypatch):
    def query(command, **kwargs):
        return "GPU-selected\n" if "--query-gpu=uuid" in command else "GPU-other\n"

    monkeypatch.setattr(subprocess, "check_output", query)
    assert batch.check_gpu("1") == "GPU-selected"
    monkeypatch.setattr(
        subprocess, "check_output", lambda *args, **kwargs: "GPU-selected\n"
    )
    with pytest.raises(RuntimeError, match="compute workload"):
        batch.check_gpu("1")


def test_failed_raw_check_stops_before_conversion_and_preserves_input(
    tmp_path, monkeypatch
):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "sentinel"
    source.write_text("preserve me")
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: "abc" if kwargs.get("text") else b"diff",
    )
    calls = []

    def fail(command, cwd, logs, name, **kwargs):
        calls.append(name)
        raise RuntimeError("raw validation failed")

    monkeypatch.setattr(batch, "run", fail)
    assert (
        batch.main([*options(tmp_path), "--raw-root", str(raw), "--levels", "0"]) == 1
    )
    assert calls == ["dr0-validate"]
    assert source.read_text() == "preserve me"
    run_root = next((tmp_path / "batches").iterdir())
    assert (run_root / "failure.json").exists()
    assert not (run_root / "result.json").exists()


def test_run_timeout_reaps_process_and_records_exit(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        batch.run(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            tmp_path,
            tmp_path,
            "timeout",
            timeout=0.1,
        )
    assert (
        json.loads((tmp_path / "timeout-exit.json").read_text())["exit_code"]
        is not None
    )
    with pytest.raises(FileExistsError):
        batch.run([sys.executable, "-c", "pass"], tmp_path, tmp_path, "timeout")


def test_existing_run_is_never_modified(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    args, plan = batch.build_plan([*options(tmp_path), "--raw-root", str(raw)])
    root = Path(plan["run_root"])
    root.mkdir(parents=True)
    (root / "sentinel").write_text("existing run")
    monkeypatch.setattr(batch, "build_plan", lambda argv: (args, plan))
    assert batch.main([]) == 1
    assert list(root.iterdir()) == [root / "sentinel"]


def test_timeout_also_stops_descendant_ignoring_sigint(tmp_path):
    child_pid = tmp_path / "descendant.pid"
    program = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); time.sleep(60)']); "
        f"open({str(child_pid)!r},'w').write(str(p.pid)); time.sleep(60)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        batch.run(
            [sys.executable, "-c", program],
            tmp_path,
            tmp_path,
            "descendant",
            timeout=0.5,
        )
    pid = int(child_pid.read_text())
    # An orphan may remain briefly as a zombie until init reaps it.
    stat = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 2
    while stat.exists() and stat.read_text().split(")", 1)[1].split()[0] != "Z":
        assert time.monotonic() < deadline, "descendant survived SIGKILL"
        time.sleep(0.01)
