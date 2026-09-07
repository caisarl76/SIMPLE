import ast
from pathlib import Path

from simple.output_paths import DEFAULT_OUTPUT_ROOT, output_path


def test_default_output_root(monkeypatch):
    monkeypatch.delenv("SIMPLE_OUTPUT_ROOT", raising=False)
    assert output_path("datagen") == f"{DEFAULT_OUTPUT_ROOT}/datagen"


def test_output_root_env_override(monkeypatch):
    monkeypatch.setenv("SIMPLE_OUTPUT_ROOT", "~/simple-output")
    assert output_path("evals") == str(Path("~/simple-output").expanduser() / "evals")


def test_cli_defaults_use_output_path():
    tree = ast.parse(open("src/simple/cli/datagen.py").read())
    assert any(
        isinstance(n, ast.Call) and getattr(n.func, "id", None) == "output_path"
        for n in ast.walk(tree)
    )


def test_evaluation_defaults_preserve_inputs_and_explicit_outputs(monkeypatch):
    from simple.evals.api import EvalConfig as APIConfig
    from simple.evals.env_runner import EvalConfig as RunnerConfig

    monkeypatch.setenv("SIMPLE_OUTPUT_ROOT", "/tmp/simple-test-output")
    for config_type in (APIConfig, RunnerConfig):
        config = (
            config_type(env_id="simple/Test-v0", policy="test")
            if config_type is APIConfig
            else config_type(env_id="simple/Test-v0")
        )
        assert config.eval_dir == "/tmp/simple-test-output/evals"
        assert config.data_dir == "data/datagen"
        if config_type is APIConfig:
            assert config.rollout_save_dir is None
        config = (
            config_type(env_id="simple/Test-v0", policy="test", eval_dir="/explicit")
            if config_type is APIConfig
            else config_type(env_id="simple/Test-v0", eval_dir="/explicit")
        )
        assert config.eval_dir == "/explicit"
