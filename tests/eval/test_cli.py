"""Smoke tests for the eval CLI subapp via CliRunner."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from minicoil_v2.cli import app

runner = CliRunner()


def test_eval_help_works():
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "splits" in result.stdout
    assert "run" in result.stdout
    assert "baselines" in result.stdout


def test_run_unknown_retriever_lists_available(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MINICOIL_EVAL_RETRIEVER_MODULES", "tests.eval.fakes")
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "no-such-retriever",
            "--reports-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "no-such-retriever" in result.stdout or "no-such-retriever" in result.stderr


def test_splits_build_help_works():
    result = runner.invoke(app, ["eval", "splits", "build", "--help"])
    assert result.exit_code == 0
    assert "--seed" in result.stdout


def test_baselines_lock_help_works():
    result = runner.invoke(app, ["eval", "baselines", "lock", "--help"])
    assert result.exit_code == 0
    assert "--report" in result.stdout
    assert "--as" in result.stdout


def test_baselines_check_help_works():
    result = runner.invoke(app, ["eval", "baselines", "check", "--help"])
    assert result.exit_code == 0
    assert "--report" in result.stdout
