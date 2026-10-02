"""End-to-end CLI test: `minicoil eval run` with a fake retriever on real MLQA."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from minicoil_v2.cli import app

runner = CliRunner()


@pytest.fixture
def splits_dir(tmp_path: Path) -> Path:
    """Build a tiny set of splits over MLQA for the e2e test."""
    out = tmp_path / "splits"
    result = runner.invoke(
        app,
        [
            "eval",
            "splits",
            "build",
            "--output-dir",
            str(out),
            "--dev-size",
            "10",
            "--val-size",
            "20",
            "--seed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.stdout
    return out


def test_e2e_dev_split_does_not_touch_audit_log(
    monkeypatch: pytest.MonkeyPatch,
    splits_dir: Path,
    tmp_path: Path,
):
    monkeypatch.setenv("MINICOIL_EVAL_RETRIEVER_MODULES", "tests.eval.fakes_e2e")
    reports_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "fake-perfect-e2e",
            "--pair",
            "eng-eng",
            "--split",
            "dev",
            "--reports-dir",
            str(reports_dir),
            "--splits-dir",
            str(splits_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    json_reports = list(reports_dir.glob("*.json"))
    assert len(json_reports) == 1
    payload = json.loads(json_reports[0].read_text())
    # FakePerfect always returns gold first; all metrics 1.0
    assert payload["results"]["eng-eng"]["dev"]["ndcg10"]["mean"] == 1.0
    # dev split must NOT have triggered audit
    assert not (reports_dir / "audit.log").exists()


def test_e2e_test_split_writes_audit_log(
    monkeypatch: pytest.MonkeyPatch,
    splits_dir: Path,
    tmp_path: Path,
):
    monkeypatch.setenv("MINICOIL_EVAL_RETRIEVER_MODULES", "tests.eval.fakes_e2e")
    monkeypatch.setenv("MINICOIL_EVAL_ALLOW_TEST", "1")  # sealed final-gate run
    reports_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "fake-perfect-e2e",
            "--pair",
            "eng-eng",
            "--split",
            "test",
            "--reports-dir",
            str(reports_dir),
            "--splits-dir",
            str(splits_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    audit_path = reports_dir / "audit.log"
    assert audit_path.exists()
    content = audit_path.read_text().strip()
    assert "fake-perfect-e2e" in content
    assert "eng-eng" in content
