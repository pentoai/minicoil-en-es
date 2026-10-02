"""Tests for the split-access audit log."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from minicoil_v2.eval.audit import log_test_access


def test_appends_line_to_audit_log(tmp_path: Path):
    log_dir = tmp_path / "reports" / "eval"
    log_test_access(
        retriever="fake-perfect",
        pair="eng-eng",
        git_sha="abc1234",
        caller="pytest",
        reports_dir=log_dir,
    )
    audit_path = log_dir / "audit.log"
    assert audit_path.exists()
    content = audit_path.read_text().strip()
    line = content.splitlines()[-1]
    parts = line.split("\t")
    assert len(parts) == 5
    # parts[0] is UTC timestamp; just check it parses-looking
    assert parts[1] == "fake-perfect"
    assert parts[2] == "eng-eng"
    assert parts[3] == "abc1234"
    assert parts[4] == "pytest"


def test_creates_reports_dir(tmp_path: Path):
    log_dir = tmp_path / "does" / "not" / "exist"
    log_test_access(
        retriever="r",
        pair="p",
        git_sha="s",
        caller="c",
        reports_dir=log_dir,
    )
    assert (log_dir / "audit.log").exists()


def test_emits_warning(tmp_path: Path):
    """Add a captured loguru sink; assert the warning message was emitted."""
    log_dir = tmp_path / "reports" / "eval"
    captured: list[str] = []
    handler_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    try:
        log_test_access(
            retriever="r",
            pair="p",
            git_sha="s",
            caller="c",
            reports_dir=log_dir,
        )
    finally:
        logger.remove(handler_id)
    assert any("TEST SPLIT ACCESS" in line for line in captured)
