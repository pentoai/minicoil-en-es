"""Audit trail for test-split access.

Every time the framework loads a `test` split, a line is appended here AND
a warning is logged. The trail is gitignored (local only) but exists so
test-set peeking is never silent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


def log_test_access(
    *,
    retriever: str,
    pair: str,
    git_sha: str,
    caller: str,
    reports_dir: Path,
) -> None:
    """Append a TSV line to `<reports_dir>/audit.log` and emit a warning."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit_path = reports_dir / "audit.log"
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    line = "\t".join([timestamp, retriever, pair, git_sha, caller])
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    logger.warning(
        "TEST SPLIT ACCESS: retriever={} pair={} git_sha={} caller={}",
        retriever,
        pair,
        git_sha,
        caller,
    )
