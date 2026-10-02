"""Git SHA and dirty-tree helpers for reproducibility metadata in reports."""

from __future__ import annotations

import subprocess
from pathlib import Path


def current_sha(cwd: Path | None = None) -> str:
    """Return the current HEAD SHA. Raises RuntimeError if not in a git repo."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git rev-parse failed (cwd={cwd}): {result.stderr.strip()}")
    return result.stdout.strip()


def is_dirty(cwd: Path | None = None) -> bool:
    """True if the working tree has uncommitted changes or untracked files."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git status failed (cwd={cwd}): {result.stderr.strip()}")
    return bool(result.stdout.strip())
