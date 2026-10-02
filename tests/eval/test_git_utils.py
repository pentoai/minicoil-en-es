"""Tests for git_utils."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from minicoil_v2.eval.git_utils import current_sha, is_dirty


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=T",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "init",
        ],
        cwd=path,
        check=True,
    )


def test_current_sha_returns_hex(tmp_path: Path):
    _init_repo(tmp_path)
    sha = current_sha(cwd=tmp_path)
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_is_dirty_false_on_clean_tree(tmp_path: Path):
    _init_repo(tmp_path)
    assert is_dirty(cwd=tmp_path) is False


def test_is_dirty_true_with_untracked(tmp_path: Path):
    _init_repo(tmp_path)
    (tmp_path / "new.txt").write_text("hi")
    assert is_dirty(cwd=tmp_path) is True


def test_is_dirty_true_with_modified(tmp_path: Path):
    _init_repo(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("v1")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=T",
            "commit",
            "-q",
            "-m",
            "add a",
        ],
        cwd=tmp_path,
        check=True,
    )
    f.write_text("v2")
    assert is_dirty(cwd=tmp_path) is True


def test_current_sha_raises_outside_repo(tmp_path: Path):
    with pytest.raises(RuntimeError):
        current_sha(cwd=tmp_path)  # tmp_path is not a git repo
