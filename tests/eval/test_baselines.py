"""Tests for baselines lock + check workflow."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from minicoil_v2.eval.baselines.lock import (
    BaselineLockError,
    lock_baseline,
)


def _init_repo(path: Path) -> None:
    """Initialize a git repo with a .gitignore that hides fixture-file noise.

    Tests write `report.json`, `baselines.json`, and `manifest.json` into the
    repo root; those would otherwise make `is_dirty()` return True and short
    circuit the lock checks we actually want to exercise. `junk.txt` stays
    *outside* the ignore list so the dirty-tree test can still trip the check.
    """
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    (path / ".gitignore").write_text(
        "report.json\nbaselines.json\nmanifest.json\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", ".gitignore"], cwd=path, check=True)
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
            "init",
        ],
        cwd=path,
        check=True,
    )


def _write_report(path: Path, splits_sha: str = "m" * 64) -> None:
    payload = {
        "retriever": "minicoil-v1",
        "k": 100,
        "git_sha": "abc1234",
        "dirty": False,
        "command": "minicoil eval run minicoil-v1",
        "timestamp_utc": "2026-05-25T16:30:00+00:00",
        "dataset": {"name": "mlqa", "revision": "rev-x"},
        "splits_manifest_sha": splits_sha,
        "results": {
            "eng-eng": {
                "test": {
                    "n_queries": 100,
                    "runtime_sec": 1.0,
                    "index_runtime_sec": 0.5,
                    "ndcg10": {"mean": 0.7517, "std": 0.1, "per_query": {}},
                    "r100": {"mean": 0.9, "std": 0.05, "per_query": {}},
                    "mrr10": {"mean": 0.8, "std": 0.05, "per_query": {}},
                },
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_lock_writes_baselines_json(tmp_path: Path):
    _init_repo(tmp_path)
    report = tmp_path / "report.json"
    _write_report(report, splits_sha="abc" * 21 + "f")  # 64 chars
    splits_manifest = tmp_path / "manifest.json"
    splits_manifest.write_text("")  # content irrelevant for hash check helper
    # patch the disk sha to match
    disk_sha = "abc" * 21 + "f"
    baselines_path = tmp_path / "baselines.json"
    lock_baseline(
        report_path=report,
        baseline_name="minicoil-v1",
        baselines_path=baselines_path,
        current_splits_manifest_sha=disk_sha,
        cwd=tmp_path,
    )
    data = json.loads(baselines_path.read_text())
    assert data["minicoil-v1"]["eng-eng"]["test"]["ndcg10"]["value"] == 0.7517


def test_lock_refuses_on_splits_sha_mismatch(tmp_path: Path):
    _init_repo(tmp_path)
    report = tmp_path / "report.json"
    _write_report(report, splits_sha="m" * 64)
    baselines_path = tmp_path / "baselines.json"
    with pytest.raises(BaselineLockError, match="splits_manifest_sha"):
        lock_baseline(
            report_path=report,
            baseline_name="minicoil-v1",
            baselines_path=baselines_path,
            current_splits_manifest_sha="x" * 64,
            cwd=tmp_path,
        )


def test_lock_refuses_on_dirty_tree(tmp_path: Path):
    _init_repo(tmp_path)
    (tmp_path / "junk.txt").write_text("uncommitted")
    report = tmp_path / "report.json"
    _write_report(report, splits_sha="m" * 64)
    baselines_path = tmp_path / "baselines.json"
    with pytest.raises(BaselineLockError, match="dirty"):
        lock_baseline(
            report_path=report,
            baseline_name="minicoil-v1",
            baselines_path=baselines_path,
            current_splits_manifest_sha="m" * 64,
            cwd=tmp_path,
        )


def test_lock_allows_dirty_with_flag(tmp_path: Path):
    _init_repo(tmp_path)
    (tmp_path / "junk.txt").write_text("uncommitted")
    report = tmp_path / "report.json"
    _write_report(report, splits_sha="m" * 64)
    baselines_path = tmp_path / "baselines.json"
    lock_baseline(
        report_path=report,
        baseline_name="minicoil-v1",
        baselines_path=baselines_path,
        current_splits_manifest_sha="m" * 64,
        allow_dirty=True,
        cwd=tmp_path,
    )
    assert baselines_path.exists()


def test_lock_overwrites_existing_entry(tmp_path: Path):
    _init_repo(tmp_path)
    report = tmp_path / "report.json"
    _write_report(report, splits_sha="m" * 64)
    baselines_path = tmp_path / "baselines.json"
    lock_baseline(
        report_path=report,
        baseline_name="minicoil-v1",
        baselines_path=baselines_path,
        current_splits_manifest_sha="m" * 64,
        cwd=tmp_path,
    )
    # Mutate the report and re-lock
    payload = json.loads(report.read_text())
    payload["results"]["eng-eng"]["test"]["ndcg10"]["mean"] = 0.8
    report.write_text(json.dumps(payload))
    lock_baseline(
        report_path=report,
        baseline_name="minicoil-v1",
        baselines_path=baselines_path,
        current_splits_manifest_sha="m" * 64,
        cwd=tmp_path,
    )
    data = json.loads(baselines_path.read_text())
    assert data["minicoil-v1"]["eng-eng"]["test"]["ndcg10"]["value"] == 0.8


from minicoil_v2.eval.baselines.check import (  # noqa: E402
    WIN_CONDITION,
    check_report,
)


def _new_report(
    eng_eng_mrr: float = 0.80,
    spa_spa_mrr: float = 0.65,
    eng_spa_mrr: float = 0.65,
    spa_eng_mrr: float = 0.60,
) -> dict:
    def block(v: float) -> dict:
        return {
            "test": {
                "n_queries": 100,
                "runtime_sec": 1.0,
                "index_runtime_sec": 0.5,
                "ndcg10": {"mean": 0.5, "std": 0.1, "per_query": {}},
                "r100": {"mean": 0.9, "std": 0.05, "per_query": {}},
                "mrr10": {"mean": v, "std": 0.05, "per_query": {}},
            },
        }

    return {
        "retriever": "v2-candidate",
        "k": 100,
        "git_sha": "abc1234",
        "dirty": False,
        "command": "x",
        "timestamp_utc": "2026-05-25T16:30:00+00:00",
        "dataset": {"name": "mlqa", "revision": "rev-x"},
        "splits_manifest_sha": "m" * 64,
        "results": {
            "eng-eng": block(eng_eng_mrr),
            "spa-spa": block(spa_spa_mrr),
            "eng-spa": block(eng_spa_mrr),
            "spa-eng": block(spa_eng_mrr),
        },
    }


def _baselines() -> dict:
    return {
        "minicoil-v1": {
            "eng-eng": {
                "test": {
                    "mrr10": {
                        "value": 0.7517,
                        "report_path": "p",
                        "git_sha": "s",
                        "dataset_revision": "rev-x",
                        "splits_manifest_sha": "m" * 64,
                    }
                }
            }
        },
        "bm25": {
            "spa-spa": {
                "test": {
                    "mrr10": {
                        "value": 0.6186,
                        "report_path": "p",
                        "git_sha": "s",
                        "dataset_revision": "rev-x",
                        "splits_manifest_sha": "m" * 64,
                    }
                }
            }
        },
        "translate-bm25": {
            "eng-spa": {
                "test": {
                    "mrr10": {
                        "value": 0.6387,
                        "report_path": "p",
                        "git_sha": "s",
                        "dataset_revision": "rev-x",
                        "splits_manifest_sha": "m" * 64,
                    }
                }
            },
            "spa-eng": {
                "test": {
                    "mrr10": {
                        "value": 0.5761,
                        "report_path": "p",
                        "git_sha": "s",
                        "dataset_revision": "rev-x",
                        "splits_manifest_sha": "m" * 64,
                    }
                }
            },
        },
    }


def test_check_all_wins_returns_zero():
    report = _new_report(0.80, 0.65, 0.65, 0.60)
    outcome = check_report(report=report, baselines=_baselines())
    assert outcome.passed is True
    assert outcome.exit_code == 0
    assert all(row.passed for row in outcome.rows)


def test_check_regression_returns_one():
    report = _new_report(0.70, 0.65, 0.65, 0.60)  # eng-eng regressed
    outcome = check_report(report=report, baselines=_baselines())
    assert outcome.passed is False
    assert outcome.exit_code == 1
    # find the failing row
    failing = [row for row in outcome.rows if not row.passed]
    assert len(failing) == 1
    assert failing[0].pair == "eng-eng"


def test_check_missing_baseline_treated_as_pass():
    report = _new_report(0.80, 0.65, 0.65, 0.60)
    partial = {"minicoil-v1": _baselines()["minicoil-v1"]}
    outcome = check_report(report=report, baselines=partial)
    assert outcome.passed is True
    # spa-spa, eng-spa, spa-eng all have "baseline missing"
    missing = [row for row in outcome.rows if row.baseline_value is None]
    assert {row.pair for row in missing} == {"spa-spa", "eng-spa", "spa-eng"}


def test_win_condition_matches_spec():
    assert WIN_CONDITION == {
        ("eng-eng", "test", "mrr10"): "minicoil-v1",
        ("spa-spa", "test", "mrr10"): "bm25",
        ("eng-spa", "test", "mrr10"): "translate-bm25",
        ("spa-eng", "test", "mrr10"): "translate-bm25",
    }
