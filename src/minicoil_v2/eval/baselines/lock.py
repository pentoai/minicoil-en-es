"""Lock baseline numbers from an eval report into data/eval/baselines.json."""

from __future__ import annotations

import json
from pathlib import Path

from minicoil_v2.eval.git_utils import is_dirty


class BaselineLockError(RuntimeError):
    """Raised when a baseline cannot be locked safely."""


def lock_baseline(
    *,
    report_path: Path,
    baseline_name: str,
    baselines_path: Path,
    current_splits_manifest_sha: str,
    allow_dirty: bool = False,
    cwd: Path | None = None,
) -> None:
    """Append/overwrite this baseline's entries in baselines.json.

    Refuses if the working tree is dirty (unless allow_dirty=True).
    Refuses if the report's splits_manifest_sha doesn't match the current one.
    """
    if not allow_dirty and is_dirty(cwd=cwd):
        raise BaselineLockError(
            "git tree is dirty; commit your changes or pass allow_dirty=True",
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_splits_sha = report.get("splits_manifest_sha")
    if report_splits_sha != current_splits_manifest_sha:
        raise BaselineLockError(
            "splits_manifest_sha in report does not match current splits on disk "
            f"(report={report_splits_sha[:12]}... "
            f"disk={current_splits_manifest_sha[:12]}...). "
            "Re-run the baseline against the current splits.",
        )

    if baselines_path.exists():
        store = json.loads(baselines_path.read_text(encoding="utf-8"))
    else:
        store = {}

    entry_for_name = store.setdefault(baseline_name, {})
    for pair, by_split in report.get("results", {}).items():
        pair_entry = entry_for_name.setdefault(pair, {})
        for split, metrics in by_split.items():
            split_entry = pair_entry.setdefault(split, {})
            for metric_key in ("ndcg10", "r100", "mrr10"):
                agg = metrics.get(metric_key)
                if agg is None:
                    continue
                split_entry[metric_key] = {
                    "value": agg["mean"],
                    "report_path": str(report_path),
                    "git_sha": report["git_sha"],
                    "dataset_revision": report["dataset"]["revision"],
                    "splits_manifest_sha": report_splits_sha,
                }

    baselines_path.parent.mkdir(parents=True, exist_ok=True)
    baselines_path.write_text(
        json.dumps(store, indent=2, sort_keys=True),
        encoding="utf-8",
    )
