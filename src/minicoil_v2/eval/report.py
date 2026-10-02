"""JSON + Markdown report writer for eval runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from minicoil_v2.eval.retriever import EvalResult

# mMARCO dev judgments are ~1 gold/query (mean 1.065), so MRR@10 — the official
# MS MARCO metric — is the headline; nDCG@10/r100 are still reported alongside.
HEADLINE_METRIC = "mrr10"


@dataclass(frozen=True)
class ReportMeta:
    git_sha: str
    dirty: bool
    command: str
    dataset_name: str
    dataset_revision: str
    splits_manifest_sha: str
    timestamp_utc: str  # ISO 8601


@dataclass(frozen=True)
class ReportPaths:
    json_path: Path
    md_path: Path


def _ts_for_filename(iso_ts: str) -> str:
    # 2026-05-25T16:30:00+00:00 -> 2026-05-25T163000Z
    parsed = datetime.fromisoformat(iso_ts)
    return parsed.strftime("%Y-%m-%dT%H%M%SZ")


def _build_json_payload(result: EvalResult, meta: ReportMeta) -> dict:
    return {
        "retriever": result.retriever,
        "k": result.k,
        "git_sha": meta.git_sha,
        "dirty": meta.dirty,
        "command": meta.command,
        "timestamp_utc": meta.timestamp_utc,
        "dataset": {"name": meta.dataset_name, "revision": meta.dataset_revision},
        "splits_manifest_sha": meta.splits_manifest_sha,
        "results": {
            pair: {
                pr.split: {
                    "n_queries": pr.n_queries,
                    "runtime_sec": pr.runtime_sec,
                    "index_runtime_sec": pr.index_runtime_sec,
                    "rank_digest_ordered": pr.rank_digest_ordered,
                    "rank_digest_sorted": pr.rank_digest_sorted,
                    **pr.metrics,
                },
            }
            for pair, pr in result.per_pair.items()
        },
    }


_METRIC_LABEL = {"ndcg10": "nDCG@10", "r100": "R@100", "mrr10": "MRR@10"}


def _build_markdown(
    result: EvalResult,
    meta: ReportMeta,
    baselines: dict | None,
) -> str:
    lines: list[str] = []
    header_warn = " ⚠ DIRTY TREE" if meta.dirty else ""
    lines.append(f"# Eval report: {result.retriever}{header_warn}")
    lines.append("")
    lines.append(f"- git_sha: `{meta.git_sha}`")
    lines.append(f"- dataset: `{meta.dataset_name}` (revision `{meta.dataset_revision}`)")
    lines.append(f"- splits_manifest_sha: `{meta.splits_manifest_sha[:16]}...`")
    lines.append(f"- k: {result.k}")
    lines.append(f"- timestamp_utc: {meta.timestamp_utc}")
    lines.append(f"- command: `{meta.command}`")
    lines.append("")

    for pair, pr in result.per_pair.items():
        lines.append(f"## {pair} {pr.split} (n={pr.n_queries})")
        lines.append("")
        # Build header row dynamically depending on whether deltas exist
        delta_baselines = set()
        if baselines is not None:
            for name, by_pair in baselines.items():
                if pr.metrics.get(HEADLINE_METRIC) is not None and (
                    by_pair.get(pair, {}).get(pr.split, {}).get(HEADLINE_METRIC) is not None
                ):
                    delta_baselines.add(name)
        delta_baselines_sorted = sorted(delta_baselines)
        delta_headers = "".join(f" | Δ vs {n}" for n in delta_baselines_sorted)
        lines.append(f"| metric | mean | std{delta_headers} |")
        lines.append("|--------|------|-----" + "|----" * len(delta_baselines_sorted) + "|")
        for metric_key in ("ndcg10", "r100", "mrr10"):
            agg = pr.metrics.get(metric_key)
            if agg is None:
                continue
            mean_val = agg["mean"]
            std_val = agg["std"]
            label = _METRIC_LABEL[metric_key]
            delta_cells = ""
            if metric_key == HEADLINE_METRIC and delta_baselines_sorted:
                for name in delta_baselines_sorted:
                    base_entry = baselines[name][pair][pr.split][HEADLINE_METRIC]
                    delta = mean_val - base_entry["value"]
                    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "·")
                    delta_cells += f" | {delta:+.4f} {arrow}"
            lines.append(f"| {label} | {mean_val:.4f} | {std_val:.4f}{delta_cells} |")
        lines.append("")
    return "\n".join(lines)


def write_report(
    result: EvalResult,
    meta: ReportMeta,
    *,
    reports_dir: Path,
    baselines: dict | None = None,
) -> ReportPaths:
    """Write JSON + Markdown report files. Returns the two paths."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = _ts_for_filename(meta.timestamp_utc)
    sha_short = meta.git_sha[:7]
    stem = f"{ts}__{result.retriever}__{sha_short}"
    json_path = reports_dir / f"{stem}.json"
    md_path = reports_dir / f"{stem}.md"

    payload = _build_json_payload(result, meta)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_build_markdown(result, meta, baselines), encoding="utf-8")
    return ReportPaths(json_path=json_path, md_path=md_path)
