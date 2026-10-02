"""Tests for the eval report writer."""

from __future__ import annotations

import json
from pathlib import Path

from minicoil_v2.eval.report import ReportMeta, write_report
from minicoil_v2.eval.retriever import EvalResult, PairResult


def _make_result() -> EvalResult:
    return EvalResult(
        retriever="fake-perfect",
        k=100,
        per_pair={
            "eng-eng": PairResult(
                pair="eng-eng",
                split="dev",
                n_queries=3,
                metrics={
                    "ndcg10": {"mean": 1.0, "std": 0.0, "per_query": {"q1": 1.0}},
                    "r100": {"mean": 1.0, "std": 0.0, "per_query": {"q1": 1.0}},
                    "mrr10": {"mean": 1.0, "std": 0.0, "per_query": {"q1": 1.0}},
                },
                runtime_sec=0.01,
                index_runtime_sec=0.001,
                rank_digest_ordered="a" * 64,
                rank_digest_sorted="b" * 64,
            ),
        },
    )


def _make_meta() -> ReportMeta:
    return ReportMeta(
        git_sha="abc1234567890",
        dirty=False,
        command="minicoil eval run fake-perfect",
        dataset_name="mlqa",
        dataset_revision="rev-x",
        splits_manifest_sha="m" * 64,
        timestamp_utc="2026-05-25T16:30:00+00:00",
    )


def test_writes_json_and_md(tmp_path: Path):
    result = _make_result()
    meta = _make_meta()
    paths = write_report(result, meta, reports_dir=tmp_path)
    assert paths.json_path.exists()
    assert paths.md_path.exists()
    assert paths.json_path.suffix == ".json"
    assert paths.md_path.suffix == ".md"


def test_filename_includes_retriever_and_sha(tmp_path: Path):
    result = _make_result()
    meta = _make_meta()
    paths = write_report(result, meta, reports_dir=tmp_path)
    name = paths.json_path.name
    assert "fake-perfect" in name
    assert meta.git_sha[:7] in name


def test_json_round_trip(tmp_path: Path):
    result = _make_result()
    meta = _make_meta()
    paths = write_report(result, meta, reports_dir=tmp_path)
    payload = json.loads(paths.json_path.read_text())
    assert payload["retriever"] == "fake-perfect"
    assert payload["git_sha"] == meta.git_sha
    assert payload["dirty"] is False
    assert payload["dataset"]["name"] == "mlqa"
    assert payload["dataset"]["revision"] == "rev-x"
    assert payload["k"] == 100
    assert payload["results"]["eng-eng"]["dev"]["ndcg10"]["mean"] == 1.0
    assert payload["results"]["eng-eng"]["dev"]["rank_digest_ordered"] == "a" * 64
    assert payload["results"]["eng-eng"]["dev"]["rank_digest_sorted"] == "b" * 64


def test_markdown_has_row_per_pair_and_headline_column(tmp_path: Path):
    result = _make_result()
    meta = _make_meta()
    paths = write_report(result, meta, reports_dir=tmp_path)
    md = paths.md_path.read_text()
    assert "eng-eng" in md
    assert "nDCG@10" in md
    assert "R@100" in md
    assert "MRR@10" in md


def test_markdown_no_delta_columns_when_no_baselines(tmp_path: Path):
    result = _make_result()
    meta = _make_meta()
    paths = write_report(result, meta, reports_dir=tmp_path, baselines=None)
    md = paths.md_path.read_text()
    assert "Δ" not in md


def test_markdown_has_delta_columns_when_baselines_given(tmp_path: Path):
    result = _make_result()
    meta = _make_meta()
    baselines = {
        "minicoil-v1": {
            "eng-eng": {
                "dev": {
                    "mrr10": {
                        "value": 0.5,
                        "report_path": "x",
                        "git_sha": "y",
                        "dataset_revision": "rev-x",
                        "splits_manifest_sha": "m" * 64,
                    },
                },
            },
        },
    }
    paths = write_report(result, meta, reports_dir=tmp_path, baselines=baselines)
    md = paths.md_path.read_text()
    assert "Δ vs minicoil-v1" in md
    # +0.5 because new (1.0) - baseline (0.5) = 0.5
    assert "↑" in md


def test_dirty_flag_marks_markdown_header(tmp_path: Path):
    result = _make_result()
    meta = _make_meta()
    meta_dirty = ReportMeta(**{**meta.__dict__, "dirty": True})
    paths = write_report(result, meta_dirty, reports_dir=tmp_path)
    md = paths.md_path.read_text()
    assert "⚠" in md  # dirty marker in header
    json_payload = json.loads(paths.json_path.read_text())
    assert json_payload["dirty"] is True
