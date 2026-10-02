"""Tests for the splits loader (sha256 verification)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minicoil_v2.eval.splits.builder import build_splits
from minicoil_v2.eval.splits.loader import (
    SplitVerificationError,
    load_manifest,
    load_split,
    manifest_sha,
)


def _build(tmp_path: Path, synth_dataset) -> Path:
    build_splits(
        dataset=synth_dataset,
        output_dir=tmp_path,
        seed=0,
        dev_size=200,
        val_size=2000,
        pairs=("eng-eng",),
        dataset_revision="rev-x",
    )
    return tmp_path


def test_loads_qids_when_sha_matches(synth_dataset, tmp_path: Path):
    d = _build(tmp_path, synth_dataset)
    qids = load_split(d, dataset_name="synth", pair="eng-eng", split="dev")
    assert len(qids) == 200
    assert all(isinstance(q, str) for q in qids)


def test_raises_on_tampered_split_file(synth_dataset, tmp_path: Path):
    d = _build(tmp_path, synth_dataset)
    # tamper: append a qid to the dev file without updating the manifest sha
    f = d / "synth_eng-eng_dev.json"
    content = json.loads(f.read_text())
    content.append("intruder")
    f.write_text(json.dumps(content, separators=(",", ":")))
    with pytest.raises(SplitVerificationError, match="sha256 mismatch"):
        load_split(d, dataset_name="synth", pair="eng-eng", split="dev")


def test_raises_on_missing_manifest(synth_dataset, tmp_path: Path):
    d = _build(tmp_path, synth_dataset)
    (d / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_manifest(d)


def test_raises_on_missing_split_file(synth_dataset, tmp_path: Path):
    d = _build(tmp_path, synth_dataset)
    (d / "synth_eng-eng_dev.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_split(d, dataset_name="synth", pair="eng-eng", split="dev")


def test_manifest_sha_changes_when_manifest_changes(synth_dataset, tmp_path: Path):
    d = _build(tmp_path, synth_dataset)
    sha_before = manifest_sha(d)
    # rebuild with a different seed (force) -> manifest content changes
    build_splits(
        dataset=synth_dataset,
        output_dir=d,
        seed=1,
        dev_size=200,
        val_size=2000,
        pairs=("eng-eng",),
        dataset_revision="rev-x",
        force=True,
    )
    sha_after = manifest_sha(d)
    assert sha_before != sha_after
