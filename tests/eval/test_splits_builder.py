"""Tests for the splits builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from minicoil_v2.eval.splits.builder import build_splits

PAIRS = ("eng-eng", "spa-spa", "eng-spa", "spa-eng")


def test_writes_three_splits_per_pair(synth_dataset, tmp_path: Path):
    build_splits(
        dataset=synth_dataset,
        output_dir=tmp_path,
        seed=0,
        dev_size=200,
        val_size=2000,
        pairs=PAIRS,
        dataset_revision="rev-x",
    )
    for pair in PAIRS:
        for split in ("dev", "val", "test"):
            f = tmp_path / f"synth_{pair}_{split}.json"
            assert f.exists()


def test_sizes_match_spec(synth_dataset, tmp_path: Path):
    build_splits(
        dataset=synth_dataset,
        output_dir=tmp_path,
        seed=0,
        dev_size=200,
        val_size=2000,
        pairs=PAIRS,
        dataset_revision="rev-x",
    )
    for pair in PAIRS:
        dev = json.loads((tmp_path / f"synth_{pair}_dev.json").read_text())
        val = json.loads((tmp_path / f"synth_{pair}_val.json").read_text())
        test = json.loads((tmp_path / f"synth_{pair}_test.json").read_text())
        assert len(dev) == 200
        assert len(val) == 2000
        assert len(test) == 5500 - 200 - 2000  # = 3300


def test_no_qid_in_multiple_splits(synth_dataset, tmp_path: Path):
    build_splits(
        dataset=synth_dataset,
        output_dir=tmp_path,
        seed=0,
        dev_size=200,
        val_size=2000,
        pairs=PAIRS,
        dataset_revision="rev-x",
    )
    for pair in PAIRS:
        dev = set(json.loads((tmp_path / f"synth_{pair}_dev.json").read_text()))
        val = set(json.loads((tmp_path / f"synth_{pair}_val.json").read_text()))
        test = set(json.loads((tmp_path / f"synth_{pair}_test.json").read_text()))
        assert dev.isdisjoint(val)
        assert dev.isdisjoint(test)
        assert val.isdisjoint(test)


def test_determinism_same_seed(synth_dataset, tmp_path: Path):
    build_splits(
        dataset=synth_dataset,
        output_dir=tmp_path / "a",
        seed=42,
        dev_size=200,
        val_size=2000,
        pairs=PAIRS,
        dataset_revision="rev-x",
    )
    build_splits(
        dataset=synth_dataset,
        output_dir=tmp_path / "b",
        seed=42,
        dev_size=200,
        val_size=2000,
        pairs=PAIRS,
        dataset_revision="rev-x",
    )
    for pair in PAIRS:
        for split in ("dev", "val", "test"):
            a = (tmp_path / "a" / f"synth_{pair}_{split}.json").read_bytes()
            b = (tmp_path / "b" / f"synth_{pair}_{split}.json").read_bytes()
            assert a == b


def test_different_seed_changes_split(synth_dataset, tmp_path: Path):
    build_splits(
        dataset=synth_dataset,
        output_dir=tmp_path / "a",
        seed=0,
        dev_size=200,
        val_size=2000,
        pairs=PAIRS,
        dataset_revision="rev-x",
    )
    build_splits(
        dataset=synth_dataset,
        output_dir=tmp_path / "b",
        seed=1,
        dev_size=200,
        val_size=2000,
        pairs=PAIRS,
        dataset_revision="rev-x",
    )
    dev_a = (tmp_path / "a" / "synth_eng-eng_dev.json").read_bytes()
    dev_b = (tmp_path / "b" / "synth_eng-eng_dev.json").read_bytes()
    assert dev_a != dev_b


def test_manifest_records_sha256_of_files(synth_dataset, tmp_path: Path):
    build_splits(
        dataset=synth_dataset,
        output_dir=tmp_path,
        seed=0,
        dev_size=200,
        val_size=2000,
        pairs=PAIRS,
        dataset_revision="rev-x",
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["seed"] == 0
    assert manifest["dataset_revision"] == "rev-x"
    assert manifest["dataset_name"] == "synth"
    for pair in PAIRS:
        for split in ("dev", "val", "test"):
            key = f"synth_{pair}_{split}.json"
            file_bytes = (tmp_path / key).read_bytes()
            expected = hashlib.sha256(file_bytes).hexdigest()
            assert manifest["files"][key]["sha256"] == expected


def test_refuses_overwrite_without_force(synth_dataset, tmp_path: Path):
    build_splits(
        dataset=synth_dataset,
        output_dir=tmp_path,
        seed=0,
        dev_size=200,
        val_size=2000,
        pairs=PAIRS,
        dataset_revision="rev-x",
    )
    with pytest.raises(FileExistsError):
        build_splits(
            dataset=synth_dataset,
            output_dir=tmp_path,
            seed=0,
            dev_size=200,
            val_size=2000,
            pairs=PAIRS,
            dataset_revision="rev-x",
        )


def test_force_overwrites(synth_dataset, tmp_path: Path):
    build_splits(
        dataset=synth_dataset,
        output_dir=tmp_path,
        seed=0,
        dev_size=200,
        val_size=2000,
        pairs=PAIRS,
        dataset_revision="rev-x",
    )
    # different seed -> different content, should succeed with force
    build_splits(
        dataset=synth_dataset,
        output_dir=tmp_path,
        seed=1,
        dev_size=200,
        val_size=2000,
        pairs=PAIRS,
        dataset_revision="rev-x",
        force=True,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["seed"] == 1
