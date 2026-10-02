"""Tests for the --dataset selector and the MMARCO covered-slice splits path."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from minicoil_v2.cli import app
from minicoil_v2.eval import cli as eval_cli

runner = CliRunner()


def test_covered_view_restricts_queries_and_qrels():
    class _Base:
        name = "mmarco"
        revision = "rev1"

        def corpus(self, lang):
            return {"d": "x"}

        def queries(self, pair, split):
            return {"a": "qa", "b": "qb", "c": "qc"}

        def qrels(self, pair, split):
            return {"a": {"d1"}, "b": {"d2"}, "c": {"d3"}}

    view = eval_cli._CoveredView(_Base(), {"eng-eng": {"a", "c"}})
    assert view.name == "mmarco"
    assert view.revision == "rev1"
    assert set(view.queries("eng-eng", "validation")) == {"a", "c"}
    assert set(view.qrels("eng-eng", "validation")) == {"a", "c"}
    assert view.corpus("en") == {"d": "x"}


def test_make_dataset_known_and_unknown():
    from minicoil_v2.eval.datasets.mmarco import MMARCODataset

    assert isinstance(eval_cli._make_dataset("mmarco"), MMARCODataset)
    import pytest
    import typer

    with pytest.raises(typer.BadParameter):
        eval_cli._make_dataset("nope")


class _FakeMMARCO:
    name = "mmarco"
    revision = "revX"
    _pairs = ("eng-eng", "spa-spa", "eng-spa", "spa-eng")

    def corpus(self, lang):
        return {f"d{i}": f"text {i}" for i in range(5)}

    def queries(self, pair, split):
        if split != "validation":
            return {}
        return {f"q{i}": f"query {i}" for i in range(20)}

    def qrels(self, pair, split):
        if split != "validation":
            return {}
        return {f"q{i}": {"d0"} for i in range(20)}


def test_splits_build_mmarco_carves_only_covered(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(eval_cli, "_make_dataset", lambda name: _FakeMMARCO())
    covered = {"pairs": {p: [f"q{i}" for i in range(10)] for p in _FakeMMARCO._pairs}}
    covered_path = tmp_path / "covered.json"
    covered_path.write_text(json.dumps(covered))
    out = tmp_path / "splits_mmarco"

    result = runner.invoke(
        app,
        [
            "eval",
            "splits",
            "build",
            "--dataset",
            "mmarco",
            "--dev-size",
            "3",
            "--val-size",
            "3",
            "--covered-qids",
            str(covered_path),
            "--output-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["dataset_name"] == "mmarco"
    assert manifest["dataset_revision"] == "revX"
    # 10 covered qids -> dev3 + val3 + test4
    dev = json.loads((out / "mmarco_eng-eng_dev.json").read_text())
    val = json.loads((out / "mmarco_eng-eng_val.json").read_text())
    test = json.loads((out / "mmarco_eng-eng_test.json").read_text())
    allq = set(dev) | set(val) | set(test)
    assert allq == {f"q{i}" for i in range(10)}
    assert len(dev) == 3 and len(val) == 3 and len(test) == 4


def test_splits_build_help_shows_dataset():
    result = runner.invoke(app, ["eval", "splits", "build", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.stdout
