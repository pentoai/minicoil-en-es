import json
from pathlib import Path

from qdrant_client import QdrantClient
from typer.testing import CliRunner

from minicoil_v2.cli import app


def test_cli_skips_unsupported_pair_with_warning(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINICOIL_QDRANT_META_DIR", str(tmp_path / "qdrant_meta"))
    shared_client = QdrantClient(path=str(tmp_path / "qdrant_storage"))
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._build_qdrant_client",
        lambda url: shared_client,
    )
    # Avoid downloading FastEmbed Qdrant/minicoil-v1 when constructing the retriever
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common.QdrantSparseRetriever._build_encoder",
        lambda self: None,
    )

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    from minicoil_v2.eval.datasets.mlqa import MLQADataset
    from minicoil_v2.eval.retriever import PAIR_LANGS
    from minicoil_v2.eval.splits.builder import build_splits

    ds = MLQADataset()
    build_splits(
        dataset=ds,
        output_dir=splits_dir,
        seed=0,
        dev_size=10,
        val_size=10,
        pairs=tuple(PAIR_LANGS),
        dataset_revision=ds.revision,
        force=True,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "minicoil-v1",
            "--pair",
            "spa-spa",
            "--split",
            "dev",
            "--splits-dir",
            str(splits_dir),
            "--reports-dir",
            str(tmp_path / "reports"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "skipping" in result.output.lower() or "unsupported" in result.output.lower()
    report_files = list((tmp_path / "reports").glob("*.json"))
    assert len(report_files) == 1
    payload = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert "spa-spa" not in payload["results"]
