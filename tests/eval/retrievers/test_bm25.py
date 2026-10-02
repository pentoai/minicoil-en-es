import os
from pathlib import Path

import pytest
from qdrant_client import QdrantClient


@pytest.mark.skipif(
    os.environ.get("MINICOIL_SKIP_FASTEMBED_TESTS") == "1",
    reason="set MINICOIL_SKIP_FASTEMBED_TESTS=1 to skip fastembed model download",
)
def test_bm25_retriever_registers_and_retrieves(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINICOIL_QDRANT_META_DIR", str(tmp_path / "qdrant_meta"))
    shared_client = QdrantClient(path=str(tmp_path / "qdrant_storage"))
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._build_qdrant_client",
        lambda url: shared_client,
    )

    import minicoil_v2.eval.retrievers.bm25  # noqa: F401
    from minicoil_v2.eval.retriever import get

    cls = get("bm25")
    r = cls()
    corpus = {
        "d_cat": "cats are small domesticated felines",
        "d_dog": "dogs are loyal canines kept as pets",
        "d_car": "cars are wheeled motor vehicles",
    }
    r.index(corpus, lang="en")
    results = r.search("feline pet", k=3)
    top_ids = [doc_id for doc_id, _ in results]
    assert "d_cat" in top_ids, f"expected d_cat in top results, got {top_ids}"
