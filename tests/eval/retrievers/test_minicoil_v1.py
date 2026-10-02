import pytest
from qdrant_client import QdrantClient


def test_minicoil_v1_raises_on_non_eng_eng_pair(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICOIL_QDRANT_META_DIR", str(tmp_path / "qdrant_meta"))
    shared_client = QdrantClient(path=str(tmp_path / "qdrant_storage"))
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._build_qdrant_client",
        lambda url: shared_client,
    )
    # Stub out the encoder factory so __init__ doesn't download Qdrant/minicoil-v1
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common.QdrantSparseRetriever._build_encoder",
        lambda self: None,
    )

    import minicoil_v2.eval.retrievers.minicoil_v1  # noqa: F401
    from minicoil_v2.eval.retriever import get
    from minicoil_v2.eval.retrievers._common import UnsupportedPairError

    cls = get("minicoil-v1")
    r = cls()

    class StubDataset:
        name = "stub"
        revision = "stub"

        def corpus(self, lang):
            return {"d1": "x"}

        def queries(self, pair, split):
            return {"q1": "y"}

        def qrels(self, pair, split):
            return {"q1": {"d1"}}

    with pytest.raises(UnsupportedPairError):
        r.evaluate(StubDataset(), pair="eng-spa", split="dev", k=10)
    with pytest.raises(UnsupportedPairError):
        r.evaluate(StubDataset(), pair="spa-spa", split="dev", k=10)


def test_minicoil_v1_supported_pairs_is_eng_eng_only():
    import minicoil_v2.eval.retrievers.minicoil_v1  # noqa: F401
    from minicoil_v2.eval.retriever import get

    cls = get("minicoil-v1")
    assert cls.supported_pairs == frozenset({"eng-eng"})
