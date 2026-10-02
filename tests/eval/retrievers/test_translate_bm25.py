import os

import pytest
from qdrant_client import QdrantClient


def test_translate_bm25_raises_on_monolingual_pair(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICOIL_QDRANT_META_DIR", str(tmp_path / "qdrant_meta"))
    shared_client = QdrantClient(path=str(tmp_path / "qdrant_storage"))
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._build_qdrant_client",
        lambda url: shared_client,
    )
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._nllb_revision_cache_path",
        lambda: tmp_path / "nllb_revision",
    )
    (tmp_path / "nllb_revision").write_text("test-sha\n", encoding="utf-8")
    # Avoid downloading FastEmbed Qdrant/bm25 when constructing the inner BM25Retriever
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common.QdrantSparseRetriever._build_encoder",
        lambda self: None,
    )

    import minicoil_v2.eval.retrievers.translate_bm25  # noqa: F401
    from minicoil_v2.eval.retriever import get
    from minicoil_v2.eval.retrievers._common import UnsupportedPairError

    cls = get("translate-bm25")
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
        r.evaluate(StubDataset(), pair="eng-eng", split="dev", k=10)


@pytest.mark.skipif(
    os.environ.get("MINICOIL_SKIP_FASTEMBED_TESTS") == "1",
    reason="set MINICOIL_SKIP_FASTEMBED_TESTS=1 to skip fastembed model download",
)
def test_translate_bm25_pretranslates_then_delegates_to_bm25(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICOIL_QDRANT_META_DIR", str(tmp_path / "qdrant_meta"))
    shared_client = QdrantClient(path=str(tmp_path / "qdrant_storage"))
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._build_qdrant_client",
        lambda url: shared_client,
    )
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._nllb_revision_cache_path",
        lambda: tmp_path / "nllb_revision",
    )
    (tmp_path / "nllb_revision").write_text("test-sha\n", encoding="utf-8")

    import minicoil_v2.eval.retrievers.bm25  # noqa: F401
    import minicoil_v2.eval.retrievers.translate_bm25  # noqa: F401
    from minicoil_v2.eval.retriever import get
    from minicoil_v2.eval.retrievers._common import Translator

    def stub_translate_batch(self, texts, src, tgt, batch_size=32):
        return [f"translated:{x}" for x in texts]

    monkeypatch.setattr(Translator, "translate_batch", stub_translate_batch)
    monkeypatch.setattr(Translator, "_lazy_load", lambda self: None)

    cls = get("translate-bm25")
    r = cls()

    class StubDataset:
        name = "stub"
        revision = "stub"

        def corpus(self, lang):
            if lang == "es":
                return {
                    "d_es_cat": "translated:cat translated:feline",
                    "d_es_dog": "translated:dog translated:canine",
                }
            return {"d_en_cat": "cat feline", "d_en_dog": "dog canine"}

        def queries(self, pair, split):
            return {"q1": "cat", "q2": "dog"}

        def qrels(self, pair, split):
            return {"q1": {"d_es_cat"}, "q2": {"d_es_dog"}}

    result = r.evaluate(StubDataset(), pair="eng-spa", split="dev", k=10)
    assert result.n_queries == 2
    assert result.metrics["ndcg10"]["mean"] > 0.0
