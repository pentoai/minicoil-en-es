from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from minicoil_v2.eval.retrievers._common import QdrantSparseRetriever


class FakeSparseEmbedding:
    def __init__(self, indices, values):
        self.indices = np.array(indices, dtype=np.uint32)
        self.values = np.array(values, dtype=np.float32)


class FakeEncoder:
    """In-test stand-in for FastEmbed SparseTextEmbedding.

    Each call returns one nonzero feature per text, with feature index = hash(text) % 1000
    and value = 1.0. Deterministic, so the same corpus produces the same vectors.
    """

    def passage_embed(self, texts, **kwargs) -> Iterator[FakeSparseEmbedding]:
        # **kwargs mirrors the real FastEmbed signature (texts, **kwargs); the
        # index path passes batch_size= to bound pad-to-longest cost.
        for t in texts:
            yield FakeSparseEmbedding(indices=[hash(t) % 1000], values=[1.0])

    def query_embed(self, queries) -> Iterator[FakeSparseEmbedding]:
        yield from self.passage_embed(queries)


class _TestRetriever(QdrantSparseRetriever):
    name = "_test"
    collection_prefix = "_test"

    def _build_encoder(self):
        return FakeEncoder()


@pytest.fixture
def qdrant_url(tmp_path: Path) -> str:
    return str(tmp_path / "qdrant")


def test_index_creates_collection_and_writes_sidecar(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINICOIL_QDRANT_META_DIR", str(tmp_path / "qdrant_meta"))
    shared_client = QdrantClient(path=str(tmp_path / "qdrant_storage"))
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._build_qdrant_client",
        lambda url: shared_client,
    )

    corpus = {"d1": "hola mundo", "d2": "adios mundo", "d3": "buenas noches"}
    r = _TestRetriever()
    r.index(corpus, lang="es")

    sidecar = tmp_path / "qdrant_meta" / "_test_es_mlqa.json"
    assert sidecar.exists()
    import json

    meta = json.loads(sidecar.read_text())
    from minicoil_v2.eval.retrievers._common import sha256_corpus

    assert meta["corpus_sha"] == sha256_corpus(corpus)
    assert meta["n_docs"] == 3

    assert shared_client.collection_exists("_test_es_mlqa")
    count = shared_client.count(collection_name="_test_es_mlqa").count
    assert count == 3


def test_index_twice_with_same_corpus_skips_reupload(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINICOIL_QDRANT_META_DIR", str(tmp_path / "qdrant_meta"))
    shared_client = QdrantClient(path=str(tmp_path / "qdrant_storage"))
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._build_qdrant_client",
        lambda url: shared_client,
    )

    corpus = {"d1": "alpha", "d2": "beta"}
    r1 = _TestRetriever()
    r1.index(corpus, lang="en")

    call_count = {"n": 0}
    orig_passage_embed = FakeEncoder.passage_embed

    def counting_passage_embed(self, texts, **kwargs):
        call_count["n"] += 1
        return orig_passage_embed(self, texts, **kwargs)

    monkeypatch.setattr(FakeEncoder, "passage_embed", counting_passage_embed)

    r2 = _TestRetriever()
    r2.index(corpus, lang="en")

    assert call_count["n"] == 0, "should reuse, not re-encode"


def test_index_with_different_corpus_raises_mismatch(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINICOIL_QDRANT_META_DIR", str(tmp_path / "qdrant_meta"))
    shared_client = QdrantClient(path=str(tmp_path / "qdrant_storage"))
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._build_qdrant_client",
        lambda url: shared_client,
    )

    r1 = _TestRetriever()
    r1.index({"d1": "alpha"}, lang="en")

    from minicoil_v2.eval.retrievers._common import CorpusHashMismatch

    r2 = _TestRetriever()
    with pytest.raises(CorpusHashMismatch) as exc:
        r2.index({"d1": "alpha", "d2": "beta"}, lang="en")
    assert "_test_en_mlqa" in str(exc.value)
    assert "--rebuild" in str(exc.value)


def test_rebuild_overwrites_existing_collection(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINICOIL_QDRANT_META_DIR", str(tmp_path / "qdrant_meta"))
    shared_client = QdrantClient(path=str(tmp_path / "qdrant_storage"))
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._build_qdrant_client",
        lambda url: shared_client,
    )

    r1 = _TestRetriever()
    r1.index({"d1": "alpha"}, lang="en")

    r2 = _TestRetriever(rebuild=True)
    r2.index({"d1": "alpha", "d2": "beta"}, lang="en")

    assert shared_client.count(collection_name="_test_en_mlqa").count == 2


def test_search_returns_doc_id_score_pairs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINICOIL_QDRANT_META_DIR", str(tmp_path / "qdrant_meta"))
    shared_client = QdrantClient(path=str(tmp_path / "qdrant_storage"))
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._build_qdrant_client",
        lambda url: shared_client,
    )

    r = _TestRetriever()
    r.index({"d1": "x", "d2": "y", "d3": "z"}, lang="en")

    results = r.search("y", k=10)
    assert isinstance(results, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in results)
    assert all(isinstance(t[0], str) and isinstance(t[1], int | float) for t in results)
    assert len(results) <= 10
    assert results[0][0] == "d2"


def test_evaluate_raises_unsupported_pair_when_pair_not_in_supported_set(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("MINICOIL_QDRANT_META_DIR", str(tmp_path / "qdrant_meta"))
    shared_client = QdrantClient(path=str(tmp_path / "qdrant_storage"))
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._build_qdrant_client",
        lambda url: shared_client,
    )

    class _RestrictedRetriever(_TestRetriever):
        name = "_restricted"
        collection_prefix = "_restricted"
        supported_pairs = frozenset({"eng-eng"})

    from minicoil_v2.eval.retrievers._common import UnsupportedPairError

    class StubDataset:
        name = "stub"
        revision = "stub"

        def corpus(self, lang):
            return {"d1": "x"}

        def queries(self, pair, split):
            return {"q1": "y"}

        def qrels(self, pair, split):
            return {"q1": {"d1"}}

    r = _RestrictedRetriever()
    with pytest.raises(UnsupportedPairError):
        r.evaluate(StubDataset(), pair="spa-spa", split="dev", k=10)
