"""Tests for the Retriever Protocol + registry."""

from __future__ import annotations

import pytest

from minicoil_v2.eval.retriever import (
    _REGISTRY,
    BaseRetriever,
    Retriever,
    get,
    list_registered,
    register,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the registry around each test."""
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def test_register_and_get():
    @register("dummy-1")
    class Dummy(BaseRetriever):
        name = "dummy-1"

        def index(self, corpus):
            pass

        def search(self, query, k):
            return []

    assert get("dummy-1") is Dummy
    instance = get("dummy-1")()
    assert isinstance(instance, Retriever)  # Protocol check via runtime_checkable


def test_register_duplicate_raises():
    @register("dummy-dup")
    class A(BaseRetriever):
        name = "dummy-dup"

        def index(self, corpus):
            pass

        def search(self, query, k):
            return []

    with pytest.raises(ValueError, match="already registered"):

        @register("dummy-dup")
        class B(BaseRetriever):
            name = "dummy-dup"

            def index(self, corpus):
                pass

            def search(self, query, k):
                return []


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="unknown retriever"):
        get("does-not-exist")


def test_list_registered_includes_name():
    @register("dummy-list")
    class Dummy(BaseRetriever):
        name = "dummy-list"

        def index(self, corpus):
            pass

        def search(self, query, k):
            return []

    assert "dummy-list" in list_registered()


def test_evaluate_perfect_retriever(tiny_dataset):
    from tests.eval.fakes import FakePerfectRetriever

    r = FakePerfectRetriever()
    # inject qrels so FakePerfect can cheat
    r._qrels = {"cat": {"d1"}, "dog": {"d2"}, "bird": {"d3"}}
    result = r.evaluate(tiny_dataset, pair="eng-eng", split="dev", k=100)
    assert result.n_queries == 3
    assert result.metrics["ndcg10"]["mean"] == 1.0
    assert result.metrics["r100"]["mean"] == 1.0
    assert result.metrics["mrr10"]["mean"] == 1.0


def test_evaluate_reverse_retriever(tiny_dataset):
    from tests.eval.fakes import FakeReverseRetriever

    r = FakeReverseRetriever()
    r._qrels = {"cat": {"d1"}, "dog": {"d2"}, "bird": {"d3"}}
    result = r.evaluate(tiny_dataset, pair="eng-eng", split="dev", k=100)
    # gold is always last, so MRR < 1 and nDCG < 1
    assert result.metrics["mrr10"]["mean"] < 1.0
    # all 4 docs in corpus, k=100 covers everything, so R@100 = 1.0
    assert result.metrics["r100"]["mean"] == 1.0


def test_evaluate_random_retriever_deterministic(tiny_dataset):
    from tests.eval.fakes import FakeRandomRetriever

    a = FakeRandomRetriever(seed=42)
    b = FakeRandomRetriever(seed=42)
    ra = a.evaluate(tiny_dataset, pair="eng-eng", split="dev", k=100)
    rb = b.evaluate(tiny_dataset, pair="eng-eng", split="dev", k=100)
    assert ra.metrics["ndcg10"]["mean"] == rb.metrics["ndcg10"]["mean"]
    # ordered digest catches reorderings even when means agree
    assert ra.rank_digest_ordered == rb.rank_digest_ordered
    assert ra.rank_digest_sorted == rb.rank_digest_sorted


def test_evaluate_digest_differs_on_different_seeds(tiny_dataset):
    from tests.eval.fakes import FakeRandomRetriever

    a = FakeRandomRetriever(seed=0)
    b = FakeRandomRetriever(seed=1)
    ra = a.evaluate(tiny_dataset, pair="eng-eng", split="dev", k=100)
    rb = b.evaluate(tiny_dataset, pair="eng-eng", split="dev", k=100)
    assert ra.rank_digest_ordered != rb.rank_digest_ordered


def test_evaluate_unknown_pair_raises(tiny_dataset):
    from tests.eval.fakes import FakePerfectRetriever

    r = FakePerfectRetriever()
    with pytest.raises(ValueError, match="unknown pair"):
        r.evaluate(tiny_dataset, pair="zz-yy", split="dev", k=10)
