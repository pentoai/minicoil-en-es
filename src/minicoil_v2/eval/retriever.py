"""Retriever Protocol, registry, and BaseRetriever helper.

The Protocol is deliberately tiny so any retriever paradigm (sparse, dense,
hybrid, FastEmbed-wrapped v1, miniCOIL v2) can satisfy it. The framework
never reaches into a retriever's internals.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from statistics import mean, pstdev
from time import perf_counter
from typing import Protocol, TypeVar, runtime_checkable

from loguru import logger

from minicoil_v2.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k


@runtime_checkable
class Retriever(Protocol):
    """Minimal retriever contract."""

    name: str

    def index(self, corpus: Mapping[str, str], lang: str) -> None: ...

    def search(self, query: str, k: int) -> list[tuple[str, float]]: ...


PAIR_LANGS: dict[str, tuple[str, str]] = {
    "eng-eng": ("en", "en"),
    "spa-spa": ("es", "es"),
    "eng-spa": ("en", "es"),
    "spa-eng": ("es", "en"),
}


@dataclass
class PairResult:
    pair: str
    split: str
    n_queries: int
    metrics: dict[str, dict[str, float | dict[str, float]]] = field(default_factory=dict)
    runtime_sec: float = 0.0
    index_runtime_sec: float = 0.0
    # Determinism diagnostics: sha256 of the concatenated rank lists, one
    # variant ordered (catches reordering), one sorted (catches set drift).
    rank_digest_ordered: str = ""
    rank_digest_sorted: str = ""


@dataclass
class EvalResult:
    retriever: str
    k: int
    per_pair: dict[str, PairResult] = field(default_factory=dict)


def _aggregate(per_query: dict[str, float]) -> dict[str, float | dict[str, float]]:
    values = list(per_query.values())
    return {
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "per_query": per_query,
    }


class BaseRetriever:
    """Optional convenience base. Subclasses override `index` and `search`."""

    name: str = ""

    def index(self, corpus: Mapping[str, str], lang: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[tuple[str, float]]:  # pragma: no cover
        raise NotImplementedError

    def evaluate(
        self,
        dataset,  # Dataset Protocol
        pair: str,
        split: str,
        k: int = 100,
    ) -> PairResult:
        if pair not in PAIR_LANGS:
            raise ValueError(f"unknown pair: {pair!r}. allowed: {sorted(PAIR_LANGS)}")
        _, corpus_lang = PAIR_LANGS[pair]
        queries = dataset.queries(pair, split)
        qrels = dataset.qrels(pair, split)

        t0 = perf_counter()
        self.index(dataset.corpus(corpus_lang), corpus_lang)
        index_runtime = perf_counter() - t0

        ndcg: dict[str, float] = {}
        recall: dict[str, float] = {}
        mrr: dict[str, float] = {}
        ordered_hasher = hashlib.sha256()
        sorted_hasher = hashlib.sha256()

        n = len(queries)
        logger.info(f"{self.name}: searching {n} queries on {pair}/{split} (k={k})")
        t1 = perf_counter()
        for i, (qid, query) in enumerate(queries.items(), start=1):
            relevant = qrels.get(qid, set())
            ranked_with_scores = self.search(query, k=k)
            ranked = [doc_id for doc_id, _score in ranked_with_scores]
            ndcg[qid] = ndcg_at_k(ranked, relevant, k=10)
            recall[qid] = recall_at_k(ranked, relevant, k=100)
            mrr[qid] = mrr_at_k(ranked, relevant, k=10)
            ordered_hasher.update(qid.encode("utf-8"))
            ordered_hasher.update(b"\x1f")
            ordered_hasher.update("|".join(ranked).encode("utf-8"))
            ordered_hasher.update(b"\x1e")
            sorted_hasher.update(qid.encode("utf-8"))
            sorted_hasher.update(b"\x1f")
            sorted_hasher.update("|".join(sorted(ranked)).encode("utf-8"))
            sorted_hasher.update(b"\x1e")
            if i % 1000 == 0:
                rate = i / max(perf_counter() - t1, 1e-9)
                logger.info(f"  {self.name} {pair}: {i}/{n} searched ({rate:.0f}/s)")
        runtime = perf_counter() - t1

        ndcg_agg = _aggregate(ndcg)
        recall_agg = _aggregate(recall)
        mrr_agg = _aggregate(mrr)
        logger.info(
            f"{self.name} {pair}/{split} done: ndcg10={ndcg_agg['mean']:.4f} "
            f"r100={recall_agg['mean']:.4f} mrr10={mrr_agg['mean']:.4f} "
            f"(search {runtime:.1f}s, index {index_runtime:.1f}s)"
        )
        return PairResult(
            pair=pair,
            split=split,
            n_queries=len(queries),
            metrics={
                "ndcg10": ndcg_agg,
                "r100": recall_agg,
                "mrr10": mrr_agg,
            },
            runtime_sec=runtime,
            index_runtime_sec=index_runtime,
            rank_digest_ordered=ordered_hasher.hexdigest(),
            rank_digest_sorted=sorted_hasher.hexdigest(),
        )


# --- registry ---

_REGISTRY: dict[str, type[BaseRetriever]] = {}

R = TypeVar("R", bound=type[BaseRetriever])


def register(name: str) -> Callable[[R], R]:
    """Decorator: register a Retriever class under `name`."""

    def decorate(cls: R) -> R:
        if name in _REGISTRY:
            raise ValueError(f"retriever already registered under name: {name}")
        _REGISTRY[name] = cls
        return cls

    return decorate


def get(name: str) -> type[BaseRetriever]:
    """Return the registered class for `name`, or raise KeyError."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown retriever: {name!r}. registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_registered() -> list[str]:
    """Return sorted list of registered retriever names."""
    return sorted(_REGISTRY)
