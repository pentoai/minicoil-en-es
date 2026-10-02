"""Fake retrievers for testing AND canonical examples of the Protocol."""

from __future__ import annotations

import random
from collections.abc import Mapping

from minicoil_v2.eval.retriever import BaseRetriever, register


@register("fake-perfect")
class FakePerfectRetriever(BaseRetriever):
    """Always returns the gold doc first (then arbitrary filler).

    Cheats by inspecting the qrels via a `_qrels` attribute set on the
    instance before `search`. Tests inject it; production code never does.
    """

    name = "fake-perfect"

    def __init__(self) -> None:
        self._corpus_ids: list[str] = []
        self._qrels: dict[str, set[str]] = {}

    def index(self, corpus: Mapping[str, str], lang: str) -> None:  # noqa: ARG002
        self._corpus_ids = list(corpus.keys())

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        # caller (test harness) sets self._qrels[query] to the gold set.
        gold = self._qrels.get(query, set())
        gold_first = [(g, 1.0) for g in gold]
        filler = [(d, 0.5) for d in self._corpus_ids if d not in gold][: max(k - len(gold), 0)]
        return (gold_first + filler)[:k]


@register("fake-random")
class FakeRandomRetriever(BaseRetriever):
    """Random permutation of the corpus, deterministic per seed."""

    name = "fake-random"

    def __init__(self, seed: int = 0) -> None:
        self._corpus_ids: list[str] = []
        self._seed = seed

    def index(self, corpus: Mapping[str, str], lang: str) -> None:  # noqa: ARG002
        self._corpus_ids = list(corpus.keys())

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        rng = random.Random(f"{self._seed}::{query}")
        order = list(self._corpus_ids)
        rng.shuffle(order)
        return [(doc_id, 1.0 / (i + 1)) for i, doc_id in enumerate(order[:k])]


@register("fake-reverse")
class FakeReverseRetriever(BaseRetriever):
    """Gold last, everything else first (worst-case for nDCG/MRR)."""

    name = "fake-reverse"

    def __init__(self) -> None:
        self._corpus_ids: list[str] = []
        self._qrels: dict[str, set[str]] = {}

    def index(self, corpus: Mapping[str, str], lang: str) -> None:  # noqa: ARG002
        self._corpus_ids = list(corpus.keys())

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        gold = self._qrels.get(query, set())
        non_gold = [d for d in self._corpus_ids if d not in gold]
        gold_list = [g for g in self._corpus_ids if g in gold]
        ordered = non_gold + gold_list
        return [(d, 1.0 / (i + 1)) for i, d in enumerate(ordered[:k])]
