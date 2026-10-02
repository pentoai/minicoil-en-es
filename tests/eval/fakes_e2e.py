"""E2E-only fake retriever that primes a query-to-gold lookup from the Dataset.

Production retrievers never have access to qrels. This is a TEST helper
that overrides BaseRetriever.evaluate to prime an internal cache, so the
search() loop can return the gold doc first without modifying the
Protocol.
"""

from __future__ import annotations

from minicoil_v2.eval.retriever import BaseRetriever, register


@register("fake-perfect-e2e")
class FakePerfectE2E(BaseRetriever):
    name = "fake-perfect-e2e"

    def __init__(self) -> None:
        self._corpus_ids: list[str] = []
        self._gold_by_qtext: dict[str, set[str]] = {}

    def index(self, corpus, lang):  # noqa: ARG002
        self._corpus_ids = list(corpus.keys())

    def search(self, query, k):
        gold = self._gold_by_qtext.get(query, set())
        gold_first = [(g, 1.0) for g in gold if g in self._corpus_ids]
        filler = [(d, 0.5) for d in self._corpus_ids if d not in gold][
            : max(k - len(gold_first), 0)
        ]
        return (gold_first + filler)[:k]

    def evaluate(self, dataset, pair, split, k=100):
        for qid, q in dataset.queries(pair, split).items():
            self._gold_by_qtext[q] = set(dataset.qrels(pair, split).get(qid, set()))
        return super().evaluate(dataset, pair, split, k=k)
