"""Pure-function retrieval metrics: nDCG@k, Recall@k, MRR@k."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Binary-relevance nDCG@k.

    DCG numerator is `(2^rel - 1) / log2(rank + 1)` where rank is 1-indexed.
    IDCG is computed against the ideal ranking of all relevant docs (capped at k).
    """
    if not ranked or not relevant:
        return 0.0
    relevant_set = set(relevant)
    dcg = 0.0
    for rank_minus_1, doc_id in enumerate(ranked[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(rank_minus_1 + 2)  # +2 because rank is 1-indexed
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of relevant docs that appear in the top-k."""
    if not ranked or not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant_set)
    return hits / len(relevant_set)


def mrr_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Reciprocal of the rank (1-indexed) of the first relevant doc within top-k."""
    if not ranked or not relevant:
        return 0.0
    relevant_set = set(relevant)
    for rank_minus_1, doc_id in enumerate(ranked[:k]):
        if doc_id in relevant_set:
            return 1.0 / (rank_minus_1 + 1)
    return 0.0
