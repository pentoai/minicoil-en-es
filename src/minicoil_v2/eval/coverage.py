"""Concept-coverage filter for the Phase-0 MMARCO slice.

A query is "covered" for a pair iff the query and its gold passage share a
*trained* (viable) concept, matched per language. This isolates exactly where a
per-concept head can influence the score: if the query carries no viable concept,
v2's query vector has no concept dimension to fire on, so such queries would only
dilute the slice.

Correctness invariant: this reuses ``concept_match.match_concepts`` — the SAME
matcher the inference encoder uses. "Covered" therefore provably means "the
encoder fires concept C here". No parallel regex.
"""

from __future__ import annotations

from collections.abc import Set

from minicoil_v2.concept_match import match_concepts
from minicoil_v2.eval.retriever import PAIR_LANGS


def query_covered(
    query: str,
    gold: str,
    query_lang: str,
    corpus_lang: str,
    word_to_concept: dict[str, dict[str, str]],
    viable: Set[str],
) -> bool:
    """True iff query (in ``query_lang``) and gold passage (in ``corpus_lang``)
    share a viable concept. Cross-lingual pairs share via the concept id: the
    query fires C via its query-lang surface form, the gold via its corpus-lang
    surface form (per the spec's per-pair table)."""
    q = set(match_concepts(query, query_lang, word_to_concept, viable))
    g = set(match_concepts(gold, corpus_lang, word_to_concept, viable))
    return bool(q & g)


def covered_qids(
    dataset,
    pair: str,
    word_to_concept: dict[str, dict[str, str]],
    viable: Set[str],
    split: str = "validation",
    gold_only: bool = False,
) -> list[str]:
    """Return the sorted qids covered for ``pair``.

    Default rule (``gold_only=False``): query AND gold share a viable concept.
    Relaxed rule (``gold_only=True``): the gold passage alone contains a viable
    concept — used only for pairs flagged underpowered after measuring counts.
    """
    query_lang, corpus_lang = PAIR_LANGS[pair]
    queries = dataset.queries(pair, split)
    qrels = dataset.qrels(pair, split)
    corpus = dataset.corpus(corpus_lang)

    out: list[str] = []
    for qid, query in queries.items():
        gold_ids = qrels.get(qid, set())
        gold_text = " ".join(corpus[d] for d in gold_ids if d in corpus)
        if not gold_text:
            continue
        if gold_only:
            covered = bool(match_concepts(gold_text, corpus_lang, word_to_concept, viable))
        else:
            covered = query_covered(
                query, gold_text, query_lang, corpus_lang, word_to_concept, viable
            )
        if covered:
            out.append(qid)
    return sorted(out)
