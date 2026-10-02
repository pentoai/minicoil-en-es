"""Tests for the MLQA dataset adapter.

These tests download MLQA from HuggingFace on first run (cached afterward).
"""

from __future__ import annotations

import pytest

from minicoil_v2.eval.datasets.mlqa import MLQADataset


@pytest.fixture(scope="session")
def mlqa() -> MLQADataset:
    return MLQADataset()


@pytest.mark.parametrize(
    "pair,split",
    [
        ("eng-eng", "validation"),
        ("spa-spa", "validation"),
        ("eng-spa", "validation"),
        ("spa-eng", "validation"),
    ],
)
def test_pair_queries_and_qrels_align(mlqa: MLQADataset, pair: str, split: str):
    queries = mlqa.queries(pair, split)
    qrels = mlqa.qrels(pair, split)
    assert set(queries) == set(qrels), "every query must have qrels"
    assert len(queries) > 0


@pytest.mark.parametrize(
    "pair,corpus_lang",
    [
        ("eng-eng", "en"),
        ("spa-spa", "es"),
        ("eng-spa", "es"),
        ("spa-eng", "en"),
    ],
)
def test_qrels_doc_ids_exist_in_corpus(mlqa: MLQADataset, pair: str, corpus_lang: str):
    qrels = mlqa.qrels(pair, "validation")
    corpus = mlqa.corpus(corpus_lang)
    corpus_ids = set(corpus.keys())
    for qid, gold_set in qrels.items():
        for gold_id in gold_set:
            assert gold_id in corpus_ids, (
                f"gold {gold_id} for {qid} missing from {corpus_lang} corpus"
            )


@pytest.mark.parametrize(
    "pair,query_lang_hint",
    [
        ("eng-eng", "the"),  # English stop word likely in many queries
        ("spa-spa", "el"),  # Spanish stop word
        ("eng-spa", "the"),
        ("spa-eng", "el"),
    ],
)
def test_queries_are_in_source_language(mlqa: MLQADataset, pair: str, query_lang_hint: str):
    queries = list(mlqa.queries(pair, "validation").values())
    # at least one query in the first 200 should contain the language-hint word
    assert any(query_lang_hint in q.lower() for q in queries[:200])


def test_corpus_doc_ids_stable_across_loads(mlqa: MLQADataset):
    """Two calls to corpus() return the same doc_id set."""
    a = set(mlqa.corpus("en"))
    b = set(mlqa.corpus("en"))
    assert a == b


def test_revision_is_recorded(mlqa: MLQADataset):
    """The loader exposes the HF dataset revision for the manifest."""
    rev = mlqa.revision
    assert isinstance(rev, str) and len(rev) > 0
