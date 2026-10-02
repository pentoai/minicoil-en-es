"""Tests for retrieval metrics."""

import pytest

from minicoil_v2.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_gold_first_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a"}, k=10) == pytest.approx(1.0, abs=1e-6)


def test_ndcg_gold_absent_is_zero():
    assert ndcg_at_k(["a", "b", "c"], {"z"}, k=10) == pytest.approx(0.0, abs=1e-6)


def test_ndcg_gold_past_k_is_zero():
    ranked = [f"d{i}" for i in range(20)]
    assert ndcg_at_k(ranked, {"d15"}, k=10) == pytest.approx(0.0, abs=1e-6)


def test_ndcg_single_relevant_at_rank_3():
    # rank 3 (1-indexed), binary relevance:
    # DCG = (2^1 - 1) / log2(3 + 1) = 1 / 2 = 0.5
    # IDCG (1 relevant doc) = 1 / log2(1 + 1) = 1.0
    # nDCG = 0.5
    assert ndcg_at_k(["a", "b", "c", "d"], {"c"}, k=10) == pytest.approx(0.5, abs=1e-6)


def test_ndcg_two_relevant():
    # Both relevant in top 2: DCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309... = 1.6309...
    # IDCG (2 relevant) = same = 1.6309...
    # nDCG = 1.0
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=10) == pytest.approx(1.0, abs=1e-6)


def test_recall_gold_first_is_one():
    assert recall_at_k(["a", "b", "c"], {"a"}, k=100) == pytest.approx(1.0, abs=1e-6)


def test_recall_gold_absent_is_zero():
    assert recall_at_k(["a", "b", "c"], {"z"}, k=100) == pytest.approx(0.0, abs=1e-6)


def test_recall_partial():
    # 2 of 4 relevant docs retrieved in top 10
    ranked = ["a", "b"] + [f"x{i}" for i in range(8)]
    assert recall_at_k(ranked, {"a", "b", "c", "d"}, k=10) == pytest.approx(0.5, abs=1e-6)


def test_recall_gold_past_k_is_zero():
    ranked = [f"d{i}" for i in range(20)]
    assert recall_at_k(ranked, {"d15"}, k=10) == pytest.approx(0.0, abs=1e-6)


def test_mrr_gold_first_is_one():
    assert mrr_at_k(["a", "b", "c"], {"a"}, k=10) == pytest.approx(1.0, abs=1e-6)


def test_mrr_gold_absent_is_zero():
    assert mrr_at_k(["a", "b", "c"], {"z"}, k=10) == pytest.approx(0.0, abs=1e-6)


def test_mrr_at_rank_3():
    # 1 / 3
    assert mrr_at_k(["a", "b", "c", "d"], {"c"}, k=10) == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_mrr_only_first_relevant_counts():
    # Two relevant docs at ranks 2 and 4 -> MRR = 1/2
    assert mrr_at_k(["a", "b", "c", "d"], {"b", "d"}, k=10) == pytest.approx(0.5, abs=1e-6)


def test_mrr_gold_past_k_is_zero():
    ranked = [f"d{i}" for i in range(20)]
    assert mrr_at_k(ranked, {"d15"}, k=10) == pytest.approx(0.0, abs=1e-6)


def test_metrics_empty_ranked_is_zero():
    assert ndcg_at_k([], {"a"}, k=10) == 0.0
    assert recall_at_k([], {"a"}, k=10) == 0.0
    assert mrr_at_k([], {"a"}, k=10) == 0.0


def test_metrics_empty_relevant_is_zero():
    assert ndcg_at_k(["a", "b"], set(), k=10) == 0.0
    assert recall_at_k(["a", "b"], set(), k=10) == 0.0
    assert mrr_at_k(["a", "b"], set(), k=10) == 0.0
