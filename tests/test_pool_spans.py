import torch

from minicoil_v2.token_pooling import pool_spans


def test_pool_spans_matches_manual_concept_pooling():
    # 4 tokens, dim 3. offsets: special(0,0), word A spans token1, word B spans tokens2-3.
    hidden = torch.tensor([[9.0, 9.0, 9.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    offsets = [(0, 0), (0, 4), (5, 9), (5, 9)]
    out = pool_spans(hidden, offsets, [(0, 4), (5, 9)])
    occ1 = torch.tensor([1.0, 0.0, 0.0])
    occ2 = torch.tensor([0.0, 0.5, 0.5])
    occ2 = occ2 / occ2.norm()
    expected = (occ1 + occ2) / 2
    expected = expected / expected.norm()
    assert out is not None
    assert torch.allclose(out, expected, atol=1e-6)


def test_pool_spans_returns_none_when_no_tokens():
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    offsets = [(0, 0), (0, 0)]  # all special
    assert pool_spans(hidden, offsets, [(0, 4)]) is None


def test_pool_spans_single_occurrence_is_l2_normalized():
    hidden = torch.tensor([[3.0, 4.0]])
    offsets = [(0, 5)]
    out = pool_spans(hidden, offsets, [(0, 5)])
    assert out is not None
    assert torch.allclose(out, torch.tensor([0.6, 0.8]), atol=1e-6)
