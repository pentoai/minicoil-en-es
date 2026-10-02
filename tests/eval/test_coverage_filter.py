import pytest

from minicoil_v2.concept_match import match_concepts
from minicoil_v2.eval.coverage import covered_qids, query_covered

W2C = {"en": {"bank": "C-1", "river": "C-9"}, "es": {"banco": "C-1", "rio": "C-9"}}
VIABLE = {"C-1"}  # only the bank concept is "trained"


def test_eng_eng_shared_concept_is_covered():
    assert query_covered("a bank here", "the bank closed", "en", "en", W2C, VIABLE)


def test_cross_lingual_bank_banco_is_covered():
    # eng-spa: EN query has 'bank', ES gold has 'banco' -> shared C-1
    assert query_covered("the bank", "el banco grande", "en", "es", W2C, VIABLE)


def test_spa_eng_banco_bank_is_covered():
    assert query_covered("el banco", "the bank vault", "es", "en", W2C, VIABLE)


def test_only_gold_has_concept_is_not_covered():
    # query has no viable concept -> v2 query vector can't fire -> excluded
    assert not query_covered("the river", "el banco", "en", "es", W2C, VIABLE)


def test_only_query_has_concept_is_not_covered():
    assert not query_covered("the bank", "el rio", "en", "es", W2C, VIABLE)


def test_non_viable_shared_concept_is_not_covered():
    # river/rio is shared but C-9 is not in VIABLE
    assert not query_covered("the river", "el rio", "en", "es", W2C, {"C-1"})


class _StubDataset:
    name = "mmarco"

    def __init__(self, queries, qrels, corpus):
        self._q, self._qr, self._c = queries, qrels, corpus

    def queries(self, pair, split):
        return self._q[pair]

    def qrels(self, pair, split):
        return self._qr[pair]

    def corpus(self, lang):
        return self._c[lang]


def test_covered_qids_eng_spa():
    corpus = {"en": {}, "es": {"d1": "el banco grande", "d2": "el rio"}}
    ds = _StubDataset(
        queries={"eng-spa": {"q1": "the bank", "q2": "the river"}},
        qrels={"eng-spa": {"q1": {"d1"}, "q2": {"d2"}}},
        corpus=corpus,
    )
    # q1: EN bank + ES banco -> covered. q2: EN river not viable -> not covered.
    assert covered_qids(ds, "eng-spa", W2C, VIABLE) == ["q1"]


@pytest.mark.slow
def test_filter_matcher_agrees_with_encoder(tmp_path):
    """The encoder must fire exactly the concepts the filter matches (same matcher)."""
    import json

    import torch

    data = tmp_path / "data"
    (data / "concept_models").mkdir(parents=True)
    json.dump(
        {"en": {"bank": "C-1", "banks": "C-1"}, "es": {}},
        (data / "word_to_concept.json").open("w"),
    )
    torch.save(
        {"C-1": {"weight": torch.zeros(4, 384), "bias": torch.zeros(4)}},
        data / "concept_models" / "concept_layers.pt",
    )
    from minicoil_v2.encoder import MiniCoilEncoder
    from minicoil_v2.token_pooling import PASSAGE_PREFIX

    enc = MiniCoilEncoder(str(data), device="cpu")
    text = "the bank and other banks"  # short -> no truncation
    w2c = json.load((data / "word_to_concept.json").open())
    fired = set(enc.encode_concept_vectors(text, lang="en"))
    matched = set(match_concepts((PASSAGE_PREFIX + text).lower(), "en", w2c, enc._loaded_cids))
    assert fired == matched == {"C-1"}
