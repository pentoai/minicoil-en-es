"""Parity tests: the v2 backbone must reproduce FastEmbed's `Qdrant/bm25`
weights exactly (same tokenization, stopwords, stemming, k/b/avg_len), because
that is the implementation the locked baselines ran. With identical weights,
the only difference is the index remap into the backbone's disjoint range —
so backbone-vs-baseline comparisons are apples-to-apples by construction.

Slow: instantiating Bm25 fetches the stopword files from HF (cached after).
"""

import pytest

from minicoil_v2.encoder import (
    _BACKBONE_SPAN,
    BACKBONE_BASE,
    backbone_terms,
    load_backbone_bm25,
)


@pytest.fixture(scope="module")
def bm25():
    return load_backbone_bm25()


TEXT = "Reykjavik fishermen hauled the nets while the tides shifted under midnight sun"
TEXT_ES = "Los pescadores de Reykjavik recogieron las redes mientras cambiaban las mareas"


@pytest.mark.slow
@pytest.mark.parametrize("text", [TEXT, TEXT_ES])
def test_doc_terms_match_fastembed_exactly(bm25, text):
    ours = backbone_terms(text, set(), bm25)
    theirs = next(iter(bm25.raw_embed([text])))
    remapped = {
        BACKBONE_BASE + int(i) % _BACKBONE_SPAN: float(v)
        for i, v in zip(theirs.indices, theirs.values, strict=True)
    }
    assert ours == pytest.approx(remapped)


@pytest.mark.slow
def test_query_terms_match_fastembed_query_embed(bm25):
    ours = backbone_terms(TEXT, set(), bm25, is_query=True)
    theirs = next(iter(bm25.query_embed([TEXT])))
    assert set(ours) == {BACKBONE_BASE + int(i) % _BACKBONE_SPAN for i in theirs.indices}
    assert all(v == 1.0 for v in ours.values())


@pytest.mark.slow
def test_backbone_uses_baseline_defaults(bm25):
    # the locked baselines ran SparseTextEmbedding("Qdrant/bm25") defaults:
    # k=1.2, b=0.75, avg_len=256, english stemmer/stopwords for BOTH languages.
    # Parity holds only if the backbone replicates those exact settings.
    assert (bm25.k, bm25.b, bm25.avg_len) == (1.2, 0.75, 256.0)
    assert bm25.language == "english"
