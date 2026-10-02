"""Unit tests for the BM25 lexical backbone of the miniCOIL v2 sparse encoding.

The backbone reuses FastEmbed's `Qdrant/bm25` pipeline (tokenize -> filter ->
stem -> BM25 tf weight with document-length normalization) instead of a
hand-rolled BM15, so the backbone matches the locked bm25 baseline by
construction. `backbone_terms` duck-types on the Bm25 instance (tokenizer,
punctuation, stopwords, token_max_length, stemmer, k, b, avg_len), so these
fast tests use a lightweight stand-in; exact parity with the real FastEmbed
model is pinned by tests/test_backbone_parity.py (slow).

Index-range constraint: Qdrant sparse indices are int32-safe, and concept
indices are `concept_num*OUTPUT_DIM+offset` (~<10**5), so backbone indices must live in
a disjoint sub-range of [2**27, 2**31).
"""

from fastembed.common.utils import get_all_punctuation
from fastembed.sparse.utils.tokenizer import SimpleTokenizer

from minicoil_v2.encoder import BACKBONE_BASE, backbone_index, backbone_terms


class _UpperStemmer:
    """Toy stemmer: maps inflections food/foods -> food by stripping a trailing s."""

    def stem_word(self, word: str) -> str:
        return word.removesuffix("s")


class FakeBm25:
    """Stand-in exposing the attributes `backbone_terms` reads off fastembed's Bm25."""

    def __init__(self, k=1.2, b=0.75, avg_len=4.0, stopwords=(), stemmer=None):
        self.tokenizer = SimpleTokenizer
        self.punctuation = set(get_all_punctuation())
        self.stopwords = set(stopwords)
        self.token_max_length = 40
        self.stemmer = stemmer
        self.k = k
        self.b = b
        self.avg_len = avg_len


def _doc_weight(tf: int, doc_len: int, bm25: FakeBm25) -> float:
    return tf * (bm25.k + 1) / (tf + bm25.k * (1 - bm25.b + bm25.b * doc_len / bm25.avg_len))


def test_backbone_index_is_deterministic_and_int32_safe_disjoint():
    idx = backbone_index("reykjavik")
    assert BACKBONE_BASE <= idx < 2**31  # disjoint from concept space, int32-safe
    assert backbone_index("reykjavik") == idx  # process-stable (not salted hash())
    assert backbone_index("reykjavik") != backbone_index("barnabus")


def test_doc_weight_applies_length_normalization():
    bm25 = FakeBm25(avg_len=4.0)
    short = backbone_terms("foo", set(), bm25)
    long = backbone_terms("foo bar baz qux quux", set(), bm25)
    idx = backbone_index("foo")
    assert short[idx] == _doc_weight(1, 1, bm25)
    assert long[idx] == _doc_weight(1, 5, bm25)
    assert long[idx] < short[idx]  # longer doc -> smaller weight: the b-term works


def test_doc_weight_saturates_with_term_frequency():
    bm25 = FakeBm25()
    three = backbone_terms("foo foo foo", set(), bm25)
    assert three[backbone_index("foo")] == _doc_weight(3, 3, bm25)


def test_query_mode_emits_flat_ones():
    # fastembed's Bm25.query_embed hashes distinct tokens at weight 1.0; tf and
    # doc length play no role on the query side.
    bm25 = FakeBm25()
    terms = backbone_terms("foo foo foo bar", set(), bm25, is_query=True)
    assert terms == {backbone_index("foo"): 1.0, backbone_index("bar"): 1.0}


def test_concept_words_carved_but_still_count_toward_doc_len():
    # concept words are represented by their learned concept block, so they must NOT
    # also emit a backbone term (no double counting) — but the document is still
    # as long as it is: doc_len for length normalization includes them.
    bm25 = FakeBm25()
    terms = backbone_terms("cat sat reykjavik", {"cat"}, bm25)
    assert backbone_index("cat") not in terms
    assert backbone_index("reykjavik") in terms
    assert terms[backbone_index("sat")] == _doc_weight(1, 3, bm25)  # doc_len 3, not 2


def test_stopwords_and_punctuation_filtered():
    bm25 = FakeBm25(stopwords={"the"})
    terms = backbone_terms("the cat , !", set(), bm25)
    assert backbone_index("the") not in terms
    assert terms == {backbone_index("cat"): _doc_weight(1, 1, bm25)}  # doc_len excludes them


def test_stemmer_collapses_inflections_to_one_index():
    bm25 = FakeBm25(stemmer=_UpperStemmer())
    terms = backbone_terms("food foods", set(), bm25)
    assert terms == {backbone_index("food"): _doc_weight(2, 2, bm25)}


def test_carve_matches_on_surface_form_not_stem():
    # the carve is by the surface forms the concept matcher located, so an
    # unlisted inflection sharing the stem still contributes a backbone term.
    bm25 = FakeBm25(stemmer=_UpperStemmer())
    terms = backbone_terms("food foods", {"food"}, bm25)
    assert terms == {backbone_index("food"): _doc_weight(1, 2, bm25)}  # only "foods" left


def test_empty_when_every_token_is_a_concept_word():
    assert backbone_terms("cat dog", {"cat", "dog"}, FakeBm25()) == {}
