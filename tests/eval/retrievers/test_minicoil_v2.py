"""Unit tests for the minicoil-v2 retriever's encoder adapter + registration.

The adapter is the bug-prone glue: it must call MiniCoilEncoder with the right
language and prefix per direction, and turn the {index: value} sparse dict into
aligned indices/values arrays. Heavy end-to-end behavior is checked by a real
dev-split run, not here.
"""

import numpy as np


class FakeMiniCoil:
    """Stand-in for MiniCoilEncoder; records (texts, lang, prefix) per call."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def encode_batch_sparse(self, texts, lang, prefix, is_query=False, **kwargs):
        self.calls.append((tuple(texts), lang, prefix, is_query))
        # deliberately unsorted dict to prove the adapter sorts by index
        return [{7: 0.5, 3: -0.2, 11: 0.9} for _ in texts]


def test_adapter_passage_uses_doc_lang_and_passage_prefix():
    from minicoil_v2.eval.retrievers.minicoil_v2 import _MiniCoilAdapter

    fake = FakeMiniCoil()
    ad = _MiniCoilAdapter(fake, "passage: ", "query: ")
    ad.doc_lang = "es"
    out = list(ad.passage_embed(["hola"]))
    assert fake.calls[-1] == (("hola",), "es", "passage: ", False)
    assert out[0].indices.tolist() == [3, 7, 11]
    assert np.allclose(out[0].values.tolist(), [-0.2, 0.5, 0.9])


def test_adapter_query_uses_query_lang_and_query_prefix():
    # is_query=True switches the encoder to fastembed's query weighting
    # (flat 1.0 backbone terms, unit concept blocks).
    from minicoil_v2.eval.retrievers.minicoil_v2 import _MiniCoilAdapter

    fake = FakeMiniCoil()
    ad = _MiniCoilAdapter(fake, "passage: ", "query: ")
    ad.query_lang = "en"
    out = list(ad.query_embed(["what"]))
    assert fake.calls[-1] == (("what",), "en", "query: ", True)
    assert out[0].indices.tolist() == [3, 7, 11]


def test_minicoil_v2_registered_and_supports_all_pairs():
    import minicoil_v2.eval.retrievers.minicoil_v2  # noqa: F401
    from minicoil_v2.eval.retriever import get

    cls = get("minicoil-v2")
    assert cls.name == "minicoil-v2"
    assert cls.supported_pairs is None  # bilingual: all four pairs
