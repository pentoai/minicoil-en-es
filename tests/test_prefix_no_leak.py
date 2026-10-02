"""The mE5 instruction prefixes must condition the encoder without firing concepts.

"passage" and "query" are both concept surface forms in the EN-ES vocabulary. When
concepts were matched over ``prefix + text``, every English document carried the
``passage`` block and every English query the ``query`` block, whatever the text
said. These tests build a three-concept model on disk (random weights are enough:
what matters is which blocks are emitted, not their values) and check that only
the text's own concept words produce blocks, on both sides, and that
``is_query=True`` selects the query prefix by default.
"""

import json

import pytest
import torch

from minicoil_v2.constants import OUTPUT_DIM


@pytest.fixture(scope="module")
def encoder(tmp_path_factory):
    data = tmp_path_factory.mktemp("prefix_leak_model")
    (data / "concept_models").mkdir()
    w2c = {
        "en": {"passage": "C-00001", "query": "C-00002", "dog": "C-00003"},
        "es": {"pasaje": "C-00001", "consulta": "C-00002", "perro": "C-00003"},
    }
    (data / "word_to_concept.json").write_text(json.dumps(w2c))
    torch.manual_seed(0)
    layers = {
        cid: {"weight": torch.randn(OUTPUT_DIM, 384)} for cid in ("C-00001", "C-00002", "C-00003")
    }
    torch.save(layers, data / "concept_models" / "concept_layers.pt")

    from minicoil_v2.encoder import MiniCoilEncoder

    return MiniCoilEncoder(str(data), device="cpu", lemma_match=False)


def _concept_blocks(sparse: dict[int, float]) -> set[int]:
    from minicoil_v2.encoder import BACKBONE_BASE

    return {i // OUTPUT_DIM for i in sparse if i < BACKBONE_BASE}


@pytest.mark.slow
@pytest.mark.parametrize("is_query", [False, True])
def test_prefix_words_do_not_fire_concepts(encoder, is_query):
    out = encoder.encode_sparse("the weather in reykjavik", lang="en", is_query=is_query)
    assert _concept_blocks(out) == set()


@pytest.mark.slow
@pytest.mark.parametrize("is_query", [False, True])
def test_text_concepts_still_fire(encoder, is_query):
    out = encoder.encode_sparse("my dog found a secret passage", lang="en", is_query=is_query)
    assert _concept_blocks(out) == {1, 3}  # passage and dog, from the text itself


@pytest.mark.slow
def test_concept_vectors_ignore_the_prefix(encoder):
    assert set(encoder.encode_concept_vectors("the weather in reykjavik", lang="en")) == set()
    assert set(encoder.encode_concept_vectors("a dog", lang="en")) == {"C-00003"}


@pytest.mark.slow
def test_is_query_defaults_to_query_prefix(encoder):
    from minicoil_v2.token_pooling import PASSAGE_PREFIX, QUERY_PREFIX

    text = "dog running"
    default = encoder.encode_sparse(text, lang="en", is_query=True)
    assert default == encoder.encode_sparse(text, lang="en", is_query=True, prefix=QUERY_PREFIX)
    assert default != encoder.encode_sparse(text, lang="en", is_query=True, prefix=PASSAGE_PREFIX)
    doc = encoder.encode_sparse(text, lang="en")
    assert doc == encoder.encode_sparse(text, lang="en", prefix=PASSAGE_PREFIX)
