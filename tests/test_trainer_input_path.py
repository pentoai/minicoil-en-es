"""The spec's correctness invariant: the trainer's input encoding must reproduce
MiniCoilEncoder's pre-Linear representation byte-for-byte (token-pool, not
sentence-pool). A mismatch silently trains one model and serves another."""

import json

import pytest
import torch

from minicoil_v2.constants import INPUT_ENCODER
from minicoil_v2.token_pooling import PASSAGE_PREFIX, load_token_pool_model
from minicoil_v2.utils import select_device


@pytest.mark.slow
def test_trainer_input_equals_encoder_representation(tmp_path):
    data = tmp_path / "data"
    (data / "concept_models").mkdir(parents=True)
    json.dump(
        {"en": {"bank": "C-1", "banks": "C-1"}, "es": {"banco": "C-1"}},
        (data / "word_to_concept.json").open("w"),
    )
    json.dump(
        {"concepts": {"C-1": {"en": ["bank", "banks"], "es": ["banco"]}}},
        (data / "concept_vocabulary.json").open("w"),
    )
    torch.save(
        {"C-1": {"weight": torch.zeros(4, 384), "bias": torch.zeros(4)}},
        data / "concept_models" / "concept_layers.pt",
    )

    from minicoil_v2.encoder import MiniCoilEncoder

    enc = MiniCoilEncoder(str(data), device="cpu")
    # sentence with TWO surface forms of the concept -> exercises all-occurrence avg
    sentence = "the bank near the river and other banks downtown"
    enc_vec = enc.encode_concept_vectors(sentence, lang="en")["C-1"]

    from minicoil_v2.train_concept_layers import encode_inputs_token_pooled

    device = select_device("cpu")
    tok, model = load_token_pool_model(INPUT_ENCODER, device)
    w2c = json.load((data / "word_to_concept.json").open())
    embs, keep = encode_inputs_token_pooled(
        model, tok, device, [sentence], ["C-1"], ["en"], w2c, prefix=PASSAGE_PREFIX
    )
    assert keep == [0]
    assert embs.shape == (1, 384)
    assert torch.allclose(embs[0], enc_vec, atol=1e-5)


@pytest.mark.slow
def test_trainer_input_drops_unlocated_concept(tmp_path):
    data = tmp_path / "data"
    (data / "concept_models").mkdir(parents=True)
    json.dump({"en": {"bank": "C-1"}, "es": {}}, (data / "word_to_concept.json").open("w"))

    from minicoil_v2.train_concept_layers import encode_inputs_token_pooled

    device = select_device("cpu")
    tok, model = load_token_pool_model(INPUT_ENCODER, device)
    w2c = json.load((data / "word_to_concept.json").open())
    # concept C-1 (surface form "bank") does not appear in the 2nd sentence ->
    # that row is dropped, not zero-filled. This is the realistic live case:
    # the row carries C-1 in concept_ids but C-1 falls outside the 256-token
    # window or only co-occurs with another concept here.
    embs, keep = encode_inputs_token_pooled(
        model,
        tok,
        device,
        ["a sentence with bank", "no concept here"],
        ["C-1", "C-1"],
        ["en", "en"],
        w2c,
    )
    assert keep == [0]
    assert embs.shape == (1, 384)
