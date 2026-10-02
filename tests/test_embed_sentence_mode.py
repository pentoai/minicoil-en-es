"""Tests for sentence-pooling mining mode in the embed pipeline.

Covers:
  (a) sentence mode: one identical vector per (sentence, focal) pair when a
      sentence has multiple focals; vector comes from the fake ST model's
      encode() and size matches get_sentence_embedding_dimension().
  (b) token_pooled mode: encode_sentences_with_focals is still called (default
      path unchanged).
  (c) settings: mining_pooling defaults to "token_pooled"; invalid value raises.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from minicoil_v2.settings import EmbedSettings

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class FakeSentenceModel:
    """Fake ST model: returns distinct unit-norm vectors per sentence index."""

    DIM = 16

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 64,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        # One deterministic unit vector per sentence position; norm^2 = 1.0 >= 0.5.
        out = np.zeros((len(sentences), self.DIM), dtype=np.float32)
        for i in range(len(sentences)):
            out[i, i % self.DIM] = 1.0
        return out

    def get_sentence_embedding_dimension(self) -> int:
        return self.DIM


class FakeTokenizer:
    """Minimal fake tokenizer for token_pooled mode tests."""

    pass


class FakeTokenPoolModel:
    """Minimal stand-in so _flush_buffer type-checks in token_pooled mode."""

    class config:
        hidden_size = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_buffer_two_sentences() -> dict[str, dict]:
    """Buffer with sentence-A (2 focals) and sentence-B (1 focal)."""
    return {
        "The bank near the river": {
            "pairs": {"bank": "C-1", "river": "C-2"},
            "article_id": "art-1",
        },
        "A cat sat on the mat": {
            "pairs": {"cat": "C-3"},
            "article_id": "art-2",
        },
    }


# ---------------------------------------------------------------------------
# (a) sentence mode
# ---------------------------------------------------------------------------


def test_sentence_mode_identical_vector_per_focal(monkeypatch):
    """Each focal of the same sentence gets an identical vector from ST encode()."""
    import minicoil_v2.embed_wiki_sentences as em

    captured_points: list = []

    def fake_upsert(client, collection_name, points, batch_size, wait):
        captured_points.extend(points)
        return len(points)

    monkeypatch.setattr(em, "upsert_batch", fake_upsert)

    fake_model = FakeSentenceModel()
    buffer = _make_buffer_two_sentences()

    n, enc_s, up_s = em._flush_buffer(
        buffer=buffer,
        lang="en",
        tokenizer=None,
        model=fake_model,
        device=torch.device("cpu"),
        client=None,
        collection_name="test_col",
        encode_batch_size=64,
        upsert_batch_size=500,
        upsert_wait=False,
        mining_pooling="sentence",
    )

    assert n == 3  # 2 focals from sentence-A + 1 focal from sentence-B

    # Build a map: sentence -> list of mining vectors
    sent_to_vecs: dict[str, list[list[float]]] = {}
    for p in captured_points:
        s = p.payload["sentence"]
        sent_to_vecs.setdefault(s, []).append(p.vector["mining"])

    # sentence-A has 2 focals and both should have the same vector
    sentence_a = "The bank near the river"
    assert len(sent_to_vecs[sentence_a]) == 2
    vec0, vec1 = sent_to_vecs[sentence_a]
    assert vec0 == vec1, "Both focals of same sentence must share the same vector"

    # The vectors must have the right dimension
    assert len(vec0) == FakeSentenceModel.DIM

    # The vector should match what encode() returns for that sentence
    expected = fake_model.encode([sentence_a])[0]
    assert np.allclose(vec0, expected)


def test_sentence_mode_vector_size_from_model(monkeypatch):
    """Vector length stored equals get_sentence_embedding_dimension()."""
    import minicoil_v2.embed_wiki_sentences as em

    captured_points: list = []

    def fake_upsert(client, collection_name, points, batch_size, wait):
        captured_points.extend(points)
        return len(points)

    monkeypatch.setattr(em, "upsert_batch", fake_upsert)

    fake_model = FakeSentenceModel()
    buffer = _make_buffer_two_sentences()

    em._flush_buffer(
        buffer=buffer,
        lang="en",
        tokenizer=None,
        model=fake_model,
        device=torch.device("cpu"),
        client=None,
        collection_name="test_col",
        encode_batch_size=64,
        upsert_batch_size=500,
        upsert_wait=False,
        mining_pooling="sentence",
    )

    expected_dim = fake_model.get_sentence_embedding_dimension()
    for p in captured_points:
        assert len(p.vector["mining"]) == expected_dim


# ---------------------------------------------------------------------------
# (b) token_pooled mode still calls encode_sentences_with_focals
# ---------------------------------------------------------------------------


def test_token_pooled_mode_calls_encode_sentences_with_focals(monkeypatch):
    """Default token_pooled path delegates to encode_sentences_with_focals."""
    import minicoil_v2.embed_wiki_sentences as em

    esf_called: list[bool] = []
    captured_points: list = []

    def fake_encode_sentences_with_focals(
        sentences, focals_per_sentence, tokenizer, model, device, batch_size
    ):
        esf_called.append(True)
        # Return unit-norm nonzero vecs so they pass the norm check
        dim = FakeTokenPoolModel.config.hidden_size
        return [
            [np.full(dim, 1.0 / np.sqrt(dim), dtype=np.float32)] * len(focals)
            for focals in focals_per_sentence
        ]

    def fake_upsert(client, collection_name, points, batch_size, wait):
        captured_points.extend(points)
        return len(points)

    monkeypatch.setattr(em, "encode_sentences_with_focals", fake_encode_sentences_with_focals)
    monkeypatch.setattr(em, "upsert_batch", fake_upsert)

    buffer = _make_buffer_two_sentences()
    em._flush_buffer(
        buffer=buffer,
        lang="en",
        tokenizer=FakeTokenizer(),
        model=FakeTokenPoolModel(),
        device=torch.device("cpu"),
        client=None,
        collection_name="test_col",
        encode_batch_size=64,
        upsert_batch_size=500,
        upsert_wait=False,
        mining_pooling="token_pooled",
    )

    assert esf_called, "encode_sentences_with_focals must be called in token_pooled mode"
    assert captured_points, "upsert_batch must receive points for token_pooled mode"


# ---------------------------------------------------------------------------
# (b2) sentence model loading caps max_seq_length
# ---------------------------------------------------------------------------


def test_load_sentence_model_caps_max_seq_length(monkeypatch):
    """Qwen3-Embedding ships with max_seq_length=32k; unbounded batches of long
    Wikipedia sentences OOM a 24GB GPU. load_sentence_model must cap it."""
    import minicoil_v2.embed_wiki_sentences as em

    class FakeST:
        def __init__(self, model_name, device=None):
            self.model_name = model_name
            self.device = device
            self.max_seq_length = 32768

        def get_sentence_embedding_dimension(self):
            return 1024

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeST)

    model = em.load_sentence_model("Qwen/Qwen3-Embedding-0.6B", torch.device("cpu"))
    assert model.max_seq_length == em.MINING_SENTENCE_MAX_SEQ
    assert model.max_seq_length <= 512


def test_load_sentence_model_keeps_smaller_native_limit(monkeypatch):
    """A model with a native limit below the cap keeps its own limit."""
    import minicoil_v2.embed_wiki_sentences as em

    class FakeST:
        def __init__(self, model_name, device=None):
            self.max_seq_length = 256

        def get_sentence_embedding_dimension(self):
            return 384

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeST)

    model = em.load_sentence_model("intfloat/multilingual-e5-small", torch.device("cpu"))
    assert model.max_seq_length == 256


# ---------------------------------------------------------------------------
# (c) settings validation
# ---------------------------------------------------------------------------


def test_embed_settings_mining_pooling_default():
    """mining_pooling defaults to 'token_pooled'."""
    s = EmbedSettings()
    assert s.mining_pooling == "token_pooled"


def test_embed_settings_mining_pooling_sentence():
    """'sentence' is accepted."""
    s = EmbedSettings(mining_pooling="sentence")
    assert s.mining_pooling == "sentence"


def test_embed_settings_mining_pooling_invalid():
    """An invalid value raises ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EmbedSettings(mining_pooling="full_doc")
