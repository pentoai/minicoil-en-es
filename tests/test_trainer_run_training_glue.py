"""End-to-end glue test for run_training with mining_dim(4096) != input_dim(384).

The byte-for-byte and smoke tests exercise the helpers directly; this exercises
the actual run_training wiring: the token-pooled re-encode call site, the
old_to_new index remapping after dropping unlocated rows, the concept_sent_langs
rebuild, and checkpoint save/merge. A fixture row whose sentence does not contain
the concept's surface forms forces a drop so the remap path is actually taken.
"""

import json

import pytest
import torch

from minicoil_v2 import qdrant_store
from minicoil_v2.constants import OUTPUT_DIM
from minicoil_v2.settings import TrainSettings
from minicoil_v2.train_concept_layers import run_training


@pytest.mark.slow
def test_run_training_decoupled_end_to_end(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    out = tmp_path / "out"
    json.dump(
        {"concepts": {"C-1": {"en": ["bank", "banks"], "es": ["banco"]}}},
        (data / "concept_vocabulary.json").open("w"),
    )
    json.dump(
        {"en": {"bank": "C-1", "banks": "C-1"}, "es": {"banco": "C-1"}},
        (data / "word_to_concept.json").open("w"),
    )

    # 12 locatable rows (6 en / 6 es) + 1 droppable row (no surface form of C-1) ->
    # 13 in, 12 after the drop, forcing the old_to_new remap to actually do work.
    en_sents = [f"the bank number {i} approved a loan" for i in range(6)]
    es_sents = [f"el banco numero {i} aprobo un prestamo" for i in range(6)]
    sentences = en_sents + es_sents + ["nothing concept here at all"]
    focals = ["bank"] * 6 + ["banco"] * 6 + ["ghost"]
    langs = ["en"] * 6 + ["es"] * 6 + ["en"]
    torch.manual_seed(0)
    mining = torch.randn(len(sentences), 4096)  # Qwen3 mining dim != 384 input

    def fake_scroll(client, collection, cid, max_per_lang=2000, lang_ratio=0.5):
        return sentences, focals, langs, mining

    monkeypatch.setattr(qdrant_store, "scroll_concept", fake_scroll)
    monkeypatch.setattr(qdrant_store, "get_client", lambda settings: object())

    settings = TrainSettings(
        data_dir=data,
        output_dir=out,
        concepts="C-1",
        epochs=1,
        train_epoch_size=64,
        val_epoch_size=32,
        sample_batch_size=8,
        trim_augment_ratio=0.0,
        wandb_enabled=False,
        device="cpu",
    )

    run_training(settings)

    merged = torch.load(out / "concept_layers.pt", map_location="cpu", weights_only=True)
    assert "C-1" in merged
    assert merged["C-1"]["weight"].shape == (OUTPUT_DIM, 384)
