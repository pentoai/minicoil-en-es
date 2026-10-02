"""Smoke test: train ONE concept end-to-end with mining_dim(4096) != input_dim(384).

This is the Phase-0 proof that the trainer is dim-decoupled — the Linear input is
384D while the distance matrix is built from a 4096D mining vector. No Qdrant.
"""

import pytest
import torch

from minicoil_v2.constants import OUTPUT_DIM
from minicoil_v2.settings import TrainSettings


@pytest.mark.slow
def test_train_one_concept_decoupled_dims():
    from minicoil_v2.train_concept_layers import train_one_concept

    torch.manual_seed(0)
    n = 40
    c_input = torch.randn(n, 384)  # mE5 input dim
    c_mining = torch.randn(n, 4096)  # Qwen3 mining dim (!= input)
    c_langs = ["en"] * 20 + ["es"] * 20
    settings = TrainSettings(
        epochs=1,
        train_epoch_size=128,
        val_epoch_size=64,
        sample_batch_size=16,
        wandb_enabled=False,
    )

    layer, train_loss, val_loss = train_one_concept(c_input, c_mining, c_langs, settings, "cpu")

    assert layer is not None
    assert layer.weight.shape == (OUTPUT_DIM, 384)
    assert torch.isfinite(torch.tensor(train_loss))
    assert torch.isfinite(torch.tensor(val_loss))
