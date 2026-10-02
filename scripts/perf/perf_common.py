"""Shared helpers for the batched-training validation/benchmark scripts.

These scripts validate ``minicoil_v2.batched_training`` against the reference
path in ``minicoil_v2.train_concept_layers``. They are self-contained: they read
one encode-cache batch (``data/enc_cache_phase2/batch_*.pt``) and never touch
Qdrant, the eval gate, or the full training run.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from minicoil_v2.batched_training import ConceptSpec, prepare_concept
from minicoil_v2.settings import TrainSettings

DEFAULT_CACHE_DIR = "data/enc_cache_phase2"


def smallest_cache_file(cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """Smallest bucket. For FAST correctness/equivalence checks ONLY.

    Do NOT use for performance measurement: buckets range to ~2.5x this size, and
    the O(rows) / O(rows^2) phases (sampling, distance, argsort) are
    under-represented here. Perf work must use ``representative_cache_file``.
    """
    files = sorted(glob.glob(os.path.join(cache_dir, "batch_*.pt")), key=os.path.getsize)
    if not files:
        raise FileNotFoundError(f"no batch_*.pt under {cache_dir}")
    return files[0]


def representative_cache_file(cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """Largest bucket: the fixture for performance measurement.

    Real training buckets are full (~50 concepts, ~80k rows); the largest cache
    bucket matches production scale, so phase timings here reflect the real run.
    """
    files = sorted(glob.glob(os.path.join(cache_dir, "batch_*.pt")), key=os.path.getsize)
    if not files:
        raise FileNotFoundError(f"no batch_*.pt under {cache_dir}")
    return files[-1]


def load_cache(path: str | None = None) -> dict:
    path = path or smallest_cache_file()
    return torch.load(path, map_location="cpu", weights_only=False)


def make_settings(**overrides) -> TrainSettings:
    """A small, fast TrainSettings for validation (wandb off, tiny epochs)."""
    base = dict(
        dropout=0.0,
        lr=2e-3,
        train_epoch_size=512,
        val_epoch_size=256,
        sample_batch_size=256,
        margin_scale=0.0,
        val_size=0.2,
        epochs=8,
        lr_factor=0.5,
        lr_patience=2,
        wandb_enabled=False,
        trim_augment_ratio=0.0,
    )
    base.update(overrides)
    return TrainSettings(**base)


def concept_inputs(d: dict, cid: str, force_lang: str | None = None):
    """Raw (pre-prepare) per-concept tensors from the cache, as run_training slices them."""
    idxs = d["concept_indices"][cid]
    it = torch.tensor(idxs, dtype=torch.long)
    c_input = d["input_embs"][it].clone()
    c_mining = d["mining_embs_all"][it].clone()
    c_langs = list(d["concept_sent_langs"][cid])
    if force_lang is not None:
        c_langs = [force_lang] * len(c_langs)
    return c_input, c_mining, c_langs


def build_spec(d: dict, cid: str, settings: TrainSettings, force_lang: str | None = None):
    c_input, c_mining, c_langs = concept_inputs(d, cid, force_lang)
    return prepare_concept(cid, c_input, c_mining, c_langs, settings)


def pick_concepts(d: dict, n: int) -> list[str]:
    """Pick `n` concepts spread across the row-count range (small..large)."""
    items = sorted(d["concept_indices"], key=lambda k: len(d["concept_indices"][k]))
    if n >= len(items):
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


class FrozenSampler:
    """Feeds a fixed, pre-generated epoch to the reference ``train_concept_layer``.

    The reference trainer iterates ``train_sampler`` each epoch and reads
    ``train_sampler.input_embs``. This wrapper yields the SAME batches every epoch
    so both paths see identical triplets (isolating optimizer/forward from
    sampling).
    """

    def __init__(self, input_embs: torch.Tensor, epoch: dict[str, np.ndarray], batch_size: int):
        self.input_embs = input_embs
        self.epoch = epoch
        self.batch_size = batch_size

    def __iter__(self):
        n = len(self.epoch["anchor"])
        for start in range(0, n, self.batch_size):
            sl = slice(start, start + self.batch_size)
            yield {k: self.epoch[k][sl] for k in self.epoch}

    def log_epoch_stats(self, label: str = "") -> None:  # noqa: D102
        pass


def frozen_epoch(spec: ConceptSpec, epoch_size: int, range_kind: str, seed: int) -> dict:
    """Generate one frozen epoch (local indices) for a concept via the vectorized sampler."""
    from minicoil_v2.batched_training import VectorizedSampler

    rng = np.random.default_rng(seed)
    if range_kind == "train":
        rf, rt = 0, spec.train_to
    else:
        rf, rt = spec.train_to, spec.n
    s = VectorizedSampler(spec.distance_matrix, spec.langs, rf, rt, spec.min_margin, epoch_size)
    return s.sample_epoch(rng)
