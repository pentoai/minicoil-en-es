"""Faithful end-to-end decomposition of one 50-concept batched train batch.

Reproduces the real ``run_training`` train section for a single bucket from the
encode cache (no Qdrant): per-concept trim augmentation + re-encode (the model
forward that is COMMON to the reference and batched paths), then
``train_concept_bucket`` with the phase profiler on. Prints where the measured
~380s actually goes, so the synthetic-7x-vs-real-2x gap is attributed to real
phases instead of assumed.

Run on the same machine/device as the A/B run:
  MINICOIL_BUCKET_PROFILE=1 uv run python scripts/perf/profile_full_bucket.py --device mps
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from loguru import logger

sys.path.insert(0, os.path.dirname(__file__))
from perf_common import load_cache, representative_cache_file  # noqa: E402

import minicoil_v2.batched_training as bt  # noqa: E402
from minicoil_v2.batched_training import prepare_concept, train_concept_bucket  # noqa: E402
from minicoil_v2.settings import TrainSettings  # noqa: E402
from minicoil_v2.token_pooling import load_token_pool_model  # noqa: E402
from minicoil_v2.train_concept_layers import (  # noqa: E402
    _load_or_compute_aug,
    load_vocab,
)
from minicoil_v2.utils import select_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--train-epoch-size", type=int, default=2000)
    ap.add_argument("--val-epoch-size", type=int, default=6400)
    ap.add_argument("--margin-scale", type=float, default=2.0)
    ap.add_argument("--sample-batch-size", type=int, default=256)
    ap.add_argument("--mine-every", type=int, default=1)
    ap.add_argument("--trim-augment-ratio", type=float, default=1.0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument(
        "--cache-file", default=None, help="path to a batch_*.pt (default: representative/largest)"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="regression gate: fail if the benchmark is non-representative (toy bucket / "
        "augmentation off) or the one-time setup share exceeds its bound",
    )
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    device = select_device(args.device)
    settings = TrainSettings(
        data_dir=args.data_dir,
        epochs=args.epochs,
        train_epoch_size=args.train_epoch_size,
        val_epoch_size=args.val_epoch_size,
        margin_scale=args.margin_scale,
        sample_batch_size=args.sample_batch_size,
        mine_every=args.mine_every,
        trim_augment_ratio=args.trim_augment_ratio,
        lr_patience=3,
        lr_factor=0.5,
        wandb_enabled=False,
    )

    vocab = load_vocab(settings.data_dir)
    import json

    with open(settings.data_dir / "word_to_concept.json") as f:
        word_to_concept = json.load(f)

    tokenizer, model = load_token_pool_model("intfloat/multilingual-e5-small", device)
    d = load_cache(args.cache_file or representative_cache_file())

    input_embs = d["input_embs"]
    mining_embs_all = d["mining_embs_all"]
    unique_sentences = d["unique_sentences"]
    concept_indices = d["concept_indices"]
    concept_sent_langs = d["concept_sent_langs"]

    # --- Reproduce the train-loop augmentation + spec prep for the whole bucket ---
    # Uses the production _load_or_compute_aug path so the disk cache (A1) is
    # exercised: first run computes+caches (cache MISS), reruns hit it (~0s).
    concept_keys = list(concept_indices.keys())
    t0 = time.perf_counter()
    aug_by_cid = _load_or_compute_aug(
        settings,
        concept_keys,
        0,
        concept_indices,
        unique_sentences,
        concept_sent_langs,
        vocab,
        word_to_concept,
        model,
        tokenizer,
        device,
    )
    t_aug = time.perf_counter() - t0

    t_prepare = 0.0
    specs = []
    base_rows = 0
    aug_rows = 0
    for cid, idx_list in concept_indices.items():
        it = torch.tensor(idx_list, dtype=torch.long)
        c_input = input_embs[it]
        c_mining = mining_embs_all[it]
        c_langs = list(concept_sent_langs[cid])
        base_rows += len(idx_list)

        aug = aug_by_cid.get(cid)
        if aug is not None and aug["kept_src"]:
            kept_src = aug["kept_src"]
            src_t = torch.tensor(kept_src, dtype=torch.long)
            c_input = torch.cat([c_input, aug["aug_input"]], dim=0)
            c_mining = torch.cat([c_mining, c_mining[src_t]], dim=0)
            c_langs = c_langs + [c_langs[i] for i in kept_src]

        t0 = time.perf_counter()
        spec = prepare_concept(cid, c_input, c_mining, c_langs, settings)
        t_prepare += time.perf_counter() - t0
        if spec is not None:
            specs.append(spec)
            aug_rows += spec.n

    # --- Train the bucket with the phase profiler on ---
    bt.reset_profile()
    t0 = time.perf_counter()
    train_concept_bucket(specs, settings, device, "vectorized", seed=0)
    t_bucket = time.perf_counter() - t0

    prof = dict(bt.PROFILE)
    train_total = t_aug + t_prepare + t_bucket

    print("\n" + "=" * 64)
    print(f"device={device}  concepts={len(specs)}  epochs={settings.epochs}")
    print(f"base_rows={base_rows}  augmented_rows={aug_rows}")
    print("=" * 64)
    print(f"  augmentation re-encode (COMMON to both paths) : {t_aug:8.2f}s")
    print(f"  prepare_concept / distance build (one-time)   : {t_prepare:8.2f}s")
    print(f"  --- train_concept_bucket total                : {t_bucket:8.2f}s")
    for k in ("sampler_init", "sample_epoch", "host_marshal", "forward", "backward", "scheduler"):
        if k in prof:
            print(f"        {k:24s}: {prof[k]:8.2f}s")
    accounted = sum(prof.values())
    print(f"        {'(bucket unaccounted)':24s}: {t_bucket - accounted:8.2f}s")
    print("-" * 64)
    print(f"  TRAIN-SECTION TOTAL                           : {train_total:8.2f}s")
    print("=" * 64)

    # Latent-config-bug warning: val sampling silently exceeds train sampling when
    # --train-epoch-size is lowered without lowering val_epoch_size.
    if settings.val_epoch_size > settings.train_epoch_size:
        print(
            f"  WARNING: val_epoch_size ({settings.val_epoch_size}) > train_epoch_size "
            f"({settings.train_epoch_size}); validation does more work than training every "
            f"epoch (drives only the LR scheduler)."
        )

    if args.check:
        # Regression gate. The synthetic benchmark mispredicted real perf by being
        # non-representative three ways (toy bucket, tiny concepts, val=train//8).
        # These checks refuse to "pass" a benchmark that under-represents reality,
        # plus a floor on the one-time setup share (the originally-suspected cap).
        setup_share = (prof.get("sampler_init", 0.0) + t_prepare) / max(t_bucket, 1e-9)
        failures = []
        if aug_rows < 60000:
            failures.append(f"toy bucket: augmented_rows={aug_rows} < 60000 (use the largest)")
        if settings.trim_augment_ratio <= 0:
            failures.append("augmentation OFF: trim_augment_ratio<=0 hides the common re-encode")
        if setup_share > 0.20:
            failures.append(f"one-time setup share {setup_share:.0%} > 20% (setup regressed)")
        if failures:
            print("\nCHECK FAILED:")
            for f in failures:
                print(f"  - {f}")
            return 1
        print(f"\nCHECK PASSED (setup share {setup_share:.0%}, augmented_rows {aug_rows})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
