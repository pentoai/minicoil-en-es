"""Speedup benchmark: reference per-concept training vs batched bucket.

NOTE: this is a MICRO benchmark and its ratio is OPTIMISTIC. It runs augmentation
off, can use small concepts, and uses val=train//8, so it under-counts the costs
that dominate a real bucket (sampling, the common augmentation re-encode, host
marshaling). For real-world performance use ``profile_full_bucket.py --check``,
which is the source of truth. See README.md for the full story.


Reports wall-clock, concepts/sec, and the projected full-vocab (~12k concept)
training time for the reference path and the batched path, at a fixed (bounded)
config. Also a sampler-only micro-benchmark at the production epoch size, which
isolates the single largest reference bottleneck (the per-sample Python loop).

Defaults run on CPU (deterministic; the GPU box is assumed busy). On CPU the win
comes from eliminating the per-sample Python loop and the per-concept Python
training loop; the additional kernel-launch win from one bmm-per-step is realized
on GPU (noted in the report, not measured here).

Run: ``uv run python scripts/perf/benchmark.py``
     ``uv run python scripts/perf/benchmark.py --k 16 --epochs 40 --device cpu``
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from perf_common import (  # noqa: E402
    build_spec,
    concept_inputs,
    load_cache,
    make_settings,
    pick_concepts,
)

from minicoil_v2.batched_training import (  # noqa: E402
    VectorizedSampler,
    train_concept_bucket,
)
from minicoil_v2.train_concept_layers import BilingualSampler, train_one_concept  # noqa: E402

FULL_VOCAB = 12000


def bench_reference(d, cids, st, device) -> float:
    t0 = time.time()
    for cid in cids:
        c_input, c_mining, c_langs = concept_inputs(d, cid)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        train_one_concept(c_input, c_mining, c_langs, st, device)
    return time.time() - t0


def bench_batched(d, cids, st, device, mode) -> float:
    # build_spec (distance-matrix build) is timed here because the reference
    # charges the equivalent one-time work inside train_one_concept.
    t0 = time.time()
    specs = [build_spec(d, c, st) for c in cids]
    train_concept_bucket(specs, st, device, mode, seed=0, epochs=st.epochs)
    return time.time() - t0


def bench_sampler(d, cids, st_epoch) -> tuple[float, float]:
    """One-epoch sampler cost at production epoch size, averaged over concepts."""
    old_t = 0.0
    new_t = 0.0
    for cid in cids:
        spec = build_spec(d, cid, make_settings())
        random.seed(0)
        np.random.seed(0)
        old = BilingualSampler(
            spec.input_embs,
            spec.distance_matrix,
            spec.langs,
            0,
            spec.train_to,
            min_margin=spec.min_margin,
            batch_size=256,
            epoch_size=st_epoch,
        )
        t0 = time.time()
        for _ in old:
            pass
        old_t += time.time() - t0

        new = VectorizedSampler(
            spec.distance_matrix, spec.langs, 0, spec.train_to, spec.min_margin, st_epoch
        )
        rng = np.random.default_rng(0)
        t0 = time.time()
        new.sample_epoch(rng)
        new_t += time.time() - t0
    return old_t / len(cids), new_t / len(cids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train-epoch-size", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    d = load_cache()
    cids = pick_concepts(d, args.k)
    st = make_settings(
        dropout=0.05,
        epochs=args.epochs,
        train_epoch_size=args.train_epoch_size,
        val_epoch_size=max(args.train_epoch_size // 8, 256),
        sample_batch_size=256,
        lr_patience=3,
    )
    print(
        f"device={device} K={len(cids)} epochs={st.epochs} train_epoch_size={st.train_epoch_size}"
    )

    print("\n=== training throughput (same K, same device) ===")
    t_ref = bench_reference(d, cids, st, device)
    t_bat = bench_batched(d, cids, st, device, "vectorized")
    for name, t in [
        ("reference (per-concept loop)", t_ref),
        ("batched (vectorized bucket)", t_bat),
    ]:
        cps = len(cids) / t
        proj_h = FULL_VOCAB / cps / 3600
        print(f"  {name:32s} {t:7.2f}s  {cps:6.3f} concepts/s  proj 12k: {proj_h:6.2f}h")
    print(f"  speedup: {t_ref / t_bat:.1f}x")

    print("\n=== sampler micro-benchmark (1 epoch @ production epoch_size=64000) ===")
    old_s, new_s = bench_sampler(d, cids, 64000)
    print(f"  reference BilingualSampler: {old_s * 1000:7.1f} ms/epoch")
    print(
        f"  VectorizedSampler:          {new_s * 1000:7.1f} ms/epoch  ({old_s / new_s:.0f}x faster)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
