"""Profile the CPU (numpy) phases of the batched training path at realistic sizes.

Goal: settle whether the batched path's end-to-end 2x (vs the synthetic 7x) is
Amdahl-limited by the ONE-TIME per-concept setup (distance matrix + argsort init,
the stated hypothesis) or by PER-EPOCH sampling (``sample_epoch`` x epochs x 2),
or neither. Both phases are pure numpy and device-independent, so this measures
them faithfully on any machine; the GPU/MPS kernel is NOT measured here (it is the
remainder of the known per-batch train wall).

Realistic augmented row counts are reproduced by tiling each concept's real cache
rows by ``--aug-factor`` (the real trim augmentation expands rows ~2.3x: 76065
augmented vs 32200 pre-aug for a 48-concept bucket). Tiling preserves n and the
distance distribution, which is what the O(n^2)/O(n^2 log n) costs depend on.

Run:
  uv run python scripts/perf/profile_real_batch.py
  uv run python scripts/perf/profile_real_batch.py --aug-factor 2.3 --epochs 80
  uv run python scripts/perf/profile_real_batch.py --scaling   # n,2n,3n scaling
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from perf_common import load_cache, make_settings  # noqa: E402

from minicoil_v2.batched_training import VectorizedSampler, prepare_concept  # noqa: E402


def tile_rows(c_input, c_mining, c_langs, factor: float):
    """Tile a concept's rows to ~factor x to simulate trim augmentation."""
    n = c_input.shape[0]
    target = int(round(n * factor))
    if target <= n:
        return c_input, c_mining, c_langs
    reps = int(np.ceil(target / n))
    idx = np.tile(np.arange(n), reps)[:target]
    it = torch.as_tensor(idx, dtype=torch.long)
    return c_input[it].clone(), c_mining[it].clone(), [c_langs[i] for i in idx]


def time_call(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


def profile_concept(c_input, c_mining, c_langs, settings, train_epoch, val_epoch, n_epochs, n_reps):
    """Return per-phase seconds (one-time) and per-epoch sampling extrapolated x n_epochs."""
    # One-time: distance matrix build (inside prepare_concept).
    spec, t_distance = time_call(prepare_concept, "C", c_input, c_mining, c_langs, settings)
    if spec is None:
        return None

    # One-time: sampler init (argsort) for train and val ranges.
    tr_sampler, t_init_tr = time_call(
        VectorizedSampler,
        spec.distance_matrix,
        spec.langs,
        0,
        spec.train_to,
        spec.min_margin,
        train_epoch,
    )
    va_sampler, t_init_va = time_call(
        VectorizedSampler,
        spec.distance_matrix,
        spec.langs,
        spec.train_to,
        spec.n,
        spec.min_margin,
        val_epoch,
    )

    # Per-epoch sampling: median of n_reps draws, then x n_epochs.
    rng = np.random.default_rng(0)
    tr_samp = [time_call(tr_sampler.sample_epoch, rng)[1] for _ in range(n_reps)]
    va_samp = [time_call(va_sampler.sample_epoch, rng)[1] for _ in range(n_reps)]
    t_sample_per_epoch = statistics.median(tr_samp) + statistics.median(va_samp)

    return {
        "n": spec.n,
        "t_distance": t_distance,
        "t_init": t_init_tr + t_init_va,
        "t_sample_total": t_sample_per_epoch * n_epochs,
        "t_sample_per_epoch": t_sample_per_epoch,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-factor", type=float, default=2.3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--train-epoch-size", type=int, default=2000)
    ap.add_argument("--val-epoch-size", type=int, default=6400)
    ap.add_argument("--margin-scale", type=float, default=2.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-concepts", type=int, default=0, help="0 = all in the bucket")
    ap.add_argument("--scaling", action="store_true", help="report n,2n,3n scaling for 5 concepts")
    args = ap.parse_args()

    settings = make_settings(
        epochs=args.epochs,
        train_epoch_size=args.train_epoch_size,
        val_epoch_size=args.val_epoch_size,
        margin_scale=args.margin_scale,
        trim_augment_ratio=0.0,
    )
    d = load_cache()
    cids = list(d["concept_indices"].keys())

    if args.scaling:
        print("=== scaling: one-time setup (distance + argsort init) vs n ===")
        sample_cids = cids[:5]
        for cid in sample_cids:
            idxs = d["concept_indices"][cid]
            it = torch.tensor(idxs, dtype=torch.long)
            ci, cm = d["input_embs"][it].clone(), d["mining_embs_all"][it].clone()
            cl = list(d["concept_sent_langs"][cid])
            row = []
            for fac in (1.0, 2.0, 3.0):
                ti, tm, tl = tile_rows(ci, cm, cl, fac)
                r = profile_concept(
                    ti, tm, tl, settings, args.train_epoch_size, args.val_epoch_size, args.epochs, 1
                )
                row.append((r["n"], r["t_distance"], r["t_init"]))
            base_setup = row[0][1] + row[0][2]
            parts = " | ".join(
                f"n={n:5d} setup={(td + ti) * 1000:7.1f}ms (x{(td + ti) / base_setup:.1f})"
                for n, td, ti in row
            )
            print(f"  {cid}: {parts}")
        return 0

    if args.max_concepts > 0:
        cids = cids[: args.max_concepts]

    print(
        f"bucket: {len(cids)} concepts | aug_factor={args.aug_factor} | epochs={args.epochs} "
        f"| train_epoch={args.train_epoch_size} val_epoch={args.val_epoch_size}"
    )
    agg = {"t_distance": 0.0, "t_init": 0.0, "t_sample_total": 0.0}
    ns = []
    for cid in cids:
        idxs = d["concept_indices"][cid]
        it = torch.tensor(idxs, dtype=torch.long)
        ci, cm = d["input_embs"][it].clone(), d["mining_embs_all"][it].clone()
        cl = list(d["concept_sent_langs"][cid])
        ti, tm, tl = tile_rows(ci, cm, cl, args.aug_factor)
        r = profile_concept(
            ti, tm, tl, settings, args.train_epoch_size, args.val_epoch_size, args.epochs, args.reps
        )
        if r is None:
            continue
        for k in agg:
            agg[k] += r[k]
        ns.append(r["n"])

    one_time = agg["t_distance"] + agg["t_init"]
    per_epoch = agg["t_sample_total"]
    cpu_total = one_time + per_epoch
    print(f"\naugmented rows: total={sum(ns)} mean={int(np.mean(ns))} max={max(ns)}")
    print("\n=== CPU (numpy) phase totals across the bucket ===")
    print(f"  distance-matrix build (one-time)   : {agg['t_distance']:7.2f}s")
    print(f"  sampler init / argsort (one-time)  : {agg['t_init']:7.2f}s")
    print(f"  --> ONE-TIME setup subtotal        : {one_time:7.2f}s")
    print(f"  per-epoch sampling (x{args.epochs} epochs): {per_epoch:7.2f}s")
    print(f"  === CPU-serial total (no kernel)   : {cpu_total:7.2f}s")
    print("\n(compare against measured batched train wall ~380s; kernel+marshal = remainder)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
