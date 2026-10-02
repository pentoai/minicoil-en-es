"""Trainer equivalence (proof part b).

Decomposed so each numerical risk is isolated:

  b0. UNIT: BatchedAdam vs torch.optim.Adam (per-concept lr), 1000 random steps.
  b1. UNIT: BatchedPlateau vs N x torch ReduceLROnPlateau, random val sequences.
  b2. old <-> A (Milestone A: batched forward + B separate torch optimizers/
      schedulers). Frozen triplets, dropout=0, same init -> expect BIT-EXACT.
      Proves the batched forward/loss/backward.
  b3. A <-> B (Milestone B: stacked custom Adam + vectorized plateau), same frozen
      triplets -> expect <= 1e-5 (the only deviation is fused addcdiv vs manual).
  b4. B>1 NON-INTERFERENCE: a concept trained alone (B=1) vs in a bucket of B
      yields the same weight; proves summed-loss/one-backward routes each
      concept exactly its own gradient.

Run: ``uv run python scripts/perf/validate_trainer.py``
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from perf_common import (  # noqa: E402
    FrozenSampler,
    build_spec,
    frozen_epoch,
    load_cache,
    make_settings,
    pick_concepts,
)

from minicoil_v2.batched_training import (  # noqa: E402
    BatchedAdam,
    BatchedPlateau,
    train_concept_bucket,
)
from minicoil_v2.constants import INPUT_DIM, OUTPUT_DIM  # noqa: E402
from minicoil_v2.train_concept_layers import train_concept_layer  # noqa: E402

DEVICE = torch.device("cpu")


def test_adam() -> float:
    b = 4
    torch.manual_seed(1)
    init = torch.randn(b, OUTPUT_DIM, INPUT_DIM, dtype=torch.float64)
    lrs = [1e-3, 2e-3, 5e-4, 3e-3]

    # reference: B separate torch.optim.Adam
    ref_params = [torch.nn.Parameter(init[i].clone()) for i in range(b)]
    ref_opt = torch.optim.Adam([{"params": [ref_params[i]], "lr": lrs[i]} for i in range(b)])
    # ours
    w = init.clone()
    lr_vec = torch.tensor(lrs, dtype=torch.float64).reshape(b, 1, 1)
    adam = BatchedAdam(w, lr_vec)

    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(1000):
        g = torch.as_tensor(rng.standard_normal((b, OUTPUT_DIM, INPUT_DIM)), dtype=torch.float64)
        for i in range(b):
            ref_params[i].grad = g[i].clone()
        ref_opt.step()
        w.grad = g.clone()
        adam.step()
        ref_stack = torch.stack([p.data for p in ref_params])
        max_diff = max(max_diff, (ref_stack - w).abs().max().item())
    return max_diff


def test_plateau() -> bool:
    b = 5
    factor, patience = 0.5, 3
    lr0 = 2e-3
    # reference: B torch schedulers, each on its own dummy optimizer
    ref_lrs = []
    ref_scheds = []
    for _ in range(b):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=lr0)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=factor, patience=patience, threshold=1e-4
        )
        ref_lrs.append(opt)
        ref_scheds.append(sch)
    # ours
    lr_vec = torch.full((b, 1, 1), lr0, dtype=torch.float64)
    plateau = BatchedPlateau(lr_vec, factor=factor, patience=patience)

    rng = np.random.default_rng(2)
    # mix of improving / plateauing sequences across concepts
    base = rng.uniform(0.1, 0.5, size=b)
    ok = True
    for t in range(60):
        # some concepts steadily improve, some plateau, some get noisy
        vals = []
        for i in range(b):
            if i % 3 == 0:
                vals.append(base[i] * (0.99**t))
            elif i % 3 == 1:
                vals.append(base[i] + 0.001 * np.sin(t))
            else:
                vals.append(base[i] + rng.normal(0, 0.01))
        vals_t = torch.tensor(vals, dtype=torch.float64)
        for i in range(b):
            ref_scheds[i].step(float(vals_t[i]))
        plateau.step(vals_t)
        ref = [opt.param_groups[0]["lr"] for opt in ref_lrs]
        ours = lr_vec.reshape(-1).tolist()
        for i in range(b):
            if abs(ref[i] - ours[i]) > 1e-12:
                ok = False
    return ok


def _frozen_for(spec, st, seed):
    return (
        [frozen_epoch(spec, st.train_epoch_size, "train", seed)],
        [frozen_epoch(spec, st.val_epoch_size, "val", seed + 1)],
    )


def init_for(seed_i: int) -> torch.Tensor:
    """The exact weight a reference ``nn.Linear`` draws after seeding ``seed_i``."""
    torch.manual_seed(seed_i)
    return torch.nn.Linear(INPUT_DIM, OUTPUT_DIM, bias=False).weight.data.clone()


def run_reference(spec, st, train_ep, val_ep, seed):
    torch.manual_seed(seed)
    layer, _, _ = train_concept_layer(
        FrozenSampler(spec.input_embs, train_ep, st.sample_batch_size),
        FrozenSampler(spec.input_embs, val_ep, st.sample_batch_size),
        DEVICE,
        epochs=st.epochs,
        lr=st.lr,
        dropout=st.dropout,
        lr_factor=st.lr_factor,
        lr_patience=st.lr_patience,
    )
    return layer.weight.data


def main() -> int:
    ok = True

    print("=== b0: BatchedAdam vs torch.optim.Adam (1000 steps, per-concept lr) ===")
    dmax = test_adam()
    print(f"  max|dw| = {dmax:.2e}  ({'OK' if dmax < 1e-6 else 'FAIL'})")
    ok = ok and dmax < 1e-6

    print("\n=== b1: BatchedPlateau vs torch ReduceLROnPlateau (60 epochs, 5 concepts) ===")
    p_ok = test_plateau()
    print(f"  lr trajectories identical: {p_ok}  ({'OK' if p_ok else 'FAIL'})")
    ok = ok and p_ok

    d = load_cache()
    st = make_settings(epochs=10, lr_patience=2)
    cids = pick_concepts(d, 4)
    specs = [build_spec(d, c, st) for c in cids]

    print("\n=== b2/b3: frozen-triplet equivalence (dropout=0, same init) ===")
    # Per-concept seeds so each reference run and its bucket slice start identically
    # (a single global seed would couple init to bucket position).
    seeds = [42 + 1000 * i for i in range(len(specs))]
    frozen_tr = []
    frozen_va = []
    ref_w = {}
    for spec, sd in zip(specs, seeds, strict=True):
        tr, va = _frozen_for(spec, st, sd)
        frozen_tr.append(tr[0])
        frozen_va.append(va[0])
        ref_w[spec.cid] = run_reference(spec, st, tr[0], va[0], sd)
    init_w = torch.stack([init_for(sd) for sd in seeds])

    resA = train_concept_bucket(
        specs,
        st,
        DEVICE,
        "per_concept_torch",
        frozen_train=frozen_tr,
        frozen_val=frozen_va,
        epochs=st.epochs,
        init_weight=init_w,
    )
    resB = train_concept_bucket(
        specs,
        st,
        DEVICE,
        "vectorized",
        frozen_train=frozen_tr,
        frozen_val=frozen_va,
        epochs=st.epochs,
        init_weight=init_w,
    )
    for spec in specs:
        d_oa = (ref_w[spec.cid] - resA[spec.cid]["weight"]).abs().max().item()
        d_ab = (resA[spec.cid]["weight"] - resB[spec.cid]["weight"]).abs().max().item()
        print(f"  {spec.cid} n={spec.n}: old<->A={d_oa:.2e}  A<->B={d_ab:.2e}")
        ok = ok and d_oa <= 1e-5 and d_ab <= 1e-5

    print("\n=== b4: B>1 non-interference (alone vs bucketed) ===")
    # Train each concept alone (B=1) and in the full bucket; weights must match.
    for i, spec in enumerate(specs):
        solo = train_concept_bucket(
            [spec],
            st,
            DEVICE,
            "vectorized",
            frozen_train=[frozen_tr[i]],
            frozen_val=[frozen_va[i]],
            epochs=st.epochs,
            init_weight=init_w[i : i + 1],
        )[spec.cid]["weight"]
        bucketed = resB[spec.cid]["weight"]
        d_sb = (solo - bucketed).abs().max().item()
        print(f"  {spec.cid}: solo<->bucket={d_sb:.2e}")
        ok = ok and d_sb <= 1e-6

    print("\n" + ("ALL TRAINER CHECKS PASSED" if ok else "TRAINER CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
