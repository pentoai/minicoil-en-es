"""End-to-end equivalence (proof part c).

Trains a small representative set of concepts BOTH ways to completion with
dropout ON and real per-epoch re-mining (the production setting), then checks the
final per-concept val losses match within tolerance.

Because the reference and batched paths consume independent RNG streams (sampling
+ dropout), bit-exact end-to-end is impossible by construction (proven instead in
parts a/b). The fair test is whether the batched path differs from the reference
by MORE than the reference differs from itself across a sampling-seed change. So
we measure a NOISE FLOOR: two reference runs with the same pinned init but
different sampling seeds. If the batched-vs-reference delta is within the same
band as reference-vs-reference, the refactor adds no systematic drift.

Init is pinned per concept across all runs so the comparison isolates the
sampling/dropout stochasticity (not init variance).

Run: ``uv run python scripts/perf/validate_end_to_end.py``
"""

from __future__ import annotations

import os
import random
import sys

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

from minicoil_v2.batched_training import train_concept_bucket  # noqa: E402
from minicoil_v2.constants import INPUT_DIM, OUTPUT_DIM  # noqa: E402
from minicoil_v2.train_concept_layers import train_one_concept  # noqa: E402

DEVICE = torch.device("cpu")


def init_for(seed_i: int) -> torch.Tensor:
    torch.manual_seed(seed_i)
    return torch.nn.Linear(INPUT_DIM, OUTPUT_DIM, bias=False).weight.data.clone()


def reference_val(d, cids, st, init_seeds, sampling_seed) -> dict[str, float]:
    """Reference path val loss per concept; init pinned, sampling seed varied."""
    out = {}
    for cid, sd in zip(cids, init_seeds, strict=True):
        c_input, c_mining, c_langs = concept_inputs(d, cid)
        torch.manual_seed(sd)  # pins the internal nn.Linear init == init_for(sd)
        random.seed(sampling_seed)
        np.random.seed(sampling_seed)
        _, _, val = train_one_concept(c_input, c_mining, c_langs, st, DEVICE)
        out[cid] = val
    return out


def batched_val(d, cids, st, init_seeds, sampling_seed) -> dict[str, float]:
    """Batched path final val loss per concept (vectorized), same pinned init."""
    specs = [build_spec(d, c, st) for c in cids]
    init_w = torch.stack([init_for(sd) for sd in init_seeds])
    torch.manual_seed(sampling_seed)  # dropout stream
    res = train_concept_bucket(
        specs,
        st,
        DEVICE,
        "vectorized",
        epochs=st.epochs,
        init_weight=init_w,
        seed=sampling_seed,
    )
    return {sp.cid: res[sp.cid]["val_loss"] for sp in specs}


def summarize(name, deltas):
    arr = np.array(list(deltas.values()))
    print(
        f"  {name}: n={len(arr)} mean={arr.mean():.5f} median={np.median(arr):.5f} "
        f"p90={np.percentile(arr, 90):.5f} max={arr.max():.5f}"
    )
    return arr


def main() -> int:
    d = load_cache()
    # Slightly larger epochs than the unit tests, but still small/fast.
    st = make_settings(
        dropout=0.05,
        epochs=30,
        train_epoch_size=2000,
        val_epoch_size=500,
        sample_batch_size=256,
        lr_patience=3,
    )
    cids = pick_concepts(d, 24)
    init_seeds = [1234 + 7 * i for i in range(len(cids))]
    print(f"end-to-end on {len(cids)} concepts, epochs={st.epochs}, dropout={st.dropout}")

    ref_a = reference_val(d, cids, st, init_seeds, sampling_seed=11)
    ref_b = reference_val(d, cids, st, init_seeds, sampling_seed=22)
    bat = batched_val(d, cids, st, init_seeds, sampling_seed=33)

    noise = {c: abs(ref_a[c] - ref_b[c]) for c in cids}
    cross = {c: abs(bat[c] - ref_a[c]) for c in cids}

    print("\nfinal val-loss deltas:")
    n_arr = summarize("reference-vs-reference (noise floor)", noise)
    c_arr = summarize("batched-vs-reference", cross)

    # batched should not drift beyond the reference's own run-to-run noise.
    # Allow a small multiplier on the mean and a modest absolute cap.
    ok = c_arr.mean() <= max(2.0 * n_arr.mean(), 0.02) and c_arr.max() <= max(
        3.0 * n_arr.max(), 0.05
    )
    print(f"\nmean(cross)={c_arr.mean():.5f} vs 2x mean(noise)={2 * n_arr.mean():.5f}")
    print(
        "ALL END-TO-END CHECKS PASSED"
        if ok
        else "END-TO-END CHECK FAILED (drift beyond noise floor)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
