"""Byte-equality snapshot of ``sample_epoch`` -- certifies a sampler rewrite is no-gate.

grid_check (validate_sampler.py a1) proves ``pick_pairs_vectorized`` is bit-exact
in ISOLATION with injected pos_ranks, and a3 only matches distributions. Neither
catches a change in ``sample_epoch``'s RNG consumption (draw order/count): such a
change shifts the exact triplets, silently altering trained weights, while every
existing check still passes. That is the unvalidated-equivalence trap.

This test pins the assembled epoch byte-for-byte. ``capture`` writes golden dicts
(run on the PRE-rewrite code); ``check`` re-runs and asserts np.array_equal on
every key (run POST-rewrite). Same seed + same code => identical bytes.

  uv run python scripts/perf/validate_sampler_rng.py capture
  uv run python scripts/perf/validate_sampler_rng.py check
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from perf_common import build_spec, load_cache, make_settings  # noqa: E402

from minicoil_v2.batched_training import VectorizedSampler  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(__file__), "sampler_rng_golden.pkl")
SEEDS = (0, 1, 7)
KEYS = (
    "anchor",
    "sl_pos",
    "sl_neg",
    "xl_pos",
    "xl_neg",
    "margin_sl",
    "margin_xl",
    "has_xl",
)


def epochs_for(spec) -> dict:
    """Assemble train+val sample_epoch dicts for several seeds (the RNG-sensitive output)."""
    out: dict = {}
    for seed in SEEDS:
        tr = VectorizedSampler(
            spec.distance_matrix, spec.langs, 0, spec.train_to, spec.min_margin, epoch_size=4000
        )
        va = VectorizedSampler(
            spec.distance_matrix,
            spec.langs,
            spec.train_to,
            spec.n,
            spec.min_margin,
            epoch_size=2000,
        )
        out[f"train_{seed}"] = tr.sample_epoch(np.random.default_rng(seed))
        out[f"val_{seed}"] = va.sample_epoch(np.random.default_rng(seed + 100))
    return out


def collect() -> dict:
    d = load_cache()
    st = make_settings()
    items = sorted(d["concept_indices"], key=lambda k: len(d["concept_indices"][k]))
    picks = {"smallest": items[0], "median": items[len(items) // 2], "largest": items[-1]}
    snap: dict = {}
    for name, cid in picks.items():
        snap[name] = epochs_for(build_spec(d, cid, st))
    return snap


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    snap = collect()
    if mode == "capture":
        with open(GOLDEN, "wb") as f:
            pickle.dump(snap, f)
        n = sum(len(v) for v in snap.values())
        print(f"captured golden: {len(snap)} concepts x {len(SEEDS)} seeds x 2 ranges = {n} epochs")
        return 0

    if not os.path.exists(GOLDEN):
        print("no golden snapshot; run `capture` on the pre-rewrite code first")
        return 1
    with open(GOLDEN, "rb") as f:
        gold = pickle.load(f)

    mismatches: list[str] = []
    for name, epochs in snap.items():
        for ep_key, ep in epochs.items():
            g = gold[name][ep_key]
            for k in KEYS:
                if not np.array_equal(g[k], ep[k]):
                    mismatches.append(f"{name}/{ep_key}/{k}")
    if mismatches:
        print(f"BYTE-EQUALITY FAILED ({len(mismatches)} keys differ): {mismatches[:8]}")
        print("=> the rewrite CHANGED the sampled triplets; it is NOT no-gate.")
        return 1
    print(f"BYTE-EQUALITY PASSED: every sample_epoch key identical across {len(snap)} concepts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
