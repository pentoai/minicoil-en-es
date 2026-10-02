"""Micro-profile the sub-phases of one ``sample_epoch`` on the largest concept.

Confirms (does not assume) where the dominant ``sample_epoch`` cost lives: the
per-chunk ``[c, m]`` fancy gather of the sorted candidate matrices and the
``[c, m]`` boolean-sum that computes the negative position ``j``, vs everything
else. This decides whether the searchsorted/top-k rewrite (which eliminates the
``[c, m]`` materialization) is worth doing.

Run: uv run python scripts/perf/profile_sampler_internals.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from perf_common import build_spec, load_cache, make_settings  # noqa: E402

from minicoil_v2.batched_training import VectorizedSampler  # noqa: E402


def time_block(fn, reps: int) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    d = load_cache()
    st = make_settings()
    # Largest concept = biggest pool = worst-case [c, m].
    cid = max(d["concept_indices"], key=lambda k: len(d["concept_indices"][k]))
    spec = build_spec(d, cid, st)
    s = VectorizedSampler(
        spec.distance_matrix, spec.langs, 0, spec.train_to, spec.min_margin, epoch_size=2000
    )
    lang = s.all_langs_list[0]
    pool = s.lang_indices[lang]
    sorted_idx = s._sorted_idx[lang]  # [N, m]
    sorted_dist = s._sorted_dist[lang]
    m = sorted_idx.shape[1]
    c = 4096
    rng = np.random.default_rng(0)
    a = pool[rng.integers(0, len(pool), size=c)]
    pos_ranks = rng.integers(0, 5, size=c)
    reps = 20

    print(f"concept={cid}  n={spec.n}  pool(|{lang}|)={len(pool)}  m={m}  c={c}")

    # Phase 1: the [c, m] fancy gathers (idx + dist).
    def gather():
        _ = sorted_idx[a]
        _ = sorted_dist[a]

    t_gather = time_block(gather, reps)

    # Phase 2: exclude_a masked copy ([c, m] bool mask + masked reshape).
    sc = sorted_idx[a]
    sd = sorted_dist[a]

    def exclude():
        keep = sc != a[:, None]
        _ = sc[keep].reshape(c, m - 1)
        _ = sd[keep].reshape(c, m - 1)

    t_exclude = time_block(exclude, reps)

    # Phase 3: the [c, m] boolean-sum that finds j.
    d_pos = sd[np.arange(c), pos_ranks]
    thr = d_pos + s.min_margin

    def boolsum():
        _ = (sd < thr[:, None]).sum(axis=1)

    t_boolsum = time_block(boolsum, reps)

    # Phase 4: full sample_epoch (the real per-round cost, both langs + xl).
    def full():
        s.sample_epoch(np.random.default_rng(1))

    t_full = time_block(full, max(3, reps // 4))

    print(f"  [c,m] gather (idx+dist)        : {t_gather * 1000:8.2f} ms")
    print(f"  exclude_a mask + reshape       : {t_exclude * 1000:8.2f} ms")
    print(f"  boolean-sum j = (sd<thr).sum   : {t_boolsum * 1000:8.2f} ms")
    print(f"  --> sum of [c,m] ops above     : {(t_gather + t_exclude + t_boolsum) * 1000:8.2f} ms")
    print(f"  full sample_epoch (1 round)    : {t_full * 1000:8.2f} ms")
    print(
        f"  [c,m]-op share of one round    : "
        f"{(t_gather + t_exclude + t_boolsum) / max(t_full, 1e-9):.0%} "
        f"(x2+ langs/xl per round amplify this)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
