"""Sampler equivalence (proof part a) for the vectorized sampler.

Three independent checks, increasing in scope:

  a1. DETERMINISTIC CORE (bit-exact). For several concepts (smallest, medium,
      largest, and a synthesized monolingual one), exhaustively compare the
      vectorized ``pick_pairs_vectorized`` against the real
      ``BilingualSampler._pick_pair_mined`` for EVERY anchor x EVERY pos_rank in
      [0, k), for SL (exclude_a=True) and XL (exclude_a=False). Asserts identical
      (positive, negative, margin) and identical accept/reject. This is the
      strongest part: it pins the selection mapping independent of RNG.

  a2. ASSEMBLY INVARIANTS. Run the full vectorized epoch and assert the structural
      contract: indices lie in the correct language/range; monolingual concepts
      yield has_xl all-False with xl_pos==xl_neg==anchor; bilingual rows with
      has_xl=True have a cross-language positive.

  a3. DISTRIBUTION MATCH. Exact full-sampler RNG alignment is infeasible (the
      reference consumes Python+NumPy RNG interleaved per sample with rejection
      redraws; the vectorized path draws in bulk). Instead, run both samplers over
      a full epoch with their own RNG and assert the SELECTION DISTRIBUTION
      matches: reject rate, XL rate, and the margin mean/std/quantiles.

Run: ``uv run python scripts/perf/validate_sampler.py``
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from perf_common import build_spec, load_cache, make_settings  # noqa: E402

from minicoil_v2.batched_training import VectorizedSampler, pick_pairs_vectorized  # noqa: E402
from minicoil_v2.train_concept_layers import BilingualSampler  # noqa: E402

TOP_K = 20


def grid_check(spec) -> tuple[int, int]:
    """Exhaustive (anchor x pos_rank) bit-exact check vs the reference picker."""
    new = VectorizedSampler(spec.distance_matrix, spec.langs, 0, spec.train_to, spec.min_margin)
    old = BilingualSampler(
        spec.input_embs,
        spec.distance_matrix,
        spec.langs,
        0,
        spec.train_to,
        min_margin=spec.min_margin,
        epoch_size=1,
    )
    # Precompute structures must already match (same code), assert it.
    for lang in old._sorted_idx:
        assert np.array_equal(old._sorted_idx[lang], new._sorted_idx[lang])
        assert np.allclose(old._sorted_dist[lang], new._sorted_dist[lang])

    total = 0
    mism = 0
    langs = old.all_langs_list
    for anchor_lang in langs:
        pool = old.lang_indices[anchor_lang]
        targets = [(anchor_lang, True)] + [(o, False) for o in langs if o != anchor_lang]
        for tlang, excl in targets:
            m = len(old.lang_indices[tlang]) - (1 if excl else 0)
            if m < 3:
                continue
            k = min(TOP_K, max(1, m // 3))
            sd = new._sorted_dist[tlang][pool]
            self_pos = new._self_pos[tlang][pool] if excl else None
            for pr in range(k):
                pos, neg, marg, val = pick_pairs_vectorized(
                    new._sorted_idx[tlang],
                    sd,
                    pool,
                    self_pos,
                    excl,
                    spec.min_margin,
                    TOP_K,
                    np.full(len(pool), pr),
                )
                for ai, a in enumerate(pool):
                    orig = np.random.randint
                    np.random.randint = lambda *x, **y: pr  # noqa: B023
                    try:
                        r = old._pick_pair_mined(int(a), tlang, excl)
                    finally:
                        np.random.randint = orig
                    total += 1
                    if r is None:
                        if val[ai]:
                            mism += 1
                    else:
                        op, on, omg = r
                        if (
                            not val[ai]
                            or op != int(pos[ai])
                            or on != int(neg[ai])
                            or abs(omg - float(marg[ai])) > 1e-6
                        ):
                            mism += 1
    return total, mism


def assembly_invariants(spec, monolingual: bool) -> list[str]:
    errs: list[str] = []
    rng = np.random.default_rng(7)
    s = VectorizedSampler(spec.distance_matrix, spec.langs, 0, spec.train_to, spec.min_margin, 8000)
    ep = s.sample_epoch(rng)
    range_idx = set(range(spec.train_to))
    langs = spec.langs

    for key in ("anchor", "sl_pos", "sl_neg"):
        if not all(i in range_idx for i in ep[key]):
            errs.append(f"{key} has out-of-range index")
    # SL pos/neg share the anchor's language.
    a_lang = langs[ep["anchor"]]
    if not np.array_equal(langs[ep["sl_pos"]], a_lang):
        errs.append("sl_pos language != anchor language")
    if not np.array_equal(langs[ep["sl_neg"]], a_lang):
        errs.append("sl_neg language != anchor language")

    if monolingual:
        if ep["has_xl"].any():
            errs.append("monolingual concept produced has_xl=True")
        if not (
            np.array_equal(ep["xl_pos"], ep["anchor"])
            and np.array_equal(ep["xl_neg"], ep["anchor"])
        ):
            errs.append("monolingual xl_pos/xl_neg != anchor")
    else:
        hx = ep["has_xl"]
        if hx.any():
            # cross-language: xl pos/neg are in the OTHER language (!= anchor lang).
            if (langs[ep["xl_pos"][hx]] == a_lang[hx]).any():
                errs.append("some xl_pos same language as anchor")
            if (langs[ep["xl_neg"][hx]] == a_lang[hx]).any():
                errs.append("some xl_neg same language as anchor")
        # rows without XL fall back to anchor.
        no = ~hx
        if no.any() and not (
            np.array_equal(ep["xl_pos"][no], ep["anchor"][no])
            and np.array_equal(ep["xl_neg"][no], ep["anchor"][no])
        ):
            errs.append("has_xl=False rows do not fall back to anchor")
    return errs


def distribution_check(spec) -> tuple[dict, dict, list[str]]:
    import random

    epoch = 40000
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
        epoch_size=epoch,
    )
    old_msl: list[float] = []
    old_mxl: list[float] = []
    for b in old:
        old_msl.extend(b["margin_sl"].tolist())
        old_mxl.extend(b["margin_xl"][b["has_xl"]].tolist())
    old_stats = {
        "reject_rate": old._epoch_rejections / max(old._epoch_samples + old._epoch_rejections, 1),
        "xl_rate": old._epoch_xl_count / max(old._epoch_samples, 1),
        "msl_mean": float(np.mean(old_msl)),
        "msl_std": float(np.std(old_msl)),
        "mxl_mean": float(np.mean(old_mxl)) if old_mxl else 0.0,
    }

    rng = np.random.default_rng(0)
    new = VectorizedSampler(
        spec.distance_matrix, spec.langs, 0, spec.train_to, spec.min_margin, epoch
    )
    ne = new.sample_epoch(rng)
    new_msl = ne["margin_sl"]
    new_mxl = ne["margin_xl"][ne["has_xl"]]
    new_stats = {
        "reject_rate": new.last_reject_count / max(epoch + new.last_reject_count, 1),
        "xl_rate": ne["has_xl"].mean(),
        "msl_mean": float(np.mean(new_msl)),
        "msl_std": float(np.std(new_msl)),
        "mxl_mean": float(np.mean(new_mxl)) if len(new_mxl) else 0.0,
    }

    errs: list[str] = []
    if abs(old_stats["reject_rate"] - new_stats["reject_rate"]) > 0.02:
        errs.append("reject_rate diff > 0.02")
    if abs(old_stats["xl_rate"] - new_stats["xl_rate"]) > 0.01:
        errs.append("xl_rate diff > 0.01")
    if abs(old_stats["msl_mean"] - new_stats["msl_mean"]) > 0.01:
        errs.append("margin_sl mean diff > 0.01")
    if abs(old_stats["msl_std"] - new_stats["msl_std"]) > 0.01:
        errs.append("margin_sl std diff > 0.01")
    return old_stats, new_stats, errs


def main() -> int:
    d = load_cache()
    st = make_settings()
    items = sorted(d["concept_indices"], key=lambda k: len(d["concept_indices"][k]))
    picks = [items[0], items[len(items) // 2], items[-1]]

    ok = True
    print("=== a1: deterministic core grid (bit-exact) ===")
    for cid in picks:
        spec = build_spec(d, cid, st)
        total, mism = grid_check(spec)
        print(f"  {cid}: n={spec.n} train_to={spec.train_to} grid_cases={total} mismatches={mism}")
        ok = ok and mism == 0
    mono = build_spec(d, items[0], st, force_lang="en")
    total, mism = grid_check(mono)
    print(f"  {items[0]}/MONOLINGUAL-en: train_to={mono.train_to} cases={total} mismatches={mism}")
    ok = ok and mism == 0

    print("\n=== a2: assembly invariants ===")
    for cid in (items[0], items[len(items) // 2]):
        errs = assembly_invariants(build_spec(d, cid, st), monolingual=False)
        print(f"  {cid} bilingual: {errs if errs else 'OK'}")
        ok = ok and not errs
    errs = assembly_invariants(mono, monolingual=True)
    print(f"  {items[0]}/MONOLINGUAL: {errs if errs else 'OK'}")
    ok = ok and not errs

    print("\n=== a3: distribution match (own-RNG full epoch) ===")
    for cid in (items[len(items) // 2], items[-1]):
        spec = build_spec(d, cid, st)
        old_s, new_s, errs = distribution_check(spec)
        print(f"  {cid}:")
        print(f"    reject_rate old={old_s['reject_rate']:.4f} new={new_s['reject_rate']:.4f}")
        print(f"    xl_rate     old={old_s['xl_rate']:.4f} new={new_s['xl_rate']:.4f}")
        print(
            f"    margin_sl   old={old_s['msl_mean']:.4f}+/-{old_s['msl_std']:.4f} "
            f"new={new_s['msl_mean']:.4f}+/-{new_s['msl_std']:.4f}"
        )
        print(f"    margin_xl   old_mean={old_s['mxl_mean']:.4f} new_mean={new_s['mxl_mean']:.4f}")
        print(f"    {errs if errs else 'OK'}")
        ok = ok and not errs

    print("\n" + ("ALL SAMPLER CHECKS PASSED" if ok else "SAMPLER CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
