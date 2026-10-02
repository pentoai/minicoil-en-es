"""Compare raw 384D intra-cosine to 4D intra-cosine for every trained concept.

The 4D-only findings can't tell whether the trained per-concept layer
*adds* signal over the raw mE5 input or *subtracts* signal in exchange
for sense separation. This script answers that.

For each trained concept and each sense bucket:
  - intra-cosine in raw 384D (the input vectors)
  - intra-cosine in 4D (output of the trained layer)
  - count + language breakdown (to flag language-confound buckets)

Run twice: once for token-pool collection + layers, once for sentence-pool.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from qdrant_client import QdrantClient

# Reuse bucketing + scrolling from the main validator.
sys.path.insert(0, str(Path(__file__).parent))
from validate_4d_polysemy import (  # noqa: E402
    CONCEPT_LABEL,
    POLYSEMOUS,
    TRAINED_CONCEPTS_ORDER,
    apply_layer,
    bucket_rows,
    scroll_concept_all,
    upper_mean,
)


def normed(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--layers-path", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    args = ap.parse_args()

    qc = QdrantClient(url=args.qdrant_url, prefer_grpc=False)
    layers = torch.load(args.layers_path, map_location="cpu", weights_only=True)

    print(f"\n=== Baseline comparison [{args.label}] ===\n")
    print(
        f"{'cid':>9s}  {'label':<22s} {'bucket':<22s} {'N':>4s} {'en/es':>9s}  "
        f"{'intra_384':>10s} {'intra_4D':>10s} {'Δ(4D-384)':>10s}"
    )

    # global aggregates
    mono_intra_384: list[float] = []
    mono_intra_4d: list[float] = []
    poly_intra_384: list[float] = []
    poly_intra_4d: list[float] = []
    poly_cross_384: list[float] = []
    poly_cross_4d: list[float] = []

    for cid in TRAINED_CONCEPTS_ORDER:
        if cid not in layers:
            continue
        rows = scroll_concept_all(qc, args.collection, cid)
        if not rows:
            continue

        vecs384 = np.stack([r["vec"] for r in rows])
        vecs4 = apply_layer(vecs384, layers[cid])
        for r, v3, v4 in zip(rows, vecs384, vecs4):
            r["v384n"] = v3 / (np.linalg.norm(v3) + 1e-12)
            r["v4n"] = v4 / (np.linalg.norm(v4) + 1e-12)

        buckets = bucket_rows(cid, rows)
        bucket_stats = []
        for bname, br in sorted(buckets.items()):
            if len(br) < 3:
                continue
            en = sum(1 for r in br if r["lang"] == "en")
            es = sum(1 for r in br if r["lang"] == "es")
            V3 = np.stack([r["v384n"] for r in br])
            V4 = np.stack([r["v4n"] for r in br])
            i3 = upper_mean(V3 @ V3.T)
            i4 = upper_mean(V4 @ V4.T)
            label = CONCEPT_LABEL.get(cid, "?")[:22]
            print(
                f"{cid:>9s}  {label:<22s} {bname[:22]:<22s} {len(br):>4d} {f'{en}/{es}':>9s}  "
                f"{i3:>+10.3f} {i4:>+10.3f} {i4 - i3:>+10.3f}"
            )
            bucket_stats.append((bname, V3, V4, i3, i4))

            if cid in POLYSEMOUS:
                poly_intra_384.append(i3)
                poly_intra_4d.append(i4)
            else:
                mono_intra_384.append(i3)
                mono_intra_4d.append(i4)

        # cross-bucket stats for polysemous concepts
        if cid in POLYSEMOUS and len(bucket_stats) >= 2:
            for i, (_, Vi3, Vi4, _, _) in enumerate(bucket_stats):
                for j, (_, Vj3, Vj4, _, _) in enumerate(bucket_stats):
                    if j <= i:
                        continue
                    poly_cross_384.append(float((Vi3 @ Vj3.T).mean()))
                    poly_cross_4d.append(float((Vi4 @ Vj4.T).mean()))
        print()

    # ----- global summary -----
    print("\n=== Global aggregates ===")

    def stat(name: str, vals: list[float]) -> None:
        if not vals:
            print(f"  {name}: (no data)")
            return
        a = np.array(vals)
        print(
            f"  {name}: n={len(a)} mean={a.mean():+.3f} median={np.median(a):+.3f} std={a.std():.3f}"
        )

    stat("monosemic intra 384D", mono_intra_384)
    stat("monosemic intra  4D", mono_intra_4d)
    if mono_intra_384 and mono_intra_4d:
        deltas = np.array(mono_intra_4d) - np.array(mono_intra_384)
        print(
            f"  monosemic Δ (4D - 384D): mean={deltas.mean():+.3f}  "
            f"won_by_4D={int((deltas > 0).sum())}/{len(deltas)}"
        )

    stat("polysemic intra 384D", poly_intra_384)
    stat("polysemic intra  4D", poly_intra_4d)
    if poly_intra_384 and poly_intra_4d:
        deltas = np.array(poly_intra_4d) - np.array(poly_intra_384)
        print(f"  polysemic intra Δ (4D - 384D): mean={deltas.mean():+.3f}")
    stat("polysemic cross 384D", poly_cross_384)
    stat("polysemic cross  4D", poly_cross_4d)
    if poly_cross_384 and poly_cross_4d:
        deltas = np.array(poly_cross_4d) - np.array(poly_cross_384)
        print(
            f"  polysemic cross Δ (4D - 384D): mean={deltas.mean():+.3f} "
            f"(more negative = layer pushes senses further apart than raw input)"
        )


if __name__ == "__main__":
    main()
