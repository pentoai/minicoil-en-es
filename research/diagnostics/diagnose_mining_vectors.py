"""Mining encoder diagnostic: measure whether stored vectors discriminate concepts.

Pulls a sample of points from Qdrant, groups by concept_id (used as an opaque
grouping key — payload IDs may reference a stale vocab, which is fine here),
and computes three signals to decide whether the mining encoder is the cause
of high triplet rejection.

1. Same-concept vs different-concept cosine gap
2. Within-concept cross-lingual cosine alignment
3. Triplet rejection rate vs margin (general + cross-lingual variant)

Usage:
    QDRANT_URL=https://...  QDRANT_API_KEY=...  uv run python research/diagnostics/diagnose_mining_vectors.py
    uv run python research/diagnostics/diagnose_mining_vectors.py --url ... --api-key ...
"""

import argparse
import os
import random
from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient


def discover_top_concepts(
    client: QdrantClient, collection: str, scan_n: int, top_k: int
) -> list[str]:
    """Scroll payloads only (no vectors) and return the top_k most-frequent concept_ids."""
    from collections import Counter

    counter: Counter[str] = Counter()
    offset = None
    seen = 0
    while seen < scan_n:
        batch, next_offset = client.scroll(
            collection_name=collection,
            limit=min(512, scan_n - seen),
            offset=offset,
            with_payload=["concept_ids"],
            with_vectors=False,
        )
        for p in batch:
            for cid in p.payload["concept_ids"]:
                counter[cid] += 1
        seen += len(batch)
        if next_offset is None:
            break
        offset = next_offset
    return [c for c, _ in counter.most_common(top_k)]


def fetch_concept_points(
    client: QdrantClient, collection: str, concept_id: str, max_points: int
) -> list[tuple[list[float], frozenset[str], str]]:
    """Filtered scroll for one concept_id, return (vector, concept_ids, lang) tuples."""
    from qdrant_client import models

    out = []
    offset = None
    while len(out) < max_points:
        batch, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="concept_ids", match=models.MatchValue(value=concept_id)
                    )
                ]
            ),
            limit=min(256, max_points - len(out)),
            offset=offset,
            with_payload=True,
            with_vectors=["mining"],
        )
        for p in batch:
            out.append((p.vector["mining"], frozenset(p.payload["concept_ids"]), p.payload["lang"]))
        if next_offset is None:
            break
        offset = next_offset
    return out


def fetch_targeted(client: QdrantClient, collection: str, concept_ids: list[str], per_concept: int):
    """Pull up to per_concept points per concept; return arrays."""
    vecs: list[list[float]] = []
    cids: list[frozenset[str]] = []
    langs: list[str] = []
    for cid in concept_ids:
        rows = fetch_concept_points(client, collection, cid, per_concept)
        for v, c, lg in rows:
            vecs.append(v)
            cids.append(c)
            langs.append(lg)
    V = np.asarray(vecs, dtype=np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    return V, cids, langs


def stats(name: str, arr) -> None:
    a = np.asarray(arr)
    print(
        f"  {name:<32} n={len(a):>5}  mean={a.mean():+.3f}  std={a.std():.3f}  "
        f"p10={np.percentile(a, 10):+.3f}  p50={np.percentile(a, 50):+.3f}  p90={np.percentile(a, 90):+.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL"))
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY"))
    parser.add_argument("--collection", default="minicoil_sentences")
    parser.add_argument(
        "--discover-scan", type=int, default=30_000, help="Points to scan to find top concepts"
    )
    parser.add_argument("--max-concepts", type=int, default=60)
    parser.add_argument("--per-concept", type=int, default=80, help="Points to pull per concept")
    parser.add_argument("--min-points-per-concept", type=int, default=20)
    parser.add_argument("--n-pairs", type=int, default=5000)
    parser.add_argument("--n-triplets", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.url or not args.api_key:
        raise SystemExit("Set QDRANT_URL and QDRANT_API_KEY (or pass --url / --api-key)")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print(f"Connecting to {args.url}...")
    client = QdrantClient(url=args.url, api_key=args.api_key)
    print(f"Discovering top concepts (scanning {args.discover_scan} points)...")
    top_cids = discover_top_concepts(
        client, args.collection, args.discover_scan, args.max_concepts * 2
    )
    print(
        f"  Found {len(top_cids)} candidate concepts; pulling up to {args.per_concept} points each"
    )
    V, CIDS, LANGS = fetch_targeted(
        client, args.collection, top_cids[: args.max_concepts], args.per_concept
    )
    n, d = V.shape
    print(f"Got {n} points, dim {d}  |  EN: {LANGS.count('en'):,}  ES: {LANGS.count('es'):,}")

    cid_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, cs in enumerate(CIDS):
        for cid in cs:
            cid_to_idx[cid].append(i)
    qualifying = [c for c in top_cids if len(cid_to_idx[c]) >= args.min_points_per_concept]
    qualifying = qualifying[: args.max_concepts]
    print(f"{len(qualifying)} concepts with >= {args.min_points_per_concept} points pulled")
    if not qualifying:
        raise SystemExit(
            "No qualifying concepts; raise --per-concept or lower --min-points-per-concept"
        )

    # ---------- 1. Intra vs inter concept ----------
    print("\n=== 1. Same-concept vs different-concept cosine ===")
    pairs_per_concept = max(1, args.n_pairs // len(qualifying))
    intra: list[float] = []
    intra_xl: list[float] = []
    for cid in qualifying:
        idx = cid_to_idx[cid]
        for _ in range(pairs_per_concept):
            i, j = rng.sample(idx, 2)
            c = float(V[i] @ V[j])
            intra.append(c)
            if LANGS[i] != LANGS[j]:
                intra_xl.append(c)

    inter: list[float] = []
    inter_xl: list[float] = []
    while len(inter) < args.n_pairs:
        i, j = rng.sample(range(n), 2)
        if CIDS[i] & CIDS[j]:
            continue
        c = float(V[i] @ V[j])
        inter.append(c)
        if LANGS[i] != LANGS[j]:
            inter_xl.append(c)

    stats("intra (same concept)", intra)
    stats("inter (different concept)", inter)
    print(f"  gap (intra - inter) = {np.mean(intra) - np.mean(inter):+.3f}")

    # ---------- 2. Cross-lingual alignment ----------
    print("\n=== 2. Cross-lingual alignment within concept ===")
    if intra_xl and inter_xl:
        stats("intra EN-ES (same concept)", intra_xl)
        stats("inter EN-ES (diff concept)", inter_xl)
        print(f"  gap = {np.mean(intra_xl) - np.mean(inter_xl):+.3f}")
    else:
        print("  (insufficient cross-lingual pairs)")

    # ---------- 3. Triplet rejection at margin ----------
    print("\n=== 3. Triplet rejection vs margin (any-lang anchor/pos/neg) ===")
    gaps: list[float] = []
    while len(gaps) < args.n_triplets:
        cid = rng.choice(qualifying)
        idx = cid_to_idx[cid]
        a, p = rng.sample(idx, 2)
        neg = -1
        for _ in range(20):
            cand = rng.randrange(n)
            if not (CIDS[a] & CIDS[cand]):
                neg = cand
                break
        if neg < 0:
            continue
        sap = float(V[a] @ V[p])
        san = float(V[a] @ V[neg])
        gaps.append(san - sap)
    arr = np.asarray(gaps)
    print(
        f"  cos(a,n) - cos(a,p):  mean={arr.mean():+.3f}  p10={np.percentile(arr, 10):+.3f}  "
        f"p50={np.percentile(arr, 50):+.3f}  p90={np.percentile(arr, 90):+.3f}"
    )
    print("  Triplet rejected when cos(a,n) - cos(a,p) + margin >= 0:")
    for m in [0.01, 0.05, 0.10, 0.20, 0.30]:
        print(f"    margin={m:.2f}: rejection = {float(np.mean(arr + m >= 0)) * 100:5.1f}%")

    # ---------- 3b. Cross-lingual triplet variant ----------
    print(
        "\n=== 3b. Cross-lingual triplet (anchor EN, pos ES same concept, neg ES diff concept) ==="
    )
    xl_gaps: list[float] = []
    attempts = 0
    while len(xl_gaps) < args.n_triplets and attempts < args.n_triplets * 30:
        attempts += 1
        cid = rng.choice(qualifying)
        idx = cid_to_idx[cid]
        en_idx = [i for i in idx if LANGS[i] == "en"]
        es_idx = [i for i in idx if LANGS[i] == "es"]
        if not en_idx or not es_idx:
            continue
        a = rng.choice(en_idx)
        p = rng.choice(es_idx)
        neg = -1
        for _ in range(40):
            cand = rng.randrange(n)
            if LANGS[cand] == "es" and not (CIDS[a] & CIDS[cand]):
                neg = cand
                break
        if neg < 0:
            continue
        sap = float(V[a] @ V[p])
        san = float(V[a] @ V[neg])
        xl_gaps.append(san - sap)
    if xl_gaps:
        xl = np.asarray(xl_gaps)
        print(f"  n={len(xl)}  mean gap={xl.mean():+.3f}  p50={np.percentile(xl, 50):+.3f}")
        for m in [0.01, 0.05, 0.10, 0.20]:
            print(f"    margin={m:.2f}: rejection = {float(np.mean(xl + m >= 0)) * 100:5.1f}%")
    else:
        print("  (no cross-lingual triplets sampled)")


if __name__ == "__main__":
    main()
