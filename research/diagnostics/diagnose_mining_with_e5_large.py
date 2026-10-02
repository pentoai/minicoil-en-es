"""Re-encode the same Qdrant sentences with e5-large and rerun the diagnostic.

Same statistical methodology as diagnose_mining_vectors.py, but instead of using
the stored 384D e5-small vectors, we pull the sentences and encode them with
intfloat/multilingual-e5-large (1024D) locally. This isolates the encoder as
the variable: same data, same concept grouping, different geometry.

Usage:
    QDRANT_URL=...  QDRANT_API_KEY=...  uv run python research/diagnostics/diagnose_mining_with_e5_large.py
"""

import argparse
import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-large"


def discover_top_concepts(
    client: QdrantClient, collection: str, scan_n: int, top_k: int
) -> list[str]:
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


def fetch_concept_text(
    client: QdrantClient, collection: str, concept_id: str, max_points: int
) -> list[tuple[str, frozenset[str], str]]:
    out: list[tuple[str, frozenset[str], str]] = []
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
            with_payload=["sentence", "concept_ids", "lang"],
            with_vectors=False,
        )
        for p in batch:
            out.append(
                (p.payload["sentence"], frozenset(p.payload["concept_ids"]), p.payload["lang"])
            )
        if next_offset is None:
            break
        offset = next_offset
    return out


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
    parser.add_argument("--discover-scan", type=int, default=30_000)
    parser.add_argument("--max-concepts", type=int, default=60)
    parser.add_argument("--per-concept", type=int, default=80)
    parser.add_argument("--min-points-per-concept", type=int, default=20)
    parser.add_argument("--n-pairs", type=int, default=5000)
    parser.add_argument("--n-triplets", type=int, default=5000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encode-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.url or not args.api_key:
        raise SystemExit("Set QDRANT_URL and QDRANT_API_KEY (or pass --url / --api-key)")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Connecting to {args.url}...")
    client = QdrantClient(url=args.url, api_key=args.api_key)
    print(f"Discovering top concepts (scanning {args.discover_scan} points)...")
    top_cids = discover_top_concepts(
        client, args.collection, args.discover_scan, args.max_concepts * 2
    )
    print(f"  Found {len(top_cids)} candidates; pulling {args.per_concept} sentences each")

    sentences: list[str] = []
    cids: list[frozenset[str]] = []
    langs: list[str] = []
    for cid in top_cids[: args.max_concepts]:
        rows = fetch_concept_text(client, args.collection, cid, args.per_concept)
        for s, c, lg in rows:
            sentences.append(s)
            cids.append(c)
            langs.append(lg)
    n = len(sentences)
    print(f"Pulled {n} sentences  |  EN: {langs.count('en'):,}  ES: {langs.count('es'):,}")

    print(f"Loading {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"Encoding {n} sentences (batch={args.encode_batch})...")
    embs = model.encode(
        ["passage: " + s for s in sentences],
        batch_size=args.encode_batch,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    V = embs.astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    print(f"Got {V.shape[0]} vectors, dim {V.shape[1]}")

    cid_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, cs in enumerate(cids):
        for cid in cs:
            cid_to_idx[cid].append(i)
    qualifying = [c for c in top_cids if len(cid_to_idx[c]) >= args.min_points_per_concept][
        : args.max_concepts
    ]
    print(f"{len(qualifying)} concepts qualify")

    # ---------- 1. Intra vs inter concept ----------
    print("\n=== 1. Same-concept vs different-concept cosine (e5-large) ===")
    pairs_per_concept = max(1, args.n_pairs // len(qualifying))
    intra: list[float] = []
    intra_xl: list[float] = []
    for cid in qualifying:
        idx = cid_to_idx[cid]
        for _ in range(pairs_per_concept):
            i, j = rng.sample(idx, 2)
            c = float(V[i] @ V[j])
            intra.append(c)
            if langs[i] != langs[j]:
                intra_xl.append(c)

    inter: list[float] = []
    inter_xl: list[float] = []
    while len(inter) < args.n_pairs:
        i, j = rng.sample(range(n), 2)
        if cids[i] & cids[j]:
            continue
        c = float(V[i] @ V[j])
        inter.append(c)
        if langs[i] != langs[j]:
            inter_xl.append(c)

    stats("intra (same concept)", intra)
    stats("inter (different concept)", inter)
    print(f"  gap (intra - inter) = {np.mean(intra) - np.mean(inter):+.3f}")

    print("\n=== 2. Cross-lingual alignment within concept (e5-large) ===")
    if intra_xl and inter_xl:
        stats("intra EN-ES (same concept)", intra_xl)
        stats("inter EN-ES (diff concept)", inter_xl)
        print(f"  gap = {np.mean(intra_xl) - np.mean(inter_xl):+.3f}")

    # ---------- 3. Triplet rejection ----------
    print("\n=== 3. Triplet rejection vs margin (e5-large, any-lang) ===")
    gaps: list[float] = []
    while len(gaps) < args.n_triplets:
        cid = rng.choice(qualifying)
        idx = cid_to_idx[cid]
        a, p = rng.sample(idx, 2)
        neg = -1
        for _ in range(20):
            cand = rng.randrange(n)
            if not (cids[a] & cids[cand]):
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
    for m in [0.01, 0.05, 0.10, 0.20, 0.30]:
        print(f"    margin={m:.2f}: rejection = {float(np.mean(arr + m >= 0)) * 100:5.1f}%")

    print(
        "\n=== 3b. Cross-lingual triplet (anchor EN, pos ES same concept, neg ES diff concept) ==="
    )
    xl_gaps: list[float] = []
    attempts = 0
    while len(xl_gaps) < args.n_triplets and attempts < args.n_triplets * 30:
        attempts += 1
        cid = rng.choice(qualifying)
        idx = cid_to_idx[cid]
        en_idx = [i for i in idx if langs[i] == "en"]
        es_idx = [i for i in idx if langs[i] == "es"]
        if not en_idx or not es_idx:
            continue
        a = rng.choice(en_idx)
        p = rng.choice(es_idx)
        neg = -1
        for _ in range(40):
            cand = rng.randrange(n)
            if langs[cand] == "es" and not (cids[a] & cids[cand]):
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


if __name__ == "__main__":
    main()
