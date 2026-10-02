"""Diagnostic: contextualized concept-word token pooling instead of full-sentence pooling.

For each sampled sentence, we:
1. Find a word that maps to a current concept (via word_to_concept.json)
2. Tokenize the sentence with the e5-small tokenizer, get offset mappings
3. Forward through the model, take the last hidden state
4. Average the hidden states of the tokens that fall inside the target word's
   character span (handles SentencePiece subword splits)
5. Use that averaged vector as the sentence's representation for diagnostics

If the same-concept vs different-concept cosine gap is meaningfully larger
than the full-sentence baseline (+0.016), token pooling is the way.

Usage:
    QDRANT_URL=...  QDRANT_API_KEY=...  uv run python research/diagnostics/diagnose_word_token_pooling.py
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from qdrant_client import QdrantClient
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "intfloat/multilingual-e5-small"
PREFIX = "passage: "
TOKEN_RE = re.compile(r"[a-záéíóúüñàèìòùâêîôûäëïöü]+", re.IGNORECASE)


def discover_concepts_from_text(
    client: QdrantClient,
    collection: str,
    scan_n: int,
    word_to_concept: dict[str, dict[str, str]],
    top_k: int,
) -> list[str]:
    """Scroll sentences, count current-vocab concepts they hit, return top_k."""
    counter: Counter[str] = Counter()
    offset = None
    seen = 0
    while seen < scan_n:
        batch, next_offset = client.scroll(
            collection_name=collection,
            limit=min(512, scan_n - seen),
            offset=offset,
            with_payload=["sentence", "lang"],
            with_vectors=False,
        )
        for p in batch:
            lang = p.payload["lang"]
            w2c = word_to_concept.get(lang, {})
            for tok in TOKEN_RE.findall(p.payload["sentence"].lower()):
                cid = w2c.get(tok)
                if cid:
                    counter[cid] += 1
        seen += len(batch)
        if next_offset is None:
            break
        offset = next_offset
    return [c for c, _ in counter.most_common(top_k)]


def fetch_sentences_for_concept(
    client: QdrantClient,
    collection: str,
    concept_id: str,
    word_to_concept: dict[str, dict[str, str]],
    concepts_data: dict,
    max_points: int,
) -> list[tuple[str, str, str]]:
    """Return (sentence, target_word, lang) tuples by scanning for current concept words.

    Note: cluster payloads have stale concept_ids, so we can't filter by current
    concept_id. Instead we scroll and check each sentence for current concept words.
    """
    concept_words_en = set(concepts_data["concepts"][concept_id]["en"])
    concept_words_es = set(concepts_data["concepts"][concept_id]["es"])
    out: list[tuple[str, str, str]] = []
    offset = None
    while len(out) < max_points:
        batch, next_offset = client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=["sentence", "lang"],
            with_vectors=False,
        )
        for p in batch:
            lang = p.payload["lang"]
            sentence = p.payload["sentence"]
            target_words = concept_words_en if lang == "en" else concept_words_es
            for tok in TOKEN_RE.findall(sentence.lower()):
                if tok in target_words:
                    out.append((sentence, tok, lang))
                    break
            if len(out) >= max_points:
                break
        if next_offset is None:
            break
        offset = next_offset
    return out


def encode_word_in_context(
    sentences: list[str],
    target_words: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Average the last-hidden-state tokens that overlap each target word."""
    all_vecs: list[np.ndarray] = []
    for start in range(0, len(sentences), batch_size):
        chunk_s = sentences[start : start + batch_size]
        chunk_w = target_words[start : start + batch_size]
        texts = [PREFIX + s for s in chunk_s]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=256,
        )
        offset_mapping = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        hidden = out.last_hidden_state.cpu()  # [B, T, D]

        for b, (text, target) in enumerate(zip(texts, chunk_w, strict=True)):
            text_lower = text.lower()
            pos = text_lower.find(target.lower())
            if pos < 0:
                all_vecs.append(np.zeros(hidden.shape[-1], dtype=np.float32))
                continue
            word_end = pos + len(target)
            offsets = offset_mapping[b].tolist()
            tok_idx: list[int] = []
            for i, (s, e) in enumerate(offsets):
                if s == 0 and e == 0:
                    continue
                if s < word_end and e > pos:
                    tok_idx.append(i)
            if not tok_idx:
                all_vecs.append(np.zeros(hidden.shape[-1], dtype=np.float32))
                continue
            vec = hidden[b, tok_idx].mean(dim=0)
            vec = vec / (vec.norm() + 1e-12)
            all_vecs.append(vec.numpy().astype(np.float32))
    return np.vstack(all_vecs)


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
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--discover-scan", type=int, default=20_000)
    parser.add_argument("--max-concepts", type=int, default=50)
    parser.add_argument("--per-concept", type=int, default=80)
    parser.add_argument("--min-points-per-concept", type=int, default=20)
    parser.add_argument("--n-pairs", type=int, default=5000)
    parser.add_argument("--n-triplets", type=int, default=5000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encode-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.url or not args.api_key:
        raise SystemExit("Set QDRANT_URL and QDRANT_API_KEY")

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

    print("Loading current vocab + word_to_concept...")
    with open(args.data_dir / "word_to_concept.json") as f:
        w2c = json.load(f)
    with open(args.data_dir / "concept_vocabulary.json") as f:
        vocab = json.load(f)

    print(f"Connecting to {args.url}...")
    client = QdrantClient(url=args.url, api_key=args.api_key)

    print(f"Discovering current-vocab concepts in cluster (scan {args.discover_scan})...")
    top_cids = discover_concepts_from_text(
        client, args.collection, args.discover_scan, w2c, args.max_concepts * 3
    )
    print(f"  Found {len(top_cids)} candidates")

    print(f"Pulling up to {args.per_concept} sentences per concept...")
    sentences: list[str] = []
    target_words: list[str] = []
    cids: list[str] = []
    langs: list[str] = []
    for cid in top_cids[: args.max_concepts * 2]:
        rows = fetch_sentences_for_concept(
            client, args.collection, cid, w2c, vocab, args.per_concept
        )
        if len(rows) < args.min_points_per_concept:
            continue
        for sent, target, lg in rows:
            sentences.append(sent)
            target_words.append(target)
            cids.append(cid)
            langs.append(lg)
        if len({c for c in cids}) >= args.max_concepts:
            break
    n = len(sentences)
    unique_cids = list(dict.fromkeys(cids))
    print(
        f"Got {n} sentences across {len(unique_cids)} concepts  |  EN: {langs.count('en')}  ES: {langs.count('es')}"
    )

    print(f"Loading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    print(f"Encoding {n} sentences with token pooling around target words...")
    V = encode_word_in_context(
        sentences, target_words, tokenizer, model, device, batch_size=args.encode_batch
    )
    print(f"Got {V.shape[0]} vectors, dim {V.shape[1]}")

    cid_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, cid in enumerate(cids):
        cid_to_idx[cid].append(i)
    qualifying = [c for c in unique_cids if len(cid_to_idx[c]) >= args.min_points_per_concept]
    print(f"{len(qualifying)} concepts qualify for analysis")

    # ---------- 1. Intra vs inter ----------
    print("\n=== 1. Same-concept vs different-concept cosine (token-pooled e5-small) ===")
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
        if cids[i] == cids[j]:
            continue
        c = float(V[i] @ V[j])
        inter.append(c)
        if langs[i] != langs[j]:
            inter_xl.append(c)

    stats("intra (same concept)", intra)
    stats("inter (different concept)", inter)
    print(f"  gap (intra - inter) = {np.mean(intra) - np.mean(inter):+.3f}")

    print("\n=== 2. Cross-lingual alignment within concept ===")
    if intra_xl and inter_xl:
        stats("intra EN-ES (same concept)", intra_xl)
        stats("inter EN-ES (diff concept)", inter_xl)
        print(f"  gap = {np.mean(intra_xl) - np.mean(inter_xl):+.3f}")

    # ---------- 3. Triplet rejection ----------
    print("\n=== 3. Triplet rejection vs margin (any-lang) ===")
    gaps: list[float] = []
    while len(gaps) < args.n_triplets:
        cid = rng.choice(qualifying)
        idx = cid_to_idx[cid]
        a, p = rng.sample(idx, 2)
        neg = -1
        for _ in range(20):
            cand = rng.randrange(n)
            if cids[cand] != cid:
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
            if langs[cand] == "es" and cids[cand] != cid:
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
