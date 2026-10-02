"""Validate that token pooling captures CONCEPT, not just word identity.

The previous diagnostic (diagnose_word_token_pooling.py) showed a gap of +0.108
between same-concept and different-concept cosine. But the target word was the
first concept word in each sentence — for many concepts, that's the SAME word
across sentences, so we may have been measuring "the cat token vs the cat token"
which is trivially close.

This script breaks the intra-concept gap into:
  (A) same-target-word    : pos pair uses identical surface word (e.g. cat / cat)
  (B) different-target-word: pos pair uses different words for same concept
                             (e.g. cat / kitten in EN, or cat / gato cross-lingual)
  (C) inter-concept        : different concepts (typically different words)

The meaningful gap for concept-vs-not-concept signal is (B) - (C).
If (A) >> (B) ≈ (C), token pooling is mostly measuring lexical identity.
If (B) ~ (A) >> (C), token pooling captures concept across surface forms.

Cross-lingual variant: anchor in EN with one word, positive in ES with the
translation. This is the strongest test (forces different surface, same concept).

Usage:
    QDRANT_URL=...  QDRANT_API_KEY=...  uv run python research/diagnostics/validate_token_pooling.py
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


def discover_concepts(client, collection, scan_n, w2c, top_k):
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
            for tok in TOKEN_RE.findall(p.payload["sentence"].lower()):
                cid = w2c.get(lang, {}).get(tok)
                if cid:
                    counter[cid] += 1
        seen += len(batch)
        if next_offset is None:
            break
        offset = next_offset
    return [c for c, _ in counter.most_common(top_k)]


def fetch_sentences_balanced(
    client,
    collection,
    concept_ids: set[str],
    vocab,
    max_per_concept: int,
    max_per_word_per_lang: int,
):
    """Pull sentences such that each (concept, word, lang) cell is roughly capped.

    This forces diversity of target words within each concept — without it,
    we'd be dominated by the most common surface word.
    """
    word_to_cid_lang: dict[tuple[str, str], str] = {}
    for cid in concept_ids:
        for w in vocab["concepts"][cid]["en"]:
            word_to_cid_lang[(w, "en")] = cid
        for w in vocab["concepts"][cid]["es"]:
            word_to_cid_lang[(w, "es")] = cid

    counts_concept: dict[str, int] = defaultdict(int)
    counts_word_lang: dict[tuple[str, str], int] = defaultdict(int)
    out: list[tuple[str, str, str, str]] = []  # (sentence, target_word, lang, cid)

    offset = None
    while True:
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
            tokens = TOKEN_RE.findall(sentence.lower())
            for tok in tokens:
                cid = word_to_cid_lang.get((tok, lang))
                if not cid:
                    continue
                if counts_concept[cid] >= max_per_concept:
                    continue
                if counts_word_lang[(tok, lang)] >= max_per_word_per_lang:
                    continue
                out.append((sentence, tok, lang, cid))
                counts_concept[cid] += 1
                counts_word_lang[(tok, lang)] += 1
                break  # one target word per sentence
        if next_offset is None:
            break
        offset = next_offset
        if all(counts_concept.get(c, 0) >= max_per_concept for c in concept_ids):
            break
    return out


def encode_word_in_context(sentences, target_words, tokenizer, model, device, batch_size=16):
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
        offsets_all = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        hidden = out.last_hidden_state.cpu()
        for b, (text, target) in enumerate(zip(texts, chunk_w, strict=True)):
            text_lower = text.lower()
            pos = text_lower.find(target.lower())
            if pos < 0:
                all_vecs.append(np.zeros(hidden.shape[-1], dtype=np.float32))
                continue
            word_end = pos + len(target)
            tok_idx: list[int] = []
            for i, (s, e) in enumerate(offsets_all[b].tolist()):
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


def stats(name, arr):
    a = np.asarray(arr)
    if len(a) == 0:
        print(f"  {name:<48} n=    0  (no pairs)")
        return
    print(
        f"  {name:<48} n={len(a):>5}  mean={a.mean():+.3f}  std={a.std():.3f}  "
        f"p10={np.percentile(a, 10):+.3f}  p50={np.percentile(a, 50):+.3f}  p90={np.percentile(a, 90):+.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("QDRANT_URL"))
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY"))
    parser.add_argument("--collection", default="minicoil_sentences")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--discover-scan", type=int, default=50_000)
    parser.add_argument("--max-concepts", type=int, default=80)
    parser.add_argument("--per-concept", type=int, default=120)
    parser.add_argument(
        "--per-word-per-lang",
        type=int,
        default=15,
        help="Cap per (word, lang) so no single surface form dominates a concept",
    )
    parser.add_argument("--n-target-pairs", type=int, default=10000)
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

    print("Loading current vocab...")
    with open(args.data_dir / "word_to_concept.json") as f:
        w2c = json.load(f)
    with open(args.data_dir / "concept_vocabulary.json") as f:
        vocab = json.load(f)

    # Restrict to concepts with multiple words in BOTH languages — without that
    # we can't even measure the different-word case.
    eligible = {
        cid for cid, c in vocab["concepts"].items() if len(c["en"]) >= 2 and len(c["es"]) >= 2
    }
    print(f"  {len(eligible)} concepts have >=2 words in both EN and ES")

    print(f"Connecting to {args.url}...")
    client = QdrantClient(url=args.url, api_key=args.api_key)
    print(f"Discovering most-frequent eligible concepts (scan {args.discover_scan})...")
    top_cids_all = discover_concepts(
        client, args.collection, args.discover_scan, w2c, args.max_concepts * 5
    )
    top_cids = [c for c in top_cids_all if c in eligible][: args.max_concepts]
    print(f"  Selected {len(top_cids)} eligible concepts")

    print(
        f"Pulling sentences (per_concept={args.per_concept}, per_word_per_lang={args.per_word_per_lang})..."
    )
    rows = fetch_sentences_balanced(
        client,
        args.collection,
        set(top_cids),
        vocab,
        args.per_concept,
        args.per_word_per_lang,
    )
    sentences = [r[0] for r in rows]
    target_words = [r[1] for r in rows]
    langs = [r[2] for r in rows]
    cids = [r[3] for r in rows]
    n = len(rows)
    print(f"Got {n} sentences  |  EN: {langs.count('en')}  ES: {langs.count('es')}")

    # Word-form diversity per concept
    cid_words: dict[str, Counter] = defaultdict(Counter)
    for cid, w, lg in zip(cids, target_words, langs, strict=True):
        cid_words[cid][(w, lg)] += 1
    multi_form_concepts = [c for c, ctr in cid_words.items() if len({w for (w, _) in ctr}) >= 2]
    multi_form_xl_concepts = [
        c
        for c, ctr in cid_words.items()
        if any(lg == "en" for (_, lg) in ctr) and any(lg == "es" for (_, lg) in ctr)
    ]
    print(f"  Concepts with >=2 distinct surface words: {len(multi_form_concepts)}")
    print(f"  Concepts with both EN and ES sentences:   {len(multi_form_xl_concepts)}")

    print(f"Loading {MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    print(f"Encoding {n} sentences (token pooling)...")
    V = encode_word_in_context(
        sentences, target_words, tokenizer, model, device, batch_size=args.encode_batch
    )

    cid_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, cid in enumerate(cids):
        cid_to_idx[cid].append(i)

    # Sample pairs
    intra_same_word: list[float] = []  # same concept, same surface form
    intra_diff_word: list[float] = []  # same concept, different surface forms
    intra_xl_diff_word: list[float] = []  # cross-lingual + different surface (strongest pos)
    inter: list[float] = []
    inter_xl: list[float] = []

    qualifying = [c for c in top_cids if len(cid_to_idx[c]) >= 10]

    pairs_per_concept = max(1, args.n_target_pairs // max(1, len(qualifying)))
    for cid in qualifying:
        idx = cid_to_idx[cid]
        for _ in range(pairs_per_concept):
            i, j = rng.sample(idx, 2)
            c = float(V[i] @ V[j])
            if target_words[i] == target_words[j] and langs[i] == langs[j]:
                intra_same_word.append(c)
            else:
                intra_diff_word.append(c)
                if langs[i] != langs[j]:
                    intra_xl_diff_word.append(c)

    while len(inter) < args.n_target_pairs:
        i, j = rng.sample(range(n), 2)
        if cids[i] == cids[j]:
            continue
        c = float(V[i] @ V[j])
        inter.append(c)
        if langs[i] != langs[j]:
            inter_xl.append(c)

    print("\n=== Intra-concept similarity, broken down ===")
    stats("(A) intra, SAME surface word", intra_same_word)
    stats("(B) intra, DIFFERENT surface word", intra_diff_word)
    stats("(B-xl) intra, DIFF word, cross-lingual", intra_xl_diff_word)
    stats("(C) inter (different concepts)", inter)
    stats("(C-xl) inter cross-lingual", inter_xl)

    def _mean(arr):
        return float(np.mean(arr)) if arr else float("nan")

    print("\n=== Honest concept gap ===")
    print(f"  same-word advantage  (A) - (C)        = {_mean(intra_same_word) - _mean(inter):+.3f}")
    print(f"  TRUE concept gap     (B) - (C)        = {_mean(intra_diff_word) - _mean(inter):+.3f}")
    print(
        f"  cross-lingual gap    (B-xl) - (C-xl)  = {_mean(intra_xl_diff_word) - _mean(inter_xl):+.3f}"
    )

    # Triplet rejection using ONLY different-word positives — the honest test
    print("\n=== Triplet rejection with different-surface-word positives ===")
    gaps: list[float] = []
    xl_gaps: list[float] = []
    attempts_total = 0
    while len(gaps) < args.n_target_pairs and attempts_total < args.n_target_pairs * 50:
        attempts_total += 1
        cid = rng.choice(qualifying)
        idx = cid_to_idx[cid]
        a = rng.choice(idx)
        # positive must differ in surface word OR language
        candidates = [k for k in idx if target_words[k] != target_words[a] or langs[k] != langs[a]]
        if not candidates:
            continue
        p = rng.choice(candidates)
        # negative
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
        if langs[a] == "en" and langs[p] == "es" and langs[neg] == "es":
            xl_gaps.append(san - sap)

    arr = np.asarray(gaps)
    print(f"  triplets sampled = {len(arr)}")
    print(f"  mean gap = {arr.mean():+.3f}  p50 = {np.percentile(arr, 50):+.3f}")
    for m in [0.01, 0.05, 0.10, 0.20]:
        print(f"    margin={m:.2f}: rejection = {float(np.mean(arr + m >= 0)) * 100:5.1f}%")

    if xl_gaps:
        xl = np.asarray(xl_gaps)
        print(f"\n  cross-lingual subset: n={len(xl)}  mean={xl.mean():+.3f}")
        for m in [0.01, 0.05, 0.10, 0.20]:
            print(f"    margin={m:.2f}: rejection = {float(np.mean(xl + m >= 0)) * 100:5.1f}%")


if __name__ == "__main__":
    main()
