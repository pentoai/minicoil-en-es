"""Test JPS's hypothesis: does more data per concept widen the cosine gap?

Picks ~30 high-frequency content-word concepts (skipping stopwords), streams
Wikipedia until each concept has up to N sentences per language, encodes
locally with mE5-small (full-sentence pooling — same as cluster), and runs
the same intra-vs-inter diagnostic as before.

Comparison point: cluster baseline at ~100/lang gave gap +0.017,
rejection at margin 0.10 = 93.5%.

If JPS is right and more data widens the distribution, gap should grow
meaningfully and rejection should drop. If the geometry is encoder-bound,
both stay flat regardless of sample size.

Usage:
    uv run python research/diagnostics/test_more_data_hypothesis.py
    uv run python research/diagnostics/test_more_data_hypothesis.py --per-lang 500 --max-concepts 20
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-small"
WIKI_DUMP = "20231101"
TOKEN_RE = re.compile(r"[a-záéíóúüñàèìòùâêîôûäëïöü]+", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Function words / stopwords to filter out concepts whose words are all in this set
STOPWORDS = {
    # English
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "and",
    "or",
    "but",
    "if",
    "as",
    "at",
    "by",
    "for",
    "from",
    "on",
    "with",
    "without",
    "into",
    "onto",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "not",
    "no",
    "yes",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "her",
    "his",
    "him",
    "they",
    "them",
    "their",
    "we",
    "us",
    "our",
    "you",
    "your",
    "i",
    "my",
    "me",
    "who",
    "whom",
    "whose",
    "what",
    "which",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "could",
    "will",
    "would",
    "should",
    "may",
    "might",
    "must",
    "one",
    "two",
    "three",
    "first",
    "second",
    "another",
    "early",
    "late",
    "after",
    "before",
    "while",
    "also",
    "however",
    "though",
    "because",
    "since",
    "until",
    "about",
    "above",
    "below",
    "up",
    "down",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "now",
    "still",
    "already",
    "yet",
    "even",
    "ever",
    "never",
    "always",
    "often",
    "sometimes",
    "usually",
    "well",
    "many",
    "much",
    "less",
    "least",
    "great",
    "new",
    "old",
    "good",
    "bad",
    "high",
    "low",
    # Spanish
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "y",
    "o",
    "pero",
    "si",
    "como",
    "en",
    "de",
    "del",
    "al",
    "por",
    "para",
    "con",
    "sin",
    "es",
    "son",
    "era",
    "fue",
    "eran",
    "fueron",
    "estaba",
    "están",
    "está",
    "ser",
    "estar",
    "ha",
    "han",
    "había",
    "haber",
    "tiene",
    "tienen",
    "sí",
    "este",
    "esto",
    "esta",
    "esos",
    "esas",
    "él",
    "ella",
    "ellos",
    "ellas",
    "su",
    "sus",
    "nosotros",
    "vosotros",
    "tú",
    "yo",
    "mi",
    "mis",
    "qué",
    "quién",
    "quiénes",
    "cómo",
    "cuándo",
    "dónde",
    "porqué",
    "cuál",
    "todo",
    "todos",
    "todas",
    "alguno",
    "alguna",
    "más",
    "menos",
    "muy",
    "tan",
    "puede",
    "podría",
    "deberá",
    "debería",
    "uno",
    "dos",
    "tres",
    "primer",
    "segundo",
    "otro",
    "otra",
    "antes",
    "después",
    "mientras",
    "porque",
    "desde",
    "hasta",
    "acerca",
    "sobre",
    "ahora",
    "aquí",
    "allí",
    "ya",
    "todavía",
    "siempre",
    "nunca",
    "bien",
    "mal",
    "nuevo",
    "viejo",
    "bueno",
    "que",
    "se",
    "lo",
    "le",
    "les",
    "te",
    "nos",
    "novedad",
    "novelty",
    "novel",
}


def is_content_concept(words_en: list[str], words_es: list[str]) -> bool:
    """Concept is content-bearing if it has at least one word >=4 chars not in stopwords."""
    for w in words_en + words_es:
        if len(w) >= 4 and w.lower() not in STOPWORDS:
            return True
    return False


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def stream_for_concepts(
    lang: str,
    concept_words_by_cid: dict[str, set[str]],
    max_per_concept: int,
    max_articles: int,
) -> dict[str, list[str]]:
    """Stream Wikipedia for one language, collect sentences per concept."""
    word_to_cid: dict[str, str] = {}
    for cid, words in concept_words_by_cid.items():
        for w in words:
            word_to_cid[w] = cid
    known_words = set(word_to_cid.keys())

    sents_by_cid: dict[str, list[str]] = {cid: [] for cid in concept_words_by_cid}
    saturated: set[str] = set()

    print(f"[{lang}] Streaming wikimedia/wikipedia {WIKI_DUMP}.{lang}...")
    ds = load_dataset("wikimedia/wikipedia", f"{WIKI_DUMP}.{lang}", split="train", streaming=True)

    n_articles = 0
    n_sentences = 0
    t0 = time.perf_counter()
    for article in ds:
        if n_articles >= max_articles:
            break
        if len(saturated) >= len(concept_words_by_cid):
            break
        text = article.get("text", "") or ""
        if len(text) < 50:
            n_articles += 1
            continue
        for sentence in split_sentences(text):
            words = TOKEN_RE.findall(sentence.lower())
            if len(words) < 5 or len(words) > 100:
                continue
            n_sentences += 1
            words_set = set(words)
            matching = words_set & known_words
            if not matching:
                continue
            cids = {word_to_cid[w] for w in matching}
            for cid in cids:
                if cid in saturated:
                    continue
                if len(sents_by_cid[cid]) >= max_per_concept:
                    saturated.add(cid)
                    continue
                sents_by_cid[cid].append(sentence)
        n_articles += 1
        if n_articles % 5000 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{lang}] {n_articles:,} articles | {n_sentences:,} sent | "
                f"{len(saturated)}/{len(concept_words_by_cid)} concepts saturated | "
                f"{n_articles / elapsed:.0f} art/s"
            )
    elapsed = time.perf_counter() - t0
    print(
        f"[{lang}] done: {n_articles:,} articles in {elapsed:.0f}s, "
        f"{len(saturated)}/{len(concept_words_by_cid)} concepts saturated"
    )
    return sents_by_cid


def select_concepts(vocab, counts_en, counts_es, target: int, min_count: int) -> list[str]:
    candidates = []
    for cid, c in vocab["concepts"].items():
        en = counts_en.get(cid, 0)
        es = counts_es.get(cid, 0)
        if min(en, es) < min_count:
            continue
        if not is_content_concept(c["en"], c["es"]):
            continue
        candidates.append((cid, min(en, es)))
    candidates.sort(key=lambda x: -x[1])
    return [c for c, _ in candidates[:target]]


def stats(name, arr):
    a = np.asarray(arr)
    if len(a) == 0:
        print(f"  {name:<32} n=    0")
        return
    print(
        f"  {name:<32} n={len(a):>5}  mean={a.mean():+.3f}  std={a.std():.3f}  "
        f"p10={np.percentile(a, 10):+.3f}  p50={np.percentile(a, 50):+.3f}  p90={np.percentile(a, 90):+.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-concepts", type=int, default=30)
    parser.add_argument(
        "--per-lang", type=int, default=1000, help="Max sentences per concept per language"
    )
    parser.add_argument(
        "--min-available", type=int, default=5000, help="Min scanned count to qualify"
    )
    parser.add_argument("--max-articles", type=int, default=200_000)
    parser.add_argument("--n-pairs", type=int, default=8000)
    parser.add_argument("--n-triplets", type=int, default=8000)
    parser.add_argument("--encode-batch", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = args.device

    with open(args.data_dir / "concept_vocabulary.json") as f:
        vocab = json.load(f)
    counts = json.load(open(args.data_dir / "concept_sentence_counts.json"))["counts"]
    concept_ids = select_concepts(
        vocab, counts["en"], counts["es"], args.max_concepts, args.min_available
    )
    print(
        f"Selected {len(concept_ids)} content-word concepts with >={args.min_available} in both langs"
    )
    for cid in concept_ids[:10]:
        c = vocab["concepts"][cid]
        print(f"  {cid}  EN={c['en'][:3]}  ES={c['es'][:3]}")
    if len(concept_ids) > 10:
        print(f"  ... and {len(concept_ids) - 10} more")

    concept_words_en = {cid: set(vocab["concepts"][cid]["en"]) for cid in concept_ids}
    concept_words_es = {cid: set(vocab["concepts"][cid]["es"]) for cid in concept_ids}

    # Stream Wikipedia
    en_sents = stream_for_concepts("en", concept_words_en, args.per_lang, args.max_articles)
    es_sents = stream_for_concepts("es", concept_words_es, args.per_lang, args.max_articles)

    # Build flat arrays
    sentences: list[str] = []
    cids: list[str] = []
    langs: list[str] = []
    for cid in concept_ids:
        for s in en_sents.get(cid, []):
            sentences.append(s)
            cids.append(cid)
            langs.append("en")
        for s in es_sents.get(cid, []):
            sentences.append(s)
            cids.append(cid)
            langs.append("es")
    n = len(sentences)
    print(f"\nCollected {n} sentences total | EN: {langs.count('en')}  ES: {langs.count('es')}")
    by_concept = defaultdict(lambda: {"en": 0, "es": 0})
    for c, lg in zip(cids, langs, strict=True):
        by_concept[c][lg] += 1
    counts_minmax = [(c, min(by_concept[c]["en"], by_concept[c]["es"])) for c in concept_ids]
    counts_minmax.sort(key=lambda x: x[1])
    print(f"Min(EN,ES) per concept — worst 5: {counts_minmax[:5]}")
    print(f"Min(EN,ES) per concept — best 5: {counts_minmax[-5:]}")

    print(f"\nLoading {MODEL} on {device}...")
    model = SentenceTransformer(MODEL, device=device)
    print(f"Encoding {n} sentences with sentence pooling (batch {args.encode_batch})...")
    t0 = time.perf_counter()
    embs = model.encode(
        ["passage: " + s for s in sentences],
        batch_size=args.encode_batch,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    print(f"Encoded in {time.perf_counter() - t0:.0f}s")
    V = embs.astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12

    # Group by concept
    cid_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(cids):
        cid_to_idx[c].append(i)

    # ---------- 1. Intra vs inter ----------
    print("\n=== 1. Same-concept vs different-concept cosine (sentence pooling, MORE data) ===")
    intra: list[float] = []
    intra_xl: list[float] = []
    pairs_per_concept = max(1, args.n_pairs // len(concept_ids))
    for cid in concept_ids:
        idx = cid_to_idx[cid]
        if len(idx) < 2:
            continue
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

    print("\n=== 2. Cross-lingual ===")
    if intra_xl and inter_xl:
        stats("intra EN-ES (same concept)", intra_xl)
        stats("inter EN-ES (diff concept)", inter_xl)
        print(f"  gap = {np.mean(intra_xl) - np.mean(inter_xl):+.3f}")

    # ---------- 3. Triplet rejection ----------
    print("\n=== 3. Triplet rejection vs margin ===")
    gaps: list[float] = []
    while len(gaps) < args.n_triplets:
        cid = rng.choice(concept_ids)
        idx = cid_to_idx[cid]
        if len(idx) < 2:
            continue
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
        f"  cos(a,n) - cos(a,p):  mean={arr.mean():+.3f}  std={arr.std():.3f}  "
        f"p10={np.percentile(arr, 10):+.3f}  p50={np.percentile(arr, 50):+.3f}  p90={np.percentile(arr, 90):+.3f}"
    )
    for m in [0.01, 0.05, 0.10, 0.20, 0.30]:
        print(f"    margin={m:.2f}: rejection = {float(np.mean(arr + m >= 0)) * 100:5.1f}%")

    print("\n=== Comparison to baseline (cluster, ~100/lang, sentence pooling) ===")
    print("  baseline gap     : +0.017")
    print(f"  this run gap     : {np.mean(intra) - np.mean(inter):+.3f}")
    print("  baseline reject@0.10: 93.5%")
    print(f"  this run  reject@0.10: {float(np.mean(arr + 0.10 >= 0)) * 100:.1f}%")


if __name__ == "__main__":
    main()
