"""Triplet quality audit: dump sampled triplets as readable markdown.

Streams Wikipedia EN+ES for a handful of concept word families, encodes each
sentence with token pooling around the concept word, builds the within-concept
cosine distance matrix, runs the BilingualSampler's within-concept logic
(random pair, label by distance, reject if margin < min_margin), and dumps N
triplets per concept with the actual sentences and distances side by side.

No Qdrant needed: all sentences pulled fresh from streaming Wikipedia, all
vectors computed locally.

Goal: judge whether sl_pos/xl_pos are "more similar" than sl_neg/xl_neg in a
semantically meaningful way (same sense, same register), or if the distances
look arbitrary.

Usage:
    uv run python research/diagnostics/diagnose_triplet_quality.py
"""

import argparse
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

MODEL = "intfloat/multilingual-e5-small"
PREFIX = "passage: "
TOKEN_RE = re.compile(r"[a-záéíóúüñàèìòùâêîôûäëïöü]+", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WIKI_DUMP = "20231101"
MIN_MARGIN = 0.1

# (concept_id, label, EN targets, ES targets)
CONCEPTS = [
    {
        "id": "USE",
        "label": "use/usar (functional verb, expected weakly polysemic)",
        "en": {"use", "used", "uses", "using"},
        "es": {
            "usar",
            "usa",
            "usan",
            "usado",
            "usada",
            "usados",
            "usadas",
            "utilizar",
            "utiliza",
            "utilizan",
            "utilizado",
            "utilizada",
            "utilizados",
            "utilizadas",
        },
    },
    {
        "id": "PLANT",
        "label": "plant/planta (polysemic: botanical / industrial / verb)",
        "en": {"plant", "plants", "planted", "planting"},
        "es": {"planta", "plantas", "plantar", "plantó", "plantado", "plantada", "plantando"},
    },
    {
        "id": "SPRING",
        "label": "spring/primavera (highly polysemic: season/water/mechanical/verb)",
        "en": {"spring", "springs"},
        "es": {
            "primavera",
            "primaveras",
            "manantial",
            "manantiales",
            "muelle",
            "muelles",
            "resorte",
            "resortes",
        },
    },
]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def stream_for_concepts(
    lang: str, concepts: list[dict], n_per_concept: int, max_articles: int = 200_000
) -> dict[str, list[str]]:
    """One pass through wiki(lang); fill buckets for all concepts until each has n_per_concept."""
    targets_per_cid = {c["id"]: c[lang] for c in concepts}
    all_targets: set[str] = set()
    for t in targets_per_cid.values():
        all_targets |= t

    buckets: dict[str, list[str]] = {c["id"]: [] for c in concepts}
    print(f"[{lang}] streaming wikimedia/wikipedia {WIKI_DUMP}.{lang}...")
    ds = load_dataset("wikimedia/wikipedia", f"{WIKI_DUMP}.{lang}", split="train", streaming=True)
    t0 = time.perf_counter()
    n_articles = 0
    for article in ds:
        if n_articles >= max_articles or all(len(buckets[cid]) >= n_per_concept for cid in buckets):
            break
        text = article.get("text", "") or ""
        if len(text) < 50:
            n_articles += 1
            continue
        for sent in split_sentences(text):
            words = TOKEN_RE.findall(sent.lower())
            if len(words) < 5 or len(words) > 80:
                continue
            wset = set(words)
            if not (wset & all_targets):
                continue
            for cid, tgt in targets_per_cid.items():
                if len(buckets[cid]) >= n_per_concept:
                    continue
                if wset & tgt:
                    buckets[cid].append(sent)
                    break  # don't double-count one sentence into multiple buckets
        n_articles += 1
        if n_articles % 5000 == 0:
            filled = " ".join(f"{cid}={len(buckets[cid])}" for cid in buckets)
            print(f"  [{lang}] {n_articles:,} arts | {filled} ({(time.perf_counter() - t0):.0f}s)")
    elapsed = time.perf_counter() - t0
    filled = " ".join(f"{cid}={len(buckets[cid])}" for cid in buckets)
    print(f"[{lang}] done: {filled} in {n_articles:,} articles ({elapsed:.0f}s)")
    return buckets


def find_target(sentence: str, target_words: set[str]) -> str | None:
    for w in TOKEN_RE.findall(sentence.lower()):
        if w in target_words:
            return w
    return None


def encode_token_pooled(sentences, targets, tokenizer, model, device, batch_size=32):
    all_vecs = []
    valid = []
    for start in range(0, len(sentences), batch_size):
        chunk_s = sentences[start : start + batch_size]
        chunk_t = targets[start : start + batch_size]
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
        for b, (text, target) in enumerate(zip(texts, chunk_t, strict=True)):
            if target is None:
                valid.append(False)
                all_vecs.append(np.zeros(hidden.shape[-1], dtype=np.float32))
                continue
            text_lower = text.lower()
            pos = text_lower.find(target.lower())
            if pos < 0:
                valid.append(False)
                all_vecs.append(np.zeros(hidden.shape[-1], dtype=np.float32))
                continue
            word_end = pos + len(target)
            tok_idx = []
            for i, (s, e) in enumerate(offsets_all[b].tolist()):
                if s == 0 and e == 0:
                    continue
                if s < word_end and e > pos:
                    tok_idx.append(i)
            if not tok_idx:
                valid.append(False)
                all_vecs.append(np.zeros(hidden.shape[-1], dtype=np.float32))
                continue
            v = hidden[b, tok_idx].mean(dim=0)
            v = v / (v.norm() + 1e-12)
            all_vecs.append(v.numpy().astype(np.float32))
            valid.append(True)
    return np.vstack(all_vecs), np.asarray(valid)


def sample_triplets(vectors, langs, n_triplets, min_margin, seed=42):
    """Replicate BilingualSampler._sample_one + _pick_pair within-concept logic."""
    rng = random.Random(seed)
    n = len(vectors)
    if n < 5:
        return [], {"attempts": 0, "rejections": 0}

    sim = vectors @ vectors.T
    dist = 1.0 - sim

    by_lang: dict[str, list[int]] = {}
    for i, lg in enumerate(langs):
        by_lang.setdefault(lg, []).append(i)
    all_langs = list(by_lang.keys())
    bilingual = len(all_langs) >= 2

    triplets = []
    rejections = 0
    attempts = 0
    max_attempts = n_triplets * 200

    while len(triplets) < n_triplets and attempts < max_attempts:
        attempts += 1
        anchor_lang = rng.choice(all_langs)
        a_pool = by_lang[anchor_lang]
        if len(a_pool) < 3:
            rejections += 1
            continue
        a = rng.choice(a_pool)
        a_dists = dist[a]
        same = [i for i in a_pool if i != a]
        if len(same) < 2:
            rejections += 1
            continue

        x, y = rng.sample(same, 2)
        m_sl = abs(a_dists[x] - a_dists[y])
        if m_sl < min_margin:
            rejections += 1
            continue
        sl_pos, sl_neg = (x, y) if a_dists[x] < a_dists[y] else (y, x)

        xl_pos, xl_neg, m_xl, xl_valid = a, a, 0.0, False
        if bilingual:
            others = [lg for lg in all_langs if lg != anchor_lang]
            o = rng.choice(others)
            o_pool = by_lang[o]
            if len(o_pool) >= 2:
                x, y = rng.sample(o_pool, 2)
                m_xl = abs(a_dists[x] - a_dists[y])
                if m_xl >= min_margin:
                    xl_pos, xl_neg = (x, y) if a_dists[x] < a_dists[y] else (y, x)
                    xl_valid = True

        if not xl_valid:
            rejections += 1
            continue

        triplets.append(
            {
                "a": a,
                "sl_pos": sl_pos,
                "sl_neg": sl_neg,
                "xl_pos": xl_pos,
                "xl_neg": xl_neg,
                "m_sl": float(m_sl),
                "m_xl": float(m_xl),
                "d_sl_pos": float(a_dists[sl_pos]),
                "d_sl_neg": float(a_dists[sl_neg]),
                "d_xl_pos": float(a_dists[xl_pos]),
                "d_xl_neg": float(a_dists[xl_neg]),
            }
        )

    return triplets, {"attempts": attempts, "rejections": rejections}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/triplet_audit.md")
    parser.add_argument("--n-triplets", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-articles", type=int, default=200_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    else:
        device = args.device

    print(f"Loading {MODEL} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()

    # Pull EN and ES in two passes through Wikipedia
    en_buckets = stream_for_concepts("en", CONCEPTS, args.n_samples, args.max_articles)
    es_buckets = stream_for_concepts("es", CONCEPTS, args.n_samples, args.max_articles)

    out_lines = [
        "# Triplet quality audit (token-pooled mE5-small, Wikipedia streamed)\n\n",
        f"Sampler: BilingualSampler within-concept logic, min_margin={MIN_MARGIN}.\n",
        f"Per concept: up to {args.n_samples}/lang sentences from streamed Wikipedia, ",
        "token-pooled around the concept word, distance matrix built locally, ",
        f"{args.n_triplets} triplets dumped.\n\n",
    ]

    for concept in CONCEPTS:
        cid = concept["id"]
        label = concept["label"]
        target_set = concept["en"] | concept["es"]
        en_sents = en_buckets.get(cid, [])
        es_sents = es_buckets.get(cid, [])
        sentences = en_sents + es_sents
        langs = ["en"] * len(en_sents) + ["es"] * len(es_sents)
        print(f"\n=== {cid}: {label} ===")
        print(f"Sentences: {len(en_sents)} EN, {len(es_sents)} ES")
        if len(en_sents) < 5 or len(es_sents) < 5:
            out_lines.append(
                f"\n## {cid} — {label}\n\n_Skipped: not enough sentences ({len(en_sents)} EN, {len(es_sents)} ES)_\n"
            )
            continue

        targets = [find_target(s, target_set) for s in sentences]
        vecs, valid = encode_token_pooled(sentences, targets, tokenizer, model, device)
        idx = np.where(valid)[0]
        vecs = vecs[idx]
        sentences = [sentences[i] for i in idx]
        langs = [langs[i] for i in idx]
        targets = [targets[i] for i in idx]
        print(f"Valid token-pool: {len(idx)} ({langs.count('en')} EN, {langs.count('es')} ES)")

        triplets, stats = sample_triplets(vecs, langs, args.n_triplets, MIN_MARGIN, seed=42)
        reject_rate = stats["rejections"] / max(stats["attempts"], 1)
        print(f"Sampled {len(triplets)} triplets, rejection rate {reject_rate:.1%}")

        if triplets:
            m_sl = float(np.mean([t["m_sl"] for t in triplets]))
            m_xl = float(np.mean([t["m_xl"] for t in triplets]))
            d_sl_p = float(np.mean([t["d_sl_pos"] for t in triplets]))
            d_sl_n = float(np.mean([t["d_sl_neg"] for t in triplets]))
            d_xl_p = float(np.mean([t["d_xl_pos"] for t in triplets]))
            d_xl_n = float(np.mean([t["d_xl_neg"] for t in triplets]))
        else:
            m_sl = m_xl = d_sl_p = d_sl_n = d_xl_p = d_xl_n = 0.0

        out_lines.append(f"\n## {cid} — {label}\n\n")
        out_lines.append(
            f"- N sentences: {len(idx)} ({langs.count('en')} EN, {langs.count('es')} ES)\n"
        )
        out_lines.append(f"- Sampler rejection at min_margin={MIN_MARGIN}: **{reject_rate:.1%}**\n")
        out_lines.append(f"- Mean margins: sl={m_sl:.3f}, xl={m_xl:.3f}\n")
        out_lines.append(f"- Mean d(anchor, sl_pos)={d_sl_p:.3f}  d(anchor, sl_neg)={d_sl_n:.3f}\n")
        out_lines.append(
            f"- Mean d(anchor, xl_pos)={d_xl_p:.3f}  d(anchor, xl_neg)={d_xl_n:.3f}\n\n"
        )

        for ti, t in enumerate(triplets):
            out_lines.append(f"### {cid} triplet {ti + 1}\n\n")
            out_lines.append(
                "| role | lang | target | d(a,·) | sentence |\n|---|---|---|---|---|\n"
            )
            out_lines.append(
                f"| **anchor** | {langs[t['a']]} | `{targets[t['a']]}` | — | {sentences[t['a']]} |\n"
            )
            out_lines.append(
                f"| sl_pos | {langs[t['sl_pos']]} | `{targets[t['sl_pos']]}` | {t['d_sl_pos']:.3f} | {sentences[t['sl_pos']]} |\n"
            )
            out_lines.append(
                f"| sl_neg | {langs[t['sl_neg']]} | `{targets[t['sl_neg']]}` | {t['d_sl_neg']:.3f} | {sentences[t['sl_neg']]} |\n"
            )
            out_lines.append(
                f"| xl_pos | {langs[t['xl_pos']]} | `{targets[t['xl_pos']]}` | {t['d_xl_pos']:.3f} | {sentences[t['xl_pos']]} |\n"
            )
            out_lines.append(
                f"| xl_neg | {langs[t['xl_neg']]} | `{targets[t['xl_neg']]}` | {t['d_xl_neg']:.3f} | {sentences[t['xl_neg']]} |\n\n"
            )
            out_lines.append(f"margin_sl={t['m_sl']:.3f}, margin_xl={t['m_xl']:.3f}\n\n---\n\n")

    Path(args.out).write_text("".join(out_lines))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
