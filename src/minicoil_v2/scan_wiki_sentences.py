"""Scan Wikipedia and count concept occurrences per language.

Streams Wikipedia articles, splits into sentences, tokenizes, and counts how
many times each concept appears. Pure CPU: no encoding, no Qdrant. Output is
consumed by the prune step to decide which concepts survive.

Usage:
    minicoil scan --max-articles 1000
    minicoil scan --max-articles 500000
"""

import json
import time
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from loguru import logger

from minicoil_v2.constants import (
    MAX_SENTENCE_WORDS,
    MIN_ARTICLE_LENGTH,
    MIN_SENTENCE_WORDS,
    PROGRESS_EARLY_THRESHOLD,
    PROGRESS_EVERY,
    PROGRESS_EVERY_EARLY,
    SENTENCE_SPLIT_RE,
    WIKI_DUMP_DATE,
)
from minicoil_v2.settings import ScanSettings
from minicoil_v2.utils import tokenize_alpha


def split_sentences(text: str) -> list[str]:
    """Split text into sentences at sentence-ending punctuation."""
    parts = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def count_concepts_for_language(
    lang: str,
    word_to_concept: dict[str, str],
    max_articles: int,
) -> dict[str, int]:
    """Stream Wikipedia for one language and count concept occurrences."""
    known_words = set(word_to_concept.keys())
    all_concept_ids = set(word_to_concept.values())
    total_concepts = len(all_concept_ids)

    concept_counts: dict[str, int] = defaultdict(int)

    wiki_config = f"{WIKI_DUMP_DATE}.{lang}"
    logger.info(f"[{lang}] Loading wikimedia/wikipedia ({wiki_config}) streaming...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        wiki_config,
        split="train",
        streaming=True,
    )

    n_articles = 0
    n_sentences_scanned = 0
    t0 = time.perf_counter()

    for article in dataset:
        if n_articles >= max_articles:
            break

        text = article.get("text", "")
        if not text or len(text) < MIN_ARTICLE_LENGTH:
            n_articles += 1
            continue

        for sentence in split_sentences(text):
            words = tokenize_alpha(sentence)
            n_words = len(words)
            if n_words < MIN_SENTENCE_WORDS or n_words > MAX_SENTENCE_WORDS:
                continue

            n_sentences_scanned += 1
            matching = words & known_words
            if not matching:
                continue

            for w in matching:
                concept_counts[word_to_concept[w]] += 1

        n_articles += 1

        early = n_articles <= PROGRESS_EARLY_THRESHOLD
        step = PROGRESS_EVERY_EARLY if early else PROGRESS_EVERY
        if n_articles % step == 0:
            elapsed = time.perf_counter() - t0
            rate = n_articles / elapsed if elapsed > 0 else 0
            logger.info(
                f"[{lang}] {n_articles:,} articles | "
                f"{n_sentences_scanned:,} sentences | "
                f"{len(concept_counts):,}/{total_concepts:,} concepts seen | "
                f"{rate:.0f} art/s"
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        f"[{lang}] Done — {n_articles:,} articles in {elapsed:.1f}s | "
        f"{n_sentences_scanned:,} sentences | "
        f"{len(concept_counts):,}/{total_concepts:,} concepts seen"
    )
    return dict(concept_counts)


def run_scan(settings: ScanSettings) -> None:
    """Scan Wikipedia and write per-concept sentence counts."""
    data_dir = Path(settings.data_dir)

    logger.info("Loading word_to_concept...")
    with open(data_dir / "word_to_concept.json") as f:
        w2c = json.load(f)

    with open(data_dir / "concept_vocabulary.json") as f:
        total_concepts = len(json.load(f)["concepts"])
    logger.info(
        f"Vocabulary: {total_concepts:,} concepts | "
        f"{len(w2c['en']):,} EN words | {len(w2c['es']):,} ES words"
    )

    langs = ["en", "es"] if settings.lang == "both" else [settings.lang]
    all_counts: dict[str, dict[str, int]] = {}

    for lang in langs:
        logger.info(f"Wikipedia ({lang}) — max {settings.max_articles:,} articles")
        all_counts[lang] = count_concepts_for_language(
            lang=lang,
            word_to_concept=w2c[lang],
            max_articles=settings.max_articles,
        )

    en_c = all_counts.get("en", {})
    es_c = all_counts.get("es", {})
    if en_c and es_c:
        all_cids = set(en_c) | set(es_c)
        both = sum(1 for cid in all_cids if en_c.get(cid, 0) > 0 and es_c.get(cid, 0) > 0)
        logger.info(
            f"Coverage: {len(all_cids):,}/{total_concepts:,} concepts, both langs: {both:,}"
        )

    output = {
        "config": {"max_articles": settings.max_articles},
        "counts": all_counts,
    }
    counts_path = data_dir / "concept_sentence_counts.json"
    with open(counts_path, "w") as f:
        json.dump(output, f)
    logger.info(f"Counts saved to {counts_path}")


if __name__ == "__main__":
    run_scan(ScanSettings())
