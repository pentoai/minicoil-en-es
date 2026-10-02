"""Encode Wikipedia sentences with the teacher (mining encoder) and upsert to Qdrant.

Each stored row is one (sentence, focal_word, lang) triple. The "mining" vector is
the teacher's view of that row:

- ``--mining-pooling token_pooled`` (default): the teacher's last hidden states
  averaged over the subword tokens that overlap the focal word's character span,
  then L2-normalized.
- ``--mining-pooling sentence``: the normalized sentence embedding (the published
  checkpoint used Qwen/Qwen3-Embedding-0.6B this way), shared by every focal word of
  the sentence.

Why per-(sentence, focal_word) rows: the concept-discrimination signal lives at
the word, not at the whole sentence. A sentence containing two matched concept
words is two training instances. The teacher vectors only rank positives and
negatives at training time; the heads' input is re-encoded by the trainer.

Usage:
    minicoil embed --max-articles 1000
    minicoil embed --max-articles 500000
"""

import hashlib
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_dataset
from loguru import logger
from qdrant_client import QdrantClient, models

from minicoil_v2.constants import (
    INPUT_ENCODER,
    MAX_PER_ARTICLE_PER_CONCEPT,
    MAX_SENTENCE_WORDS,
    MIN_ARTICLE_LENGTH,
    MIN_SENTENCE_WORDS,
    PROGRESS_EARLY_THRESHOLD,
    PROGRESS_EVERY,
    PROGRESS_EVERY_EARLY,
    SENTENCE_SPLIT_RE,
    WIKI_DUMP_DATE,
)
from minicoil_v2.qdrant_store import (
    ensure_collection,
    get_client,
    upsert_batch,
)
from minicoil_v2.settings import EmbedSettings
from minicoil_v2.token_pooling import encode_sentences_with_focals, load_token_pool_model
from minicoil_v2.utils import select_device, tokenize_alpha

if sys.platform == "darwin":
    torch.Tensor.share_memory_ = lambda self: self  # type: ignore[assignment]


# Cap for sentence-mode mining encoders. Qwen3-Embedding ships with a 32k
# max_seq_length; SentenceTransformer sorts inputs by length, so one batch of
# the longest Wikipedia "sentences" can OOM a 24GB GPU (observed on A10G,
# 2026-06-10). 512 tokens covers real sentences with a wide margin.
MINING_SENTENCE_MAX_SEQ = 512


def load_sentence_model(model_name: str, device: torch.device):
    """Load a SentenceTransformer mining encoder with a bounded max_seq_length."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=str(device))
    model.max_seq_length = min(model.max_seq_length, MINING_SENTENCE_MAX_SEQ)
    return model


def row_uuid(sentence: str, focal_word: str, lang: str) -> str:
    """Deterministic UUID from (lang, sentence, focal_word) for point dedup."""
    h = hashlib.sha256(f"{lang}:{focal_word}:{sentence}".encode()).digest()
    return str(uuid.UUID(bytes=h[:16]))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences at sentence-ending punctuation."""
    parts = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def _flush_buffer(
    buffer: dict[str, dict],
    lang: str,
    tokenizer,
    model,
    device: torch.device,
    client: QdrantClient,
    collection_name: str,
    encode_batch_size: int,
    upsert_batch_size: int,
    upsert_wait: bool,
    mining_pooling: str = "token_pooled",
    cloud_inference: bool = False,
    mining_encoder: str = "",
    openrouter_api_key: str | None = None,
) -> tuple[int, float, float]:
    """Encode each (sentence, focal_word) and upsert. Returns (rows, encode_s, upsert_s)."""
    if not buffer:
        return 0, 0.0, 0.0

    sentences = list(buffer.keys())
    focals_per_sentence: list[list[str]] = [sorted(buffer[s]["pairs"].keys()) for s in sentences]

    if cloud_inference:
        assert mining_encoder, (
            "mining_encoder must be a non-empty model name when cloud_inference is True"
        )
        encode_s = 0.0
        vecs_per_sentence = [[None for _ in focals] for focals in focals_per_sentence]
    else:
        t_enc = time.perf_counter()
        if mining_pooling == "sentence":
            encoded = model.encode(
                sentences,
                batch_size=encode_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vecs_per_sentence = [
                [encoded[i].copy() for _ in focals] for i, focals in enumerate(focals_per_sentence)
            ]
        else:
            assert tokenizer is not None, "tokenizer must be provided for token_pooled mode"
            vecs_per_sentence = encode_sentences_with_focals(
                sentences,
                focals_per_sentence,
                tokenizer,
                model,
                device,
                batch_size=encode_batch_size,
            )
        encode_s = time.perf_counter() - t_enc

    if cloud_inference and mining_pooling != "sentence":
        raise ValueError("Qdrant cloud inference only supports sentence mining_pooling")

    points: list[models.PointStruct] = []
    n_dropped = 0
    for sentence, focals, vecs in zip(
        sentences, focals_per_sentence, vecs_per_sentence, strict=True
    ):
        meta = buffer[sentence]
        for focal, vec in zip(focals, vecs, strict=True):
            if not cloud_inference and float((vec * vec).sum()) < 0.5:
                n_dropped += 1
                continue
            cid = meta["pairs"][focal]
            uid = row_uuid(sentence, focal, lang)
            vector: dict
            if cloud_inference:
                options = {"openrouter-api-key": openrouter_api_key} if openrouter_api_key else {}
                vector = {
                    "mining": models.Document(
                        text="passage: " + sentence, model=mining_encoder, options=options
                    )
                }
            else:
                vector = {"mining": vec.tolist()}
            points.append(
                models.PointStruct(
                    id=uid,
                    vector=vector,
                    payload={
                        "concept_ids": [cid],
                        "focal_word": focal,
                        "lang": lang,
                        "sentence": sentence,
                        "article_id": meta["article_id"],
                    },
                )
            )

    if n_dropped:
        logger.debug(f"[{lang}] dropped {n_dropped} rows with no matching tokens")

    t_up = time.perf_counter()
    n = upsert_batch(
        client,
        collection_name,
        points,
        batch_size=upsert_batch_size,
        wait=upsert_wait,
    )
    upsert_s = time.perf_counter() - t_up

    return n, encode_s, upsert_s


def process_language(
    lang: str,
    word_to_concept: dict[str, str],
    client: QdrantClient,
    collection_name: str,
    tokenizer,
    model,
    device: torch.device,
    max_articles: int,
    store_cap: int,
    encode_batch_size: int,
    flush_buffer_size: int,
    upsert_batch_size: int,
    upsert_wait: bool,
    mining_pooling: str = "token_pooled",
    cloud_inference: bool = False,
    mining_encoder: str = "",
    openrouter_api_key: str | None = None,
) -> None:
    """Stream Wikipedia, encode concept sentences, upsert to Qdrant."""
    known_words = set(word_to_concept.keys())
    all_concept_ids = set(word_to_concept.values())
    total_concepts = len(all_concept_ids)

    stored_counts: dict[str, int] = defaultdict(int)
    saturated: set[str] = set()

    wiki_config = f"{WIKI_DUMP_DATE}.{lang}"
    logger.info(f"[{lang}] Loading wikimedia/wikipedia ({wiki_config}) streaming...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        wiki_config,
        split="train",
        streaming=True,
    )

    buffer: dict[str, dict] = {}
    n_articles = 0
    n_sentences_scanned = 0
    n_stored = 0
    n_upserted = 0
    n_flushes = 0
    total_encode_s = 0.0
    total_upsert_s = 0.0
    t0 = time.perf_counter()
    t_last_progress = t0

    def _do_flush() -> None:
        nonlocal n_upserted, n_flushes, total_encode_s, total_upsert_s
        n, enc_s, up_s = _flush_buffer(
            buffer,
            lang,
            tokenizer,
            model,
            device,
            client,
            collection_name,
            encode_batch_size,
            upsert_batch_size,
            upsert_wait,
            mining_pooling,
            cloud_inference=cloud_inference,
            mining_encoder=mining_encoder,
            openrouter_api_key=openrouter_api_key,
        )
        n_upserted += n
        n_flushes += 1
        total_encode_s += enc_s
        total_upsert_s += up_s
        buffer.clear()

    def _should_log_progress() -> bool:
        if n_articles <= PROGRESS_EARLY_THRESHOLD:
            return n_articles % PROGRESS_EVERY_EARLY == 0
        return n_articles % PROGRESS_EVERY == 0

    def _log_progress() -> None:
        nonlocal t_last_progress
        now = time.perf_counter()
        elapsed = now - t0
        interval = now - t_last_progress
        rate = n_articles / elapsed if elapsed > 0 else 0
        n_sat = len(saturated)
        scan_s = elapsed - total_encode_s - total_upsert_s
        logger.info(
            f"[{lang}] {n_articles:,} articles | "
            f"{n_sentences_scanned:,} sent scanned | "
            f"{n_stored:,} stored | {n_upserted:,} upserted | "
            f"{total_concepts - n_sat:,}/{total_concepts:,} concepts open | "
            f"{rate:.0f} art/s | "
            f"scan={scan_s:.1f}s encode={total_encode_s:.1f}s "
            f"upsert={total_upsert_s:.1f}s "
            f"({n_flushes} flushes, interval={interval:.1f}s)"
        )
        t_last_progress = now

    for article in dataset:
        if n_articles >= max_articles:
            break

        text = article.get("text", "")
        if not text or len(text) < MIN_ARTICLE_LENGTH:
            n_articles += 1
            continue

        article_id = str(article.get("id", n_articles))
        sentences = split_sentences(text)
        article_concept_hits: dict[str, int] = defaultdict(int)

        for sentence in sentences:
            words = tokenize_alpha(sentence)
            n_words = len(words)
            if n_words < MIN_SENTENCE_WORDS or n_words > MAX_SENTENCE_WORDS:
                continue

            n_sentences_scanned += 1
            matching = words & known_words
            if not matching:
                continue

            store_pairs: dict[str, str] = {}
            for w in matching:
                cid = word_to_concept[w]
                if cid in saturated:
                    continue
                if stored_counts[cid] >= store_cap:
                    saturated.add(cid)
                    continue
                if article_concept_hits[cid] >= MAX_PER_ARTICLE_PER_CONCEPT:
                    continue
                store_pairs[w] = cid
                stored_counts[cid] += 1
                article_concept_hits[cid] += 1
                n_stored += 1

            if store_pairs:
                if sentence in buffer:
                    buffer[sentence]["pairs"].update(store_pairs)
                else:
                    buffer[sentence] = {
                        "pairs": store_pairs,
                        "article_id": article_id,
                    }

        n_articles += 1

        if len(buffer) >= flush_buffer_size:
            _do_flush()

        if _should_log_progress():
            _log_progress()

    if buffer:
        _do_flush()

    elapsed = time.perf_counter() - t0
    scan_s = elapsed - total_encode_s - total_upsert_s
    logger.info(
        f"[{lang}] Done — {n_articles:,} articles in {elapsed:.1f}s | "
        f"{n_sentences_scanned:,} sentences | "
        f"{n_upserted:,} upserted ({n_flushes} flushes) | "
        f"{len(stored_counts):,}/{total_concepts:,} concepts stored | "
        f"scan={scan_s:.1f}s encode={total_encode_s:.1f}s upsert={total_upsert_s:.1f}s"
    )


def run_embedding(settings: EmbedSettings) -> None:
    """Encode Wikipedia sentences with token pooling and upsert into Qdrant."""
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

    device = select_device(settings.device)
    model_name = settings.mining_encoder or INPUT_ENCODER
    cloud_inference = settings.cloud_inference
    if cloud_inference and settings.mining_pooling != "sentence":
        raise ValueError("Qdrant cloud inference only supports sentence mining_pooling")

    if cloud_inference:
        logger.info(
            "Cloud inference enabled, skipping local encoder load "
            f"(model: {settings.mining_encoder})"
        )
        tokenizer = None
        model = None
        vector_size = None
    elif settings.mining_pooling == "sentence":
        logger.info(f"Device: {device}")
        logger.info(f"Loading sentence model: {model_name}...")
        tokenizer = None
        model = load_sentence_model(model_name, device)
        vector_size = model.get_sentence_embedding_dimension()
        logger.info(f"Sentence embedding dim: {vector_size} (max_seq={model.max_seq_length})")
    else:
        logger.info(f"Device: {device}")
        logger.info(f"Loading token-pool model: {model_name}...")
        tokenizer, model = load_token_pool_model(model_name, device)
        vector_size = model.config.hidden_size
        logger.info(f"Hidden dim: {vector_size}")

    logger.info(f"Connecting to Qdrant at {settings.qdrant.url}...")
    client = get_client(settings.qdrant, cloud_inference=cloud_inference)
    ensure_collection(
        client,
        settings.qdrant.collection_name,
        vector_size=vector_size,
        recreate=settings.recreate,
        mining_model=settings.mining_encoder if cloud_inference else None,
    )

    langs = ["en", "es"] if settings.lang == "both" else [settings.lang]

    for lang in langs:
        logger.info(
            f"Wikipedia ({lang}) — max {settings.max_articles:,} articles, "
            f"store cap {settings.store_cap}, "
            f"flush buffer {settings.flush_buffer_size}, "
            f"encode batch {settings.encode_batch_size}, "
            f"upsert batch {settings.upsert_batch_size}, "
            f"wait={settings.upsert_wait}"
        )
        process_language(
            lang=lang,
            word_to_concept=w2c[lang],
            client=client,
            collection_name=settings.qdrant.collection_name,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_articles=settings.max_articles,
            store_cap=settings.store_cap,
            encode_batch_size=settings.encode_batch_size,
            flush_buffer_size=settings.flush_buffer_size,
            upsert_batch_size=settings.upsert_batch_size,
            upsert_wait=settings.upsert_wait,
            mining_pooling=settings.mining_pooling,
            cloud_inference=cloud_inference,
            mining_encoder=settings.mining_encoder,
            openrouter_api_key=settings.openrouter_api_key,
        )

    info = client.get_collection(settings.qdrant.collection_name)
    logger.info(f"Qdrant collection: {info.points_count} points")


if __name__ == "__main__":
    run_embedding(EmbedSettings())
