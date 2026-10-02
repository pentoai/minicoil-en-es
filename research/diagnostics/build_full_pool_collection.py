"""Build minicoil_full_pool: mirror of minicoil_token_pool with sentence-pool vectors.

For each point in minicoil_token_pool, write a matching point in minicoil_full_pool
(same point ID, same payload), but with the mE5-small sentence-pooled vector
(mean over all token last-hidden-states with attention mask) in place of the
token-pooled vector.

Two passes:
  1. Scroll token_pool, collect unique (sentence, lang) pairs.
  2. Encode unique pairs once in batches.
  3. Scroll token_pool again, upsert into full_pool with the sentence-pool vector.

The two collections are now identical in (point_id, payload) and differ only in
the stored vector — exact apples-to-apples for training.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator

import numpy as np
import torch
from loguru import logger
from qdrant_client import QdrantClient, models
from transformers import AutoModel, AutoTokenizer

from minicoil_v2.constants import INPUT_ENCODER
from minicoil_v2.token_pooling import PASSAGE_PREFIX

SRC_COLLECTION = "minicoil_token_pool"
DST_COLLECTION = "minicoil_full_pool"
SCROLL_LIMIT = 1024
ENCODE_BATCH = 64
UPSERT_BATCH = 256


def pick_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def scroll_all(client: QdrantClient, collection: str, with_vectors: bool = False) -> Iterator[list]:
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection,
            limit=SCROLL_LIMIT,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        if not batch:
            break
        yield batch
        if next_offset is None:
            break
        offset = next_offset


@torch.no_grad()
def encode_sentence_pool(
    sentences: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:
    """mE5 mean-pool over all tokens (with attention mask), L2 normalized."""
    inputs = tokenizer(
        [PASSAGE_PREFIX + s for s in sentences],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)
    out = model(**inputs)
    hidden = out.last_hidden_state  # (B, T, D)
    mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
    summed = (hidden * mask).sum(dim=1)  # (B, D)
    counts = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
    pooled = summed / counts
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--rebuild", action="store_true", help="Drop dst collection if exists.")
    parser.add_argument(
        "--limit-concepts",
        type=str,
        default=None,
        help="Comma-separated concept IDs to limit the build to. Otherwise all.",
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    logger.info(f"Device: {device}")

    client = QdrantClient(url=args.qdrant_url, prefer_grpc=False)

    # Determine the set of point IDs to copy.
    filt = None
    if args.limit_concepts:
        ids = [c.strip() for c in args.limit_concepts.split(",") if c.strip()]
        filt = models.Filter(
            must=[
                models.FieldCondition(
                    key="concept_ids",
                    match=models.MatchAny(any=ids),
                )
            ]
        )
        logger.info(f"Limiting to {len(ids)} concept(s)")

    # --- ensure dst collection ---
    existing = {c.name for c in client.get_collections().collections}
    if DST_COLLECTION in existing and args.rebuild:
        client.delete_collection(DST_COLLECTION)
        logger.info(f"Deleted existing {DST_COLLECTION}")
        existing.discard(DST_COLLECTION)
    if DST_COLLECTION not in existing:
        client.create_collection(
            collection_name=DST_COLLECTION,
            vectors_config={
                "mining": models.VectorParams(size=384, distance=models.Distance.COSINE)
            },
        )
        client.create_payload_index(
            collection_name=DST_COLLECTION,
            field_name="concept_ids",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=DST_COLLECTION,
            field_name="lang",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.info(f"Created {DST_COLLECTION}")

    # --- pass 1: collect unique (sentence, lang) and corresponding point IDs ---
    logger.info(f"Pass 1: scrolling {SRC_COLLECTION} for unique sentences...")
    t0 = time.time()
    seen_sentences: dict[tuple[str, str], int] = {}
    point_records: list[tuple] = []  # (point_id, payload, sentence_key)
    offset = None
    n = 0
    while True:
        batch, next_offset = client.scroll(
            SRC_COLLECTION,
            limit=SCROLL_LIMIT,
            offset=offset,
            scroll_filter=filt,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            break
        for p in batch:
            key = (p.payload["sentence"], p.payload["lang"])
            if key not in seen_sentences:
                seen_sentences[key] = len(seen_sentences)
            point_records.append((p.id, p.payload, key))
        n += len(batch)
        if n % 10240 == 0:
            logger.info(f"  scrolled {n} points, {len(seen_sentences)} unique sentences so far")
        if next_offset is None:
            break
        offset = next_offset

    n_unique = len(seen_sentences)
    n_total = len(point_records)
    logger.info(
        f"Pass 1 done in {time.time() - t0:.1f}s: {n_total} points, {n_unique} unique sentences"
    )

    # --- encode unique sentences ---
    logger.info(f"Loading {INPUT_ENCODER}...")
    tokenizer = AutoTokenizer.from_pretrained(INPUT_ENCODER)
    model = AutoModel.from_pretrained(INPUT_ENCODER).to(device).eval()

    sentences_ordered: list[str] = [None] * n_unique  # type: ignore[list-item]
    for (sent, _lang), idx in seen_sentences.items():
        sentences_ordered[idx] = sent

    logger.info(f"Encoding {n_unique} unique sentences in batches of {ENCODE_BATCH}...")
    t0 = time.time()
    sentence_vecs = np.zeros((n_unique, 384), dtype=np.float32)
    for start in range(0, n_unique, ENCODE_BATCH):
        end = min(start + ENCODE_BATCH, n_unique)
        chunk = sentences_ordered[start:end]
        sentence_vecs[start:end] = encode_sentence_pool(chunk, tokenizer, model, device)
        if (start // ENCODE_BATCH) % 50 == 0:
            elapsed = time.time() - t0
            rate = (end) / elapsed if elapsed > 0 else 0
            eta = (n_unique - end) / rate if rate > 0 else 0
            logger.info(f"  encoded {end}/{n_unique}  rate={rate:.0f}/s  eta={eta:.0f}s")
    logger.info(f"Encoding done in {time.time() - t0:.1f}s")

    # --- pass 2: upsert into dst ---
    logger.info(f"Pass 2: upserting {n_total} points into {DST_COLLECTION}...")
    t0 = time.time()
    buf: list[models.PointStruct] = []
    written = 0
    for pid, payload, key in point_records:
        vec_idx = seen_sentences[key]
        buf.append(
            models.PointStruct(
                id=pid,
                payload=payload,
                vector={"mining": sentence_vecs[vec_idx].tolist()},
            )
        )
        if len(buf) >= UPSERT_BATCH:
            client.upsert(collection_name=DST_COLLECTION, points=buf, wait=False)
            written += len(buf)
            buf.clear()
            if written % (UPSERT_BATCH * 20) == 0:
                logger.info(f"  upserted {written}/{n_total}")
    if buf:
        client.upsert(collection_name=DST_COLLECTION, points=buf, wait=True)
        written += len(buf)
    logger.info(f"Pass 2 done in {time.time() - t0:.1f}s: {written} points written")

    info = client.get_collection(DST_COLLECTION)
    logger.info(f"{DST_COLLECTION}: points={info.points_count}, status={info.status}")


if __name__ == "__main__":
    main()
