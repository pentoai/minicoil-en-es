"""Qdrant connection management and collection operations for miniCOIL v2."""

import torch
from loguru import logger
from qdrant_client import QdrantClient, models

from minicoil_v2.constants import KNOWN_MODEL_DIMS, QDRANT_SCROLL_LIMIT
from minicoil_v2.settings import QdrantSettings


def get_client(settings: QdrantSettings, cloud_inference: bool = False) -> QdrantClient:
    """Create a Qdrant client from settings."""
    return QdrantClient(
        url=settings.url,
        api_key=settings.api_key,
        prefer_grpc=False if cloud_inference else settings.prefer_grpc,
        cloud_inference=cloud_inference,
        timeout=120 if cloud_inference else settings.timeout,
    )


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int | None,
    recreate: bool = False,
    mining_model: str | None = None,
) -> None:
    """Create the training sentences collection if it doesn't exist.

    Args:
        client: Qdrant client instance.
        collection_name: Name for the collection.
        vector_size: Dimension of the mining encoder embeddings. When None,
            mining_model must be known and is used to look up the dimension.
        recreate: If True, drop and recreate the collection.
        mining_model: Model name used for cloud inference dimension lookup.
    """
    if vector_size is None:
        if mining_model is None or mining_model not in KNOWN_MODEL_DIMS:
            raise ValueError(
                f"vector_size is None but model '{mining_model}' is not in KNOWN_MODEL_DIMS. "
                f"Known models: {list(KNOWN_MODEL_DIMS)}"
            )
        vector_size = KNOWN_MODEL_DIMS[mining_model]

    if recreate:
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection '{collection_name}'")
        except Exception:
            pass

    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        logger.info(f"Collection '{collection_name}' already exists, skipping creation")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "mining": models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
        },
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="concept_ids",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="lang",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    logger.info(
        f"Created collection '{collection_name}' "
        f"(mining vector dim={vector_size}, indexes on concept_ids + lang)"
    )


def scroll_concept(
    client: QdrantClient,
    collection_name: str,
    concept_id: str,
    max_per_lang: int = 2000,
    lang_ratio: float = 0.5,
) -> tuple[list[str], list[str], list[str], torch.Tensor]:
    """Scroll all rows for a concept, balanced across languages.

    Each row is one (sentence, focal_word, lang) tuple with a token-pooled
    vector. Returns (sentences, focal_words, langs, mining_embeddings_tensor)
    in parallel order.
    """
    filt = models.Filter(
        must=[
            models.FieldCondition(
                key="concept_ids",
                match=models.MatchValue(value=concept_id),
            )
        ]
    )

    en_sentences: list[str] = []
    en_focals: list[str] = []
    en_vectors: list[list[float]] = []
    es_sentences: list[str] = []
    es_focals: list[str] = []
    es_vectors: list[list[float]] = []

    offset = None
    while True:
        result = client.scroll(
            collection_name=collection_name,
            scroll_filter=filt,
            limit=QDRANT_SCROLL_LIMIT,
            offset=offset,
            with_vectors=["mining"],
            with_payload=True,
        )
        points, next_offset = result

        for point in points:
            lang = point.payload["lang"]  # type: ignore[index]
            sentence = point.payload["sentence"]  # type: ignore[index]
            focal = point.payload.get("focal_word", "")  # type: ignore[index]
            vector = point.vector["mining"]  # type: ignore[index]

            if lang == "en" and len(en_sentences) < max_per_lang:
                en_sentences.append(sentence)
                en_focals.append(focal)
                en_vectors.append(vector)
            elif lang == "es" and len(es_sentences) < max_per_lang:
                es_sentences.append(sentence)
                es_focals.append(focal)
                es_vectors.append(vector)

        if next_offset is None:
            break
        if len(en_sentences) >= max_per_lang and len(es_sentences) >= max_per_lang:
            break
        offset = next_offset

    n_en = len(en_sentences)
    n_es = len(es_sentences)

    if n_en > 0 and n_es > 0 and 0 < lang_ratio < 1:
        target_en = int(lang_ratio * (n_en + n_es))
        target_es = (n_en + n_es) - target_en
        target_en = min(target_en, n_en)
        target_es = min(target_es, n_es)

        if target_en < n_en:
            import random

            indices = random.sample(range(n_en), target_en)
            indices.sort()
            en_sentences = [en_sentences[i] for i in indices]
            en_focals = [en_focals[i] for i in indices]
            en_vectors = [en_vectors[i] for i in indices]
        if target_es < n_es:
            import random

            indices = random.sample(range(n_es), target_es)
            indices.sort()
            es_sentences = [es_sentences[i] for i in indices]
            es_focals = [es_focals[i] for i in indices]
            es_vectors = [es_vectors[i] for i in indices]

    sentences = en_sentences + es_sentences
    focals = en_focals + es_focals
    langs = ["en"] * len(en_sentences) + ["es"] * len(es_sentences)
    all_vectors = en_vectors + es_vectors

    if not all_vectors:
        return [], [], [], torch.empty(0)

    mining_embs = torch.tensor(all_vectors, dtype=torch.float32)
    return sentences, focals, langs, mining_embs


def scroll_concept_rematch(
    client: QdrantClient,
    collection_name: str,
    concept_id: str,
    bucket: dict,
    max_per_lang: int = 2000,
    lang_ratio: float = 0.5,
) -> tuple[list[str], list[str], list[str], torch.Tensor]:
    """Source rows for an ON-DISK concept from pre-matched point-id buckets.

    The live ``minicoil_sentences`` collection was embedded with a different
    vocab vintage, so its ``concept_ids`` payload does NOT correspond to the
    on-disk pruned vocabulary the encoder/eval use (cloud ``C-00001`` fires
    on-disk ``C-00037``, etc.). ``bucket`` (from a one-time on-disk
    ``match_concepts`` scan, ``research/scan_viable_coverage.py``) lists cloud point-ids
    whose text fires ``concept_id`` UNDER THE ON-DISK VOCAB. We fetch only those
    points' mining vectors (capped per language so the retrieve cost is bounded)
    and return the same ``(sentences, focals, langs, mining)`` contract as
    ``scroll_concept``. Focals are ``""`` placeholders: unused downstream, since
    the trainer pools by concept id, not by a stored focal word.
    """
    import random

    def _take(lang: str) -> tuple[list[str], list[list[float]]]:
        ids = (bucket.get(lang) or {}).get("ids", [])[:max_per_lang]
        sents: list[str] = []
        vecs: list[list[float]] = []
        for start in range(0, len(ids), 256):
            chunk = ids[start : start + 256]
            pts = client.retrieve(
                collection_name,
                ids=chunk,
                with_payload=["sentence", "lang"],
                with_vectors=["mining"],
            )
            for p in pts:
                sents.append(p.payload["sentence"])  # type: ignore[index]
                vecs.append(p.vector["mining"])  # type: ignore[index]
        return sents, vecs

    en_sentences, en_vectors = _take("en")
    es_sentences, es_vectors = _take("es")

    n_en = len(en_sentences)
    n_es = len(es_sentences)

    if n_en > 0 and n_es > 0 and 0 < lang_ratio < 1:
        target_en = int(lang_ratio * (n_en + n_es))
        target_es = (n_en + n_es) - target_en
        target_en = min(target_en, n_en)
        target_es = min(target_es, n_es)

        if target_en < n_en:
            idx = sorted(random.sample(range(n_en), target_en))
            en_sentences = [en_sentences[i] for i in idx]
            en_vectors = [en_vectors[i] for i in idx]
        if target_es < n_es:
            idx = sorted(random.sample(range(n_es), target_es))
            es_sentences = [es_sentences[i] for i in idx]
            es_vectors = [es_vectors[i] for i in idx]

    sentences = en_sentences + es_sentences
    langs = ["en"] * len(en_sentences) + ["es"] * len(es_sentences)
    focals = [""] * len(sentences)
    all_vectors = en_vectors + es_vectors

    if not all_vectors:
        return [], [], [], torch.empty(0)

    mining_embs = torch.tensor(all_vectors, dtype=torch.float32)
    return sentences, focals, langs, mining_embs


def upsert_batch(
    client: QdrantClient,
    collection_name: str,
    points: list[models.PointStruct],
    batch_size: int = 100,
    wait: bool = True,
) -> int:
    """Batch upsert points to Qdrant. Returns total upserted count."""
    total = 0
    for i in range(0, len(points), batch_size):
        chunk = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=chunk, wait=wait)
        total += len(chunk)
    return total
