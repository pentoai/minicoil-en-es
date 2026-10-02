"""Shared utilities for baseline retrievers.

Contains:
- sha256_corpus: deterministic hash of a corpus.
- CorpusHashMismatch, UnsupportedPairError: typed errors.
- QdrantSparseRetriever: base class for sparse retrievers backed by Qdrant.
- Translator: NLLB-200-distilled-600M wrapper with disk cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient, models

from minicoil_v2.eval.retriever import BaseRetriever


def sha256_corpus(corpus: Mapping[str, str]) -> str:
    """Return a deterministic SHA-256 hex digest of a corpus.

    Independent of dict insertion order. Doc IDs are sorted before hashing.
    """
    h = hashlib.sha256()
    for doc_id in sorted(corpus):
        h.update(doc_id.encode("utf-8"))
        h.update(b"\x1f")
        h.update(corpus[doc_id].encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


class CorpusHashMismatch(RuntimeError):
    """Raised when a Qdrant collection exists but its sidecar corpus_sha
    does not match the corpus being indexed."""

    def __init__(self, collection: str, stored_sha: str, new_sha: str) -> None:
        self.collection = collection
        self.stored_sha = stored_sha
        self.new_sha = new_sha
        super().__init__(
            f"corpus mismatch on collection {collection!r}: "
            f"stored={stored_sha[:8]} new={new_sha[:8]}. "
            f"pass --rebuild or delete the collection to re-index."
        )


class UnsupportedPairError(ValueError):
    """Raised when a retriever cannot handle a (src_lang, tgt_lang) pair."""

    def __init__(self, retriever: str, pair: str) -> None:
        self.retriever = retriever
        self.pair = pair
        super().__init__(f"retriever {retriever!r} does not support pair {pair!r}")


def _build_qdrant_client(url: str | None) -> QdrantClient:
    """Factory wrapped so tests can monkeypatch.

    Uses gRPC (one multiplexed connection) instead of HTTP. The eval search
    loop issues tens of thousands of sequential queries; HTTP opens a new
    connection per call and exhausts local ephemeral ports (Errno 49).
    """
    return QdrantClient(
        url=url or os.environ.get("QDRANT_URL", "http://localhost:6333"),
        prefer_grpc=True,
    )


def _meta_dir() -> Path:
    return Path(os.environ.get("MINICOIL_QDRANT_META_DIR", "data/eval/qdrant_meta"))


class QdrantSparseRetriever(BaseRetriever):
    """Base for retrievers whose encoder produces sparse vectors stored in
    Qdrant with Modifier.IDF. Subclasses set `collection_prefix` and
    `fastembed_model`, and optionally `supported_pairs`."""

    name: str = ""
    collection_prefix: str = ""
    fastembed_model: str = ""
    supported_pairs: frozenset[str] | None = None
    # Distinguishes collections per dataset so mlqa and mmarco corpora never share
    # a collection (a corpus_sha mismatch there would raise). Defaults to "mlqa"
    # for backward compatibility; the CLI sets it from --dataset.
    dataset_tag: str = "mlqa"

    def __init__(self, qdrant_url: str | None = None, rebuild: bool = False) -> None:
        self._client = _build_qdrant_client(qdrant_url)
        self._encoder = self._build_encoder()
        self._rebuild = rebuild
        self._collection: str | None = None

    def _build_encoder(self):
        """Hook for tests to inject a fake encoder."""
        from fastembed import SparseTextEmbedding

        # Cap ONNX intra-op threads. Uncapped, FastEmbed oversubscribes a small
        # box (observed load ~54 on 4 vCPU) and the encode crawls. Tunable via env.
        threads = int(os.environ.get("MINICOIL_FASTEMBED_THREADS", "4"))
        return SparseTextEmbedding(model_name=self.fastembed_model, threads=threads)

    def _sidecar_path(self, collection: str) -> Path:
        return _meta_dir() / f"{collection}.json"

    def _read_sidecar(self, collection: str) -> dict | None:
        path = self._sidecar_path(collection)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_sidecar(self, collection: str, corpus_sha: str, n_docs: int) -> None:
        path = self._sidecar_path(collection)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "corpus_sha": corpus_sha,
                    "n_docs": n_docs,
                    "indexed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def index(self, corpus: Mapping[str, str], lang: str) -> None:
        self._collection = f"{self.collection_prefix}_{lang}_{self.dataset_tag}"
        new_sha = sha256_corpus(corpus)
        sidecar = self._read_sidecar(self._collection)
        collection_exists = self._client.collection_exists(self._collection)

        if sidecar and collection_exists and not self._rebuild:
            if sidecar["corpus_sha"] == new_sha:
                logger.info(f"reusing collection {self._collection} (corpus_sha match)")
                return
            raise CorpusHashMismatch(self._collection, sidecar["corpus_sha"], new_sha)

        if collection_exists:
            self._client.delete_collection(self._collection)
        sidecar_path = self._sidecar_path(self._collection)
        if sidecar_path.exists():
            sidecar_path.unlink()

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={},
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )

        doc_ids = sorted(corpus)
        logger.info(
            f"indexing {len(doc_ids)} docs into {self._collection} "
            f"via {self.fastembed_model or self.name}"
        )
        # Encode + upload in chunks so peak memory stays bounded. Encoding the
        # whole corpus at once (list(passage_embed(all))) OOMs on a 16GB box for
        # a ~11k-doc corpus of long Wikipedia paragraphs.
        #
        # encode_batch bounds the FastEmbed forward batch. FastEmbed pads every
        # batch to the longest sequence in it, and MLQA docs run to ~2.3k tokens
        # (median 153). With the default batch_size=256, one long doc forces a
        # 256 x 2330 padded forward that takes minutes on CPU and spikes memory
        # (a 16-doc batch of the longest docs is ~1900x slower than 16 short
        # ones). A small batch keeps that padding waste local: the full corpus
        # indexes in ~7 min instead of stalling on the first chunk.
        upload_batch = 512
        encode_batch = 16
        uploaded = 0
        for start in range(0, len(doc_ids), upload_batch):
            chunk_ids = doc_ids[start : start + upload_batch]
            chunk_texts = [corpus[d] for d in chunk_ids]
            points = [
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self._collection}/{doc_id}")),
                    vector={
                        "sparse": models.SparseVector(
                            indices=emb.indices.tolist(),
                            values=emb.values.tolist(),
                        ),
                    },
                    payload={"doc_id": doc_id},
                )
                for doc_id, emb in zip(
                    chunk_ids,
                    self._encoder.passage_embed(chunk_texts, batch_size=encode_batch),
                    strict=True,
                )
            ]
            self._client.upload_points(collection_name=self._collection, points=points, wait=True)
            uploaded += len(points)
            logger.info(f"  indexed {uploaded}/{len(doc_ids)} docs")
        self._write_sidecar(self._collection, new_sha, len(doc_ids))
        logger.info(f"indexed {len(doc_ids)} docs into {self._collection}")

    def evaluate(self, dataset, pair: str, split: str, k: int = 100):
        if self.supported_pairs is not None and pair not in self.supported_pairs:
            raise UnsupportedPairError(self.name, pair)
        return super().evaluate(dataset, pair=pair, split=split, k=k)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._collection is None:
            raise RuntimeError("search() called before index()")
        sparse = next(iter(self._encoder.query_embed([query])))
        hits = self._client.query_points(
            collection_name=self._collection,
            query=models.SparseVector(
                indices=sparse.indices.tolist(),
                values=sparse.values.tolist(),
            ),
            using="sparse",
            limit=k,
            with_payload=["doc_id"],
        ).points
        return [(p.payload["doc_id"], float(p.score)) for p in hits]


NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
NLLB_FALLBACK_SHA = "distilled-600M-fallback"


def _nllb_revision_cache_path() -> Path:
    return Path.home() / ".cache" / "minicoil_v2" / "nllb_revision"


def _fetch_nllb_revision_from_hf() -> str:
    """Fetch the snapshot SHA of NLLB-200-distilled-600M from the HF API.

    Returns the commit SHA string. Raises if offline or API unreachable.
    """
    from huggingface_hub import HfApi

    info = HfApi().model_info(NLLB_MODEL_NAME)
    return info.sha


def resolve_nllb_revision_sha() -> str:
    """Return the cached NLLB revision SHA, fetching once if absent.

    On first call, tries the HF API; on failure, writes a sentinel.
    Either way, persists the value to a per-user cache so subsequent
    calls return it without network access.
    """
    cache_path = _nllb_revision_cache_path()
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8").strip()
    try:
        sha = _fetch_nllb_revision_from_hf()
    except Exception:  # noqa: BLE001
        sha = NLLB_FALLBACK_SHA
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(sha + "\n", encoding="utf-8")
    return sha


NLLB_LANG_CODES = {"en": "eng_Latn", "es": "spa_Latn"}


def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class Translator:
    """Disk-cached NLLB-200-distilled-600M translator."""

    def __init__(self, cache_dir: Path, device: str | None = None) -> None:
        self._cache_dir = Path(cache_dir)
        self._device = device or _auto_device()
        self._model = None
        self._tokenizer = None
        self._model_sha = resolve_nllb_revision_sha()

    def _cache_path(self, text: str, src: str, tgt: str) -> Path:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self._cache_dir / self._model_sha / f"{src}-{tgt}" / f"{key}.txt"

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info(f"loading {NLLB_MODEL_NAME} onto {self._device}")
        self._tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
        self._model.to(self._device)
        self._model.eval()
        logger.info("NLLB model ready")

    def translate(self, text: str, src: str, tgt: str) -> str:
        cache_path = self._cache_path(text, src, tgt)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        self._lazy_load()
        out = self._translate_uncached([text], src, tgt)[0]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(out, encoding="utf-8")
        return out

    def translate_batch(
        self, texts: list[str], src: str, tgt: str, batch_size: int = 32
    ) -> list[str]:
        """Translate many texts, hitting cache where possible and batching misses."""
        results: list[str | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        for i, text in enumerate(texts):
            path = self._cache_path(text, src, tgt)
            if path.exists():
                results[i] = path.read_text(encoding="utf-8")
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        logger.info(
            f"translate {src}->{tgt}: {len(texts)} texts "
            f"({len(texts) - len(miss_texts)} cached, {len(miss_texts)} to generate) "
            f"on {self._device}"
        )
        if miss_texts:
            self._lazy_load()
            for start in range(0, len(miss_texts), batch_size):
                batch = miss_texts[start : start + batch_size]
                translated = self._translate_uncached(batch, src, tgt)
                for j, out in enumerate(translated):
                    idx = miss_indices[start + j]
                    results[idx] = out
                    path = self._cache_path(texts[idx], src, tgt)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(out, encoding="utf-8")
                done = min(start + batch_size, len(miss_texts))
                logger.info(f"  translated {done}/{len(miss_texts)}")

        assert all(r is not None for r in results)
        return [r for r in results if r is not None]

    def _translate_uncached(self, texts: list[str], src: str, tgt: str) -> list[str]:
        """Generate translations for `texts`. Caller is responsible for caching."""
        import torch

        src_code = NLLB_LANG_CODES[src]
        tgt_code = NLLB_LANG_CODES[tgt]
        self._tokenizer.src_lang = src_code
        encoded = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_code)
        with torch.no_grad():
            generated = self._model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                max_length=256,
            )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)
