"""BM25 baseline using FastEmbed's Qdrant/bm25 sparse encoder."""

from __future__ import annotations

from minicoil_v2.eval.retriever import register
from minicoil_v2.eval.retrievers._common import QdrantSparseRetriever


@register("bm25")
class BM25Retriever(QdrantSparseRetriever):
    name = "bm25"
    collection_prefix = "bm25"
    fastembed_model = "Qdrant/bm25"
