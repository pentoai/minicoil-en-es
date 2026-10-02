"""miniCOIL v1 baseline using FastEmbed's Qdrant/minicoil-v1 sparse encoder."""

from __future__ import annotations

from minicoil_v2.eval.retriever import register
from minicoil_v2.eval.retrievers._common import QdrantSparseRetriever


@register("minicoil-v1")
class MiniCoilV1Retriever(QdrantSparseRetriever):
    name = "minicoil-v1"
    collection_prefix = "minicoil_v1"
    fastembed_model = "Qdrant/minicoil-v1"
    supported_pairs = frozenset({"eng-eng"})
