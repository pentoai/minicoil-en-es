"""miniCOIL v2 retriever: the per-concept bilingual model, wrapped for the eval
harness.

Encodes text to sparse vectors (8-value concept blocks plus a BM25 backbone) with
`MiniCoilEncoder` and scores
them in Qdrant with `Modifier.IDF`, the exact scoring path used by miniCOIL v1,
so the eng-eng comparison is apples-to-apples. It is bilingual: documents are
encoded in the corpus language and queries in the query language, so all four
pairs (eng-eng, spa-spa, eng-spa, spa-eng) are supported.

The checkpoint defaults to the published model on the Hugging Face Hub
(`constants.PUBLISHED_MODEL_ID`, downloaded and loaded with its recorded inference
defaults). Point `MINICOIL_V2_CHECKPOINT` (or the `model_path` ctor arg) at a local
`concept_layers.pt` / `model.safetensors` to score any trained checkpoint instead,
with `MINICOIL_V2_DATA_DIR` naming the vocabulary it was trained against.

NOTE: this module is imported at retriever-discovery time, so its top-level
imports stay light (no torch). The encoder and token-pooling prefixes are
imported lazily in `_build_encoder`.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping

import numpy as np

from minicoil_v2.constants import PUBLISHED_MODEL_ID, XL_BACKBONE_WEIGHT
from minicoil_v2.eval.retriever import PAIR_LANGS, register
from minicoil_v2.eval.retrievers._common import QdrantSparseRetriever


class _SparseVec:
    """Minimal FastEmbed-style sparse embedding exposing `.indices`/`.values`
    (sorted by index) so it drops into QdrantSparseRetriever's index/search."""

    def __init__(self, sparse: Mapping[int, float]) -> None:
        items = sorted(sparse.items())
        self.indices = np.array([i for i, _ in items], dtype=np.int64)
        self.values = np.array([v for _, v in items], dtype=np.float32)


class _MiniCoilAdapter:
    """Adapt MiniCoilEncoder to the passage_embed/query_embed interface
    QdrantSparseRetriever expects. `doc_lang` and `query_lang` are set by the
    retriever so the bilingual model encodes each side in the right language."""

    def __init__(self, encoder, passage_prefix: str, query_prefix: str) -> None:
        self._encoder = encoder
        self._passage_prefix = passage_prefix
        self._query_prefix = query_prefix
        self.doc_lang = "en"
        self.query_lang = "en"
        # Backbone downweight applied to cross-lingual queries (query_lang != doc_lang).
        # Default from constants; overridable for sweeps via the env var.
        self._xl_backbone_weight = float(
            os.environ.get("MINICOIL_V2_XL_BACKBONE_WEIGHT", XL_BACKBONE_WEIGHT)
        )

    def passage_embed(self, texts: Iterable[str], **kwargs) -> Iterator[_SparseVec]:
        dicts = self._encoder.encode_batch_sparse(
            list(texts), lang=self.doc_lang, prefix=self._passage_prefix
        )
        for d in dicts:
            yield _SparseVec(d)

    def query_embed(self, texts: Iterable[str], **kwargs) -> Iterator[_SparseVec]:
        # is_query: fastembed-style BM25 asymmetry — flat 1.0 weights on the
        # query side, tf/length-normalized weights on the document side.
        # Cross-lingual queries downweight the BM25 backbone (mostly noise across
        # languages); same-language queries keep it at full weight. The doc index
        # is encoded at full backbone, so IDF and same-language scoring are unchanged.
        backbone_weight = self._xl_backbone_weight if self.query_lang != self.doc_lang else 1.0
        dicts = self._encoder.encode_batch_sparse(
            list(texts),
            lang=self.query_lang,
            prefix=self._query_prefix,
            is_query=True,
            backbone_weight=backbone_weight,
        )
        for d in dicts:
            yield _SparseVec(d)


@register("minicoil-v2")
class MiniCoilV2Retriever(QdrantSparseRetriever):
    """Per-concept bilingual miniCOIL v2, scored with Qdrant IDF like v1."""

    name = "minicoil-v2"
    collection_prefix = "minicoil_v2"
    fastembed_model = ""  # unused; encoder is built in _build_encoder
    supported_pairs = None  # bilingual: all four pairs

    def __init__(
        self,
        qdrant_url: str | None = None,
        rebuild: bool = False,
        model_path: str | None = None,
        data_dir: str = "data",
    ) -> None:
        # None = the published model (see _build_encoder).
        self._model_path = model_path or os.environ.get("MINICOIL_V2_CHECKPOINT")
        # The vocab vintage must match the trained checkpoint's concept ids. The
        # phase2 model is trained against data/phase2's vocab, so the encoder must
        # load the same dir; overridable via env so the harness call stays untouched.
        self._data_dir = (
            data_dir if data_dir != "data" else os.environ.get("MINICOIL_V2_DATA_DIR", "data")
        )
        super().__init__(qdrant_url=qdrant_url, rebuild=rebuild)

    def _build_encoder(self):
        from minicoil_v2.encoder import MiniCoilEncoder
        from minicoil_v2.token_pooling import PASSAGE_PREFIX, QUERY_PREFIX

        if self._model_path is None:
            encoder = MiniCoilEncoder.from_pretrained(PUBLISHED_MODEL_ID)
        else:
            encoder = MiniCoilEncoder(self._data_dir, model_path=self._model_path)
        return _MiniCoilAdapter(encoder, PASSAGE_PREFIX, QUERY_PREFIX)

    def index(self, corpus, lang):
        self._encoder.doc_lang = lang
        super().index(corpus, lang)

    def evaluate(self, dataset, pair: str, split: str, k: int = 100):
        q_lang, _ = PAIR_LANGS[pair]
        self._encoder.query_lang = q_lang
        return super().evaluate(dataset, pair=pair, split=split, k=k)
