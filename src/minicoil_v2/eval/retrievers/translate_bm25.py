"""Translate-then-BM25 baseline. Translates queries to corpus language with
NLLB-200-distilled-600M, then runs BM25 on the foreign-language corpus."""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

from loguru import logger

from minicoil_v2.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from minicoil_v2.eval.retriever import (
    PAIR_LANGS,
    BaseRetriever,
    PairResult,
    _aggregate,
    register,
)
from minicoil_v2.eval.retrievers._common import (
    Translator,
    UnsupportedPairError,
)
from minicoil_v2.eval.retrievers.bm25 import BM25Retriever

DEFAULT_CACHE_DIR = Path("data/eval/translation_cache")


@register("translate-bm25")
class TranslateBM25Retriever(BaseRetriever):
    """Translates queries with NLLB to the corpus language, then BM25."""

    name = "translate-bm25"
    supported_pairs = frozenset({"eng-spa", "spa-eng"})

    def __init__(
        self,
        qdrant_url: str | None = None,
        rebuild: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        self._bm25 = BM25Retriever(qdrant_url=qdrant_url, rebuild=rebuild)
        self._translator = Translator(cache_dir=cache_dir or DEFAULT_CACHE_DIR)

    def index(self, corpus, lang):
        self._bm25.index(corpus, lang)

    def search(self, query, k):
        return self._bm25.search(query, k)

    def evaluate(self, dataset, pair: str, split: str, k: int = 100) -> PairResult:
        if pair not in self.supported_pairs:
            raise UnsupportedPairError(self.name, pair)
        src, tgt = PAIR_LANGS[pair]

        queries = dataset.queries(pair, split)
        qrels = dataset.qrels(pair, split)
        qids = list(queries.keys())
        texts = list(queries.values())

        logger.info(f"translate-bm25 {pair}: translating {len(texts)} queries {src}->{tgt}")
        translated = self._translator.translate_batch(texts, src=src, tgt=tgt)
        translation_map = dict(zip(qids, translated, strict=True))

        logger.info(f"translate-bm25 {pair}: indexing corpus ({tgt})")
        t0 = perf_counter()
        self._bm25.index(dataset.corpus(tgt), tgt)
        index_runtime = perf_counter() - t0

        ndcg: dict[str, float] = {}
        recall: dict[str, float] = {}
        mrr: dict[str, float] = {}
        ordered_hasher = hashlib.sha256()
        sorted_hasher = hashlib.sha256()

        n = len(qids)
        logger.info(f"translate-bm25 {pair}: searching {n} translated queries (k={k})")
        t1 = perf_counter()
        for i, qid in enumerate(qids, start=1):
            translated_q = translation_map[qid]
            ranked_with_scores = self._bm25.search(translated_q, k=k)
            ranked = [doc_id for doc_id, _ in ranked_with_scores]
            relevant = qrels.get(qid, set())
            ndcg[qid] = ndcg_at_k(ranked, relevant, k=10)
            recall[qid] = recall_at_k(ranked, relevant, k=100)
            mrr[qid] = mrr_at_k(ranked, relevant, k=10)
            ordered_hasher.update(qid.encode("utf-8") + b"\x1f")
            ordered_hasher.update("|".join(ranked).encode("utf-8") + b"\x1e")
            sorted_hasher.update(qid.encode("utf-8") + b"\x1f")
            sorted_hasher.update("|".join(sorted(ranked)).encode("utf-8") + b"\x1e")
            if i % 1000 == 0:
                rate = i / max(perf_counter() - t1, 1e-9)
                logger.info(f"  translate-bm25 {pair}: {i}/{n} searched ({rate:.0f}/s)")
        runtime = perf_counter() - t1

        ndcg_agg = _aggregate(ndcg)
        recall_agg = _aggregate(recall)
        mrr_agg = _aggregate(mrr)
        logger.info(
            f"translate-bm25 {pair}/{split} done: ndcg10={ndcg_agg['mean']:.4f} "
            f"r100={recall_agg['mean']:.4f} mrr10={mrr_agg['mean']:.4f} "
            f"(translate+search {runtime:.1f}s, index {index_runtime:.1f}s)"
        )
        return PairResult(
            pair=pair,
            split=split,
            n_queries=len(qids),
            metrics={
                "ndcg10": ndcg_agg,
                "r100": recall_agg,
                "mrr10": mrr_agg,
            },
            runtime_sec=runtime,
            index_runtime_sec=index_runtime,
            rank_digest_ordered=ordered_hasher.hexdigest(),
            rank_digest_sorted=sorted_hasher.hexdigest(),
        )
