"""mMARCO dataset adapter for the eval framework.

Sourced from the raw Google-translation TSVs in the ``unicamp-dl/mmarco`` HF
dataset repo, NOT its `datasets` loader script (which is unusable on
`datasets >= 3.x`, exactly like MLQA's HF mirror). The small files (queries,
qrels) are downloaded via ``hf_hub_download``; the two multi-GB collections are
*streamed* and reduced to only the judged (gold) passages, so the on-disk cache
stays small. Everything is cached under ``eval_cache/`` (gitignored) and the
dataset revision is pinned for content-addressing.

We expose (matching ``MLQADataset``):
  - corpus(lang) = {doc_id: passage_text}; doc_id is sha1 of the passage text.
    The Phase-0 corpus is the dev.small gold-passage pool rendered in `lang`.
  - queries(pair, split) = {qid: query_text} in the source language of pair.
  - qrels(pair, split) = {qid: {gold_doc_id}}; the cross-lingual gold is the same
    MS MARCO passage id translated into the pair's corpus language.

Cross-lingual alignment (verified against sampled rows): the same integer `qid`
indexes both the English and Spanish query files, and the same integer `pid`
indexes both collections; `qrels.dev.small.tsv` maps `qid -> pid`
language-independently. MS MARCO passage ranking has no public test qrels, so the
judged `dev.small` pool is exposed under split ``"validation"``; ``"test"`` is empty.

Pair to (query_lang, corpus_lang):
  eng-eng -> (en, en)
  spa-spa -> (es, es)
  eng-spa -> (en, es)    # EN query, ES corpus
  spa-eng -> (es, en)    # ES query, EN corpus
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path

from loguru import logger

MMARCO_REPO = "unicamp-dl/mmarco"
MMARCO_REVISION = "6d039c4638c0ba3e46a9cb7b498b145e7edc6230"

PAIR_LANGS: dict[str, tuple[str, str]] = {
    "eng-eng": ("en", "en"),
    "spa-spa": ("es", "es"),
    "eng-spa": ("en", "es"),
    "spa-eng": ("es", "en"),
}

LANG_NAME: dict[str, str] = {"en": "english", "es": "spanish"}

# Remote paths within the dataset repo (Google translation variant).
_REMOTE_QUERIES = {
    "en": "data/google/queries/dev/english_queries.dev.small.tsv",
    "es": "data/google/queries/dev/spanish_queries.dev.small.tsv",
}
_REMOTE_QRELS = "data/qrels.dev.small.tsv"
_REMOTE_COLLECTION = {
    "en": "data/google/collections/english_collection.tsv",
    "es": "data/google/collections/spanish_collection.tsv",
}


def _doc_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class MMARCODataset:
    """mMARCO loader sourced from raw Google-translation TSVs."""

    name = "mmarco"

    def __init__(
        self,
        cache_dir: Path | str = "eval_cache/mmarco",
        revision: str = MMARCO_REVISION,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._revision = revision

    # --- file acquisition -------------------------------------------------

    def _local(self, name: str) -> Path:
        return self._cache_dir / name

    def _ensure_files(self) -> None:
        """Download queries+qrels and stream-extract the gold sub-collections."""
        from huggingface_hub import hf_hub_download

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        small = {
            "english_queries.dev.small.tsv": _REMOTE_QUERIES["en"],
            "spanish_queries.dev.small.tsv": _REMOTE_QUERIES["es"],
            "qrels.dev.small.tsv": _REMOTE_QRELS,
        }
        for local_name, remote in small.items():
            dst = self._local(local_name)
            if dst.exists():
                continue
            logger.info("mMARCO: downloading {}", remote)
            src = hf_hub_download(
                repo_id=MMARCO_REPO,
                repo_type="dataset",
                filename=remote,
                revision=self._revision,
            )
            shutil.copy(src, dst)

        need_collections = [
            lang
            for lang in ("en", "es")
            if not self._local(f"{LANG_NAME[lang]}_collection.tsv").exists()
        ]
        if need_collections:
            gold_pids = {
                line.split("\t")[2]
                for line in self._local("qrels.dev.small.tsv")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            }
            for lang in need_collections:
                self._extract_gold(lang, gold_pids)

    def _extract_gold(self, lang: str, gold_pids: set[str]) -> None:
        """Stream the (multi-GB) collection and keep only gold-pid rows."""
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        remote = f"datasets/{MMARCO_REPO}/{_REMOTE_COLLECTION[lang]}"
        dst = self._local(f"{LANG_NAME[lang]}_collection.tsv")
        logger.info("mMARCO: extracting {} gold passages from {}", len(gold_pids), remote)
        found = 0
        with (
            fs.open(remote, "r", encoding="utf-8", revision=self._revision) as src,
            dst.open("w", encoding="utf-8") as out,
        ):
            for line in src:
                pid = line.split("\t", 1)[0]
                if pid in gold_pids:
                    out.write(line if line.endswith("\n") else line + "\n")
                    found += 1
        logger.info("mMARCO: wrote {} gold passages for {}", found, lang)

    # --- parsing ----------------------------------------------------------

    @staticmethod
    def _read_tsv2(path: Path) -> dict[str, str]:
        rows: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            key, _, val = line.partition("\t")
            rows[key] = val
        return rows

    @cached_property
    def _collection(self) -> dict[str, dict[str, str]]:
        self._ensure_files()
        return {
            lang: self._read_tsv2(self._local(f"{LANG_NAME[lang]}_collection.tsv"))
            for lang in ("en", "es")
        }

    @cached_property
    def _queries_by_lang(self) -> dict[str, dict[str, str]]:
        self._ensure_files()
        return {
            "en": self._read_tsv2(self._local("english_queries.dev.small.tsv")),
            "es": self._read_tsv2(self._local("spanish_queries.dev.small.tsv")),
        }

    @cached_property
    def _qid_to_pids(self) -> dict[str, set[str]]:
        self._ensure_files()
        out: dict[str, set[str]] = {}
        for line in self._local("qrels.dev.small.tsv").read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            qid, _zero, pid, _rel = line.split("\t")
            out.setdefault(qid, set()).add(pid)
        return out

    @cached_property
    def _corpora(self) -> dict[str, dict[str, str]]:
        return {
            lang: {_doc_id(text): text for text in self._collection[lang].values()}
            for lang in ("en", "es")
        }

    # --- Dataset protocol -------------------------------------------------

    @property
    def revision(self) -> str:
        return self._revision

    def corpus(self, lang: str) -> Mapping[str, str]:
        if lang not in self._corpora:
            raise ValueError(f"unknown lang: {lang!r}. allowed: en, es")
        return self._corpora[lang]

    def queries(self, pair: str, split: str) -> Mapping[str, str]:
        if pair not in PAIR_LANGS:
            raise ValueError(f"unknown pair: {pair!r}. allowed: {sorted(PAIR_LANGS)}")
        if split != "validation":
            return {}
        query_lang, _ = PAIR_LANGS[pair]
        q = self._queries_by_lang[query_lang]
        return {qid: q[qid] for qid in self._qid_to_pids if qid in q}

    def qrels(self, pair: str, split: str) -> Mapping[str, set[str]]:
        if pair not in PAIR_LANGS:
            raise ValueError(f"unknown pair: {pair!r}. allowed: {sorted(PAIR_LANGS)}")
        if split != "validation":
            return {}
        _, corpus_lang = PAIR_LANGS[pair]
        coll = self._collection[corpus_lang]
        out: dict[str, set[str]] = {}
        for qid, pids in self._qid_to_pids.items():
            golds = {_doc_id(coll[pid]) for pid in pids if pid in coll}
            if golds:
                out[qid] = golds
        return out
