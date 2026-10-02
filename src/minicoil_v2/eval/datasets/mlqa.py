"""MLQA dataset adapter for the eval framework.

Loads from the official Meta tarball (https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip),
not the HuggingFace `facebook/mlqa` mirror; the HF mirror ships a legacy loader
script and is unusable with `datasets >= 3.x`. The tarball is small (~75 MB)
and content-stable since 2019, so vendoring its contents into `eval_cache/`
(gitignored) decouples eval from HuggingFace's evolving script policy.

We expose:
  - corpus(lang) = {doc_id: context_text} per language; doc_id is sha1 of the
    context text (stable, content-addressed).
  - queries(pair, split) = {qid: question_text} in the source language of pair.
  - qrels(pair, split) = {qid: {gold_doc_id}} where gold_doc_id is the sha1
    of the context in the target language of the pair.

Pair to (corpus_lang, query_lang, mlqa_file_lang_pair):
  eng-eng -> (en, en, context-en-question-en)
  spa-spa -> (es, es, context-es-question-es)
  eng-spa -> (es, en, context-es-question-en)    # EN query, ES corpus
  spa-eng -> (en, es, context-en-question-es)    # ES query, EN corpus

`split="validation"` is mapped to the tarball's `dev` directory; `split="test"`
maps to `test`. This matches the HuggingFace `datasets` split convention used
elsewhere in this project.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from urllib.request import urlopen

from loguru import logger

MLQA_URL = "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip"

PAIR_FILE: dict[str, tuple[str, str, str]] = {
    # pair: (corpus_lang, query_lang, file_lang_pair)
    "eng-eng": ("en", "en", "context-en-question-en"),
    "spa-spa": ("es", "es", "context-es-question-es"),
    "eng-spa": ("es", "en", "context-es-question-en"),
    "spa-eng": ("en", "es", "context-en-question-es"),
}

SPLIT_DIR: dict[str, str] = {
    "validation": "dev",
    "test": "test",
}


def _doc_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class MLQADataset:
    """MLQA loader sourced from the official Meta tarball.

    The zip is downloaded once into `cache_dir` and reused on subsequent runs.
    Subsequent reads access the unzipped JSON directly (the zip is kept too
    so `revision` can stay content-addressed).
    """

    name = "mlqa"

    def __init__(
        self,
        cache_dir: Path | str = "eval_cache/mlqa",
        url: str = MLQA_URL,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._url = url
        self._zip_path = self._cache_dir / "MLQA_V1.zip"
        self._extract_root = self._cache_dir / "MLQA_V1"

    def _ensure_downloaded(self) -> None:
        if self._extract_root.exists() and any(self._extract_root.iterdir()):
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        if not self._zip_path.exists():
            logger.info("Downloading MLQA tarball from {} ...", self._url)
            with (
                urlopen(self._url) as resp,  # noqa: S310 (trusted public URL)
                self._zip_path.open("wb") as out,
            ):
                out.write(resp.read())
        logger.info("Extracting MLQA tarball to {}", self._cache_dir)
        with zipfile.ZipFile(self._zip_path) as zf:
            zf.extractall(self._cache_dir)

    @cached_property
    def revision(self) -> str:
        """SHA256 of the MLQA zip; content-addressed and stable across runs."""
        self._ensure_downloaded()
        h = hashlib.sha256()
        with self._zip_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _file_path(self, pair: str, split: str) -> Path:
        if pair not in PAIR_FILE:
            raise ValueError(f"unknown pair: {pair!r}. allowed: {sorted(PAIR_FILE)}")
        if split not in SPLIT_DIR:
            raise ValueError(f"unknown split: {split!r}. allowed: {sorted(SPLIT_DIR)}")
        _, _, file_pair = PAIR_FILE[pair]
        sub = SPLIT_DIR[split]
        return self._extract_root / sub / f"{sub}-{file_pair}.json"

    @cached_property
    def _per_pair(self) -> dict[tuple[str, str], dict[str, tuple[str, str, str]]]:
        """Lazy {(pair, split): {qid -> (question, context, context_doc_id)}}."""
        self._ensure_downloaded()
        cache: dict[tuple[str, str], dict[str, tuple[str, str, str]]] = {}
        for pair in PAIR_FILE:
            for split in SPLIT_DIR:
                path = self._file_path(pair, split)
                with path.open(encoding="utf-8") as f:
                    payload = json.load(f)
                rows: dict[str, tuple[str, str, str]] = {}
                for article in payload["data"]:
                    for para in article["paragraphs"]:
                        context = para["context"]
                        cid = _doc_id(context)
                        for qa in para["qas"]:
                            rows[qa["id"]] = (qa["question"], context, cid)
                cache[(pair, split)] = rows
        return cache

    @cached_property
    def _corpora(self) -> dict[str, dict[str, str]]:
        """{lang: {doc_id: context_text}} aggregated across configs using that lang."""
        corpora: dict[str, dict[str, str]] = {"en": {}, "es": {}}
        for pair, (corpus_lang, _query_lang, _file_pair) in PAIR_FILE.items():
            for split in SPLIT_DIR:
                for _qid, (_q, ctx, doc_id) in self._per_pair[(pair, split)].items():
                    corpora[corpus_lang][doc_id] = ctx
        return corpora

    def corpus(self, lang: str) -> Mapping[str, str]:
        if lang not in self._corpora:
            raise ValueError(f"unknown lang: {lang!r}. allowed: en, es")
        return self._corpora[lang]

    def queries(self, pair: str, split: str) -> Mapping[str, str]:
        rows = self._per_pair[(pair, split)]
        return {qid: q for qid, (q, _c, _d) in rows.items()}

    def qrels(self, pair: str, split: str) -> Mapping[str, set[str]]:
        rows = self._per_pair[(pair, split)]
        return {qid: {doc_id} for qid, (_q, _c, doc_id) in rows.items()}
