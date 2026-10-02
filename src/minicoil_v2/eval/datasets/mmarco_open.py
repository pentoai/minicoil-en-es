"""Open-domain mMARCO adapter: gold pool + non-gold distractors.

Wraps MMARCODataset to turn its gold-only corpus into a real open-domain
haystack by mixing in N non-gold passages streamed from the full collection.
Gold passages, queries, and qrels are unchanged, so this masquerades as the
``mmarco`` dataset (same name + revision) and reuses the frozen
splits_mmarco_phase2 qids; only ``corpus()`` grows. Distractors appear in no
qrels, so they are pure ranking noise. Read the fix ON/OFF delta, not absolutes
(MS MARCO qrels are sparse, so some distractors are relevant-but-unjudged and
depress absolutes uniformly; the delta cancels that).

N is set per language via env (MINICOIL_MMARCO_OPEN_N_EN / _N_ES), default 0.
At N=0 this is byte-identical to mmarco (the validity gate).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from loguru import logger

from minicoil_v2.eval.datasets.mmarco import (
    _REMOTE_COLLECTION,
    LANG_NAME,
    MMARCO_REPO,
    MMARCODataset,
    _doc_id,
)


class MMARCOOpenDataset:
    """mMARCO with a distractor-augmented corpus (true open-domain haystack)."""

    name = "mmarco"  # masquerade: reuse mmarco splits + manifest revision gate

    def __init__(
        self,
        cache_dir: Path | str = "eval_cache/mmarco_open",
        n_en: int | None = None,
        n_es: int | None = None,
        base: MMARCODataset | None = None,
    ) -> None:
        self._base = base or MMARCODataset()
        self._cache_dir = Path(cache_dir)
        self._n = {
            "en": n_en
            if n_en is not None
            else int(os.environ.get("MINICOIL_MMARCO_OPEN_N_EN", "0")),
            "es": n_es
            if n_es is not None
            else int(os.environ.get("MINICOIL_MMARCO_OPEN_N_ES", "0")),
        }

    @property
    def revision(self) -> str:
        # Delegate so the splits manifest revision gate passes against the frozen
        # mmarco splits. Distractor count is deliberately NOT content-addressed:
        # qids/qrels are identical across N, only the haystack grows.
        return self._base.revision

    def _gold_pids(self) -> set[str]:
        self._base._ensure_files()  # noqa: SLF001
        return {
            line.split("\t")[2]
            for line in self._base._local("qrels.dev.small.tsv")  # noqa: SLF001
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        }

    def _distractor_cache(self, lang: str) -> Path:
        return self._cache_dir / f"{LANG_NAME[lang]}_distractors.tsv"

    def _ensure_distractors(self, lang: str, n: int) -> None:
        """Stream the first n non-gold passages for `lang` into the cache TSV."""
        dst = self._distractor_cache(lang)
        if dst.exists():
            have = sum(1 for line in dst.open(encoding="utf-8") if line.strip())
            if have >= n:
                return
        from huggingface_hub import HfFileSystem

        gold = self._gold_pids()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        remote = f"datasets/{MMARCO_REPO}/{_REMOTE_COLLECTION[lang]}"
        logger.info("mmarco_open: streaming {} non-gold {} distractors from {}", n, lang, remote)
        fs = HfFileSystem()
        written = 0
        with (
            fs.open(remote, "r", encoding="utf-8", revision=self._base.revision) as src,
            dst.open("w", encoding="utf-8") as out,
        ):
            for line in src:
                pid = line.split("\t", 1)[0]
                if pid in gold:
                    continue
                out.write(line if line.endswith("\n") else line + "\n")
                written += 1
                if written >= n:
                    break
        logger.info("mmarco_open: cached {} {} distractors", written, lang)

    def _distractors(self, lang: str, n: int) -> dict[str, str]:
        self._ensure_distractors(lang, n)
        out: dict[str, str] = {}
        for line in self._distractor_cache(lang).read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            _pid, _, text = line.partition("\t")
            out[_doc_id(text)] = text
            if len(out) >= n:
                break
        return out

    def corpus(self, lang: str) -> Mapping[str, str]:
        base = dict(self._base.corpus(lang))
        n = self._n.get(lang, 0)
        if n > 0:
            for doc_id, text in self._distractors(lang, n).items():
                base.setdefault(doc_id, text)  # gold wins on collision
        return base

    def queries(self, pair: str, split: str) -> Mapping[str, str]:
        return self._base.queries(pair, split)

    def qrels(self, pair: str, split: str) -> Mapping[str, set[str]]:
        return self._base.qrels(pair, split)
