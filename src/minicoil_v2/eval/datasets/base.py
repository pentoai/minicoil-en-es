"""Dataset Protocol for the eval framework."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class Dataset(Protocol):
    """A retrieval evaluation dataset.

    Implementations expose a corpus per language, queries per (pair, split),
    and qrels per (pair, split). `qrels` returns a SET of relevant doc IDs
    so the API supports multi-gold corpora; single-gold corpora (like MLQA)
    just return singleton sets.

    `pair` is a string like `"eng-eng"`, `"eng-spa"`, `"spa-eng"`, `"spa-spa"`.
    `split` is one of `"dev"`, `"val"`, `"test"`.
    """

    name: str

    def corpus(self, lang: str) -> Mapping[str, str]:
        """Return {doc_id: text} for the given language."""
        ...

    def queries(self, pair: str, split: str) -> Mapping[str, str]:
        """Return {qid: query_text} for the given pair/split."""
        ...

    def qrels(self, pair: str, split: str) -> Mapping[str, set[str]]:
        """Return {qid: {relevant_doc_id, ...}} for the given pair/split."""
        ...
