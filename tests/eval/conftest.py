"""Shared fixtures for eval tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pytest

from minicoil_v2.eval.datasets.base import Dataset


@dataclass
class _TinyDataset:
    """4-doc, 3-query in-memory dataset for unit tests."""

    name: str = "tiny"
    _corpus: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "en": {
                "d1": "cat sat",
                "d2": "dog ran",
                "d3": "bird flew",
                "d4": "fish swam",
            },
            "es": {
                "d1": "gato sentado",
                "d2": "perro corrió",
                "d3": "pájaro voló",
                "d4": "pez nadó",
            },
        }
    )
    _queries: dict[str, dict[str, dict[str, str]]] = field(
        default_factory=lambda: {
            "eng-eng": {
                "dev": {"q1": "cat", "q2": "dog", "q3": "bird"},
                "val": {"q1": "cat", "q2": "dog", "q3": "bird"},
                "test": {"q1": "cat", "q2": "dog", "q3": "bird"},
            },
        }
    )
    _qrels: dict[str, dict[str, dict[str, set[str]]]] = field(
        default_factory=lambda: {
            "eng-eng": {
                "dev": {"q1": {"d1"}, "q2": {"d2"}, "q3": {"d3"}},
                "val": {"q1": {"d1"}, "q2": {"d2"}, "q3": {"d3"}},
                "test": {"q1": {"d1"}, "q2": {"d2"}, "q3": {"d3"}},
            },
        }
    )

    def corpus(self, lang: str) -> Mapping[str, str]:
        return self._corpus[lang]

    def queries(self, pair: str, split: str) -> Mapping[str, str]:
        return self._queries[pair][split]

    def qrels(self, pair: str, split: str) -> Mapping[str, set[str]]:
        return self._qrels[pair][split]


@pytest.fixture
def tiny_dataset() -> Dataset:
    return _TinyDataset()


@dataclass
class _BigSyntheticDataset:
    """Larger in-memory dataset for split-builder tests (5500 queries per pair).

    Provides both `validation` and `test` splits so the builder can union them.
    """

    name: str = "synth"

    def corpus(self, lang: str) -> Mapping[str, str]:
        return {f"d{i}": f"text {i}" for i in range(100)}

    def queries(self, pair: str, split: str) -> Mapping[str, str]:
        # validation has 1000 qids, test has 4500 qids; union = 5500
        if split == "validation":
            return {f"v{i}": f"q {i}" for i in range(1000)}
        if split == "test":
            return {f"t{i}": f"q {i}" for i in range(4500)}
        raise KeyError(split)

    def qrels(self, pair: str, split: str) -> Mapping[str, set[str]]:
        return {qid: {"d0"} for qid in self.queries(pair, split)}


@pytest.fixture
def synth_dataset() -> Dataset:
    return _BigSyntheticDataset()
