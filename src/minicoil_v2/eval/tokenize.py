"""Shared tokenizer for the eval framework.

Re-exports the project's TOKEN_RE so the BM25 baseline and any miniCOIL
retriever stay aligned with concept-matching at retrieval time.
"""

from __future__ import annotations

from minicoil_v2.constants import TOKEN_RE

__all__ = ["TOKEN_RE", "tokenize"]


def tokenize(text: str) -> list[str]:
    """Lowercase and apply TOKEN_RE."""
    return TOKEN_RE.findall(text.lower())
