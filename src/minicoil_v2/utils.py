"""Shared utilities for miniCOIL v2."""

import torch

from minicoil_v2.constants import TOKEN_RE, WORD_RE


def select_device(preference: str = "auto") -> torch.device:
    """Select compute device (MPS > CUDA > CPU)."""
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tokenize_unicode(text: str) -> list[str]:
    """Tokenize: lowercase, extract word tokens (Unicode letters only)."""
    return TOKEN_RE.findall(text.lower())


def tokenize_alpha(text: str) -> set[str]:
    """Tokenize: lowercase alphabetic words (ASCII + accented)."""
    return set(WORD_RE.findall(text.lower()))
