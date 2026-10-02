"""Token-pooling around concept words.

Replaces sentence-level pooling with concept-word-focused token pooling.
Diagnostic showed this gives a ~7x larger intra/inter cosine gap (+0.12 vs +0.02).

Shared between embed (training data), encoder (inference), and diagnostics.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def load_token_pool_model(
    model_name: str, device: torch.device
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load tokenizer + transformer in eval mode on the given device."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model


def _find_token_indices(
    offsets: list[tuple[int, int]],
    word_start: int,
    word_end: int,
) -> list[int]:
    """Indices of tokens whose char-span overlaps [word_start, word_end).

    Skips special tokens (offset (0, 0))."""
    out: list[int] = []
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if s < word_end and e > word_start:
            out.append(i)
    return out


def pool_spans(
    hidden: torch.Tensor,
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int]],
    eps: float = 1e-12,
) -> torch.Tensor | None:
    """Pool one concept's vector from a single text's hidden states.

    For each char-span: mean of overlapping (non-special) token rows, then
    L2-normalize. Average those per-occurrence vectors, then L2-normalize again.
    Returns None if no span overlapped any token. This is byte-for-byte the
    pooling the inference encoder performs per concept (see encoder.py).
    """
    per_occ: list[torch.Tensor] = []
    for word_start, word_end in spans:
        tok_idx = [
            i
            for i, (s, e) in enumerate(offsets)
            if not (s == 0 and e == 0) and s < word_end and e > word_start
        ]
        if not tok_idx:
            continue
        idx_t = torch.tensor(tok_idx, dtype=torch.long, device=hidden.device)
        vec = hidden[idx_t].mean(dim=0)
        vec = vec / (vec.norm() + eps)
        per_occ.append(vec)
    if not per_occ:
        return None
    pooled = torch.stack(per_occ).mean(dim=0)
    return pooled / (pooled.norm() + eps)


@torch.no_grad()
def encode_sentences_with_focals(
    sentences: list[str],
    focals_per_sentence: list[list[str]],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: torch.device,
    prefix: str = PASSAGE_PREFIX,
    batch_size: int = 16,
    max_length: int = 256,
) -> list[list[np.ndarray]]:
    """One transformer pass per sentence; emit one pooled vector per focal word.

    Returns: parallel to focals_per_sentence — for each sentence a list of
    L2-normalized 384D vectors, one per focal word. A zero vector signals the
    focal word was not located in the tokenized text.
    """
    assert len(sentences) == len(focals_per_sentence), "lengths must match"
    if not sentences:
        return []

    results: list[list[np.ndarray]] = []
    for start in range(0, len(sentences), batch_size):
        chunk_s = sentences[start : start + batch_size]
        chunk_focals = focals_per_sentence[start : start + batch_size]
        texts = [prefix + s for s in chunk_s]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        offset_mapping = enc.pop("offset_mapping").tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        hidden = out.last_hidden_state.cpu()

        for b, (text, focals) in enumerate(zip(texts, chunk_focals, strict=True)):
            text_lower = text.lower()
            offsets = offset_mapping[b]
            per_focal: list[np.ndarray] = []
            for focal in focals:
                target_lower = focal.lower()
                pos = text_lower.find(target_lower)
                if pos < 0:
                    per_focal.append(np.zeros(hidden.shape[-1], dtype=np.float32))
                    continue
                tok_idx = _find_token_indices(offsets, pos, pos + len(target_lower))
                if not tok_idx:
                    per_focal.append(np.zeros(hidden.shape[-1], dtype=np.float32))
                    continue
                vec = hidden[b, tok_idx].mean(dim=0)
                vec = vec / (vec.norm() + 1e-12)
                per_focal.append(vec.numpy().astype(np.float32))
            results.append(per_focal)

    return results
