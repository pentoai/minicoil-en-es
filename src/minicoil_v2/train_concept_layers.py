"""Per-concept linear layer training for miniCOIL v2.

For each concept in the vocabulary, trains a Linear(384, OUTPUT_DIM, bias=False) + tanh
head using a bilingual 4-term cosine triplet loss on sentences stored in Qdrant. The
stored teacher (mining) vectors only rank positives and negatives; the head's input is
always a fresh token-pooled mE5-small encode, the same representation as inference.

Pipeline:
1. Load mE5-small (input encoder)
2. For batches of concepts:
   a. Scroll sentences + teacher vectors from Qdrant
   b. Re-encode each (sentence, concept) row as token-pooled mE5-small ("passage: ")
   c. Optionally add trimmed copies (trim augmentation), also token-pooled
   d. Build the distance matrix from the teacher vectors
   e. Train the head with the hard-mining bilingual sampler (or the batched trainer)
3. Save trained weights per batch, merge at end

See docs/04-training.md and docs/05-four-quadrant-contrastive.md.

Usage:
    minicoil train
    minicoil train --trim-augment-ratio 0.3 --trim-window 5
    minicoil train --concepts 0:10
    minicoil train --resume
"""

import hashlib
import json
import os
import random
import time
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from minicoil_v2.concept_match import match_concepts_after_prefix
from minicoil_v2.constants import (
    DEFAULT_DROPOUT,
    DEFAULT_SAMPLE_BATCH_SIZE,
    DEFAULT_TRAIN_EPOCH_SIZE,
    INPUT_DIM,
    INPUT_ENCODER,
    MIN_SENTENCES_PER_CONCEPT,
    MIN_TRIPLET_MARGIN,
    OUTPUT_DIM,
)
from minicoil_v2.settings import TrainSettings
from minicoil_v2.token_pooling import PASSAGE_PREFIX, load_token_pool_model, pool_spans
from minicoil_v2.utils import select_device

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_vocab(data_dir: Path) -> dict:
    """Load concept vocabulary."""
    with open(data_dir / "concept_vocabulary.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_inputs_token_pooled(
    model,
    tokenizer,
    device,
    sentences: list[str],
    cids: list[str],
    langs: list[str],
    word_to_concept: dict[str, dict[str, str]],
    prefix: str = PASSAGE_PREFIX,
    batch_size: int = 32,
    max_length: int = 256,
) -> tuple[torch.Tensor, list[int]]:
    """Token-pooled mE5 input per row, matching ``encoder.encode_concept_vectors``.

    For each (sentence, cid, lang) the vector is pooled over ALL surface-form
    occurrences of concept ``cid`` in the sentence (the prefix itself never
    matches), in the prefixed sentence's token offsets (exactly as inference
    does via ``match_concepts`` + ``pool_spans``). The concept id is given
    explicitly (the live ``minicoil_sentences`` collection stores ``concept_ids``
    per row but no focal word). Returns ``(input_embs[K, 384], kept_indices)``.
    Rows whose concept is not located in the tokenized window are DROPPED (never
    fed to the layer as a zero vector).
    """
    vecs: list[torch.Tensor] = []
    kept: list[int] = []
    n = len(sentences)
    for start in range(0, n, batch_size):
        chunk = sentences[start : start + batch_size]
        prefixed = [prefix + s for s in chunk]
        enc = tokenizer(
            prefixed,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        offsets = enc.pop("offset_mapping").tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state.cpu()
        for b in range(len(chunk)):
            gi = start + b
            cid = cids[gi]
            if cid is None:
                continue
            spans = match_concepts_after_prefix(
                prefix, chunk[b], langs[gi], word_to_concept, {cid}
            ).get(cid, [])
            vec = pool_spans(hidden[b], offsets[b], spans)
            if vec is None:
                continue
            vecs.append(vec)
            kept.append(gi)
    if not vecs:
        return torch.empty(0, model.config.hidden_size), []
    return torch.stack(vecs), kept


# ---------------------------------------------------------------------------
# Trimming augmentation
# ---------------------------------------------------------------------------


def get_concept_words(vocab: dict, concept_id: str, lang: str) -> set[str]:
    """Get the set of words for a concept in a given language."""
    concept_data = vocab.get("concepts", {}).get(concept_id, {})
    return set(concept_data.get(lang, []))


def trim_around_word(sentence: str, target_words: set[str], window: int = 5) -> str:
    """Trim sentence to a window of words around the first matching target word."""
    tokens = sentence.split()
    for i, tok in enumerate(tokens):
        cleaned = tok.lower().strip(".,!?;:\"'()-")
        if cleaned in target_words:
            start = max(0, i - window)
            end = min(len(tokens), i + window + 1)
            return " ".join(tokens[start:end])
    return sentence


def collect_trimmed_augmentations(
    sentences: list[str],
    langs: list[str],
    vocab: dict,
    concept_id: str,
    trim_ratio: float,
    trim_window: int,
) -> tuple[list[str], list[int]]:
    """Build trimmed copies and source indices for data augmentation."""
    if trim_ratio <= 0:
        return [], []

    augmented_sentences: list[str] = []
    source_indices: list[int] = []

    for idx, (sent, lang) in enumerate(zip(sentences, langs, strict=True)):
        if random.random() >= trim_ratio:
            continue
        words = get_concept_words(vocab, concept_id, lang)
        trimmed = trim_around_word(sent, words, window=trim_window)
        if trimmed != sent:
            augmented_sentences.append(trimmed)
            source_indices.append(idx)

    return augmented_sentences, source_indices


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def interleave_by_language(
    input_embs: torch.Tensor,
    mining_embs: torch.Tensor,
    langs: list[str],
    sentences: list[str],
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    """Interleave sentences by language so train/val splits get both.

    Alternates EN and ES indices, ensuring both languages appear throughout
    the index range. Leftover sentences from the majority language are appended
    at the end.
    """
    en_idx = [i for i, lang in enumerate(langs) if lang == "en"]
    es_idx = [i for i, lang in enumerate(langs) if lang == "es"]

    interleaved: list[int] = []
    i_en, i_es = 0, 0
    while i_en < len(en_idx) and i_es < len(es_idx):
        interleaved.append(en_idx[i_en])
        i_en += 1
        interleaved.append(es_idx[i_es])
        i_es += 1

    interleaved.extend(en_idx[i_en:])
    interleaved.extend(es_idx[i_es:])

    perm = torch.tensor(interleaved, dtype=torch.long)
    return (
        input_embs[perm],
        mining_embs[perm],
        [langs[i] for i in interleaved],
        [sentences[i] for i in interleaved],
    )


def cosine_distance(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Calculate cosine distance between paired rows."""
    dot_product = torch.sum(x1 * x2, dim=1)
    x1_norm = torch.norm(x1, p=2, dim=1)
    x2_norm = torch.norm(x2, p=2, dim=1)
    return 1.0 - dot_product / (x1_norm * x2_norm + eps)


def bilingual_cosine_loss(
    anchor: torch.Tensor,
    sl_pos: torch.Tensor,
    sl_neg: torch.Tensor,
    xl_pos: torch.Tensor,
    xl_neg: torch.Tensor,
    margin_sl: torch.Tensor,
    margin_xl: torch.Tensor,
    has_xl: torch.Tensor,
) -> torch.Tensor:
    """4-term cosine triplet loss with per-sample adaptive margins.

    SL = same-language, XL = cross-language. has_xl is a boolean mask indicating
    which samples have valid cross-lingual pairs (some concepts may be monolingual).
    """
    sl_loss = torch.relu(
        cosine_distance(anchor, sl_pos) - cosine_distance(anchor, sl_neg) + margin_sl
    )

    if has_xl.any():
        xl_loss_raw = torch.relu(
            cosine_distance(anchor, xl_pos) - cosine_distance(anchor, xl_neg) + margin_xl
        )
        xl_loss = xl_loss_raw * has_xl.float()
    else:
        xl_loss = torch.zeros_like(sl_loss)

    return (sl_loss + xl_loss).mean()


# ---------------------------------------------------------------------------
# Bilingual sampler (v1-style infinite generator)
# ---------------------------------------------------------------------------


class BilingualSampler:
    """Yields batches of bilingual 5-point samples with per-epoch re-mining.

    Each sample is (anchor, sl_pos, sl_neg, xl_pos, xl_neg) with adaptive margins
    for both the same-language and cross-language pairs.

    With `hard_mining=True` (default), positives are drawn from the top-K nearest
    candidates to the anchor and negatives are the hardest semi-hard candidate
    (smallest distance still greater than the positive's distance + min_margin).
    Mirrors miniCOIL v1's mining strategy and gives every batch real margin signal.

    With `hard_mining=False`, two random candidates are drawn and the closer one
    becomes the positive. This is the legacy v2 behavior — keep for ablation.
    """

    def __init__(
        self,
        input_embs: torch.Tensor,
        distance_matrix: np.ndarray,
        langs: np.ndarray,
        range_from: int,
        range_to: int,
        min_margin: float = MIN_TRIPLET_MARGIN,
        batch_size: int = DEFAULT_SAMPLE_BATCH_SIZE,
        epoch_size: int = DEFAULT_TRAIN_EPOCH_SIZE,
        hard_mining: bool = True,
        top_k_pos: int = 20,
    ):
        self.input_embs = input_embs
        self.distance_matrix = distance_matrix
        self.langs = langs
        self.range_from = range_from
        self.range_to = range_to
        self.min_margin = min_margin
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.hard_mining = hard_mining
        self.top_k_pos = top_k_pos

        indices_in_range = np.arange(range_from, range_to)
        unique_langs = np.unique(langs[indices_in_range])

        self.lang_indices: dict[str, np.ndarray] = {}
        for lang in unique_langs:
            self.lang_indices[lang] = indices_in_range[langs[indices_in_range] == lang]

        self.all_langs_list = list(self.lang_indices.keys())
        self.bilingual = len(self.all_langs_list) >= 2
        # Precompute the other-language list per anchor language (constant per sample).
        self._other_langs = {
            lang: [o for o in self.all_langs_list if o != lang] for lang in self.all_langs_list
        }

        # Hard mining needs, per anchor, candidates sorted by distance. The distance
        # matrix is fixed, so sort ONCE here instead of argsort-ing on every sample
        # (was the dominant per-sample cost). For each language L, store the global
        # candidate indices and their distances, sorted ascending per anchor row.
        self._sorted_idx: dict[str, np.ndarray] = {}
        self._sorted_dist: dict[str, np.ndarray] = {}
        if self.hard_mining:
            for lang, pool in self.lang_indices.items():
                sub = self.distance_matrix[:, pool]  # [N, |L|] distances to lang-L candidates
                order = np.argsort(sub, axis=1)
                self._sorted_idx[lang] = pool[order]  # global indices, sorted per row
                self._sorted_dist[lang] = np.take_along_axis(sub, order, axis=1)

        self._epoch_rejections = 0
        self._epoch_samples = 0
        self._epoch_xl_count = 0

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """Yield batches of 5-point samples. Fresh random draws each call."""
        self._epoch_rejections = 0
        self._epoch_samples = 0
        self._epoch_xl_count = 0
        self._epoch_margins_sl: list[float] = []
        self._epoch_margins_xl: list[float] = []

        anchors, sl_pos, sl_neg, xl_pos, xl_neg = [], [], [], [], []
        margin_sl_list, margin_xl_list, has_xl_list = [], [], []
        n = 0
        max_attempts = self.epoch_size * 20
        attempts = 0

        while n < self.epoch_size and attempts < max_attempts:
            attempts += 1
            sample = self._sample_one()
            if sample is None:
                self._epoch_rejections += 1
                continue

            a, sp, sn, xp, xn, m_sl, m_xl, xl_valid = sample
            self._epoch_samples += 1
            self._epoch_margins_sl.append(m_sl)
            if xl_valid:
                self._epoch_xl_count += 1
                self._epoch_margins_xl.append(m_xl)
            anchors.append(a)
            sl_pos.append(sp)
            sl_neg.append(sn)
            xl_pos.append(xp)
            xl_neg.append(xn)
            margin_sl_list.append(m_sl)
            margin_xl_list.append(m_xl)
            has_xl_list.append(xl_valid)
            n += 1

            if len(anchors) >= self.batch_size:
                yield self._make_batch(
                    anchors,
                    sl_pos,
                    sl_neg,
                    xl_pos,
                    xl_neg,
                    margin_sl_list,
                    margin_xl_list,
                    has_xl_list,
                )
                anchors, sl_pos, sl_neg, xl_pos, xl_neg = [], [], [], [], []
                margin_sl_list, margin_xl_list, has_xl_list = [], [], []

        if anchors:
            yield self._make_batch(
                anchors,
                sl_pos,
                sl_neg,
                xl_pos,
                xl_neg,
                margin_sl_list,
                margin_xl_list,
                has_xl_list,
            )

    def _sample_one(self) -> tuple[int, int, int, int, int, float, float, bool] | None:
        """Try to sample one valid 5-point tuple. Returns None on rejection."""
        anchor_lang = random.choice(self.all_langs_list)
        anchor_lang_indices = self.lang_indices[anchor_lang]
        if len(anchor_lang_indices) < 3:
            return None

        a = int(anchor_lang_indices[random.randrange(len(anchor_lang_indices))])

        sl_result = self._pick_pair(a, anchor_lang, exclude_a=True)
        if sl_result is None:
            return None
        sp, sn, m_sl = sl_result

        xl_valid = False
        xp, xn, m_xl = a, a, 0.0

        if self.bilingual:
            other_lang = random.choice(self._other_langs[anchor_lang])
            if len(self.lang_indices[other_lang]) >= 2:
                xl_result = self._pick_pair(a, other_lang, exclude_a=False)
                if xl_result is not None:
                    xp, xn, m_xl = xl_result
                    xl_valid = True

        return a, sp, sn, xp, xn, m_sl, m_xl, xl_valid

    def _pick_pair(self, a: int, lang: str, exclude_a: bool) -> tuple[int, int, float] | None:
        """Pick a positive/negative pair among ``lang`` candidates for anchor ``a``."""
        if self.hard_mining:
            return self._pick_pair_mined(a, lang, exclude_a)
        return self._pick_pair_random(a, lang, exclude_a)

    def _pick_pair_random(
        self, a: int, lang: str, exclude_a: bool
    ) -> tuple[int, int, float] | None:
        """Legacy: pick 2 random candidates, closer one becomes positive."""
        cand = self.lang_indices[lang]
        if exclude_a:
            cand = cand[cand != a]
        if len(cand) < 2:
            return None
        x, y = random.sample(cand.tolist(), 2)
        a_dists = self.distance_matrix[a]
        dx, dy = a_dists[x], a_dists[y]
        margin = abs(dx - dy)
        if margin < self.min_margin:
            return None
        if dx < dy:
            return x, y, margin
        return y, x, margin

    def _pick_pair_mined(self, a: int, lang: str, exclude_a: bool) -> tuple[int, int, float] | None:
        """v1-style hard mining over ``lang`` candidates for anchor ``a``.

        - Positive: random pick from the top-K candidates closest to the anchor.
        - Negative: closest candidate (outside the positive pool) whose distance
          exceeds the positive's by at least ``min_margin``.

        Reads the per-anchor sorted candidate order precomputed in ``__init__`` (no
        per-sample argsort). Behavior-identical to the prior argsort+linear-scan;
        ties may resolve to a different equal-distance index (same margin).
        """
        sc = self._sorted_idx[lang][a]
        sd = self._sorted_dist[lang][a]
        if exclude_a:
            keep = sc != a
            sc = sc[keep]
            sd = sd[keep]
        m = len(sc)
        if m < 3:
            return None
        k = min(self.top_k_pos, max(1, m // 3))
        # Positive: random among the k nearest (sorted positions 0..k-1).
        pos_rank = int(np.random.randint(k))
        d_pos = float(sd[pos_rank])
        # Negative: first sorted position >= k with distance >= d_pos + min_margin.
        j = int(np.searchsorted(sd, d_pos + self.min_margin, side="left"))
        if j < k:
            j = k
        if j >= m:
            return None
        return int(sc[pos_rank]), int(sc[j]), float(sd[j] - d_pos)

    @staticmethod
    def _make_batch(
        anchors: list[int],
        sl_pos: list[int],
        sl_neg: list[int],
        xl_pos: list[int],
        xl_neg: list[int],
        margin_sl: list[float],
        margin_xl: list[float],
        has_xl: list[bool],
    ) -> dict[str, np.ndarray]:
        return {
            "anchor": np.array(anchors, dtype=np.int64),
            "sl_pos": np.array(sl_pos, dtype=np.int64),
            "sl_neg": np.array(sl_neg, dtype=np.int64),
            "xl_pos": np.array(xl_pos, dtype=np.int64),
            "xl_neg": np.array(xl_neg, dtype=np.int64),
            "margin_sl": np.array(margin_sl, dtype=np.float32),
            "margin_xl": np.array(margin_xl, dtype=np.float32),
            "has_xl": np.array(has_xl, dtype=np.bool_),
        }

    def log_epoch_stats(self, label: str = "sampler") -> None:
        """Log diagnostic stats from the last epoch. Call after iterating."""
        total_attempts = self._epoch_samples + self._epoch_rejections
        reject_rate = self._epoch_rejections / max(total_attempts, 1)
        xl_rate = self._epoch_xl_count / max(self._epoch_samples, 1)

        avg_m_sl = np.mean(self._epoch_margins_sl) if self._epoch_margins_sl else 0.0
        avg_m_xl = np.mean(self._epoch_margins_xl) if self._epoch_margins_xl else 0.0
        std_m_sl = np.std(self._epoch_margins_sl) if self._epoch_margins_sl else 0.0
        std_m_xl = np.std(self._epoch_margins_xl) if self._epoch_margins_xl else 0.0

        logger.info(
            f"[{label}] samples={self._epoch_samples} "
            f"reject_rate={reject_rate:.1%} "
            f"xl_rate={xl_rate:.1%} "
            f"margin_sl={avg_m_sl:.3f}±{std_m_sl:.3f} "
            f"margin_xl={avg_m_xl:.3f}±{std_m_xl:.3f}"
        )


# ---------------------------------------------------------------------------
# Per-concept training
# ---------------------------------------------------------------------------


def train_concept_layer(
    train_sampler: BilingualSampler,
    val_sampler: BilingualSampler,
    device: torch.device,
    epochs: int = 500,
    lr: float = 2e-3,
    dropout: float = DEFAULT_DROPOUT,
    lr_factor: float = 0.5,
    lr_patience: int = 5,
) -> tuple[nn.Linear, float, float]:
    """Train a single per-concept linear layer with bilingual cosine loss.

    Returns: (trained layer on CPU, final train loss, final val loss)
    """
    embs = train_sampler.input_embs.to(device)

    layer = nn.Linear(INPUT_DIM, OUTPUT_DIM, bias=False).to(device)
    dropout_layer = nn.Dropout(dropout).to(device)
    optimizer = torch.optim.Adam(layer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        threshold=1e-4,
    )

    final_train_loss = 0.0
    final_val_loss = 0.0

    def forward_batch(batch: dict[str, np.ndarray], training: bool) -> torch.Tensor:
        if training:
            layer.train()
            dropout_layer.train()
        else:
            layer.eval()
            dropout_layer.eval()

        idx_a = torch.from_numpy(batch["anchor"]).to(device)
        idx_sp = torch.from_numpy(batch["sl_pos"]).to(device)
        idx_sn = torch.from_numpy(batch["sl_neg"]).to(device)
        idx_xp = torch.from_numpy(batch["xl_pos"]).to(device)
        idx_xn = torch.from_numpy(batch["xl_neg"]).to(device)
        m_sl = torch.from_numpy(batch["margin_sl"]).to(device)
        m_xl = torch.from_numpy(batch["margin_xl"]).to(device)
        h_xl = torch.from_numpy(batch["has_xl"]).to(device)

        def project(idx: torch.Tensor) -> torch.Tensor:
            return torch.tanh(layer(dropout_layer(embs[idx])))

        return bilingual_cosine_loss(
            project(idx_a),
            project(idx_sp),
            project(idx_sn),
            project(idx_xp),
            project(idx_xn),
            m_sl,
            m_xl,
            h_xl,
        )

    for _epoch in range(epochs):
        epoch_train_loss = 0.0
        n_train_batches = 0
        for batch in train_sampler:
            loss = forward_batch(batch, training=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            n_train_batches += 1

        epoch_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_sampler:
                loss = forward_batch(batch, training=False)
                epoch_val_loss += loss.item()
                n_val_batches += 1

        final_train_loss = epoch_train_loss / max(n_train_batches, 1)
        final_val_loss = epoch_val_loss / max(n_val_batches, 1)
        scheduler.step(final_val_loss)

        log_every = max(1, epochs // 5)
        if _epoch == 0 or (_epoch + 1) % log_every == 0 or _epoch == epochs - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.debug(
                f"  epoch {_epoch + 1}/{epochs} | "
                f"train={final_train_loss:.4f} val={final_val_loss:.4f} lr={cur_lr:.2e}"
            )
            train_sampler.log_epoch_stats("train")
            val_sampler.log_epoch_stats("val")

    layer.eval()
    return layer.cpu(), final_train_loss, final_val_loss


def dynamic_min_margin(dist_matrix: np.ndarray, scale: float, floor: float) -> float:
    """Per-concept mining margin floor, scaled to the concept's own distance spread.

    A fixed floor ignores that each concept's mining-distance distribution has a
    different scale: tight (monosemous) concepts have small gaps everywhere, so a
    global 0.1 floor never bites (reject_rate ~0, margins pin to the floor); spread
    (polysemous) concepts have large inter-sense gaps where 0.1 is trivial. Scaling
    by the concept's own std restores uniform selectivity and forces harder negatives
    where senses actually separate. ``floor`` is the absolute minimum so the margin
    never collapses toward 0. ``scale <= 0`` keeps the legacy fixed floor.
    """
    if scale <= 0 or dist_matrix.shape[0] < 3:
        return floor
    iu = np.triu_indices(dist_matrix.shape[0], k=1)
    spread = float(np.std(dist_matrix[iu]))
    return max(floor, scale * spread)


def train_one_concept(
    c_input: torch.Tensor,
    c_mining: torch.Tensor,
    c_langs: list[str],
    settings: TrainSettings,
    device,
) -> tuple[nn.Linear | None, float, float]:
    """Train one concept's Linear(384, OUTPUT_DIM) head, Qdrant-free.

    ``c_input`` is the token-pooled mE5 input (384D, the inference representation);
    ``c_mining`` is the mining vector (any dim, e.g. 4096D Qwen3) used ONLY to build
    the positive/negative distance matrix. This decoupling is the Phase-0 fix: the
    layer never sees the mining vector. Returns ``(None, 0.0, 0.0)`` if there are
    fewer than ``MIN_SENTENCES_PER_CONCEPT`` rows.
    """
    c_input, c_mining, c_langs, _ = interleave_by_language(
        c_input, c_mining, c_langs, [""] * len(c_langs)
    )

    n = c_input.shape[0]
    if n < MIN_SENTENCES_PER_CONCEPT:
        return None, 0.0, 0.0

    c_mining_np = c_mining.numpy()
    norms = np.linalg.norm(c_mining_np, axis=1, keepdims=True) + 1e-9
    c_mining_normed = c_mining_np / norms
    sim_matrix = c_mining_normed @ c_mining_normed.T
    np.fill_diagonal(sim_matrix, 1.0)
    dist_matrix = 1.0 - sim_matrix

    lang_arr = np.array(c_langs)

    # Per-concept margin floor, scaled to this concept's own distance spread. A
    # global floor never bites on rich/tight concepts (reject_rate ~0, margins
    # pinned at the floor); scaling by std restores uniform mining selectivity.
    min_margin = dynamic_min_margin(dist_matrix, settings.margin_scale, settings.min_triplet_margin)

    n_val = int(n * settings.val_size)
    n_val = min(max(n_val, 1), n - 3) if n > 3 else 0
    train_to = n - n_val

    train_sampler = BilingualSampler(
        input_embs=c_input,
        distance_matrix=dist_matrix,
        langs=lang_arr,
        range_from=0,
        range_to=train_to,
        min_margin=min_margin,
        batch_size=settings.sample_batch_size,
        epoch_size=settings.train_epoch_size,
    )
    val_sampler = BilingualSampler(
        input_embs=c_input,
        distance_matrix=dist_matrix,
        langs=lang_arr,
        range_from=train_to,
        range_to=n,
        min_margin=min_margin,
        batch_size=settings.sample_batch_size,
        epoch_size=settings.val_epoch_size,
    )

    return train_concept_layer(
        train_sampler,
        val_sampler,
        device,
        epochs=settings.epochs,
        lr=settings.lr,
        dropout=settings.dropout,
        lr_factor=settings.lr_factor,
        lr_patience=settings.lr_patience,
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    output_dir: Path,
    trained_layers: dict[str, dict],
    batch_idx: int,
) -> Path:
    """Save a batch of trained layers as a checkpoint."""
    path = output_dir / f"checkpoint_batch_{batch_idx:04d}.pt"
    torch.save(trained_layers, path)
    return path


def merge_checkpoints(output_dir: Path) -> Path:
    """Merge all batch checkpoints into a single model file."""
    merged = {}
    for ckpt_path in sorted(output_dir.glob("checkpoint_batch_*.pt")):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        merged.update(checkpoint)

    merged_path = output_dir / "concept_layers.pt"
    torch.save(merged, merged_path)
    return merged_path


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup_concept_ids(vocab: dict, settings: TrainSettings) -> list[str]:
    """Prepare concept list for training, applying range filter and resume.

    `settings.concepts` accepts either a range "start:end" or a comma-separated
    list "C-001,C-042" to target specific concepts (useful for hand-picked smoke trains).
    """
    concept_ids = sorted(vocab["concepts"].keys())
    logger.info(f"Total concepts in vocabulary: {len(concept_ids)}")

    if settings.concepts:
        spec = settings.concepts.strip()
        if "," in spec or spec.startswith("C-"):
            wanted = {c.strip() for c in spec.split(",") if c.strip()}
            concept_ids = [cid for cid in concept_ids if cid in wanted]
            logger.info(
                f"Filtered to {len(concept_ids)} concepts (explicit list, {len(wanted)} requested)"
            )
        else:
            start, end = map(int, spec.split(":"))
            concept_ids = [cid for cid in concept_ids if start <= int(cid.split("-")[1]) < end]
            logger.info(f"Filtered to {len(concept_ids)} concepts (range {spec})")

    if settings.resume:
        trained_set: set[str] = set()
        for ckpt_path in settings.output_dir.glob("checkpoint_batch_*.pt"):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            trained_set.update(ckpt.keys())
        if trained_set:
            concept_ids = [c for c in concept_ids if c not in trained_set]
            logger.info(
                f"Resuming: {len(trained_set)} already trained, {len(concept_ids)} remaining"
            )

    return concept_ids


# ---------------------------------------------------------------------------
# Batch input preparation (scroll + token-pool encode) with optional disk cache
# ---------------------------------------------------------------------------


def _prepare_batch_inputs(
    client,
    settings: TrainSettings,
    batch_concepts: list[str],
    batch_idx: int,
    rematch: dict | None,
    input_model,
    input_tokenizer,
    device,
    word_to_concept: dict,
) -> tuple[dict | None, int]:
    """Scroll each concept's rows, token-pool the mE5 inputs, compact dropped rows.

    Returns ``(batch_data, n_skipped_delta)``. ``batch_data`` is ``None`` when the
    batch yields no trainable rows. The dict is recipe-independent (depends only on
    the collection, concept set, and scroll/encode params), so it is safe to cache.
    """
    from minicoil_v2.qdrant_store import scroll_concept, scroll_concept_rematch

    n_skipped = 0
    # Dedup unit is (sentence, concept_id, lang): the same sentence pooled for
    # different concepts yields different input vectors, so concept is part of the key.
    row_to_idx: dict[tuple[str, str, str], int] = {}
    unique_sentences: list[str] = []
    unique_cids: list[str] = []
    unique_langs: list[str] = []
    unique_mining_vecs: list[torch.Tensor] = []
    concept_indices: dict[str, list[int]] = {}
    concept_sent_langs: dict[str, list[str]] = {}

    for cid in batch_concepts:
        if rematch is not None:
            sentences, _focals, langs, mining_embs = scroll_concept_rematch(
                client,
                settings.qdrant.collection_name,
                cid,
                rematch.get(cid, {}),
                max_per_lang=settings.max_sentences,
                lang_ratio=settings.lang_ratio,
            )
        else:
            sentences, _focals, langs, mining_embs = scroll_concept(
                client,
                settings.qdrant.collection_name,
                cid,
                max_per_lang=settings.max_sentences,
                lang_ratio=settings.lang_ratio,
            )
        if len(sentences) < MIN_SENTENCES_PER_CONCEPT:
            n_skipped += 1
            continue

        indices = []
        s_langs = []
        for sent, lang, mvec in zip(sentences, langs, mining_embs, strict=True):
            key = (sent, cid, lang)
            if key not in row_to_idx:
                row_to_idx[key] = len(unique_sentences)
                unique_sentences.append(sent)
                unique_cids.append(cid)
                unique_langs.append(lang)
                unique_mining_vecs.append(mvec)
            indices.append(row_to_idx[key])
            s_langs.append(lang)
        concept_indices[cid] = indices
        concept_sent_langs[cid] = s_langs

    if not concept_indices:
        return None, n_skipped

    n_unique = len(unique_sentences)
    n_total = sum(len(v) for v in concept_indices.values())
    dedup_ratio = 1 - n_unique / n_total if n_total > 0 else 0
    logger.info(
        f"[Batch {batch_idx}] {n_unique:,} unique (sentence, concept) rows "
        f"({n_total:,} total, {dedup_ratio:.0%} dedup) for {len(concept_indices)} concepts"
    )

    mining_embs_all = torch.stack(unique_mining_vecs)  # mining-only (e.g. Qwen3 1024D)

    # Decouple input from mining: re-encode every row as token-pooled mE5 (384D),
    # reproducing encoder.encode_concept_vectors byte-for-byte. Rows whose concept
    # cannot be located are dropped and the row-index space compacted.
    input_embs, kept = encode_inputs_token_pooled(
        input_model,
        input_tokenizer,
        device,
        unique_sentences,
        unique_cids,
        unique_langs,
        word_to_concept,
        prefix=PASSAGE_PREFIX,
        batch_size=settings.encode_batch_size,
    )
    if not kept:
        logger.warning(f"[Batch {batch_idx}] no rows located by encoder; skipping batch")
        return None, n_skipped
    if len(kept) < n_unique:
        logger.info(f"[Batch {batch_idx}] dropped {n_unique - len(kept)} unlocated rows")
    old_to_new = {old: new for new, old in enumerate(kept)}
    keep_t = torch.tensor(kept, dtype=torch.long)
    mining_embs_all = mining_embs_all[keep_t]
    unique_sentences = [unique_sentences[i] for i in kept]
    unique_langs = [unique_langs[i] for i in kept]
    concept_indices = {
        cid: [old_to_new[i] for i in idxs if i in old_to_new]
        for cid, idxs in concept_indices.items()
    }
    concept_sent_langs = {
        cid: [unique_langs[i] for i in concept_indices[cid]] for cid in concept_indices
    }
    return (
        {
            "input_embs": input_embs,
            "mining_embs_all": mining_embs_all,
            "unique_sentences": unique_sentences,
            "unique_langs": unique_langs,
            "concept_indices": concept_indices,
            "concept_sent_langs": concept_sent_langs,
        },
        n_skipped,
    )


def _load_or_prepare_batch(
    client,
    settings: TrainSettings,
    batch_concepts: list[str],
    batch_idx: int,
    rematch: dict | None,
    input_model,
    input_tokenizer,
    device,
    word_to_concept: dict,
) -> tuple[dict | None, int]:
    """``_prepare_batch_inputs`` fronted by an optional disk cache keyed by the
    recipe-independent inputs (collection, concept set, scroll/encode params)."""
    cache_path = None
    if settings.encode_cache_dir is not None:
        key_src = "|".join(
            [
                settings.qdrant.collection_name,
                str(settings.max_sentences),
                str(settings.lang_ratio),
                INPUT_ENCODER,
                "rematch" if rematch is not None else "scroll",
                ",".join(batch_concepts),
            ]
        )
        key = hashlib.sha1(key_src.encode()).hexdigest()
        cache_path = settings.encode_cache_dir / f"batch_{key}.pt"
        if cache_path.exists():
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            logger.info(
                f"[Batch {batch_idx}] encode cache HIT "
                f"({data['input_embs'].shape[0]} rows, {cache_path.name})"
            )
            return data, 0

    data, n_skipped = _prepare_batch_inputs(
        client,
        settings,
        batch_concepts,
        batch_idx,
        rematch,
        input_model,
        input_tokenizer,
        device,
        word_to_concept,
    )
    if cache_path is not None and data is not None:
        settings.encode_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, cache_path)
        logger.info(f"[Batch {batch_idx}] encode cache WRITE ({cache_path.name})")
    return data, n_skipped


def _load_or_compute_aug(
    settings: TrainSettings,
    batch_concepts: list[str],
    batch_idx: int,
    concept_indices: dict[str, list[int]],
    unique_sentences: list[str],
    concept_sent_langs: dict[str, list[str]],
    vocab: dict,
    word_to_concept: dict,
    input_model,
    input_tokenizer,
    device,
    rematch: dict | None = None,
) -> dict[str, dict]:
    """Per-concept trim-augmentation input embeddings, fronted by a disk cache.

    The mE5 re-encode of trimmed copies is the single largest cost in the train
    section and is recomputed identically every run. Caching it (keyed by the same
    recipe hash as the encode cache plus the trim params) lets re-runs on a fixed
    cache skip it. Deterministic at ``trim_augment_ratio=1.0`` (every sentence is
    trimmed), so the cached embeddings reproduce the inline path bit-for-bit; the
    key embeds the ratio/window so a different recipe recomputes.

    Returns ``{cid: {"aug_input": Tensor[K, INPUT_DIM], "kept_src": list[int]}}``.
    """
    if settings.trim_augment_ratio <= 0:
        return {}

    cache_path = None
    if settings.encode_cache_dir is not None:
        # Strict superset of the encode-cache key (same fields, same order,
        # including the scroll/rematch discriminator) plus the trim params, so an
        # aug cache can never collide across sourcing modes or recipes.
        key_src = "|".join(
            [
                settings.qdrant.collection_name,
                str(settings.max_sentences),
                str(settings.lang_ratio),
                INPUT_ENCODER,
                "rematch" if rematch is not None else "scroll",
                f"aug{settings.trim_augment_ratio}:{settings.trim_window}",
                ",".join(batch_concepts),
            ]
        )
        key = hashlib.sha1(key_src.encode()).hexdigest()
        cache_path = settings.encode_cache_dir / f"aug_{key}.pt"
        if cache_path.exists():
            aug = torch.load(cache_path, map_location="cpu", weights_only=False)
            logger.info(f"[Batch {batch_idx}] aug cache HIT ({cache_path.name})")
            return aug

    out: dict[str, dict] = {}
    for cid, idx_list in concept_indices.items():
        c_langs = list(concept_sent_langs[cid])
        c_sentences = [unique_sentences[i] for i in idx_list]
        aug_sentences, source_indices = collect_trimmed_augmentations(
            c_sentences, c_langs, vocab, cid, settings.trim_augment_ratio, settings.trim_window
        )
        if not aug_sentences:
            out[cid] = {"aug_input": torch.empty(0, INPUT_DIM), "kept_src": []}
            continue
        aug_cids = [cid] * len(aug_sentences)
        aug_langs = [c_langs[i] for i in source_indices]
        aug_input, aug_kept = encode_inputs_token_pooled(
            input_model,
            input_tokenizer,
            device,
            aug_sentences,
            aug_cids,
            aug_langs,
            word_to_concept,
            prefix=PASSAGE_PREFIX,
            batch_size=settings.encode_batch_size,
        )
        out[cid] = {"aug_input": aug_input, "kept_src": [source_indices[k] for k in aug_kept]}

    if cache_path is not None:
        settings.encode_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(out, cache_path)
        logger.info(f"[Batch {batch_idx}] aug cache WRITE ({cache_path.name})")
    return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_training(settings: TrainSettings) -> None:
    """Run the full training pipeline using Qdrant as sentence source."""
    from minicoil_v2.qdrant_store import get_client

    device = select_device(settings.device)
    logger.info(f"Device: {device}")

    settings.output_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(settings.data_dir)
    with open(settings.data_dir / "word_to_concept.json") as f:
        word_to_concept: dict[str, dict[str, str]] = json.load(f)
    concept_ids = setup_concept_ids(vocab, settings)

    # The live cloud's `concept_ids` payload is a different vocab vintage than the
    # on-disk pruned vocab the encoder/eval use, so filtering by it yields rows for
    # the wrong concept. When rematch buckets are supplied, source each on-disk
    # concept's rows from those pre-matched point-ids instead.
    rematch: dict | None = None
    if settings.rematch_buckets is not None:
        with open(settings.rematch_buckets) as f:
            rematch = json.load(f)
        logger.info(f"Rematch mode: sourcing rows from {settings.rematch_buckets} (on-disk vocab)")

    if not concept_ids:
        logger.info("All concepts already trained!")
        return

    total_to_train = len(concept_ids)

    logger.info(f"Loading input encoder (token-pooled): {INPUT_ENCODER}...")
    input_tokenizer, input_model = load_token_pool_model(INPUT_ENCODER, device)

    logger.info(f"Connecting to Qdrant at {settings.qdrant.url}...")
    client = get_client(settings.qdrant)

    wandb_run = _init_wandb(settings, total_to_train)

    n_trained = 0
    n_skipped = 0
    n_monolingual = 0
    all_train_losses: list[float] = []
    all_val_losses: list[float] = []
    t0 = time.time()

    for batch_start in range(0, total_to_train, settings.concept_batch_size):
        batch_end = min(batch_start + settings.concept_batch_size, total_to_train)
        batch_concepts = concept_ids[batch_start:batch_end]
        batch_idx = batch_start // settings.concept_batch_size
        batch_t0 = time.time()

        batch_data, skip_delta = _load_or_prepare_batch(
            client,
            settings,
            batch_concepts,
            batch_idx,
            rematch,
            input_model,
            input_tokenizer,
            device,
            word_to_concept,
        )
        n_skipped += skip_delta
        if batch_data is None:
            continue
        input_embs = batch_data["input_embs"]
        mining_embs_all = batch_data["mining_embs_all"]
        unique_sentences = batch_data["unique_sentences"]
        concept_indices = batch_data["concept_indices"]
        concept_sent_langs = batch_data["concept_sent_langs"]

        encode_time = time.time() - batch_t0

        # Train per-concept layers
        train_t0 = time.time()
        batch_trained: dict[str, dict] = {}
        batch_train_losses: list[float] = []
        batch_val_losses: list[float] = []
        batch_augmented = 0
        batch_specs: list = []  # collected for the opt-in batched path

        # Trim-augmentation re-encode (the dominant common cost) is computed once
        # per batch and disk-cached; on a cache hit the model never runs here. The
        # token-pooled trimmed copies match inference exactly (see encode helper).
        aug_by_cid = _load_or_compute_aug(
            settings,
            batch_concepts,
            batch_idx,
            concept_indices,
            unique_sentences,
            concept_sent_langs,
            vocab,
            word_to_concept,
            input_model,
            input_tokenizer,
            device,
            rematch=rematch,
        )

        for cid, idx_list in concept_indices.items():
            idx_tensor = torch.tensor(idx_list, dtype=torch.long)
            c_input = input_embs[idx_tensor]
            c_mining = mining_embs_all[idx_tensor]
            c_langs = concept_sent_langs[cid]

            aug = aug_by_cid.get(cid)
            if aug is not None and aug["kept_src"]:
                kept_src = aug["kept_src"]
                source_tensor = torch.tensor(kept_src, dtype=torch.long)
                c_input = torch.cat([c_input, aug["aug_input"]], dim=0)
                c_mining = torch.cat([c_mining, c_mining[source_tensor]], dim=0)
                c_langs = c_langs + [c_langs[i] for i in kept_src]
                batch_augmented += len(kept_src)

            is_bilingual = len(set(c_langs)) >= 2

            if settings.batched:
                # Defer training: collect this concept's spec and train the whole
                # bucket with one batched forward/step after the loop.
                from minicoil_v2.batched_training import prepare_concept

                spec = prepare_concept(cid, c_input, c_mining, c_langs, settings)
                if spec is None:
                    n_skipped += 1
                    continue
                batch_specs.append(spec)
                continue

            layer, final_train_loss, final_val_loss = train_one_concept(
                c_input, c_mining, c_langs, settings, device
            )
            if layer is None:
                n_skipped += 1
                continue
            if not is_bilingual:
                n_monolingual += 1

            logger.info(
                f"  [{cid}] train_loss={final_train_loss:.4f} val_loss={final_val_loss:.4f} "
                f"rows={len(idx_list)} {'bilingual' if is_bilingual else 'MONOLINGUAL'}"
            )

            batch_trained[cid] = {
                "weight": layer.weight.data,
            }
            batch_train_losses.append(final_train_loss)
            batch_val_losses.append(final_val_loss)
            n_trained += 1

        if settings.batched and batch_specs:
            from minicoil_v2.batched_training import train_concept_bucket

            results = train_concept_bucket(
                batch_specs, settings, device, settings.optimizer_mode, seed=batch_idx
            )
            for spec in batch_specs:
                r = results[spec.cid]
                if not spec.is_bilingual:
                    n_monolingual += 1
                logger.info(
                    f"  [{spec.cid}] train_loss={r['train_loss']:.4f} "
                    f"val_loss={r['val_loss']:.4f} rows={spec.n} "
                    f"{'bilingual' if spec.is_bilingual else 'MONOLINGUAL'}"
                )
                batch_trained[spec.cid] = {"weight": r["weight"]}
                batch_train_losses.append(r["train_loss"])
                batch_val_losses.append(r["val_loss"])
                n_trained += 1

        train_time = time.time() - train_t0
        batch_time = time.time() - batch_t0

        if batch_trained:
            save_checkpoint(settings.output_dir, batch_trained, batch_idx)

            avg_train_loss = sum(batch_train_losses) / len(batch_train_losses)
            avg_val_loss = sum(batch_val_losses) / len(batch_val_losses)
            all_train_losses.extend(batch_train_losses)
            all_val_losses.extend(batch_val_losses)
            elapsed = time.time() - t0
            rate = n_trained / elapsed if elapsed > 0 else 0
            remaining = total_to_train - (batch_end + n_skipped)
            eta_h = remaining / rate / 3600 if rate > 0 else 0

            logger.info(
                f"[Batch {batch_idx}] {len(batch_trained)} trained | "
                f"avg train: {avg_train_loss:.4f} | avg val: {avg_val_loss:.4f} | "
                f"augmented: {batch_augmented} | "
                f"encode: {encode_time:.1f}s | train: {train_time:.1f}s | "
                f"batch: {batch_time:.1f}s | "
                f"progress: {n_trained}/{total_to_train} "
                f"({n_trained / total_to_train * 100:.1f}%) | "
                f"ETA: {eta_h:.1f}h"
            )

            if wandb_run:
                import wandb

                wandb.log(
                    {
                        "batch": batch_idx,
                        "concepts_trained": n_trained,
                        "avg_train_loss": avg_train_loss,
                        "avg_val_loss": avg_val_loss,
                        "encode_time_s": encode_time,
                        "train_time_s": train_time,
                        "batch_time_s": batch_time,
                        "sentences_in_batch": input_embs.shape[0],
                        "augmented_sentences": batch_augmented,
                    }
                )

    # Final merge
    logger.info(f"Training complete: {n_trained} trained, {n_skipped} skipped")
    if n_monolingual > 0:
        logger.warning(
            f"{n_monolingual} concepts had only one language (monolingual-only training)"
        )

    if n_trained > 0:
        merged_path = merge_checkpoints(settings.output_dir)
        merged = torch.load(merged_path, map_location="cpu", weights_only=True)
        logger.info(f"Merged model: {len(merged)} concepts -> {merged_path}")

        overall_avg_train_loss = (
            sum(all_train_losses) / len(all_train_losses) if all_train_losses else 0
        )
        overall_avg_val_loss = sum(all_val_losses) / len(all_val_losses) if all_val_losses else 0
        total_time = time.time() - t0
        logger.info(f"Average final train loss: {overall_avg_train_loss:.4f}")
        logger.info(f"Average final val loss: {overall_avg_val_loss:.4f}")
        logger.info(f"Total time: {total_time:.1f}s ({total_time / 3600:.2f}h)")

        if wandb_run:
            import wandb

            wandb.log(
                {
                    "final/concepts_trained": n_trained,
                    "final/concepts_skipped": n_skipped,
                    "final/monolingual_concepts": n_monolingual,
                    "final/avg_train_loss": overall_avg_train_loss,
                    "final/avg_val_loss": overall_avg_val_loss,
                    "final/total_time_hours": total_time / 3600,
                }
            )

    if wandb_run:
        import wandb

        wandb.finish()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_wandb(settings: TrainSettings, total_to_train: int) -> object | None:
    """Initialize W&B if enabled."""
    if not settings.wandb_enabled:
        return None
    try:
        import wandb

        return wandb.init(
            project=settings.wandb_project,
            name=settings.wandb_name or f"concept-training-{total_to_train}",
            config=settings.model_dump(mode="json"),
        )
    except Exception as e:
        logger.warning(f"wandb init failed ({e}), continuing without")
        return None


if __name__ == "__main__":
    run_training(TrainSettings())
