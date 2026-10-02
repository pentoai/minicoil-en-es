"""Batched, GPU-friendly training for miniCOIL v2 per-concept layers.

This is a PERFORMANCE refactor of ``train_concept_layers.py``. It preserves the
exact training semantics (5-point bilingual sampling, semi-hard mining, the
4-term cosine-margin loss, Adam, per-concept ReduceLROnPlateau) while removing
the two bottlenecks of the reference path:

1. The sampler built an epoch one tuple at a time in a Python while-loop. Here we
   draw a whole epoch at once with vectorized gather/searchsorted over the
   pre-sorted candidate rows (``VectorizedSampler``).
2. Concepts were trained one at a time, each a microscopic GPU op. Here a bucket
   of concepts is stacked (``[B, OUTPUT_DIM, INPUT_DIM]``) and projected with a
   single ``bmm``; the per-concept losses are summed into one backward and one
   optimizer step (``train_concept_bucket``).

Two optimizer back-ends are provided so the numerics can be validated in
isolation (see ``scripts/perf/``):

- ``optimizer_mode="per_concept_torch"`` (Milestone A): one ``torch.optim.Adam`` +
  one ``torch.optim.lr_scheduler.ReduceLROnPlateau`` per concept. Exact by
  construction; the only new thing is the batched forward/loss/backward. Used as
  the oracle for the vectorized back-end.
- ``optimizer_mode="vectorized"`` (Milestone B): a single stacked Adam with a
  per-concept learning rate ``[B, 1, 1]`` and a vectorized per-concept plateau
  scheduler. This is the launch-count win that makes the GPU the bottleneck.

The reference path in ``train_concept_layers.py`` stays untouched and is the
equivalence oracle for the validation scripts.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from minicoil_v2.constants import (
    INPUT_DIM,
    MIN_SENTENCES_PER_CONCEPT,
    MIN_TRIPLET_MARGIN,
    OUTPUT_DIM,
)
from minicoil_v2.settings import TrainSettings
from minicoil_v2.train_concept_layers import (
    dynamic_min_margin,
    interleave_by_language,
)

DEFAULT_TOP_K_POS = 20

# ---------------------------------------------------------------------------
# Optional phase profiler (opt-in via MINICOIL_BUCKET_PROFILE=1)
# ---------------------------------------------------------------------------
# Accumulates wall-clock per named phase across a ``train_concept_bucket`` call so
# the synthetic-vs-real speedup gap can be attributed (one-time setup vs per-epoch
# sampling vs host marshaling vs GPU forward/backward) without a separate harness.
# Synchronizes MPS/CUDA around device phases so async kernels are charged to the
# right bucket. Zero overhead when the env var is unset.

PROFILE: dict[str, float] = {}
_PROFILE_ON = os.environ.get("MINICOIL_BUCKET_PROFILE") == "1"


@contextlib.contextmanager
def _ptime(key: str, device: torch.device | None = None):
    if not _PROFILE_ON:
        yield
        return
    if device is not None and device.type in ("mps", "cuda"):
        torch.__dict__[device.type].synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if device is not None and device.type in ("mps", "cuda"):
            torch.__dict__[device.type].synchronize()
        PROFILE[key] = PROFILE.get(key, 0.0) + (time.perf_counter() - t0)


def reset_profile() -> None:
    PROFILE.clear()


# ---------------------------------------------------------------------------
# Vectorized semi-hard pair selection (the deterministic core)
# ---------------------------------------------------------------------------


def pick_pairs_vectorized(
    sorted_idx_full: np.ndarray,
    sorted_dist_rows: np.ndarray,
    anchors: np.ndarray,
    self_pos: np.ndarray | None,
    exclude_a: bool,
    min_margin: float,
    top_k_pos: int,
    pos_ranks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized equivalent of ``BilingualSampler._pick_pair_mined``.

    ``sorted_idx_full`` is the lang's full ``[N, m_pool]`` candidate-index
    structure (sorted ascending per query row); ``sorted_dist_rows[i]`` is the
    matching sorted-distance row for ``anchors[i]`` (gathered by the caller, which
    needs it for the threshold scan anyway). ``self_pos[i]`` is the column of the
    anchor's own entry in its sorted row, used only when ``exclude_a`` (pass
    ``None`` otherwise). Selecting positive/negative via per-row column indexing
    into ``sorted_idx_full`` avoids materializing the ``[g, m_pool]`` *index*
    gather and the ``exclude_a`` boolean-mask reshape (the dominant ``sample_epoch``
    cost); the result is bit-identical to the previous implementation.

    Returns ``(pos, neg, margin, valid)`` of length ``len(anchors)``; ``valid[i]``
    is False where the reference returns ``None`` (too few candidates, or no
    negative beyond the margin).

    Selection rule, identical to the scalar reference:
      - candidates sorted ascending by mining distance to the anchor;
      - with ``exclude_a`` the anchor itself (exactly one column) is removed;
      - ``m`` = candidate count, reject if ``m < 3``;
      - ``k = min(top_k_pos, max(1, m // 3))``;
      - positive at reduced position ``pos_ranks[i]`` (caller draws in ``[0, k)``);
      - ``j = searchsorted(dist, d_pos + min_margin, side="left")``, clamped up to
        ``k``; reject if ``j >= m``;
      - negative at reduced position ``j``; margin = ``dist[j] - d_pos``.
    """
    sd = sorted_dist_rows
    g, m_full = sd.shape
    m = m_full - 1 if exclude_a else m_full

    if m < 3:
        zeros = np.zeros(g, dtype=np.int64)
        return zeros, zeros, np.zeros(g, dtype=np.float32), np.zeros(g, dtype=bool)

    k = min(top_k_pos, max(1, m // 3))
    rows = np.arange(g)
    valid = np.ones(g, dtype=bool)

    # Reduced (exclude-self) positions map to full columns by skipping the self
    # column: full = reduced + (reduced >= self_pos). Self has distance 0 (the
    # matrix diagonal is exactly 0), so it sorts to a column <= any positive and
    # is always counted in the full-row scan below -- hence ``j_full - 1``.
    pos_col = pos_ranks + (pos_ranks >= self_pos).astype(np.int64) if exclude_a else pos_ranks
    d_pos = sd[rows, pos_col]
    thr = d_pos + min_margin
    # searchsorted(side="left") per row == count of candidates strictly less than
    # the threshold. Vectorized as a boolean row-sum over the FULL sorted row.
    j = (sd < thr[:, None]).sum(axis=1)
    if exclude_a:
        j = j - 1  # drop the always-counted self entry to get the reduced count
    j = np.maximum(j, k)
    valid &= j < m

    j_safe = np.minimum(j, m - 1)
    neg_col = j_safe + (j_safe >= self_pos).astype(np.int64) if exclude_a else j_safe
    pos = sorted_idx_full[anchors, pos_col].astype(np.int64)
    neg = sorted_idx_full[anchors, neg_col].astype(np.int64)
    margin = (sd[rows, neg_col] - d_pos).astype(np.float32)
    # Zero-out invalid rows so callers never read a bogus negative.
    margin = np.where(valid, margin, 0.0).astype(np.float32)
    return pos, neg, margin, valid


# ---------------------------------------------------------------------------
# Vectorized epoch sampler
# ---------------------------------------------------------------------------


class VectorizedSampler:
    """Draws a full epoch of 5-point samples without a per-sample Python loop.

    Mirrors ``BilingualSampler`` (hard-mining path) exactly in its precompute and
    selection logic. ``sample_epoch`` produces ``epoch_size`` valid samples
    (back-filling rejections by oversampling, as the reference loop does via its
    ``max_attempts`` budget). Indices are local to the concept's row range; the
    caller adds the concept's flat offset for the shared embedding gather.
    """

    def __init__(
        self,
        distance_matrix: np.ndarray,
        langs: np.ndarray,
        range_from: int,
        range_to: int,
        min_margin: float = MIN_TRIPLET_MARGIN,
        epoch_size: int = 64000,
        top_k_pos: int = DEFAULT_TOP_K_POS,
    ):
        self.min_margin = float(min_margin)
        self.epoch_size = epoch_size
        self.top_k_pos = top_k_pos

        indices_in_range = np.arange(range_from, range_to)
        unique_langs = np.unique(langs[indices_in_range])
        self.lang_indices: dict[str, np.ndarray] = {}
        for lang in unique_langs:
            self.lang_indices[lang] = indices_in_range[langs[indices_in_range] == lang]
        self.all_langs_list = list(self.lang_indices.keys())
        self.bilingual = len(self.all_langs_list) >= 2
        self._other_langs = {
            lang: [o for o in self.all_langs_list if o != lang] for lang in self.all_langs_list
        }

        # Per-language, per-anchor candidate order sorted ascending by distance.
        # Identical to BilingualSampler.__init__.
        n_rows = distance_matrix.shape[0]
        self._sorted_idx: dict[str, np.ndarray] = {}
        self._sorted_dist: dict[str, np.ndarray] = {}
        # Column of each own-language anchor's self-entry within its sorted row,
        # so ``pick_pairs_vectorized`` can drop self by index (exclude_a) without
        # the per-chunk [g, m] boolean-mask reshape. Only own-language anchors
        # (members of ``pool``) need it; -1 elsewhere (unused).
        self._self_pos: dict[str, np.ndarray] = {}
        for lang, pool in self.lang_indices.items():
            sub = distance_matrix[:, pool]
            order = np.argsort(sub, axis=1)
            self._sorted_idx[lang] = pool[order]
            self._sorted_dist[lang] = np.take_along_axis(sub, order, axis=1)
            sp = np.full(n_rows, -1, dtype=np.int64)
            sp[pool] = (self._sorted_idx[lang][pool] == pool[:, None]).argmax(axis=1)
            self._self_pos[lang] = sp

        self.last_reject_count = 0
        self.last_xl_count = 0

    def _k_for(self, lang: str, exclude_a: bool) -> int:
        m = len(self.lang_indices[lang]) - (1 if exclude_a else 0)
        if m < 3:
            return 0
        return min(self.top_k_pos, max(1, m // 3))

    def sample_epoch(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Return ``epoch_size`` valid samples as arrays (local row indices)."""
        if not self.all_langs_list:
            empty = np.zeros(0, dtype=np.int64)
            return {
                "anchor": empty,
                "sl_pos": empty,
                "sl_neg": empty,
                "xl_pos": empty,
                "xl_neg": empty,
                "margin_sl": np.zeros(0, np.float32),
                "margin_xl": np.zeros(0, np.float32),
                "has_xl": np.zeros(0, bool),
            }

        out: dict[str, list[np.ndarray]] = {
            k: [] for k in ("anchor", "sl_pos", "sl_neg", "xl_pos", "xl_neg")
        }
        out_m_sl: list[np.ndarray] = []
        out_m_xl: list[np.ndarray] = []
        out_has_xl: list[np.ndarray] = []

        n_have = 0
        reject_total = 0
        xl_total = 0
        # Oversample factor mirrors the reference max_attempts budget (epoch*20).
        # We draw in chunks until epoch_size valid samples accumulate.
        chunk = max(self.epoch_size, 4096)
        attempts = 0
        max_attempts = self.epoch_size * 20

        while n_have < self.epoch_size and attempts < max_attempts:
            need = self.epoch_size - n_have
            c = min(chunk, max(need * 2, 4096))
            attempts += c
            res = self._draw_chunk(rng, c)
            valid = res["valid"]
            reject_total += int((~valid).sum())
            keep = np.where(valid)[0]
            if len(keep) > need:
                keep = keep[:need]
            if len(keep) == 0:
                continue
            for k in out:
                out[k].append(res[k][keep])
            out_m_sl.append(res["margin_sl"][keep])
            out_m_xl.append(res["margin_xl"][keep])
            out_has_xl.append(res["has_xl"][keep])
            xl_total += int(res["has_xl"][keep].sum())
            n_have += len(keep)

        self.last_reject_count = reject_total
        self.last_xl_count = xl_total

        def cat(parts: list[np.ndarray], dtype) -> np.ndarray:
            if not parts:
                return np.zeros(0, dtype=dtype)
            return np.concatenate(parts).astype(dtype)

        return {
            "anchor": cat(out["anchor"], np.int64),
            "sl_pos": cat(out["sl_pos"], np.int64),
            "sl_neg": cat(out["sl_neg"], np.int64),
            "xl_pos": cat(out["xl_pos"], np.int64),
            "xl_neg": cat(out["xl_neg"], np.int64),
            "margin_sl": cat(out_m_sl, np.float32),
            "margin_xl": cat(out_m_xl, np.float32),
            "has_xl": cat(out_has_xl, np.bool_),
        }

    def _draw_chunk(self, rng: np.random.Generator, c: int) -> dict[str, np.ndarray]:
        """Draw ``c`` candidate samples (before validity filtering)."""
        anchor = np.zeros(c, dtype=np.int64)
        sl_pos = np.zeros(c, dtype=np.int64)
        sl_neg = np.zeros(c, dtype=np.int64)
        xl_pos = np.zeros(c, dtype=np.int64)
        xl_neg = np.zeros(c, dtype=np.int64)
        margin_sl = np.zeros(c, dtype=np.float32)
        margin_xl = np.zeros(c, dtype=np.float32)
        has_xl = np.zeros(c, dtype=bool)
        valid = np.zeros(c, dtype=bool)

        anchor_lang = rng.choice(self.all_langs_list, size=c)
        for lang in self.all_langs_list:
            grp = np.where(anchor_lang == lang)[0]
            if len(grp) == 0:
                continue
            pool = self.lang_indices[lang]
            if len(pool) < 3:
                # anchor_lang has too few rows: every such sample is rejected.
                continue
            a = pool[rng.integers(0, len(pool), size=len(grp))]
            anchor[grp] = a
            # XL defaults: xp = xn = a, margin 0, has_xl False (monolingual fallback).
            xl_pos[grp] = a
            xl_neg[grp] = a

            sl_rows_dist = self._sorted_dist[lang][a]
            k_sl = self._k_for(lang, exclude_a=True)
            if k_sl == 0:
                continue
            pr_sl = rng.integers(0, k_sl, size=len(grp))
            sp, sn, msl, vsl = pick_pairs_vectorized(
                self._sorted_idx[lang],
                sl_rows_dist,
                a,
                self._self_pos[lang][a],
                True,
                self.min_margin,
                self.top_k_pos,
                pr_sl,
            )
            sl_pos[grp] = sp
            sl_neg[grp] = sn
            margin_sl[grp] = msl
            valid[grp] = vsl

            if self.bilingual:
                others = self._other_langs[lang]
                other_lang = others[0] if len(others) == 1 else rng.choice(others, size=len(grp))
                # With two languages there is a single "other"; generalize anyway.
                if isinstance(other_lang, str):
                    other_lang_arr = np.full(len(grp), other_lang)
                else:
                    other_lang_arr = other_lang
                for ol in np.unique(other_lang_arr):
                    sub = np.where(other_lang_arr == ol)[0]  # positions within grp
                    if len(self.lang_indices[ol]) < 2:
                        continue
                    k_xl = self._k_for(ol, exclude_a=False)
                    if k_xl == 0:
                        continue
                    a_sub = a[sub]
                    xl_rows_dist = self._sorted_dist[ol][a_sub]
                    pr_xl = rng.integers(0, k_xl, size=len(sub))
                    xp, xn, mxl, vxl = pick_pairs_vectorized(
                        self._sorted_idx[ol],
                        xl_rows_dist,
                        a_sub,
                        None,
                        False,
                        self.min_margin,
                        self.top_k_pos,
                        pr_xl,
                    )
                    gpos = grp[sub]
                    # Only set XL where the pick succeeded (else keep a,a / has_xl False).
                    set_idx = gpos[vxl]
                    xl_pos[set_idx] = xp[vxl]
                    xl_neg[set_idx] = xn[vxl]
                    margin_xl[set_idx] = mxl[vxl]
                    has_xl[set_idx] = True

        return {
            "anchor": anchor,
            "sl_pos": sl_pos,
            "sl_neg": sl_neg,
            "xl_pos": xl_pos,
            "xl_neg": xl_neg,
            "margin_sl": margin_sl,
            "margin_xl": margin_xl,
            "has_xl": has_xl,
            "valid": valid,
        }


# ---------------------------------------------------------------------------
# Vectorized Adam (per-concept learning rate) and plateau scheduler
# ---------------------------------------------------------------------------


class BatchedAdam:
    """Adam over a stacked ``[B, *]`` parameter with a per-concept lr ``[B,1,1]``.

    Replicates ``torch.optim.Adam`` (single-tensor, non-fused) op-for-op:
      ``exp_avg = b1*exp_avg + (1-b1)*g`` (lerp)
      ``exp_avg_sq = b2*exp_avg_sq + (1-b2)*g*g``
      ``denom = exp_avg_sq.sqrt()/sqrt(bc2) + eps``
      ``p -= (lr/bc1) * exp_avg/denom``
    The only deviation is that the final step size is a per-concept tensor, so the
    fused ``addcdiv_`` (scalar value) is expanded to ``p -= step*exp_avg/denom``;
    this is the same sequence of rounded ops and matches torch to < 1e-6.
    """

    def __init__(
        self,
        param: torch.Tensor,
        lr: torch.Tensor,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.param = param
        self.lr = lr  # [B, 1, 1]
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.exp_avg = torch.zeros_like(param)
        self.exp_avg_sq = torch.zeros_like(param)
        self.step_t = 0

    def zero_grad(self) -> None:
        if self.param.grad is not None:
            self.param.grad = None

    @torch.no_grad()
    def step(self) -> None:
        grad = self.param.grad
        self.step_t += 1
        self.exp_avg.lerp_(grad, 1 - self.beta1)
        self.exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        bc1 = 1 - self.beta1**self.step_t
        bc2 = 1 - self.beta2**self.step_t
        step_size = self.lr / bc1  # [B,1,1]
        denom = (self.exp_avg_sq.sqrt() / (bc2**0.5)).add_(self.eps)
        self.param.add_(step_size * (self.exp_avg / denom), alpha=-1.0)


class BatchedPlateau:
    """Vectorized ``ReduceLROnPlateau`` (mode='min', threshold_mode='rel').

    Tracks ``best``/``num_bad_epochs`` per concept and reduces that concept's lr
    independently, replicating torch's algorithm (cooldown, eps guard, rel
    threshold) elementwise across the lr vector.
    """

    def __init__(
        self,
        lr: torch.Tensor,  # [B] (or [B,1,1]); we operate on a [B] view
        factor: float,
        patience: int,
        threshold: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        b = lr.shape[0]
        self.best = torch.full((b,), float("inf"))
        self.num_bad = torch.zeros(b, dtype=torch.long)
        self.cooldown_counter = torch.zeros(b, dtype=torch.long)

    @torch.no_grad()
    def step(self, metrics: torch.Tensor) -> None:
        # All bookkeeping is on CPU in float64 to match torch's Python-float
        # comparisons exactly (and because MPS lacks float64). Only the final lr
        # vector lives on the compute device; we copy the reduced values back.
        current = metrics.detach().cpu().to(torch.float64)
        better = current < self.best * (1.0 - self.threshold)
        self.best = torch.where(better, current, self.best)
        self.num_bad = torch.where(better, torch.zeros_like(self.num_bad), self.num_bad + 1)

        in_cooldown = self.cooldown_counter > 0
        self.cooldown_counter = torch.where(
            in_cooldown, self.cooldown_counter - 1, self.cooldown_counter
        )
        self.num_bad = torch.where(in_cooldown, torch.zeros_like(self.num_bad), self.num_bad)

        trigger = self.num_bad > self.patience
        lr_flat = self.lr.detach().cpu().to(torch.float64).reshape(-1)
        new_lr = torch.clamp(lr_flat * self.factor, min=self.min_lr)
        apply = trigger & ((lr_flat - new_lr) > self.eps)
        lr_flat = torch.where(apply, new_lr, lr_flat)
        self.lr.reshape(-1).copy_(lr_flat.to(self.lr.dtype))
        self.cooldown_counter = torch.where(
            trigger, torch.full_like(self.cooldown_counter, self.cooldown), self.cooldown_counter
        )
        self.num_bad = torch.where(trigger, torch.zeros_like(self.num_bad), self.num_bad)


# ---------------------------------------------------------------------------
# Batched forward / loss
# ---------------------------------------------------------------------------


def _cosine_distance_bt(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine distance over the last dim of ``[B, S, D]`` tensors.

    Identical formula to ``train_concept_layers.cosine_distance`` but applied per
    row of a batched tensor (``dim=-1`` instead of ``dim=1``).
    """
    dot = torch.sum(x1 * x2, dim=-1)
    n1 = torch.norm(x1, p=2, dim=-1)
    n2 = torch.norm(x2, p=2, dim=-1)
    return 1.0 - dot / (n1 * n2 + eps)


def batched_bilingual_loss(
    proj_a: torch.Tensor,
    proj_sp: torch.Tensor,
    proj_sn: torch.Tensor,
    proj_xp: torch.Tensor,
    proj_xn: torch.Tensor,
    margin_sl: torch.Tensor,
    margin_xl: torch.Tensor,
    has_xl: torch.Tensor,
) -> torch.Tensor:
    """Per-concept 4-term cosine triplet loss for a ``[B, S, OUTPUT_DIM]`` batch.

    Returns the per-concept mean over the ``S`` samples (shape ``[B]``). Equivalent
    to calling ``bilingual_cosine_loss`` on each concept's batch: the monolingual
    ``has_xl.any()`` short-circuit is replaced by an unconditional ``* has_xl``,
    which is numerically identical (masked terms contribute 0).
    """
    sl = torch.relu(
        _cosine_distance_bt(proj_a, proj_sp) - _cosine_distance_bt(proj_a, proj_sn) + margin_sl
    )
    xl_raw = torch.relu(
        _cosine_distance_bt(proj_a, proj_xp) - _cosine_distance_bt(proj_a, proj_xn) + margin_xl
    )
    xl = xl_raw * has_xl.to(xl_raw.dtype)
    return (sl + xl).mean(dim=1)


@dataclass
class ConceptSpec:
    """Everything needed to train one concept, ready for bucketing."""

    cid: str
    input_embs: torch.Tensor  # [n, INPUT_DIM], interleaved
    distance_matrix: np.ndarray  # [n, n]
    langs: np.ndarray  # [n]
    n: int
    train_to: int
    min_margin: float
    is_bilingual: bool


def prepare_concept(
    cid: str,
    c_input: torch.Tensor,
    c_mining: torch.Tensor,
    c_langs: list[str],
    settings: TrainSettings,
) -> ConceptSpec | None:
    """Reproduce ``train_one_concept`` setup (interleave, distance, split, margin).

    Returns ``None`` when the concept has fewer than ``MIN_SENTENCES_PER_CONCEPT``
    rows (the reference path skips it).
    """
    c_input, c_mining, c_langs, _ = interleave_by_language(
        c_input, c_mining, c_langs, [""] * len(c_langs)
    )
    n = c_input.shape[0]
    if n < MIN_SENTENCES_PER_CONCEPT:
        return None

    c_mining_np = c_mining.numpy()
    norms = np.linalg.norm(c_mining_np, axis=1, keepdims=True) + 1e-9
    normed = c_mining_np / norms
    sim = normed @ normed.T
    np.fill_diagonal(sim, 1.0)
    dist = 1.0 - sim

    lang_arr = np.array(c_langs)
    min_margin = dynamic_min_margin(dist, settings.margin_scale, settings.min_triplet_margin)

    n_val = int(n * settings.val_size)
    n_val = min(max(n_val, 1), n - 3) if n > 3 else 0
    train_to = n - n_val
    return ConceptSpec(
        cid=cid,
        input_embs=c_input,
        distance_matrix=dist,
        langs=lang_arr,
        n=n,
        train_to=train_to,
        min_margin=float(min_margin),
        is_bilingual=len(set(c_langs)) >= 2,
    )


# ---------------------------------------------------------------------------
# Bucket trainer
# ---------------------------------------------------------------------------


def _init_stacked_weights(b: int, device, seed: int | None) -> torch.Tensor:
    """Stack ``b`` independently-initialized ``Linear(INPUT_DIM, OUTPUT_DIM)``
    weights into a ``[b, OUTPUT_DIM, INPUT_DIM]`` tensor.

    Each weight is drawn by an actual ``nn.Linear`` so the init matches the
    reference path bit-for-bit when the global RNG state is aligned.
    """
    if seed is not None:
        torch.manual_seed(seed)
    weights = []
    for _ in range(b):
        layer = torch.nn.Linear(INPUT_DIM, OUTPUT_DIM, bias=False)
        weights.append(layer.weight.data.clone())
    return torch.stack(weights).to(device)


def train_concept_bucket(
    specs: list[ConceptSpec],
    settings: TrainSettings,
    device: torch.device,
    optimizer_mode: str = "vectorized",
    seed: int | None = None,
    frozen_train: list[dict[str, np.ndarray]] | None = None,
    frozen_val: list[dict[str, np.ndarray]] | None = None,
    epochs: int | None = None,
    init_weight: torch.Tensor | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Train a bucket of concepts together with one batched forward per step.

    ``frozen_train``/``frozen_val`` (one epoch dict per concept, local indices) let
    the equivalence tests feed identical triplets to both paths; when None the
    ``VectorizedSampler`` re-mines every epoch.

    Returns ``{cid: {"weight": [OUTPUT_DIM, INPUT_DIM]}}`` matching the reference
    checkpoint layout.
    """
    b = len(specs)
    if b == 0:
        return {}
    epochs = settings.epochs if epochs is None else epochs
    bs = settings.sample_batch_size

    # Flat embedding bank: concat all concepts' interleaved inputs, offset per
    # concept so the sampler's local indices index a single shared tensor.
    offsets = np.zeros(b, dtype=np.int64)
    acc = 0
    embs_parts = []
    for i, sp in enumerate(specs):
        offsets[i] = acc
        embs_parts.append(sp.input_embs)
        acc += sp.n
    flat_embs = torch.cat(embs_parts, dim=0).to(device)

    # Per-concept samplers (skip when frozen triplets are supplied).
    train_samplers: list[VectorizedSampler] = []
    val_samplers: list[VectorizedSampler] = []
    if frozen_train is None:
        with _ptime("sampler_init"):
            for sp in specs:
                train_samplers.append(
                    VectorizedSampler(
                        sp.distance_matrix,
                        sp.langs,
                        0,
                        sp.train_to,
                        min_margin=sp.min_margin,
                        epoch_size=settings.train_epoch_size,
                    )
                )
                val_samplers.append(
                    VectorizedSampler(
                        sp.distance_matrix,
                        sp.langs,
                        sp.train_to,
                        sp.n,
                        min_margin=sp.min_margin,
                        epoch_size=settings.val_epoch_size,
                    )
                )

    # ``init_weight`` lets equivalence tests pin the per-concept initialization so
    # bucketed and reference runs start identically (the global-RNG draw order
    # otherwise makes init depend on bucket position). Production uses the
    # per-concept nn.Linear init.
    if init_weight is not None:
        weight = init_weight.detach().clone().to(device)
    else:
        weight = _init_stacked_weights(b, device, seed)
    weight.requires_grad_(True)

    dropout_p = settings.dropout
    base_lr = settings.lr

    # Optimizer back-ends.
    if optimizer_mode == "per_concept_torch":
        params = [torch.nn.Parameter(weight[i].clone()) for i in range(b)]
        optimizers = [torch.optim.Adam([p], lr=base_lr) for p in params]
        schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=settings.lr_factor,
                patience=settings.lr_patience,
                threshold=1e-4,
            )
            for opt in optimizers
        ]
    elif optimizer_mode == "vectorized":
        lr_vec = torch.full((b, 1, 1), base_lr, dtype=weight.dtype, device=device)
        adam = BatchedAdam(weight, lr_vec)
        plateau = BatchedPlateau(lr_vec, factor=settings.lr_factor, patience=settings.lr_patience)
    else:
        raise ValueError(f"unknown optimizer_mode: {optimizer_mode}")

    rng = np.random.default_rng(seed if seed is not None else 0)
    final_train = np.zeros(b, dtype=np.float64)
    final_val = np.zeros(b, dtype=np.float64)

    def stacked_param() -> torch.Tensor:
        # Recomputed per call: in per_concept_torch mode the leaf params are
        # updated in place by their optimizers, so the stacked view must be
        # rebuilt every batch (a stale epoch-start stack would freeze the
        # weights). In vectorized mode this returns the in-place-updated tensor.
        if optimizer_mode == "per_concept_torch":
            return torch.stack(params)
        return weight

    def project(w: torch.Tensor, emb: torch.Tensor, training: bool) -> torch.Tensor:
        if dropout_p > 0:
            emb = F.dropout(emb, p=dropout_p, training=training)
        return torch.tanh(torch.bmm(emb, w.transpose(1, 2)))

    def run_epoch(ep: list[dict], epoch_size_eff: int, training: bool) -> torch.Tensor:
        per_batch_means: list[torch.Tensor] = []
        # Host marshaling hoisted out of the batch loop: build the five per-role
        # flat gather-index tensors and the three margin/mask tensors ONCE per epoch
        # (one np.stack + one H2D copy each) and then slice them on-device per batch.
        # This is index-only -- the [B, epoch, INPUT_DIM] embeddings are NEVER
        # materialized for the whole epoch (that would OOM); only [B, epoch] int64
        # index tensors are. Results-equivalent to the old per-batch np.stack path:
        # identical indices, identical gather, identical dropout draw order.
        with _ptime("host_marshal", device):
            idx_full = {
                key: torch.as_tensor(
                    np.stack([ep[i][key][:epoch_size_eff] + offsets[i] for i in range(b)]),
                    dtype=torch.long,
                    device=device,
                )
                for key in ("anchor", "sl_pos", "sl_neg", "xl_pos", "xl_neg")
            }
            m_sl_full = torch.as_tensor(
                np.stack([ep[i]["margin_sl"][:epoch_size_eff] for i in range(b)]), device=device
            )
            m_xl_full = torch.as_tensor(
                np.stack([ep[i]["margin_xl"][:epoch_size_eff] for i in range(b)]), device=device
            )
            h_xl_full = torch.as_tensor(
                np.stack([ep[i]["has_xl"][:epoch_size_eff] for i in range(b)]), device=device
            )
        for start in range(0, epoch_size_eff, bs):
            sl = slice(start, start + bs)
            cur = min(bs, epoch_size_eff - start)
            if cur <= 0:
                break
            # Rebuild the forward weight every batch so in-place optimizer
            # updates from the previous step are reflected (see stacked_param).
            w_for_fwd = stacked_param() if training else stacked_param().detach()
            with _ptime("host_marshal", device):
                m_sl = m_sl_full[:, sl]
                m_xl = m_xl_full[:, sl]
                h_xl = h_xl_full[:, sl]
                emb_a = flat_embs[idx_full["anchor"][:, sl]]
                emb_sp = flat_embs[idx_full["sl_pos"][:, sl]]
                emb_sn = flat_embs[idx_full["sl_neg"][:, sl]]
                emb_xp = flat_embs[idx_full["xl_pos"][:, sl]]
                emb_xn = flat_embs[idx_full["xl_neg"][:, sl]]
            with _ptime("forward", device):
                pa = project(w_for_fwd, emb_a, training)
                psp = project(w_for_fwd, emb_sp, training)
                psn = project(w_for_fwd, emb_sn, training)
                pxp = project(w_for_fwd, emb_xp, training)
                pxn = project(w_for_fwd, emb_xn, training)
                loss_per_c = batched_bilingual_loss(pa, psp, psn, pxp, pxn, m_sl, m_xl, h_xl)
            if training:
                total = loss_per_c.sum()
                with _ptime("backward", device):
                    if optimizer_mode == "per_concept_torch":
                        for opt in optimizers:
                            opt.zero_grad()
                        total.backward()
                        for opt in optimizers:
                            opt.step()
                    else:
                        adam.zero_grad()
                        total.backward()
                        adam.step()
                per_batch_means.append(loss_per_c.detach())
            else:
                per_batch_means.append(loss_per_c.detach())
        if not per_batch_means:
            return torch.zeros(b, device=device)
        return torch.stack(per_batch_means).mean(dim=0)  # mean of per-batch means

    mine_every = max(1, settings.mine_every)
    train_ep: list[dict[str, np.ndarray]] = []
    val_ep: list[dict[str, np.ndarray]] = []
    for epoch in range(epochs):
        if frozen_train is not None:
            train_ep = frozen_train
            val_ep = frozen_val if frozen_val is not None else frozen_train
        elif epoch % mine_every == 0:
            # Re-mine triplets; otherwise reuse the previous epoch's draw (B4).
            with _ptime("sample_epoch"):
                train_ep = [s.sample_epoch(rng) for s in train_samplers]
                val_ep = [s.sample_epoch(rng) for s in val_samplers]

        # In practice every concept back-fills to exactly epoch_size (measured
        # even for the smallest concepts), so these mins equal epoch_size. The min
        # guards the rectangular batch shape if a degenerate concept under-fills
        # under extreme rejection: all concepts are then truncated to the shortest.
        tr_size = min((len(e["anchor"]) for e in train_ep), default=0)
        va_size = min((len(e["anchor"]) for e in val_ep), default=0)

        train_means = run_epoch(train_ep, tr_size, training=True)
        with torch.no_grad():
            val_means = run_epoch(val_ep, va_size, training=False)

        final_train = train_means.detach().cpu().numpy()
        final_val = val_means.detach().cpu().numpy()

        with _ptime("scheduler", device):
            if optimizer_mode == "per_concept_torch":
                for i, sch in enumerate(schedulers):
                    sch.step(float(final_val[i]))
            else:
                plateau.step(val_means)

        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            logger.debug(
                f"  [bucket B={b}] epoch {epoch + 1}/{epochs} "
                f"train={final_train.mean():.4f} val={final_val.mean():.4f}"
            )

    final_weight = stacked_param().detach().cpu()
    return {
        sp.cid: {
            "weight": final_weight[i].clone(),
            "train_loss": float(final_train[i]),
            "val_loss": float(final_val[i]),
        }
        for i, sp in enumerate(specs)
    }
