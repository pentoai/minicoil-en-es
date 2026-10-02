"""Smoke test for the bilingual training pipeline with synthetic data.

Validates BilingualSampler, bilingual_cosine_loss, and train_concept_layer
without requiring Qdrant or real encoders.

Usage:
    uv run python tests/test_bilingual_training_smoke.py
"""

import sys

import numpy as np
import torch
from loguru import logger

from minicoil_v2.constants import OUTPUT_DIM
from minicoil_v2.train_concept_layers import (  # type: ignore[import-untyped]
    BilingualSampler,
    bilingual_cosine_loss,
    cosine_distance,
    interleave_by_language,
    train_concept_layer,
)

logger.remove()
logger.add(sys.stderr, level="DEBUG")


def make_synthetic_data(
    n_en: int = 200, n_es: int = 200, dim: int = 384, n_senses: int = 3
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, list[str]]:
    """Create synthetic bilingual data with clustered senses.

    Each language has n_senses clusters. EN cluster i and ES cluster i are
    translations (close in embedding space). Different clusters are far apart.
    """
    rng = np.random.default_rng(42)

    sense_centers = rng.standard_normal((n_senses, dim)).astype(np.float32)
    sense_centers /= np.linalg.norm(sense_centers, axis=1, keepdims=True)

    embeddings = []
    langs = []
    sentences = []

    for lang, n in [("en", n_en), ("es", n_es)]:
        for i in range(n):
            sense = i % n_senses
            noise = rng.standard_normal(dim).astype(np.float32) * 0.15
            if lang == "es":
                noise += rng.standard_normal(dim).astype(np.float32) * 0.05
            vec = sense_centers[sense] + noise
            vec /= np.linalg.norm(vec)
            embeddings.append(vec)
            langs.append(lang)
            sentences.append(f"{lang}_s{sense}_i{i}")

    emb_tensor = torch.tensor(np.stack(embeddings), dtype=torch.float32)
    lang_arr = np.array(langs)

    return emb_tensor, lang_arr, np.stack(embeddings), sentences


def build_distance_matrix(mining_embs: np.ndarray, langs: np.ndarray) -> np.ndarray:
    """Build and validate distance matrix."""
    norms = np.linalg.norm(mining_embs, axis=1, keepdims=True) + 1e-9
    normed = mining_embs / norms
    sim = normed @ normed.T
    np.fill_diagonal(sim, 1.0)
    dist = 1.0 - sim

    logger.info(f"Distance matrix: shape={dist.shape}")
    logger.info(f"  self-distance range: [{np.diag(dist).min():.4f}, {np.diag(dist).max():.4f}]")

    en_mask = langs == "en"
    es_mask = langs == "es"

    en_en = dist[np.ix_(en_mask, en_mask)]
    es_es = dist[np.ix_(es_mask, es_mask)]
    en_es = dist[np.ix_(en_mask, es_mask)]

    logger.info(f"  EN-EN distances: mean={en_en.mean():.4f} std={en_en.std():.4f}")
    logger.info(f"  ES-ES distances: mean={es_es.mean():.4f} std={es_es.std():.4f}")
    logger.info(f"  EN-ES distances: mean={en_es.mean():.4f} std={en_es.std():.4f}")

    return dist  # type: ignore


def run_sampler_smoke(
    input_embs: torch.Tensor,
    dist_matrix: np.ndarray,
    langs: np.ndarray,
    sentences: list[str],
) -> None:
    """Exercise BilingualSampler with small epoch size and inspect samples."""
    N = len(langs)
    train_to = int(N * 0.8)

    sampler = BilingualSampler(
        input_embs=input_embs,
        distance_matrix=dist_matrix,
        langs=langs,
        range_from=0,
        range_to=train_to,
        min_margin=0.1,
        batch_size=32,
        epoch_size=200,
    )

    logger.info(f"Sampler: range=[0, {train_to}), bilingual={sampler.bilingual}")
    for lang, idx in sampler.lang_indices.items():
        logger.info(f"  {lang}: {len(idx)} sentences in train range")

    n_batches = 0
    n_total = 0
    n_xl = 0
    all_margins_sl = []
    all_margins_xl = []

    for batch in sampler:
        n_batches += 1
        bs = len(batch["anchor"])
        n_total += bs
        n_xl += batch["has_xl"].sum()
        all_margins_sl.extend(batch["margin_sl"].tolist())
        all_margins_xl.extend(batch["margin_xl"][batch["has_xl"]].tolist())

        if n_batches == 1:
            logger.info(f"  First batch: {bs} samples")
            for i in range(min(5, bs)):
                a_idx = batch["anchor"][i]
                sp_idx = batch["sl_pos"][i]
                sn_idx = batch["sl_neg"][i]
                xp_idx = batch["xl_pos"][i]
                xn_idx = batch["xl_neg"][i]
                m_sl = batch["margin_sl"][i]
                m_xl = batch["margin_xl"][i]
                has = batch["has_xl"][i]

                a_lang = langs[a_idx]
                sp_lang = langs[sp_idx]
                sn_lang = langs[sn_idx]

                d_a_sp = dist_matrix[a_idx, sp_idx]
                d_a_sn = dist_matrix[a_idx, sn_idx]

                logger.info(
                    f"    sample {i}: a={sentences[a_idx]}({a_lang}) "
                    f"sl_pos={sentences[sp_idx]}({sp_lang}) d={d_a_sp:.3f} "
                    f"sl_neg={sentences[sn_idx]}({sn_lang}) d={d_a_sn:.3f} "
                    f"m_sl={m_sl:.3f} has_xl={has}"
                )
                if has:
                    xp_lang = langs[xp_idx]
                    xn_lang = langs[xn_idx]
                    d_a_xp = dist_matrix[a_idx, xp_idx]
                    d_a_xn = dist_matrix[a_idx, xn_idx]
                    logger.info(
                        f"           xl_pos={sentences[xp_idx]}({xp_lang}) d={d_a_xp:.3f} "
                        f"xl_neg={sentences[xn_idx]}({xn_lang}) d={d_a_xn:.3f} "
                        f"m_xl={m_xl:.3f}"
                    )

                    assert xp_lang != a_lang, (
                        f"XL positive should be different language: {xp_lang} vs {a_lang}"
                    )
                    assert xn_lang != a_lang, (
                        f"XL negative should be different language: {xn_lang} vs {a_lang}"
                    )
                    assert d_a_xp < d_a_xn, f"XL positive should be closer: {d_a_xp} >= {d_a_xn}"

                assert sp_lang == a_lang, (
                    f"SL positive should be same language: {sp_lang} vs {a_lang}"
                )
                assert sn_lang == a_lang, (
                    f"SL negative should be same language: {sn_lang} vs {a_lang}"
                )
                assert d_a_sp < d_a_sn, f"SL positive should be closer: {d_a_sp} >= {d_a_sn}"

    sampler.log_epoch_stats("test-sampler")

    logger.info(
        f"Sampler output: {n_total} samples in {n_batches} batches, "
        f"xl_rate={n_xl / max(n_total, 1):.1%}"
    )
    logger.info(f"  margin_sl: mean={np.mean(all_margins_sl):.3f} std={np.std(all_margins_sl):.3f}")
    if all_margins_xl:
        logger.info(
            f"  margin_xl: mean={np.mean(all_margins_xl):.3f} std={np.std(all_margins_xl):.3f}"
        )


def run_loss_smoke() -> None:
    """Test bilingual_cosine_loss with known values."""
    torch.manual_seed(42)
    B = 8
    D = 4

    anchor = torch.randn(B, D)
    sl_pos = anchor + torch.randn(B, D) * 0.1
    sl_neg = anchor + torch.randn(B, D) * 0.5
    xl_pos = anchor + torch.randn(B, D) * 0.15
    xl_neg = anchor + torch.randn(B, D) * 0.6
    margin_sl = torch.full((B,), 0.1)
    margin_xl = torch.full((B,), 0.1)
    has_xl = torch.ones(B, dtype=torch.bool)

    loss = bilingual_cosine_loss(
        anchor, sl_pos, sl_neg, xl_pos, xl_neg, margin_sl, margin_xl, has_xl
    )
    logger.info(f"Loss (all XL): {loss.item():.4f}")
    assert loss.item() >= 0, "Loss should be non-negative"

    d_a_sp = cosine_distance(anchor, sl_pos)
    d_a_sn = cosine_distance(anchor, sl_neg)
    logger.info(f"  SL: d(a,pos)={d_a_sp.mean():.4f} d(a,neg)={d_a_sn.mean():.4f}")
    assert d_a_sp.mean() < d_a_sn.mean(), "SL positive should be closer on average"

    has_xl_half = torch.zeros(B, dtype=torch.bool)
    has_xl_half[:4] = True
    loss_half = bilingual_cosine_loss(
        anchor, sl_pos, sl_neg, xl_pos, xl_neg, margin_sl, margin_xl, has_xl_half
    )
    logger.info(f"Loss (50% XL): {loss_half.item():.4f}")

    has_xl_none = torch.zeros(B, dtype=torch.bool)
    loss_none = bilingual_cosine_loss(
        anchor, sl_pos, sl_neg, xl_pos, xl_neg, margin_sl, margin_xl, has_xl_none
    )
    logger.info(f"Loss (no XL): {loss_none.item():.4f}")
    assert loss_none.item() <= loss.item(), "Monolingual loss should be <= full bilingual"


def run_training_smoke(
    input_embs: torch.Tensor,
    dist_matrix: np.ndarray,
    langs: np.ndarray,
) -> None:
    """Run a short training smoke pass with small epoch size."""
    N = len(langs)
    train_to = int(N * 0.8)

    train_sampler = BilingualSampler(
        input_embs=input_embs,
        distance_matrix=dist_matrix,
        langs=langs,
        range_from=0,
        range_to=train_to,
        min_margin=0.1,
        batch_size=64,
        epoch_size=500,
    )
    val_sampler = BilingualSampler(
        input_embs=input_embs,
        distance_matrix=dist_matrix,
        langs=langs,
        range_from=train_to,
        range_to=N,
        min_margin=0.1,
        batch_size=64,
        epoch_size=100,
    )

    device = torch.device("cpu")
    logger.info(f"Training: {train_to} train / {N - train_to} val sentences, 10 epochs")

    layer, final_train, final_val = train_concept_layer(
        train_sampler,
        val_sampler,
        device,
        epochs=10,
        lr=2e-3,
        dropout=0.05,
        lr_factor=0.5,
        lr_patience=3,
    )

    logger.info(f"Final: train={final_train:.4f} val={final_val:.4f}")
    logger.info(f"Layer weight shape: {layer.weight.shape}")
    assert layer.weight.shape == (OUTPUT_DIM, 384), (
        f"Expected ({OUTPUT_DIM}, 384), got {layer.weight.shape}"
    )
    assert final_train >= 0, "Train loss should be non-negative"
    assert final_val >= 0, "Val loss should be non-negative"


def test_bilingual_training_smoke() -> None:
    """Pytest entry: synthetic data, sampler, loss, and short training complete."""
    input_embs, lang_arr, mining_embs, sentences = make_synthetic_data(
        n_en=80, n_es=80, dim=384, n_senses=3
    )
    build_distance_matrix(mining_embs, lang_arr)
    input_embs, mining_tensor, lang_list, sent_list = interleave_by_language(
        input_embs,
        torch.from_numpy(mining_embs),
        lang_arr.tolist(),
        sentences,
    )
    lang_arr = np.array(lang_list)
    mining_embs_np = mining_tensor.numpy()
    norms = np.linalg.norm(mining_embs_np, axis=1, keepdims=True) + 1e-9
    normed = mining_embs_np / norms
    dist_matrix = 1.0 - normed @ normed.T
    np.fill_diagonal(dist_matrix, 0.0)

    run_loss_smoke()
    run_sampler_smoke(input_embs, dist_matrix, lang_arr, sent_list)
    run_training_smoke(input_embs, dist_matrix, lang_arr)


def main() -> None:
    logger.info("=" * 60)
    logger.info("Bilingual training pipeline smoke test")
    logger.info("=" * 60)

    logger.info("\n--- Generating synthetic data ---")
    input_embs, lang_arr, mining_embs, sentences = make_synthetic_data(
        n_en=200, n_es=200, dim=384, n_senses=3
    )
    n_en = int((lang_arr == "en").sum())
    n_es = int((lang_arr == "es").sum())
    logger.info(f"Data: {len(lang_arr)} sentences, {n_en} EN, {n_es} ES")

    logger.info("\n--- Testing distance matrix ---")
    dist_matrix = build_distance_matrix(mining_embs, lang_arr)

    logger.info("\n--- Interleaving by language ---")
    input_embs, mining_tensor, lang_list, sent_list = interleave_by_language(
        input_embs,
        torch.from_numpy(mining_embs),
        lang_arr.tolist(),
        sentences,
    )
    lang_arr = np.array(lang_list)
    mining_embs_np = mining_tensor.numpy()

    norms = np.linalg.norm(mining_embs_np, axis=1, keepdims=True) + 1e-9
    normed = mining_embs_np / norms
    dist_matrix = 1.0 - normed @ normed.T
    np.fill_diagonal(dist_matrix, 0.0)

    N = len(lang_arr)
    first_10 = [f"{lang_arr[i]}" for i in range(min(10, N))]
    logger.info(f"First 10 langs after interleave: {first_10}")

    logger.info("\n--- Testing loss function ---")
    run_loss_smoke()

    logger.info("\n--- Testing sampler ---")
    run_sampler_smoke(input_embs, dist_matrix, lang_arr, sent_list)

    logger.info("\n--- Testing training loop ---")
    run_training_smoke(input_embs, dist_matrix, lang_arr)

    logger.info("\n" + "=" * 60)
    logger.info("All smoke tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
