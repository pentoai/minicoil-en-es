"""Frequency-prune concept vocabulary using Wikipedia sentence counts.

Keeps only concepts with >= min_sentences in BOTH English and Spanish.
Rewrites concept_vocabulary.json and word_to_concept.json with:
- Pruned concept set (contiguous re-numbered IDs)
- Updated word mappings
- Pruning metadata in stats

Usage:
    minicoil vocab prune --threshold 50
"""

import json
from pathlib import Path

from loguru import logger

from minicoil_v2.settings import VocabPruneSettings


def load_pruning_data(
    data_dir: Path,
) -> tuple[dict, dict[str, int], dict[str, int]]:
    """Load vocabulary and sentence counts."""
    with open(data_dir / "concept_vocabulary.json") as f:
        vocab = json.load(f)
    with open(data_dir / "concept_sentence_counts.json") as f:
        counts_data = json.load(f)
    return vocab, counts_data["counts"]["en"], counts_data["counts"]["es"]


def identify_viable_concepts(
    concepts: dict,
    en_counts: dict[str, int],
    es_counts: dict[str, int],
    threshold: int,
) -> list[str]:
    """Return concept IDs with >= threshold sentences in BOTH languages."""
    viable = [
        cid for cid in concepts if min(en_counts.get(cid, 0), es_counts.get(cid, 0)) >= threshold
    ]
    viable.sort(key=lambda x: int(x.split("-")[1]))
    return viable


def renumber_concepts(old_concepts: dict, viable_ids: list[str]) -> dict:
    """Re-number viable concepts to contiguous IDs, largest first."""
    sorted_viable_ids = sorted(
        viable_ids,
        key=lambda cid: (
            -(len(old_concepts[cid]["en"]) + len(old_concepts[cid]["es"])),
            cid,
        ),
    )
    return {
        f"C-{new_idx:05d}": old_concepts[old_cid]
        for new_idx, old_cid in enumerate(sorted_viable_ids)
    }


def rebuild_word_lookup(concepts: dict) -> dict[str, dict[str, str]]:
    """Rebuild word_to_concept lookup from concept vocabulary."""
    w2c: dict[str, dict[str, str]] = {"en": {}, "es": {}}
    for cid, concept in concepts.items():
        for w in concept["en"]:
            w2c["en"][w] = cid
        for w in concept["es"]:
            w2c["es"][w] = cid
    return w2c


def prune_and_save(settings: VocabPruneSettings) -> None:
    """Prune concept vocabulary by sentence frequency and save results."""
    data_dir = settings.data_dir
    threshold = settings.threshold

    vocab, en_counts, es_counts = load_pruning_data(data_dir)
    old_concepts = vocab["concepts"]
    old_total = len(old_concepts)

    viable_ids = identify_viable_concepts(old_concepts, en_counts, es_counts, threshold)
    new_concepts = renumber_concepts(old_concepts, viable_ids)
    new_w2c = rebuild_word_lookup(new_concepts)

    total_en_words = sum(len(c["en"]) for c in new_concepts.values())
    total_es_words = sum(len(c["es"]) for c in new_concepts.values())

    new_vocab = {
        "concepts": new_concepts,
        "stats": {
            "total_concepts": len(new_concepts),
            "total_en_words": total_en_words,
            "total_es_words": total_es_words,
            "pruning": {
                "threshold": threshold,
                "original_concepts": old_total,
                "kept": len(new_concepts),
                "pruned": old_total - len(new_concepts),
                "kept_pct": round(100 * len(new_concepts) / old_total, 1),
            },
        },
    }

    # Save pruned files (backup originals first)
    vocab_path = data_dir / "concept_vocabulary.json"
    w2c_path = data_dir / "word_to_concept.json"
    backup_vocab = data_dir / "concept_vocabulary_full.json"
    backup_w2c = data_dir / "word_to_concept_full.json"

    if not backup_vocab.exists():
        vocab_path.rename(backup_vocab)
        logger.info(f"Backed up original to {backup_vocab}")
    else:
        logger.info(f"Backup already exists: {backup_vocab}")

    if not backup_w2c.exists():
        w2c_path.rename(backup_w2c)
        logger.info(f"Backed up original to {backup_w2c}")
    else:
        logger.info(f"Backup already exists: {backup_w2c}")

    with open(vocab_path, "w") as f:
        json.dump(new_vocab, f, ensure_ascii=False)
    with open(w2c_path, "w") as f:
        json.dump(new_w2c, f, ensure_ascii=False)

    kept_pct = new_vocab["stats"]["pruning"]["kept_pct"]
    logger.info(
        f"Pruning complete (threshold={threshold}): "
        f"{old_total} -> {len(new_concepts)} concepts ({kept_pct}% kept)"
    )
    logger.info(f"EN words: {vocab['stats']['total_en_words']} -> {total_en_words}")
    logger.info(f"ES words: {vocab['stats']['total_es_words']} -> {total_es_words}")


if __name__ == "__main__":
    prune_and_save(VocabPruneSettings())
