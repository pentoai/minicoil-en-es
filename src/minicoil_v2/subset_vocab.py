"""Create a subset vocabulary from a concepts JSON file.

Extracts a named list of concept IDs from the full vocabulary and writes
concept_vocabulary.json and word_to_concept.json into a new output directory.
The output is identical in format to vocab build output, so the standard
scan → prune → embed pipeline works unchanged against that directory.

Usage:
    minicoil vocab subset 
    --concepts-file data/sample_200_pruned_concepts.json \
    --output-dir data/mini
"""

import json
from pathlib import Path

from loguru import logger

from minicoil_v2.settings import SubsetSettings


def run_subset(settings: SubsetSettings) -> None:
    """Extract a concept subset and write vocabulary files to output_dir."""
    concepts_file = Path(settings.concepts_file)
    source_dir = Path(settings.source_dir)
    output_dir = Path(settings.output_dir)

    logger.info(f"Loading concepts list from {concepts_file}...")
    with open(concepts_file) as f:
        concepts_data = json.load(f)
    concept_ids: list[str] = concepts_data["concept_ids"]
    logger.info(f"{len(concept_ids)} concept IDs to extract")

    logger.info(f"Loading full vocabulary from {source_dir}...")
    with open(source_dir / "concept_vocabulary.json") as f:
        full_vocab = json.load(f)
    full_concepts = full_vocab["concepts"]

    missing = [cid for cid in concept_ids if cid not in full_concepts]
    if missing:
        sample = f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        raise ValueError(f"{len(missing)} concept IDs not found in vocabulary: {sample}")

    subset_concepts = {cid: full_concepts[cid] for cid in concept_ids}

    total_en = sum(len(c["en"]) for c in subset_concepts.values())
    total_es = sum(len(c["es"]) for c in subset_concepts.values())

    new_vocab = {
        "concepts": subset_concepts,
        "stats": {
            "total_concepts": len(subset_concepts),
            "total_en_words": total_en,
            "total_es_words": total_es,
            "subset": {
                "source_file": str(concepts_file),
                "source_total": len(full_concepts),
                "kept": len(subset_concepts),
            },
        },
    }

    w2c: dict[str, dict[str, str]] = {"en": {}, "es": {}}
    for cid, concept in subset_concepts.items():
        for w in concept["en"]:
            w2c["en"][w] = cid
        for w in concept["es"]:
            w2c["es"][w] = cid

    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = output_dir / "concept_vocabulary.json"
    w2c_path = output_dir / "word_to_concept.json"

    with open(vocab_path, "w") as f:
        json.dump(new_vocab, f, ensure_ascii=False, indent=2)
    with open(w2c_path, "w") as f:
        json.dump(w2c, f, ensure_ascii=False)

    logger.info(
        f"Subset written to {output_dir}: "
        f"{len(subset_concepts)} concepts, "
        f"{total_en} EN words, {total_es} ES words"
    )
    logger.info(
        f"Next steps:\n"
        f"  uv run minicoil scan --data-dir {output_dir}\n"
        f"  uv run minicoil vocab prune --data-dir {output_dir} --threshold 23\n"
        f"  uv run minicoil embed --data-dir {output_dir} --store-cap 10000 --cloud-inference"
    )


if __name__ == "__main__":
    run_subset(SubsetSettings())
