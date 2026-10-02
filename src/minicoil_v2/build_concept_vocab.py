"""Build concept vocabulary from MUSE bilingual dictionaries.

Takes en-es and es-en MUSE dictionary files and produces a concept vocabulary
where each concept groups words from both languages that are translations of
each other.

Algorithm:
1. Parse both dictionaries, filter to valid words, keep top-K per source word
2. Build a weighted bipartite translation graph
3. Louvain community detection to find dense bilingual clusters
4. Recursively split oversized communities at increasing resolution
5. Filter out monolingual/undersized clusters, assign concept IDs
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
from loguru import logger

from minicoil_v2.settings import VocabBuildSettings

_VALID_WORD_RE = re.compile(r"^[a-záéíóúüñ]+$")


def parse_muse_dict(path: Path) -> list[tuple[str, str]]:
    """Parse a MUSE dictionary file into (source, target) pairs."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((parts[0].lower(), parts[1].lower()))
    return pairs


def is_valid_word(word: str) -> bool:
    """Keep only alphabetic words with 2+ chars (EN + ES charset)."""
    return len(word) >= 2 and _VALID_WORD_RE.match(word) is not None


def _find_communities(
    graph: nx.Graph,
    resolution: float,
    max_size: int,
    seed: int,
) -> list[set[str]]:
    """Louvain community detection, recursively splitting oversized communities."""
    communities = list(
        nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=resolution,
            seed=seed,
        )
    )

    result: list[set[str]] = []
    for comm in communities:
        if len(comm) <= max_size:
            result.append(comm)
            continue

        subs = list(
            nx.community.louvain_communities(
                graph.subgraph(comm),
                weight="weight",
                resolution=resolution + 0.5,
                seed=seed,
            )
        )
        if len(subs) <= 1:
            logger.warning(f"Unsplittable community: {len(comm)} nodes (max={max_size})")
            result.append(comm)
            continue

        for sub in subs:
            if len(sub) <= max_size:
                result.append(sub)
            else:
                result.extend(
                    _find_communities(
                        graph.subgraph(sub),
                        resolution + 0.5,
                        max_size,
                        seed,
                    )
                )

    return result


def build_concept_vocabulary(
    en_es_path: Path,
    es_en_path: Path,
    min_cluster_size: int = 2,
    max_cluster_size: int = 20,
    max_translations_per_word: int = 2,
    resolution: float = 1.5,
    seed: int = 42,
) -> dict:
    """Build concept vocabulary using Louvain on a weighted translation graph."""
    en_es_pairs = parse_muse_dict(en_es_path)
    es_en_pairs = parse_muse_dict(es_en_path)
    logger.info(f"Parsed {len(en_es_pairs)} en->es pairs, {len(es_en_pairs)} es->en pairs")

    # Build weighted bipartite graph, pruning to top-K translations per source
    # word (MUSE is frequency-ordered). Mutual translations (appearing in both
    # dictionaries) naturally accumulate higher weight.
    graph = nx.Graph()
    for pairs, src_lang, tgt_lang in [
        (en_es_pairs, "en", "es"),
        (es_en_pairs, "es", "en"),
    ]:
        counts: dict[str, int] = defaultdict(int)
        for src, tgt in pairs:
            if (
                is_valid_word(src)
                and is_valid_word(tgt)
                and counts[src] < max_translations_per_word
            ):
                src_key = f"{src_lang}:{src}"
                tgt_key = f"{tgt_lang}:{tgt}"
                if graph.has_edge(src_key, tgt_key):
                    graph[src_key][tgt_key]["weight"] += 1
                else:
                    graph.add_edge(src_key, tgt_key, weight=1)
                counts[src] += 1

    logger.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Find communities with recursive splitting for oversized ones
    communities = _find_communities(graph, resolution, max_cluster_size, seed)
    logger.info(
        f"Found {len(communities)} communities "
        f"(resolution={resolution}, max_size={max_cluster_size})"
    )

    # Filter to bilingual clusters above minimum size
    clusters: list[dict[str, list[str]]] = []
    monolingual = 0
    undersized = 0

    for comm in communities:
        en = sorted(w.removeprefix("en:") for w in comm if w.startswith("en:"))
        es = sorted(w.removeprefix("es:") for w in comm if w.startswith("es:"))

        if not en or not es:
            monolingual += 1
        elif len(en) + len(es) < min_cluster_size:
            undersized += 1
        else:
            clusters.append({"en": en, "es": es})

    # Assign IDs, largest concepts first
    clusters.sort(key=lambda c: -(len(c["en"]) + len(c["es"])))
    concepts = {f"C-{i:05d}": c for i, c in enumerate(clusters)}

    # Build reverse lookups
    en_word_to_concept: dict[str, str] = {}
    es_word_to_concept: dict[str, str] = {}
    for cid, c in concepts.items():
        for w in c["en"]:
            en_word_to_concept[w] = cid
        for w in c["es"]:
            es_word_to_concept[w] = cid

    # Stats
    sizes = [len(c["en"]) + len(c["es"]) for c in concepts.values()]
    sorted_sizes = sorted(sizes) if sizes else []
    n = len(sorted_sizes)

    logger.info(
        f"Built {len(concepts)} concepts, "
        f"{len(en_word_to_concept)} EN words, {len(es_word_to_concept)} ES words "
        f"(filtered {monolingual} monolingual, {undersized} undersized)"
    )

    return {
        "concepts": concepts,
        "en_word_to_concept": en_word_to_concept,
        "es_word_to_concept": es_word_to_concept,
        "stats": {
            "total_concepts": len(concepts),
            "total_en_words": len(en_word_to_concept),
            "total_es_words": len(es_word_to_concept),
            "filtered_monolingual": monolingual,
            "filtered_undersized": undersized,
            "cluster_size_distribution": {
                "min": sorted_sizes[0] if n else 0,
                "max": sorted_sizes[-1] if n else 0,
                "mean": round(sum(sizes) / n, 2) if n else 0,
                "median": sorted_sizes[n // 2] if n else 0,
                "p90": sorted_sizes[int(n * 0.9)] if n else 0,
                "p99": sorted_sizes[int(n * 0.99)] if n else 0,
            },
        },
    }


def build_and_save(settings: VocabBuildSettings) -> None:
    """Build concept vocabulary and save to disk."""
    if not settings.en_es_path.exists() or not settings.es_en_path.exists():
        logger.error(f"MUSE dictionaries not found at {settings.en_es_path.parent}")
        logger.info("Download them first:")
        logger.info("  curl -O https://dl.fbaipublicfiles.com/arrival/dictionaries/en-es.txt")
        logger.info("  curl -O https://dl.fbaipublicfiles.com/arrival/dictionaries/es-en.txt")
        return

    result = build_concept_vocabulary(
        settings.en_es_path,
        settings.es_en_path,
        min_cluster_size=settings.min_cluster_size,
        max_cluster_size=settings.max_cluster_size,
        max_translations_per_word=settings.max_translations_per_word,
        resolution=settings.resolution,
    )

    vocab_path = settings.output_dir / "concept_vocabulary.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {"concepts": result["concepts"], "stats": result["stats"]},
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Saved concept vocabulary to {vocab_path}")

    lookup_path = settings.output_dir / "word_to_concept.json"
    with open(lookup_path, "w", encoding="utf-8") as f:
        json.dump(
            {"en": result["en_word_to_concept"], "es": result["es_word_to_concept"]},
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Saved word -> concept lookup to {lookup_path}")


if __name__ == "__main__":
    build_and_save(VocabBuildSettings())
