"""Empirically compare old (CC + salvage) vs new (Louvain) vocab algorithms.

Suite of metrics, each measuring a different aspect of "is this partition good":

1. MUSE pair recall stratified by edge weight (training-set fit + bridge-cutting)
2. Orphan count (does the salvage pass still earn its keep?)
3. Held-out MUSE validation (generalization, not just fit)
4. Translation density per cluster (how tight are the clusters structurally?)
5. Modularity on the weighted graph (sanity)
6. Polysemy probe (qualitative readable evidence on known-polysemous words)

Final verdict tallies wins/ties/losses across the quantitative metrics.

Usage:
    uv run python research/compare_vocab_quality.py
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import networkx as nx
from loguru import logger

from minicoil_v2.build_concept_vocab import (
    _find_communities,
    is_valid_word,
    parse_muse_dict,
)

EN_ES_PATH = Path("data/muse/en-es.txt")
ES_EN_PATH = Path("data/muse/es-en.txt")
MIN_CLUSTER_SIZE = 2
MAX_CLUSTER_SIZE = 20
MAX_TRANSLATIONS_PER_WORD = 2
RESOLUTION = 1.5
SEED = 42

HELDOUT_FRAC = 0.10
PROBE_WORDS = [
    # English polysemes
    "bank",
    "light",
    "bat",
    "plant",
    "crane",
    "club",
    "organ",
    "mole",
    "plot",
    "spring",
    "match",
    "pitch",
    "seal",
    "bark",
    # Spanish polysemes
    "banco",
    "planta",
    "vela",
    "cura",
    "llama",
    "gato",
    "muñeca",
    "carta",
    "sierra",
    "pila",
]


def build_weighted_graph(
    en_es_pairs: list[tuple[str, str]],
    es_en_pairs: list[tuple[str, str]],
    max_translations: int,
) -> nx.Graph:
    """Build the bipartite translation graph both algorithms operate on."""
    graph = nx.Graph()
    for pairs, src_lang, tgt_lang in [
        (en_es_pairs, "en", "es"),
        (es_en_pairs, "es", "en"),
    ]:
        counts: dict[str, int] = defaultdict(int)
        for src, tgt in pairs:
            if is_valid_word(src) and is_valid_word(tgt) and counts[src] < max_translations:
                src_key = f"{src_lang}:{src}"
                tgt_key = f"{tgt_lang}:{tgt}"
                if graph.has_edge(src_key, tgt_key):
                    graph[src_key][tgt_key]["weight"] += 1
                else:
                    graph.add_edge(src_key, tgt_key, weight=1)
                counts[src] += 1
    return graph


def partition_louvain(graph: nx.Graph) -> dict[str, int]:
    """Run Louvain + recursive split + bilingual/size filter.

    Returns word -> cluster_id. Words filtered out (monolingual / undersized)
    are absent from the mapping (treated as orphaned).
    """
    communities = _find_communities(graph, RESOLUTION, MAX_CLUSTER_SIZE, SEED)

    word_to_cluster: dict[str, int] = {}
    cid = 0
    for comm in communities:
        en = [w for w in comm if w.startswith("en:")]
        es = [w for w in comm if w.startswith("es:")]
        if not en or not es:
            continue
        if len(en) + len(es) < MIN_CLUSTER_SIZE:
            continue
        for w in comm:
            word_to_cluster[w] = cid
        cid += 1
    return word_to_cluster


def partition_cc_salvage(
    en_es_pairs: list[tuple[str, str]],
    es_en_pairs: list[tuple[str, str]],
    graph: nx.Graph,
) -> dict[str, int]:
    """Run BFS connected components + size filter + salvage pass (legacy).

    Ported from build_concept_vocab.py at commit 1e21418^.

    Returns word -> cluster_id. Includes salvaged pairs.
    """
    # Connected components via BFS on the same graph
    visited: set[str] = set()
    components: list[set[str]] = []
    for node in graph.nodes:
        if node in visited:
            continue
        component: set[str] = set()
        queue = [node]
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)

    word_to_cluster: dict[str, int] = {}
    en_covered: set[str] = set()
    es_covered: set[str] = set()
    en_to_cid: dict[str, int] = {}
    es_to_cid: dict[str, int] = {}
    cid = 0

    for comp in components:
        en_words = [w[3:] for w in comp if w.startswith("en:")]
        es_words = [w[3:] for w in comp if w.startswith("es:")]
        total = len(en_words) + len(es_words)
        if not en_words or not es_words:
            continue
        if total < MIN_CLUSTER_SIZE or total > MAX_CLUSTER_SIZE:
            continue
        for w in comp:
            word_to_cluster[w] = cid
        for w in en_words:
            en_covered.add(w)
            en_to_cid[w] = cid
        for w in es_words:
            es_covered.add(w)
            es_to_cid[w] = cid
        cid += 1

    # Salvage pass: top-1 translations for words dropped from oversized clusters
    en_to_es_top1: dict[str, str] = {}
    for en_w, es_w in en_es_pairs:
        if (
            is_valid_word(en_w)
            and is_valid_word(es_w)
            and en_w != es_w
            and en_w not in en_to_es_top1
        ):
            en_to_es_top1[en_w] = es_w
    es_to_en_top1: dict[str, str] = {}
    for es_w, en_w in es_en_pairs:
        if (
            is_valid_word(es_w)
            and is_valid_word(en_w)
            and en_w != es_w
            and es_w not in es_to_en_top1
        ):
            es_to_en_top1[es_w] = en_w

    for en_w, es_w in en_to_es_top1.items():
        if en_w in en_covered:
            continue
        if es_w in es_covered:
            target = es_to_cid[es_w]
        else:
            target = cid
            cid += 1
            word_to_cluster[f"es:{es_w}"] = target
            es_covered.add(es_w)
            es_to_cid[es_w] = target
        word_to_cluster[f"en:{en_w}"] = target
        en_covered.add(en_w)
        en_to_cid[en_w] = target

    for es_w, en_w in es_to_en_top1.items():
        if es_w in es_covered:
            continue
        if en_w in en_covered:
            target = en_to_cid[en_w]
        else:
            target = cid
            cid += 1
            word_to_cluster[f"en:{en_w}"] = target
            en_covered.add(en_w)
            en_to_cid[en_w] = target
        word_to_cluster[f"es:{es_w}"] = target
        es_covered.add(es_w)
        es_to_cid[es_w] = target

    return word_to_cluster


def cluster_contents(partition: dict[str, int]) -> dict[int, dict[str, list[str]]]:
    """Invert word->cid mapping into cid -> {en, es} word lists."""
    out: dict[int, dict[str, list[str]]] = defaultdict(lambda: {"en": [], "es": []})
    for node, cid in partition.items():
        if node.startswith("en:"):
            out[cid]["en"].append(node[3:])
        else:
            out[cid]["es"].append(node[3:])
    return out


def modularity(partition: dict[str, int], graph: nx.Graph) -> float:
    """Modularity of the partition on the weighted graph (Louvain's objective)."""
    by_cid: dict[int, set[str]] = defaultdict(set)
    for node, cid in partition.items():
        by_cid[cid].add(node)
    # Add singletons for nodes not in partition (so coverage is complete)
    placed = set(partition.keys())
    singletons = [{n} for n in graph.nodes if n not in placed]
    communities = list(by_cid.values()) + singletons
    return nx.community.modularity(graph, communities, weight="weight")


def translation_density(
    partition: dict[str, int],
    en_to_es: dict[str, set[str]],
) -> dict[str, float]:
    """For each cluster, fraction of (en, es) word pairs that are real MUSE edges.

    Density = MUSE-supported pairs / |EN| * |ES|. Higher = tighter cluster.
    """
    contents = cluster_contents(partition)
    densities = []
    for c in contents.values():
        en, es = c["en"], c["es"]
        if not en or not es:
            continue
        possible = len(en) * len(es)
        supported = sum(1 for e in en for s in es if s in en_to_es.get(e, set()))
        densities.append(supported / possible)
    if not densities:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0}
    densities.sort()
    n = len(densities)
    return {
        "mean": sum(densities) / n,
        "median": densities[n // 2],
        "p10": densities[max(0, n // 10)],
    }


def heldout_recall(
    en_es_pairs: list[tuple[str, str]],
    es_en_pairs: list[tuple[str, str]],
    heldout_pairs: list[tuple[str, str]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Build graph on a 90% subset, partition with both algos, score on held-out 10%.

    Returns (old_results, new_results) each with {recall, n_evaluated}.
    """
    graph = build_weighted_graph(en_es_pairs, es_en_pairs, MAX_TRANSLATIONS_PER_WORD)

    old_part = partition_cc_salvage(en_es_pairs, es_en_pairs, graph)
    new_part = partition_louvain(graph)

    def score(partition: dict[str, int]) -> dict[str, float]:
        evaluated = 0
        co = 0
        for en_w, es_w in heldout_pairs:
            en_key, es_key = f"en:{en_w}", f"es:{es_w}"
            cu = partition.get(en_key)
            cv = partition.get(es_key)
            if cu is None or cv is None:
                continue
            evaluated += 1
            if cu == cv:
                co += 1
        return {
            "recall": co / evaluated if evaluated else 0.0,
            "n_evaluated": evaluated,
            "n_total_heldout": len(heldout_pairs),
        }

    return score(old_part), score(new_part)


def evaluate(name: str, partition: dict[str, int], graph: nx.Graph) -> dict[str, float | int]:
    """Compute co-clustering rate per edge-weight bucket."""
    by_weight: dict[int, list[bool]] = defaultdict(list)
    for u, v, data in graph.edges(data=True):
        weight = int(data.get("weight", 1))
        cu = partition.get(u)
        cv = partition.get(v)
        co = cu is not None and cv is not None and cu == cv
        by_weight[weight].append(co)

    n_clusters = len(set(partition.values())) if partition else 0
    n_words = len(partition)
    total_nodes = graph.number_of_nodes()
    orphaned = total_nodes - n_words

    print(f"\n=== {name} ===")
    print(f"clusters:        {n_clusters}")
    print(f"words placed:    {n_words} / {total_nodes}")
    print(f"orphaned words:  {orphaned}")

    print("\n  edge weight    pairs    co-clustered    recall")
    print("  -----------    -----    ------------    ------")

    recalls: dict[int, float] = {}
    for w in sorted(by_weight):
        flags = by_weight[w]
        n = len(flags)
        co = sum(flags)
        rate = co / n if n else 0.0
        recalls[w] = rate
        print(f"  weight={w}       {n:6d}    {co:12d}    {rate:.3f}")

    if 1 in recalls and 2 in recalls:
        ratio = recalls[2] / recalls[1] if recalls[1] > 0 else float("inf")
        print(f"\n  ratio (w2 / w1 recall): {ratio:.3f}  <-- higher = better")
    return {
        "clusters": n_clusters,
        "words_placed": n_words,
        "orphaned": orphaned,
        **{f"recall_w{w}": r for w, r in recalls.items()},
    }


def main() -> None:
    if not EN_ES_PATH.exists() or not ES_EN_PATH.exists():
        logger.error(f"MUSE dicts not found at {EN_ES_PATH} / {ES_EN_PATH}")
        return

    en_es_pairs = parse_muse_dict(EN_ES_PATH)
    es_en_pairs = parse_muse_dict(ES_EN_PATH)
    logger.info(f"Parsed {len(en_es_pairs)} en->es, {len(es_en_pairs)} es->en pairs")

    graph = build_weighted_graph(en_es_pairs, es_en_pairs, MAX_TRANSLATIONS_PER_WORD)
    logger.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    by_w = defaultdict(int)
    for _, _, d in graph.edges(data=True):
        by_w[int(d.get("weight", 1))] += 1
    logger.info(f"Edge weights: {dict(by_w)}")

    logger.info("Running OLD partition (CC + salvage)...")
    old = partition_cc_salvage(en_es_pairs, es_en_pairs, graph)
    logger.info("Running NEW partition (Louvain + recursive split)...")
    new = partition_louvain(graph)

    old_stats = evaluate("OLD: connected components + salvage", old, graph)
    new_stats = evaluate("NEW: Louvain + recursive split", new, graph)

    # ------------------------------------------------------------------
    # Modularity (Louvain's objective; sanity check)
    # ------------------------------------------------------------------
    print("\n=== modularity on weighted graph ===")
    mod_old = modularity(old, graph)
    mod_new = modularity(new, graph)
    print(f"  OLD: {mod_old:.4f}")
    print(f"  NEW: {mod_new:.4f}   (delta {mod_new - mod_old:+.4f})")

    # ------------------------------------------------------------------
    # Translation density per cluster
    # ------------------------------------------------------------------
    print("\n=== translation density per cluster ===")
    print("  fraction of (en, es) word pairs in a cluster that are real MUSE edges")
    en_to_es: dict[str, set[str]] = defaultdict(set)
    for en_w, es_w in en_es_pairs:
        en_to_es[en_w].add(es_w)
    for es_w, en_w in es_en_pairs:
        en_to_es[en_w].add(es_w)

    old_dens = translation_density(old, en_to_es)
    new_dens = translation_density(new, en_to_es)
    for k in ("mean", "median", "p10"):
        delta = new_dens[k] - old_dens[k]
        print(f"  {k:7s}  OLD {old_dens[k]:.3f}  NEW {new_dens[k]:.3f}  (delta {delta:+.3f})")

    # ------------------------------------------------------------------
    # Held-out MUSE validation (generalization)
    # ------------------------------------------------------------------
    print("\n=== held-out MUSE validation ===")
    print(f"  hold {HELDOUT_FRAC:.0%} of dictionary pairs, build graph on rest, score on held-out")
    rng = random.Random(SEED)
    all_pairs = en_es_pairs + [(e, s) for s, e in es_en_pairs]
    # Dedup as (en, es) tuples
    unique = list({p for p in all_pairs if is_valid_word(p[0]) and is_valid_word(p[1])})
    rng.shuffle(unique)
    n_held = int(len(unique) * HELDOUT_FRAC)
    heldout = unique[:n_held]
    held_set = set(heldout)

    train_en_es = [(e, s) for e, s in en_es_pairs if (e, s) not in held_set]
    train_es_en = [(s, e) for s, e in es_en_pairs if (e, s) not in held_set]

    old_held, new_held = heldout_recall(train_en_es, train_es_en, heldout)
    print(
        f"  held-out pairs: {n_held} total, "
        f"OLD evaluated {old_held['n_evaluated']}, "
        f"NEW evaluated {new_held['n_evaluated']}"
    )
    print(f"  OLD co-cluster recall: {old_held['recall']:.3f}")
    print(
        f"  NEW co-cluster recall: {new_held['recall']:.3f}   "
        f"(delta {new_held['recall'] - old_held['recall']:+.3f})"
    )

    # ------------------------------------------------------------------
    # Polysemy probe (qualitative)
    # ------------------------------------------------------------------
    print("\n=== polysemy probe (cluster contents per known-polysemous word) ===")
    old_contents = cluster_contents(old)
    new_contents = cluster_contents(new)

    def fmt(c: dict[str, list[str]] | None) -> str:
        if c is None:
            return "(orphaned)"
        en = ",".join(sorted(c["en"])[:8]) + ("..." if len(c["en"]) > 8 else "")
        es = ",".join(sorted(c["es"])[:8]) + ("..." if len(c["es"]) > 8 else "")
        return f"EN={{{en}}} | ES={{{es}}}"

    for word in PROBE_WORDS:
        en_key, es_key = f"en:{word}", f"es:{word}"
        for key, lang in ((en_key, "EN"), (es_key, "ES")):
            old_cid = old.get(key)
            new_cid = new.get(key)
            if old_cid is None and new_cid is None:
                continue
            print(f"\n  [{lang}] {word!r}")
            print(f"    OLD: {fmt(old_contents.get(old_cid) if old_cid is not None else None)}")
            print(f"    NEW: {fmt(new_contents.get(new_cid) if new_cid is not None else None)}")

    # ------------------------------------------------------------------
    # Verdict tally
    # ------------------------------------------------------------------
    print("\n=== VERDICT TALLY ===")
    metrics = [
        ("orphan count (lower better)", old_stats["orphaned"], new_stats["orphaned"], "lower"),
        (
            "recall on weight=2 edges (higher better)",
            old_stats["recall_w2"],
            new_stats["recall_w2"],
            "higher",
        ),
        ("recall on weight=1 edges", old_stats["recall_w1"], new_stats["recall_w1"], "context"),
        ("modularity (higher better)", mod_old, mod_new, "higher"),
        ("translation density mean (higher better)", old_dens["mean"], new_dens["mean"], "higher"),
        ("translation density p10 (higher better)", old_dens["p10"], new_dens["p10"], "higher"),
        ("held-out recall (higher better)", old_held["recall"], new_held["recall"], "higher"),
    ]
    wins = ties = losses = 0
    for label, ov, nv, direction in metrics:
        if direction == "context":
            verdict = "context"
        elif direction == "higher":
            verdict = "WIN" if nv > ov else ("TIE" if nv == ov else "LOSS")
        else:
            verdict = "WIN" if nv < ov else ("TIE" if nv == ov else "LOSS")
        if verdict == "WIN":
            wins += 1
        elif verdict == "TIE":
            ties += 1
        elif verdict == "LOSS":
            losses += 1
        ov_s = f"{ov:.4f}" if isinstance(ov, float) else str(ov)
        nv_s = f"{nv:.4f}" if isinstance(nv, float) else str(nv)
        print(f"  [{verdict:6s}]  {label:48s}  OLD {ov_s}  NEW {nv_s}")

    print(
        f"\n  Quantitative metrics (excluding context): {wins} wins / {ties} ties / {losses} losses"
    )
    if losses == 0 and wins > 0:
        print("  >>> NEW IS BETTER on every measured axis.")
    elif wins > losses:
        print("  >>> NEW IS BETTER overall (some mixed signals).")
    elif wins == losses:
        print("  >>> MIXED: no clear winner.")
    else:
        print("  >>> OLD IS BETTER overall — investigate.")


if __name__ == "__main__":
    main()
