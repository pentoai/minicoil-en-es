"""Empirical check on the recursive Louvain resolution schedule.

The current code uses `resolution * 2` at each recursion level (exponential).
Reviewer asked whether a linear schedule (`+0.5` per level) would be safer.

This script runs both schedules on the same MUSE graph, reports recursion
depth, partition quality, and bridge-cutting behavior, so we can answer
empirically rather than by assertion.

Usage:
    uv run python research/compare_resolution_schedules.py
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import networkx as nx
from loguru import logger

from minicoil_v2.build_concept_vocab import is_valid_word, parse_muse_dict

EN_ES_PATH = Path("data/muse/en-es.txt")
ES_EN_PATH = Path("data/muse/es-en.txt")
MAX_CLUSTER_SIZE = 20
MAX_TRANSLATIONS = 2
BASE_RESOLUTION = 1.5
SEED = 42

ScheduleFn = Callable[[float, int], float]


def schedule_exponential(base: float, depth: int) -> float:
    """Current behaviour: resolution doubles each recursion level."""
    return base * (2**depth)


def schedule_linear(base: float, depth: int) -> float:
    """Conservative: add 0.5 per recursion level."""
    return base + 0.5 * depth


def build_weighted_graph() -> nx.Graph:
    en_es = parse_muse_dict(EN_ES_PATH)
    es_en = parse_muse_dict(ES_EN_PATH)
    graph = nx.Graph()
    for pairs, src_lang, tgt_lang in [(en_es, "en", "es"), (es_en, "es", "en")]:
        counts: dict[str, int] = defaultdict(int)
        for src, tgt in pairs:
            if is_valid_word(src) and is_valid_word(tgt) and counts[src] < MAX_TRANSLATIONS:
                u, v = f"{src_lang}:{src}", f"{tgt_lang}:{tgt}"
                if graph.has_edge(u, v):
                    graph[u][v]["weight"] += 1
                else:
                    graph.add_edge(u, v, weight=1)
                counts[src] += 1
    return graph


def find_communities_with_schedule(
    graph: nx.Graph,
    base_resolution: float,
    max_size: int,
    seed: int,
    schedule: ScheduleFn,
) -> tuple[list[set[str]], Counter, int]:
    """Louvain + recursive split using a parametric resolution schedule.

    Returns (communities, depth_histogram, n_unsplittable).
    """
    depth_hist: Counter = Counter()
    unsplittable = 0

    def recurse(subgraph: nx.Graph, nodes: set[str], depth: int) -> list[set[str]]:
        nonlocal unsplittable
        depth_hist[depth] += 1
        if depth == 0:
            comms = list(
                nx.community.louvain_communities(
                    subgraph,
                    weight="weight",
                    resolution=base_resolution,
                    seed=seed,
                )
            )
        else:
            res = schedule(base_resolution, depth)
            comms = list(
                nx.community.louvain_communities(
                    subgraph,
                    weight="weight",
                    resolution=res,
                    seed=seed,
                )
            )

        out: list[set[str]] = []
        # If the recursive call produced exactly one cluster (no progress), bail.
        if depth > 0 and len(comms) <= 1:
            unsplittable += 1
            return [nodes]

        for c in comms:
            if len(c) <= max_size:
                out.append(c)
            else:
                out.extend(recurse(graph.subgraph(c), c, depth + 1))
        return out

    return recurse(graph, set(graph.nodes), 0), depth_hist, unsplittable


def filter_to_concepts(communities: list[set[str]], min_size: int = 2) -> dict[str, int]:
    """Drop monolingual / undersized clusters; return word -> cluster_id."""
    out: dict[str, int] = {}
    cid = 0
    for comm in communities:
        en = [w for w in comm if w.startswith("en:")]
        es = [w for w in comm if w.startswith("es:")]
        if not en or not es:
            continue
        if len(en) + len(es) < min_size:
            continue
        for w in comm:
            out[w] = cid
        cid += 1
    return out


def co_cluster_recall(partition: dict[str, int], graph: nx.Graph) -> dict[int, float]:
    """Recall per edge weight bucket."""
    by_weight: dict[int, list[bool]] = defaultdict(list)
    for u, v, data in graph.edges(data=True):
        weight = int(data.get("weight", 1))
        cu = partition.get(u)
        cv = partition.get(v)
        by_weight[weight].append(cu is not None and cv is not None and cu == cv)
    return {w: (sum(flags) / len(flags) if flags else 0.0) for w, flags in by_weight.items()}


def modularity_score(partition: dict[str, int], graph: nx.Graph) -> float:
    by_cid: dict[int, set[str]] = defaultdict(set)
    for node, cid in partition.items():
        by_cid[cid].add(node)
    placed = set(partition.keys())
    singletons = [{n} for n in graph.nodes if n not in placed]
    return nx.community.modularity(graph, list(by_cid.values()) + singletons, weight="weight")


def cluster_size_stats(partition: dict[str, int]) -> dict[str, float]:
    sizes_by_cid: dict[int, int] = defaultdict(int)
    for cid in partition.values():
        sizes_by_cid[cid] += 1
    sizes = sorted(sizes_by_cid.values())
    if not sizes:
        return {"n_clusters": 0, "min": 0, "max": 0, "median": 0, "mean": 0.0}
    n = len(sizes)
    return {
        "n_clusters": n,
        "min": sizes[0],
        "max": sizes[-1],
        "median": sizes[n // 2],
        "mean": round(sum(sizes) / n, 2),
        "n_size_2": sum(1 for s in sizes if s == 2),
    }


def run_one(name: str, schedule: ScheduleFn, graph: nx.Graph) -> dict:
    logger.info(f"--- running schedule: {name} ---")
    communities, depth_hist, unsplittable = find_communities_with_schedule(
        graph, BASE_RESOLUTION, MAX_CLUSTER_SIZE, SEED, schedule
    )
    partition = filter_to_concepts(communities)
    recall = co_cluster_recall(partition, graph)
    mod = modularity_score(partition, graph)
    sizes = cluster_size_stats(partition)

    return {
        "name": name,
        "depth_hist": depth_hist,
        "unsplittable": unsplittable,
        "n_communities_pre_filter": len(communities),
        "partition": partition,
        "recall_w1": recall.get(1, 0.0),
        "recall_w2": recall.get(2, 0.0),
        "modularity": mod,
        "sizes": sizes,
    }


def report(results: list[dict]) -> None:
    print("\n=== recursion depth histogram (how often we ran Louvain at each level) ===")
    print("  depth        " + "  ".join(f"{r['name']:>14s}" for r in results))
    all_depths = sorted({d for r in results for d in r["depth_hist"]})
    for d in all_depths:
        row = "  ".join(f"{r['depth_hist'].get(d, 0):>14d}" for r in results)
        label = f"depth={d}" + ("  (initial)" if d == 0 else "")
        print(f"  {label:11s}  {row}")
    print(f"  unsplittable {' ' * 0}" + "  ".join(f"{r['unsplittable']:>14d}" for r in results))

    print("\n=== partition quality ===")
    keys = [
        ("n clusters", lambda r: r["sizes"]["n_clusters"]),
        ("max cluster size", lambda r: r["sizes"]["max"]),
        ("median cluster size", lambda r: r["sizes"]["median"]),
        ("mean cluster size", lambda r: r["sizes"]["mean"]),
        ("# size-2 clusters", lambda r: r["sizes"].get("n_size_2", 0)),
        ("modularity", lambda r: round(r["modularity"], 4)),
        ("recall on w=2 edges", lambda r: round(r["recall_w2"], 4)),
        ("recall on w=1 edges", lambda r: round(r["recall_w1"], 4)),
    ]
    print(f"  {'metric':24s}" + "  ".join(f"{r['name']:>14s}" for r in results))
    for label, fn in keys:
        row = "  ".join(f"{fn(r):>14}" for r in results)
        print(f"  {label:24s}{row}")

    if len(results) >= 2:
        a, b = results[0], results[1]
        print(f"\n=== verdict: {b['name']} vs {a['name']} ===")
        deltas = [
            ("modularity", b["modularity"] - a["modularity"]),
            ("recall on w=2 edges", b["recall_w2"] - a["recall_w2"]),
            ("recall on w=1 edges", b["recall_w1"] - a["recall_w1"]),
            ("n size-2 clusters", b["sizes"].get("n_size_2", 0) - a["sizes"].get("n_size_2", 0)),
        ]
        for label, delta in deltas:
            if isinstance(delta, float):
                print(f"  {label:24s}  delta {delta:+.4f}")
            else:
                print(f"  {label:24s}  delta {delta:+d}")


def main() -> None:
    if not EN_ES_PATH.exists() or not ES_EN_PATH.exists():
        logger.error(f"MUSE dicts not found at {EN_ES_PATH}, {ES_EN_PATH}")
        return
    graph = build_weighted_graph()
    logger.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    exp = run_one("exponential", schedule_exponential, graph)
    lin = run_one("linear (+0.5)", schedule_linear, graph)

    report([exp, lin])


if __name__ == "__main__":
    main()
