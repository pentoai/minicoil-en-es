"""Phase-2 demand analysis: how many concepts (full on-disk vocab) are needed to
cover the mMARCO query content words?

Demand-driven vocabulary selection: instead of training the concepts Wikipedia
happens to supply (Phase 0/1: 64), find the MINIMAL concept set that covers the
vast majority of what the eval will ask. Metric: fraction of content-token
occurrences (stopword/punct-filtered, per fastembed's own language stopword
lists) in queries that belong to a selected concept, selecting concepts
greedily by demand count.

Leakage guard: concepts are selected on an 80% slice of the full judged query
pool (sha1(qid)%5 != 0); the held-out 20% is reserved for the Phase-2 sealed
test carve. Coverage is reported on both slices so generalization is visible.

Writes the frozen selection artifact `data/eval/mmarco/phase2_concepts.json`:
the demand-ranked list to the 90% mark, with per-concept demand counts and the
coverage table (spec: docs/research/phase2-scale-concepts-design.md).
"""

import hashlib
import json
from collections import Counter

from fastembed.sparse.bm25 import Bm25

from minicoil_v2.eval.datasets.mmarco import MMARCODataset

with open("data/word_to_concept.json") as f:
    w2c: dict[str, dict[str, str]] = json.load(f)

ds = MMARCODataset()
bm25 = {
    lang: Bm25("Qdrant/bm25", language=name)
    for lang, name in [("en", "english"), ("es", "spanish")]
}


def in_select_pool(qid: str) -> bool:
    return int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 5 != 0


def content_tokens(text: str, lang: str) -> list[str]:
    m = bm25[lang]
    toks = []
    for tok in m.tokenizer.tokenize(text):
        lower = tok.lower()
        if tok in m.punctuation or lower in m.stopwords or len(tok) > m.token_max_length:
            continue
        toks.append(lower)
    return toks


# demand per concept (occurrences in selection-pool queries, en+es), plus totals
demand: Counter[str] = Counter()
totals = {"select": 0, "heldout": 0}
matched = {"select": Counter(), "heldout": Counter()}  # cid -> occurrences per pool
for pair, lang in [("eng-eng", "en"), ("spa-spa", "es")]:
    queries = ds.queries(pair, "validation")
    for qid, text in queries.items():
        pool = "select" if in_select_pool(qid) else "heldout"
        for tok in content_tokens(text, lang):
            totals[pool] += 1
            cid = w2c[lang].get(tok)
            if cid is not None:
                matched[pool][cid] += 1
                if pool == "select":
                    demand[cid] += 1

vocab_cover = {p: sum(c.values()) / totals[p] for p, c in matched.items()}
print(f"queries: {sum(len(ds.queries(p, 'validation')) for p in ('eng-eng', 'spa-spa'))} texts")
print(f"content-token occurrences: select={totals['select']} heldout={totals['heldout']}")
print(
    f"in-vocab ceiling (ALL {len(set(w2c['en'].values()) | set(w2c['es'].values()))} "
    f"concepts): select={vocab_cover['select']:.3f} heldout={vocab_cover['heldout']:.3f}"
)
print(f"distinct concepts in demand: {len(demand)}")

ranked = [cid for cid, _ in demand.most_common()]
cum = 0
total_matched = sum(demand.values())
marks = {0.80: None, 0.90: None, 0.95: None, 0.99: None}
for i, cid in enumerate(ranked, 1):
    cum += demand[cid]
    for m in marks:
        if marks[m] is None and cum / total_matched >= m:
            marks[m] = i

coverage_table = {}
print("\nconcepts needed for X% of MATCHED occurrences (and abs. coverage incl. OOV):")
for m, n in marks.items():
    sel = set(ranked[:n])
    abs_sel = sum(v for c, v in matched["select"].items() if c in sel) / totals["select"]
    abs_held = sum(v for c, v in matched["heldout"].items() if c in sel) / totals["heldout"]
    coverage_table[f"{m:.0%}"] = {
        "concepts": n,
        "abs_coverage_select": round(abs_sel, 4),
        "abs_coverage_heldout": round(abs_held, 4),
    }
    print(
        f"  {m:.0%} -> {n:5d} concepts   abs coverage: select={abs_sel:.3f} heldout={abs_held:.3f}"
    )

# Freeze the 90%-mark selection (spec default), demand-ranked so callers can
# walk the list down or truncate further by supply.
n90 = marks[0.90]
out = {
    "selection_rule": "demand-ranked to 90% of matched occurrences; "
    "select pool = sha1(qid)%5 != 0 over all judged dev.small qids (en+es)",
    "vocab_ceiling_select": round(vocab_cover["select"], 4),
    "vocab_ceiling_heldout": round(vocab_cover["heldout"], 4),
    "coverage_table": coverage_table,
    "concepts": [{"cid": c, "demand": demand[c]} for c in ranked[:n90]],
}
path = "data/eval/mmarco/phase2_concepts.json"
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print(f"\nwrote {path} ({n90} concepts)")
