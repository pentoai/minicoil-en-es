# Evaluation

**Package:** `src/minicoil_v2/eval/`
**Command:** `minicoil eval` (`run`, `splits`, `baselines`)

A deterministic, approach-agnostic harness. Any retriever that implements `index` and
`search` runs on frozen splits, gets a stamped report, and is checked against locked
baselines by a hardcoded gate. It is built so that iterating on the model cannot leak
into the test numbers.

## Benchmark

**mMARCO** (`unicamp-dl/mmarco`, pinned revision), the machine-translated MS MARCO, in
English and Spanish:

- **Queries:** the 6,980 judged `dev.small` queries in each language.
- **Corpus:** the gold sub-corpus, about 7.4k passages per language.
- **Cross-lingual gold:** the same passage id in the other language.
- **Pairs** are named `<query>-<corpus>`: `eng-eng`, `spa-spa`, `eng-spa` (English
  queries, Spanish passages), `spa-eng`.
- **Metrics:** MRR@10 (the gate metric, the official MS MARCO metric), nDCG@10 and R@100.

MLQA is still supported (`--dataset mlqa`) and was the benchmark before mMARCO; its numbers
are not comparable. `mmarco_open` adds N non-gold distractor passages for an open-domain
haystack: set `MINICOIL_MMARCO_OPEN_N_EN` / `_N_ES`. At N=0 it reproduces the gold-pool
numbers exactly, which is its validity check.

## Two gates

| Gate | Splits | Baselines | What it measures |
|---|---|---|---|
| **Covered slice** (headline, model card) | `data/eval/splits_mmarco_phase2/` | `data/eval/baselines_mmarco_phase2.json` | Only queries that share a trained concept with their gold passage (`eval/coverage.py`, the encoder's matcher on surface forms; frozen before the lemma fallback existed). Per pair: dev 200, val 2,000, test 3,491 / 3,341 / 3,145 / 3,248 |
| **All queries** | `data/eval/splits_mmarco_all/` | `data/eval/baselines_mmarco_all.json` | Every query (`--no-coverage-filter`). Shows the coverage cliff: about 22% of queries have no shared concept, and cross-lingually ~40% of those share no sparse index with their gold at all |

**Caveat on concept selection.** The Phase-2 design reserved a 20% qid slice
(`sha1(qid) % 5 == 0`) for the test carve, and concepts were selected on the other 80%.
The split builder, however, shuffled all covered qids, so only about 18% of each test
split (650 of 3,491 for eng-eng) comes from the held-out slice. Selection used query text
only, never relevance labels, but the concept set was chosen with most test queries in
view. Concept coverage on test queries is therefore optimistic: on unseen queries, fewer
would be covered, and the all-queries numbers would likely be lower. A clean re-carve
would restrict test to the held-out slice and re-lock the baselines.

## Baselines and win conditions

| Retriever | What it is |
|---|---|
| `bm25` | FastEmbed `Qdrant/bm25` |
| `minicoil-v1` | FastEmbed `Qdrant/minicoil-v1` (English only) |
| `translate-bm25` | NLLB-200-distilled-600M translation of the query, then BM25 (translations cached on disk) |
| `minicoil-v2` | This model |

The gate (`eval/baselines/check.py`, `WIN_CONDITION`) requires beating, on test MRR@10:

| Pair | Bar | Locked test MRR@10 (covered slice) |
|---|---|---|
| eng-eng | `minicoil-v1` | 0.8921 |
| spa-spa | `bm25` | 0.7953 |
| eng-spa | `translate-bm25` | 0.7028 |
| spa-eng | `translate-bm25` | 0.7741 |

## Running it

Qdrant must be running (`docker compose up -d`). Retrievers index into a collection per
(retriever, language) and reuse it unless `--rebuild` is passed.

```bash
# Iterate on val (default model: the published checkpoint from the Hub)
uv run minicoil eval run minicoil-v2 --dataset mmarco --split val \
  --splits-dir data/eval/splits_mmarco_phase2 \
  --baselines-path data/eval/baselines_mmarco_phase2.json \
  --reports-dir reports/eval_phase2

# A local checkpoint instead
MINICOIL_V2_CHECKPOINT=data/concept_models_phase2/concept_layers.pt MINICOIL_V2_DATA_DIR=data/phase2 \
MINICOIL_V2_LEMMA_MATCH=1 uv run minicoil eval run minicoil-v2 --dataset mmarco --split val --rebuild \
  --splits-dir data/eval/splits_mmarco_phase2 --baselines-path data/eval/baselines_mmarco_phase2.json
```

- **Local checkpoints** do not carry a `config.json`, so set `MINICOIL_V2_LEMMA_MATCH=1`
  to match the published inference config.
- **`--rebuild`** is needed whenever encoding changes, because documents are re-encoded.
- **`--pair eng-spa`** restricts a run to one pair.
- **Reports:** each run writes `<reports-dir>/<timestamp>__<retriever>__<sha>.json` and a
  `.md`. They are stamped with git sha, dirty-tree flag, dataset revision and splits
  manifest sha, so every number traces back to its code and data.

## The sealed test split

The test split refuses to run unless explicitly unlocked, and every access is appended to
`<reports-dir>/audit.log`. Run it once, on the final candidate:

```bash
MINICOIL_EVAL_ALLOW_TEST=1 uv run minicoil eval run minicoil-v2 --dataset mmarco --split test --rebuild \
  --splits-dir data/eval/splits_mmarco_phase2 --baselines-path data/eval/baselines_mmarco_phase2.json \
  --reports-dir reports/eval_phase2
uv run minicoil eval baselines check --report reports/eval_phase2/<report>.json \
  --baselines-path data/eval/baselines_mmarco_phase2.json   # exit 1 if any locked bar is not beaten
```

## Guardrails

- **Frozen splits** with sha256 per file in `manifest.json`. A modified or reshuffled split
  fails to load.
- **Dataset revision pin:** downloads are pinned to one mMARCO revision, and a splits
  manifest built on a different revision is refused.
- **Sealed test:** environment unlock plus an audit log.
- **Hardcoded win conditions:** one metric and one comparator per pair, so a flattering
  metric cannot be cherry-picked. `baselines check` skips pairs the report does not
  contain and bars that are not locked, so run the final test on all four pairs (the
  default) and with the locked baselines file present.
- **Stamped reports** (above).

## Re-locking baselines

`scripts/lock_phase2_baselines.sh` and `scripts/lock_all_queries_baselines.sh` re-run
`bm25`, `minicoil-v1` and `translate-bm25` on val and lock them with
`minicoil eval baselines lock`. `scripts/lock_phase2_baselines_test.sh` does the same for
test. Re-lock only when the splits change; the locked files are the record that results
are compared against.

## Adding a retriever

Drop a module in `src/minicoil_v2/eval/retrievers/` and register it. Modules there are
discovered automatically; `MINICOIL_EVAL_RETRIEVER_MODULES` adds modules from elsewhere.

```python
from minicoil_v2.eval.retriever import BaseRetriever, register

@register("my-retriever")
class MyRetriever(BaseRetriever):
    def index(self, corpus, lang): ...
    def search(self, query, k): ...
```

`BaseRetriever.evaluate` runs the metric loop. `search(query, k)` receives no language,
so a retriever that supports cross-lingual pairs overrides `evaluate` to learn the query
language (as `translate_bm25.py` and `minicoil_v2.py` do). `_common.QdrantSparseRetriever` provides the Qdrant
plumbing.

If you index through FastEmbed or onnxruntime, cap the encode batch size (the harness uses
16). Long documents pad every batch to the longest one, and the default of 256 can hang a
run or exhaust memory.

## Current results

Published checkpoint, covered slice, **test**, MRR@10
(`reports/eval_phase2/2026-07-06T180957Z__minicoil-v2__3056c00.md`):

| Pair | v2 | Bar | Δ |
|---|---|---|---|
| eng-eng | 0.8889 | v1 0.8921 | −0.0032 |
| spa-spa | 0.8398 | bm25 0.7953 | +0.0446 |
| eng-spa | 0.7109 | translate-bm25 0.7028 | +0.0081 |
| spa-eng | 0.7154 | translate-bm25 0.7741 | −0.0588 |

The gate passes on 2 of 4 pairs. Against plain BM25, cross-lingual MRR@10 rises by +0.38
(eng-spa) and +0.46 (spa-eng).

These numbers predate the prefix-leak fix. On val, the fix moved eng-eng 0.8845 → 0.8889
and eng-spa 0.7188 → 0.7304, and left the Spanish-query pairs unchanged
([research log](research-log.md) 1.15). The test split has not been re-run since.

All-queries val numbers are in `reports/eval_all/`. The covered-slice numbers overstate
cross-lingual quality: on uncovered queries v2 trails translate-bm25 by 0.11 (eng-spa) and
0.18 (spa-eng) MRR@10.

`reports/eval_final/` holds the significance test and the recomputed per-query metrics
for an earlier (2026-06-18) candidate.
