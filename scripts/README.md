# scripts/

Helpers that reproduce the published model and its evaluation. Run them from the repo
root with `uv run`.

| Script | Purpose | Docs |
|---|---|---|
| `phase2_concept_demand.py` | Rank concepts by how often they fire on mMARCO query content words (80% qid slice); writes `data/eval/mmarco/phase2_concepts.json`, the 2,398-concept selection | [03-vocabulary-pruning.md](../docs/03-vocabulary-pruning.md) |
| `phase2_make_datadir.py` | Filter the pruned vocabulary to that selection into `data/phase2/` (ids not renumbered) | same |
| `lock_phase2_baselines.sh` | Run and lock `bm25`, `minicoil-v1` and `translate-bm25` on the covered-slice val split | [07-evaluation.md](../docs/07-evaluation.md) |
| `lock_phase2_baselines_test.sh` | Same, on the sealed test split | same |
| `lock_all_queries_baselines.sh` | Same, on the all-queries val split | same |
| `eval/published_model_retrieval_check.py <cache-dir>` | End-to-end check of the published model from a clean download (needs Qdrant): index a small bilingual corpus in Qdrant with `Modifier.IDF` and query across languages | [06-inference-and-scoring.md](../docs/06-inference-and-scoring.md) |
| `perf/` | Batched-trainer equivalence checks, profiler and benchmarks | [perf/README.md](perf/README.md) |

The published model already ships the selected vocabulary (`concept_vocabulary.json`,
`word_to_concept.json`), so the two `phase2_*` scripts are only needed to redo the
selection, for example on a different vocabulary or query set.
