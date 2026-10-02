# research/

One-off exploration scripts. None of them is needed to train or use the published
model; they are kept because they produced findings the design rests on. Each script's
docstring says what it measures and how to run it. Scripts here may target older data
layouts or collections, so expect to adapt paths and flags before re-running.

| Script | Question it answered | Write-up |
|---|---|---|
| `compare_resolution_schedules.py` | Exponential vs linear Louvain resolution schedule for recursive splitting | [docs/01](../docs/01-concept-vocabulary.md), [research log](../docs/research-log.md) 1.0 |
| `compare_vocab_quality.py` | Connected components + salvage vs Louvain partitions (pair recall, orphans, fragments) | [docs/01](../docs/01-concept-vocabulary.md) |
| `20260519_qwen8b_cost.py` | Cost of embedding the corpus with a hosted Qwen3-8B teacher | [research log](../docs/research-log.md) 1.7 |
| `20260525_cluster_metrics.py` | Do concepts have distinct sense sub-clusters in teacher space (k-means silhouette)? | — |
| `20260525_umap_clusters.py` | UMAP plots of the same clusters (needs `umap-learn` and `matplotlib`) | — |
| `scan_viable_coverage.py` | Phase 0/1: which concepts a cloud collection could supply; writes the bucket file `minicoil train --rematch-buckets` reads | [research log](../docs/research-log.md) 1.7–1.8 |

## diagnostics/

Frozen copies of the May 2026 diagnostics behind the pooling and polysemy findings,
written against the trainer and collections of that time (4D heads, 384-D teacher
vectors). They are auto-formatted but otherwise unchanged, and are not
maintained.

| Script | Used in |
|---|---|
| `diagnose_mining_vectors.py`, `diagnose_mining_with_e5_large.py` | [Pooling strategy](../docs/research/pooling-strategy.md): sentence-pool baseline, encoder-size control |
| `diagnose_word_token_pooling.py`, `validate_token_pooling.py` | Pooling strategy: token-pool gap and the different-surface-word control |
| `test_more_data_hypothesis.py` | Pooling strategy: 10x data per concept |
| `diagnose_triplet_quality.py` | [Pooling verification](../docs/research/pooling-verification.md): per-concept triplet audit |
| `build_full_pool_collection.py`, `validate_4d_polysemy.py`, `baseline_384d_per_concept.py` | [4D polysemy validation](../docs/research/4d-polysemy-validation.md) |
