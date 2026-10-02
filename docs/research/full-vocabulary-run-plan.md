# Full-vocabulary run: plan and lessons

> Planning note (2026-08), not yet executed. It describes how to train heads for all
> 12,357 pruned concepts (the published model trains 2,398) and evaluate them against the
> locked mMARCO gate. Estimates come from the Phase-2 run and carry roughly ±2x error.

## Goal

Train every concept in the pruned vocabulary, then check whether the extra coverage
closes the cross-lingual gap. At the time of writing es→en trailed translate-bm25 by
−0.0588 MRR@10 on the covered-slice test split. The vocabulary is already built and
pruned, so the work is embed → train → eval.

## Stage 1: embed (teacher vectors for all concepts)

The Phase-2 command, pointed at a new collection:

```bash
minicoil embed --data-dir data --max-articles 400000 --store-cap 800 --lang both \
  --mining-encoder Qwen/Qwen3-Embedding-0.6B --mining-pooling sentence \
  --encode-batch-size 64 --collection-name minicoil_sentences_full --recreate
```

Decisions:

- **Teacher.** Use Qwen3-Embedding-0.6B to reproduce the published recipe. The code
  default (mE5-small) is about 5x cheaper, but it is a different recipe and will change
  results. The teacher only builds the distance matrix, so its dimension is free.
- **`--max-articles`.** Phase 2 filled its 2,398 concepts to about 775 of 800 sentences
  per language from 400k articles. The extra ~10k concepts are rarer, so 400k articles will
  not fill them. Check their frequency distribution first, then either raise
  `--max-articles` or accept partial fill (the floor is `MIN_SENTENCES_PER_CONCEPT = 10`).
- **Collection.** Write a new collection; do not overwrite the Phase-2 one.

Known issues:

- Qwen3's native `max_seq_length` is 32k, and length-sorted batching can run the GPU out
  of memory on long sentences. The 512 cap in sentence mode is already in the code.
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` also helped.
- The embed has no resume logic. A failed run restarts from scratch.

Estimate: Phase 2 took about 5.5 h for 2,398 concepts on one A10G-class GPU (370–505
sentences/s). The full vocabulary has about 5x the concepts plus a long rare tail: plan for
10–20 h on a faster consumer GPU.

## Stage 2: train

The published recipe, with no `--concepts` flag, so every concept trains:

```bash
minicoil train --data-dir data --collection-name minicoil_sentences_full \
  --encode-cache-dir data/enc_cache_full --epochs 80 --train-epoch-size 2000 \
  --margin-scale 2.0 --lr-patience 3 --dropout 0.10 --batched --device cuda
```

Training does two encode passes, and they drive the cost:

1. **Pass 1** is the embed above: teacher vectors stored in Qdrant.
2. **Pass 2** happens inside training: every sentence is re-encoded live as token-pooled
   mE5-small (384-D), because the heads consume the inference representation, not the
   stored teacher vector. `--encode-cache-dir` caches it. The Phase-2 cache was 25 GB for
   2,398 concepts, so expect about 125 GB for the full vocabulary. The cold build is part
   of the first run's wall clock.

Estimate: about 5.5 h of training per 2,398 concepts on a laptop GPU, roughly linear in
concept count, so about 28 h laptop-class and less on CUDA. Calibrate early on a small
`--concepts` range; it cold-builds the cache, so it measures pass 2 as well.

Before trusting a fresh build, run the test suite and the batched-trainer equivalence
checks in `scripts/perf/`.

## Stage 3: evaluate

Every system must use the same corpus. Three options:

| Corpus | Status | Notes |
|---|---|---|
| Gold pool (dev.small, about 7.4k docs per language) | Locked baselines ready | Primary comparison; about 30 min for v2 |
| Open domain, N=1M distractors (`mmarco_open`) | Baselines must be re-run at N=1M | Set `MINICOIL_MMARCO_OPEN_N_EN` / `_N_ES`. Validity gate: at N=0, es→en val MRR@10 = 0.70766 |
| Full 8.8M collection | Not supported yet | Needs a sharded, parallel indexer; multi-day per system otherwise |

Splits: reuse the locked Phase-2 splits for an apples-to-apples comparison. Rebuilding them
on the larger coverage means re-running every baseline. The locked splits may also
understate a larger model's advantage, because they only hold queries the 2,398 concepts
cover.

Eval encoding is CPU-bound (measured 119 docs/s on CPU against 62 on Apple MPS), so run
indexing on CPU. A GPU does not help; core count does, once indexing is sharded.

## Lessons to carry over

- **Eval batch sizes** are fixed in `eval/retrievers/_common.py` (encode 16, upload 512).
  Larger encode batches pad every batch to the longest document and can exhaust memory.
- **Sparse layout:** concept indices are `concept_num * OUTPUT_DIM + offset` with
  `OUTPUT_DIM = 8`; backbone indices start at `1 << 27`.
- **Query stopword fix:** `MINICOIL_V2_BACKBONE_STOPWORD_FIX` defaults to on. Keep it on.
- **Teacher and input dimensions differ by design:** the teacher is any dimension and
  supervision only; the input is mE5-small, 384-D.
- **Qdrant must be running** for both embed (writes) and eval (index and search). It can
  be killed by the OOM killer on low-RAM machines.
- **The test split is sealed** (`MINICOIL_EVAL_ALLOW_TEST=1`) and access is logged.
  Iterate on val.
- **Hardware sizing:** budget at least 96 GB RAM and 500 GB of disk for the cache plus the
  collection.
