# Step 3: Prune the Concept Vocabulary

**Script:** `src/minicoil_v2/prune_concept_vocab.py`
**Command:** `minicoil vocab prune`
**Settings class:** `VocabPruneSettings`

## Goal

Keep only concepts with sufficient training evidence in **both** English and
Spanish. Renumber surviving concept IDs contiguously (largest first).

## Why This Step Exists

Step 1 produces ~79k concepts. Many have few or no matching sentences in Wikipedia.
Training on sparse data produces noisy weights. Pruning ensures every surviving
concept has enough bilingual data to learn a meaningful projection.

## Command

```bash
# Recommended
uv run minicoil vocab prune --threshold 50

# Smoke test (keeps almost everything)
uv run minicoil vocab prune --threshold 1
```

## Inputs

| File | Source |
|------|--------|
| `data/concept_vocabulary.json` | Step 1 |
| `data/concept_sentence_counts.json` | Step 2a (`minicoil scan`) |

## Outputs

| File | Description |
|------|-------------|
| `data/concept_vocabulary.json` | Pruned vocabulary (overwrites original) |
| `data/word_to_concept.json` | Pruned lookup (overwrites original) |
| `data/concept_vocabulary_full.json` | Backup of pre-pruned vocabulary |
| `data/word_to_concept_full.json` | Backup of pre-pruned lookup |

## Algorithm

1. **Load** concept vocabulary and sentence counts from extraction.

2. **Threshold filter.** A concept survives only if `min(en_count, es_count) >= threshold`. Both languages must independently meet the bar.

3. **Renumber contiguously.** Surviving concepts are sorted by total word count (`len(en) + len(es)`) descending, then assigned new IDs starting at `C-00000`. Ties broken by old ID for determinism.

4. **Rebuild** `word_to_concept.json` from the pruned vocabulary with new IDs.

5. **Save.** Originals backed up as `*_full.json` (first prune only), then overwritten.

## Why Contiguous IDs

Each concept's sparse slots are `concept_number * OUTPUT_DIM + offset` (`OUTPUT_DIM = 8`), so the concept index range is `max_concept_number * 8`. Gaps in IDs would waste sparse dimensions on unused slots.

The demand-selected training subset (below) does **not** renumber: it keeps the pruned IDs, so the trained concepts are not contiguous. That is harmless, because only fired concepts are emitted and Qdrant stores sparse vectors.

## Hyperparameters

| Parameter | CLI flag | Default | Effect |
|-----------|----------|---------|--------|
| `threshold` | `--threshold` | `50` | Min sentences in both EN and ES |
| `data_dir` | `--data-dir` | `data` | Directory with vocab and count files |

**Guidance:** `50` on a 100k-article scan reduces ~79k to ~19.5k concepts. `100` is more aggressive (fewer, better-trained concepts). `1` keeps every concept seen at least once in both languages.

## Typical Output

With `threshold=50` and 100k articles per language:

| Metric | Value |
|--------|-------|
| Original concepts | ~79.6k |
| Surviving concepts | ~19,485 (24.5%) |
| EN words kept | ~30,600 |
| ES words kept | ~34,100 |

The published model's vocabulary was pruned at `threshold=1` over a smaller scan than the
one above: 79,652 → 12,357 concepts (21,761 EN / 25,453 ES words).

## Selecting the concepts to train (published model)

Pruning decides which concepts *can* be trained; it does not decide which ones *should*
be. For the published model, the 12,357 pruned concepts were narrowed by query demand
rather than by Wikipedia supply:

1. `scripts/phase2_concept_demand.py` counts how often each concept fires on the content
   words of mMARCO dev.small queries (EN and ES), selecting on an 80% qid slice and
   reserving 20% for the test carve (see the caveat in
   [07-evaluation.md](07-evaluation.md): the carve did not end up restricted to it). It
   writes the demand-ranked list to the 90%
   coverage mark: **2,398 concepts**.
2. `scripts/phase2_make_datadir.py` filters `concept_vocabulary.json` and
   `word_to_concept.json` to that list into `data/phase2/`, keeping the pruned IDs.
   `minicoil embed --data-dir data/phase2` then stores sentences only for those concepts.

The selected vocabulary ships with the published model (`concept_vocabulary.json` and
`word_to_concept.json` in the Hugging Face repo), so it can be reused as a `--data-dir`
without re-running the selection. The demand table and the reasoning are in
[research/phase2-scale-concepts-design.md](research/phase2-scale-concepts-design.md).
`minicoil vocab subset` does the same filtering for any hand-picked concept list.

## Important Notes

- Pruning overwrites the original vocabulary files. Backups are automatic on first run.
- Running prune again operates on the already-pruned vocabulary. Restore from `*_full.json` to re-prune from scratch.
- Qdrant is populated **after** pruning (`minicoil embed`), so its `concept_ids` payload uses the final contiguous IDs. No ID mapping is required.
