# Step 4: Train Per-Concept Heads

**Script:** `src/minicoil_v2/train_concept_layers.py` (batched path: `batched_training.py`)
**Command:** `minicoil train`
**Settings class:** `TrainSettings`

## Goal

Train one head per concept, `tanh(Linear(384, 8, bias=False))`, with a bilingual 4-term
cosine triplet loss. A head maps the token-pooled mE5-small vector of a concept word
(384-D) to an 8-D sense vector (`OUTPUT_DIM = 8`). Heads are independent: there are no
cross-concept parameters at training or inference time.

Two encoders play different roles:

| Role | Model | Where it runs | What it does |
|---|---|---|---|
| **Input encoder** | `intfloat/multilingual-e5-small` (384-D) | Training and inference | Produces the vector the head consumes: the last hidden states of the concept word's subword tokens, averaged and L2-normalized |
| **Teacher** (mining encoder) | Set at embed time; the published model used `Qwen/Qwen3-Embedding-0.6B` (1024-D, sentence pooling) | Embed step only | Its stored vectors build the distance matrix that picks positives and negatives. The head never sees it |

## Command

```bash
# Smoke test (10 concepts)
uv run minicoil train --concepts 0:10 --no-wandb

# Published recipe (see "Reproducing the published checkpoint" in the README)
uv run minicoil train --data-dir data/phase2 --collection-name minicoil_sentences_phase2 \
  --encode-cache-dir data/enc_cache_phase2 --epochs 80 --train-epoch-size 2000 \
  --margin-scale 2.0 --lr-patience 3 --dropout 0.10 --batched --no-wandb

# Resume after interruption
uv run minicoil train --resume --no-wandb
```

The recipe above is the one recorded for the published heads; the exact arguments of each
run are in its W&B metadata. The code defaults (below) are more conservative and much
slower.

## Inputs

| Input | Source |
|-------|--------|
| `<data-dir>/concept_vocabulary.json`, `word_to_concept.json` | Step 3, or the demand-selected subset |
| Qdrant collection | Step 2b: sentences, language, concept ids and the teacher vector |
| `intfloat/multilingual-e5-small` | Hugging Face, loaded as a plain transformer for token pooling |

## Outputs

| File | Description |
|------|-------------|
| `<output-dir>/checkpoint_batch_XXXX.pt` | One checkpoint per concept batch |
| `<output-dir>/concept_layers.pt` | Final merged model (all concepts) |

`concept_layers.pt` is a dict of `{concept_id: {"weight": Tensor[8, 384]}}`. `minicoil
export-hf` stacks it into the published `model.safetensors`.

## Pipeline

For each batch of concepts (`--concept-batch-size`, default 50):

1. **Scroll from Qdrant.** For each concept, fetch up to `max_sentences` per language,
   balanced by `lang_ratio` (0.5 = equal EN/ES). Each point brings its sentence, language
   and teacher vector.
2. **Deduplicate** on `(sentence, concept, language)`. The same sentence pooled for two
   concepts gives two different input vectors, so the concept is part of the key.
3. **Encode the input.** Every row is re-encoded as token-pooled mE5-small with the
   `"passage: "` prefix, through the same code path as inference
   (`encode_inputs_token_pooled`, matched by `concept_match.match_concepts_after_prefix`).
   Rows whose concept word cannot be located are dropped. Tests pin this input to the
   inference encoding of a document (`"passage: "`, surface-form matches) byte for byte.
   Matching here is surface-form only: the lemma fallback is an inference-time feature.
4. **Trim augmentation** (`trim_augment_ratio`, default 1.0). Each sentence also gets a
   copy cut to `trim_window` words on each side of the concept word, re-encoded with
   token pooling. The copy reuses the original sentence's teacher vector.
5. **Cache** (optional, `--encode-cache-dir`). Steps 1–4 do not depend on the training
   recipe, so they are cached per batch. A tuning loop encodes once, and recipe
   comparisons are not confounded by re-sampled rows. The Phase-2 cache was 25 GB for
   2,398 concepts.
6. **Distance matrix.** `1 − cosine` between the concept's teacher vectors. Rows are
   interleaved by language, then split 80/20 into train and validation ranges.
7. **Sample and train.** Each epoch draws fresh bilingual 5-point samples with hard
   mining (next section) and minimizes the 4-term loss with Adam
   (`lr = 2e-3`) and `ReduceLROnPlateau` on validation loss. Forward pass:
   `tanh(W · dropout(x))`.
8. **Checkpoint** the batch, and merge all batches at the end.

Concepts with fewer than `MIN_SENTENCES_PER_CONCEPT` (10) usable rows are skipped.

## Sampling and loss

Full design: [05-four-quadrant-contrastive.md](05-four-quadrant-contrastive.md). In short,
each sample is `(anchor, sl_pos, sl_neg, xl_pos, xl_neg)`:

- **Positive:** a random pick among the 20 candidates nearest to the anchor in teacher
  space (fewer for small concepts: at most a third of the candidates).
- **Negative:** the nearest candidate outside that pool whose distance exceeds the
  positive's by at least the margin floor (semi-hard).
- **Margin floor:** `max(min_triplet_margin, margin_scale × std(concept distances))`.
  With `margin_scale = 0` it is the fixed 0.1.
- **Loss margin:** the per-sample gap `d(a, neg) − d(a, pos)` measured by the teacher.
- **Cross-language pair:** drawn the same way from the other language. If none
  qualifies, the sample keeps only its same-language triplet (masked by `has_xl`).

```
loss = relu(d(a, sl_pos) − d(a, sl_neg) + margin_sl)
     + relu(d(a, xl_pos) − d(a, xl_neg) + margin_xl) · has_xl
```

`d` is cosine distance on the 8-D head output.

## Batched trainer

`--batched` trains a whole concept batch at once: stacked weights `[B, 8, 384]`, one
`bmm` per step, a summed per-concept loss, one backward pass and a vectorized Adam step
with per-concept learning-rate scheduling. Because the heads are independent, this equals
`B` separate runs up to sampling and dropout randomness. The equivalence checks,
profiler and benchmarks are in [`scripts/perf/`](../scripts/perf/README.md); the report
is [research/batched-gpu-training.md](research/batched-gpu-training.md). On a real
50-concept bucket it is about 2x faster than the reference loop, because per-epoch
sampling, not the GPU kernel, dominates.

`--mine-every N` (batched only) reuses each drawn epoch for N epochs. It changes training
dynamics, so check it against the eval gate, not the equivalence tests.

## Hyperparameters

### Training

| Parameter | CLI flag | Default | Published run | Effect |
|-----------|----------|---------|---------------|--------|
| `epochs` | `--epochs` | `500` | `80` | Epochs per concept |
| `lr` | `--lr` | `2e-3` | `2e-3` | Adam learning rate |
| `min_triplet_margin` | `--min-triplet-margin` | `0.1` | `0.1` | Absolute margin floor |
| `margin_scale` | `--margin-scale` | `0.0` | `2.0` | Scales the floor by each concept's distance spread. A larger floor selects *easier* negatives |
| `train_epoch_size` | `--train-epoch-size` | `64,000` | `2,000` | 5-point samples per training epoch |
| `val_epoch_size` | `--val-epoch-size` | `6,400` | `6,400` | Samples per validation epoch (only drives the scheduler; a warning fires if it exceeds the train size) |
| `sample_batch_size` | `--sample-batch-size` | `256` | `256` | Mini-batch size |
| `dropout` | `--dropout` | `0.05` | `0.10` | Dropout on the input vector |
| `val_size` | `--val-size` | `0.2` | `0.2` | Validation share of each concept's rows |
| `lr_factor` | `--lr-factor` | `0.5` | `0.5` | Scheduler factor |
| `lr_patience` | `--lr-patience` | `5` | `3` | Scheduler patience (epochs) |
| `batched` | `--batched` | off | on | Batched multi-concept trainer |
| `mine_every` | `--mine-every` | `1` | `1` | Re-mine every N epochs (batched only) |

### Data

| Parameter | CLI flag | Default | Effect |
|-----------|----------|---------|--------|
| `data_dir` | `--data-dir` | `data` | Vocabulary to train |
| `max_sentences` | `--max-sentences` | `2000` | Max rows per concept per language |
| `lang_ratio` | `--lang-ratio` | `0.5` | Target EN share |
| `concepts` | `--concepts` | all | Concept-number range (`0:100` = `C-00000` to `C-00099`, whichever exist in the vocabulary) or a comma-separated id list (`C-00042,C-01647`) |
| `concept_batch_size` | `--concept-batch-size` | `50` | Concepts per batch (and per checkpoint) |
| `encode_batch_size` | `--encode-batch-size` | `64` | Sentences per mE5 forward |
| `encode_cache_dir` | `--encode-cache-dir` | none | Cache for steps 1–4 |
| `rematch_buckets` | `--rematch-buckets` | none | Phase 0/1 only: source rows from a pre-matched bucket file when a collection's concept ids come from a different vocabulary (produced by `research/scan_viable_coverage.py`) |

### Augmentation

| Parameter | CLI flag | Default | Effect |
|-----------|----------|---------|--------|
| `trim_augment_ratio` | `--trim-augment-ratio` | `1.0` | Share of rows that get a trimmed copy |
| `trim_window` | `--trim-window` | `5` | Words kept on each side of the concept word |

### Infrastructure

| Parameter | CLI flag | Default | Effect |
|-----------|----------|---------|--------|
| `device` | `--device` | `auto` | `auto` (MPS > CUDA > CPU), `mps`, `cuda`, `cpu` |
| `resume` | `--resume` | off | Skip concepts already in checkpoints |
| `wandb` | `--wandb/--no-wandb` | on | W&B logging (`--wandb-project`, `--wandb-name`) |
| `collection_name` | `--collection-name` | `minicoil_sentences` | Qdrant collection |
| `qdrant_url`, `qdrant_api_key` | `--qdrant-url`, `--qdrant-api-key` | `MINICOIL_QDRANT_*` env / `.env`, else local | Qdrant connection |

## Resume behavior

With `--resume`, the trainer scans the output directory for checkpoints, skips concepts
already trained, and merges everything at the end.

> **Known issue:** checkpoint files are numbered from `checkpoint_batch_0000.pt` again on
> every run, so a resumed run overwrites the earlier run's files and the final merge loses
> those concepts. Until this is fixed, resume into a fresh `--output-dir` and merge the
> checkpoint dicts yourself, or copy the earlier checkpoints aside first.

## Time and resources

Measured for the published model (2,398 concepts):

- Training took about 5.5 h on an Apple-silicon laptop with the recipe above, split across
  parallel processes and merged.
- Building the cold input-encode cache took 1.5–2 h; the cache occupies 25 GB.

Cost grows roughly linearly with the number of concepts. See
[research/full-vocabulary-run-plan.md](research/full-vocabulary-run-plan.md) for
projections to the full 12,357-concept vocabulary.
