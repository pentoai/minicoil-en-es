# Step 2: Wikipedia Sentence Extraction (Scan + Embed)

Wikipedia extraction is split into two phases around the prune step:

| Phase | Script | Command | When |
|-------|--------|---------|------|
| **2a. Scan** | `scan_wiki_sentences.py` | `minicoil scan` | Before prune |
| **2b. Embed** | `embed_wiki_sentences.py` | `minicoil embed` | After prune |

The scan phase produces the per-concept counts used by prune. The embed phase
runs against the already-pruned vocabulary, so Qdrant is populated with the
final concept IDs and no remapping is needed afterwards. This keeps encoder
compute bounded to the surviving vocabulary and eliminates any old/new ID
mismatch between Qdrant and the training stage.

---

## Phase 2a: Scan (count only)

**Script:** `src/minicoil_v2/scan_wiki_sentences.py`
**Command:** `minicoil scan`
**Settings class:** `ScanSettings`

### Goal

Stream Wikipedia, tokenize each sentence, and produce uncapped per-concept
occurrence counts per language. Pure CPU: no encoder, no Qdrant.

Step 1 outputs (`concept_vocabulary.json`, `word_to_concept.json`) must exist.

## Inputs

| Input | Source |
|-------|--------|
| `data/word_to_concept.json` | Step 1 |
| `data/concept_vocabulary.json` | Step 1 (size stats only) |
| Wikipedia dumps | HuggingFace `wikimedia/wikipedia` (streamed) |

### Outputs

| Output | Description |
|--------|-------------|
| `data/concept_sentence_counts.json` | Uncapped per-concept counts per language (`config` and `counts`); coverage stats are logged |

### Command

```bash
# Smoke test
uv run minicoil scan --max-articles 1000

# Full scan
uv run minicoil scan --max-articles 100000
```

### Algorithm

1. Load `word_to_concept.json` (token -> concept ID per language).
2. Stream Wikipedia per language.
3. For each article: split into sentences, drop those outside `[MIN_SENTENCE_WORDS, MAX_SENTENCE_WORDS]`.
4. Tokenize each sentence to a set of lowercase alpha words, intersect with the vocabulary, map matches to concepts, increment counts.
5. Write `concept_sentence_counts.json`.

### Hyperparameters

| Parameter | CLI flag | Default | Effect |
|-----------|----------|---------|--------|
| `max_articles` | `--max-articles` | `100,000` | Articles per language |
| `lang` | `--lang` | `both` | `en`, `es`, or `both` |

---

## Phase 2b: Embed (encode + upsert)

**Script:** `src/minicoil_v2/embed_wiki_sentences.py`
**Command:** `minicoil embed`
**Settings class:** `EmbedSettings`

### Goal

Stream Wikipedia again, this time encoding sentences for surviving concepts
with the teacher (mining encoder) and upserting to Qdrant. The stored vectors are
only used at training time, to rank positives and negatives; the heads never see them
(their input is re-encoded with mE5-small, see [04-training.md](04-training.md)).

### Prerequisites

- Qdrant must be running: `docker compose up -d`.
- Prune must have been run. `concept_vocabulary.json` and `word_to_concept.json` contain final contiguous IDs.

### Inputs

Without `--cloud-inference`, embed connects to `--qdrant-url` (default `localhost:6333`) with no API key. A Qdrant Cloud cluster is reached only with `--cloud-inference`, which reads `MINICOIL_QDRANT_URL` / `MINICOIL_QDRANT_API_KEY` (or `.env`) and has the cluster compute the teacher vectors server-side (sentence pooling only).

| Input | Source |
|-------|--------|
| `data/concept_vocabulary.json` | Step 3 (post-prune) |
| `data/word_to_concept.json` | Step 3 (post-prune) |
| Wikipedia dumps | HuggingFace `wikimedia/wikipedia` (streamed) |
| Teacher (mining encoder) | `--mining-encoder` and `--mining-pooling`. Published checkpoint: `Qwen/Qwen3-Embedding-0.6B`, `sentence`, 1024-D. Phase 0/1: `openrouter/qwen/qwen3-embedding-8b`, 4096-D, via Qdrant cloud inference. Code default: mE5-small, `token_pooled`, 384-D |

### Outputs

| Output | Description |
|--------|-------------|
| Qdrant collection `minicoil_sentences` | Sentences + mining embeddings + concept metadata |

### Command

```bash
# Smoke test (code-default teacher, mE5-small token-pooled)
uv run minicoil embed --max-articles 1000

# Published checkpoint's run (demand-selected vocabulary, Qwen3 teacher)
uv run minicoil embed --data-dir data/phase2 --max-articles 400000 --store-cap 800 --lang both \
  --mining-encoder Qwen/Qwen3-Embedding-0.6B --mining-pooling sentence \
  --encode-batch-size 64 --collection-name minicoil_sentences_phase2 --recreate

# Qdrant Cloud inference
uv run minicoil embed --max-articles 100000 --store-cap 4000 --mining-pooling sentence --cloud-inference
```

### Algorithm

1. Load the pruned `word_to_concept.json`.
2. Initialize the mining encoder on best available device (MPS > CUDA > CPU). The collection's vector size is the encoder's output size (`get_sentence_embedding_dimension()` in sentence mode, the transformer's `hidden_size` in token-pooled mode).
3. Create the Qdrant collection with named vector `mining` (cosine) and payload indexes on `concept_ids` and `lang`. Use `--recreate` to drop and rebuild after changing `--mining-encoder` (dimension must match).
4. Stream Wikipedia, apply the same sentence-length filter as scan.
5. For each sentence, gate storage through three caps:

| Filter | Value | Purpose |
|--------|-------|---------|
| Sentence length | 5-100 words | Exclude fragments and full paragraphs |
| Store cap | Configurable (default 100) | Max stored sentences per concept per language |
| Per-article cap | 2 per concept | Reduce topical skew |

6. Each stored row is one `(sentence, focal word, language)` triple: a sentence with two
   matched concept words gives two rows. Rows are buffered and, when the buffer reaches
   `flush_buffer_size`, encoded in one batch and upserted in chunks of `upsert_batch_size`.
7. **Teacher vector.** With `--mining-pooling token_pooled` (code default), it is the
   teacher's last hidden states averaged over the subword tokens of the focal word's
   first occurrence in the `"passage: "`-prefixed sentence, L2-normalized; rows whose
   focal word cannot be located are dropped. Known issue: the occurrence is found by
   substring search, so a focal word can match inside the prefix ("age" in "passage") or
   inside a longer word. This only affects the token-pooled teacher, not the heads'
   inputs, which use the shared concept matcher. With
   `--mining-pooling sentence` (published checkpoint), it is the normalized sentence
   embedding, shared by all focal words of the sentence. Sentence mode caps the teacher's
   `max_seq_length` at 512 tokens: Qwen3's native 32k, combined with length-sorted
   batching, ran the GPU out of memory.
8. Point ids are a deterministic UUID of `(sentence, focal word, language)`, so re-runs
   overwrite rather than duplicate.

### Qdrant Point Schema

```json
{
  "id": "deterministic-uuid",
  "vector": {"mining": [0.123, -0.456, ...]},
  "payload": {
    "sentence": "The cat sat on the mat",
    "focal_word": "cat",
    "lang": "en",
    "concept_ids": ["C-01647"],
    "article_id": "12345"
  }
}
```

### Hyperparameters

| Parameter | CLI flag | Default | Effect |
|-----------|----------|---------|--------|
| `max_articles` | `--max-articles` | `100,000` | Articles per language |
| `store_cap` | `--store-cap` | `100` | Max stored sentences per concept per language |
| `lang` | `--lang` | `both` | `en`, `es`, or `both` |
| `mining_encoder` | `--mining-encoder` | `intfloat/multilingual-e5-small` | Encoder for mining embeddings in Qdrant (published checkpoint: `Qwen/Qwen3-Embedding-0.6B`) |
| `mining_pooling` | `--mining-pooling` | `token_pooled` | `token_pooled` (focal-word subwords) or `sentence` (published checkpoint) |
| `encode_batch_size` | `--encode-batch-size` | `64` | Sentences per encoder call |
| `flush_buffer_size` | `--flush-buffer-size` | `512` | Buffer size triggering encode+upsert |
| `upsert_batch_size` | `--upsert-batch-size` | `500` | Points per Qdrant upsert call |
| `upsert_wait` | `--upsert-wait/--no-upsert-wait` | `False` | Wait for Qdrant ack per batch |
| `device` | `--device` | `auto` | `auto`, `mps`, `cuda`, `cpu` |
| `recreate` | `--recreate` | `False` | Drop and recreate collection |
| `qdrant_url` | `--qdrant-url` | `http://localhost:6333` | Qdrant server URL |
| `qdrant_api_key` | `--qdrant-api-key` | `None` | Qdrant API key (used with `--cloud-inference` only) |
| `collection_name` | `--collection-name` | `minicoil_sentences` | Collection name |
| `cloud_inference` | `--cloud-inference/--no-cloud-inference` | `False` | Encode in Qdrant Cloud instead of locally |

**Local (default):** connects to `localhost:6333`, encodes sentences with the mining encoder locally.

**Qdrant Cloud inference:** pass `--cloud-inference` to skip local encoding and have Qdrant Cloud generate embeddings server-side. The cluster URL and API key are read from `.env` (`MINICOIL_QDRANT_URL`, `MINICOIL_QDRANT_API_KEY`) unless overridden with `--qdrant-url` / `--qdrant-api-key`.

### Performance Notes

Progress is logged every 500 articles during warm-up, every 5000 after. Each log line shows time breakdown (`scan=`, `encode=`, `upsert=`) so you can identify the bottleneck.

Typical profile on MPS: encoding dominates (~90% of wall time), Qdrant writes are ~6% with default settings (`flush_buffer_size=512`, `wait=False`).

The published checkpoint's embed (Qwen3-0.6B, 400k articles per language, store cap 800)
was too slow on a laptop (14.5 sentences/s, 38–77 h projected) and ran on a single
A10G-class cloud GPU at 340–505 sentences/s, about 5.5 h. The embed has no resume logic.

---

## Why Split Scan and Embed?

The encoder is the pipeline bottleneck. Running it before prune means encoding
embeddings for ~75% of concepts that will be discarded anyway. Splitting the
stages keeps the encoder bounded to the pruned vocabulary.

It also removes an ID-consistency hazard: with the single-stage pipeline,
Qdrant stored pre-prune IDs in its payload while the trainer expected
post-prune IDs. Splitting the stages means Qdrant is only ever written with
final IDs, so no mapping file or remap pass is needed.

The only new cost is scanning Wikipedia twice. Scan is CPU-only and processes
articles at tens of thousands per second, so its cost is dwarfed by the
encoder time it saves.
