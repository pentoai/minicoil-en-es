# miniCOIL EN-ES: English-Spanish Sparse Retrieval

miniCOIL EN-ES is a bilingual sparse retrieval model for English and Spanish: BM25 with a
learned semantic overlay, extended across languages. It builds on
[miniCOIL v1](https://qdrant.tech/articles/minicoil/) (English-only, one small head per
word), with three changes:

- **Concepts instead of words.** Each head belongs to a **bilingual concept**, a cluster
  of translations built from the MUSE dictionaries, such as `{dog, dogs}` /
  `{perro, perros, perrita}`.
- **One block per concept, in both languages.** A word in a concept becomes a block of 8
  values from that concept's head. English and Spanish words of the same concept write to
  the same block, so `dog` and `perro` match with no translation step.
- **BM25 for everything else.** Every other token stays a plain BM25 term.

The output is an ordinary sparse vector for any inverted index; in Qdrant it is scored
with `Modifier.IDF`.

The published model is [`Jocana/minicoil-en-es`](https://huggingface.co/Jocana/minicoil-en-es):
2,398 concepts, `Linear(384 → 8) + tanh` heads over `multilingual-e5-small`. The Python
package is `minicoil_v2`.

## Results

mMARCO test, MRR@10, concept-covered slice (queries that share a concept with their gold
passage):

| Pair | miniCOIL EN-ES | Bar to beat | Δ vs bar | Δ vs BM25 |
|---|---|---|---|---|
| en → en | 0.8889 | miniCOIL v1 0.8921 | −0.0032 | +0.0084 |
| es → es | 0.8398 | BM25 0.7953 | +0.0446 | +0.0446 |
| en → es | 0.7109 | translate-bm25 0.7028 | +0.0081 | +0.3833 |
| es → en | 0.7154 | translate-bm25 0.7741 | −0.0588 | +0.4625 |

`translate-bm25` translates the query with NLLB-200 (600M) and then runs BM25.

- **All queries:** on the full query set, which includes queries with no shared concept,
  the cross-lingual numbers are lower. Those are the coverage-cliff queries;
  see [docs/07-evaluation.md](docs/07-evaluation.md).
- **Prefix-leak fix:** these numbers predate the fix, which raised en → en and en → es on
  val. The test split has not been re-run since.

## Use the published model

```bash
pip install torch transformers fastembed safetensors mmh3 simplemma huggingface-hub qdrant-client
```

```python
import sys

from huggingface_hub import snapshot_download

model_dir = snapshot_download("Jocana/minicoil-en-es")  # ships its own encoder code
sys.path.insert(0, model_dir)

from minicoil_v2.encoder import MiniCoilEncoder

encoder = MiniCoilEncoder.from_pretrained(model_dir)

doc = encoder.encode_sparse("El perro corrió por el parque", lang="es")
query = encoder.encode_sparse("dog running", lang="en", is_query=True)
```

- **Languages:** encode documents in the corpus language and queries in the query
  language.
- **Queries:** always pass `is_query=True` for queries.
- **Inside this repo:** `from minicoil_v2.encoder import MiniCoilEncoder` works directly
  after `uv sync`, and `MiniCoilEncoder.from_pretrained("Jocana/minicoil-en-es")`
  downloads the model.

The index must apply IDF, because the vectors carry term weights only:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient("http://localhost:6333")
client.create_collection(
    "docs",
    vectors_config={},
    sparse_vectors_config={"minicoil": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)
```

`scripts/eval/published_model_retrieval_check.py <cache-dir>` is a complete cross-lingual
example (it needs a running Qdrant).
Encoding, scoring and limitations are described in
[docs/06-inference-and-scoring.md](docs/06-inference-and-scoring.md).

## How it works

```text
MUSE EN-ES dictionaries ─► 1. vocab build ─► 2a. scan Wikipedia (counts)
                                                   │
                                              3. vocab prune  (+ demand-driven selection)
                                                   │
                                              2b. embed Wikipedia sentences with a teacher → Qdrant
                                                   │
                                              4. train one head per concept
                                                   │
                                              5. encode: concept blocks + BM25 backbone
```

| Step | Command | What it does |
|------|---------|-------------|
| 1 | `minicoil vocab build` | Cluster MUSE translation pairs into bilingual concepts (Louvain, ~79.6k concepts) |
| 2a | `minicoil scan` | Stream Wikipedia and count concept occurrences per language (CPU only) |
| 3 | `minicoil vocab prune` | Keep concepts with data in both languages (12,357 for the published model). `scripts/phase2_*` then select the 2,398 concepts mMARCO queries need |
| 2b | `minicoil embed` | Store Wikipedia sentences per concept in Qdrant with a teacher vector (`Qwen/Qwen3-Embedding-0.6B` for the published model) |
| 4 | `minicoil train` | Train `tanh(Linear(384 → 8))` per concept on token-pooled mE5-small inputs, with hard-mined bilingual cosine triplets |
| 5 | `MiniCoilEncoder` | Encode text to sparse vectors; `minicoil export-hf` packages a checkpoint |
| – | `minicoil eval` | Evaluate on frozen mMARCO splits against locked baselines |

Scan runs before prune, which needs its counts. Embed runs after prune, so Qdrant only
holds final concept ids. The reasons behind each choice are in
[docs/architecture.md](docs/architecture.md).

## Setup

Prerequisites: Python 3.12, [`uv`](https://docs.astral.sh/uv/), and Docker for Qdrant
(needed by embed, train and eval).

```bash
uv sync --group dev
docker compose up -d          # Qdrant on localhost:6333 (REST) and 6334 (gRPC)
uv run minicoil --help
uv run pytest -q -m "not slow"   # fast suite; plain `pytest` adds the model-loading tests
uv run ruff check . && uv run ruff format --check .
```

## Train from scratch

Fetch the MUSE dictionaries. The archive only contains raw pair files;
`concept_vocabulary.json` and `word_to_concept.json` are created by `minicoil vocab build`.

```bash
mkdir -p data/muse
curl -L "https://dl.fbaipublicfiles.com/arrival/dictionaries.tar.gz" -o "data/dictionaries.tar.gz"
tar -xzf "data/dictionaries.tar.gz" -C "data"
cp "data/dictionaries/en-es.txt" "data/muse/en-es.txt"
cp "data/dictionaries/es-en.txt" "data/muse/es-en.txt"
```

End-to-end smoke test, which uses the code-default teacher (mE5-small):

```bash
uv run minicoil vocab build
uv run minicoil scan --max-articles 1000
uv run minicoil vocab prune --threshold 1
uv run minicoil embed --max-articles 1000
uv run minicoil train --concepts 0:10 --no-wandb
```

### Reproducing the published checkpoint

The published model ships its 2,398-concept vocabulary, so it can serve directly as the
data directory. Alternatively, rebuild the selection with `scripts/phase2_concept_demand.py`
and `scripts/phase2_make_datadir.py` ([docs/03](docs/03-vocabulary-pruning.md)).

```bash
uv run hf download Jocana/minicoil-en-es concept_vocabulary.json word_to_concept.json \
  --local-dir data/phase2

# Teacher vectors: about 5.5 h on one A10G-class GPU
uv run minicoil embed --data-dir data/phase2 --max-articles 400000 --store-cap 800 --lang both \
  --mining-encoder Qwen/Qwen3-Embedding-0.6B --mining-pooling sentence \
  --encode-batch-size 64 --collection-name minicoil_sentences_phase2 --recreate

# Heads: the recorded recipe
uv run minicoil train --data-dir data/phase2 --collection-name minicoil_sentences_phase2 \
  --output-dir data/concept_models_phase2 --encode-cache-dir data/enc_cache_phase2 \
  --epochs 80 --train-epoch-size 2000 --margin-scale 2.0 --lr-patience 3 --dropout 0.10 \
  --batched --no-wandb
```

- **Exact arguments:** the published heads were trained in parallel runs and merged. The
  arguments of each run are in its W&B metadata.
- **Randomness:** Wikipedia sampling, mining and dropout are random, so a re-run gives
  equivalent heads, not identical ones.
- **Evaluate before trusting** a fresh checkpoint: see [docs/07](docs/07-evaluation.md).

### Publishing a model

`export-hf` packages a checkpoint as a Hugging Face model repo:

- stacked heads in `model.safetensors`;
- `config.json` with the concept order and inference defaults;
- both vocabulary files;
- the vendored inference code;
- a model card whose metrics are rendered from an eval report, so the card cannot drift
  from the run it cites.

```bash
uv run minicoil export-hf \
  --checkpoint data/concept_models_phase2/concept_layers.pt \
  --data-dir data/phase2 \
  --out data/hf/minicoil-en-es \
  --repo-id <org>/<model-name> \
  --report reports/eval_phase2/<report>.json \
  --baselines data/eval/baselines_mmarco_phase2.json
```

- **Uploading:** add `--push` (run `hf auth login` first, or set `HF_TOKEN`).
- **Visibility:** repos are created private unless `--no-private` is passed.
- **Inference defaults:** recorded at export time (currently `lemma_match`) and stored in
  `config.json`, so a download reproduces the reported numbers without extra settings.

## Repository layout

```text
.
├── src/minicoil_v2/        # the package: pipeline stages, encoder, export, eval harness (eval/)
├── tests/                  # pytest; -m slow for model-loading tests
├── scripts/                # concept selection, baseline locking, published-model check, perf/
├── research/               # one-off exploration scripts behind the documented findings
├── docs/                   # step-by-step docs, design decisions, research log and notes
├── data/eval/              # frozen eval splits and locked baselines (rest of data/ is generated)
├── reports/                # eval reports cited by the docs and the model card
├── docker-compose.yml      # Qdrant
└── AGENTS.md               # contributor and coding-agent guide (CLAUDE.md points to it)
```

Generated artifacts (all under `data/`, gitignored):

| Artifact | Produced by | Description |
|----------|-------------|-------------|
| `concept_vocabulary.json` | `vocab build` / `vocab prune` | Concepts (EN + ES words per concept) |
| `word_to_concept.json` | `vocab build` / `vocab prune` | Surface form → concept id, per language |
| `*_full.json` | `vocab prune` | Backups of the pre-prune vocabulary |
| `concept_sentence_counts.json` | `scan` | Per-concept sentence counts for pruning |
| Qdrant collection | `embed` | Sentences, focal words, concept ids and teacher vectors |
| `<output-dir>/concept_layers.pt` | `train` | Merged heads (plus per-batch checkpoints) |

## Configuration

Configure runs with CLI flags (`minicoil <command> --help`). Each stage also has a
pydantic-settings class with its own environment prefix, which applies when the class is
constructed from Python. The `minicoil` CLI passes every option explicitly, so for CLI runs
those variables have no effect. The exceptions are the Qdrant connection
(`MINICOIL_QDRANT_URL`, `MINICOIL_QDRANT_API_KEY`, `MINICOIL_QDRANT_COLLECTION_NAME`) for
`train` and for `embed --cloud-inference`, and `MINICOIL_EMBED_OPENROUTER_API_KEY`. Those
also read a `.env` file (see `.env.example`).

| Stage | Env prefix | Settings class |
|-------|-----------|---------------|
| Vocabulary build | `MINICOIL_VOCAB_` | `VocabBuildSettings` |
| Vocabulary prune | `MINICOIL_PRUNE_` | `VocabPruneSettings` |
| Vocabulary subset | `MINICOIL_SUBSET_` | `SubsetSettings` |
| Qdrant connection | `MINICOIL_QDRANT_` | `QdrantSettings` |
| Scan | `MINICOIL_SCAN_` | `ScanSettings` |
| Embed | `MINICOIL_EMBED_` | `EmbedSettings` |
| Training | `MINICOIL_TRAIN_` | `TrainSettings` |

Inference and eval switches (`MINICOIL_V2_*`, `MINICOIL_EVAL_*`) are listed in
[docs/06](docs/06-inference-and-scoring.md) and [docs/07](docs/07-evaluation.md).

## Documentation

| Document | Contents |
|----------|----------|
| [01: Concept vocabulary](docs/01-concept-vocabulary.md) | MUSE graph construction, Louvain clustering, recursive splitting |
| [02: Sentence extraction](docs/02-sentence-extraction.md) | Wikipedia scan and embed, teacher vectors, Qdrant schema |
| [03: Vocabulary pruning](docs/03-vocabulary-pruning.md) | Frequency pruning, id renumbering, demand-driven concept selection |
| [04: Training](docs/04-training.md) | Input vs teacher, pipeline, batched trainer, hyperparameters, published recipe |
| [05: Contrastive sampling](docs/05-four-quadrant-contrastive.md) | Bilingual 5-point samples, hard mining, 4-term cosine loss |
| [06: Inference and scoring](docs/06-inference-and-scoring.md) | Encoder pipeline, BM25 backbone, sparse layout, limitations, export |
| [07: Evaluation](docs/07-evaluation.md) | mMARCO harness, gates, baselines, sealed test, current results |
| [Architecture decisions](docs/architecture.md) | DD-001 to DD-015 with evidence |
| [Research log](docs/research-log.md) | How the model got here: every phase, decision, dead end and open question |
| [Research notes](docs/research/README.md) | Archived findings, specs and reports |
| [miniCOIL v1 background](docs/minicoil-v1-background.md) | v1 design and benchmarks, and how EN-ES differs |

## License

This repository and the published model are licensed under
[CC BY-NC 4.0](LICENSE) (Attribution-NonCommercial 4.0 International). You may share and
adapt them with attribution, but not for commercial purposes.

> **Why non-commercial, and why this will change.** Every concept is built from the
> [MUSE](https://github.com/facebookresearch/MUSE) bilingual dictionaries, which are
> licensed CC BY-NC 4.0, so the vocabulary and the model trained on it inherit that
> restriction. This is a temporary choice: we plan to rebuild the concept vocabulary from
> permissively licensed sources and relicense under more permissive terms. Until a new
> license is published here, CC BY-NC 4.0 applies.

The base encoder (`multilingual-e5-small`, MIT), the teacher (`Qwen3-Embedding-0.6B`,
Apache-2.0) and the libraries keep their own licenses. Wikipedia text (CC BY-SA 4.0) is
used for training but not redistributed. The evaluation baselines and datasets also have
their own terms: NLLB is CC BY-NC 4.0, and mMARCO derives from MS MARCO, which is for
non-commercial research. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
