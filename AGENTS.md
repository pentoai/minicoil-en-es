# miniCOIL EN-ES: guide for contributors and coding agents

miniCOIL EN-ES is a bilingual (English/Spanish) sparse retrieval model. It extends
[miniCOIL v1](https://qdrant.tech/articles/minicoil/) (English-only, one head per word)
to one head per **bilingual concept**: a cluster of translations such as
`{dog, dogs}` / `{perro, perros, perrita}`. The Python package is `minicoil_v2`; the
published model is [`Jocana/minicoil-en-es`](https://huggingface.co/Jocana/minicoil-en-es).

## How it works

- **Concepts.** MUSE EN-ES dictionaries are clustered with Louvain into ~79.6k concepts.
  Pruning leaves 12,357; 2,398 were selected by query demand and trained.
- **Input.** `multilingual-e5-small` hidden states, token-pooled over the concept word's
  subwords (384-D).
- **Head.** One `tanh(Linear(384 → 8))` per concept, trained with a bilingual 4-term
  cosine triplet loss. Positives and negatives are hard-mined from a teacher's distances
  (`Qwen3-Embedding-0.6B` for the published model).
- **Output.** One sparse vector:
  - an 8-value block per fired concept, unit-normalized and scaled by its BM25 weight;
  - plus a BM25 backbone term (FastEmbed `Qdrant/bm25`) for every other token.
  Qdrant scores it with `Modifier.IDF`.

The why behind each choice is in [docs/architecture.md](docs/architecture.md); the history
is in [docs/research-log.md](docs/research-log.md).

## Pipeline

| Step | Command | Doc |
|------|---------|-----|
| 1 | `minicoil vocab build` | [01-concept-vocabulary.md](docs/01-concept-vocabulary.md) |
| 2a | `minicoil scan` (CPU only: concept counts) | [02-sentence-extraction.md](docs/02-sentence-extraction.md) |
| 3 | `minicoil vocab prune` (+ demand selection scripts) | [03-vocabulary-pruning.md](docs/03-vocabulary-pruning.md) |
| 2b | `minicoil embed` (teacher vectors into Qdrant, final ids) | [02-sentence-extraction.md](docs/02-sentence-extraction.md) |
| 4 | `minicoil train` | [04-training.md](docs/04-training.md), [05-four-quadrant-contrastive.md](docs/05-four-quadrant-contrastive.md) |
| 5 | `MiniCoilEncoder` / `minicoil export-hf` | [06-inference-and-scoring.md](docs/06-inference-and-scoring.md) |
| – | `minicoil eval` | [07-evaluation.md](docs/07-evaluation.md) |

Scan runs before prune because prune needs its counts; embed runs after prune, so Qdrant
only ever holds final concept ids.

## Repository layout

- `src/minicoil_v2/`:
  - `cli.py`: Typer entrypoint (`minicoil`).
  - `constants.py`: shared constants (encoders, `OUTPUT_DIM`, defaults, published model id).
  - `settings.py`: one pydantic-settings class per stage, env prefix `MINICOIL_<STAGE>_`
    (effective from Python; the CLI passes options explicitly, see the README).
  - `build_concept_vocab.py`, `scan_wiki_sentences.py`, `prune_concept_vocab.py`,
    `subset_vocab.py`, `embed_wiki_sentences.py`: steps 1–3.
  - `train_concept_layers.py`: step 4, the reference trainer and `BilingualSampler`.
  - `batched_training.py`: opt-in batched trainer (`--batched`).
  - `concept_match.py`: the single matcher for "does concept C fire on this text",
    shared by encoder, trainer and eval.
  - `token_pooling.py`: the pooled input vector.
  - `encoder.py`: inference (`MiniCoilEncoder`).
  - `hf_export.py`: `export-hf`.
  - `qdrant_store.py`: Qdrant client, collection setup, scroll and upsert.
  - `eval/`: eval harness (datasets, frozen splits, retrievers, baselines gate, reports).
- `tests/`: pytest. `-m slow` tests load transformer models or the published model.
- `scripts/`: what reproduces the published model and its eval
  ([scripts/README.md](scripts/README.md)); `scripts/perf/` holds batched-trainer checks.
- `research/`: one-off exploration scripts, kept for their findings
  ([research/README.md](research/README.md)).
- `data/eval/`: frozen splits and locked baselines (tracked); the rest of `data/` is
  generated and gitignored.
- `reports/`: eval reports cited by the docs and the model card.

## Commands

```bash
uv sync --group dev
uv run pytest -q -m "not slow"   # fast suite
uv run pytest -q                 # everything, incl. model-loading tests (downloads models)
uv run ruff check . && uv run ruff format --check .
docker compose up -d             # Qdrant, for embed / train / eval
```

## Invariants: do not break these

- **Training input equals inference input.** The trainer must pool exactly what
  `MiniCoilEncoder` pools for a document (`token_pooling.pool_spans`, `concept_match`,
  the `"passage: "` prefix, surface-form matches). `tests/test_trainer_input_path.py`
  pins it.
- **Prefixes never fire concepts.** Match on the text, shift spans by `len(prefix)`
  (`match_concepts_after_prefix`).
- **Backbone parity.** The backbone must stay identical to FastEmbed `Qdrant/bm25` on the
  English path (`tests/test_backbone_parity.py`), so comparisons with the `bm25` baseline
  hold.
- **Index layout.** Concept blocks at `concept_num * OUTPUT_DIM + offset`; backbone at
  `BACKBONE_BASE (1 << 27)` and above, below 2³¹. Changing either invalidates every index
  and the published config.
- **Export order.** The stacked tensor has no ids; `config.json`'s `concept_ids` defines
  row order. `tests/test_hf_export.py` checks the export against the source checkpoint.
- **Eval hygiene.** Iterate on val. The test split is sealed (`MINICOIL_EVAL_ALLOW_TEST=1`)
  and audited. Do not edit frozen splits or locked baselines by hand.
- **Teacher vs input.** The teacher (mining encoder) only builds distance matrices; the
  heads never see it. Do not feed teacher vectors to the heads.

## License

CC BY-NC 4.0 (`LICENSE`), inherited from the MUSE dictionaries, and meant to change once the
vocabulary is rebuilt from permissive sources. Do not add dependencies or data with
licenses that are incompatible with redistribution; record new third-party inputs in
`THIRD_PARTY_NOTICES.md`.

## Conventions

- `uv run` instead of `python`; Python 3.12.
- Ruff for lint and format (config in `pyproject.toml`, line length 100).
- `loguru` for logging, `typer` for the CLI, `pydantic-settings` for configuration.
- Modern typing (`list[str]`, `X | None`); no elaborate pydantic logic unless needed.
- Keep `research/` scripts out of the import graph of `src/`.
- Any change that alters training dynamics or scoring must be checked on the val gate
  (`minicoil eval run ... --split val`), not only unit tests.
