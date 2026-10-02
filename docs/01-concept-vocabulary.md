# Step 1: Build the Concept Vocabulary

**Script:** `src/minicoil_v2/build_concept_vocab.py`
**Command:** `minicoil vocab build`
**Settings class:** `VocabBuildSettings`

## Goal

Construct a bilingual concept vocabulary where English and Spanish words that are
translations of each other share a single concept ID. This concept ID later
determines which sparse slots are activated during inference.

## Prerequisites

Download MUSE dictionary files and place them under `data/muse/`:

```bash
mkdir -p data/muse
curl -L "https://dl.fbaipublicfiles.com/arrival/dictionaries.tar.gz" -o "data/dictionaries.tar.gz"
tar -xzf "data/dictionaries.tar.gz" -C "data"
cp "data/dictionaries/en-es.txt" "data/muse/en-es.txt"
cp "data/dictionaries/es-en.txt" "data/muse/es-en.txt"
```

Source: [MUSE bilingual dictionaries](https://github.com/facebookresearch/MUSE).

## Command

```bash
uv run minicoil vocab build
```

## Inputs

| Path (default) | Description |
|----------------|-------------|
| `data/muse/en-es.txt` | English -> Spanish translation pairs (~112k) |
| `data/muse/es-en.txt` | Spanish -> English translation pairs (~112k) |

Each line: two whitespace-separated tokens (source, target). Parsed by `parse_muse_dict`.

## Outputs

| File | Description |
|------|-------------|
| `data/concept_vocabulary.json` | `{concept_id: {en: [words], es: [words]}}` |
| `data/word_to_concept.json` | `{en: {word: concept_id}, es: {word: concept_id}}` |

These files only exist after running the build command, not after downloading MUSE.

## Algorithm

1. **Parse MUSE dictionaries.** Lowercase, keep only alphabetic words with 2+ characters (including accented: `á`, `ñ`, `ü`).

2. **Prune to top-K translations per source word** (default K=2). MUSE files are frequency-ordered. Without pruning, polysemous words like "bajo" (low/bass/short/under) bridge 80+ unrelated English words into one cluster.

3. **Build bipartite graph.** EN words on one side, ES on the other, edges from pruned translation pairs. Words prefixed `en:`/`es:` to avoid name collisions.

4. **Louvain community detection.** Each community is a candidate concept. Communities larger than `max_cluster_size` are recursively split by re-running Louvain at higher resolution (`resolution + 0.5`). Communities that cannot be split further are kept with a warning.

5. **Filter by cluster size.**

| Filter | Condition | Reason |
|--------|-----------|--------|
| Monolingual | Only EN or only ES words | Not useful for cross-lingual retrieval |
| Undersized | Total words < `min_cluster_size` (2) | Too small |

6. **Assign concept IDs by descending size.** The final set is sorted by `len(en) + len(es)` descending before IDs are assigned. `C-00000` is the largest concept. Ties broken by alphabetical order for determinism.

## Examples

Entries from the published vocabulary (IDs are post-prune):

```json
{"C-06550": {"en": ["water"], "es": ["agua"]}}
{"C-01647": {"en": ["cat", "cats"], "es": ["cat", "gata", "gato", "gatos"]}}
{"C-01641": {"en": ["home", "household"], "es": ["casa", "doméstico", "hogar", "inicio"]}}
{"C-03373": {"en": ["cottage", "house"], "es": ["casita", "house"]}}
```

Two things these show:

- **Inflections are separate MUSE entries,** so they land in the same concept
  (`cat`/`cats`, `gato`/`gatos`). Inflections MUSE does not list are handled at inference
  by the lemma fallback ([06-inference-and-scoring.md](06-inference-and-scoring.md)).
- **MUSE is noisy.** It pairs some words with themselves (`cat` appears on the Spanish
  side) and it splits near-synonyms: `home` and `house` are different concepts, and
  `casa` joined `home`.

The `word_to_concept.json` lookup maps every surface form to its concept:

```json
{"en": {"cat": "C-01647", "cats": "C-01647", "home": "C-01641", "house": "C-03373"},
 "es": {"gato": "C-01647", "gatos": "C-01647", "casa": "C-01641", "hogar": "C-01641"}}
```

## Hyperparameters

| Parameter | CLI flag | Default | Effect |
|-----------|----------|---------|--------|
| `max_translations_per_word` | `--max-translations` | `2` | Top-K translations per source word. Higher = more connectivity, more polysemy risk |
| `min_cluster_size` | `--min-cluster-size` | `2` | Minimum words per concept |
| `max_cluster_size` | `--max-cluster-size` | `20` | Maximum words per concept |
| `resolution` | `--resolution` | `1.5` | Louvain resolution. Higher = smaller, more communities. Recursive splits use `resolution + 0.5` |
| `en_es_path` | `--en-es-path` | `data/muse/en-es.txt` | MUSE en->es dictionary |
| `es_en_path` | `--es-en-path` | `data/muse/es-en.txt` | MUSE es->en dictionary |
| `output_dir` | `--output-dir` | `data` | Output directory |

## Typical Output

| Metric | Value |
|--------|-------|
| Total concepts | ~79.6k (79,652 for the published vocabulary) |
| EN words covered | ~91,180 |
| ES words covered | ~94,481 |
| Median cluster size | 2 |
| P99 cluster size | 9 |

## Limitations

- 79.6k concepts is more than needed. Step 3 (pruning) cut the published model's vocabulary
  to 12,357 concepts, and demand-driven selection then picked the 2,398 that were trained
  (see [03-vocabulary-pruning.md](03-vocabulary-pruning.md)).
- Some clusters still contain mild polysemy bridging. The per-concept linear layer handles sense disambiguation via contextual embeddings.
- Only alphabetic words are indexed. Numbers, punctuation, and subword fragments are excluded.

## Design history

- Concepts were first built from connected components of the translation graph plus a
  salvage pass for orphans. Louvain replaced them; `research/compare_vocab_quality.py`
  compares the two partitions (pair recall by edge weight, orphans, fragments).
- Recursive splits originally doubled the resolution at each level. The linear `+0.5`
  schedule gave +0.003 modularity, +0.017 weight-1 recall and 303 fewer two-word
  fragments (`research/compare_resolution_schedules.py`).
- Rationale and numbers: [architecture.md](architecture.md) DD-002 and the
  [research log](research-log.md), section 1.0.
