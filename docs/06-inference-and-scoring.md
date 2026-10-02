# Step 5: Inference and Scoring

**Module:** `src/minicoil_v2/encoder.py` (`MiniCoilEncoder`)
**Shared helpers:** `concept_match.py` (which concepts fire), `token_pooling.py` (the input vector)

## Goal

Turn a text into one sparse vector that an inverted index can score: BM25 with a learned
semantic overlay. Words that belong to a trained concept become an 8-value block from that
concept's head; every other token stays a plain BM25 term. English and Spanish words of
the same concept write to the same block, which is what makes cross-lingual matching
possible without translation.

## Usage

```python
from minicoil_v2.encoder import MiniCoilEncoder

encoder = MiniCoilEncoder.from_pretrained("Jocana/minicoil-en-es")  # or a local export dir

doc = encoder.encode_sparse("El perro corrió por el parque", lang="es")
query = encoder.encode_sparse("dog running", lang="en", is_query=True)
docs = encoder.encode_batch_sparse(texts, lang="en", batch_size=32)
```

- **`lang`** is the language of the text (`"en"` or `"es"`). Encode documents in the
  corpus language and queries in the query language.
- **`is_query`** switches BM25 weighting and the mE5 prefix (below). Always set it for
  queries.
- **Output:** `{sparse_index: value}`.
- **Local checkpoint:** `MiniCoilEncoder(data_dir, model_path=".../concept_layers.pt")`
  loads a checkpoint trained against the vocabulary in `data_dir`.
- **`from_pretrained`** also applies the inference defaults recorded in `config.json`
  (currently `lemma_match`).

The index **must apply IDF** at query time. In Qdrant, declare the sparse vector with
`Modifier.IDF`; the vectors carry term weights only.

```python
from qdrant_client import models

client.create_collection(
    "docs",
    vectors_config={},
    sparse_vectors_config={"minicoil": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)
```

## Encoding pipeline

For each text:

1. **Prefix.** The mE5 instruction prefix is prepended: `"query: "` when `is_query=True`,
   `"passage: "` otherwise. The prefix conditions the encoder only.
2. **Match concepts** (`concept_match.match_concepts_after_prefix`). The lowercased text,
   *without* the prefix, is split by `TOKEN_RE`. Each token is looked up in
   `word_to_concept[lang]`. Spans are then shifted by the prefix length so they line up
   with the tokenizer offsets of the prefixed string.
3. **Lemma fallback** (when `lemma_match` is on, as in the published model). A token that
   misses the surface lookup is lemmatized with simplemma and looked up again
   ("gatos" → "gato", "injections" → "injection"). Both the token and its lemma are
   skipped if they are stopwords: without that gate, auxiliaries ("is"/"are" → "be",
   "es" → "ser") would fire a near-stopword concept on almost every text.
4. **Encode once.** One mE5-small forward pass over the prefixed text, truncated at 256
   tokens.
5. **Pool.** For each fired concept, take the mean of the last hidden states of the
   subword tokens overlapping each occurrence and L2-normalize it, then average over
   occurrences and normalize again (`token_pooling.pool_spans`). For a document word
   matched on its surface form this is byte for byte what the heads were trained on.
   Queries (`"query: "` prefix) and lemma-matched words go through the same pooling, but
   the heads never saw them in training (see Limitations).
6. **Project.** `tanh(W_c · x)` with the concept's `[8, 384]` head.
7. **Calibrate the block.** The heads are trained with a scale-invariant cosine objective,
   so raw `tanh` values are small (around 0.17) and arbitrary in scale. Each block is
   normalized to unit length and scaled by the BM25 weight the displaced word would have
   had: 1.0 on the query side, the tf-saturated, length-normalized BM25 weight on the
   document side. A same-sense match then scores like a lexical match, and opposite senses
   (negative cosine) subtract.
8. **Emit** the block at indices `concept_num * 8 + offset` (`concept_num` is the number
   in the concept id, `C-01647` → 1647), dropping values with `|v| ≤ 1e-6`.
9. **Add the BM25 backbone** for every token that is not a concept word (next section).

## BM25 backbone

The backbone is what keeps lexical recall: names, numbers and rare words have no concept
but still have to match.

- **Same pipeline as the baseline.** It reuses FastEmbed's `Qdrant/bm25`: tokenize,
  stopword and punctuation filter, Snowball stem, BM25 tf weight with `k = 1.2`,
  `b = 0.75`, `avg_len = 256`. That is the exact implementation of the locked `bm25`
  baseline, so backbone-vs-baseline comparisons hold by construction
  (`tests/test_backbone_parity.py`).
- **English rules for both languages.** Like the baseline, it uses the English stemmer
  and stopwords for every text.
- **Weights:** 1.0 per distinct query stem; the BM25 tf weight for document stems.
- **Concept words are excluded,** because their block already carries them.
- **Indices:** `BACKBONE_BASE + |mmh3(stem)| mod (2³¹ − 1 − BACKBONE_BASE)`, with
  `BACKBONE_BASE = 1 << 27`. This range is disjoint from the concept indices (far below
  2²⁷) and stays under Qdrant's 2³¹ sparse-index limit.
- **Query-side stopwords per language.** With English stopwords only, a Spanish query's
  function words ("de", "la", "que") leaked into the backbone and scored as high-IDF
  noise against English documents. About 94% of es→en queries emitted them. Spanish
  queries now also drop Spanish stopwords. Documents are unchanged, and the step is a
  no-op for English queries. It raised es→en val MRR@10 from 0.55 to about 0.71.
- **Cross-lingual matching through the backbone is kept.** Shared cognates, numbers and
  names are genuine signal: scoping the backbone by language cost 0.09 (en→es) and 0.04
  (es→en) nDCG@10 in Phase 1.

## Score

Qdrant scores a document as the dot product of the query and document vectors, with IDF
applied to each index:

```
score(q, d) = Σ_backbone  IDF(t) · 1.0 · w_d(t)                       # plain BM25
            + Σ_concepts  IDF(c-block) · Σ_offsets  q_c[i] · d_c[i]   # BM25 weight × cosine
```

Because both blocks are unit vectors scaled by their BM25 weights, a concept contributes
its BM25 term weight times the cosine between the query's and the document's sense vectors.
That is the miniCOIL v1 formula
([minicoil-v1-background.md](minicoil-v1-background.md)), applied to bilingual concepts
instead of English words.

## Options and environment variables

| Setting | Default | Effect |
|---|---|---|
| `lemma_match` (ctor) / `MINICOIL_V2_LEMMA_MATCH=1` | from `config.json` for `from_pretrained` (true for the published model); otherwise off | Lemma fallback in concept matching. Applies to queries and documents, so re-index after changing it |
| `MINICOIL_V2_BACKBONE_STOPWORD_FIX` | `1` | Set `0` to disable the query-side language stopwords (ablation only) |
| `MINICOIL_V2_CONCEPT_MODE` | `learned` | `lexical` replaces each learned block with one BM25-weighted term at the concept index, an ablation of the "shared concept index" bridge without the heads |
| `backbone_weight` (arg) / `MINICOIL_V2_XL_BACKBONE_WEIGHT` (eval harness, cross-lingual pairs) | `1.0` | Scales backbone terms. Superseded by the stopword fix; kept for experiments |
| `include_backbone` (arg) | `True` | Emit concept blocks only when `False` |
| `max_length` (arg) | `256` | mE5 truncation. A concept word past the cut is matched but yields no block |

## Limitations

- **Coverage cliff.** Cross-lingual matching happens only through shared concepts and the
  backbone's cognates. A query whose content words are all outside the 2,398 trained
  concepts can return nothing against a corpus in the other language. For example,
  "court ruling witnesses" against Spanish documents that say "tribunal" and "testigos":
  only `court` fires, and its concept has `juzgado`, not `tribunal`. Same-language
  retrieval always keeps full BM25 behavior.
- **Truncation.** Documents are cut at 256 mE5 tokens for concept blocks. The backbone
  sees the full text.
- **Train/inference gap on queries and lemmas.** The heads were trained only on
  `"passage: "`-prefixed sentences and surface-form matches. Query encodings use
  `"query: "`, and the lemma fallback pools words the trainer never matched. Both worked
  on the eval gate, but neither is in the training distribution.
- **English stemming for Spanish.** The backbone stems Spanish with the English Snowball
  stemmer, to stay identical to the `bm25` baseline.

## Exporting a model

`minicoil export-hf` packages a checkpoint as a Hugging Face model repository:

- stacked heads in `model.safetensors`;
- `config.json` with the concept id order (which defines the row order), the sparse index
  layout and the inference defaults;
- both vocabulary files;
- the inference code it imports (vendored), so users only need PyPI packages.

The export directory is itself a valid `MiniCoilEncoder` input, so there is no second
inference path. The model card's metric table is rendered from an eval report and the
locked baselines, so it cannot drift from the run it cites. See the README for the
command.
