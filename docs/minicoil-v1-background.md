# miniCOIL Research Background

## What is miniCOIL?

miniCOIL is a lightweight sparse neural retrieval model created by Qdrant. It acts as a
semantically-aware upgrade to BM25 — keeping BM25's efficiency and inverted-index compatibility
while adding contextual meaning to keyword matching.

The core problem it solves: **the same word has different meanings in different contexts**
(e.g., "fruit bat" vs. "baseball bat"), and BM25 is blind to this.

## Scoring Formula

miniCOIL extends BM25 with a semantic similarity component:

```
score(D, Q) = Σ IDF(q_i) · Importance(q_i, D) · Meaning(q_i, d_j)
```

| Component     | Source    | What it captures                                         |
|---------------|-----------|----------------------------------------------------------|
| IDF(q_i)      | BM25      | Inverse document frequency — rare terms matter more      |
| Importance    | BM25      | Term frequency signal in the document                    |
| Meaning       | miniCOIL  | Semantic similarity between matched keywords in Q and D  |

When miniCOIL can't encode a word (not in vocabulary), it falls back to pure BM25 scoring.

## Model Architecture

The architecture is intentionally simple:

- **One trainable linear layer per vocabulary word** (no cross-word interactions)
- **Input:** 512D dense embedding from `jina-embeddings-v2-small-en`
- **Output:** 4D vector per word (compressed via linear layer + tanh activation)
- **Vocabulary:** 30,000 most common English words (cleaned, stemmed, >3 characters)
- **Sparse representation:** Each word gets 4 consecutive cells in the sparse vector
  (one per dimension), making it compatible with standard inverted indexes

Total model = stack of 30k independent linear layers (512 -> 4).

## Training Process

### Data

- 40 million sentences from OpenWebText
- No labeled relevance pairs needed (self-supervised)

### Method

1. For each word in vocabulary, collect ~8,000 sentences containing that word
2. Encode sentences with `mxbai-embed-large-v1` (external dense encoder)
3. Cluster sentences by meaning — the assumption is that sentences sharing a word
   will naturally cluster by the word's different senses
4. Sample triplets (anchor, positive, negative) based on cluster distances
5. Train per-word linear layer with **triplet loss** (margin = 0.1)

### Data Augmentation

- Cut surrounding context: keep target word ± 1-3 neighboring words
- Forces the model to learn from limited context

### Training Parameters

| Parameter       | Value                           |
|-----------------|---------------------------------|
| Epochs          | 60                              |
| Optimizer       | Adam (lr = 1e-4)                |
| Input encoder   | jina-embeddings-v2-small-en     |
| Triplet encoder | mxbai-embed-large-v1            |
| Time per word   | ~50 seconds (single CPU)        |
| Output dim      | 4                               |

## Sparse Vector Encoding

miniCOIL converts the 4D meaning vectors into a sparse bag-of-words format:

1. Each vocabulary word is assigned a base index (word_index × 4)
2. The 4D vector occupies 4 consecutive positions starting at that base index
3. Values are normalized before insertion
4. The resulting sparse vector is compatible with any standard inverted index

This "4D semantic space" technique means the sparse vector has at most `4 × num_matched_words`
non-zero entries per document.

## Comparison with Alternatives

| Property                    | BM25 | SPLADE | COIL   | miniCOIL |
|-----------------------------|------|--------|--------|----------|
| Semantic awareness          | No   | Yes    | Yes    | Yes      |
| Inverted index compatible   | Yes  | Yes*   | No     | Yes      |
| Domain-independent          | Yes  | No     | No     | Yes      |
| Needs relevance labels      | No   | Yes    | Yes    | No       |
| Computational cost          | Low  | High   | Medium | Low      |
| Document expansion          | No   | Yes    | No     | No       |

*SPLADE uses document expansion which reduces sparsity and increases index size.

### Why not SPLADE?

SPLADE expands documents with related terms, which increases index size and computational
cost. It also requires relevance-labeled training data, making it domain-dependent.

### Why not COIL?

COIL uses subword tokenization and end-to-end relevance training, making it
domain-dependent and incompatible with standard inverted indexes.

### Why not dense-only retrieval?

Dense vectors excel at semantic similarity but miss exact keyword matches.
Hybrid search (dense + sparse) consistently outperforms either alone.
miniCOIL provides the sparse component with better quality than BM25.

## Benchmarks (BEIR, NDCG@10)

Evaluated on BEIR datasets **without any prior training on them**:

| Dataset    | BM25    | miniCOIL | Delta   |
|------------|---------|----------|---------|
| MS MARCO   | 0.237   | **0.244**| +0.007  |
| NQ         | 0.304   | **0.319**| +0.015  |
| Quora      | 0.784   | **0.802**| +0.018  |
| FiQA-2018  | 0.252   | **0.257**| +0.005  |
| HotpotQA   | **0.634**| 0.633   | -0.001  |

Outperforms BM25 in 4/5 domains. The gains are modest but consistent, and achieved
without any domain-specific training.

## Integration with Qdrant

- Ships with FastEmbed v0.7.0+
- Sparse vectors must be configured with `Modifier.IDF` (Qdrant calculates IDF at query time)
- In hybrid search, reuses the dense encoder outputs — no extra encoding overhead
- Available on HuggingFace as `Qdrant/minicoil-v1`

## Key References

- [miniCOIL article (Qdrant)](https://qdrant.tech/articles/minicoil/)
- [GitHub - qdrant/miniCOIL](https://github.com/qdrant/miniCOIL)
- [FastEmbed integration docs](https://qdrant.tech/documentation/fastembed/fastembed-minicoil/)
- [HuggingFace model](https://huggingface.co/Qdrant/minicoil-v1)
- [Interactive demo](https://minicoil.qdrant.tech/)

## How miniCOIL EN-ES differs from v1

| | miniCOIL v1 | miniCOIL EN-ES |
|---|---|---|
| Languages | English | English and Spanish, including cross-lingual |
| Unit | One head per English word (30k) | One head per bilingual concept (2,398 trained, from 12,357 pruned MUSE clusters) |
| Input | `jina-embeddings-v2-small-en` token vectors (512-D) | `multilingual-e5-small` vectors token-pooled around the concept word (384-D) |
| Head | `Linear(512 → 4) + tanh` | `Linear(384 → 8) + tanh` |
| Teacher | `mxbai-embed-large-v1` | `Qwen3-Embedding-0.6B`, sentence mode (1024-D) |
| Training data | OpenWebText | Wikipedia EN + ES |
| Loss | Triplet loss, hard-mined | Bilingual 4-term cosine triplet loss, hard-mined, adaptive margins |
| Out-of-vocabulary words | BM25 | BM25 backbone (FastEmbed `Qdrant/bm25`), with query-side stopwords per language |

Details: [architecture.md](architecture.md) and the step docs `01`–`06`.
