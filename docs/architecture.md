# Architecture Decisions

Design decisions behind miniCOIL EN-ES. Each entry gives the context, the options
considered, the evidence and the decision. Superseded entries are kept, marked, so the
reasoning stays traceable. The step docs (`01-*` to `07-*`) describe the implementation;
the [research log](research-log.md) has the full history and every number's source.

| ID | Decision | Status |
|---|---|---|
| DD-001 | mE5-small as the input encoder | Current |
| DD-002 | Concepts from MUSE graph clustering | Current |
| DD-003 | Separate teacher and input encoders | Current |
| DD-004 | Sentence-level input encoding | **Superseded by DD-005** |
| DD-005 | Token pooling around the concept word | Current |
| DD-006 | Hard mining with per-concept margins | Current |
| DD-007 | Qwen3-Embedding-0.6B as the teacher | Current |
| DD-008 | 8-D heads | Current |
| DD-009 | One scorer: BM25 backbone plus calibrated concept blocks | Current |
| DD-010 | Language-aware query stopwords | Current |
| DD-011 | Demand-driven concept selection | Current |
| DD-012 | Stopword-gated lemma fallback | Current |
| DD-013 | Instruction prefixes never fire concepts | Current |
| DD-014 | mMARCO gate on MRR@10, covered slice plus all queries | Current |
| DD-015 | CC BY-NC 4.0, until the vocabulary is rebuilt | Current (temporary) |

## DD-001: Multilingual Encoder Selection

**Context:** v1 uses English-only encoders (jina-embeddings-v2-small-en + mxbai-embed-large-v1). v2 needs a multilingual encoder with strong EN/ES cross-lingual alignment.

**Experiment:** Cross-lingual alignment test on 30 EN/ES sentence pairs:

| Model | Dim | Params | Cross-sim | Retrieval Acc | Min sim |
|-------|-----|--------|-----------|---------------|---------|
| mE5-small | 384 | 118M | 0.9126 | 100% | 0.8448 |
| mE5-base | 768 | 278M | 0.9190 | 100% | 0.8792 |
| mE5-large | 1024 | 560M | 0.9253 | 100% | 0.8807 |

**Decision:** `intfloat/multilingual-e5-small` (384-D) as the input encoder. All three models hit 100% retrieval accuracy, and extra dimensions give marginal benefit through a small per-concept bottleneck. Each head is `Linear(384 → 8)` (DD-008).

**Revisited:** in May 2026 Qwen3-Embedding-0.6B was tried as the *input* encoder (word-token pooled, `Linear(1024 → 4)`). Despite better sense silhouettes it lowered cross-lingual MLQA nDCG@10 (−0.0117 / −0.0066) with monolingual flat, and mE5-small stayed (research log 1.5).

## DD-002: Concept Vocabulary via Graph Clustering

**Context:** v1 has a per-word vocabulary (30k English words). v2 needs shared bilingual concepts where EN and ES translations activate the same sparse slots.

**Approach:** MUSE bilingual dictionaries → bipartite graph → Louvain community detection → recursive splitting of oversized communities → cluster-size filtering. Full algorithm in [01-concept-vocabulary.md](01-concept-vocabulary.md).

**Key finding:** Without top-K translation pruning, polysemous words create mega-clusters (up to 589 words). Top-2 pruning + max_cluster=20 (Louvain with recursive splitting) produces ~79.6k clean concepts (median size 2, P99 size 9). A linear resolution schedule (`+0.5` per level) beat doubling: +0.003 modularity, +0.017 weight-1 recall, 303 fewer two-word fragments.

## DD-003: Separate Teacher and Input Encoders

**Context:** Training needs two signals: a vector the head consumes (the *input*), and a similarity judgment that picks positives and negatives (the *teacher*, or mining encoder). v1 uses a large external encoder as teacher (mxbai-embed-large-v1, 1024-D). Loading two encoders during training caused memory pressure on small GPUs.

**Early design:** mE5-small for both. A 50-concept run reached 4-D cross-lingual cosine 0.988+ with triplet loss ~0.096.

**Decision:** decouple them.

- **Teacher:** runs once, at embed time. Its vectors are stored in Qdrant, and at training time they only build each concept's distance matrix, so their dimension is free.
- **Input:** always token-pooled mE5-small (384-D). The trainer re-encodes every row through the same code path as inference, and tests pin training input to inference input byte for byte (`tests/test_trainer_input_path.py`).

This made 4096-D and 1024-D teachers possible without changing the heads.

## DD-004: Sentence-Level vs. Per-Token Input Encoding — superseded

> **Superseded by DD-005.** Kept for the reasoning; the "revisit if" condition below was met in May 2026.

**Context:** v1 feeds each word's head a per-token embedding: the subword states of the target word inside its sentence, averaged (512-D). The first v2 design fed the head a single pooled sentence embedding instead.

**Rationale at the time:**

- **Alignment for free.** The multilingual encoder already maps "She won a prize" and "Ella ganó un premio" to nearby points (cosine 0.91+).
- **Simplicity.** Sentence encoding avoids multilingual word-to-subword resolution, tracking which concept word triggered a sentence, and a policy for sentences with several concept words.
- **Augmentation.** Trimming to ~11 words around the concept word was expected to focus the sentence embedding on the word.

**Revisit if:** concept discrimination or monolingual retrieval came out weak and other causes had been ruled out.

**What happened:** diagnostics showed sentence pooling carries almost no concept signal (intra-minus-inter concept cosine gap +0.017), and per-token pooling recovers it (+0.116), cross-lingually too (DD-005).

## DD-005: Token Pooling Around the Concept Word

**Context:** training rejected ~98% of triplets. The question was whether the cause was the sampler, the margin, the data volume, the encoder size or the representation.

**Evidence** ([research/pooling-strategy.md](research/pooling-strategy.md), independently re-run in [research/pooling-verification.md](research/pooling-verification.md)):

| Strategy | Concept gap | Cross-lingual gap | Rejection @ 0.10 |
|---|---|---|---|
| mE5-small, sentence pool | +0.017 | – | 93.5% |
| mE5-large, sentence pool | +0.016 | – | 93.7% |
| mE5-small, sentence pool, 10x data | +0.007 | – | 95.4% |
| **mE5-small, token pool** | **+0.116** | **+0.123** | 44.8% |

- **More data and a larger encoder do not help.** Both leave the gap where it is, or make it worse.
- **The signal is not lexical identity.** It holds when the positive must use a different surface word (cat vs kitten, cat vs gato).
- **Trained heads separate senses on token-pooled input.** They won 11/11 monosemic concepts on intra-concept cosine and 7/9 polysemous ones on sense separation; heads trained on sentence-pool input were worse on every metric ([research/4d-polysemy-validation.md](research/4d-polysemy-validation.md)).

**Decision:** the head's input is the mean of the last hidden states of the subword tokens overlapping the concept word, L2-normalized, averaged over occurrences (`token_pooling.pool_spans`). The same function serves embed, training and inference.

## DD-006: Hard Mining with Per-Concept Margins

**Context:** the verification of DD-005 found the other half of the rejection problem: v2 drew two *random* same-concept sentences and labeled the closer one positive. Inside a concept, random pairs are nearly equidistant from the anchor, so almost every sample fails a margin. v1 mines hard by construction.

**Decision** (`BilingualSampler._pick_pair_mined`):

- **Positive:** a random pick among the 20 teacher-nearest candidates.
- **Negative:** the nearest candidate beyond that pool that is at least the margin floor further away (semi-hard).
- **Loss margin:** each sample uses the teacher's own gap, `d(a, neg) − d(a, pos)`.
- **Margin floor:** scaled per concept, `max(0.1, margin_scale × std(distances))`, because a fixed 0.1 never bites on tight concepts and is trivial on spread ones. The published model used `margin_scale = 2.0`.

The random sampler is kept as `hard_mining=False` for ablations. Details: [05-four-quadrant-contrastive.md](05-four-quadrant-contrastive.md).

**Alternative tried:** symmetric in-batch InfoNCE (τ = 0.1, cross-language teacher-nearest positive). It missed the cross-lingual bar by 0.109 and lowered monolingual scores, so the triplet loss stayed.

## DD-007: Qwen3-Embedding-0.6B as the Teacher

**Context:** the teacher sets what "same sense" means for each concept. History:

- CLIP text (512-D) first.
- Then mE5-small itself.
- Then Qwen3-Embedding-8B (4096-D, through Qdrant Cloud inference) for Phase 0/1.

**Decision:** for the published model (Phase 2), `Qwen/Qwen3-Embedding-0.6B` in sentence mode (1024-D), computed on a GPU machine into a self-hosted Qdrant (not through cloud inference). The 8B model was judged only marginally better, while 0.6B is ~10x cheaper to run and 4x smaller to store, which bought the scale to 2,398 concepts. `max_seq_length` is capped at 512, because Qwen3's native 32k combined with length-sorted batching ran the GPU out of memory.

**Open:** sentence pooling for the teacher was never ablated against word-token pooling, although a May silhouette study scored Qwen3 sentence pooling lowest for sense separation. The code default teacher is still mE5-small (token-pooled), which reproduces an earlier recipe, not the published one.

## DD-008: 8-D Heads

**Context:** with 4-D heads, the cosine-trained outputs collapsed in norm (mean 0.063 on the 1,000-concept MLQA run) and ranking was compressed. The 4-D bottleneck was the named limiter for eng-eng against v1's per-word model and for the cross-lingual ranking failures.

**Decision:** `OUTPUT_DIM = 8`. Concept indices stay `concept_num * OUTPUT_DIM + offset`, far below the backbone range.

**Open:** there is no isolated 4-D vs 8-D ablation at a fixed recipe.

## DD-009: One Scorer, BM25 Backbone plus Calibrated Concept Blocks

**Context:** concept blocks alone cannot retrieve. Names, numbers and rare words have no concept: eng-eng dev nDCG@10 was 0.177 against v1's 0.746. Combining a separate BM25 score with a miniCOIL score (`α·BM25 + (1−α)·miniCOIL`) looked good on validation (eng-spa +0.0246) but lost on test (−0.0129).

**Decision:** one sparse vector, one dot product ([06-inference-and-scoring.md](06-inference-and-scoring.md)):

- **BM25 backbone inside the encoder.** Every non-concept token is a BM25 term from FastEmbed's `Qdrant/bm25`, the same implementation as the `bm25` baseline (parity tests). Backbone indices are hashed into a range disjoint from concept indices. Replacing an earlier hand-rolled BM15 with it gained +0.02 to +0.09 val nDCG@10.
- **Calibrated blocks.** Each concept block is normalized to unit length and scaled by the BM25 weight its word would have had. Raw `tanh` outputs (~0.17) were drowned by full-weight lexical terms; calibrating raised Phase-1 eng-eng val nDCG@10 from 0.7275 to 0.8414.
- **Cross-lingual backbone matching is kept.** Shared cognates and numbers are genuine signal; scoping the backbone by language cost 0.09 / 0.04.
- **IDF comes from Qdrant** (`Modifier.IDF`), as for v1.

## DD-010: Language-Aware Query Stopwords

**Context:** the backbone applies English stopwords to all text, like the baseline. Spanish function words in queries ("de", "la", "que") survived and, being rare in an English corpus, scored as high-IDF noise. About 94% of es→en queries emitted them; it was the dominant failure at ranks 11–100.

**Decision:** queries also drop the stopwords of their own language. Documents are unchanged, so IDF and document length stay identical and the other pairs are untouched. es→en val MRR@10 rose from 0.55 to ~0.71. This replaced an earlier cross-lingual backbone down-weight (0.5), which is still available as `MINICOIL_V2_XL_BACKBONE_WEIGHT` (default 1.0).

## DD-011: Demand-Driven Concept Selection

**Context:** Phase 0/1 trained the concepts Wikipedia supplied enough sentences for (64). The resulting gap on every pair was a coverage gap.

**Decision:** train the smallest concept set that covers the content words of mMARCO queries.

- **Selection:** concepts ranked by query demand on an 80% qid slice, cut at the 90% mark: **2,398 concepts**, covering 67.1% of content tokens absolute (60.5% on the held-out 20%).
- **Ceiling:** all 12,357 concepts reach only 74.6%. Names, numbers and unlisted inflections ride the backbone.

| Coverage target | Concepts |
|---|---|
| 80% | 1,479 |
| 90% (chosen) | 2,398 |
| 95% | 3,145 |
| 99% | 3,922 |

Spec: [research/phase2-scale-concepts-design.md](research/phase2-scale-concepts-design.md).

**Caveat:** the frozen test splits were not restricted to the held-out 20%; only ~18% of test queries are from it, so selection saw most test query texts (no labels). See [07-evaluation.md](07-evaluation.md).

## DD-012: Stopword-Gated Lemma Fallback

**Context:** MUSE lists some inflections but not all, so "injections" or an unlisted Spanish plural missed its concept.

**Decision:** a token that misses the surface lookup is lemmatized (simplemma) and looked up again. Both token and lemma are gated on stopwords. Without the gate, auxiliaries ("is" → "be", "es" → "ser") fired a near-stopword concept on nearly every text. This improved all four test pairs and turned eng-spa from a loss into a win against translate-bm25 (+0.0081 MRR@10). It is part of the published config (`lemma_match` in `config.json`).

## DD-013: Instruction Prefixes Never Fire Concepts

**Context:** mE5 expects `"query: "` / `"passage: "` prefixes. Concepts were matched over `prefix + text`, and "passage" and "query" are both concept words, so every English text emitted those two blocks whatever it said. The trainer had the same bug.

**Decision:** concepts are matched on the text alone and spans are shifted by the prefix length (`concept_match.match_concepts_after_prefix`), in the encoder, the trainer and the coverage scan. `encode_sparse` picks the prefix from `is_query`. The fix moved val MRR@10 en-en 0.8845 → 0.8889 and en-es 0.7188 → 0.7304; the Spanish-query pairs were unaffected.

## DD-014: mMARCO Gate on MRR@10, Covered Slice plus All Queries

**Context:**

- MLQA (parallel Wikipedia) was the first benchmark. It is small enough for a laptop but not comparable to public results.
- MIRACL-es (10.4M passages) and full mMARCO (8.8M) did not fit locally.

**Decision:**

- **Benchmark:** the mMARCO `dev.small` queries against their gold sub-corpus.
- **Gate:** test MRR@10, beating v1 (eng-eng), BM25 (spa-spa) and NLLB translate-then-BM25 (both cross-lingual pairs).
- **Two splits:**
  - The covered slice (queries sharing a trained concept with their gold) is the headline.
  - An all-queries gate keeps the coverage cliff visible: on uncovered queries v2 trails translate-bm25 by 0.11–0.18 MRR@10.
- **Guardrails:** sha-verified frozen splits, a sealed and audited test split, and hardcoded win conditions keep tuning off the test set.

Details: [07-evaluation.md](07-evaluation.md).

## DD-015: CC BY-NC 4.0, Until the Vocabulary Is Rebuilt

**Context:** every concept is derived from the MUSE EN-ES dictionaries (CC BY-NC 4.0), and the vocabulary files ship with the model. The other inputs are permissive: mE5-small (MIT), Qwen3-Embedding-0.6B (Apache-2.0), FastEmbed (Apache-2.0) and simplemma (MIT code). The most restrictive input governs.

**Decision:** release the repository and the model under CC BY-NC 4.0 (`LICENSE`, `pyproject.toml`, the model card; `export-hf --license` defaults to `cc-by-nc-4.0`). The intent is to relicense once the concept vocabulary is rebuilt from permissively licensed bilingual sources, which means re-running the pipeline from step 1 and re-training the heads. Third-party terms are summarized in `THIRD_PARTY_NOTICES.md`.
