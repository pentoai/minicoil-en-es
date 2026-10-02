# Research log: miniCOIL EN-ES

Curated history of how miniCOIL EN-ES (bilingual English/Spanish sparse retriever, one small trainable head per bilingual concept) got to its published form. It condenses research notes, run logs and commit messages from the internal development history, most of which lived on branches that were never merged, so the reasoning survives without them.

- **What this is:** a digest, not a reproduction guide. Each section names its sources. Notes that were ported into this repo link to their copy under [`docs/research/`](research/README.md); the rest, and the branch names and commit shas in the [source index](#6-source-index), refer to the internal history and are not published.
- **Where the numbers come from:** internal runs on the setups described. They are only comparable within a phase, because benchmark, slice, BM25 implementation, translator, and metric all changed between phases.
- **Superseded results (kept for history, do not cite as current):**
  - All MLQA numbers (May 2026), which predate the eval framework and used a hand-rolled BM25, opus-mt translation, and a different scorer.
  - The α-hybrid results.
  - The Qwen3-0.6B base-encoder and InfoNCE results.
  - mMARCO Phase 0/1 (64 concepts, 4-D heads).
  - The 2026-06-18 Phase-2 test run, superseded by the 2026-07-06 lemma-fallback run.
- **Current reference numbers:** `reports/eval_phase2/2026-07-06T180957Z__minicoil-v2__3056c00.md` (covered slice) and `reports/eval_all/` (all queries). Both predate the prefix-leak fix (1.15).

### Conventions (read before comparing tables)

- **Pair labels mean different things in different eras:**
  - **MLQA goal-branch scripts (May, after fix M9.1)** use the MTEB `<corpus>-<query>` order: `eng-spa` = Spanish queries against English documents.
  - **The eval framework and every mMARCO number** (`src/minicoil_v2/eval/retriever.py: PAIR_LANGS`) use `<query>-<corpus>`: `eng-spa` / `en-es` = English queries against Spanish documents.
- **Metrics:** nDCG@10 was the headline metric through Phase 1. The `main` gate (`eval/baselines/check.py`) uses MRR@10, which the code comment calls the official MS MARCO metric. Every table states which metric it shows.
- **"Teacher" = "mining encoder".** It only builds the distance matrix that picks triplets and is never run at inference. **"Input encoder"** is what the heads consume at train and inference time.

---

## 1. Timeline of phases

### 1.0 Early mE5-only pipeline (2026-03-17 to 2026-05-07, on `main`)

- **Setup:** input encoder `intfloat/multilingual-e5-small` (384-D); one `Linear(384,4)+tanh` head per concept (a9de51e).
- **Concept vocabulary:** MUSE EN-ES dictionaries, first clustered by connected components (2a02ac1), then by Louvain with recursive splitting (1e21418; merged 2026-05-07). A linear resolution schedule (`+0.5` instead of `*2`) gave +0.003 modularity, +0.002 weight-2 recall, +0.017 weight-1 recall, and 303 fewer 2-word fragments (5fcad89).
- **Teacher:** CLIP text (512-D), then unified with the input encoder as mE5-small (b6462dc, 2026-04-30).
- **Encoder choice** (`docs/architecture.md` DD-001): cross-lingual similarity on 30 EN/ES pairs gave mE5-small 0.9126, mE5-base 0.9190, mE5-large 0.9253, all at 100% retrieval accuracy. Chose mE5-small. An early 50-concept run reached 4-D cross-lingual cosine 0.988+ with triplet loss ~0.096 (DD-003).
- **Pipeline bug:** extraction ran before pruning, so every `scroll_concept()` in training returned zero sentences. Fixed by splitting extraction into scan and embed around prune (3a593a3); validated with 53,422 points from 200 articles per language.
- **Training defaults:** epochs 500 → 60 and `min_triplet_margin` 0.1 → 0.01 (fbdf346, 2026-05-07). The margin was restored to 0.1 later (83be5de, 2026-06-10).
- **Input encoding:** DD-004 chose sentence-level pooling. **Superseded by 1.1.**

### 1.1 Pooling strategy finding (diagnostics before 2026-05-11, committed 2026-05-13)

**Sources:** [pooling-strategy.md](research/pooling-strategy.md) and [pooling-verification.md](research/pooling-verification.md).

- **Question:** why did v2 training reject ~98% of triplets?
- **Setup:** mE5-small vectors over 4-8k Wikipedia EN+ES sentences. Cosine gap = mean intra-concept similarity − mean inter-concept similarity.

| Strategy | Gap | Cross-lingual gap | Rejection @ margin 0.10 |
|---|---|---|---|
| mE5-small, sentence pool | +0.017 | – | 93.5% |
| mE5-large, sentence pool | +0.016 | – | 93.7% |
| mE5-small, token pool (focal-word subwords) | **+0.116** | +0.123 | 44.8% |
| token pool, different-surface positives only | – | – | 42.1% |
| sentence pool, 10× data (1,000 sentences/lang) | +0.007 | – | 95.4% |

- **Independent re-run** (verification report): every claim reproduced. Sentence pool +0.021, token pool +0.114, cross-lingual +0.113 (all pairs) / +0.126 (different-surface pairs), mE5-large +0.021 with 92.0% rejection, 10× data exactly +0.007 / 95.4%.
- **Per-concept audit:** USE 99.6%, PLANT 99.8%, SPRING 96.6% rejection. PLANT "negatives" were Wikipedia footer blocks (a data artifact); SPRING negatives did track real senses (muelle/manantial/resorte).
- **Root cause found by the verification:** the v2 sampler drew two random within-concept sentences, with no hard mining. v1 mines hard by construction. So the high rejection rate was structural, not a bug.
- **Hygiene issue:** `constants.py` claimed the teacher was mE5-large (1024-D), but the stored vectors were 384-D (mE5-small).
- **Provenance caveat:** "~98%" came from the triplet-quality diagnostic on three concepts. The training loop never logged rejection rate.
- **Decision:** token-pool around the concept word, and add hard mining. Do not scale data or encoder size while pooling is wrong. Supersedes DD-004 (sentence pooling) on `main`.

### 1.2 4-D polysemy validation (spec 2026-05-11, findings committed 2026-05-13)

**Sources:** [4d-polysemy-validation.md](research/4d-polysemy-validation.md) and its design spec (internal).

- **Setup:** 20 hand-picked concepts with `Linear(384,4)+tanh` heads; one sentence universe (2,912 sentences: 1,527 EN, 1,385 ES) encoded twice, token-pool and sentence-pool. 30 epochs, 8k samples/epoch, lr 2e-3, dropout 0.05, hard mining.
- **Polysemy gap** (min intra-sense − max cross-sense): token pool positive on 4/9 concepts (type/write +1.221, work/obra +0.592, set/fixed +0.449, way/road +0.268); sentence pool positive on 3/9, all weak (largest +0.275). The type/write +1.221 is partly a language confound; a sense-only estimate is +0.45 to +0.60.
- **Monosemic intra-concept 4-D cosine:** token pool won all 11 concepts, e.g. water 0.740 vs 0.343, animal 0.727 vs 0.350.
- **4-D output norm:** mean 0.151 (token) vs 0.104 (sentence); p95 0.212 (token).
- **Layer "tax" vs raw 384-D input:** with token pool, monosemic intra-concept cosine drops 0.189 while polysemous cross-sense cosine drops 0.465 (+0.729 → +0.264), about a 2.5× return. Sentence pool pays 0.464 for 0.547, about 1.2×.
- **Cross-lingual top-1 same-concept** (chance = 5%): 36.0% token vs 13.2% sentence. An earlier small-sample 50% did not hold at scale.
- **Sentence-pool training health:** 80-90% validation rejection (vs ~38-40% for token pool) and 10-15% cross-lingual validation (vs 40%).
- **Decision:** the per-concept head on token-pooled input works for sense separation; sentence pooling is falsified. Retrieval was still untested and named "the decision-grade test".

### 1.3 MLQA goal run: mE5-small, 1,000 concepts (2026-05-13 to 2026-05-14)

**Source:** the goal-branch run log (commits 86698ef through 2e1878d). Autonomous agent sessions with reviewer checkpoints.

- **Setup:** 12,357 bilingual concepts (~21.8k EN words, ~25.5k ES words); 251,681 token-pooled mE5-small sentences; teacher = input encoder = mE5-small; 4-D heads with bilingual cosine triplet loss.
- **Benchmark:** MLQA (fits in 16 GB; parallel EN/ES Wikipedia). MIRACL-es (10.4M passages) and mMARCO (8.8M) were too large to fit locally. BM25 via `rank_bm25`, `TOKEN_RE=[^\W\d_]+`, k1=0.9, b=0.4.
- **BM25 test nDCG@10** (MLQA order `<corpus>-<query>`): eng-eng 0.6804, spa-spa 0.6186, eng-spa 0.2578, spa-eng 0.1665.
- **M4, training at scale:** 1,000 concepts selected by MLQA-validation coverage (98.6% of validation queries); 20.6 min on Apple MPS; validation loss 0.0118. Head output norm mean 0.063 (p95 0.105): **norm collapse** from cosine triplet + tanh.
- **M5, α-hybrid** `α·BM25 + (1−α)·miniCOIL`: the validation cross-lingual win (eng-spa +0.0246) **did not transfer to test** (eng-spa −0.0129, spa-eng −0.0124). The framing was rejected.
- **M5, standalone single formula:** BM25 over non-concept tokens + concept-level BM25 × learned 4-D cosine. `sim=cosine` won validation on all 4 pairs.

| MLQA test nDCG@10 | BM25 | v2 (pre-fix, bbccfc2) | v2 after fix M9.1 (e7808b5) | v2 after re-keying H1 (4f101c6) |
|---|---|---|---|---|
| eng-eng | 0.6804 | 0.6934 | 0.6934 | 0.6890 |
| spa-spa | 0.6186 | 0.6365 | 0.6365 | 0.6460 |
| eng-spa | 0.2578 | 0.3132 | 0.3969 | 0.5739 |
| spa-eng | 0.1665 | 0.2844 | 0.3792 | 0.5404 |

- **Fix M9.1 (pair names):** three scripts flipped the MTEB `<corpus>-<query>` order. Monolingual results were byte-identical before and after the fix.
- **H1 (re-keying):** re-key concept maps to all 12,357 bilingual concept ids, so untrained concepts score as concept-aggregated BM25 (`sim=1`); no retraining. Cross-lingual R@100 after H1: eng-spa 0.8596 vs translate+BM25 0.8874; spa-eng 0.8144 vs 0.8230. Conflict: the 4f101c6 commit message lists 0.3991 / 0.3792; the goal log marks those as pre-patch artifacts and gives the recheck values in the table.
- **Translate+BM25 (opus-mt), conflicting values:** 0.2494 / 0.1620 at M5b (bbccfc2) vs 0.6387 / 0.5761 from M9.1 on. Not reconciled in the sources. The script's usage example passes `--pair eng-spa --src-lang en --tgt-lang es`, which is reversed under the `<corpus>-<query>` order. Every later section uses 0.6387 / 0.5761 as the bar.
- **Generalization** (1,000-concept checkpoint, before M9.1): SciFact 0.6629 vs BM25 0.6653, Δ −0.0024, 95% CI [−0.0095, +0.0049], a tie. XQuAD-es 0.9496 vs 0.9436, Δ +0.0060, 95% CI [+0.0005, +0.0112], a significant win.
- **Pre-registered retrain M9.2** (concept selection by Wikipedia frequency): not executed, because the Qdrant volume was lost when the container runtime stopped. The frequency list overlapped 68.6% with the shipped list and lost 7.9 points of validation coverage (76.1% vs 84.0%).
- **Decisions:** one standalone scorer, never an α-hybrid (recorded as a binding learning in the branch's agent notes). Validation-only selection, with the test set treated as spent after the shipped result.

### 1.4 Formula levers H2/H3 and vocabulary diagnostics (2026-05-14 to 2026-05-15)

**Source:** the snapshot-branch run log (ccc864b, 4dcd23a, 92d8eae, 7bcd8ee).

- **H2, vocabulary vs scoring:** of queries that translate+BM25 hit at @100 but H1 missed, 87-88% are scoring-limited (eng-spa 321/363, spa-eng 350/402) and 12-13% vocabulary-limited. Contradicts M9.3's "vocabulary breadth is the ceiling".
- **Concept/literal weight sweep:** best at 1.0/1.0. w_con=8 cost −10.7 (eng-spa) and −11.9 points (spa-eng); concept-only (w_lit=0) scored 0.4312 / 0.3782.
- **Decoupled concept k1/b:** k1=1.5, b=0.75 gave eng-spa 0.5982 (+2.43 points) but spa-eng 0.5308 (−0.96 points); k1=0.9, b=0.75 gave 0.5940 / 0.5322. Neither reached translate+BM25.
- **Similarity distribution:** 23.0% (eng-spa) and 16.9% (spa-eng) of matched-concept cosines are negative.
- **H3: clipping negative similarities to 0** hurt eng-spa by 0.91 points. The negative cosines are sense penalties, not noise.
- **Similarity-off test (`sim=one`)** on the 1,000-concept checkpoint: the head adds +1.61 / +1.22 / **+0.31** / +2.35 points (eng-eng / spa-spa / eng-spa / spa-eng). The trained head barely moves cross-lingual scores.
- **Concept-eligibility ceiling** (min(en,es) ≥ N sentences): N=10 → 2,025 concepts, N=5 → 3,434, N=3 → 4,945. A "5k retrain" at the N=10 floor was impossible.
- **Coverage diagnostic** (all 10,504 cross-lingual test queries): median vocabulary coverage 0.714 (eng-spa) and 0.857 (spa-eng); zero-coverage queries 3 and 0. The "missing-token coverage" hypothesis was rejected on MLQA.

### 1.5 Qwen3 base-encoder swap and InfoNCE snapshot (2026-05-14 to 2026-05-15)

**Sources:** the snapshot-branch run log (4dcd23a, 43b7d59, b3d131b, 331f83a) and an MLQA target table on the perf branch.

> Correction to a common label: this snapshot used **Qwen3-Embedding-0.6B (1024-D) as both input and teacher encoder** (word-token pooled, `Linear(1024→4)`). Qwen3-**8B** (4096-D) appears later, as the cloud teacher collection in Phase 0/1 (1.7).

- **Motivation,** a silhouette study supplied by the team (language / sense silhouette): mE5-small word token +0.194 / +0.187; Qwen3-0.6B word token +0.154 / **+0.309**; Qwen3-0.6B sentence (last token) +0.059 / +0.094.
- **Triplet run:** 500 articles per language (137k sentences: 86k EN, 51k ES; 586k points). Embed 3h10m on Apple MPS; training 18 min, validation loss 0.0662.

| MLQA test nDCG@10 (`sim=cosine` unless noted) | mE5 (H1) | Qwen3 triplet | Qwen3 InfoNCE | InfoNCE, boost | Bar (translate+BM25) |
|---|---|---|---|---|---|
| eng-eng | 0.6890 | 0.6903 | 0.6603 | 0.6561 | – |
| spa-spa | 0.6460 | 0.6459 | 0.6229 | 0.5947 | – |
| eng-spa | 0.5739 | 0.5622 | 0.5297 | 0.5679 | 0.6387 |
| spa-eng | 0.5404 | 0.5338 | 0.5162 | 0.4773 | 0.5761 |

- **Sanity check:** `sim=one` columns were bit-identical across encoders (0.6729 / 0.6338 / 0.5708 / 0.5169).
- **Loss curves:** train 0.0272 / validation 0.0662, i.e. mild overfit, not underfit.
- **InfoNCE setup:** symmetric in-batch loss, opposite-language teacher-nearest positive, τ=0.1, batch size 64.
- **InfoNCE validation divergence (3.83 → 9.00) was an artifact.** On a shared pool, val/train = 1.005. The real cause was duplicate anchors from a ~40-sample held-out pool.
- **Mislabel risk:** a "miniCOIL v1" row in the snapshot goal log and `baseline_targets.json` (0.693 / 0.637 / 0.313 / 0.284) equals the v2 pre-fix MLQA numbers from bbccfc2 to three decimals. Treat it as mislabeled. Real v1 is English-only.
- **State at the snapshot (2026-05-15 evening):** encoder "locked" to Qwen3-0.6B; OUTPUT_DIM 4→16 back on the table; Tier-1 diagnostics planned (projection alignment, silhouette after the head, InfoNCE `sim=one`), with no recorded results.
- **Superseded:** from Phase 0 on, inference returned to mE5-small, with Qwen3 used only as teacher. No source records an explicit rationale for reverting the "locked" decision.

### 1.6 Eval framework and baselines gate (2026-05-25 to 2026-06-03)

**Sources:** the baseline-retrievers design spec, the eval experiment guide (now [docs/07-evaluation.md](07-evaluation.md)) and the MLQA baselines file (perf branch). Also the eval-framework branch (c837fc5) and a1fdbff.

- **What was built:** an approach-agnostic harness (`Retriever` protocol with `index` and `search`, auto-discovered retrievers). Guardrails: sha-verified frozen splits, a dataset-revision pin, a sealed test split (environment-variable unlock plus audit log), hardcoded win conditions, stamped reports.
- **Baselines:** FastEmbed `Qdrant/bm25`; FastEmbed `Qdrant/minicoil-v1` (English only); `translate-bm25` = NLLB-200-distilled-600M + BM25.
- **Locked MLQA test nDCG@10** (framework order `<query>-<corpus>`): BM25 0.7367 / 0.6375 / 0.3079 / 0.2542 (eng-eng / spa-spa / eng-spa / spa-eng); v1 eng-eng 0.7460; translate-bm25 eng-spa 0.6004, spa-eng 0.6589. Not comparable with 1.3 (different BM25, translator, splits).
- **Backbone finding:** v2 with concept blocks only scored eng-eng dev nDCG@10 0.177 vs v1's 0.746 (a1fdbff). Decision: add a BM25 lexical backbone inside the encoder, with hashed indices disjoint from concept indices. Known defect, flagged in code: the backbone was BM15 (no document-length normalization).
- **Indexing fix:** cap the FastEmbed encode batch at 16; padding to the longest document (~2.3k tokens) hung the run and caused out-of-memory errors (3059751).
- **Superseded:** MLQA as the primary benchmark. Collaborators asked for MMARCO numbers comparable to public baselines, and Phase 0 chose mMARCO as the leaderboard-comparable yardstick.

### 1.7 mMARCO Phase 0: concept-restricted gate, 64 concepts (2026-06-08)

**Sources:** the Phase-0 run log and the Phase-0 gate design spec (perf branch).

- **Setup:** teacher = a cloud Qdrant collection with 173 concepts, ~760k EN+ES sentences (ES 433k / EN 327k), 4096-D Qwen3 teacher vectors. Input encoder mE5-small token-pooled, 384-D → 4. Bilingual viability: 47 concepts with ≥2k sentences/lang, 64 with ≥1k, 26 with ≥5k; median min(EN,ES) 415.
- **Trainer change (bd8421d):** inputs are re-encoded with token-pooled mE5; the teacher vector (any dimension) feeds only the distance matrix. Tests pin the training input byte-for-byte to the inference representation.
- **mMARCO:** raw Google-translation TSVs; `dev.small` has 6,980 queries; gold sub-corpus 7,434 EN / 7,433 ES passages; cross-lingual gold = the same pid in the other language.
- **Coverage rule:** the query and its gold passage must share a trained concept, matched per language.

| Pair | Covered @ ≥2k (47 concepts) | Covered @ ≥1k (64 concepts) |
|---|---|---|
| eng-eng | 285 | 423 |
| spa-spa | 286 | 383 |
| eng-spa | 258 | 363 |
| spa-eng | 256 | 366 |

- **Decision:** the ≥1k floor (64 concepts). Splits: dev 50, val 110, test the remainder (203-263 per pair).
- **Locked test nDCG@10 bars:**

| Pair | Bar to beat | nDCG@10 |
|---|---|---|
| eng-eng | v1 (BM25 0.9323) | 0.9342 |
| spa-spa | BM25 | 0.8481 |
| eng-spa | translate-bm25 (BM25 0.3709) | 0.7446 |
| spa-eng | translate-bm25 (BM25 0.2356) | 0.8273 |

- Self-check: a baseline ties itself, so `baselines check` exits 1. This shows the gate discriminates.

### 1.8 mMARCO Phase 1: train 64 heads against the gate (2026-06-08 to 2026-06-10)

**Source:** the Phase-1 run log (perf branch).

- **Blocker fixed (4849675):** the cloud `concept_ids` were a different vocabulary vintage (cloud C-00001 = "bid", on-disk C-00001 = "open"). A rematch loader sources rows by on-disk surface forms; all 64 concepts had ≥100 sentences/lang (median min 804).
- **Run t1:** 64 heads, 60 epochs.
- **Scoring fix (fa01622):** cosine-trained heads emit tiny tanh values (~0.17), and those blocks displaced full-weight lexical terms. Fix: L2-normalize each block and scale it by tf-saturation.

| Validation nDCG@10 | t1 raw | after fa01622 | Phase-0 bar (test) |
|---|---|---|---|
| eng-eng | 0.7275 | 0.8414 | 0.9342 |
| spa-spa | 0.6682 | 0.7947 | 0.8481 |
| eng-spa | 0.2273 | 0.3429 | 0.7446 |
| spa-eng | 0.1890 | 0.2854 | 0.8273 |

- **Language-scoped backbone, reverted:** eng-spa 0.3429 → 0.2550 and spa-eng 0.2854 → 0.2441. Cross-lingual cognate overlap is genuine signal.
- **Sealed test (t1):** 0.8729 / 0.8252 / 0.3975 / 0.3227 (nDCG@10). `baselines check` **FAIL** on all 4 pairs.
- **t2 (120 epochs):** within ±0.005 of t1 on validation, loss plateaued ~0.16. The heads were converged, not undertrained.
- **Concept-lift isolation** (same backbone in both arms, BM15): +0.1285 / +0.1434 / +0.1461 / +0.1066. This **overturned** an earlier "concepts are neutral" claim, which had been confounded by comparison with external BM25.
- **Backbone swap to FastEmbed `Qdrant/bm25` (78373c1):** validation 0.8395 → 0.9210, 0.7991 → 0.8396, 0.3440 → 0.4374, 0.2840 → 0.3067. Same-language ≈ plain BM25 (−0.004 / −0.008); cross-lingual +0.17 / +0.11 over plain BM25 but far below translate-bm25 (validation 0.7601 / 0.7992).
- **Decision:** every remaining gap is a concept-**coverage** gap. Scale the concept vocabulary (Phase 2) and keep translate-bm25 as the cross-lingual bar.

### 1.9 mMARCO Phase 2: demand-driven 2,398 concepts, Qwen3-0.6B teacher, 8-D heads (2026-06-10 to 2026-07-06)

**Sources:** [phase2-scale-concepts-design.md](research/phase2-scale-concepts-design.md), the Phase-2 embed run log and the full-vocabulary handoff (now [full-vocabulary-run-plan.md](research/full-vocabulary-run-plan.md)); commits b57b193, cd88e9a, e70d0d4, 67000a7. Also `reports/eval_phase2/` and `reports/eval_final/`.

- **Demand measurement:** 13,960 query texts (6,980 qids × EN/ES); concepts selected on an 80% qid slice, with a 20% slice held out for the test carve. Found in the pre-release review: the split builder shuffled all covered qids instead, so only ~18% of each test split is from the held-out slice (see [07-evaluation.md](07-evaluation.md)).

| Matched-coverage target | Concepts | Absolute coverage (select) | Absolute coverage (held-out) |
|---|---|---|---|
| 80% | 1,479 | 0.597 | 0.541 |
| 90% (chosen) | **2,398** | 0.671 | 0.605 |
| 95% | 3,145 | 0.709 | 0.636 |
| 99% | 3,922 | 0.738 | 0.661 |
| All 12,357 | 12,357 | 0.746 | 0.742 |

- Vocabulary ceiling: about 74.6% of query content tokens. Names, numbers, and unlisted inflections ride the lexical backbone.
- Supply check: 2,242 of 2,398 concepts have ≥300 sentences/lang (median 3,266).
- **Teacher: Qwen3-Embedding-0.6B (1024-D)**, replacing Qwen3-8B, which was judged "only marginally better" (no numbers in the sources); 0.6B is ~10× cheaper to run and 4× smaller to store. Sentence-mode pooling (cd88e9a), max_seq_length capped at 512 (e70d0d4). The code default teacher on `main` is mE5-small; Qwen3 is a command-line override.
- **Embed run:** local Apple MPS managed 14.5 sentences/s (38-77 h projected), so the run moved to a single A10G cloud GPU: benchmark 739 sentences/s (batch 64), observed 340-505/s. A CUDA out-of-memory error at ~1.1M points (Qwen3's 32k native max_seq_length plus length-sorted batching) was fixed by the 512 cap. 400k articles, store cap 800 per concept per language. EN finished with 1,857,059 points and all 2,398 concepts filled; ES was at 2,274,084 points at the last log entry.
- **Training (`cm_fast_stack`):** 2,398 `Linear(384→8)+tanh` heads merged from parallel runs. Recipe as recorded in the handoff (the exact arguments are in the W&B run metadata): 80 epochs, train-epoch-size 2000, margin-scale 2.0, lr-patience 3, dropout 0.10, batched. (The Phase-2 spec had planned 60 epochs.) The mE5 input cache was 25 GB for 2,398 concepts.
- **Gate re-frozen on the Phase-2 covered slice (MRR@10):** per pair dev 200, val 2,000, test 3,491 / 3,341 / 3,145 / 3,248.

| Pair | Bar to beat | Bar MRR@10 | BM25 MRR@10 |
|---|---|---|---|
| eng-eng | v1 | 0.8921 | 0.8805 |
| spa-spa | BM25 | 0.7953 | – |
| eng-spa | translate-bm25 | 0.7028 | 0.3276 |
| spa-eng | translate-bm25 | 0.7741 | 0.2528 |

**Results (test MRR@10)**

| Run | eng-eng | spa-spa | eng-spa | spa-eng |
|---|---|---|---|---|
| 2026-06-18, surface matching (703b5f5, dirty tree) | 0.8850 (−0.0071) | 0.8314 (+0.0361) | 0.6975 (−0.0053) | 0.7041 (**−0.0700**) |
| 2026-07-06, + lemma fallback (3056c00): **published config** | 0.8889 (−0.0032) | 0.8398 (+0.0446) | 0.7109 (**+0.0081**) | 0.7154 (**−0.0588**) |

- Deltas in parentheses are vs that pair's bar. 2026-07-06 nDCG@10: 0.9049 / 0.8605 / 0.7428 / 0.7459; R@100: 0.9951 / 0.9877 / 0.9634 / 0.9628.
- **Gate status:** fails on eng-eng and spa-eng (2 of 4 pairs beat their bars). For the 2026-06-18 model, the handoff notes R@100 beat every comparator on all 4 pairs; the deficits were top-10 ranking only.
- **Significance (2026-06-18 run, spa-eng vs translate-bm25):** MRR@10 Δ −0.0700, CI [−0.0847, −0.0549], p≈0.001, a loss; R@100 Δ +0.0121, CI [+0.0033, +0.0209], a win. Recomputed per-query metrics match the reported ones (`reports/eval_final/`).

### 1.10 Batched GPU training performance (2026-06-16 to 2026-06-17)

**Sources:** [batched-gpu-training.md](research/batched-gpu-training.md) and [`scripts/perf/README.md`](../scripts/perf/README.md); commits 67000a7, 89a0e7a, 703b5f5.

- **What changed:** a vectorized semi-hard sampler plus a bucketed multi-concept trainer: stacked `[B,8,384]` weights, one `bmm` per step, summed per-concept loss, one backward pass and one custom Adam step with per-concept learning rate, GPU-resident input bank. Opt-in (`TrainSettings.batched=False` keeps the reference path).
- **Equivalence:** sampler 0 mismatches over 67,854 exhaustive cases; batched Adam vs torch Adam 4.44e-16; old path vs Milestone A 0.00e+00, A vs B ≤4.1e-7; bucket non-interference 0; end-to-end mean validation-loss Δ 0.00958 < 2× noise floor (0.01351).

| Setting | Speedup | Projected full vocabulary |
|---|---|---|
| CPU (B=16) | 2.6× | – |
| Apple MPS, B=24 | 7.0× | – |
| Apple MPS, B=48 | 7.5× | 1.6 h for 12k concepts |
| Sampler micro-benchmark | ~4-6× | – |

- **Reality check:** the synthetic 7× became **~2×** on a real 50-concept, ~80k-row bucket (sampling ~53%, augmentation re-encode ~19%, host marshaling ~17%, setup ~6%, GPU kernel ~4%). The benchmark had used a toy bucket, augmentation off, and a val_epoch_size mismatch.
- Follow-up (89a0e7a), bit-exact sampler rewrite: a sampling round went 11.9 → 3.3 ms, and a bucket went 290 → 217 s.
- `--mine-every N` (703b5f5): reuses mined triplets for N epochs. Off by default, because it changes training dynamics and must pass the MRR@10 gate.

### 1.11 Query-side stopword leak fix (2026-06-18)

**Sources:** commit 1d66d44 (perf branch) and the `XL_BACKBONE_WEIGHT` comment in `constants.py`.

- **Bug:** the backbone used the English stopword list for all languages, so Spanish query function words (de/la/que) leaked into English corpora as high-IDF noise. About 94% of spa-eng queries emitted them; it was the dominant failure at ranks 11-100.
- **Fix:** strip the query's own-language stopwords. Document side unchanged; a no-op for English queries.
- **Results:** spa-eng validation MRR@10 0.55 → 0.71; +0.158 R@100 on a 107k-document open-domain haystack.
- Superseded the earlier cross-lingual backbone downweight (0.5 had been the MRR optimum before the fix; default is now 1.0).
- **Open-domain harness** (bc1046f): `mmarco_open` mixes N non-gold distractors into the gold corpus. Validity gate: at N=0 it reproduces spa-eng validation MRR@10 0.70766 exactly.

### 1.12 Stopword-gated lemma fallback (code 2026-07, committed 2026-08-24)

**Source:** cef22a1 (`main`).

- **Mechanism:** a token that misses the surface-form lookup is lemmatized (simplemma) and looked up again, e.g. "gatos" → "gato", "injections" → "injection".
- **Gating:** both the token and its lemma are gated on stopwords. Without the gate, auxiliaries (is/are → be, es → ser → C-01977) fire a near-stopword concept on almost every text.
- **Effect:** improves every pair over the surface-only run (table in 1.9). eng-spa goes from a loss to a +0.0081 win.
- Opt-in through the `MINICOIL_V2_LEMMA_MATCH` env var or a constructor argument. `export-hf` writes `lemma_match` into `config.json` (default true), so a download reproduces the published config.

### 1.13 All-queries gate and coverage cliff (2026-08-24)

**Sources:** be3bb60 and 49893b7 (`main`).

- **Coverage cliff (published-model probe):** 5 of 6 cross-lingual probe queries rank the right document first. "court ruling witnesses" against a Spanish corpus returns **nothing**: ruling, witnesses, tribunal, and testigos are all outside the 2,398-concept vocabulary, and the one concept that fires (court/juzgado) does not match a document that says "tribunal". With a language-specific backbone there is no shared sparse index.
- **All-queries gate:** the covered-slice filter drops ~22% of dev queries, and cross-lingually ~40% of those share no sparse index with their gold. `--no-coverage-filter` splits over all 6,980 queries (200 / 2,000 / 4,780 per pair).

| Validation MRR@10, all queries | v2 | Covered slice | Δ vs bar |
|---|---|---|---|
| en-en | 0.8729 | 0.8849 | BM25 +0.0008, v1 −0.0110 |
| es-es | 0.7991 | 0.8139 | BM25 +0.0370 |
| en-es | 0.6588 | 0.7177 | translate-bm25 −0.0261 (was +0.0007) |
| es-en | 0.6596 | 0.7170 | translate-bm25 −0.0780 (was −0.0510) |

- On the wider slice, BM25 gains ~+0.03 cross-lingually (the dropped queries are cognate- and entity-heavy) while translate-bm25 loses ~0.03.

Split by coverage, validation MRR@10, v2 vs translate-bm25:

| Pair | Covered | Uncovered |
|---|---|---|
| en-es | 0.7138 vs 0.7139 | 0.4805 vs 0.5909 (R@100 0.96 → 0.57) |
| es-en | 0.7193 vs 0.7676 | 0.4499 vs 0.6322 (R@100 0.97 → 0.50) |

- **Size of the prize:** raising uncovered queries to covered performance is worth +0.055 (en-es) and +0.060 (es-en).

### 1.14 Published model (2026-08-24 to 2026-08-26)

**Sources:** b6b14d6, d5f8241, 62b686d, 27419a3, 03065d1, 1274991 (`main`).

- **`minicoil export-hf` output:** stacked heads in safetensors, `config.json` holding the concept id order and inference defaults, and both vocabulary files.
- **Tests:** byte-for-byte checks against the source checkpoint, with lemma matching on and off. Concept order is the silent-failure risk.
- **Vendored inference code:** the import closure only, so a user needs PyPI packages alone. Verified from a clean virtual environment.
- **License:** CC BY-NC 4.0, because the MUSE dictionaries ship in the repo. The most restrictive input governs.
- **Model card:** reports MRR@10 explicitly on the covered slice and states the coverage-cliff failure mode.

### 1.15 After publication: fixes before going public (2026-09)

**Sources:** the commits "fix(encoder): stop the mE5 prefix from firing concepts" and "docs: record the published checkpoint's teacher" in this repo.

- **Prefix leak:**
  - The bug: concept matching ran over `prefix + text`. "passage" (C-00098) and "query" (C-00023) are concept words, so every English text fired them, whatever it said. Spanish text was unaffected. The trainer's input path had the same bug.
  - The fix: match on the text alone and shift spans by the prefix length (`concept_match.match_concepts_after_prefix`), in the encoder, the trainer and the coverage scan. `encode_sparse` now defaults the prefix from `is_query` ("query: " for queries), the pairing the eval harness scores.
  - Validation MRR@10 on the published checkpoint, before → after: en-en 0.8845 → 0.8889, es-es 0.8142 (unchanged), en-es 0.7188 → 0.7304, es-en 0.7176 (unchanged).
  - The "before" re-run lands within 0.0011 of the committed validation report (0.8849 / 0.8139 / 0.7177 / 0.7170). The cause of the small difference was not investigated; the re-run used an in-memory Qdrant.
  - The published **test** numbers were measured with the leak. Re-run the test split before re-exporting the model.
- **Teacher provenance:** the docs said Qwen3-8B. The published checkpoint used **Qwen3-Embedding-0.6B in sentence mode (1024-D)**, per the Phase-2 run log.

---

## 2. Decisions and why

| Decision | Alternatives considered | Evidence | Where in code (`main`) |
|---|---|---|---|
| Token-pool the input around concept-word spans | Sentence pool; mE5-large; 10× data | Gap +0.116 vs +0.017; 4-D validation wins on 7/9 polysemous and 11/11 monosemic concepts (1.1, 1.2) | `token_pooling.py` (`pool_spans`), `encoder.py` |
| Hard semi-hard mining from the teacher distance matrix | Random within-concept pairs; margin sweeps | Random pairs gave 96-99% rejection; the gap problem is geometric (1.1) | `train_concept_layers.py` (`_pick_pair_mined`, `dynamic_min_margin`) |
| Bilingual cosine triplet loss | InfoNCE (τ=0.1, batch 64) | InfoNCE missed the bar and lowered monolingual scores (1.5) | `train_concept_layers.py` (`bilingual_cosine_loss`); InfoNCE only on the snapshot branch (43b7d59) |
| One standalone scorer, never α·BM25 + (1−α)·miniCOIL | α-hybrid; RRF fusion (24e3804) | α validation win flipped to a test loss (1.3) | `encoder.py` (backbone + concept blocks in one sparse vector) |
| BM25 lexical backbone inside the encoder for out-of-vocabulary tokens | Concept-only vectors | eng-eng 0.177 without the backbone; concepts-only 0.2482 (1.6, 1.8) | `encoder.py` (`BACKBONE_BASE = 1<<27`) |
| Reuse the FastEmbed `Qdrant/bm25` pipeline for the backbone | Hand-rolled BM15 | +0.02 to +0.09 validation nDCG@10; parity tests (1.8) | `encoder.py`, `tests/test_backbone_parity.py` |
| Unit-normalize concept blocks × BM25 tf-weight | Raw tanh output | Validation eng-eng 0.7275 → 0.8414 (1.8) | `encoder.py` (fa01622) |
| Keep cross-lingual backbone matching | Language-scoped backbone | Scoping cost −0.09 / −0.04 (1.8) | `encoder.py` |
| Decouple input (mE5-small, 384-D) from teacher (any dimension) | Input == teacher vector | Needed for the 4096-D / 1024-D teachers; byte-for-byte tests (1.7) | `train_concept_layers.py`, `concept_match.py`, `tests/test_trainer_input_path.py` |
| mE5-small stays the inference encoder | Qwen3-0.6B as base encoder ("locked" 2026-05-15) | Qwen3 swap regressed cross-lingual (1.5); reversal rationale not recorded | `constants.py` `INPUT_ENCODER` |
| Teacher Qwen3-Embedding-0.6B, sentence mode | Qwen3-8B (4096-D cloud); mE5-small | 8B "only marginally better"; 0.6B ~10× cheaper and 4× smaller (1.9) | `embed_wiki_sentences.py` (`--mining-encoder`, `--mining-pooling`) |
| OUTPUT_DIM 4 → 8 | 4; 16 | 4-D norm collapse and ranking compression; rationale in a code comment, no isolated ablation in the sources | `constants.py` `OUTPUT_DIM` |
| Demand-driven concept selection at the 90% mark (2,398) | Supply-driven (64); 80/95/99%; all 12,357 | Demand table (1.9) | `scripts/phase2_concept_demand.py`, `scripts/phase2_make_datadir.py` |
| mMARCO with MRR@10 as the gate metric | MLQA; MIRACL-es; BEIR | Phase-0 spec: public and leaderboard-comparable; collaborators had asked for MMARCO numbers (1.6, 1.7) | `eval/datasets/mmarco.py`, `eval/baselines/check.py` |
| Per-pair comparator: v1 (en-en), BM25 (es-es), translate-bm25 (cross-lingual) | BM25 everywhere | Team decision (2026-06-10) to keep translate-bm25 as the cross-lingual bar | `eval/baselines/check.py` `WIN_CONDITION` |
| Covered-slice gate plus all-queries gate | Covered slice only | Covered slice hides the deficit on uncovered queries (1.13) | `eval/coverage.py`, `eval/cli.py --no-coverage-filter` |
| Sealed test split, frozen sha-verified splits, audit log | Ad-hoc scripts | Prevents tuning on test (1.6) | `eval/splits/`, `eval/audit.py` |
| translate-bm25 via NLLB-200-distilled-600M with a disk cache | opus-mt (May scripts) | Reproducible translations (1.6) | `eval/retrievers/translate_bm25.py`, `_common.py` |
| Language-aware query stopwords | Cross-lingual backbone weight 0.5 | spa-eng validation 0.55 → 0.71 (1.11) | `encoder.py` `load_lang_stopwords` |
| Stopword-gated lemma fallback | Surface-only; ungated lemmatization | All 4 test pairs improve (1.12) | `concept_match.py`, `encoder.py` |
| Batched trainer, opt-in, with equivalence oracle | Reference loop only | Bit-exact tests; ~2× real speedup (1.10) | `batched_training.py`, `scripts/perf/` |
| Louvain clustering with linear resolution schedule | Connected components + salvage; `resolution*2` | +0.003 modularity, +0.017 weight-1 recall (1.0) | `build_concept_vocab.py`, `research/compare_resolution_schedules.py` |
| Publish as CC BY-NC 4.0 | MIT (base encoder license) | MUSE dictionaries ship in the repo (1.14) | `hf_export.py` |

## 3. Negative results and dead ends

**Pooling and training signal**
- **Sentence pooling:** +0.017 gap. mE5-large is no better (+0.016), and 10× data is worse (+0.007, 95.4% rejection) (1.1).
- **Training heads on sentence-pool input:** 80-90% validation rejection and lower values on every metric. "The head can compensate for weak input" is falsified (1.2).
- **Lowering the triplet margin to cure rejection:** the gap is geometric, so the margin does not help (1.1).
- **Snowball stemming as a coverage hack:** 5-7% gain with 10k+ collisions (goal run log, "What NOT to revisit").

**MLQA scoring levers (1.3, 1.4)**
- **α-hybrid:** validation eng-spa +0.0246 became test −0.0129. Even α tuned on test gained only ≤0.0005.
- **`sim=boost` (γ=1):** worse than BM25 on monolingual validation (eng-eng 0.7444 vs 0.7945).
- **Concept-weight up-weighting:** w_con=8 cost −10.7 / −11.9 points. Concept-only scoring: 0.4312 / 0.3782.
- **Decoupled concept k1/b:** a +2.43-point gain on one cross-lingual pair came with a −0.96-point loss on the other.
- **Clipping negative similarities to 0:** eng-spa −0.91 points.

**Encoders, losses, concept selection, and vocabulary breadth (1.3-1.5)**
- **Trained 4-D head on MLQA cross-lingual:** only +0.31 points over `sim=one` (eng-spa).
- **Qwen3-0.6B as base encoder:** cross-lingual −0.0117 / −0.0066 with monolingual flat, despite better silhouettes.
- **InfoNCE:** missed the translate+BM25 bar by −0.109 (eng-spa, cosine) and dropped monolingual scores (eng-eng 0.6603 vs 0.6903 with triplet loss).
- **Frequency-sorted reselection (pre-registered, never run):** predicted to fail; 68.6% overlap and −7.9 points of coverage.
- **"Vocabulary breadth is the ceiling" on MLQA:** retracted twice, once by H2 (scoring-limited 87-88%) and once by the coverage diagnostic (median coverage 71% / 86%).
  - **Tension with Phase 2:** on mMARCO with 2,398 concepts, uncovered queries carry the whole cross-lingual deficit (1.13). The vocabulary size (12,357 aggregated vs 2,398 trained) and the benchmark both differ, so the two findings are not directly contradictory, but they point in different directions.

**mMARCO Phase 1 scoring (1.8)**
- **Raw tanh concept blocks:** eng-eng fell below a 2-concept smoke run (0.7275 vs 0.8437).
- **Language-scoped backbone:** eng-spa 0.3429 → 0.2550 and spa-eng 0.2854 → 0.2441. Reverted.
- **More training (t2, 2× epochs):** ±0.005; loss plateaued ~0.16.
- **BM15 backbone (no length normalization):** ~5-8 points below real BM25 on same-language pairs.
- **64 concepts:** cannot rank cross-lingually against full query translation (sealed test 0.3975 / 0.3227).

**Infrastructure and later fixes (1.9-1.13)**
- **Local Qwen3-0.6B embedding on Apple MPS:** 14.5 sentences/s, 38-77 h projected. Infeasible.
- **Qwen3 at native 32k max_seq_length:** CUDA out-of-memory at ~1.1M points.
- **Synthetic training benchmark:** promised 7×, delivered ~2× on a real bucket.
- **Ungated lemmatization:** auxiliaries fire a near-stopword concept on almost every text (reason for the gate).
- **Cross-lingual backbone downweight (0.5):** a crude proxy, superseded by the stopword fix.

## 4. Open questions and next steps (as recorded)

**From the goal and snapshot branches (May 2026)**
- No recorded result for:
  - The Tier-1 diagnostic battery: projection alignment, silhouette after the head, InfoNCE `sim=one`.
  - The Tier-2 decision matrix: MUSE translation-pair positives vs larger head capacity.
- 4-D magnitude calibration vs BM25 term weights. Handled at inference by fa01622; never addressed in the loss.
- Named-entity passthrough for out-of-vocabulary capitalized tokens: proposed, not implemented.
- The original eval plan (an internal draft) has no recorded run:
  - BEIR (5 datasets) vs v1's published table.
  - MIRACL-es dev vs published numbers; the sparse bar to beat was BGE-M3 sparse 38.8.
  - XPQA as a secondary cross-lingual set.

**From mMARCO Phases 0-2**
- **Truncation edge:** the coverage filter matches the full passage, but the encoder truncates at 256 tokens.
- **Phase-2 spec:** supply floor (≥300 sentences/lang), concept-count target, and long-term storage of teacher vectors.
- **Full-vocabulary run** ([plan](research/full-vocabulary-run-plan.md)). Target: train all 12,357 concepts and close the es-en gap (−0.0700 MRR@10 at that time). Recorded decisions:
  - Teacher choice. Recommendation: Qwen3-0.6B, to reproduce the published recipe.
  - Embed `--max-articles`: the rare-concept tail will not fill to 800 from 400k articles.
  - Pinning the exact train recipe.
  - Eval corpus. Recommendation: gold pool as primary; N=1M open-domain; full 8.8M only with a sharded indexer, which is not implemented.
  - Reusing the locked Phase-2 splits, which may **understate** a larger model's advantage, vs rebuilding them and re-running all baselines.
  - Eval encoding is CPU-bound (119 docs/s on CPU vs 62 on Apple MPS), so a GPU does not speed up indexing.

**After publication**
- A matching fallback or vocabulary expansion for uncovered queries; the measured prize is +0.055 en-es and +0.060 es-en (be3bb60).
- Re-run the test split after the prefix-leak fix before re-exporting (fae6b87).

**Not ablated anywhere in the sources**
- Phase 2 uses **sentence** pooling for the Qwen3 teacher, although the May silhouette study scored Qwen3 sentence pooling lowest (+0.094 sense silhouette) and word-token pooling highest (+0.309).
- 8-D vs 4-D heads at a fixed recipe.

## 5. Notes on conflicts between sources

- **Translate+BM25 on MLQA:** 0.2494 / 0.1620 (bbccfc2) vs 0.6387 / 0.5761 (e7808b5 onward). Not reconciled; see 1.3.
- **H1 numbers:** the 4f101c6 commit message gives 0.3991 / 0.3792; the goal log gives 0.5739 / 0.5404 and marks the former as pre-patch artifacts.
- **The "miniCOIL v1" MLQA row** (0.693 / 0.637 / 0.313 / 0.284) equals the v2 pre-fix numbers. Treat it as mislabeled (1.5).
- **Concept counts:** the Phase-0 spec says "the full ~19.5k-concept train"; the handoff and goal log give a pruned vocabulary of 12,357 (pruned from 79,652).
- **Teacher in the docs:** older docs and commit 15fb34a said Qwen3-8B (4096-D); the published checkpoint used Qwen3-0.6B (1.15). The code default is mE5-small.
- **Training sampler:** the initial commit 0ca6a47 says "semi-hard mining", but the May verification found a random within-concept sampler in use. Hard mining was re-added during the polysemy work.
- **Held-out test carve:** the Phase-2 spec reserves the 20% held-out qid slice for test; the frozen test splits are a random carve of all covered qids (~18% held-out).
- **Gate metric:** nDCG@10 in the eval guide and the Phase 0/1 logs; MRR@10 in the `main` gate code. The switch commit is not identified in the sources.
- **An internal repository overview** (dated 2026-06-04, since removed) said v2 "does not yet strictly beat BM25 on nDCG@10 cross-lingual at test scale". That reflects the α-hybrid result. The goal log's standalone and H1 numbers beat BM25 on all MLQA pairs.

## 6. Source index

Branches and commit shas below refer to the internal development history and are not published. Branch tips: goal = `goal/cross-lingual-minicoil-v2` @ 2e1878d; snapshot = `snapshot/qwen3-infonce-2026-05-15` @ 331f83a; perf = `perf/batched-gpu-training-full` @ ef73cec; eval-framework = `feat/eval-framework` @ c837fc5; fastembed = `feat/fastembed-wrapper` @ b6b565b. `feat/mMARCO-eval` @ 3056c00 was merged. A "Ported to" path means the note now lives in this repo.

| Fact cluster | File | Branch | Commit(s) |
|---|---|---|---|
| Early pipeline, encoder choice, Louvain, scan/embed split, training defaults | `docs/architecture.md`; commit bodies | main | a9de51e, 0ca6a47, 2a02ac1, 1e21418, 5fcad89, 3a593a3, b6462dc, fbdf346, 83be5de, 15fb34a, 32df0dd |
| Pooling strategy | `docs/findings-pooling-strategy.md` → ported to [research/pooling-strategy.md](research/pooling-strategy.md) | goal, snapshot (identical) | 86698ef |
| Verification re-run, v1 vs v2 sampler, constants mismatch | `docs/findings-verification-report.md` → ported to [research/pooling-verification.md](research/pooling-verification.md) | goal, snapshot | 86698ef |
| 4-D polysemy validation | `docs/findings-4d-polysemy-validation.md` → ported to [research/4d-polysemy-validation.md](research/4d-polysemy-validation.md); design spec `docs/superpowers/specs/2026-05-11-4d-polysemy-validation-design.md` | goal, snapshot | 86698ef |
| MLQA BM25, α-hybrid, standalone, M6, M9-M9.3 | `docs/goal-log.md` (586 lines) | goal | 86698ef, a3fd3e1, 3a6cf27, bbccfc2, e7808b5, defbd69, 2e1878d |
| H1, H2, H3, k1/b, sim=one, coverage diagnostic | `docs/goal-log.md` (1,319 lines; first 586 identical to goal) | snapshot | 4f101c6, ccc864b, 4dcd23a, 92d8eae, 7bcd8ee |
| Qwen3-0.6B swap, InfoNCE, silhouette study, Tier-1/2 plan | `docs/goal-log.md`; `src/minicoil_v2/train_concept_layers.py` (InfoNCE path) | snapshot | 43b7d59, b3d131b, 331f83a |
| MLQA target table (includes the mislabeled v1 row) | `data/eval/baseline_targets.json` | perf | ef73cec |
| Original eval plan (BEIR / MIRACL-es / MLQA) | `eval-issue-draft.md` | goal, snapshot | 86698ef |
| Working principles | `PRINCIPLES.md` | snapshot, perf | 331f83a, ef73cec |
| Agent learnings (no α-hybrid; encoder roles) | `.claude/CLAUDE.md` | goal/snapshot; perf | 86698ef; ef73cec |
| Eval framework design and MLQA baselines | `docs/superpowers/specs/2026-05-26-baseline-retrievers-design.md`, `docs/eval-experiment-guide.md` (folded into [07-evaluation.md](07-evaluation.md)), `data/eval/baselines.json` | perf | dfa3ed8, ac43349, 86f3240, a1fdbff, 3059751 |
| Eval framework PR (code only) | `src/`, `tests/` | eval-framework | c837fc5 |
| Phase 0 gate | `docs/phase0-goal-log.md`, `docs/superpowers/specs/2026-06-08-phase0-mmarco-baseline-gate-design.md`, `data/eval/baselines_mmarco.json` | perf | f1cd5c0, bd8421d, 39203fe, 4dad804, 3072341 |
| Phase 1 training and diagnostics | `docs/phase1-goal-log.md` | perf | 4849675, fa01622, 0d6afb9, 1cac6b4, 9e6d02c, 4b954ad, c6cef74, 5d93653, 78373c1, b9c8e99 |
| Phase 2 design and demand table | `docs/superpowers/specs/2026-06-10-phase2-scale-concepts-design.md` → ported to [research/phase2-scale-concepts-design.md](research/phase2-scale-concepts-design.md) | perf | acf7bfe, b57b193 |
| Phase 2 embed run (internal: holds infrastructure details, not ported) | `docs/phase2-embed-run-log.md`, the cloud GPU runbook | perf | ef73cec, e3c426c, cd88e9a, e70d0d4, 41c47c4 |
| Full-vocabulary plan and Phase-2 recipe | `FULL_VOCAB_RUN_HANDOFF.md` → condensed into [research/full-vocabulary-run-plan.md](research/full-vocabulary-run-plan.md) | perf | ef73cec |
| Batched training performance | `docs/perf-batched-gpu-training.md` → ported to [research/batched-gpu-training.md](research/batched-gpu-training.md); `scripts/perf/` → ported | perf | 67000a7, 89a0e7a, 703b5f5 |
| Stopword leak fix and open-domain harness | commit bodies; `src/minicoil_v2/constants.py` comment | perf; main | 1d66d44, bc1046f; 3056c00 (squash, empty body) |
| Phase 2 results, significance, verification | `reports/eval_phase2/*.md`, `reports/eval_final/*.json`, `data/eval/baselines_mmarco_phase2.json` | main | 9f59103, 9f62c27 |
| Lemma fallback | commit body | main | cef22a1 |
| Coverage cliff and all-queries gate | commit bodies; `reports/eval_all/`, `data/eval/baselines_mmarco_all.json` | main | 49893b7, be3bb60 |
| Publication | commit bodies; `src/minicoil_v2/hf_export.py` | main | b6b14d6, d5f8241, 62b686d, 27419a3, 03065d1, 1274991 |
| FastEmbed wrapper probe (173-concept, 4-D model) | `research/20260612_fastembed_integration.py` | fastembed | b6b565b |
| Prefix leak; teacher provenance | commit bodies | this repo | — |
