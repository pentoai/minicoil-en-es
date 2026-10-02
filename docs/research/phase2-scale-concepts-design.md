# Phase 2 — Scale concept coverage (demand-driven vocabulary)

> Archived design spec (2026-06-10), kept as approved. It was executed with these
> deviations: training used 80 epochs, `--train-epoch-size 2000`, `--margin-scale 2.0`,
> `--lr-patience 3`, dropout 0.10 and the batched trainer (not the 60-epoch Phase-1
> config); the embed ran on a single cloud GPU because the local projection exceeded the
> 4h gate; and the frozen test splits were carved from all covered qids, not restricted
> to the 20% held-out slice (component D). Outcomes: [research log](../research-log.md)
> section 1.9.

## Why

Phase 1 (64 concepts) established, with a baseline-parity backbone (commit 78373c1):

- Same-language: concept blocks are a near-exact replacement for the lexical terms
  they displace (v2 within 0.004-0.008 of plain bm25 on val); they add ~nothing
  beyond, because 64 concepts cover almost none of any query.
- Cross-lingual: concepts add +0.17/+0.11 nDCG over plain bm25 — genuine bridge value
  no lexical term provides — but 0.44/0.31 vs translate-bm25's 0.76/0.80.
- Every remaining gap on all 4 pairs is a concept-COVERAGE gap.

Phase 2 scales coverage. Thesis: as concept coverage of query content words grows,
v2 cross-lingual approaches v2 same-language quality (which sits at bm25-level
already), putting the translate-bm25 bar in reach. The bar stays translate-bm25
(team decision 2026-06-10).

## Decisions (approved 2026-06-10)

1. **Demand-driven concept selection.** Train the minimal concept set that covers the
   vast majority of eval query content words — not the concepts Wikipedia happens to
   supply (Phase 0/1 logic, which produced 64 concepts and a tiny covered slice).
2. **Mining encoder: Qwen3-Embedding-0.6B** (1024D) replaces Qwen3-8B (4096D). The 8B
   was validated as only marginally better; 0.6B is ~10x cheaper to run and 4x smaller
   to store, buying scale. Trainer is already dim-decoupled (input fixed at mE5-384).
3. **Compute: local-first.** Benchmark throughput; if the embed run projects >4h
   locally, STOP and report — then consider a cloud GPU.
4. **Sealed test stays sealed** until the scaled model is ready (no intermediate
   test-split runs).

## Measured demand (scripts/phase2_concept_demand.py, read-only, 2026-06-10)

Query pool: all 6,980 judged dev.small qids x {en, es} = 13,960 texts. Content tokens
per fastembed's own stopword lists (english/spanish). Leakage guard: concepts selected
on the 80% slice `sha1(qid)%5 != 0`; the 20% held-out slice is reserved for the
Phase-2 test carve and only used to verify selection generalizes.

| matched-coverage target | concepts | abs coverage (select) | abs coverage (held-out) |
|---|---|---|---|
| 80% | 1,479 | 0.597 | 0.541 |
| 90% | 2,398 | 0.671 | 0.605 |
| 95% | 3,145 | 0.709 | 0.636 |
| 99% | 3,922 | 0.738 | 0.661 |
| ceiling (all 12,357) | 12,357 | 0.746 | 0.742 |

Key facts:
- **Vocab ceiling is ~74.6%**: ~25% of query content tokens are OOV to the full
  on-disk vocabulary (names, numbers, unlisted inflections). They ride the backbone
  (cognates only, cross-lingually). This caps how far concept scaling alone can go.
- Demand is concentrated: 4,220 distinct concepts ever fire; ~2.4k cover 90% of
  what's coverable.

**Default target: the 90% mark (~2,400 concepts)**, walked down by supply (below).
The 90->99% tail (1,500 more concepts) buys only +6.7 pts absolute coverage.

## Components

### A. Supply scan (CPU-only, runs first)
Re-run `minicoil scan` over Wikipedia EN+ES with the on-disk vocabulary, restricted to
the demand-ranked concept list. Output: per-concept per-lang sentence counts.
Selection rule: walk the demand-ranked list to the 90% mark; drop concepts below a
supply floor (default >=300 sentences/lang, cf. Phase-1 heads trained fine on ~400);
report dropped concepts and the resulting effective coverage. Frozen artifact:
`data/eval/mmarco/phase2_concepts.json` (ranked list + counts + coverage numbers).

### B. Embed (the data investment, 4h local gate applies)
`minicoil embed` with: on-disk vocab (concept_ids aligned by construction — kills the
vocab-vintage rematch workaround), mining = Qwen3-Embedding-0.6B (1024D), target
~800-1500 sentences/concept/lang (decided by supply scan + throughput benchmark),
into a NEW local Qdrant collection `minicoil_sentences_phase2`. Cloud cluster
untouched. Before launching: benchmark 0.6B sentences/s locally on MPS and project
total wall time; >4h -> STOP and report (cloud GPU decision).
Storage estimate: ~3-6M sentences x 1024D fp32 = 12-24GB on-disk vectors (local
docker Qdrant, on-disk storage mode).

### C. Train at scale
`minicoil train` on the selected concepts from the new collection (no rematch path).
Config: Phase-1 t1-equivalent (60 epochs converged identically to 120 in Phase 1).
Embarrassingly parallel per concept; ~80min/64 concepts on MPS => ~50h single-process
for 2.4k, so shard across N parallel processes (MPS allows ~4-6 concurrent small
trainings) or batch overnight. Output: `data/concept_models_phase2/`.

### D. Phase-2 gate freeze
Re-run the coverage filter with the Phase-2 concept set -> new covered slice (much
larger and more representative than Phase 1's 363-423/pair). Carve splits with the
20% held-out reserve as test. Re-lock baselines (bm25, minicoil-v1, translate-bm25)
on the new slice into `data/eval/baselines_mmarco_phase2.json`. Phase-0/1 frozen files
stay UNTOUCHED (they are the Phase-1 record).

### E. Evaluate
Val first, iterate if needed; one sealed-test run at the end via
`minicoil eval baselines check`. Also re-run the concept-lift isolation at scale —
the decomposition (backbone / concepts / full) is the result that matters
scientifically, whatever the gate says.

## Cleanup enabled by this phase
- Delete `scroll_concept_rematch` + `viable_cloud_buckets.json` path once the new
  collection exists (alignment by construction).

## Open questions for review
1. Supply floor (>=300/lang default) and sentences/concept target (800-1500) — final
   numbers set after the supply scan + throughput benchmark, reported before embed.
2. Concept-count target: 90% mark default; 80% mark (~1.5k) is the fallback if the
   embed projects too long even on a cloud GPU.
3. Where the 0.6B mining vectors live long-term (local-only is fine for Phase 2; a
   cloud collection is a later ops decision).
