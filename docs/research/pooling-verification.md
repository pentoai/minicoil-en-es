# Verification report: pooling diagnostic re-run

> Archived research note, kept as written apart from editing for publication.
> Date: May 2026. Outcome: hard mining was restored in `BilingualSampler`
> (`_pick_pair_mined`); the constants mismatch was fixed. Line numbers below refer to the
> code at the time.

An earlier diagnostic summary ([pooling-strategy.md](pooling-strategy.md)) explained why
miniCOIL v2 training was hitting ~98% triplet rejection. This report verifies it end to end
by re-running every experiment (not just reading the methodology), and records what
reproduced and what did not.

## Headline

**Every numerical and qualitative claim in the prior summary reproduces.** No falsifications.
Several numbers land within sampling variance; the 10x-data result reproduces exactly.

The load-bearing finding is **not** the rejection rate itself. It is that v2's
random-pair sampler is structurally different from v1's hard-mining sampler, and v2's
"path forward #2 (hard mining)" is not a hyperparameter tweak; it is restoring a v1
design decision that v2 abandoned.

## Reproduction table

All re-runs on a clean local Qdrant instance seeded from the project's
existing `minicoil_sentences` collection (53,422 points, 384D mining vectors).

| Claim | Summary | Re-run | Verdict |
|---|---|---|---|
| Sentence-pool gap (mE5-small) | +0.017 | +0.021 | reproduces (sampling variance) |
| Token-pool gap (mE5-small) | +0.116 | +0.114 | reproduces |
| Token-pool gap, cross-lingual (all pairs) | +0.123 | +0.113 | reproduces (diagnose_word_token_pooling) |
| Token-pool gap, cross-lingual (different-surface) | +0.123 | +0.126 | reproduces (validate_token_pooling) |
| Lexical control (different-surface positives) | +0.116 | +0.121 | reproduces |
| Lexical control rejection @ 0.10 | 42.1% | 42.4% | reproduces |
| mE5-large sentence-pool gap | +0.016 | +0.021 | reproduces |
| mE5-large rejection @ 0.10 | 93.7% | 92.0% | reproduces |
| 10x data (sentence pool, 1000/lang) gap | +0.007 | +0.007 | reproduces exactly |
| 10x data rejection @ 0.10 | 95.4% | 95.4% | reproduces exactly |
| Phase-5 audit USE rejection | 99.6% | 99.6% | reproduces |
| Phase-5 audit PLANT rejection | 99.8% | 99.8% | reproduces |
| Phase-5 audit SPRING rejection | 96.6% | 96.6% | reproduces |
| PLANT format-artifact pattern | claimed | confirmed | reproduces (qualitative) |
| SPRING polysemy captured by sampler | claimed | confirmed | reproduces (qualitative) |

### Methodology spot-check

The three diagnostic scripts were code-reviewed before re-running:

- `diagnose_mining_vectors.py`: scrolls stored mining vectors, intra/inter cosine,
  triplet rejection vs margin. Uses `frozenset(concept_ids)` for multi-concept points.
  Sound.
- `diagnose_word_token_pooling.py`: tokenizes with `offset_mapping`, locates the
  target word's character span, averages last-hidden-state subword tokens overlapping
  the span, L2-normalizes. Sound.
- `validate_token_pooling.py`: splits intra-concept pairs into (A) same surface word,
  (B) different surface word, (B-xl) cross-lingual different word, (C) inter; reports
  the "honest" concept gap as (B) - (C). Sound.

## Qualitative reproductions

**PLANT format artifacts.** The audit's PLANT section shows `sl_neg` sentences are
repeatedly Wikipedia metadata blocks: *"Plantas descritas por Linnaeus / Plantas
descritas en 1753"*, *"Cereales / Plantas descritas por"*, *"Enlaces externos /
Plantas bulbosas"*. The sampler's "different plant" separation is footer-vs-prose, not
botanical-vs-industrial. A data-quality artifact, not concept discrimination.

**SPRING polysemy.** Anchors are `spring`/`primavera` (season sense, dominant in
Wikipedia). Negatives are `muelle`/`muelles`/`manantial`/`manantiales`/`resorte` (the
other Spanish surface words for spring's other senses: dock, water spring, mechanical
spring). The sampler IS picking up polysemy structurally via lexical-form separation.
This explains why SPRING rejection (96.6%) is lower than USE (99.6%) and PLANT (99.8%):
SPRING has real lexical-sense distinctions, USE/PLANT mostly do not.

## The real result: v1 vs v2 sampler architecture

The summary frames the ~98% rejection as a problem to solve via path #1 (lower
margin), path #2 (hard mining), or path #3 (rethink architecture). Reading the v1
codebase changes how to think about path #2.

**v1 mines hard by construction:**

- Positive pool: top-20 most similar sentences (cosine on `mxbai-embed-large-v1`)
- Negative: closest sentence with `neg_sim < pos_sim AND neg_sim > -1.5` (hardest
  semi-hard); fallback to least similar
- Result: triplets are already near the margin, model learns useful gradient

**v2 samples randomly within concept:**

- `BilingualSampler._pick_pair` (`src/minicoil_v2/train_concept_layers.py:355-366`):
  draw two random in-language sentences, label the closer one "positive", the further
  one "negative", reject if `|d_x - d_y| < min_margin=0.1`
- No across-concept negatives. No similarity-conditioned selection.
- Result: most random within-concept pairs have `cos(a,n) - cos(a,p) ~ 0`. The 96-99%
  rejection rate is a direct consequence, not a sampler bug.

So v2's "path forward #2 (hard mining)" is not a tweak. It is restoring a v1 design
decision. The question is not "should we add hard mining?", it is "why was random
sampling chosen over v1's mining strategy?" - and the answer is in `docs/architecture.md`,
which documents the per-concept refactor but does not justify dropping hard mining.

The training-data layer is also wrong for hard mining as written: v2 stores embeddings
in Qdrant with concept-id payload but with no within-concept similarity index. Adding
hard mining would require either pre-computing similar-sentence indices or running a
nearest-neighbor pass inside the training loop. Either is feasible, but it is not
a one-line change.

## Separate finding: constants / data mismatch

`src/minicoil_v2/constants.py:8` declares `MINING_ENCODER = "intfloat/multilingual-e5-large"`
with a comment "(dimension = model output, 1024 for large)". The stored Qdrant vectors
in `minicoil_sentences` are 384D, which is e5-small dimensionality, and `XLMRobertaModel`
output for e5-large would be 1024D.

`embed_wiki_sentences.py` accepts `--mining-encoder` as a CLI flag, so the most likely
explanation is that the actual embed run used `--mining-encoder intfloat/multilingual-e5-small`
(the docstring even shows this exact override at line 14). This means:

- The constant default is misleading - the running system uses e5-small.
- Anyone reading `constants.py` will assume mining is 1024D and reach for the wrong
  encoder when diagnosing.
- The "mining" name is itself confusing in v2 because the sampler does not actually
  mine; it samples randomly within concept (see above).

This is a documentation/constants hygiene issue, not a training bug. But it tripped up
the original diagnostic (the e5-large baseline was claimed at +0.016, but the stored data was
already e5-small; the vectors were re-encoded explicitly to test the encoder swap).

## What the summary got slightly wrong

The summary says "~98% triplet rejection in training." Training output
(`data/concept_models/training.log`) does not actually emit per-batch rejection rate;
the `_epoch_rejections` counter at `train_concept_layers.py:392-393` is computed but
never logged. The "~98%" comes from `diagnose_triplet_quality.py` runs on three
hand-picked polysemous concepts (USE, PLANT, SPRING), not from the training loop
directly. Same sampler, same data shape, so the comparison is valid in kind, but the
provenance should be stated honestly.

## Operational conclusion (carried forward from summary, now backed by re-run)

1. Pooling strategy dominates: moving aggregation from full-sentence to token-level
   (around the concept word) yields ~7x improvement in concept-discrimination gap at
   no cost. This is the cheapest available win.
2. Scaling data does not help while pooling is wrong; it makes the gap worse
   (+0.017 to +0.007).
3. Scaling encoder size does not help while pooling is wrong; e5-large lands at the
   same +0.021 as e5-small.
4. v1-style hard mining is the structural fix for high rejection rates, not margin
   tuning. Adding it requires a within-concept similarity index, not just a sampler
   flag.

## Files referenced

- [pooling-strategy.md](pooling-strategy.md) (the summary being verified)
- `research/diagnostics/diagnose_mining_vectors.py`
- `research/diagnostics/diagnose_word_token_pooling.py`
- `research/diagnostics/diagnose_mining_with_e5_large.py`
- `research/diagnostics/validate_token_pooling.py`
- `research/diagnostics/test_more_data_hypothesis.py`
- `research/diagnostics/diagnose_triplet_quality.py`
- `src/minicoil_v2/train_concept_layers.py` (BilingualSampler, at the time of writing)
- `src/minicoil_v2/constants.py` (mining encoder mismatch, since fixed)
- `data/concept_models/training.log` (no rejection-rate emit)
- Re-run logs and the triplet audit were kept locally and are not part of the repo.
