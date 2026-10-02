# Finding: pooling strategy dominates encoder size and data volume

> Archived research note, kept as written apart from editing for publication.
> Date: May 2026. Outcome: token pooling became the input representation
> (see `docs/architecture.md` DD-005). Context: [research log](../research-log.md) section 1.1.

For extracting concept-level signal from a multilingual encoder over natural
text, the aggregation strategy matters more than the encoder size or the number
of sentences per concept. Empirical measurements on Wikipedia EN+ES, comparing
the cosine gap between same-concept and different-concept pairs:

## 1. Sentence pooling collapses concept signal

mE5-small full-sentence pooling: **gap +0.017** (essentially noise). The
pooled sentence vector is dominated by the article's topic (sports, biology,
politics) rather than the concept word it contains. Concept signal is diluted
across the surrounding tokens.

| Strategy | Gap (intra − inter) | Triplet rejection @ margin 0.10 |
|---|---|---|
| mE5-small, sentence pool | +0.017 | 93.5% |
| mE5-large, sentence pool | +0.016 | 93.7% |

## 2. Token pooling around the concept word recovers the signal

Averaging the last-hidden-state vectors of the subword tokens that make up
the target word, on the same encoder:

**Gap +0.116 — 7× larger.** Holds when the positive is forced to use a
different surface word from the anchor (cat vs kitten, or cat vs gato
cross-lingual), ruling out lexical identity as the driver.

| Strategy | Gap (intra − inter) | Cross-lingual gap | Rejection @ 0.10 |
|---|---|---|---|
| mE5-small, token pool | +0.116 | +0.123 | 44.8% |
| mE5-small, token pool, different-surface positives only | — | — | 42.1% |

## 3. More data does not compensate for a representation problem

Scaling sentences per concept from 100 to 1,000 (10×) with sentence pooling:

| | 100/lang | 1000/lang |
|---|---|---|
| Gap | +0.017 | **+0.007** (smaller) |
| Rejection @ 0.10 | 93.5% | **95.4%** (worse) |

Adding samples introduces dispersion in topic dimensions, not concept
dimensions. Intra- and inter-concept distributions converge as topic noise
grows, and the gap shrinks.

## 4. Multilingual encoders align at the token level, not only at the sentence level

Cross-lingual token-pooled gap (+0.123) is slightly larger than the
monolingual gap (+0.116). The assumption that bilingual concept alignment
requires sentence-level pooling is not supported empirically — the
multilingual training pushes translation alignment into the contextualized
token representations as well.

## 5. Encoder size does not compensate for the wrong strategy

mE5-large (1024D) with sentence pooling produces statistically the same
geometry as mE5-small (384D) with sentence pooling. Same encoder family,
~3× the parameters, no measurable improvement. Strategy beats parameter
count for this task.

## 6. High triplet rejection can be a geometry symptom, not a sampler bug

When `mean(d_neg − d_pos) ≈ 0`, no margin tweak produces useful gradient.
The classical response (lower the margin, or sample more aggressively) treats
the symptom; the cause is upstream. Diagnose the cosine gap and rejection-vs-
margin curve before tuning training hyperparameters.

## Operational implication

For per-concept retrieval, moving aggregation from full-sentence to
token-level (around the word of interest) is cheap and yields
order-of-magnitude gains in concept discrimination. Test pooling strategy
before scaling data volume or upgrading the encoder.

## Methodology notes

- All measurements over 4–8k sentences sampled from `wikimedia/wikipedia`
  20231101 dumps in EN and ES.
- Cosine gap measured as `mean(intra-concept pair similarity) − mean(inter-concept pair similarity)` over 5–10k random pairs.
- Triplet rejection measured as the fraction of `(anchor, positive same concept, negative different concept)` triplets where `cos(a,n) − cos(a,p) + margin ≥ 0`.
- Lexical-identity confounder controlled by restricting intra-concept positives to pairs whose surface words differ from the anchor's.

## Reproduction scripts

Frozen copies live in [`research/diagnostics/`](../../research/diagnostics/); they were
written against an earlier trainer and collection layout and may need adapting.

- `diagnose_mining_vectors.py` — sentence-pooling baseline on stored cluster vectors
- `diagnose_mining_with_e5_large.py` — same, but vectors re-encoded locally with e5-large
- `diagnose_word_token_pooling.py` — token-pooled diagnostic
- `validate_token_pooling.py` — adds the different-surface-word control
- `test_more_data_hypothesis.py` — sentence-pooling at 10× sentence count per concept
