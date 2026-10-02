# Bilingual 5-Point Contrastive Sampling

## Design: Four Quadrants, Five Points

Every concept's sentences live in a 2D space along two axes:

- **Language**: same (SL) vs. different (XL)
- **Similarity**: similar (Sim) vs. dissimilar (Dis)

This gives four quadrants, each teaching the concept layer something distinct.

### Quadrant definitions

All examples use concept C-00000 of the published vocabulary: `{award, prize, reward,
bounty, …}` / `{premio, galardón, recompensa, …}`. "Similar" and "dissimilar" are
measured by the teacher (mining) vectors stored at embed time.

| Quadrant | Anchor | Partner | Example | Signal |
|----------|--------|---------|---------|--------|
| **SL-Sim** | EN sentence | EN sentence, high mining similarity | "won the Nobel prize" ~ "received an award for research" | Collapse same-sense, same-language representations |
| **SL-Dis** | EN sentence | EN sentence, low mining similarity | "won the Nobel prize" vs. "the bounty was claimed" | Push apart different senses within same language |
| **XL-Sim** | EN sentence | ES sentence, high mining similarity | "won the Nobel prize" ~ "ganó el premio Nobel" | Align equivalent meanings across languages |
| **XL-Dis** | EN sentence | ES sentence, low mining similarity | "won the Nobel prize" vs. "recompensa por su captura" | Separate different senses across languages |

## Implementation: BilingualSampler

The four quadrants are exercised through a **5-point sampling** strategy implemented
in `BilingualSampler` (`src/minicoil_v2/train_concept_layers.py`).

Each sample contains:

```
(anchor, sl_pos, sl_neg, xl_pos, xl_neg)
```

### Sampling algorithm

Each concept's distance matrix is `1 − cosine` between teacher vectors. For each language,
every anchor's candidates are sorted by distance once, when the sampler is built. Then,
for each sample:

1. **Pick the anchor.** Choose a language uniformly among those present, then a random
   sentence in it. (Language balance is set earlier, at fetch time, by `lang_ratio`.)

2. **Same-language pair (hard mining).** Among the anchor's same-language candidates:
   - `sl_pos` is a random pick from the `k` nearest, with `k = min(20, ⌊m/3⌋)` for `m`
     candidates (at least 1).
   - `sl_neg` is the nearest candidate ranked after those `k` whose distance is at least
     `d(a, sl_pos) + min_margin`, i.e. the hardest semi-hard negative.
   - If no candidate qualifies, the sample is rejected and redrawn.

3. **Cross-language pair.** The same procedure over the other language's candidates.
   If it fails, or the concept has one language only, the sample keeps just its
   same-language triplet and `has_xl` masks the cross-language term in the loss.

4. **Adaptive margins.** Each term's margin is the teacher's own gap for that sample:
   - `margin_sl = d(a, sl_neg) − d(a, sl_pos)`
   - `margin_xl = d(a, xl_neg) − d(a, xl_pos)`

The margin floor `min_margin` is `max(min_triplet_margin, margin_scale × std(distances))`
per concept. A fixed floor never bites on tight (monosemous) concepts and is trivial on
spread (polysemous) ones; scaling by the concept's own spread makes it concept-relative.
A larger floor picks *easier* negatives (further away), so `margin_scale` is a tuning
knob, not a guaranteed win. The published model used `margin_scale = 2.0`.

#### Why hard mining

The first v2 sampler drew two random same-concept sentences and called the closer one the
positive. Random pairs inside a concept are almost equidistant from the anchor, so
96–99% of samples failed the margin and the rest carried little gradient. miniCOIL v1
mines hard by construction, and restoring that fixed the rejection problem
([research/pooling-verification.md](research/pooling-verification.md)). The random
sampler survives as `hard_mining=False` for ablations.

### Re-mining per epoch

Following v1's approach, samples are freshly drawn every epoch. No pre-computed
triplet lists. This prevents the model from memorizing specific pairings and ensures
the hardness distribution evolves as the model learns.

- **Train:** `train_epoch_size` samples/epoch (default 64,000)
- **Val:** `val_epoch_size` samples/epoch (default 6,400)
- **Batch size:** `sample_batch_size` (default 256)

## Loss: 4-Term Bilingual Cosine Triplet

```
loss = relu(d_cos(a, sl_pos) - d_cos(a, sl_neg) + margin_sl)
     + relu(d_cos(a, xl_pos) - d_cos(a, xl_neg) + margin_xl)
```

Where `d_cos = 1 - cos_sim` (cosine distance on the 8-D head output).

The XL term is masked to zero for samples without a valid cross-language pair. When
both terms are active,
the loss simultaneously:

- Pulls same-language similar sentences together (**SL-Sim**)
- Pushes same-language dissimilar sentences apart (**SL-Dis**)
- Pulls cross-language similar sentences together (**XL-Sim**)
- Pushes cross-language dissimilar sentences apart (**XL-Dis**)

## Language Balancing

Two mechanisms ensure balanced bilingual training:

1. **Fetch-time balancing.** `scroll_concept` accepts a `lang_ratio` parameter
   (default 0.5) and downsamples the majority language after fetching from Qdrant.

2. **Interleaving.** Before the 80/20 train/val split, sentences are interleaved
   by language (alternating EN/ES indices) so both splits contain both languages.

## Debug Diagnostics

`BilingualSampler.log_epoch_stats()` emits per-epoch statistics:

| Metric | Purpose |
|--------|---------|
| Rejection rate | % of draws with no qualifying same-language pair |
| XL rate | % of accepted samples with valid cross-language pairs |
| Mean/std margin (SL) | Distribution of same-language adaptive margins |
| Mean/std margin (XL) | Distribution of cross-language adaptive margins |
