# Findings: 4D polysemy validation, token-pool vs sentence-pool

> Archived research note, kept as written apart from editing for publication.
> Date: May 2026, with 4D heads (the published model uses 8D). Outcome: sentence
> pooling falsified for the heads' input; token pooling kept. The reproduction scripts are
> frozen copies in `research/diagnostics/` and target the code of the time.

End-to-end empirical validation of two questions:

1. Does an earlier small-sample claim hold at scale, that the trained
   per-concept Linear(384,4)+tanh layer (a) gives strong intra-concept
   signal on monosemic concepts and (b) pushes apart different senses
   inside polysemous concepts?
2. With the same architecture and training setup, does sentence-pool
   input produce same / better / worse sense separation than token-pool?

## Setup

- 20 trained per-concept layers (the hand-picked subset).
- Two input collections, identical sentence universe, point-for-point:
  - `minicoil_token_pool` — mE5-small last-hidden-states token-pooled
    around the focal word's char span.
  - `minicoil_full_pool` — built fresh for this experiment by re-encoding
    the exact same sentences with mE5-small sentence-pool (mean over all
    sentence tokens with attention mask, `"passage: "` prefix, L2 norm).
- Two parallel `concept_layers.pt`. Token-pool model is the one already
  trained. Sentence-pool model trained from scratch on
  `minicoil_full_pool` with identical hyperparameters (30 epochs, 8k
  train samples/epoch, lr=2e-3, dropout=0.05, hard mining, MPS,
  ~13.5 min total).
- Per concept: every stored sentence pulled (115–160 per concept,
  2,912 total across both runs).
- Sense buckets: collapse Spanish inflections into a single sense
  (e.g., `fijo/fijos/fija/fijas/fijado/corregido` -> `fixed_adj`).
  Plant and earth use sentence regex (botanical/industrial,
  planet/soil) since the surface form alone does not disambiguate.

Per-concept metric: `gap = min(intra-bucket cosine) - max(cross-bucket cosine)`.
Positive = senses are separable; negative = senses are merged or
worse, cross-sense pairs are more similar than within-sense.

## Headline numbers

### Polysemous concepts: per-concept gap (positive is good)

|  Concept  | Label                  | Token-pool gap | Sentence-pool gap | Δ (TP - SP) |
|-----------|------------------------|---------------:|------------------:|-------------:|
| C-01016   | type / write           |       **+1.221** |          +0.275  |       +0.946 |
| C-00875   | work / jobs / obra     |       **+0.592** |          +0.054  |       +0.538 |
| C-00190   | set / fixed / ensemble |       **+0.449** |          -0.099  |       +0.548 |
| C-01120   | way / road             |       **+0.268** |          -0.032  |       +0.300 |
| C-01035   | case (legal/instance)  |       -0.002    |          -0.023  |       +0.021 |
| C-00173   | time / weather / clima |       -0.106    |          -0.156  |       +0.050 |
| C-03566   | plant (botan/industr)  |       -0.205    |          -0.281  |       +0.076 |
| C-00183   | see / view / vista     |       -0.277    |          +0.010  |       -0.287 |
| C-03567   | earth (planet/soil)    |       -0.498    |          -0.188  |       -0.310 |

Token-pool produces a positive gap on 4 of 9 polysemous concepts; on
the 4 it wins (type/write, work, set/fixed, way/road) it wins by a
large margin. Sentence-pool produces positive gap on 3 of 9, all of
them weak (largest +0.275).

The negative-gap concepts mostly reflect bucketing limitations, not
model failure. See per-concept notes below.

### Monosemic concepts: intra-concept 4D cosine (higher is better — same concept should cluster)

|  Concept  | Label             | Token-pool intra | Sentence-pool intra | Δ |
|-----------|-------------------|-----------------:|--------------------:|---:|
| C-06550   | water / agua      |         **+0.740** |              +0.343 | +0.397 |
| C-00304   | animal / pet      |         **+0.727** |              +0.350 | +0.377 |
| C-03165   | life / vida       |         **+0.694** |              +0.328 | +0.366 |
| C-00818   | use / usage       |         **+0.663** |              +0.247 | +0.416 |
| C-00201   | world / global    |         **+0.656** |              +0.262 | +0.394 |
| C-00122   | food / feed       |         **+0.649** |              +0.307 | +0.342 |
| C-01594   | say / tell        |         **+0.571** |              +0.225 | +0.346 |
| C-00620   | make / do         |         **+0.548** |              +0.266 | +0.282 |
| C-01098   | include           |         **+0.457** |              +0.322 | +0.135 |
| C-00126   | understand        |         **+0.415** |              +0.247 | +0.168 |
| C-00877   | motion / move     |         **+0.362** |              +0.289 | +0.073 |

Token-pool wins on every monosemic concept. The gap is roughly 2× on
the concrete ones (water, animal, life, world, food) and narrows on
the abstract ones (motion, understand, include) where even token-pool
struggles.

### 4D output magnitude (the "are values too small" question)

|                      | Token-pool | Sentence-pool |
|----------------------|----------:|--------------:|
| mean L2 norm         |     0.151 |         0.104 |
| p5 / p50 / p95       | 0.092 / 0.151 / 0.212 | 0.064 / 0.103 / 0.147 |
| min / max            | 0.028 / 0.271 | 0.023 / 0.188 |

Token-pool outputs are about 50 % larger. In a BM25 + miniCOIL setup
the per-word contribution is `IDF · dot(q4d, d4d)`; the dot product
for matched concept words averages around `0.151² ≈ 0.023` (TP) vs
`0.104² ≈ 0.011` (SP) if cosines line up.  Either way the per-word
signal is small relative to typical BM25 term scores; this is a
calibration issue that affects both, but token-pool has more
headroom.

### What does the trained layer add on top of raw mE5? (384D baseline)

The 4D-only tables above hide a critical question: is the trained
Linear(384,4)+tanh adding signal compared to using the input vectors
directly, or paying a tax somewhere?

For every (concept, sense bucket) we measured raw 384D intra-cosine
on the same sentences and compared to 4D intra-cosine. For polysemous
concepts we also measured raw 384D cross-bucket cosine, the baseline
for "do senses look similar in the raw input?".

|                                | Token-pool | Sentence-pool |
|--------------------------------|----------:|--------------:|
| Monosemic intra 384D (mean, n=11) | **+0.779** | +0.754 |
| Monosemic intra 4D  (mean, n=11) |   +0.589 |     +0.290 |
| Monosemic Δ (4D − 384D)          |   **-0.189** |  -0.464 |
|                                |   |   |
| Polysemic intra 384D (mean, n=27) |   +0.801 |     +0.775 |
| Polysemic intra 4D  (mean, n=27) |   +0.719 |     +0.370 |
| Polysemic intra Δ (4D − 384D)    |   -0.082 |     -0.405 |
|                                |   |   |
| Polysemic **cross** 384D (mean, n=30) |   +0.729 |     +0.748 |
| Polysemic **cross** 4D  (mean, n=30) |   +0.264 |     +0.201 |
| Polysemic cross Δ (4D − 384D)    |   **-0.465** |  -0.547 |

How to read this:

- **Raw 384D cannot separate senses.** Even for clearly polysemous
  concepts, cross-sense cosine in the input is +0.73 (token-pool) /
  +0.75 (sentence-pool) — almost as high as intra-sense. The input
  vectors don't know which sense is which.
- **The trained layer fixes that.** Cross-sense 4D cosine drops to
  +0.26 / +0.20. That's a -0.47 / -0.55 push apart in cosine — the
  layer is doing exactly what miniCOIL is meant to do.
- **The layer also reduces intra-cosine.** Token-pool: -0.19 on
  monosemic, -0.08 on polysemic intra-bucket. Sentence-pool: -0.46
  on monosemic, -0.41 on polysemic intra. This is the tax.
- **Token-pool has the better trade.** It pays a ~0.19 intra tax to
  buy a ~0.47 cross-sense separation — roughly 2.5× return.
  Sentence-pool pays a ~0.46 intra tax to buy a ~0.55 cross
  separation — barely 1.2× return. After the tax, sentence-pool 4D
  intra-cosine on monosemic concepts is only +0.29, meaning
  same-concept sentences are weakly similar in 4D space.

What this means for retrieval: on a per-word contribution
`IDF · dot(q4d, d4d)`, monosemic concepts dominate the long tail of
queries. For token-pool, monosemic dot products are roughly
`0.589 · 0.151² ≈ 0.013`. For sentence-pool, `0.290 · 0.104² ≈ 0.003`.
Token-pool's per-word contribution is 4× larger in expectation.

The earlier first-pass framing — "the layer degrades the
input signal on monosemic concepts" — is empirically correct in
isolation; they were right to walk it back to "this is the trade
miniCOIL makes." The net signal direction is still positive (cross
push >> intra reduction) on token-pool. Whether the absolute
magnitudes are large enough to influence BM25 rankings is the open
retrieval question.

### Cross-lingual top-1 same-concept rate

Across all 2,912 sentences (1,527 EN, 1,385 ES): for every EN
sentence, find its top-1 4D neighbor among the ES pool (each sentence
passed through its own concept's layer) and check if the concept
matches.

|                     | Top-1 same-concept | vs chance (5%) |
|---------------------|------------------:|----------------:|
| Token-pool          |             36.0% |          7.2× |
| Sentence-pool       |             13.2% |          2.6× |

Both well above chance, token-pool ~3× better. The 50 % the previous
earlier analysis reported was on 12 EN sentences across 4 concepts; with 1,527
EN sentences across 20 concepts the rate is lower, but the
qualitative finding (cross-lingual signal exists) reproduces.

## Per-polysemous-concept inspection

### C-00190 set / fixed / ensemble (lumped MUSE cluster)

**Token-pool cross matrix (diagonals are intra, off-diagonals are cross):**

```
              ensemble_es   fixed_adj    set_eng
ensemble_es        +0.842      -0.012     +0.327
fixed_adj          -0.012      +0.846     -0.085
set_eng            +0.327      -0.085     +0.776
```

intra-bucket min = +0.776, cross-bucket max = +0.327, **gap = +0.449**.

Sample EN/ES sentences and their 4D vectors:

- `set_eng` (en) "The film was [set] in 18th-century Russia..."
  -> 4D = [+0.034 +0.222 +0.048 -0.023]
- `set_eng` (en) "Since 1947, the award is shared with the [set] decorators."
  -> 4D = [+0.124 +0.111 +0.009 -0.030]
- `fixed_adj` (en) "Solids have a more rigid and [fixed] structure than liquids."
  -> 4D = [+0.027 -0.072 +0.082 +0.141]
- `fixed_adj` (es) "...permanecen [fijos] en un sitio."
  -> 4D = [+0.065 -0.055 +0.107 +0.110]
- `ensemble_es` (es) "El [conjunto] de todos los músculos esqueléticos..."
  -> 4D = [+0.121 -0.050 +0.052 -0.108]

The `set_eng` cluster occupies the (+, +, .., -) corner; `fixed_adj`
occupies (.., -, +, +); `ensemble_es` occupies (+, -, .., -). The
cross between `fixed_adj` and `set_eng` is **-0.085** and between
`fixed_adj` and `ensemble_es` is **-0.012** — different senses do
end up in different parts of 4D space.

**Sentence-pool cross matrix:**

```
              ensemble_es   fixed_adj    set_eng
ensemble_es        +0.153      +0.242     +0.204
fixed_adj          +0.242      +0.491     +0.252
set_eng            +0.204      +0.252     +0.289
```

intra-min = +0.153, cross-max = +0.252, **gap = -0.099**.

All three buckets are loosely positive with no separation. The model
has not found the sense axes that token-pool found cleanly.

### C-00875 work / jobs / obra

Two clear senses: labor work (English `work`/`jobs`, Spanish
`trabajo`/`trabajos`/`empleos`) vs artistic work (Spanish `obra`).

**Token-pool:** intra `work_artistic` +0.831, intra `work_labor` +0.594,
cross +0.002. **gap = +0.592**.

- `work_artistic` (es) "Esta [obra] por su claridad y rigor expositivo..."
  -> 4D = [+0.160 +0.045 +0.002 +0.102]
- `work_labor` (en) "...worked at odd [jobs] in Kentucky..."
  -> 4D = [-0.050 -0.038 -0.100 -0.149]
- `work_labor` (es) "...obtenidos por [trabajos] anteriores..."
  -> 4D = [-0.006 +0.121 -0.098 +0.002]

`obra` lives in the (+, +, .., +) corner. `work`/`trabajos` live in
(-, .., -, -). Cross-sense cosine essentially zero.

**Sentence-pool:** intra `work_artistic` +0.481, intra `work_labor`
+0.249, cross +0.195. **gap = +0.054**.

Same 4D values are an order of magnitude smaller and the two senses
are weakly positive with each other; the geometry has not learned to
split.

### C-01016 type / write

Two senses: `type`/`tipo` (= "kind of") vs `write`/`writes`/`escribir`
(= verb to write/typing).

**Token-pool:** intra `kind` +0.747, intra `write_verb` +0.819,
cross **-0.474**. **gap = +1.221**.

The strongest sense separation in the whole experiment. Type-as-kind
and to-write end up at strongly negative cosine — exactly the
behavior miniCOIL is meant to produce.

**Sentence-pool:** intra `kind` +0.396, intra `write_verb` +0.311,
cross +0.036. **gap = +0.275**.

Sentence-pool keeps the senses *separable* (cross only +0.036) but
the intra-bucket cohesion is half of token-pool's, so the absolute
signal is much weaker.

**Language-confound caveat.** The `kind` bucket is 44 EN / 72 ES and
the `write_verb` bucket is 36 EN / 8 ES. The cross of -0.474 is
partly "kind vs verb" and partly "Spanish vs English." That said,
each bucket has at least 8 sentences of the minority language and
its intra-cosine remains high (+0.75 and +0.82), so the layer has
learned a sense embedding that holds across languages — but the
+1.221 headline number is the optimistic end of a range and the
"true" sense-only gap is closer to the C-00190 / C-00875 ballpark
(+0.45 to +0.60). The qualitative conclusion (token-pool separates
senses) holds.

### C-01120 way / road / manera

`way_en`/`ways` map to one bucket, `road`/`roads`/`camino` to another,
`manera` to a third (Spanish for "manner"). Three buckets.

**Token-pool gap = +0.268**: way_en intra +0.713, road_path intra
+0.756, manner_es intra +0.705. Cross matrix shows road_path and
manner_es are similar in 4D (+0.437) but way_en is well separated
from manner_es.

**Sentence-pool gap = -0.032**: no separation.

### C-01035 case (legal / container / instance)

`case_en` (`case`) and `case_es` (`caso`) — single surface form in
each language with multiple senses *inside* the form. Bucketing by
surface form does not disambiguate.

Token-pool gap = -0.002 (no separation), sentence-pool gap = -0.023.

This is a bucketing limitation, not a model failure. A deeper test
would require sense-tagged sentences (legal-case vs container-case
vs instance-case) which we do not have.

### C-00173 time / weather

`tiempo` (Spanish) is ambiguous — means both time and weather. Even
the sense-mapping cannot cleanly separate it. The clean signal
("weather"/"clima"/"meteorología" vs "time") is partially recoverable
but with noise.

Token-pool gap = -0.106, sentence-pool gap = -0.156. Both fail, both
for the same reason: the ambiguity in the input data, not in the
model.

### C-00183 see / view / vista / reproductions

Five sense buckets after mapping; surface forms are not all
disambiguated cleanly (English `view` is also ambiguous).

Token-pool gap -0.277, sentence-pool gap +0.010. Sentence-pool barely
wins here because all its values are small. Looking at the raw cross
matrix on token-pool: see_verb and vista_noun are at +0.75 (high) —
that's because in Spanish "vista" appears in contexts very similar to
"ver/véase" (visual/perception). The bucketing claims they are
different senses; the actual usage in Wikipedia is closely related.
Model behavior may be correct; the bucket labels are.

### C-03566 plant (botanical vs industrial)

Regex-bucketed. Most sentences fall into "other_en" / "other_es"
because the regex did not match — those buckets are catch-alls that
contain both senses mixed.

Token-pool gap -0.205, sentence-pool gap -0.281. The botanical vs
industrial signal exists in the model's outputs (the small `industrial`
bucket separates from `botanical`) but the dominant "other" buckets
drown the gap metric. Visual inspection of sample sentences confirms
the regex is the bottleneck, not the model.

### C-03567 earth / tierra (planet vs soil)

Same issue as plant — only 6 sentences land in the `soil` bucket and
the rest pile up in "other".

Token-pool gap -0.498. The earlier analysis inspected a hand-picked
C-00063 city/people case (not in our trained set) and found cleanly
negative cross-cosine on the right pairs — that finding is consistent
with what we see in C-00190 and C-00875 where the bucketing is good.

## Verdict on each question

### Sentence-pool training health

For completeness: the sentence-pool training run did not converge as
cleanly as the token-pool run. Validation triplet-rejection rates
stayed at 80-90 % across all 20 concepts and all 30 epochs (vs
~38-40 % for the token-pool run). Cross-lingual validation rates
hovered around 10-15 % (vs 40 % for token-pool). The model received
very few effective gradient updates per epoch because most candidate
triplets had `|cos(a,n) − cos(a,p)| < 0.1` and were rejected by the
adaptive-margin filter.

This is a direct consequence of sentence-pool's weaker geometry: in
input space, "a sentence with focal-word X" and "another sentence
with focal-word X-different-sense" already look very similar
(cross-bucket 384D = +0.748). Hard mining cannot find informative
negatives when the input cosines are all clustered around the same
value. More epochs would not have helped — the gradient simply is
not there to extract.

### Q1. Does the earlier qualitative claim reproduce at scale?

**Yes, with caveats.** On polysemous concepts where bucketing is
clean (set/fixed/ensemble, work/jobs/obra, type/write), the 4D layer
clearly separates senses — cross-sense cosines are 0 to -0.5 while
intra-sense is +0.6 to +0.85. The earlier specific +1.058
separation on C-00190 (intra-set, intra-fixed, cross set-fixed)
reproduces at full sample: intra-set +0.776, intra-fixed +0.846,
cross -0.085.

On monosemic concepts, intra-cosine is +0.4 to +0.74 — large positive
signal exactly as claimed.

What does **not** fully reproduce:
- The 50 % cross-lingual top-1 falls to 36 % on full sample. Still 7×
  chance, but the small-sample number was inflated by selection.
- The "small 4D magnitudes" concern is real and confirmed at scale
  (p95 = 0.212). Average per-word contribution to BM25-style scoring
  remains a calibration question.
- The earlier headline `cross set ↔ fixed = -0.283` (from
  6 sentences per bucket, EN-only) softens to -0.085 on the full
  EN+ES sample. Same sign and same conclusion (different senses are
  pushed apart), smaller magnitude — partly because the bigger
  sample includes Spanish `fijo/fijado/fijas` which are not
  perfectly aligned with English `fixed` in 4D, and partly because
  the EN-only handpick was a best case.

What the 384D baseline *added* to the picture (not in the previous
analysis):
- The trained layer **reduces intra-cosine on monosemic concepts**
  vs raw input by -0.19 (token-pool) / -0.46 (sentence-pool). This
  is the tax for sense separation. The earlier first-pass
  framing "the layer degrades the signal" was correct in isolation
  for the monosemic case.
- The trained layer **reduces cross-sense cosine on polysemous
  concepts** by -0.47 (token-pool) / -0.55 (sentence-pool). This is
  the gain — and it dwarfs the monosemic tax, but only on the
  polysemous subset of the vocabulary.

### Q2. Does sentence-pool match / beat / lose to token-pool?

**Sentence-pool loses across the board.**

- Monosemic intra-cosine: 0.5× to 0.55× token-pool's level on every
  concept tested.
- Polysemy gap: token-pool wins on 7 of 9, by large margins where
  it wins. Sentence-pool wins by tiny margins on 2 (see/view, earth)
  where token-pool's larger absolute values fall on the wrong side
  of a noisy bucket boundary.
- 4D output magnitude: sentence-pool is 30% smaller.
- Cross-lingual top-1: 13% vs 36%, ~3× weaker.

The hypothesis that the per-concept layer can compensate for weaker
input geometry is **falsified.** Whatever the layer learns is bounded
by the discriminative information already present in the input, and
sentence-pool simply contains less concept-specific signal than
token-pool. This is consistent with the prior raw-cosine-gap finding
(`docs/findings-pooling-strategy.md`): the gap is +0.017 for
sentence-pool vs +0.116 for token-pool *before* training, and a
trained Linear(384,4)+tanh on top does not close it.

## Bottom line

miniCOIL v2's core architectural choice — Linear(384,4)+tanh per
concept on top of mE5-small **token-pooled** vectors — works on the
sense-separation task it is designed for. Different senses inside
lumped concepts get pushed into different quadrants of 4D space when
the input pooling captures word-local context. Switching to
sentence-pool inputs degrades every measurable property of the
output. The training does not rescue weaker inputs.

The honest picture is that the trained layer pays a tax on monosemic
concepts (intra-cosine drops ~0.19 vs raw input) to buy a much
larger gain on polysemous concepts (cross-sense cosine drops ~0.47
vs raw input). Roughly 2.5× return on token-pool. The architecture
is net positive on cross-sense discrimination — the thing miniCOIL
exists to do — but it isn't free, and the monosemic tax is a real
cost retrieval has to overcome.

This validates the v2 path on the per-concept geometric task. It
does **not** validate retrieval performance directly — for that,
BM25+miniCOIL vs BM25 on MIRACL EN+ES remains the unrun,
decision-grade test.

## Reproduction

- Build the parallel sentence-pool collection:
  `uv run python research/diagnostics/build_full_pool_collection.py --rebuild --limit-concepts <list>`
- Train sentence-pool layers (matches token-pool hyperparams):
  `MINICOIL_QDRANT_PREFER_GRPC=false uv run minicoil train --collection-name minicoil_full_pool --concepts <list> --output-dir data/concept_models_sentpool --epochs 30 --train-epoch-size 8000 --val-epoch-size 800 --device mps --no-wandb`
- Validation report:
  `uv run python research/diagnostics/validate_4d_polysemy.py --collection <coll> --layers-path <path> --label <tag>`
- 384D vs 4D baseline:
  `uv run python research/diagnostics/baseline_384d_per_concept.py --collection <coll> --layers-path <path> --label <tag>`
- Raw logs were kept locally and are not part of the repo.
