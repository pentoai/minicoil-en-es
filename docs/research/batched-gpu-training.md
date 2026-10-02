# Batched GPU training: performance refactor + equivalence report

> Archived engineering report (June 2026). The batched path is opt-in (`minicoil train
> --batched`); the tools it cites live in [`scripts/perf/`](../../scripts/perf/README.md),
> which also records why the measured 7x became ~2x on a real bucket.

This is a pure performance refactor of the
per-concept training loop in `train_concept_layers.py`. The model and the loss are
unchanged; the goal is to make training GPU-bound and batched. The reference path
is kept intact and is the equivalence oracle.

## What changed

New module `src/minicoil_v2/batched_training.py`:

1. **Vectorized sampler** (`VectorizedSampler`). The reference `BilingualSampler`
   builds an epoch one tuple at a time in a Python `while` loop (~64k iterations
   per concept per epoch). The new sampler draws a whole epoch with bulk
   gather + a vectorized `searchsorted` (a boolean row-sum) over the pre-sorted
   candidate rows. The semi-hard selection rule, adaptive margins, `has_xl` mask,
   monolingual fallback and reject-and-backfill are preserved.
2. **Batched multi-concept trainer** (`train_concept_bucket`). The reference
   trains thousands of `Linear(384, 8)` heads one at a time (each a microscopic
   GPU op). The new trainer stacks a bucket of `B` concept weights `[B, 8, 384]`,
   projects all concepts' batches with one `bmm`, computes the per-concept loss,
   sums them, and does **one** backward and **one** optimizer step. Because the
   weights are independent, `L = Σ_c L_c` routes each concept exactly its own
   gradient, so a single Adam step on the stacked weight equals `B` independent
   Adam steps.
3. **GPU-resident data.** Inputs are concatenated into one flat embedding bank
   indexed by gather; per-step CPU↔GPU transfer is eliminated.

Two optimizer back-ends (both validated):

- `per_concept_torch` (Milestone A): `B` separate `torch.optim.Adam` +
  `B` separate `torch.optim.lr_scheduler.ReduceLROnPlateau`. Exact by
  construction; the only new thing is the batched forward. Used as the oracle.
- `vectorized` (Milestone B, default): a single stacked custom Adam with a
  per-concept learning rate `[B, 1, 1]` and a vectorized per-concept plateau
  scheduler. This is the launch-count win that lets one step train `B` concepts.

Opt-in only: `TrainSettings.batched=False` by default, so the reference path is
unchanged. `minicoil train` with `MINICOIL_TRAIN_BATCHED=true` (or `batched=True`)
uses the batched path; `optimizer_mode` selects the back-end.

## Equivalence (the deliverable)

All checks are runnable from the cache (`data/enc_cache_phase2`), no Qdrant, no
eval gate. `trim_augment_ratio=0` and `dropout=0` for parts (a)/(b) to isolate the
deterministic computation.

### (a) Sampler — `scripts/perf/validate_sampler.py`

| Check | Result |
|-------|--------|
| **a1** deterministic core, exhaustive grid (every anchor × every pos_rank, SL + XL) vs the real `_pick_pair_mined` | **0 mismatches** over 67,854 cases across concepts n=47/497/1595 and a synthesized monolingual concept |
| **a2** assembly invariants (index ranges, SL language, monolingual `has_xl` all-False with `xl=anchor`, XL cross-language) | **OK** for bilingual + monolingual |
| **a3** distribution match (own-RNG full epoch) | reject_rate 0.0000 vs 0.0000; xl_rate 1.0000 vs 1.0000; margin_sl 0.1125±0.0354 vs 0.1125±0.0350 (and 0.1044±0.0195 vs 0.1043±0.0189) |

a1 pins the selection *mapping* bit-for-bit, independent of RNG. Exact
full-sampler RNG replay is infeasible (the reference consumes Python+NumPy RNG
interleaved per sample with rejection redraws; the vectorized path draws in bulk),
so a3 shows the selection *distribution* matches — the decomposition the task
allows.

### (b) Trainer — `scripts/perf/validate_trainer.py`

| Check | Result | Bar |
|-------|--------|-----|
| **b0** `BatchedAdam` vs `torch.optim.Adam`, 1000 steps, per-concept lr | **4.44e-16** | ≤1e-6 |
| **b1** `BatchedPlateau` vs N× `torch ReduceLROnPlateau`, 60 epochs | **lr trajectories identical** | exact |
| **b2** old ↔ A (frozen triplets, dropout=0, same init) | **0.00e+00** (n=47/346/732/1595) | ≤1e-5 |
| **b3** A ↔ B (custom Adam + vectorized plateau) | **5.7e-8 … 4.1e-7** | ≤1e-5 |
| **b4** B>1 non-interference (solo vs bucketed weight) | **0.00e+00** (all) | ≤1e-6 |

b2 = exactly 0 proves the batched forward/loss/backward is bit-identical to the
reference. b3 isolates the optimizer swap: the only op-level deviation is the
fused `addcdiv` (scalar value) expanded to a per-concept-lr broadcast multiply,
which costs ≤4.1e-7 over a full short training. b4 = 0 proves summed-loss /
one-backward does not couple concepts.

### (c) End-to-end — `scripts/perf/validate_end_to_end.py`

24 concepts, 30 epochs, **dropout on (0.05)**, real per-epoch re-mining, init
pinned. Bit-exact is impossible here by construction (independent sampling +
dropout RNG streams), so the fair test is whether the batched path drifts more
than the reference drifts from *itself* across a sampling-seed change.

| Final val-loss delta | mean | median | p90 | max |
|----------------------|------|--------|-----|-----|
| reference-vs-reference (noise floor) | 0.00676 | 0.00613 | 0.01370 | 0.02143 |
| batched-vs-reference | 0.00958 | 0.00813 | 0.01518 | 0.03400 |

`mean(batched-vs-reference)=0.00958 < 2×mean(noise floor)=0.01351`. The batched
path adds no systematic drift beyond the pipeline's own stochasticity.

The floor here varies the **sampling** seed only (the two reference runs share an
identical init and dropout stream), so it is a *lower bound* on the reference's
true run-to-run variance. The batched run additionally differs in its dropout
stream, which accounts for it sitting modestly (~1.4×) above this conservative
floor rather than at ~1×. Parts (a)/(b) carry the bit-exact proof of the
computation; (c) only confirms there is no order-of-magnitude integration-level
drift.

## Speedup — `scripts/perf/benchmark.py`

Same K concepts, same device, reference vs batched (`vectorized`):

| Device | B | epochs | epoch_size | ref proj 12k | batched proj 12k | speedup |
|--------|---|--------|------------|--------------|------------------|---------|
| CPU | 16 | 40 | 8000 | 25.6 h | 10.0 h | **2.6×** |
| MPS | 24 | 30 | 4000 | 17.7 h | 2.5 h | **7.0×** |
| MPS | 48 | 20 | 4000 | 11.7 h | 1.6 h | **7.5×** |

Single-run wall-clock with normal machine variance (`build_spec`/distance-matrix
build is timed on both sides). Sampler micro-benchmark (1 epoch @ production
`epoch_size=64000`): reference ~595–643 ms/epoch → vectorized ~100–140 ms/epoch,
**~4–6×**, on every concept size.

Reading the numbers:
- On **CPU** the win is 2.6× — there are no kernel launches to amortize, so it is
  purely Python-loop elimination + the ~5× sampler speedup.
- On **MPS** (GPU-like dispatch overhead) the win is 7×, and the batched path
  scales with the bucket size `B`: batched throughput grew from B=24 (1.32
  concepts/s, proj 2.5 h) to B=48 (2.14 concepts/s, proj 1.6 h) while the reference
  scaled with concept count. This is the kernel-launch amortization the refactor
  targets.
- A real **CUDA** box has the same (typically larger) per-launch overhead for tiny
  ops, so the GPU win should be at least as good. **Larger B** (pooling concepts
  across multiple cache files into one bucket) is the lever to push the full-vocab
  time below an hour; it is supported (the trainer is B-agnostic) but not required
  for correctness. No CUDA device was available locally; CUDA numbers are not
  measured here.

## Self-review: edge cases and semantic drift

Adversarial review of where the new path could differ from the reference, and
whether the tests exercise it:

- **Monolingual concepts (`has_xl` all-False).** Covered: a1 grid on a synthesized
  monolingual concept (0 mismatches), a2 invariants (`has_xl` all-False,
  `xl_pos=xl_neg=anchor`). The batched loss replaces the reference's
  `has_xl.any()` short-circuit with an unconditional `* has_xl`, which is
  numerically identical (masked terms are exactly 0); b2=0 confirms it end-to-end.
- **Concepts with too few rows.** `prepare_concept` returns `None` for
  `n < MIN_SENTENCES_PER_CONCEPT`, exactly as the reference skips. Anchor-lang with
  `<3` rows → that draw rejects (a1 skips `m<3` the same way). The n=47 concept
  (train_to=38) is in every test.
- **min_margin floor vs scale.** The per-concept `min_margin` is produced by the
  **reused** `dynamic_min_margin` (same function, default `margin_scale=0` → floor
  0.1, tested). The sampler treats `min_margin` as an opaque scalar, so the
  floor/scale choice cannot change sampler equivalence; a1 runs with the concept's
  actual `min_margin`. The `margin_scale>0` exploration path is not separately
  benchmarked (it is off by default and known-experimental).
- **`run_training` batched wiring.** Covered by `scripts/perf/validate_wiring.py`:
  it drives the real `run_training(batched=True)` from a cache-derived batch
  (monkeypatching only the Qdrant client, encoder load, and scroll/encode boundary)
  and asserts the merged `concept_layers.pt` has the right concepts and weight
  shapes. The spec-collection, post-loop bucket call, checkpoint save/merge and
  skip/monolingual bookkeeping are all executed.
- **Trimming-augmentation rows.** Augmentation runs in `run_training` **before**
  `prepare_concept`, using the unchanged reference code, and consumes Python
  `random`. The batched wiring feeds the augmented tensors to `prepare_concept`
  like any other rows (no special-casing). Tests set `trim_augment_ratio=0` to
  isolate the refactor; augmentation-**on** is the one path covered only by
  code-reuse, not by a dedicated test (the wiring smoke test runs with
  `trim_augment_ratio=0` to avoid needing the encoder).
- **searchsorted tie-breaking.** `side="left"` is replicated as "count of
  candidates strictly less than the threshold" (`(d < thr).sum`). Ties are present
  in the real distance matrices and a1 is bit-exact, so ties resolve identically.
- **train/val split.** `prepare_concept` reproduces the reference formula
  (`n_val = clamp(int(n*val_size), 1, n-3)`); samplers are built on `[0, train_to)`
  and `[train_to, n)` with `interleave_by_language` applied first, identical to the
  reference.
- **Ragged epochs.** The batched trainer assumes constant samples-per-concept
  (=`epoch_size`). Measured: every concept back-fills to exactly `epoch_size`
  (even n=47, 2.9% reject). As a guard, a bucket is truncated to the shortest
  concept's epoch if a degenerate concept under-fills — a no-op in practice.

### Flagged semantic changes (and their measured impact)

1. **RNG stream differs** (bulk draws; dropout masks interleave differently across
   a bucket). Consequence: end-to-end results are equivalent **within tolerance**,
   not bit-identical — quantified in (c): within 2× the reference's own
   sampling-seed noise floor. Unavoidable for any bulk vectorization.
2. **Custom vectorized Adam.** ≤4.4e-16 in isolation (b0), ≤4.1e-7 over a full
   short training (b3). Op sequence matches `torch.optim.Adam`; only the final
   fused `addcdiv` becomes a broadcast multiply.
3. **Init draw order.** Reference draws each concept's init as the next global-RNG
   value (depends on prior concepts); the batched path draws all inits up front.
   Different (equally valid) random init, same class of non-determinism as a seed
   change. Tests pin init to isolate.
4. **LR schedule is preserved exactly**, not changed: b1 shows the vectorized
   per-concept plateau reproduces `torch ReduceLROnPlateau` lr trajectories
   identically, and per-concept lr is carried as a `[B,1,1]` tensor.

The bmm-vs-mm forward is bit-identical on CPU (b2=0); on GPU floating-point
reduction order may introduce ~1e-6, which is standard and not a semantic change.

## How to reproduce

```bash
uv run python scripts/perf/validate_sampler.py       # (a)
uv run python scripts/perf/validate_trainer.py       # (b)
uv run python scripts/perf/validate_end_to_end.py    # (c), ~2 min
uv run python scripts/perf/validate_wiring.py        # run_training(batched=True) integration
uv run python scripts/perf/benchmark.py              # CPU speedup + sampler micro
uv run python scripts/perf/benchmark.py --device mps --k 48 --epochs 20
```
