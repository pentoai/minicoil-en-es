# Batched-training performance: benchmarks, profiler, and the regression gate

This dir validates and measures `minicoil_v2.batched_training` (the opt-in
`--batched` GPU path). It exists because a synthetic benchmark predicted a 7x
speedup that reality delivered as 2x. The gap was a measurement problem, not a
kernel problem; the tools here stop it recurring.

## Why the 7x became 2x (the three blind spots)

The batched kernel (one `bmm`/step per concept bucket) is genuinely fast. But on
a real 50-concept, ~80k-row bucket trained for 80 epochs, the train wall is
dominated by costs the synthetic benchmark never measured:

| phase | share of a real bucket | in synthetic bench? |
|---|---|---|
| per-epoch sampling (`sample_epoch` ×epochs×2) | ~53% | under-measured (tiny concepts) |
| augmentation re-encode (mE5 forward, common to ref + batched) | ~19% | **OFF** (`trim_augment_ratio=0`) |
| host marshaling (per-batch gather) | ~17% | trivial (tiny concepts) |
| one-time setup (distance matrix + argsort) | ~6% | trivial (tiny concepts) |
| GPU kernel (fwd+bwd) | ~4% | the only thing it tracked |

1. **Toy bucket.** `perf_common.smallest_cache_file` defaulted to the *smallest*
   cache bucket (~32k rows); real buckets are ~2.5x larger and the O(rows)/
   O(rows²) phases scale with it. Use `representative_cache_file` for perf.
2. **Augmentation off.** The bench ran with augmentation disabled, so it never
   paid the re-encode, a large cost *common to both paths* that compresses any
   speedup ratio on its own.
3. **`val_epoch_size` mismatch.** The bench used `val = train//8`; the real run
   lowered `--train-epoch-size` to 2000 but left `val_epoch_size` at the constant
   default 6400, so validation did ~3x the work of training every epoch.

The originally-suspected cause (the one-time `argsort`/distance setup) is real and
O(n²), but at ~6% it is not the cap. Per-epoch **sampling** dominates it ~9x.

## Tools

- **`profile_full_bucket.py`** — the source of truth. Reproduces one real bucket's
  train section end-to-end (augmentation re-encode + `train_concept_bucket`) from
  the encode cache, on the production device/config, and prints the phase table.
  Defaults to the representative (largest) bucket.
  - `--check` is the **regression gate**: it fails if the run is non-representative
    (toy bucket / augmentation off) or if the one-time setup share exceeds 20%.
  - Warns when `val_epoch_size > train_epoch_size` (the latent config bug).
- **Phase profiler** (`MINICOIL_BUCKET_PROFILE=1`) — built into
  `batched_training.train_concept_bucket`; attributes wall-clock per phase
  (`sampler_init`, `sample_epoch`, `host_marshal`, `forward`, `backward`,
  `scheduler`) with MPS/CUDA syncs. Zero overhead when unset. Use it on real runs
  to surface a regression in logs/wandb. NB: the per-phase syncs serialize the GPU
  pipeline, so the profiled *total* is an upper bound; trust the relative split.
- **`validate_trainer.py` / `validate_sampler.py` / `validate_end_to_end.py`** —
  the **correctness** gate (numerical equivalence vs the reference path). Separate
  from perf: run these after any change to `batched_training`. They use the
  *smallest* bucket on purpose (fast).
- **`benchmark.py`** — a ref-vs-batched *micro* benchmark. Its ratio is optimistic
  for the reasons above; treat `profile_full_bucket.py --check` as authoritative.

## Workflow for a perf change

1. Make the change in `batched_training.py` (or the train loop).
2. Correctness: `uv run python scripts/perf/validate_trainer.py` must pass
   (`old<->A=0.00e+00`). Note: bit-equivalence holds at `dropout=0`; changes that
   reorder dropout draws (e.g. fusing the per-role projections) are only equal at
   `dropout=0` and must clear the MRR@10 eval gate at production `dropout`.
3. Perf: `MINICOIL_BUCKET_PROFILE=1 uv run python scripts/perf/profile_full_bucket.py --check`.
4. Anything that changes training dynamics (`val_epoch_size`, mine-once/reuse) is
   **not** covered by the equivalence tests; it must pass the MRR@10 eval gate.
