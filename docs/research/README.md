# Research notes

Archived findings, design specs and engineering reports from the development of
miniCOIL EN-ES. They are kept close to how they were written, so numbers and code
references reflect the state at the time; each note opens with its date and outcome.
For the narrative that connects them, read the [research log](../research-log.md) first.

| Note | Date | What it settled |
|---|---|---|
| [pooling-strategy.md](pooling-strategy.md) | 2026-05 | Token pooling around the concept word beats sentence pooling 7x on concept gap; more data or a larger encoder does not compensate |
| [pooling-verification.md](pooling-verification.md) | 2026-05 | Independent re-run of the pooling diagnostics; found the random within-concept sampler behind the ~98% triplet rejection |
| [4d-polysemy-validation.md](4d-polysemy-validation.md) | 2026-05 | Trained heads on token-pooled input separate senses; sentence-pool input is falsified |
| [phase2-scale-concepts-design.md](phase2-scale-concepts-design.md) | 2026-06 | Demand-driven selection of the 2,398 published concepts; Qwen3-Embedding-0.6B as teacher |
| [batched-gpu-training.md](batched-gpu-training.md) | 2026-06 | Batched multi-concept trainer and its equivalence proofs |
| [full-vocabulary-run-plan.md](full-vocabulary-run-plan.md) | 2026-08 | Plan for training all 12,357 concepts (not executed) |

Frozen copies of the scripts these notes cite are in
[`research/diagnostics/`](../../research/diagnostics/). The performance tools are
maintained in [`scripts/perf/`](../../scripts/perf/README.md).
