#!/usr/bin/env bash
# Lock bm25 / minicoil-v1 / translate-bm25 on the phase2 covered slice (TEST split).
# Mirrors lock_phase2_baselines.sh but for the sealed test split: this is the final
# gate the loop's winning candidate is checked against (WIN_CONDITION is test/mrr10).
# Locking baselines on test is fine — they are fixed references, not the tuned model.
set -uo pipefail
cd "$(dirname "$0")/.."

export MINICOIL_EVAL_ALLOW_TEST=1
SPLITS=data/eval/splits_mmarco_phase2
BASE=data/eval/baselines_mmarco_phase2.json
REPORTS=reports/eval_phase2
mkdir -p "$REPORTS"

for R in bm25 minicoil-v1 translate-bm25; do
  echo "=================== running baseline (TEST): $R ==================="
  uv run minicoil eval run "$R" --dataset mmarco --split test \
    --splits-dir "$SPLITS" --baselines-path "$BASE" --reports-dir "$REPORTS" || {
      echo "!! $R run failed, skipping lock"; continue; }
  REPORT=$(ls -t "$REPORTS"/*"__${R}__"*.json 2>/dev/null | head -1)
  echo "locking $R from $REPORT"
  uv run minicoil eval baselines lock --report "$REPORT" --as "$R" \
    --baselines-path "$BASE" --splits-dir "$SPLITS" --allow-dirty || echo "!! lock $R failed"
done

echo "=================== DONE (test baselines locked) ==================="
