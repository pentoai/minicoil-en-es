#!/usr/bin/env bash
# Re-lock bm25 / minicoil-v1 / translate-bm25 on the phase2 covered slice (val split).
# Parallel gate: writes baselines_mmarco_phase2.json, leaves the old gate untouched.
set -uo pipefail
cd "$(dirname "$0")/.."

SPLITS=data/eval/splits_mmarco_phase2
BASE=data/eval/baselines_mmarco_phase2.json
REPORTS=reports/eval_phase2
mkdir -p "$REPORTS"

for R in bm25 minicoil-v1 translate-bm25; do
  echo "=================== running baseline: $R ==================="
  uv run minicoil eval run "$R" --dataset mmarco --split val \
    --splits-dir "$SPLITS" --baselines-path "$BASE" --reports-dir "$REPORTS" || {
      echo "!! $R run failed, skipping lock"; continue; }
  REPORT=$(ls -t "$REPORTS"/*.json 2>/dev/null | head -1)
  echo "locking $R from $REPORT"
  uv run minicoil eval baselines lock --report "$REPORT" --as "$R" \
    --baselines-path "$BASE" --splits-dir "$SPLITS" --allow-dirty || echo "!! lock $R failed"
done

echo "=================== DONE; locked baselines: ==================="
python3 -c "import json;print(json.dumps(json.load(open('$BASE')),indent=1)[:1500])" 2>/dev/null || cat "$BASE"
