#!/usr/bin/env bash
# Lock bm25 / minicoil-v1 / translate-bm25 on the ALL-QUERIES mMARCO slice (val split).
#
# The phase2 gate is carved from the concept-covered slice, which excludes ~22% of
# dev queries -- exactly the ones where v2 has no concept bridge and, cross-lingually,
# often no shared sparse index with the gold at all. This gate keeps every query, so
# a fallback that makes unreachable golds reachable has somewhere to show up.
#
# translate-bm25 will generate the missing NLLB translations on first run (the cache
# was built on the covered slice, so ~25% of these queries are uncached).
set -uo pipefail
cd "$(dirname "$0")/.."

SPLITS=data/eval/splits_mmarco_all
BASE=data/eval/baselines_mmarco_all.json
REPORTS=reports/eval_all
mkdir -p "$REPORTS"

for R in bm25 minicoil-v1 translate-bm25; do
  echo "=================== running baseline: $R ==================="
  uv run minicoil eval run "$R" --dataset mmarco --split val --rebuild \
    --splits-dir "$SPLITS" --baselines-path "$BASE" --reports-dir "$REPORTS" || {
      echo "!! $R run failed, skipping lock"; continue; }
  REPORT=$(ls -t "$REPORTS"/*.json 2>/dev/null | head -1)
  echo "locking $R from $REPORT"
  uv run minicoil eval baselines lock --report "$REPORT" --as "$R" \
    --baselines-path "$BASE" --splits-dir "$SPLITS" --allow-dirty || echo "!! lock $R failed"
done

echo "=================== DONE; locked baselines: ==================="
python3 -c "import json;print(json.dumps(json.load(open('$BASE')),indent=1)[:1500])" 2>/dev/null || cat "$BASE"
