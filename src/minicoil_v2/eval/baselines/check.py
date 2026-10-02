"""Win-condition gate: compare a new eval report against locked baselines."""

from __future__ import annotations

from dataclasses import dataclass

# Hardcoded win condition: (pair, split, metric) -> baseline name to beat.
# Metric is MRR@10, the official MS MARCO metric (dev judgments are ~1 gold/query).
WIN_CONDITION: dict[tuple[str, str, str], str] = {
    ("eng-eng", "test", "mrr10"): "minicoil-v1",
    ("spa-spa", "test", "mrr10"): "bm25",
    ("eng-spa", "test", "mrr10"): "translate-bm25",
    ("spa-eng", "test", "mrr10"): "translate-bm25",
}


@dataclass(frozen=True)
class CheckRow:
    pair: str
    split: str
    metric: str
    baseline_name: str
    new_value: float
    baseline_value: float | None  # None if baseline not locked yet
    passed: bool  # True if beat OR baseline missing


@dataclass(frozen=True)
class CheckOutcome:
    rows: list[CheckRow]
    passed: bool
    exit_code: int


def check_report(*, report: dict, baselines: dict) -> CheckOutcome:
    """Apply WIN_CONDITION to the report and return the outcome.

    Missing baselines (lookup miss) are treated as pass with
    baseline_value=None. Any locked baseline NOT beaten counts as failure.
    """
    rows: list[CheckRow] = []
    overall_passed = True
    for (pair, split, metric), baseline_name in WIN_CONDITION.items():
        new_agg = report.get("results", {}).get(pair, {}).get(split, {}).get(metric)
        if new_agg is None:
            # report doesn't cover this (pair, split, metric); skip as missing
            rows.append(
                CheckRow(
                    pair=pair,
                    split=split,
                    metric=metric,
                    baseline_name=baseline_name,
                    new_value=float("nan"),
                    baseline_value=None,
                    passed=True,
                )
            )
            continue
        new_value = new_agg["mean"]
        baseline_entry = baselines.get(baseline_name, {}).get(pair, {}).get(split, {}).get(metric)
        if baseline_entry is None:
            rows.append(
                CheckRow(
                    pair=pair,
                    split=split,
                    metric=metric,
                    baseline_name=baseline_name,
                    new_value=new_value,
                    baseline_value=None,
                    passed=True,
                )
            )
            continue
        baseline_value = baseline_entry["value"]
        passed = new_value > baseline_value
        if not passed:
            overall_passed = False
        rows.append(
            CheckRow(
                pair=pair,
                split=split,
                metric=metric,
                baseline_name=baseline_name,
                new_value=new_value,
                baseline_value=baseline_value,
                passed=passed,
            )
        )
    return CheckOutcome(
        rows=rows,
        passed=overall_passed,
        exit_code=0 if overall_passed else 1,
    )
