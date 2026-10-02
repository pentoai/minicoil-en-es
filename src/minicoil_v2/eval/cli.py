"""Typer subapp for `minicoil eval ...` commands."""

from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from minicoil_v2.eval.audit import log_test_access
from minicoil_v2.eval.baselines.check import check_report
from minicoil_v2.eval.baselines.lock import BaselineLockError, lock_baseline
from minicoil_v2.eval.git_utils import current_sha, is_dirty
from minicoil_v2.eval.report import ReportMeta, write_report
from minicoil_v2.eval.retriever import PAIR_LANGS, EvalResult, get
from minicoil_v2.eval.splits.builder import build_splits
from minicoil_v2.eval.splits.loader import load_manifest, load_split, manifest_sha

eval_app = typer.Typer(name="eval", help="Evaluation framework for miniCOIL v2")
splits_app = typer.Typer(help="Manage frozen dev/val/test splits")
baselines_app = typer.Typer(help="Lock/check baseline numbers")
eval_app.add_typer(splits_app, name="splits")
eval_app.add_typer(baselines_app, name="baselines")


DEFAULT_RETRIEVER_MODULES: list[str] = [
    "minicoil_v2.eval.retrievers.bm25",
    "minicoil_v2.eval.retrievers.minicoil_v1",
    "minicoil_v2.eval.retrievers.translate_bm25",
]


class _CarvedSplitView:
    """Wraps a Dataset so `queries(pair, split)` returns only the carved qids.

    The carved dev/val/test splits are produced by `splits/builder.py` as a
    deterministic partition of the union of MLQA's underlying `validation`
    and `test` splits. The base Dataset only knows the underlying split
    names, so we look each carved qid up across both and re-expose it under
    the carved split name. `corpus()` is delegated unchanged.
    """

    def __init__(self, base, pair_qids: dict[str, list[str]]) -> None:
        self._base = base
        self._pair_qids = pair_qids
        self._lookup: dict[str, dict[str, tuple[str, set[str]]]] = {}
        for pair in pair_qids:
            merged: dict[str, tuple[str, set[str]]] = {}
            for underlying in ("validation", "test"):
                queries = base.queries(pair, underlying)
                qrels = base.qrels(pair, underlying)
                for qid, q in queries.items():
                    merged[qid] = (q, set(qrels.get(qid, set())))
            self._lookup[pair] = merged

    @property
    def name(self) -> str:
        return self._base.name

    def corpus(self, lang: str):
        return self._base.corpus(lang)

    def queries(self, pair: str, split: str):  # noqa: ARG002 - split is the carved label
        lookup = self._lookup[pair]
        return {qid: lookup[qid][0] for qid in self._pair_qids[pair] if qid in lookup}

    def qrels(self, pair: str, split: str):  # noqa: ARG002 - split is the carved label
        lookup = self._lookup[pair]
        return {qid: lookup[qid][1] for qid in self._pair_qids[pair] if qid in lookup}


def _make_dataset(name: str):
    """Construct a Dataset by name. New datasets register here."""
    if name == "mlqa":
        from minicoil_v2.eval.datasets.mlqa import MLQADataset

        return MLQADataset()
    if name == "mmarco":
        from minicoil_v2.eval.datasets.mmarco import MMARCODataset

        return MMARCODataset()
    if name == "mmarco_open":
        from minicoil_v2.eval.datasets.mmarco_open import MMARCOOpenDataset

        return MMARCOOpenDataset()
    raise typer.BadParameter(f"unknown dataset: {name!r}. allowed: mlqa, mmarco")


class _CoveredView:
    """Wraps a Dataset so `queries`/`qrels` expose only the concept-covered qids.

    Used when carving the MMARCO splits: the frozen covered slice (from the
    coverage filter) restricts the query pool the splits builder partitions over.
    `corpus` and `revision` are delegated unchanged.
    """

    def __init__(self, base, covered: dict[str, set[str]]) -> None:
        self._base = base
        self._covered = covered

    @property
    def name(self) -> str:
        return self._base.name

    @property
    def revision(self) -> str:
        return self._base.revision

    def corpus(self, lang: str):
        return self._base.corpus(lang)

    def queries(self, pair: str, split: str):
        keep = self._covered.get(pair, set())
        return {k: v for k, v in self._base.queries(pair, split).items() if k in keep}

    def qrels(self, pair: str, split: str):
        keep = self._covered.get(pair, set())
        return {k: v for k, v in self._base.qrels(pair, split).items() if k in keep}


def _load_covered(path: Path) -> dict[str, set[str]]:
    """Load frozen covered qids: {"pairs": {pair: [qid, ...]}} -> {pair: set}."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {pair: set(qids) for pair, qids in payload["pairs"].items()}


def _discover_retriever_modules() -> list[str]:
    """Scan the retrievers package so an experiment is a drop-in file: add
    `retrievers/<name>.py` with `@register(...)` and it is found automatically,
    no edit to any central list. Private modules (leading underscore, e.g.
    `_common`) are skipped since they are helpers, not retrievers."""
    import pkgutil

    from minicoil_v2.eval import retrievers as pkg

    return sorted(
        f"minicoil_v2.eval.retrievers.{info.name}"
        for info in pkgutil.iter_modules(pkg.__path__)
        if not info.name.startswith("_")
    )


def _load_retriever_modules() -> None:
    modules = set(DEFAULT_RETRIEVER_MODULES)
    modules.update(_discover_retriever_modules())
    extra = os.environ.get("MINICOIL_EVAL_RETRIEVER_MODULES", "")
    if extra:
        modules.update(m.strip() for m in extra.split(",") if m.strip())
    for mod in sorted(modules):
        importlib.import_module(mod)


@splits_app.command("build")
def splits_build(
    seed: Annotated[int, typer.Option(help="RNG seed for shuffling")] = 0,
    dev_size: Annotated[int, typer.Option(help="Queries per dev split")] = 200,
    val_size: Annotated[int, typer.Option(help="Queries per val split")] = 2000,
    dataset: Annotated[str, typer.Option(help="mlqa|mmarco")] = "mlqa",
    output_dir: Annotated[Path | None, typer.Option(help="Output directory")] = None,
    covered_qids: Annotated[Path, typer.Option(help="Frozen covered qids (mmarco only)")] = Path(
        "data/eval/mmarco/covered_qids.json"
    ),
    coverage_filter: Annotated[
        bool,
        typer.Option(
            help="Keep only queries sharing a trained concept with their gold (mmarco only)"
        ),
    ] = True,
    force: Annotated[bool, typer.Option(help="Overwrite existing files")] = False,
) -> None:
    """Build deterministic dev/val/test splits over MLQA or the MMARCO covered slice.

    `--no-coverage-filter` carves over ALL queries instead. The covered slice exists
    to isolate where a per-concept head can move the score, but it also hides the
    model's worst case: a cross-lingual query sharing no concept with its gold has
    no sparse index in common with it, so the gold is unreachable at any k. Those
    queries are ~22% of mMARCO dev and never appear in a covered-slice number.
    """
    ds = _make_dataset(dataset)
    if output_dir is None:
        output_dir = Path("data/eval/splits_mmarco" if dataset == "mmarco" else "data/eval/splits")
    if dataset == "mmarco" and coverage_filter:
        ds = _CoveredView(ds, _load_covered(covered_qids))
    build_splits(
        dataset=ds,
        output_dir=output_dir,
        seed=seed,
        dev_size=dev_size,
        val_size=val_size,
        pairs=tuple(PAIR_LANGS),
        dataset_revision=ds.revision,
        force=force,
    )
    typer.echo(f"Wrote splits to {output_dir}")


@eval_app.command("run")
def run(
    retriever_name: Annotated[str, typer.Argument()],
    pair: Annotated[list[str] | None, typer.Option(help="Pairs to evaluate; default all 4")] = None,
    split: Annotated[str, typer.Option(help="dev|val|test")] = "dev",
    dataset: Annotated[str, typer.Option(help="mlqa|mmarco")] = "mlqa",
    k: Annotated[int, typer.Option(help="Retrieve top-k")] = 100,
    reports_dir: Annotated[Path, typer.Option(help="Output dir for reports")] = Path(
        "reports/eval"
    ),
    splits_dir: Annotated[Path | None, typer.Option(help="Frozen splits dir")] = None,
    baselines_path: Annotated[Path | None, typer.Option(help="Locked baselines.json")] = None,
    rebuild: Annotated[bool, typer.Option(help="Re-index even if collection exists")] = False,
    qdrant_url: Annotated[str | None, typer.Option(help="Override QDRANT_URL env")] = None,
) -> None:
    """Run a registered retriever on MLQA/MMARCO splits and write a report."""
    if splits_dir is None:
        splits_dir = Path("data/eval/splits_mmarco" if dataset == "mmarco" else "data/eval/splits")
    if baselines_path is None:
        baselines_path = Path(
            "data/eval/baselines_mmarco.json" if dataset == "mmarco" else "data/eval/baselines.json"
        )
    if split == "test" and not os.environ.get("MINICOIL_EVAL_ALLOW_TEST"):
        typer.echo(
            "the 'test' split is sealed: iterate and tune on --split val, then run "
            "the held-out test gate by setting MINICOIL_EVAL_ALLOW_TEST=1.",
            err=True,
        )
        raise typer.Exit(code=2)
    _load_retriever_modules()
    try:
        cls = get(retriever_name)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    ds = _make_dataset(dataset)
    pairs = pair or list(PAIR_LANGS)
    try:
        instance = cls(qdrant_url=qdrant_url, rebuild=rebuild)
    except TypeError:
        instance = cls()
    # Qdrant-backed retrievers key their collection by dataset so mlqa and mmarco
    # corpora never collide on the same local Qdrant.
    if hasattr(instance, "dataset_tag"):
        instance.dataset_tag = dataset

    git_sha = current_sha()
    dirty = is_dirty()
    if dirty:
        logger.warning("git tree is dirty; report will be marked accordingly")

    splits_sha = manifest_sha(splits_dir)
    manifest = load_manifest(splits_dir)
    if manifest.get("dataset_revision") != ds.revision:
        typer.echo(
            f"dataset revision mismatch: splits manifest says "
            f"{manifest.get('dataset_revision')!r}, loaded dataset is "
            f"{ds.revision!r}. Rebuild splits "
            "(`minicoil eval splits build --force`) before running eval.",
            err=True,
        )
        raise typer.Exit(code=2)

    pair_qids: dict[str, list[str]] = {}
    for p in pairs:
        if split == "test":
            log_test_access(
                retriever=retriever_name,
                pair=p,
                git_sha=git_sha,
                caller="cli",
                reports_dir=reports_dir,
            )
        # Verify the split exists + sha matches before any work
        pair_qids[p] = load_split(splits_dir, dataset_name=ds.name, pair=p, split=split)

    carved = _CarvedSplitView(ds, pair_qids)
    from minicoil_v2.eval.retrievers._common import UnsupportedPairError

    per_pair = {}
    for p in pairs:
        try:
            pr = instance.evaluate(carved, pair=p, split=split, k=k)
        except UnsupportedPairError as exc:
            logger.warning(f"{retriever_name} skipping pair {p}: {exc}")
            typer.echo(f"skipping pair {p}: {exc}")
            continue
        per_pair[p] = pr

    result = EvalResult(retriever=retriever_name, k=k, per_pair=per_pair)
    meta = ReportMeta(
        git_sha=git_sha,
        dirty=dirty,
        command="minicoil " + " ".join(sys.argv[1:]),
        dataset_name=ds.name,
        dataset_revision=ds.revision,
        splits_manifest_sha=splits_sha,
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    baselines = None
    if baselines_path.exists():
        baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
    paths = write_report(result, meta, reports_dir=reports_dir, baselines=baselines)
    typer.echo(f"Wrote {paths.json_path} + {paths.md_path}")


@baselines_app.command("lock")
def baselines_lock_cmd(
    report: Annotated[Path, typer.Option(help="Path to a JSON eval report")],
    as_name: Annotated[str, typer.Option("--as", help="Baseline name to register")],
    baselines_path: Annotated[Path, typer.Option(help="Output baselines.json")] = Path(
        "data/eval/baselines.json"
    ),
    splits_dir: Annotated[Path, typer.Option()] = Path("data/eval/splits"),
    allow_dirty: Annotated[bool, typer.Option()] = False,
) -> None:
    """Lock the baseline numbers from a report into baselines.json."""
    try:
        lock_baseline(
            report_path=report,
            baseline_name=as_name,
            baselines_path=baselines_path,
            current_splits_manifest_sha=manifest_sha(splits_dir),
            allow_dirty=allow_dirty,
        )
    except BaselineLockError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Locked baseline {as_name!r} into {baselines_path}")


@baselines_app.command("check")
def baselines_check_cmd(
    report: Annotated[Path, typer.Option(help="Path to a JSON eval report")],
    baselines_path: Annotated[Path, typer.Option(help="Locked baselines.json")] = Path(
        "data/eval/baselines.json"
    ),
) -> None:
    """Check report vs locked baselines; exit non-zero on regression."""
    if not baselines_path.exists():
        typer.echo(
            f"no baselines locked yet at {baselines_path}; treating as pass",
            err=True,
        )
        raise typer.Exit(code=0)
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
    outcome = check_report(report=report_payload, baselines=baselines)
    for row in outcome.rows:
        if row.baseline_value is None:
            typer.echo(
                f"  {row.pair} {row.split} {row.metric}: ⚠ baseline {row.baseline_name!r} missing"
            )
        else:
            arrow = "↑" if row.passed else "↓"
            typer.echo(
                f"  {row.pair} {row.split} {row.metric}: "
                f"{row.new_value:.4f} vs {row.baseline_value:.4f} "
                f"({row.baseline_name}) {arrow}"
            )
    typer.echo("PASS" if outcome.passed else "FAIL")
    raise typer.Exit(code=outcome.exit_code)
