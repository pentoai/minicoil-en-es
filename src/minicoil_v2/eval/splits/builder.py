"""Deterministic dev/val/test split carving over a Dataset's queries.

Output layout:
  <output_dir>/<dataset_name>_<pair>_dev.json
  <output_dir>/<dataset_name>_<pair>_val.json
  <output_dir>/<dataset_name>_<pair>_test.json
  <output_dir>/manifest.json

Each split file is a JSON list of qid strings. The manifest records seed,
dataset name and revision, and per-file sha256 + counts.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from pathlib import Path


def _split_for_pair(
    dataset,
    pair: str,
    seed: int,
    dev_size: int,
    val_size: int,
) -> tuple[list[str], list[str], list[str]]:
    qids: list[str] = []
    for split in ("validation", "test"):
        qids.extend(dataset.queries(pair, split).keys())
    rng = random.Random(f"{seed}::{dataset.name}::{pair}")
    rng.shuffle(qids)
    if len(qids) < dev_size + val_size:
        raise ValueError(
            f"pair {pair}: only {len(qids)} qids, need >= {dev_size + val_size}",
        )
    dev = qids[:dev_size]
    val = qids[dev_size : dev_size + val_size]
    test = qids[dev_size + val_size :]
    return dev, val, test


def build_splits(
    *,
    dataset,
    output_dir: Path,
    seed: int,
    dev_size: int,
    val_size: int,
    pairs: Iterable[str],
    dataset_revision: str,
    force: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = list(pairs)

    # Existence check (manifest + any split file) BEFORE writing.
    manifest_path = output_dir / "manifest.json"
    if not force:
        existing = [manifest_path] + [
            output_dir / f"{dataset.name}_{p}_{s}.json"
            for p in pairs
            for s in ("dev", "val", "test")
        ]
        for path in existing:
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists; pass force=True to overwrite",
                )

    files_meta: dict[str, dict[str, int | str]] = {}
    for pair in pairs:
        dev, val, test = _split_for_pair(dataset, pair, seed, dev_size, val_size)
        for split, qids in [("dev", dev), ("val", val), ("test", test)]:
            filename = f"{dataset.name}_{pair}_{split}.json"
            payload = json.dumps(qids, sort_keys=False, separators=(",", ":"))
            path = output_dir / filename
            path.write_text(payload, encoding="utf-8")
            files_meta[filename] = {
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "n": len(qids),
            }

    manifest = {
        "seed": seed,
        "dataset_name": dataset.name,
        "dataset_revision": dataset_revision,
        "pairs": pairs,
        "dev_size": dev_size,
        "val_size": val_size,
        "files": files_meta,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
