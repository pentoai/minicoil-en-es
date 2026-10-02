"""Verified loader for committed split files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class SplitVerificationError(RuntimeError):
    """Raised when a split file's sha256 doesn't match its manifest entry."""


def load_manifest(splits_dir: Path) -> dict:
    """Return the parsed manifest. Raises FileNotFoundError if missing."""
    manifest_path = splits_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def manifest_sha(splits_dir: Path) -> str:
    """SHA256 of the manifest file bytes (used as splits_manifest_sha in reports)."""
    return hashlib.sha256((splits_dir / "manifest.json").read_bytes()).hexdigest()


def load_split(
    splits_dir: Path,
    *,
    dataset_name: str,
    pair: str,
    split: str,
) -> list[str]:
    """Load and verify a split file. Returns list of qids in stored order."""
    manifest = load_manifest(splits_dir)
    filename = f"{dataset_name}_{pair}_{split}.json"
    file_path = splits_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"split file not found: {file_path}")
    if filename not in manifest.get("files", {}):
        raise SplitVerificationError(f"{filename} is not recorded in manifest.json")
    file_bytes = file_path.read_bytes()
    actual_sha = hashlib.sha256(file_bytes).hexdigest()
    expected_sha = manifest["files"][filename]["sha256"]
    if actual_sha != expected_sha:
        raise SplitVerificationError(
            f"sha256 mismatch for {filename}: "
            f"file={actual_sha[:12]}... manifest={expected_sha[:12]}..."
        )
    return json.loads(file_bytes.decode("utf-8"))
