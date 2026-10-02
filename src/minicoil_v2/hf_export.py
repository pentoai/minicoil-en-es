"""Package a trained miniCOIL v2 checkpoint as a Hugging Face model repository.

The export is a directory that `MiniCoilEncoder` can load directly, so
`from_pretrained("<org>/<repo>")` and `MiniCoilEncoder(export_dir)` are the same
code path — no second inference implementation to keep in sync:

    model.safetensors          stacked concept heads, (num_concepts, 8, 384)
    config.json                concept id order, index layout, inference defaults
    word_to_concept.json       EN/ES surface form -> concept id (read at load time)
    concept_vocabulary.json    concept -> member words (provenance, not read at load)
    README.md                  model card, with metrics rendered from a stamped report

Numbers in the card are read out of an eval report JSON and the locked baselines
file rather than typed in, so a card can never drift from the run it cites.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from minicoil_v2.constants import (
    CONCEPT_VOCABULARY_FILE,
    HF_BIASES_KEY,
    HF_CONFIG_FILE,
    HF_WEIGHTS_FILE,
    HF_WEIGHTS_KEY,
    INPUT_DIM,
    INPUT_ENCODER,
    OUTPUT_DIM,
    WORD_TO_CONCEPT_FILE,
)
from minicoil_v2.encoder import BACKBONE_BASE, BACKBONE_BM25_MODEL, load_concept_layers

# The inference-side import closure of `MiniCoilEncoder`, copied into the published
# repo so a downloader needs only PyPI packages — this repo is not on PyPI, and a
# `pip install git+...` line is dead for anyone outside the source repo. The package
# `__init__` is written fresh rather than copied: the real one imports the Typer CLI,
# which would drag the whole training stack into an inference-only install.
VENDORED_MODULES = (
    "encoder.py",
    "concept_match.py",
    "token_pooling.py",
    "utils.py",
    "constants.py",
)
VENDORED_PACKAGE = "minicoil_v2"
VENDORED_INIT = '''"""miniCOIL v2 inference code, vendored into the published model repository.

Kept in sync with the source repo by `minicoil export-hf`; the encoder here is the
same file the published numbers were measured with. Training, evaluation and CLI
code are deliberately absent.
"""
'''

# Which baseline each pair is judged against in the card. Monolingual pairs compare
# with BM25 (and miniCOIL v1 where it applies, English only); cross-lingual pairs
# compare with translate-then-BM25, an NLLB-equipped baseline v2 needs no MT to beat.
CARD_COMPARATORS: dict[str, tuple[str, ...]] = {
    "eng-eng": ("bm25", "minicoil-v1"),
    "spa-spa": ("bm25",),
    "eng-spa": ("translate-bm25",),
    "spa-eng": ("translate-bm25",),
}
PAIR_LABELS = {
    "eng-eng": "en → en",
    "spa-spa": "es → es",
    "eng-spa": "en → es",
    "spa-eng": "es → en",
}
METRIC_KEYS = (("mrr10", "MRR@10"), ("ndcg10", "nDCG@10"), ("r100", "R@100"))


def build_config(
    concept_ids: list[str],
    *,
    lemma_match: bool,
    base_encoder: str = INPUT_ENCODER,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The published `config.json`: everything an encoder needs that isn't a tensor.

    `concept_ids` is load-bearing, not metadata — safetensors stores one stacked
    tensor, so this list is what maps row i back to its concept.
    """
    return {
        "model_type": "minicoil-v2",
        "base_encoder": base_encoder,
        "languages": ["en", "es"],
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "num_concepts": len(concept_ids),
        "lemma_match": lemma_match,
        "sparse_index_layout": {
            "concept_block": "concept_num * output_dim + offset",
            "backbone_base": BACKBONE_BASE,
            "backbone_term": "backbone_base + abs(mmh3.hash(stem)) % (2**31 - 1 - backbone_base)",
            "backbone_model": BACKBONE_BM25_MODEL,
        },
        "requires_idf": True,
        "concept_ids": concept_ids,
        "source": source or {},
    }


def _metric_rows(
    report: dict[str, Any],
    baselines: dict[str, Any],
    split: str,
    metrics: tuple[tuple[str, str], ...] = METRIC_KEYS,
) -> list[str]:
    """Markdown table rows: one per (pair, metric), with the delta against each
    comparator that has a locked baseline for that pair and split."""
    rows: list[str] = []
    for pair, label in PAIR_LABELS.items():
        pair_result = report.get("results", {}).get(pair, {}).get(split)
        if pair_result is None:
            continue
        for key, metric_label in metrics:
            # A report that didn't compute a metric simply has no row; the card
            # reports what the run measured rather than inventing a placeholder.
            if key not in pair_result:
                continue
            value = pair_result[key]["mean"]
            deltas = []
            for comparator in CARD_COMPARATORS.get(pair, ()):
                # Locked baselines store the number under "value" alongside the
                # provenance of the run that produced it; live reports use "mean".
                base = baselines.get(comparator, {}).get(pair, {}).get(split, {}).get(key)
                if base is None:
                    continue
                deltas.append(f"{value - base['value']:+.4f} vs {comparator}")
            rows.append(
                f"| {label} | {metric_label} | {value:.4f} | "
                f"{pair_result['n_queries']} | {', '.join(deltas) or '—'} |"
            )
    return rows


def render_model_card(
    config: dict[str, Any],
    *,
    repo_id: str,
    license_id: str,
    report: dict[str, Any] | None = None,
    baselines: dict[str, Any] | None = None,
    split: str = "test",
    vendored: bool = True,
) -> str:
    """A short card: what the model is, how to run it, what breaks it.

    Deliberately minimal — the design rationale lives in the source repo, not here.
    Metrics still come from `report` rather than prose, so they cannot drift.
    """
    lines = [
        "---",
        f"license: {license_id}",
        "language:",
        "- en",
        "- es",
        f"base_model: {config['base_encoder']}",
        "library_name: minicoil-en-es",
        "pipeline_tag: feature-extraction",
        "tags:",
        "- sparse-retrieval",
        "- cross-lingual",
        "- minicoil",
        "- bm25",
        "---",
        "",
        "# miniCOIL EN-ES",
        "",
        "Bilingual sparse retrieval for English and Spanish: BM25 with a learned semantic",
        f"overlay. Words in one of the model's {config['num_concepts']} bilingual concepts map",
        f"to a block of {config['output_dim']} values from that concept's trained head; every",
        "other token stays a plain BM25 term. Translations share a concept, so `dog` and `perro`",
        "match across languages with no translation step. Output is an ordinary sparse",
        "vector.",
        "",
        "## Usage",
        "",
        "```bash",
        "pip install torch transformers fastembed safetensors mmh3 simplemma huggingface-hub",
        "```",
        "",
        "```python",
    ]
    if vendored:
        lines += [
            "import sys",
            "",
            "from huggingface_hub import snapshot_download",
            "",
            f'model_dir = snapshot_download("{repo_id}")  # ships its own encoder code',
            "sys.path.insert(0, model_dir)",
            "",
            "from minicoil_v2.encoder import MiniCoilEncoder",
            "",
            "encoder = MiniCoilEncoder.from_pretrained(model_dir)",
        ]
    else:
        lines += [
            "from minicoil_v2.encoder import MiniCoilEncoder",
            "",
            f'encoder = MiniCoilEncoder.from_pretrained("{repo_id}")',
        ]
    lines += [
        "",
        'doc = encoder.encode_sparse("El perro corrió por el parque", lang="es")',
        'query = encoder.encode_sparse("dog running", lang="en", is_query=True)',
        "```",
        "",
        "`encode_sparse` returns `{sparse_index: value}`. Encode documents in the corpus",
        "language and queries in the query language, and pass `is_query=True` for queries.",
        "",
        "**The index must apply IDF** — in Qdrant, declare the sparse vector with",
        "`Modifier.IDF`. These are BM25 vectors and carry none of their own; without it,",
        "ranking quality collapses.",
        "",
    ]

    if report is not None:
        rows = _metric_rows(report, baselines or {}, split, metrics=(("mrr10", "MRR@10"),))
        lines += [
            f"## Results — mMARCO {split}",
            "",
            "| pair | metric | value | queries | vs baseline |",
            "|------|--------|-------|---------|-------------|",
            *rows,
            "",
            "Measured on the concept-covered slice of mMARCO (queries sharing a concept with",
            "their gold passage). `translate-bm25` is NLLB translation followed by BM25.",
            "",
        ]

    lines += [
        "## Limitations",
        "",
        "- Cross-lingual matching happens only through shared concepts. A query whose",
        "  content words are all out of vocabulary can return **no results** against a",
        "  corpus in the other language. Same-language retrieval always keeps full BM25",
        "  behavior.",
        "- Documents are truncated at 256 tokens by the base encoder.",
        "",
        "## License",
        "",
        f"`{license_id}` — the concept vocabulary derives from the "
        "[MUSE](https://github.com/facebookresearch/MUSE) bilingual dictionaries, which "
        "are non-commercial.",
        "",
        f"Base encoder: [`{config['base_encoder']}`]"
        f"(https://huggingface.co/{config['base_encoder']}) (MIT). "
        "Trained on Wikipedia (EN + ES).",
        "",
    ]
    return "\n".join(lines)


def vendor_inference_code(out_dir: Path, package_root: Path | None = None) -> Path:
    """Copy the encoder's import closure into `out_dir/minicoil_v2/` and return it."""
    package_root = package_root or Path(__file__).resolve().parent
    target = out_dir / VENDORED_PACKAGE
    target.mkdir(parents=True, exist_ok=True)
    (target / "__init__.py").write_text(VENDORED_INIT)
    for module in VENDORED_MODULES:
        shutil.copyfile(package_root / module, target / module)
    return target


def build_export(
    *,
    checkpoint: Path,
    data_dir: Path,
    out_dir: Path,
    repo_id: str,
    lemma_match: bool = True,
    license_id: str = "cc-by-nc-4.0",  # MUSE-derived vocabulary; see THIRD_PARTY_NOTICES.md
    vendor_code: bool = True,
    report_path: Path | None = None,
    baselines_path: Path | None = None,
    split: str = "test",
    source: dict[str, Any] | None = None,
) -> Path:
    """Write a publishable model directory to `out_dir` and return it."""
    concept_ids, weights, biases = load_concept_layers(
        data_dir=checkpoint.parent, model_path=checkpoint
    )
    logger.info(f"loaded {len(concept_ids)} concept heads from {checkpoint}")

    out_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file

    tensors = {HF_WEIGHTS_KEY: weights.contiguous().to(torch.float32)}
    if biases is not None:
        tensors[HF_BIASES_KEY] = biases.contiguous().to(torch.float32)
    save_file(tensors, str(out_dir / HF_WEIGHTS_FILE))

    for filename in (WORD_TO_CONCEPT_FILE, CONCEPT_VOCABULARY_FILE):
        src = data_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"{src} is required for the export")
        shutil.copyfile(src, out_dir / filename)

    if vendor_code:
        vendor_inference_code(out_dir)

    config = build_config(concept_ids, lemma_match=lemma_match, source=source or {})
    with open(out_dir / HF_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    report = json.loads(report_path.read_text()) if report_path else None
    baselines = json.loads(baselines_path.read_text()) if baselines_path else None
    card = render_model_card(
        config,
        repo_id=repo_id,
        license_id=license_id,
        report=report,
        baselines=baselines,
        split=split,
        vendored=vendor_code,
    )
    (out_dir / "README.md").write_text(card)

    logger.info(f"export written to {out_dir}")
    return out_dir


def push_export(
    out_dir: Path,
    repo_id: str,
    *,
    private: bool = True,
    token: str | None = None,
    commit_message: str = "Upload miniCOIL v2 model",
) -> str:
    """Create (if needed) and upload the export directory. Returns the repo URL."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(out_dir),
        commit_message=commit_message,
    )
    url = f"https://huggingface.co/{repo_id}"
    logger.info(f"pushed {out_dir} -> {url}")
    return url
