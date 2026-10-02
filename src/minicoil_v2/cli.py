"""Unified CLI for miniCOIL v2."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from minicoil_v2.constants import (
    DEFAULT_EXTRACT_MINING_ENCODER,
    DEFAULT_FLUSH_BUFFER_SIZE,
)


class MiningPooling(StrEnum):
    token_pooled = "token_pooled"
    sentence = "sentence"


app = typer.Typer(name="minicoil", help="MiniCOIL v2 — sparse multilingual embeddings")
vocab_app = typer.Typer(help="Concept vocabulary commands")
app.add_typer(vocab_app, name="vocab")

from minicoil_v2.eval.cli import eval_app  # noqa: E402

app.add_typer(eval_app, name="eval")


@vocab_app.command("build")
def vocab_build(
    en_es_path: Annotated[Path, typer.Option(help="MUSE en-es dictionary")] = Path(
        "data/muse/en-es.txt"
    ),
    es_en_path: Annotated[Path, typer.Option(help="MUSE es-en dictionary")] = Path(
        "data/muse/es-en.txt"
    ),
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path("data"),
    min_cluster_size: Annotated[int, typer.Option(help="Min words per concept")] = 2,
    max_cluster_size: Annotated[int, typer.Option(help="Max words per concept")] = 20,
    max_translations: Annotated[
        int, typer.Option("--max-translations", help="Top-K translations per word")
    ] = 2,
    resolution: Annotated[
        float, typer.Option(help="Louvain resolution (higher = smaller clusters)")
    ] = 1.5,
) -> None:
    """Build concept vocabulary from MUSE bilingual dictionaries."""
    from minicoil_v2.build_concept_vocab import build_and_save
    from minicoil_v2.settings import VocabBuildSettings

    settings = VocabBuildSettings(
        en_es_path=en_es_path,
        es_en_path=es_en_path,
        output_dir=output_dir,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        max_translations_per_word=max_translations,
        resolution=resolution,
    )
    build_and_save(settings)


@vocab_app.command("subset")
def vocab_subset(
    concepts_file: Annotated[Path, typer.Option(help="JSON file with concept_ids list")] = Path(
        "data/sample_200_pruned_concepts.json"
    ),
    source_dir: Annotated[
        Path, typer.Option(help="Source data directory (full vocabulary)")
    ] = Path("data"),
    output_dir: Annotated[Path, typer.Option(help="Output directory for subset vocabulary")] = Path(
        "data/mini"
    ),
) -> None:
    """Extract a concept subset into a new data directory for isolated pipeline runs."""
    from minicoil_v2.settings import SubsetSettings
    from minicoil_v2.subset_vocab import run_subset

    settings = SubsetSettings(
        concepts_file=concepts_file,
        source_dir=source_dir,
        output_dir=output_dir,
    )
    run_subset(settings)


@vocab_app.command("prune")
def vocab_prune(
    threshold: Annotated[int, typer.Option(help="Min sentences in BOTH languages")] = 50,
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
) -> None:
    """Prune concept vocabulary by sentence frequency."""
    from minicoil_v2.prune_concept_vocab import prune_and_save
    from minicoil_v2.settings import VocabPruneSettings

    settings = VocabPruneSettings(threshold=threshold, data_dir=data_dir)
    prune_and_save(settings)


@app.command("scan")
def scan_sentences(
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
    max_articles: Annotated[int, typer.Option(help="Max Wikipedia articles")] = 100_000,
    lang: Annotated[str, typer.Option(help="Language(s): en, es, or both")] = "both",
) -> None:
    """Scan Wikipedia and count concept occurrences per language."""
    from minicoil_v2.scan_wiki_sentences import run_scan
    from minicoil_v2.settings import ScanSettings

    settings = ScanSettings(
        data_dir=data_dir,
        max_articles=max_articles,
        lang=lang,
    )
    run_scan(settings)


@app.command("embed")
def embed_sentences(
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
    max_articles: Annotated[int, typer.Option(help="Max Wikipedia articles")] = 100_000,
    store_cap: Annotated[
        int, typer.Option(help="Max stored sentences per concept per language")
    ] = 100,
    lang: Annotated[str, typer.Option(help="Language(s): en, es, or both")] = "both",
    mining_encoder: Annotated[
        str, typer.Option(help="Mining encoder model name")
    ] = DEFAULT_EXTRACT_MINING_ENCODER,
    mining_pooling: Annotated[
        MiningPooling, typer.Option(help="Mining pooling mode")
    ] = MiningPooling.token_pooled,
    encode_batch_size: Annotated[int, typer.Option(help="Encoding batch size")] = 64,
    flush_buffer_size: Annotated[
        int, typer.Option(help="Buffer size triggering encode+upsert flush")
    ] = DEFAULT_FLUSH_BUFFER_SIZE,
    upsert_batch_size: Annotated[int, typer.Option(help="Qdrant upsert batch size")] = 500,
    upsert_wait: Annotated[
        bool,
        typer.Option("--upsert-wait/--no-upsert-wait", help="Wait for Qdrant acks"),
    ] = False,
    device: Annotated[str, typer.Option(help="Device: auto, mps, cuda, cpu")] = "auto",
    recreate: Annotated[bool, typer.Option(help="Drop and recreate Qdrant collection")] = False,
    qdrant_url: Annotated[
        str | None, typer.Option(help="Qdrant URL (overrides MINICOIL_QDRANT_URL)")
    ] = None,
    qdrant_api_key: Annotated[
        str | None, typer.Option(help="Qdrant API key (overrides MINICOIL_QDRANT_API_KEY)")
    ] = None,
    collection_name: Annotated[
        str | None,
        typer.Option(help="Qdrant collection name (overrides MINICOIL_QDRANT_COLLECTION_NAME)"),
    ] = None,
    cloud_inference: Annotated[
        bool | None,
        typer.Option(
            "--cloud-inference/--no-cloud-inference",
            help="Use Qdrant Cloud Inference",
        ),
    ] = None,
) -> None:
    """Encode Wikipedia sentences with mining encoder, upsert to Qdrant."""
    from minicoil_v2.embed_wiki_sentences import run_embedding
    from minicoil_v2.settings import EmbedSettings, QdrantSettings

    use_cloud = cloud_inference is True

    if use_cloud:
        qdrant_overrides: dict[str, object] = {
            k: v
            for k, v in {
                "url": qdrant_url,
                "api_key": qdrant_api_key,
                "collection_name": collection_name,
            }.items()
            if v is not None
        }
        qdrant = QdrantSettings(**qdrant_overrides)  # type: ignore[arg-type]
    else:
        from minicoil_v2.constants import DEFAULT_QDRANT_URL

        qdrant = QdrantSettings(
            url=qdrant_url or DEFAULT_QDRANT_URL,
            api_key=None,
            collection_name=collection_name or "minicoil_sentences",
        )

    settings = EmbedSettings(
        data_dir=data_dir,
        max_articles=max_articles,
        store_cap=store_cap,
        lang=lang,
        mining_encoder=mining_encoder,
        mining_pooling=mining_pooling.value,
        encode_batch_size=encode_batch_size,
        flush_buffer_size=flush_buffer_size,
        upsert_batch_size=upsert_batch_size,
        upsert_wait=upsert_wait,
        device=device,
        recreate=recreate,
        cloud_inference=use_cloud,
        qdrant=qdrant,
    )
    run_embedding(settings)


@app.command("train")
def train(
    data_dir: Annotated[Path, typer.Option(help="Data directory")] = Path("data"),
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path(
        "data/concept_models"
    ),
    concepts: Annotated[str | None, typer.Option(help="Concept range, e.g. 0:10")] = None,
    concept_batch_size: Annotated[int, typer.Option(help="Concepts per batch")] = 50,
    device: Annotated[str, typer.Option(help="Device: auto, mps, cuda, cpu")] = "auto",
    epochs: Annotated[int, typer.Option(help="Training epochs per concept")] = 500,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 2e-3,
    min_triplet_margin: Annotated[
        float, typer.Option(help="Absolute floor for the mined-sample margin")
    ] = 0.1,
    margin_scale: Annotated[
        float,
        typer.Option(
            help="Per-concept margin = max(floor, margin_scale*std of dists); 0=fixed floor"
        ),
    ] = 0.0,
    max_sentences: Annotated[
        int, typer.Option(help="Max sentences per concept per language")
    ] = 2000,
    lang_ratio: Annotated[
        float, typer.Option(help="Target EN ratio (0.5 = equal EN/ES balance)")
    ] = 0.5,
    train_epoch_size: Annotated[
        int, typer.Option(help="5-point samples per training epoch")
    ] = 64_000,
    val_epoch_size: Annotated[
        int, typer.Option(help="5-point samples per validation epoch")
    ] = 6_400,
    sample_batch_size: Annotated[int, typer.Option(help="Samples per mini-batch")] = 256,
    encode_batch_size: Annotated[int, typer.Option(help="Encoding batch size")] = 64,
    encode_cache_dir: Annotated[
        Path | None,
        typer.Option(help="Cache scrolled+encoded inputs here; reused across runs (loop speedup)"),
    ] = None,
    val_size: Annotated[float, typer.Option(help="Validation split ratio")] = 0.2,
    dropout: Annotated[float, typer.Option(help="Dropout before linear layer")] = 0.05,
    lr_factor: Annotated[float, typer.Option(help="ReduceLROnPlateau multiplicative factor")] = 0.5,
    lr_patience: Annotated[int, typer.Option(help="ReduceLROnPlateau patience (epochs)")] = 5,
    resume: Annotated[bool, typer.Option(help="Resume from checkpoints")] = False,
    batched: Annotated[
        bool,
        typer.Option(
            "--batched/--no-batched",
            help="Use the batched/GPU training path (one bmm/step per concept bucket); "
            "results equivalent within sampling/dropout noise. ~2x end-to-end at "
            "production scale (sampling-bound, not kernel-bound; see scripts/perf/README.md)",
        ),
    ] = False,
    mine_every: Annotated[
        int,
        typer.Option(
            help="Re-mine triplets every N epochs (batched path); >1 cuts sampling cost "
            "but changes dynamics, so re-check the MRR@10 eval. 1 = mine every epoch"
        ),
    ] = 1,
    trim_augment_ratio: Annotated[
        float, typer.Option(help="Fraction of sentences to add trimmed copies for")
    ] = 1.0,
    trim_window: Annotated[
        int, typer.Option(help="Words on each side of target word for trimming")
    ] = 5,
    rematch_buckets: Annotated[
        Path | None,
        typer.Option(help="Pre-matched point-id buckets (on-disk vocab) to source rows from"),
    ] = None,
    qdrant_url: Annotated[
        str | None, typer.Option(help="Qdrant URL (overrides MINICOIL_QDRANT_URL)")
    ] = None,
    qdrant_api_key: Annotated[
        str | None, typer.Option(help="Qdrant API key (overrides MINICOIL_QDRANT_API_KEY)")
    ] = None,
    collection_name: Annotated[
        str | None,
        typer.Option(help="Qdrant collection name (overrides MINICOIL_QDRANT_COLLECTION_NAME)"),
    ] = None,
    wandb: Annotated[bool, typer.Option("--wandb/--no-wandb", help="Enable W&B logging")] = True,
    wandb_project: Annotated[str, typer.Option(help="W&B project name")] = "minicoil-v2",
    wandb_name: Annotated[str | None, typer.Option(help="W&B run name")] = None,
) -> None:
    """Train per-concept linear layers with bilingual cosine triplet loss."""
    from minicoil_v2.settings import QdrantSettings, TrainSettings
    from minicoil_v2.train_concept_layers import run_training

    qdrant_overrides: dict[str, object] = {
        k: v
        for k, v in {
            "url": qdrant_url,
            "api_key": qdrant_api_key,
            "collection_name": collection_name,
        }.items()
        if v is not None
    }

    settings = TrainSettings(
        data_dir=data_dir,
        output_dir=output_dir,
        concepts=concepts,
        concept_batch_size=concept_batch_size,
        device=device,
        epochs=epochs,
        lr=lr,
        min_triplet_margin=min_triplet_margin,
        margin_scale=margin_scale,
        max_sentences=max_sentences,
        lang_ratio=lang_ratio,
        train_epoch_size=train_epoch_size,
        val_epoch_size=val_epoch_size,
        sample_batch_size=sample_batch_size,
        encode_batch_size=encode_batch_size,
        encode_cache_dir=encode_cache_dir,
        val_size=val_size,
        dropout=dropout,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
        resume=resume,
        batched=batched,
        mine_every=mine_every,
        trim_augment_ratio=trim_augment_ratio,
        trim_window=trim_window,
        rematch_buckets=rematch_buckets,
        wandb_enabled=wandb,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        qdrant=QdrantSettings(**qdrant_overrides),  # type: ignore[arg-type]
    )
    run_training(settings)


@app.command("export-hf")
def export_hf(
    checkpoint: Annotated[Path, typer.Option(help="Trained concept_layers.pt to publish")],
    data_dir: Annotated[Path, typer.Option(help="Vocabulary dir (word_to_concept.json)")],
    out: Annotated[Path, typer.Option(help="Export directory to write")],
    repo_id: Annotated[str, typer.Option(help="HF repo id, e.g. Jocana/minicoil-en-es")],
    lemma_match: Annotated[
        bool, typer.Option(help="Record lemma-fallback matching as the inference default")
    ] = True,
    license_id: Annotated[
        str, typer.Option("--license", help="License id for the card (HF license tag)")
    ] = "cc-by-nc-4.0",
    vendor_code: Annotated[
        bool, typer.Option(help="Copy the encoder's import closure into the export")
    ] = True,
    report: Annotated[
        Path | None, typer.Option(help="Eval report JSON whose metrics go in the card")
    ] = None,
    baselines: Annotated[
        Path | None, typer.Option(help="Locked baselines JSON for the card's deltas")
    ] = None,
    split: Annotated[str, typer.Option(help="Split to report in the card")] = "test",
    push: Annotated[bool, typer.Option(help="Upload to the Hub after building")] = False,
    private: Annotated[bool, typer.Option(help="Create the Hub repo private")] = True,
) -> None:
    """Package a checkpoint as a Hugging Face model repo (and optionally upload it)."""
    from minicoil_v2.eval.git_utils import current_sha, is_dirty
    from minicoil_v2.hf_export import build_export, push_export

    # simplemma version is pinned into the export because lemma matching is part of
    # the scoring path: a different lemmatizer version can change which concepts fire.
    simplemma_version = None
    if lemma_match:
        import importlib.metadata

        simplemma_version = importlib.metadata.version("simplemma")

    source = {
        "checkpoint": str(checkpoint),
        "simplemma_version": simplemma_version,
        "data_dir": str(data_dir),
        "git_sha": current_sha(),
        "git_dirty": is_dirty(),
        "eval_report": str(report) if report else None,
    }
    out_dir = build_export(
        checkpoint=checkpoint,
        data_dir=data_dir,
        out_dir=out,
        repo_id=repo_id,
        lemma_match=lemma_match,
        license_id=license_id,
        vendor_code=vendor_code,
        report_path=report,
        baselines_path=baselines,
        split=split,
        source=source,
    )
    if push:
        push_export(out_dir, repo_id, private=private)


if __name__ == "__main__":
    app()
