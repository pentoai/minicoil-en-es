"""Per-module configuration using pydantic-settings."""

from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from minicoil_v2.constants import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DROPOUT,
    DEFAULT_EXTRACT_MINING_ENCODER,
    DEFAULT_FLUSH_BUFFER_SIZE,
    DEFAULT_LOUVAIN_RESOLUTION,
    DEFAULT_MAX_CLUSTER_SIZE,
    DEFAULT_MAX_TRANSLATIONS_PER_WORD,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_QDRANT_URL,
    DEFAULT_SAMPLE_BATCH_SIZE,
    DEFAULT_TRAIN_EPOCH_SIZE,
    DEFAULT_TRIM_WINDOW,
    DEFAULT_VAL_EPOCH_SIZE,
    MARGIN_SCALE,
    MIN_TRIPLET_MARGIN,
)


class VocabBuildSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINICOIL_VOCAB_")

    en_es_path: Path = Path("data/muse/en-es.txt")
    es_en_path: Path = Path("data/muse/es-en.txt")
    output_dir: Path = Path("data")
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE
    max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE
    max_translations_per_word: int = DEFAULT_MAX_TRANSLATIONS_PER_WORD
    resolution: float = DEFAULT_LOUVAIN_RESOLUTION


class SubsetSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINICOIL_SUBSET_")

    concepts_file: Path = Path("data/sample_200_pruned_concepts.json")
    source_dir: Path = Path("data")
    output_dir: Path = Path("data/mini")


class VocabPruneSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINICOIL_PRUNE_")

    threshold: int = 50
    data_dir: Path = Path("data")


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MINICOIL_QDRANT_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    url: str = DEFAULT_QDRANT_URL
    api_key: str | None = None
    collection_name: str = DEFAULT_COLLECTION_NAME
    prefer_grpc: bool = True
    timeout: int = 60


class ScanSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINICOIL_SCAN_")

    data_dir: Path = Path("data")
    max_articles: int = 100_000
    lang: str = "both"


class EmbedSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MINICOIL_EMBED_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = Path("data")
    max_articles: int = 100_000
    store_cap: int = 100
    lang: str = "both"
    mining_encoder: str = DEFAULT_EXTRACT_MINING_ENCODER
    mining_pooling: Literal["token_pooled", "sentence"] = "token_pooled"
    encode_batch_size: int = 64
    flush_buffer_size: int = DEFAULT_FLUSH_BUFFER_SIZE
    upsert_batch_size: int = 500
    upsert_wait: bool = False
    device: str = "auto"
    recreate: bool = False
    cloud_inference: bool = False
    openrouter_api_key: str | None = None
    qdrant: QdrantSettings = QdrantSettings()


class TrainSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINICOIL_TRAIN_")

    data_dir: Path = Path("data")
    output_dir: Path = Path("data/concept_models")
    concepts: str | None = None
    concept_batch_size: int = 50
    device: str = "auto"
    epochs: int = 500
    lr: float = 2e-3
    min_triplet_margin: float = MIN_TRIPLET_MARGIN
    margin_scale: float = MARGIN_SCALE
    max_sentences: int = 2000
    lang_ratio: float = 0.5
    train_epoch_size: int = DEFAULT_TRAIN_EPOCH_SIZE
    val_epoch_size: int = DEFAULT_VAL_EPOCH_SIZE
    sample_batch_size: int = DEFAULT_SAMPLE_BATCH_SIZE
    encode_batch_size: int = 64
    # When set, per-batch scrolled+encoded inputs (mE5 token-pooled vectors + mining
    # vectors + concept row maps) are cached here. They are recipe-independent, so a
    # tuning loop encodes once and reuses across iterations; it also fixes the training
    # rows so recipe comparisons aren't confounded by re-sampled data. None = no cache.
    encode_cache_dir: Path | None = None
    val_size: float = 0.2
    dropout: float = DEFAULT_DROPOUT
    lr_factor: float = 0.5
    lr_patience: int = 5
    resume: bool = False
    # Opt-in batched/GPU-friendly training (see batched_training.py). Default False
    # keeps the reference per-concept path bit-for-bit. When True, a whole
    # concept_batch is stacked and trained with one bmm/step (results equivalent
    # within the pipeline's own sampling/dropout stochasticity, see the perf report).
    batched: bool = False
    optimizer_mode: Literal["vectorized", "per_concept_torch"] = "vectorized"
    # Re-mine triplets every N epochs (reuse the drawn epoch in between). 1 = mine
    # every epoch (reference behavior). >1 cuts per-epoch sampling ~N-fold but
    # reduces negative diversity, so it changes training dynamics: gate on the
    # MRR@10 eval, not the equivalence tests. Used only by the batched path.
    mine_every: int = 1
    wandb_enabled: bool = True
    wandb_project: str = "minicoil-v2"
    wandb_name: str | None = None
    trim_augment_ratio: float = 1.0
    trim_window: int = DEFAULT_TRIM_WINDOW
    # When set, source training rows for each ON-DISK concept from this pre-matched
    # point-id bucket file instead of the cloud's misaligned `concept_ids` payload.
    rematch_buckets: Path | None = None
    qdrant: QdrantSettings = QdrantSettings()

    @model_validator(mode="after")
    def _warn_val_exceeds_train(self) -> "TrainSettings":
        # Latent footgun: lowering --train-epoch-size without lowering
        # --val-epoch-size makes validation do more sampling than training every
        # epoch (it only drives the LR scheduler). Surface it loudly; do not
        # silently clamp (that would change training dynamics behind the user's
        # back). See scripts/perf/README.md.
        if self.val_epoch_size > self.train_epoch_size:
            logger.warning(
                f"val_epoch_size ({self.val_epoch_size}) > train_epoch_size "
                f"({self.train_epoch_size}): validation samples more than training each "
                f"epoch and only feeds the LR scheduler. Consider --val-epoch-size "
                f"<= --train-epoch-size."
            )
        return self
