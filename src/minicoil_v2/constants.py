"""Shared constants for miniCOIL v2."""

import re

# --- Encoder models ---
INPUT_ENCODER = "intfloat/multilingual-e5-small"
# Default teacher (mining encoder) for `minicoil embed`: mE5-small, token-pooled around
# the concept word (384D). The teacher's vectors only build each concept's distance
# matrix at training time; the heads always consume a fresh token-pooled mE5-small
# re-encode (train_concept_layers.encode_inputs_token_pooled). The published checkpoint
# overrode this with Qwen/Qwen3-Embedding-0.6B in sentence mode (1024D), so this default
# reproduces an earlier recipe, not the published one.
DEFAULT_EXTRACT_MINING_ENCODER = INPUT_ENCODER
INPUT_DIM = 384
# 4 -> 8: widen each concept head to give the cosine objective more room to
# separate senses (the 4D bottleneck was the named eng-eng limiter vs v1 per-word
# and the spa-eng/eng-spa ranking-compression failure mode). Concept indices stay
# concept_num*OUTPUT_DIM+offset, well below BACKBONE_BASE (1<<27).
OUTPUT_DIM = 8

# --- Tokenization ---
TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
WORD_RE = re.compile(r"[a-záéíóúüñàèìòùâêîôûäëïöü]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# --- Published model artifacts (HF repo layout) ---
# The released checkpoint on the Hugging Face Hub. Used as the eval harness default
# when no local checkpoint is given (MINICOIL_V2_CHECKPOINT / --model-path).
PUBLISHED_MODEL_ID = "Jocana/minicoil-en-es"
# An export directory is a valid encoder `data_dir`: the stacked concept heads live
# in HF_WEIGHTS_FILE, the concept id order and inference defaults in HF_CONFIG_FILE,
# and surface-form lookup in WORD_TO_CONCEPT_FILE (same file the training tree uses).
HF_WEIGHTS_FILE = "model.safetensors"
HF_CONFIG_FILE = "config.json"
WORD_TO_CONCEPT_FILE = "word_to_concept.json"
CONCEPT_VOCABULARY_FILE = "concept_vocabulary.json"
HF_WEIGHTS_KEY = "concept_weights"
HF_BIASES_KEY = "concept_biases"

# --- Sparse encoding ---
SPARSE_EPSILON = 1e-6

# --- Cross-lingual scoring ---
# The BM25 backbone bridges across languages only via cognates/numbers/proper nouns.
# Its original defect: the backbone ran the English stopword list for ALL languages, so
# a query's own function words leaked in. Spanish "de"/"la"/"que" survived, and being
# rare in an English corpus they scored as high-IDF noise (emitted by ~94% of spa-eng
# queries, the dominant rank-11-100 failure). That is fixed at the root by language-aware
# backbone stopwords (encoder.load_lang_stopwords): with the leak gone, spa-eng val MRR@10
# rose 0.55 -> ~0.71 and the backbone no longer needs downweighting.
# This knob remains an env-overridable fallback (MINICOIL_V2_XL_BACKBONE_WEIGHT) that
# scales the backbone for cross-lingual queries (query_lang != doc_lang), query-side only.
# Default 1.0 keeps the backbone fully load-bearing (open-domain recall). An earlier sweep
# found 0.5 was the MRR optimum BEFORE the stopword fix, a crude proxy now superseded.
XL_BACKBONE_WEIGHT = 1.0

# --- Concept vocabulary defaults ---
DEFAULT_MIN_CLUSTER_SIZE = 2
DEFAULT_MAX_CLUSTER_SIZE = 20
DEFAULT_MAX_TRANSLATIONS_PER_WORD = 2
DEFAULT_LOUVAIN_RESOLUTION = 1.5

# --- Training ---
MIN_SENTENCES_PER_CONCEPT = 10
DEFAULT_DROPOUT = 0.05
MIN_TRIPLET_MARGIN = 0.1
# Per-concept margin floor = max(MIN_TRIPLET_MARGIN, MARGIN_SCALE * std(concept distances)).
# A global floor behaves inconsistently across concepts (a fixed 0.1 is the hard-negative
# boundary for spread concepts but unreachable for tight ones). Scaling by each concept's
# own distance std makes the boundary concept-relative. NOTE direction: a LARGER margin
# selects EASIER negatives (closest candidate beyond the floor), so this is an exploration
# knob, not a guaranteed win. Default 0 = legacy fixed floor (known-good hard mining).
MARGIN_SCALE = 0.0
DEFAULT_TRAIN_EPOCH_SIZE = 64000
DEFAULT_VAL_EPOCH_SIZE = 6400
DEFAULT_SAMPLE_BATCH_SIZE = 256

# --- Extraction ---
MIN_SENTENCE_WORDS = 5
MAX_SENTENCE_WORDS = 100
MAX_PER_ARTICLE_PER_CONCEPT = 2
PROGRESS_EVERY = 5000
PROGRESS_EVERY_EARLY = 500
PROGRESS_EARLY_THRESHOLD = 5000
DEFAULT_FLUSH_BUFFER_SIZE = 512
WIKI_DUMP_DATE = "20231101"
MIN_ARTICLE_LENGTH = 50

# --- Known embedding model output dimensions ---
KNOWN_MODEL_DIMS: dict[str, int] = {
    "sentence-transformers/all-minilm-l6-v2": 384,
    "intfloat/multilingual-e5-small": 384,
    "mxbai/embed-large-v1": 512,
    "openrouter/qwen/qwen3-embedding-8b": 4096,
}

# --- Qdrant ---
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = "minicoil_sentences"
DEFAULT_TRIM_WINDOW = 5
QDRANT_SCROLL_LIMIT = 256
