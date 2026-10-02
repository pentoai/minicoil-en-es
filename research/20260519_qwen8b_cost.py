"""Cost estimate for miniCOIL embed using openrouter qwen/qwen3-embedding-8b.

Methodology:
  - Load the pruned concept vocabulary
  - Randomly sample 200 concepts
  - Stream Wikipedia (EN + ES) and collect up to SENTENCES_PER_CONCEPT sentences
    per sampled concept
  - Tokenize each sentence with the Qwen3-Embedding-8B tokenizer
  - Compute average tokens/sentence, then extrapolate to full pipeline:
      full_vocab_size x TARGET_SENTENCES_PER_CONCEPT x avg_tokens x cost_per_token

Cost: $0.01 / 1M tokens  (openrouter qwen/qwen3-embedding-8b)

Resumability:
  - Progress is saved to data/cost_estimate_checkpoint.json every 50k sentences
  - If interrupted, run this script again to resume from the checkpoint
  - To start fresh, delete the checkpoint: rm data/cost_estimate_checkpoint.json
  - On successful completion, the checkpoint is automatically cleaned up
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset  # type: ignore[import-untyped]
from loguru import logger
from transformers import AutoTokenizer

from minicoil_v2.constants import (  # type: ignore[import-untyped]
    MAX_SENTENCE_WORDS,
    MIN_ARTICLE_LENGTH,
    MIN_SENTENCE_WORDS,
    WIKI_DUMP_DATE,
)
from minicoil_v2.scan_wiki_sentences import split_sentences  # type: ignore[import-untyped]
from minicoil_v2.utils import tokenize_alpha  # type: ignore[import-untyped]

# ── Config ────────────────────────────────────────────────────────────────────
VOCAB_PATH = Path("data/concept_vocabulary.json")
CHECKPOINT_PATH = Path("data/cost_estimate_checkpoint.json")
CHECKPOINT_SAVE_INTERVAL = 50_000  # Save every N sentences
SAMPLE_CONCEPTS = 200
SENTENCES_PER_CONCEPT = 10_000
COST_PER_MILLION_TOKENS = 0.01  # USD
LANGUAGES = ["en", "es"]
SEED = 42

# ── Load vocab ─────────────────────────────────────────────────────────────────
logger.info(f"Loading concept vocabulary from {VOCAB_PATH}")
with open(VOCAB_PATH) as f:
    vocab_data = json.load(f)

all_concepts: dict[str, dict] = vocab_data["concepts"]
full_vocab_size = len(all_concepts)
stats = vocab_data.get("stats", {})
pruning = stats.get("pruning", {})
if pruning:
    logger.info(
        f"Vocabulary: {full_vocab_size:,} concepts "
        f"(pruned from {pruning['original_concepts']:,} at threshold={pruning['threshold']}, "
        f"{pruning['kept_pct']}% kept)"
    )
else:
    logger.info(f"Vocabulary: {full_vocab_size:,} concepts (no pruning metadata found)")

# ── Sample 200 concepts ────────────────────────────────────────────────────────
rng = random.Random(SEED)
sampled_ids = rng.sample(list(all_concepts.keys()), SAMPLE_CONCEPTS)
sampled_concepts = {cid: all_concepts[cid] for cid in sampled_ids}

# Build per-language word→concept_id lookup for fast sentence matching
word_to_concept: dict[str, dict[str, str]] = {"en": {}, "es": {}}
for cid, entry in sampled_concepts.items():
    for word in entry.get("en", []):
        word_to_concept["en"][word.lower()] = cid
    for word in entry.get("es", []):
        word_to_concept["es"][word.lower()] = cid

logger.info(
    f"Sampled {SAMPLE_CONCEPTS} concepts — "
    f"EN words: {len(word_to_concept['en'])}, "
    f"ES words: {len(word_to_concept['es'])}"
)

# ── Load tokenizer ─────────────────────────────────────────────────────────────
logger.info("Loading Qwen3-Embedding-8B tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B", padding_side="left")
logger.info("Tokenizer loaded.")

# ── Load checkpoint if it exists ───────────────────────────────────────────────
concept_token_counts: dict[str, list[int]] = defaultdict(list)
concept_sentence_counts: dict[str, int] = defaultdict(int)
total_sentences_collected = 0

if CHECKPOINT_PATH.exists():
    logger.info(f"Loading checkpoint from {CHECKPOINT_PATH}")
    with open(CHECKPOINT_PATH) as f:
        checkpoint = json.load(f)

    if checkpoint.get("seed") != SEED or checkpoint.get("sampled_ids") != sampled_ids:
        logger.warning("Checkpoint SEED or sampled concepts differ — starting fresh")
        CHECKPOINT_PATH.unlink()
    else:
        concept_sentence_counts = {
            k: int(v) for k, v in checkpoint["concept_sentence_counts"].items()
        }
        concept_token_counts = {k: v for k, v in checkpoint["concept_token_counts"].items()}
        total_sentences_collected = checkpoint["total_sentences_collected"]
        logger.info(
            f"Resumed from checkpoint: {total_sentences_collected:,} sentences, "
            f"{sum(1 for c in concept_sentence_counts.values() if c > 0)} concepts with data"
        )


def save_checkpoint() -> None:
    """Save current progress to checkpoint file."""
    checkpoint = {
        "seed": SEED,
        "sampled_ids": sampled_ids,
        "total_sentences_collected": total_sentences_collected,
        "concept_sentence_counts": concept_sentence_counts,
        "concept_token_counts": concept_token_counts,
    }
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f)
    logger.info(f"Checkpoint saved: {total_sentences_collected:,} sentences collected")


t0 = time.perf_counter()

for lang in LANGUAGES:
    w2c = word_to_concept[lang]
    known_words = set(w2c.keys())

    logger.info(f"[{lang}] Streaming wikimedia/wikipedia {WIKI_DUMP_DATE}.{lang} ...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        f"{WIKI_DUMP_DATE}.{lang}",
        split="train",
        streaming=True,
    )

    def concepts_still_needed() -> bool:
        return any(concept_sentence_counts[cid] < SENTENCES_PER_CONCEPT for cid in sampled_ids)

    for article in dataset:
        if not concepts_still_needed():
            break

        text = article.get("text", "")
        if not text or len(text) < MIN_ARTICLE_LENGTH:
            continue

        for sentence in split_sentences(text):
            words = tokenize_alpha(sentence)
            if not (MIN_SENTENCE_WORDS <= len(words) <= MAX_SENTENCE_WORDS):
                continue

            matched_concepts = {
                w2c[w]
                for w in words & known_words
                if concept_sentence_counts[w2c[w]] < SENTENCES_PER_CONCEPT
            }
            if not matched_concepts:
                continue

            token_count = len(tokenizer.encode(sentence, add_special_tokens=True))

            for cid in matched_concepts:
                if concept_sentence_counts[cid] < SENTENCES_PER_CONCEPT:
                    concept_token_counts[cid].append(token_count)
                    concept_sentence_counts[cid] += 1
                    total_sentences_collected += 1

        if (
            total_sentences_collected % CHECKPOINT_SAVE_INTERVAL == 0
            and total_sentences_collected > 0
        ):
            elapsed = time.perf_counter() - t0
            covered = sum(1 for cid in sampled_ids if concept_sentence_counts[cid] > 0)
            logger.info(
                f"[{lang}] {total_sentences_collected:,} sentences | "
                f"{covered}/{SAMPLE_CONCEPTS} concepts covered | "
                f"{elapsed:.0f}s elapsed"
            )
            save_checkpoint()

    covered = sum(1 for cid in sampled_ids if concept_sentence_counts[cid] > 0)
    logger.info(
        f"[{lang}] Done. {total_sentences_collected:,} sentences total, "
        f"{covered}/{SAMPLE_CONCEPTS} concepts with >=1 sentence"
    )

elapsed = time.perf_counter() - t0
logger.info(f"Wikipedia streaming complete in {elapsed:.1f}s")
save_checkpoint()

# ── Statistics ─────────────────────────────────────────────────────────────────
all_token_counts = [t for counts in concept_token_counts.values() for t in counts]

if not all_token_counts:
    logger.error("No sentences collected — check vocabulary paths and Wikipedia availability.")
    raise SystemExit(1)

avg_tokens = sum(all_token_counts) / len(all_token_counts)
median_tokens = sorted(all_token_counts)[len(all_token_counts) // 2]
p95_tokens = sorted(all_token_counts)[int(len(all_token_counts) * 0.95)]

sentences_per_concept_actual = [concept_sentence_counts[cid] for cid in sampled_ids]
concepts_with_data = sum(1 for c in sentences_per_concept_actual if c > 0)
avg_sentences_per_concept = sum(sentences_per_concept_actual) / len(sentences_per_concept_actual)


# ── Cost projection ────────────────────────────────────────────────────────────
def cost_row(label: str, n_concepts: int, sents_per_concept: float) -> None:
    total_sents = n_concepts * sents_per_concept
    total_tokens = total_sents * avg_tokens
    total_cost = total_tokens / 1_000_000 * COST_PER_MILLION_TOKENS
    token_str = (
        f"{total_tokens / 1e9:.2f}B" if total_tokens >= 1e9 else f"{total_tokens / 1e6:.1f}M"
    )
    print(f"  {label:<38} {total_sents:>14,.0f} sents   {token_str:>8} tokens   ${total_cost:.2f}")


print(f"\n{'=' * 60}")
print("COST ESTIMATE — qwen/qwen3-embedding-8b via OpenRouter")
print(f"{'=' * 60}")
print(f"Rate:                     ${COST_PER_MILLION_TOKENS:.4f} / 1M tokens")
print(f"Concepts sampled:         {SAMPLE_CONCEPTS} ({concepts_with_data} with >=1 sentence)")
print(
    f"Avg sentences/concept:    {avg_sentences_per_concept:.1f}  (cap: {SENTENCES_PER_CONCEPT:,})"
)
print(f"Avg tokens/sentence:      {avg_tokens:.1f}  (median {median_tokens}, p95 {p95_tokens})")
print()
print(f"{'Scenario':<42} {'Sentences':>14}   {'Tokens':>8}   {'Cost':>6}")
print("-" * 78)
cost_row(
    f"200-concept sample @ actual avg ({avg_sentences_per_concept:.0f}/c)",
    SAMPLE_CONCEPTS,
    avg_sentences_per_concept,
)
cost_row(
    f"200-concept sample @ target cap ({SENTENCES_PER_CONCEPT:,}/c)",
    SAMPLE_CONCEPTS,
    SENTENCES_PER_CONCEPT,
)
cost_row(
    f"Full vocab ({full_vocab_size:,}c) @ actual avg ({avg_sentences_per_concept:.0f}/c)",
    full_vocab_size,
    avg_sentences_per_concept,
)
cost_row(
    f"Full vocab ({full_vocab_size:,}c) @ target cap ({SENTENCES_PER_CONCEPT:,}/c)",
    full_vocab_size,
    SENTENCES_PER_CONCEPT,
)
print(f"{'=' * 60}\n")
print(
    f"Note: avg {avg_sentences_per_concept:.0f} < cap {SENTENCES_PER_CONCEPT:,} — "
    "Wikipedia exhausted for many concepts at current scan depth.\n"
    "To get closer to the cap, increase the article count in `minicoil scan`."
)

# Clean up checkpoint on successful completion
if CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.unlink()
    logger.info("Checkpoint cleaned up")
