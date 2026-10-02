# type: ignore[import-untyped]
"""UMAP visualization of Qdrant mining-vector clusters.

Requires: uv add umap-learn matplotlib
"""

# %%
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap

from minicoil_v2.qdrant_store import get_client, scroll_concept
from minicoil_v2.settings import QdrantSettings

# --- Config (edit here) ---
N_CONCEPTS: int = 100  # number of concepts to visualize
MAX_PER_LANG: int = 100  # sentences per language per concept
SEED: int = 42
DATA_DIR: Path = Path("data/mini")
COUNTS_FILE: Path = DATA_DIR / "concept_sentence_counts.json"
VOCAB_FILE: Path = DATA_DIR / "concept_vocabulary.json"

random.seed(SEED)
np.random.seed(SEED)

# %%  --- Load concept metadata ---
counts_raw: dict = json.loads(COUNTS_FILE.read_text())
counts_en: dict[str, int] = counts_raw["counts"]["en"]
counts_es: dict[str, int] = counts_raw["counts"]["es"]

vocab_raw: dict = json.loads(VOCAB_FILE.read_text())
vocab: dict[str, dict[str, list[str]]] = vocab_raw["concepts"]

# Keep only concepts with enough data in both languages
MIN_COUNT = MAX_PER_LANG
eligible: list[str] = [
    cid
    for cid in vocab
    if counts_en.get(cid, 0) >= MIN_COUNT and counts_es.get(cid, 0) >= MIN_COUNT
]
print(f"Eligible concepts (>= {MIN_COUNT} sentences per lang): {len(eligible)}")

sampled: list[str] = random.sample(eligible, min(N_CONCEPTS, len(eligible)))
print(f"Sampled {len(sampled)} concepts: {sampled}")


def concept_label(cid: str) -> str:
    """Short human-readable label: first EN word / first ES word."""
    entry = vocab.get(cid, {})
    en_words = entry.get("en", [])
    es_words = entry.get("es", [])
    en = en_words[0] if en_words else "?"
    es = es_words[0] if es_words else "?"
    return f"{en}/{es}"


# %%  --- Fetch embeddings from Qdrant ---
settings = QdrantSettings()
client = get_client(settings)

all_vectors: list[list[float]] = []
all_concept_ids: list[str] = []
all_langs: list[str] = []

for cid in sampled:
    sentences, langs, embs = scroll_concept(
        client=client,
        collection_name=settings.collection_name,
        concept_id=cid,
        max_per_lang=MAX_PER_LANG,
        lang_ratio=0.5,
    )
    if embs.numel() == 0:
        print(f"  {cid}: no data, skipping")
        continue
    all_vectors.extend(embs.tolist())
    all_concept_ids.extend([cid] * len(langs))
    all_langs.extend(langs)
    print(
        f"  {cid} ({concept_label(cid)}): {len(langs)} sentences "
        f"({langs.count('en')} en / {langs.count('es')} es)"
    )

print(f"\nTotal points: {len(all_vectors)}")

# %%  --- Run UMAP ---
vectors_np: np.ndarray = np.array(all_vectors, dtype=np.float32)

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=SEED,
    verbose=True,
)
embedding: np.ndarray = reducer.fit_transform(vectors_np)
print(f"UMAP output shape: {embedding.shape}")

# %%  --- Plot: color by concept, marker by language ---
concept_to_idx: dict[str, int] = {cid: i for i, cid in enumerate(sampled)}
colors = plt.cm.tab20(np.linspace(0, 1, len(sampled)))  # type: ignore[attr-defined]
lang_markers: dict[str, str] = {"en": "o", "es": "^"}

fig, ax = plt.subplots(figsize=(14, 10))

for cid in sampled:
    idx = concept_to_idx[cid]
    label = concept_label(cid)
    color = colors[idx]

    for lang, marker in lang_markers.items():
        mask = [
            i
            for i, (c, sent_lang) in enumerate(zip(all_concept_ids, all_langs, strict=True))
            if c == cid and sent_lang == lang
        ]
        if not mask:
            continue
        pts = embedding[mask]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=[color],
            marker=marker,
            s=30,
            alpha=0.7,
            label=f"{label} ({lang})" if lang == "en" else None,
        )

# Concept legend (colors, circles)
concept_handles = [
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=colors[concept_to_idx[cid]],
        markersize=9,
        label=concept_label(cid),
    )
    for cid in sampled
]
concept_legend = ax.legend(
    handles=concept_handles,
    loc="upper left",
    fontsize=7,
    title="Concept",
)
ax.add_artist(concept_legend)

# Language legend (markers)
lang_handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=9, label="en"),
    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=9, label="es"),
]
ax.legend(
    handles=lang_handles,
    loc="lower left",
    fontsize=8,
    title="Language",
)

ax.set_title(f"UMAP of mining embeddings — {N_CONCEPTS} concepts, {len(all_vectors)} sentences")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
out_path = DATA_DIR / "umap_clusters.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved {out_path}")
