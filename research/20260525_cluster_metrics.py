"""Intra-concept cluster quality metrics on Qdrant mining embeddings.

For each concept, sentences are clustered with k-means (k=2..5) and the
best silhouette score is recorded. A high score means the concept has
distinct semantic sub-clusters in the embedding space — different usages
or meanings of the same word — which is what the triplet loss needs to
learn from.

  - Silhouette score  (higher = better, range -1..1)
  - Davies-Bouldin score  (lower = better, >= 0)

Requires: uv add scikit-learn
"""

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from minicoil_v2.qdrant_store import get_client, scroll_concept
from minicoil_v2.settings import QdrantSettings

# --- Config ---
MAX_PER_LANG: int = 50
K_RANGE: list[int] = [2, 3, 4, 5]
SEED: int = 42
DATA_DIR: Path = Path("data/mini")
VOCAB_FILE: Path = DATA_DIR / "concept_vocabulary.json"

np.random.seed(SEED)

# %%  --- Load vocab for labels + connect to Qdrant ---
vocab_raw: dict = json.loads(VOCAB_FILE.read_text())
vocab: dict[str, dict[str, list[str]]] = vocab_raw["concepts"]

settings = QdrantSettings()
client = get_client(settings)

# Pull concept IDs directly from Qdrant — source of truth
facet_result = client.facet(
    collection_name=settings.collection_name,
    key="concept_ids",
    limit=1000,
)
all_concept_ids_in_qdrant: list[str] = [h.value for h in facet_result.hits]
print(f"Distinct concept IDs in Qdrant: {len(all_concept_ids_in_qdrant)}")


def concept_label(cid: str) -> str:
    """Short human-readable label: first EN word / first ES word."""
    entry = vocab.get(cid, {})
    en_words = entry.get("en", [])
    es_words = entry.get("es", [])
    en = en_words[0] if en_words else "?"
    es = es_words[0] if es_words else "?"
    return f"{en}/{es}"


# %%  --- Fetch embeddings from Qdrant ---
all_vectors: list[list[float]] = []
all_concept_ids: list[str] = []
all_langs: list[str] = []

for cid in all_concept_ids_in_qdrant:
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

X: np.ndarray = np.array(all_vectors, dtype=np.float32)
norms = np.linalg.norm(X, axis=1, keepdims=True)
X_norm: np.ndarray = X / np.where(norms == 0, 1, norms)

print(f"Total points: {len(X)} across {len(set(all_concept_ids))} concepts")

# %%  --- Intra-concept silhouette via k-means ---
# For each concept: try k in K_RANGE, keep best silhouette score.
# High silhouette = distinct semantic sub-clusters = good training signal.
rows: list[dict] = []

for cid in all_concept_ids_in_qdrant:
    mask = [i for i, c in enumerate(all_concept_ids) if c == cid]
    if not mask:
        continue
    X_c = X_norm[mask]

    best_sil: float = -1.0
    best_k: int = -1
    best_db: float = float("inf")

    for k in K_RANGE:
        if len(X_c) < k * 2:
            continue
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X_c)
        sil: float = silhouette_score(X_c, labels, metric="euclidean")
        db: float = davies_bouldin_score(X_c, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_db = db

    langs_c = [all_langs[i] for i in mask]
    rows.append(
        {
            "concept_id": cid,
            "label": concept_label(cid),
            "n": len(mask),
            "best_k": best_k,
            "silhouette": round(best_sil, 4),
            "davies_bouldin": round(best_db, 4),
        }
    )

df = pd.DataFrame(rows).sort_values("silhouette", ascending=False)

# %%  --- Distribution ---
print("\n=== Intra-concept silhouette distribution ===")
print("(best k-means k per concept, higher silhouette = more distinct sub-clusters)\n")
print(df.to_string(index=False))

print("\n--- Summary ---")
print(f"  mean   : {df['silhouette'].mean():.4f}")
print(f"  median : {df['silhouette'].median():.4f}")
print(f"  std    : {df['silhouette'].std():.4f}")
print(f"  min    : {df['silhouette'].min():.4f}")
print(f"  max    : {df['silhouette'].max():.4f}")
print(f"  > 0.10 : {(df['silhouette'] > 0.10).sum()} / {len(df)} concepts")
print(f"  > 0.20 : {(df['silhouette'] > 0.20).sum()} / {len(df)} concepts")

# %%  --- Save ---
out_path = DATA_DIR / "cluster_metrics.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
