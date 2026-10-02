"""End-to-end retrieval with the PUBLISHED model: index a small bilingual corpus in
Qdrant with Modifier.IDF, then query across languages. Uses only the downloaded repo."""

import sys

from huggingface_hub import snapshot_download

model_dir = snapshot_download("Jocana/minicoil-en-es", cache_dir=sys.argv[1])
sys.path.insert(0, model_dir)
from qdrant_client import QdrantClient, models  # noqa: E402

from minicoil_v2.encoder import MiniCoilEncoder  # noqa: E402  (the downloaded repo's copy)

enc = MiniCoilEncoder.from_pretrained(model_dir)

CORPUS = [
    ("es", "El perro corrió por el parque detrás de una pelota roja."),
    ("es", "La receta de la sopa lleva cebolla, ajo y tomate fresco."),
    ("es", "El tribunal dictó sentencia tras escuchar a los testigos."),
    ("es", "Los astrónomos observaron una estrella nueva en la galaxia."),
    ("en", "The dog ran through the park chasing a red ball."),
    ("en", "The soup recipe calls for onion, garlic and fresh tomato."),
    ("en", "The court issued a ruling after hearing the witnesses."),
    ("en", "Astronomers observed a new star in the galaxy."),
]
QUERIES = [
    ("en", "dog running in the park", "es", 0),
    ("en", "soup with onion and garlic", "es", 1),
    ("en", "court ruling witnesses", "es", 2),
    ("es", "perro corriendo en el parque", "en", 4),
    ("es", "sopa con cebolla y ajo", "en", 5),
    ("es", "estrella nueva galaxia", "en", 7),
]

client = QdrantClient(":memory:")
client.create_collection(
    "probe",
    vectors_config={},
    sparse_vectors_config={"minicoil": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)

points = []
for i, (lang, text) in enumerate(CORPUS):
    vec = enc.encode_sparse(text, lang=lang)
    items = sorted(vec.items())
    points.append(
        models.PointStruct(
            id=i,
            vector={
                "minicoil": models.SparseVector(
                    indices=[k for k, _ in items], values=[v for _, v in items]
                )
            },
            payload={"lang": lang, "text": text},
        )
    )
client.upsert("probe", points)

ok = 0
for qlang, qtext, doc_lang, expected in QUERIES:
    vec = enc.encode_sparse(qtext, lang=qlang, is_query=True)
    items = sorted(vec.items())
    hits = client.query_points(
        "probe",
        query=models.SparseVector(indices=[k for k, _ in items], values=[v for _, v in items]),
        using="minicoil",
        limit=8,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="lang", match=models.MatchValue(value=doc_lang))]
        ),
    ).points
    # How much of the query is carried by concepts (the only cross-lingual bridge):
    # backbone stems are language-specific, so a query with no concept hits cannot
    # match a document in the other language at all.
    BACKBONE_BASE = 1 << 27
    n_concept_terms = sum(1 for k in vec if k < BACKBONE_BASE)
    if not hits:
        print(f"[NONE] {qlang}->{doc_lang} {qtext!r}  (concept terms in query: {n_concept_terms})")
        continue
    top = hits[0]
    hit = top.id == expected
    ok += hit
    print(
        f"[{'OK ' if hit else 'MISS'}] {qlang}->{doc_lang} {qtext!r}  "
        f"(concept terms: {n_concept_terms})"
    )
    print(f"        rank1 (score {top.score:.3f}): {top.payload['text']}")

print(f"\ncross-lingual rank-1 accuracy: {ok}/{len(QUERIES)}")
sys.exit(0 if ok == len(QUERIES) else 1)
