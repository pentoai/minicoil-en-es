"""Validate per-concept 4D layer behavior at scale.

For each trained concept in a layers file:
  - Fetch every stored sentence from the source Qdrant collection.
  - Bucket sentences by `focal_word` (surface form); for concepts whose surface
    form alone does not disambiguate the sense, apply a sense regex.
  - Apply the trained Linear(384,4)+tanh layer to each sentence's stored 384D
    vector.
  - Print 6 sample sentences per bucket, each with its 4D vector + L2 norm.
  - Compute intra-bucket cosine mean and cross-bucket cosine matrix.
  - Per-concept gap = min(intra-bucket) - max(cross-bucket).

Global stats:
  - 4D magnitude distribution across all sentences in all concepts.
  - Cross-lingual top-1 same-concept rate across all trained concepts.

Run twice from the shell: once for token-pool, once for sentence-pool. See
the bottom of this file for the canonical invocations.
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import torch
from qdrant_client import QdrantClient, models

TRAINED_CONCEPTS_ORDER = [
    "C-00122",
    "C-00126",
    "C-00173",
    "C-00183",
    "C-00190",
    "C-00201",
    "C-00304",
    "C-00620",
    "C-00818",
    "C-00875",
    "C-00877",
    "C-01016",
    "C-01035",
    "C-01098",
    "C-01120",
    "C-01594",
    "C-03165",
    "C-03566",
    "C-03567",
    "C-06550",
]

POLYSEMOUS = {
    "C-00190",
    "C-03566",
    "C-01016",
    "C-01035",
    "C-01120",
    "C-00875",
    "C-00173",
    "C-00183",
    "C-03567",
}
MONOSEMIC = {
    "C-00122",
    "C-00304",
    "C-06550",
    "C-03165",
    "C-00201",
    "C-00126",
    "C-00620",
    "C-00818",
    "C-00877",
    "C-01098",
    "C-01594",
}

# Human label per concept for printing.
CONCEPT_LABEL = {
    "C-00122": "food/feed",
    "C-00126": "understand",
    "C-00173": "time/weather",
    "C-00183": "see/view/reproductions",
    "C-00190": "set/fixed/ensemble",
    "C-00201": "world/global",
    "C-00304": "animal/pet",
    "C-00620": "make/do",
    "C-00818": "use/usage",
    "C-00875": "work/jobs/obra",
    "C-00877": "motion/move",
    "C-01016": "type/write",
    "C-01035": "case/cardboard",
    "C-01098": "include",
    "C-01120": "way/road",
    "C-01594": "say/tell",
    "C-03165": "life/vida",
    "C-03566": "plant/plants",
    "C-03567": "earth/tierra",
    "C-06550": "water/agua",
}

# Sense-collapsed mapping: focal_word (lowercased) -> sense label.
# For each polysemous concept, explicitly group inflections of the same
# sense together. Words not in the mapping fall into "<focal>_unmapped".
SENSE_OF_FOCAL: dict[str, dict[str, str]] = {
    "C-00190": {  # set / fixed / ensemble — lumped MUSE cluster
        "set": "set_eng",
        "sets": "set_eng",
        "fixed": "fixed_adj",
        "corrected": "fixed_adj",
        "fijo": "fixed_adj",
        "fijos": "fixed_adj",
        "fija": "fixed_adj",
        "fijas": "fixed_adj",
        "fijado": "fixed_adj",
        "fijada": "fixed_adj",
        "fijados": "fixed_adj",
        "corregido": "fixed_adj",
        "corregida": "fixed_adj",
        "corregidos": "fixed_adj",
        "conjunto": "ensemble_es",
        "ensemble": "ensemble_en",
        "ensembles": "ensemble_en",
    },
    "C-01016": {  # type / write
        "type": "kind",  # english "type" mostly means "kind of"
        "tipo": "kind",
        "write": "write_verb",
        "writes": "write_verb",
        "escribir": "write_verb",
        "escribe": "write_verb",
    },
    "C-01035": {  # case / cardboard — only case/caso in data
        "case": "case_en",
        "caso": "case_es",
    },
    "C-01120": {  # way / road / manera
        "way": "way_en",
        "ways": "way_en",
        "road": "road_path",
        "roads": "road_path",
        "camino": "road_path",
        "manera": "manner_es",
    },
    "C-00875": {  # work / jobs / obra
        "work": "work_labor",
        "trabajo": "work_labor",
        "trabajos": "work_labor",
        "jobs": "work_labor",
        "empleos": "work_labor",
        "obra": "work_artistic",
    },
    "C-00173": {  # time / weather / climate
        "time": "time_concept",
        "tiempo": "time_or_weather_amb",  # tiempo is ambiguous in Spanish
        "hora": "clock_hour",
        "climate": "climate",
        "clima": "climate",
        "weather": "weather",
        "meteorología": "climate",
        "meteorológicos": "climate",
        "meteorológico": "climate",
        "meteorológicas": "climate",
        "meteorológica": "climate",
    },
    "C-00183": {  # see / view / vista / reproductions
        "see": "see_verb",
        "ver": "see_verb",
        "véase": "see_verb",
        "view": "view_verb_or_noun",
        "views": "view_verb_or_noun",
        "vista": "vista_noun",
        "vistas": "vista_noun",
        "reproductions": "reproductions",
        "reproducciones": "reproductions",
        "visualizaciones": "visualizations",
    },
}

# Sense regexes for concepts where surface form is the SAME across senses.
# Bucket name and the regex that selects sentences of that sense.
SENSE_REGEX: dict[str, dict[str, re.Pattern]] = {
    "C-03566": {  # plant: botanical vs industrial
        "botanical": re.compile(
            r"\b(species|leaf|leaves|flower|fruit|seed|tree|root|grow|growing|grew|"
            r"botanic|botany|garden|forest|crop|cultivat|leafy|stems?|bloom|petal|"
            r"angiosperm|gymnosperm|chlorophyll|photosynthesis|herb|shrub|"
            r"especie|hoja|hojas|flor|flores|fruto|frutos|semilla|árbol|raíz|crec|"
            r"botánic|jardín|bosque|cultiv|hierba|arbusto|tallo|pétalo)\b",
            re.IGNORECASE,
        ),
        "industrial": re.compile(
            r"\b(power\s+plant|nuclear|chemical|treatment|industrial|manufactur|"
            r"refinery|reactor|generat\w*\s+plant|coal|gas\s+plant|water\s+treatment|"
            r"central\s+(eléctrica|nuclear|térmica)|fábrica|planta\s+(nuclear|química|"
            r"industrial|de\s+tratamiento|de\s+energía))\b",
            re.IGNORECASE,
        ),
    },
    "C-03567": {  # earth: planet vs soil
        "planet": re.compile(
            r"\b(planet|orbit|atmosphere|solar|moon|sun|space|gravity|spheric|"
            r"earth's\s+(orbit|surface|atmosphere|core|crust|axis)|equator|hemisphere|"
            r"planeta|órbita|atmósfera|solar|luna|sol|espacio|gravedad|esférico|"
            r"corteza\s+terrestre|hemisferio|ecuador)\b",
            re.IGNORECASE,
        ),
        "soil": re.compile(
            r"\b(soil|ground|dirt|buried|dig|underground|excavat|topsoil|"
            r"sediment|earthworm|earthen|earthwork|mound|"
            r"suelo|tierra\s+(firme|húmeda|fértil|seca|adentro)|enterrad|excav|"
            r"subterrán|cavar|sediment)\b",
            re.IGNORECASE,
        ),
    },
}


def scroll_concept_all(qc: QdrantClient, collection: str, cid: str) -> list[dict]:
    """All points for concept `cid`, returns list of {sentence,focal,lang,vec}."""
    rows: list[dict] = []
    offset = None
    flt = models.Filter(
        must=[models.FieldCondition(key="concept_ids", match=models.MatchValue(value=cid))]
    )
    while True:
        batch, nxt = qc.scroll(
            collection,
            limit=1024,
            offset=offset,
            scroll_filter=flt,
            with_payload=["sentence", "focal_word", "lang"],
            with_vectors=["mining"],
        )
        for p in batch:
            rows.append(
                {
                    "sentence": p.payload["sentence"],
                    "focal": p.payload.get("focal_word", ""),
                    "lang": p.payload["lang"],
                    "vec": np.asarray(p.vector["mining"], dtype=np.float32),
                }
            )
        if nxt is None:
            break
        offset = nxt
    return rows


def apply_layer(vec384: np.ndarray, layer: dict) -> np.ndarray:
    """tanh(W @ v + b) -> (..., 4)."""
    W = layer["weight"].numpy()  # (4, 384)
    b = (
        layer["bias"].numpy()
        if "bias" in layer and layer["bias"] is not None
        else np.zeros(4, dtype=np.float32)
    )
    return np.tanh(vec384 @ W.T + b).astype(np.float32)


def normed(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + 1e-12)


def bucket_rows(cid: str, rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows into sense buckets.

    Priority:
      1. If cid is in SENSE_REGEX, bucket by regex match (planet/soil, botanic/industrial).
         Sentences matching neither or both go to "other_<focal>".
      2. Else if cid is in SENSE_OF_FOCAL, bucket by mapped sense label.
         Unmapped focal_words become "<focal>_unmapped".
      3. Else (monosemic): single bucket "all".
    """
    out: dict[str, list[dict]] = {}
    if cid in SENSE_REGEX:
        senses = SENSE_REGEX[cid]
        for r in rows:
            matched = [name for name, rx in senses.items() if rx.search(r["sentence"])]
            if len(matched) == 1:
                key = matched[0]
            else:
                key = f"other_{r['lang']}"
            out.setdefault(key, []).append(r)
        return out
    if cid in SENSE_OF_FOCAL:
        mapping = SENSE_OF_FOCAL[cid]
        for r in rows:
            focal = r["focal"].lower().strip()
            key = mapping.get(focal, f"{focal}_unmapped")
            out.setdefault(key, []).append(r)
        return out
    out["all"] = list(rows)
    return out


def truncate(s: str, n: int = 90) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def highlight(s: str, focal: str) -> str:
    if not focal:
        return s
    return re.sub(rf"\b({re.escape(focal)})\b", r"[\1]", s, count=1, flags=re.IGNORECASE)


def cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    return normed(vecs) @ normed(vecs).T


def pairwise_inter(vecs_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
    return normed(vecs_a) @ normed(vecs_b).T


def upper_mean(mat: np.ndarray) -> float:
    """Mean of upper triangle (excluding diagonal)."""
    if mat.shape[0] < 2:
        return float("nan")
    iu = np.triu_indices(mat.shape[0], k=1)
    return float(mat[iu].mean())


def fmt4d(v: np.ndarray) -> str:
    return f"[{v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f} {v[3]:+.3f}]"


def analyse_concept(cid: str, layer: dict, rows: list[dict], rng: np.random.Generator) -> dict:
    """Print + return per-concept stats."""
    label = CONCEPT_LABEL.get(cid, "?")
    is_poly = cid in POLYSEMOUS
    print()
    print("=" * 100)
    print(f"  {cid}  ({label})  [{'POLY' if is_poly else 'MONO'}]  N={len(rows)}")
    print("=" * 100)

    if not rows:
        print("  (no rows)")
        return {"cid": cid, "n": 0}

    # apply layer
    vecs384 = np.stack([r["vec"] for r in rows])  # (N, 384)
    vecs4 = apply_layer(vecs384, layer)  # (N, 4)
    for r, v in zip(rows, vecs4):
        r["v4"] = v
        r["v4n"] = v / (np.linalg.norm(v) + 1e-12)

    # bucket
    buckets = bucket_rows(cid, rows)
    print(f"  buckets ({len(buckets)}):")
    for k, br in sorted(buckets.items()):
        en = sum(1 for r in br if r["lang"] == "en")
        es = sum(1 for r in br if r["lang"] == "es")
        print(f"    {k:30s} N={len(br):4d} (en={en} es={es})")

    # ---- sample print: up to 4 sentences per bucket with 4D vectors ----
    print("\n  --- sample sentences (focal in []) + 4D output ---")
    bucket_summary = []
    for bname in sorted(buckets):
        br = buckets[bname]
        # need at least 3 sentences in bucket to compute intra-cosine
        if len(br) < 3:
            print(f"\n  [bucket: {bname}]  (N={len(br)}, too few for stats)")
            continue
        # sample up to 4 sentences per bucket per language for display
        en_b = [r for r in br if r["lang"] == "en"]
        es_b = [r for r in br if r["lang"] == "es"]
        en_pick = list(rng.choice(en_b, size=min(3, len(en_b)), replace=False)) if en_b else []
        es_pick = list(rng.choice(es_b, size=min(2, len(es_b)), replace=False)) if es_b else []
        picks = list(en_pick) + list(es_pick)
        print(f"\n  [bucket: {bname}]  N={len(br)} (en={len(en_b)} es={len(es_b)})")
        for r in picks:
            v = r["v4"]
            norm = float(np.linalg.norm(v))
            print(
                f"    {r['lang']}  ‖{fmt4d(v)}‖={norm:.3f}  {truncate(highlight(r['sentence'], r['focal']))}"
            )

        # bucket stats
        V = np.stack([r["v4n"] for r in br])
        intra = upper_mean(V @ V.T)
        bucket_summary.append({"name": bname, "n": len(br), "intra": intra, "vecs": V})

    # ---- cross-bucket cosine matrix ----
    print("\n  --- intra & cross bucket cosine (4D, normalized) ---")
    if len(bucket_summary) >= 2:
        names = [b["name"] for b in bucket_summary]
        cross = np.zeros((len(names), len(names)), dtype=np.float32)
        for i, bi in enumerate(bucket_summary):
            for j, bj in enumerate(bucket_summary):
                if i == j:
                    cross[i, j] = bi["intra"]
                else:
                    cross[i, j] = float((bi["vecs"] @ bj["vecs"].T).mean())
        # print matrix with diagonals = intra
        hdr = "        " + " ".join(f"{n[:10]:>11s}" for n in names)
        print(hdr)
        for i, n in enumerate(names):
            row = " ".join(f"{cross[i, j]:>+11.3f}" for j in range(len(names)))
            print(f"  {n[:6]:>6s}  {row}")
        # per-concept gap
        intras = np.array([b["intra"] for b in bucket_summary])
        off = cross.copy()
        np.fill_diagonal(off, -np.inf)
        max_cross = float(off.max())
        min_intra = float(intras.min())
        gap = min_intra - max_cross
        print(f"\n  intra-bucket min:  {min_intra:+.3f}")
        print(f"  cross-bucket max:  {max_cross:+.3f}")
        print(f"  gap (min_intra - max_cross): {gap:+.3f}")
    else:
        gap = None
        min_intra = bucket_summary[0]["intra"] if bucket_summary else float("nan")
        max_cross = float("nan")
        print(f"  only {len(bucket_summary)} usable bucket, no cross stats")
        print(f"  intra: {min_intra:+.3f}")

    # ---- 4D magnitude stats for this concept ----
    norms = np.linalg.norm(vecs4, axis=1)
    print(
        f"\n  4D L2 norm: mean={norms.mean():.3f}  std={norms.std():.3f}  "
        f"min={norms.min():.3f}  max={norms.max():.3f}"
    )

    return {
        "cid": cid,
        "n": len(rows),
        "n_buckets_usable": len(bucket_summary),
        "intra_min": min_intra,
        "cross_max": max_cross if len(bucket_summary) >= 2 else None,
        "gap": gap,
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "vecs4": vecs4,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--layers-path", required=True)
    ap.add_argument("--label", required=True, help="Short tag, e.g. 'token-pool'")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--out", default=None, help="Write to a file in addition to stdout.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.out:
        f = open(args.out, "w")

        class T:
            def write(self, s):
                f.write(s)
                sys.__stdout__.write(s)

            def flush(self):
                f.flush()

        sys.stdout = T()  # type: ignore[assignment]

    rng = np.random.default_rng(args.seed)
    qc = QdrantClient(url=args.qdrant_url, prefer_grpc=False)

    layers = torch.load(args.layers_path, map_location="cpu", weights_only=True)
    print(f"\n###  VALIDATION RUN  [{args.label}]")
    print(f"     collection: {args.collection}")
    print(f"     layers:     {args.layers_path}")
    print(f"     trained:    {len(layers)} concepts\n")

    per_concept: list[dict] = []
    for cid in TRAINED_CONCEPTS_ORDER:
        if cid not in layers:
            print(f"  !! skipping {cid} (not in layers)")
            continue
        rows = scroll_concept_all(qc, args.collection, cid)
        per_concept.append(analyse_concept(cid, layers[cid], rows, rng))

    # ---- global summary ----
    print("\n" + "#" * 100)
    print(f"#  GLOBAL SUMMARY  [{args.label}]")
    print("#" * 100)

    all_norms = np.concatenate([np.linalg.norm(c["vecs4"], axis=1) for c in per_concept if c["n"]])
    print(f"\n  4D L2 norm across all {all_norms.size} sentences:")
    print(f"    mean={all_norms.mean():.3f}  std={all_norms.std():.3f}")
    print(
        f"    p5={np.percentile(all_norms, 5):.3f}  p50={np.percentile(all_norms, 50):.3f}  p95={np.percentile(all_norms, 95):.3f}"
    )
    print(f"    min={all_norms.min():.3f}  max={all_norms.max():.3f}")

    print("\n  Per-concept gap table (sorted by gap, descending):")
    print(
        f"    {'cid':>10s} {'label':<24s} {'class':>6s} {'N':>5s} {'intra_min':>10s} {'cross_max':>10s} {'gap':>10s} {'norm_mean':>10s}"
    )
    rows_for_sort = [c for c in per_concept if c.get("gap") is not None]
    rows_for_sort.sort(key=lambda c: -c["gap"])
    for c in rows_for_sort:
        klass = "POLY" if c["cid"] in POLYSEMOUS else "MONO"
        print(
            f"    {c['cid']:>10s} {CONCEPT_LABEL[c['cid']][:24]:<24s} {klass:>6s} {c['n']:>5d} "
            f"{c['intra_min']:>+10.3f} {c['cross_max']:>+10.3f} {c['gap']:>+10.3f} {c['norm_mean']:>10.3f}"
        )

    print("\n  Concepts without cross stats (only 1 usable bucket):")
    for c in per_concept:
        if c.get("gap") is None:
            klass = "POLY" if c["cid"] in POLYSEMOUS else "MONO"
            print(
                f"    {c['cid']:>10s} {CONCEPT_LABEL[c['cid']][:24]:<24s} {klass:>6s} N={c['n']:>4d}  intra={c['intra_min']:+.3f}  norm_mean={c['norm_mean']:.3f}"
            )

    # ---- cross-lingual top-1 across all trained concepts ----
    # Build a pool of (vec4n, cid, lang) across all concepts using each concept's OWN layer.
    pool_v: list[np.ndarray] = []
    pool_cid: list[str] = []
    pool_lang: list[str] = []
    for c in per_concept:
        if c["n"] == 0:
            continue
        v = c["vecs4"]
        vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        for k, r in enumerate(c["rows"]):
            pool_v.append(vn[k])
            pool_cid.append(c["cid"])
            pool_lang.append(r["lang"])
    V = np.stack(pool_v)
    pool_cid_arr = np.array(pool_cid)
    pool_lang_arr = np.array(pool_lang)

    # For each EN sentence, top-1 ES match across pool, check concept agreement.
    en_idx = np.where(pool_lang_arr == "en")[0]
    es_idx = np.where(pool_lang_arr == "es")[0]
    if en_idx.size and es_idx.size:
        es_V = V[es_idx]
        en_V = V[en_idx]
        sims = en_V @ es_V.T  # (Nen, Nes)
        top1 = sims.argmax(axis=1)
        match = (pool_cid_arr[en_idx] == pool_cid_arr[es_idx[top1]]).mean()
        print(
            f"\n  Cross-lingual top-1 same-concept (EN -> ES, pool size: en={en_idx.size} es={es_idx.size}):"
        )
        print(f"    {match:.1%}  (chance ~{1 / len(per_concept):.1%})")
    else:
        print("\n  Cross-lingual top-1: insufficient data")

    print(f"\n###  END  [{args.label}]\n")


if __name__ == "__main__":
    main()
