"""Read-only: scan the ENTIRE cloud corpus (payload only, no vectors) and, using
the ON-DISK pruned vocab, tally how many sentences fire each of the viable-64
concepts per language. Also caches up to CAP point-ids per (cid, lang) so a
later loader can fetch only the needed Qwen3 vectors (two-pass, never 760k x 4096).

Writes data/eval/mmarco/viable_cloud_buckets.json:
  {cid: {"en": {"count": int, "ids": [pt_id,...]}, "es": {...}}}
"""

import json
import os
import statistics
import time
from collections import defaultdict

from qdrant_client import QdrantClient

from minicoil_v2.concept_match import match_concepts

CAP = int(os.environ.get("CAP", "3000"))  # max ids cached per (cid, lang)

with open("data/word_to_concept.json") as f:
    w2c = json.load(f)
with open("data/eval/mmarco/viable_concepts.json") as f:
    viable = set(json.load(f)["concept_ids"])
print(f"viable concepts: {len(viable)}  CAP={CAP}")

c = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=120)
coll = "minicoil_sentences"
total = c.count(coll).count
print(f"corpus count: {total}")

counts = defaultdict(lambda: defaultdict(int))  # cid -> lang -> count
ids = defaultdict(lambda: defaultdict(list))  # cid -> lang -> [pt_id]

offset = None
seen = 0
t0 = time.time()
while True:
    pts, offset = c.scroll(
        collection_name=coll,
        limit=4096,
        with_payload=["sentence", "lang"],
        with_vectors=False,
        offset=offset,
    )
    if not pts:
        break
    for p in pts:
        seen += 1
        lang = p.payload.get("lang")
        sent = p.payload.get("sentence", "")
        if lang not in ("en", "es") or not sent:
            continue
        # Match exactly as the encoder does at index time (the text itself, not the
        # mE5 prefix; lowercased; restricted to viable so the regex pass is cheap).
        fired = match_concepts(sent.lower(), lang, w2c, viable)
        for cid in fired:
            counts[cid][lang] += 1
            bucket = ids[cid][lang]
            if len(bucket) < CAP:
                bucket.append(p.id)
    if seen % 40960 == 0:
        el = time.time() - t0
        print(f"  scanned {seen}/{total} ({seen / total:.0%}) in {el:.0f}s")
    if offset is None:
        break

print(f"scan done: {seen} sentences in {time.time() - t0:.0f}s")
print(f"corpus count still: {c.count(coll).count}")

out = {
    cid: {lang: {"count": counts[cid][lang], "ids": ids[cid][lang]} for lang in ("en", "es")}
    for cid in sorted(viable)
}
path = "data/eval/mmarco/viable_cloud_buckets.json"
with open(path, "w") as f:
    json.dump(out, f)
print(f"wrote {path}")

# summary
mins = []
print("\nper-concept min(en,es) coverage:")
ge1k = ge500 = ge100 = 0
for cid in sorted(viable):
    en = counts[cid]["en"]
    es = counts[cid]["es"]
    m = min(en, es)
    mins.append(m)
    ge1k += m >= 1000
    ge500 += m >= 500
    ge100 += m >= 100
mins.sort()
print(f"  >=1000/lang: {ge1k}/64   >=500: {ge500}/64   >=100: {ge100}/64")
print(f"  median min: {statistics.median(mins):.0f}  min: {mins[0]}  max: {mins[-1]}")
print(f"  worst 10 mins: {mins[:10]}")
