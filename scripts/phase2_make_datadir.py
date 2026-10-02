"""Build the Phase-2 embed data-dir: data/phase2/ with word_to_concept.json and
concept_vocabulary.json filtered to the demand-selected concepts
(data/eval/mmarco/phase2_concepts.json).

Concept ids are NOT renumbered: the filtered files keep the on-disk vintage, so
the new collection's payload concept_ids align with the encoder/eval vocabulary
by construction (no rematch workaround). `minicoil embed --data-dir data/phase2`
then only stores sentences for selected concepts.
"""

import json
from pathlib import Path

with open("data/eval/mmarco/phase2_concepts.json") as f:
    sel = {c["cid"] for c in json.load(f)["concepts"]}

out = Path("data/phase2")
out.mkdir(exist_ok=True)

with open("data/word_to_concept.json") as f:
    w2c = json.load(f)

filtered = {lang: {w: c for w, c in m.items() if c in sel} for lang, m in w2c.items()}
with open(out / "word_to_concept.json", "w") as f:
    json.dump(filtered, f, indent=1)

with open("data/concept_vocabulary.json") as f:
    vocab = json.load(f)

fv = {
    **{k: v for k, v in vocab.items() if k != "concepts"},
    "concepts": {cid: e for cid, e in vocab["concepts"].items() if cid in sel},
}
with open(out / "concept_vocabulary.json", "w") as f:
    json.dump(fv, f, indent=1)

kept = {c for m in filtered.values() for c in m.values()}
print(f"selected={len(sel)} kept_in_w2c={len(kept)} vocab_entries={len(fv['concepts'])}")
print(f"en words: {len(filtered['en'])}  es words: {len(filtered['es'])}")
missing = sel - kept
if missing:
    print(f"WARNING: {len(missing)} selected cids have no surface forms: {sorted(missing)[:5]}")
