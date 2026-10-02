"""scroll_concept_rematch: source training rows for an ON-DISK concept from
pre-matched point-id buckets (instead of the cloud's misaligned concept_ids
payload), fetching mining vectors via retrieve. Same return contract as
scroll_concept so run_training is unchanged.
"""

import torch

from minicoil_v2.qdrant_store import scroll_concept_rematch


class _Pt:
    def __init__(self, pid, lang, sentence, vec):
        self.id = pid
        self.payload = {"lang": lang, "sentence": sentence}
        self.vector = {"mining": vec}


class _FakeClient:
    """Captures retrieve() calls and serves points from an in-memory store."""

    def __init__(self, store):
        self._store = store  # pid -> _Pt
        self.retrieved_ids = []

    def retrieve(self, collection_name, ids, with_payload=None, with_vectors=None):
        self.retrieved_ids.extend(ids)
        return [self._store[i] for i in ids if i in self._store]


def _make(n_en, n_es, dim=4096):
    store = {}
    en_ids, es_ids = [], []
    for i in range(n_en):
        pid = f"en{i}"
        store[pid] = _Pt(pid, "en", f"english sentence {i}", [float(i)] * dim)
        en_ids.append(pid)
    for i in range(n_es):
        pid = f"es{i}"
        store[pid] = _Pt(pid, "es", f"oracion espanola {i}", [float(-i)] * dim)
        es_ids.append(pid)
    bucket = {"en": {"ids": en_ids}, "es": {"ids": es_ids}}
    return _FakeClient(store), bucket


def test_returns_parallel_lists_and_mining_tensor():
    client, bucket = _make(5, 4, dim=8)
    sents, focals, langs, mining = scroll_concept_rematch(
        client, "coll", "C-37", bucket, max_per_lang=10, lang_ratio=0.5
    )
    n = len(sents)
    assert n == len(focals) == len(langs) == mining.shape[0]
    assert mining.shape[1] == 8
    assert set(langs) == {"en", "es"}
    # focals are placeholders (live collection has no focal word)
    assert all(f == "" for f in focals)
    assert isinstance(mining, torch.Tensor)


def test_respects_max_per_lang():
    client, bucket = _make(50, 50, dim=4)
    sents, _f, langs, mining = scroll_concept_rematch(
        client, "coll", "C-37", bucket, max_per_lang=10, lang_ratio=0.5
    )
    assert langs.count("en") == 10
    assert langs.count("es") == 10
    assert mining.shape == (20, 4)


def test_empty_bucket_returns_empty():
    client = _FakeClient({})
    sents, focals, langs, mining = scroll_concept_rematch(
        client, "coll", "C-99", {"en": {"ids": []}, "es": {"ids": []}}, max_per_lang=10
    )
    assert sents == [] and focals == [] and langs == []
    assert mining.numel() == 0


def test_only_fetches_capped_ids_not_whole_bucket():
    # max_per_lang must bound the retrieve cost; we must NOT retrieve all 50 ids.
    client, bucket = _make(50, 50, dim=4)
    scroll_concept_rematch(client, "coll", "C-37", bucket, max_per_lang=10, lang_ratio=0.5)
    assert len(client.retrieved_ids) <= 20
