def test_resolve_model_revision_sha_uses_cache_file_when_present(tmp_path, monkeypatch):
    cache_dir = tmp_path / "nllb_cache"
    cache_dir.mkdir()
    (cache_dir / "nllb_revision").write_text("cached-sha-123\n", encoding="utf-8")

    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._nllb_revision_cache_path",
        lambda: cache_dir / "nllb_revision",
    )

    from minicoil_v2.eval.retrievers._common import resolve_nllb_revision_sha

    sha = resolve_nllb_revision_sha()
    assert sha == "cached-sha-123"


def test_resolve_model_revision_sha_falls_back_when_offline(tmp_path, monkeypatch):
    cache_path = tmp_path / "nllb_revision"
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._nllb_revision_cache_path",
        lambda: cache_path,
    )

    def fail(*a, **kw):
        raise RuntimeError("offline")

    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._fetch_nllb_revision_from_hf",
        fail,
    )

    from minicoil_v2.eval.retrievers._common import resolve_nllb_revision_sha

    sha = resolve_nllb_revision_sha()
    assert sha == "distilled-600M-fallback"
    assert cache_path.read_text(encoding="utf-8").strip() == "distilled-600M-fallback"


def test_resolve_model_revision_sha_persists_first_successful_lookup(tmp_path, monkeypatch):
    cache_path = tmp_path / "nllb_revision"
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._nllb_revision_cache_path",
        lambda: cache_path,
    )
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._fetch_nllb_revision_from_hf",
        lambda: "real-sha-from-hf",
    )

    from minicoil_v2.eval.retrievers._common import resolve_nllb_revision_sha

    sha = resolve_nllb_revision_sha()
    assert sha == "real-sha-from-hf"
    assert cache_path.read_text(encoding="utf-8").strip() == "real-sha-from-hf"


def test_translate_cache_hit_does_not_load_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._nllb_revision_cache_path",
        lambda: tmp_path / "nllb_revision",
    )
    (tmp_path / "nllb_revision").write_text("test-sha\n", encoding="utf-8")

    import hashlib

    from minicoil_v2.eval.retrievers._common import Translator

    cache_dir = tmp_path / "translation_cache"
    text = "hello world"
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cache_path = cache_dir / "test-sha" / "en-es" / f"{sha}.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("hola mundo", encoding="utf-8")

    t = Translator(cache_dir=cache_dir)

    def fail_load():
        raise AssertionError("model should not be loaded on cache hit")

    monkeypatch.setattr(t, "_lazy_load", fail_load)

    result = t.translate(text, src="en", tgt="es")
    assert result == "hola mundo"


def test_translate_cache_miss_calls_model_and_writes_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._nllb_revision_cache_path",
        lambda: tmp_path / "nllb_revision",
    )
    (tmp_path / "nllb_revision").write_text("test-sha\n", encoding="utf-8")

    from minicoil_v2.eval.retrievers._common import Translator

    cache_dir = tmp_path / "translation_cache"
    t = Translator(cache_dir=cache_dir)

    calls = []

    def fake(texts, src, tgt):
        calls.append((tuple(texts), src, tgt))
        return [f"<{src}->{tgt}>{x}" for x in texts]

    monkeypatch.setattr(t, "_lazy_load", lambda: None)
    monkeypatch.setattr(t, "_translate_uncached", fake)

    out = t.translate("hello", src="en", tgt="es")
    assert out == "<en->es>hello"
    assert len(calls) == 1

    out2 = t.translate("hello", src="en", tgt="es")
    assert out2 == "<en->es>hello"
    assert len(calls) == 1


def test_translate_batch_resolves_hits_first_and_batches_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "minicoil_v2.eval.retrievers._common._nllb_revision_cache_path",
        lambda: tmp_path / "nllb_revision",
    )
    (tmp_path / "nllb_revision").write_text("test-sha\n", encoding="utf-8")

    import hashlib

    from minicoil_v2.eval.retrievers._common import Translator

    cache_dir = tmp_path / "translation_cache"
    text_hit = "hello"
    sha_hit = hashlib.sha256(text_hit.encode("utf-8")).hexdigest()
    p = cache_dir / "test-sha" / "en-es" / f"{sha_hit}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("hola", encoding="utf-8")

    t = Translator(cache_dir=cache_dir)
    monkeypatch.setattr(t, "_lazy_load", lambda: None)

    captured: list[list[str]] = []

    def fake(texts, src, tgt):
        captured.append(list(texts))
        return [f"<es>{x}" for x in texts]

    monkeypatch.setattr(t, "_translate_uncached", fake)

    out = t.translate_batch(["hello", "world", "foo"], src="en", tgt="es")
    assert out == ["hola", "<es>world", "<es>foo"]
    assert captured == [["world", "foo"]]


def test_auto_device_prefers_cuda_when_available(monkeypatch):
    import torch

    import minicoil_v2.eval.retrievers._common as common

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert common._auto_device() == "cuda"


def test_auto_device_falls_back_to_cpu_when_no_accelerator(monkeypatch):
    import torch

    import minicoil_v2.eval.retrievers._common as common

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert common._auto_device() == "cpu"
