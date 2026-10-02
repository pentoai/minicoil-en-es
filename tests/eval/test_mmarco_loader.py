from pathlib import Path

from minicoil_v2.eval.datasets.mmarco import MMARCO_REVISION, MMARCODataset, _doc_id


def _write_fixtures(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "english_collection.tsv").write_text("0\tThe bank approved it\n1\tA river delta\n")
    (cache / "spanish_collection.tsv").write_text("0\tEl banco lo aprobo\n1\tUn delta del rio\n")
    (cache / "english_queries.dev.small.tsv").write_text("10\twhat bank\n11\twhat river\n")
    (cache / "spanish_queries.dev.small.tsv").write_text("10\tque banco\n11\tque rio\n")
    (cache / "qrels.dev.small.tsv").write_text("10\t0\t0\t1\n11\t0\t1\t1\n")


def _ds(tmp_path, monkeypatch) -> MMARCODataset:
    cache = tmp_path / "mmarco"
    monkeypatch.setattr(MMARCODataset, "_ensure_files", lambda self: _write_fixtures(cache))
    return MMARCODataset(cache_dir=cache)


def test_name_and_corpus(tmp_path, monkeypatch):
    ds = _ds(tmp_path, monkeypatch)
    assert ds.name == "mmarco"
    assert ds.corpus("en")[_doc_id("The bank approved it")] == "The bank approved it"
    assert ds.corpus("es")[_doc_id("El banco lo aprobo")] == "El banco lo aprobo"
    assert len(ds.corpus("en")) == 2


def test_eng_eng_monolingual_gold(tmp_path, monkeypatch):
    ds = _ds(tmp_path, monkeypatch)
    assert ds.queries("eng-eng", "validation")["10"] == "what bank"
    assert ds.qrels("eng-eng", "validation")["10"] == {_doc_id("The bank approved it")}


def test_eng_spa_cross_lingual_gold(tmp_path, monkeypatch):
    ds = _ds(tmp_path, monkeypatch)
    # EN query, gold is the ES passage with the same pid
    assert ds.queries("eng-spa", "validation")["10"] == "what bank"
    assert ds.qrels("eng-spa", "validation")["10"] == {_doc_id("El banco lo aprobo")}


def test_spa_eng_cross_lingual_gold(tmp_path, monkeypatch):
    ds = _ds(tmp_path, monkeypatch)
    assert ds.queries("spa-eng", "validation")["10"] == "que banco"
    assert ds.qrels("spa-eng", "validation")["10"] == {_doc_id("The bank approved it")}


def test_all_four_pairs_present(tmp_path, monkeypatch):
    ds = _ds(tmp_path, monkeypatch)
    for pair in ("eng-eng", "spa-spa", "eng-spa", "spa-eng"):
        assert len(ds.queries(pair, "validation")) == 2
        assert len(ds.qrels(pair, "validation")) == 2


def test_test_split_is_empty(tmp_path, monkeypatch):
    ds = _ds(tmp_path, monkeypatch)
    assert ds.queries("eng-eng", "test") == {}
    assert ds.qrels("eng-eng", "test") == {}


def test_revision_is_pinned():
    assert isinstance(MMARCO_REVISION, str) and len(MMARCO_REVISION) >= 7
