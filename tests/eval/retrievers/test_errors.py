from minicoil_v2.eval.retrievers._common import (
    CorpusHashMismatch,
    UnsupportedPairError,
)


def test_corpus_hash_mismatch_message_includes_collection_and_short_shas():
    err = CorpusHashMismatch(
        collection="bm25_en_mlqa",
        stored_sha="abc123def456" + "0" * 52,
        new_sha="999888777666" + "0" * 52,
    )
    msg = str(err)
    assert "bm25_en_mlqa" in msg
    assert "abc123de" in msg
    assert "99988877" in msg
    assert "--rebuild" in msg


def test_corpus_hash_mismatch_is_runtime_error():
    err = CorpusHashMismatch("x", "a" * 64, "b" * 64)
    assert isinstance(err, RuntimeError)


def test_unsupported_pair_error_message():
    err = UnsupportedPairError(retriever="minicoil-v1", pair="spa-spa")
    msg = str(err)
    assert "minicoil-v1" in msg
    assert "spa-spa" in msg


def test_unsupported_pair_error_is_value_error():
    err = UnsupportedPairError("x", "y")
    assert isinstance(err, ValueError)


def test_unsupported_pair_error_attributes_accessible():
    err = UnsupportedPairError(retriever="r", pair="p")
    assert err.retriever == "r"
    assert err.pair == "p"
