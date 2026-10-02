from minicoil_v2.eval.retrievers._common import sha256_corpus


def test_sha256_corpus_is_deterministic_regardless_of_insertion_order():
    a = {"d1": "hello", "d2": "world", "d3": "foo"}
    b = {"d3": "foo", "d1": "hello", "d2": "world"}
    assert sha256_corpus(a) == sha256_corpus(b)


def test_sha256_corpus_changes_when_a_doc_changes():
    a = {"d1": "hello", "d2": "world"}
    b = {"d1": "hello", "d2": "WORLD"}
    assert sha256_corpus(a) != sha256_corpus(b)


def test_sha256_corpus_changes_when_a_doc_id_changes():
    a = {"d1": "hello", "d2": "world"}
    b = {"d1": "hello", "d3": "world"}
    assert sha256_corpus(a) != sha256_corpus(b)


def test_sha256_corpus_returns_64_hex_chars():
    digest = sha256_corpus({"d1": "x"})
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
