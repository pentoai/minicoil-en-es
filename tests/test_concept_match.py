from minicoil_v2.concept_match import match_concepts

W2C = {
    "en": {"bank": "C-1", "banks": "C-1", "loan": "C-2"},
    "es": {"banco": "C-1", "prestamo": "C-2"},
}


def test_matches_all_occurrences_of_all_surface_forms():
    spans = match_concepts("bank and banks", "en", W2C, allowed_cids={"C-1", "C-2"})
    assert set(spans) == {"C-1"}
    assert len(spans["C-1"]) == 2  # both 'bank' and 'banks'


def test_respects_allowed_cids():
    spans = match_concepts("bank loan", "en", W2C, allowed_cids={"C-2"})
    assert set(spans) == {"C-2"}


def test_cross_lingual_form_es():
    assert set(match_concepts("el banco grande", "es", W2C, {"C-1"})) == {"C-1"}


def test_unknown_lang_returns_empty():
    assert match_concepts("bank", "fr", W2C, {"C-1"}) == {}


def test_spans_are_char_offsets():
    spans = match_concepts("a bank", "en", W2C, {"C-1"})
    s, e = spans["C-1"][0]
    assert "a bank"[s:e] == "bank"


def test_after_prefix_ignores_concept_words_in_the_prefix():
    from minicoil_v2.concept_match import match_concepts_after_prefix

    w2c = {"en": {"passage": "C-P", "query": "C-Q", "bank": "C-1"}}
    allowed = {"C-P", "C-Q", "C-1"}
    for prefix in ("passage: ", "query: "):
        assert match_concepts_after_prefix(prefix, "xqzzyflarn", "en", w2c, allowed) == {}


def test_after_prefix_spans_index_the_prefixed_string():
    from minicoil_v2.concept_match import match_concepts_after_prefix

    w2c = {"en": {"passage": "C-P", "bank": "C-1"}}
    prefix, text = "passage: ", "a bank, then a passage"
    spans = match_concepts_after_prefix(prefix, text, "en", w2c, {"C-P", "C-1"})
    prefixed = (prefix + text).lower()
    assert [prefixed[s:e] for s, e in spans["C-1"]] == ["bank"]
    # a concept word that is really in the text still fires, once
    assert [prefixed[s:e] for s, e in spans["C-P"]] == ["passage"]
    assert spans["C-P"][0][0] >= len(prefix)
