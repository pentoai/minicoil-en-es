"""Model-free concept matching.

Single source of truth for "does concept C fire on this text in language L".
Shared by the encoder, the trainer's input encoding, and the eval coverage
filter so that "covered" provably means "the encoder fires concept C here".
No torch import — pure string/regex work.
"""

from __future__ import annotations

from collections.abc import Callable, Set

from minicoil_v2.constants import TOKEN_RE


def match_concepts(
    text: str,
    lang: str,
    word_to_concept: dict[str, dict[str, str]],
    allowed_cids: Set[str],
    lemmatizer: Callable[[str, str], str] | None = None,
    lemma_skip_words: Set[str] | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Return {concept_id: [(start, end), ...]} for every ``TOKEN_RE`` token that
    is a surface form (in ``lang``) of a concept in ``allowed_cids``.

    Spans are character offsets over the lowercased ``text``. The encoder and the
    trainer's input path call it through ``match_concepts_after_prefix``; the eval
    coverage filter calls it directly (surface forms only).

    ``lemmatizer`` (the encoder's ``lemma_match``, on in the published config):
    a token that misses the surface lookup is lemmatized and looked up again, so
    inflected forms ("gatos", "injections") fire their concept.
    ``lemma_skip_words`` gates BOTH the token and its lemma: without it,
    auxiliaries dominate the expansion (is/are -> be, es -> ser) and fire a
    near-stopword concept on almost every text. Defaults keep the surface-only
    path byte-for-byte.
    """
    w2c = word_to_concept.get(lang, {})
    skip = lemma_skip_words or frozenset()
    out: dict[str, list[tuple[int, int]]] = {}
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group()
        cid = w2c.get(token)
        if cid is None and lemmatizer is not None and token not in skip:
            lemma = lemmatizer(token, lang)
            if lemma != token and lemma not in skip:
                cid = w2c.get(lemma)
        if cid is None or cid not in allowed_cids:
            continue
        out.setdefault(cid, []).append(match.span())
    return out


def match_concepts_after_prefix(
    prefix: str,
    text: str,
    lang: str,
    word_to_concept: dict[str, dict[str, str]],
    allowed_cids: Set[str],
    lemmatizer: Callable[[str, str], str] | None = None,
    lemma_skip_words: Set[str] | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """``match_concepts`` over ``text`` only, with spans in ``prefix + text`` coordinates.

    The mE5 instruction prefixes (``"passage: "``, ``"query: "``) are part of the
    string the encoder tokenizes, but they are not part of the user's text. Both
    prefix words are concept surface forms in the EN-ES vocabulary (``passage`` and
    ``query``), so matching over the prefixed string made every English document
    emit the ``passage`` concept block and every English query the ``query`` one.
    Matching over ``text`` and shifting by ``len(prefix)`` keeps the spans aligned
    with the tokenizer's offsets for the prefixed string while ignoring the prefix.
    """
    shift = len(prefix)
    spans = match_concepts(
        text,
        lang,
        word_to_concept,
        allowed_cids,
        lemmatizer=lemmatizer,
        lemma_skip_words=lemma_skip_words,
    )
    return {cid: [(s + shift, e + shift) for s, e in sp] for cid, sp in spans.items()}
