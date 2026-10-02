"""Tests for the eval tokenizer (re-exports project TOKEN_RE)."""

from minicoil_v2.constants import TOKEN_RE as PROJECT_TOKEN_RE
from minicoil_v2.eval.tokenize import TOKEN_RE, tokenize


def test_token_re_is_the_project_regex():
    """Eval must use the same tokenizer as concept matching to stay aligned."""
    assert TOKEN_RE is PROJECT_TOKEN_RE


def test_tokenize_lowercases():
    assert tokenize("Hello WORLD") == ["hello", "world"]


def test_tokenize_drops_digits_and_underscores():
    # TOKEN_RE is [^\W\d_]+, so digits and underscores are token boundaries.
    assert tokenize("foo_bar 123 baz") == ["foo", "bar", "baz"]


def test_tokenize_handles_spanish_accents():
    assert tokenize("¿Cómo estás?") == ["cómo", "estás"]


def test_tokenize_empty_string():
    assert tokenize("") == []
