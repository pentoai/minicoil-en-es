"""The published model must encode identically to the checkpoint it came from.

The export swaps the storage format (a torch dict of per-concept heads becomes one
stacked safetensors tensor plus a concept id list) and moves the vocabulary files
into a flat directory. Both are places where concept order could silently shift —
a permuted row maps every concept to the wrong head, which no smoke test catches
because the output still looks like a sparse vector. So the check is byte-for-byte
equality of the emitted sparse vectors, not a similarity threshold.
"""

import json
from pathlib import Path

import pytest
import torch

from minicoil_v2.constants import (
    CONCEPT_VOCABULARY_FILE,
    HF_CONFIG_FILE,
    HF_WEIGHTS_FILE,
    WORD_TO_CONCEPT_FILE,
)
from minicoil_v2.hf_export import build_config, build_export, render_model_card

_DATA = Path("data")
_CKPT = _DATA / "cm_fast_stack" / "concept_layers.pt"
_VOCAB_DIR = _DATA / "phase2"

_HAS_MODEL = _CKPT.exists() and (_VOCAB_DIR / WORD_TO_CONCEPT_FILE).exists()

_TEXTS_EN = ["the dog ran through the park", "reykjavik xqzzyflarn", "food for the people"]
_TEXTS_ES = ["el perro corrió por el parque", "comida para la gente"]


def test_config_records_concept_order_and_defaults():
    config = build_config(["C-00001", "C-00000"], lemma_match=True)

    # Order is preserved exactly as given: it indexes rows of the stacked tensor,
    # so re-sorting here would silently remap every concept.
    assert config["concept_ids"] == ["C-00001", "C-00000"]
    assert config["num_concepts"] == 2
    assert config["lemma_match"] is True
    assert config["requires_idf"] is True


def test_card_omits_results_without_a_report():
    config = build_config(["C-00000"], lemma_match=False)
    card = render_model_card(config, repo_id="acme/minicoil", license_id="mit")

    assert "## Results" not in card
    assert "license: mit" in card
    assert "Modifier.IDF" in card


def test_card_renders_deltas_against_locked_baselines():
    config = build_config(["C-00000"], lemma_match=True)
    report = {
        "command": "minicoil eval run minicoil-v2",
        "git_sha": "abc123",
        "results": {"eng-eng": {"test": {"n_queries": 10, "mrr10": {"mean": 0.9}}}},
    }
    baselines = {"bm25": {"eng-eng": {"test": {"mrr10": {"value": 0.8}}}}}

    card = render_model_card(
        config, repo_id="acme/minicoil", license_id="mit", report=report, baselines=baselines
    )

    assert "| en → en | MRR@10 | 0.9000 | 10 | +0.1000 vs bm25 |" in card


@pytest.mark.skipif(not _HAS_MODEL, reason=f"requires {_CKPT} and {_VOCAB_DIR}")
def test_export_layout_is_complete(tmp_path):
    out = build_export(
        checkpoint=_CKPT,
        data_dir=_VOCAB_DIR,
        out_dir=tmp_path / "export",
        repo_id="acme/minicoil-v2-en-es",
        lemma_match=True,
    )

    for filename in (
        HF_WEIGHTS_FILE,
        HF_CONFIG_FILE,
        WORD_TO_CONCEPT_FILE,
        CONCEPT_VOCABULARY_FILE,
    ):
        assert (out / filename).exists(), f"{filename} missing from export"
    assert (out / "README.md").exists()

    config = json.loads((out / HF_CONFIG_FILE).read_text())
    layers = torch.load(_CKPT, map_location="cpu", weights_only=True)
    assert config["concept_ids"] == sorted(layers)
    assert config["num_concepts"] == len(layers)


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MODEL, reason=f"requires {_CKPT} and {_VOCAB_DIR}")
@pytest.mark.parametrize("lemma_match", [False, True])
def test_exported_model_encodes_identically_to_checkpoint(tmp_path, lemma_match):
    from minicoil_v2.encoder import MiniCoilEncoder

    export_dir = build_export(
        checkpoint=_CKPT,
        data_dir=_VOCAB_DIR,
        out_dir=tmp_path / "export",
        repo_id="acme/minicoil-v2-en-es",
        lemma_match=lemma_match,
    )

    source = MiniCoilEncoder(_VOCAB_DIR, model_path=_CKPT, lemma_match=lemma_match)
    # from_pretrained on a local directory: no network, and lemma_match comes from
    # the config rather than the caller, which is how a download behaves.
    published = MiniCoilEncoder.from_pretrained(export_dir)
    assert published.lemma_match is lemma_match

    for texts, lang in ((_TEXTS_EN, "en"), (_TEXTS_ES, "es")):
        for is_query in (False, True):
            expected = source.encode_batch_sparse(texts, lang=lang, is_query=is_query)
            actual = published.encode_batch_sparse(texts, lang=lang, is_query=is_query)
            assert actual == expected, f"{lang} is_query={is_query} diverged after export"


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MODEL, reason=f"requires {_CKPT} and {_VOCAB_DIR}")
def test_vendored_code_encodes_standalone(tmp_path):
    """The published repo has to work for someone who cannot install this repo.

    Run in a subprocess whose PYTHONPATH is the export directory alone, so the
    import resolves to the vendored copy rather than `src/` — and assert that,
    since a silently-shadowed import would make this test prove nothing.
    """
    import os
    import subprocess
    import sys

    from minicoil_v2.encoder import MiniCoilEncoder

    export_dir = build_export(
        checkpoint=_CKPT,
        data_dir=_VOCAB_DIR,
        out_dir=tmp_path / "export",
        repo_id="acme/minicoil-v2-en-es",
        lemma_match=True,
        vendor_code=True,
    )

    script = (
        "import json, sys\n"
        "import minicoil_v2\n"
        "assert minicoil_v2.__file__.startswith(sys.argv[1]), minicoil_v2.__file__\n"
        "from minicoil_v2.encoder import MiniCoilEncoder\n"
        "enc = MiniCoilEncoder.from_pretrained(sys.argv[1])\n"
        "print(json.dumps(enc.encode_sparse(sys.argv[2], lang='en')))\n"
    )
    text = "the dog ran through the park"
    result = subprocess.run(
        [sys.executable, "-c", script, str(export_dir), text],
        env={**os.environ, "PYTHONPATH": str(export_dir)},
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    vendored = {int(k): v for k, v in json.loads(result.stdout).items()}
    source = MiniCoilEncoder(_VOCAB_DIR, model_path=_CKPT, lemma_match=True)
    assert vendored == source.encode_sparse(text, lang="en")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MODEL, reason=f"requires {_CKPT} and {_VOCAB_DIR}")
def test_lemma_match_fires_on_inflected_forms():
    """Guards the reason lemma matching ships on: a plural that misses the surface
    lookup must still reach its concept block, and stopword-gated auxiliaries must
    not (they map to a near-universal concept and swamp the signal)."""
    from minicoil_v2.encoder import BACKBONE_BASE, MiniCoilEncoder

    off = MiniCoilEncoder(_VOCAB_DIR, model_path=_CKPT, lemma_match=False)
    on = MiniCoilEncoder(_VOCAB_DIR, model_path=_CKPT, lemma_match=True)

    import simplemma

    from minicoil_v2.constants import TOKEN_RE

    # A single-token concept word whose plural is NOT itself a surface form, and that
    # simplemma actually folds back. Multi-word entries ("new york") are excluded:
    # their plural still tokenizes into a token that matches on its own, which would
    # make the OFF assertion below fail for reasons unrelated to lemmatization.
    w2c = off.word_to_concept["en"]
    inflected = next(
        (
            f"{word}s"
            for word, cid in w2c.items()
            if cid in off._loaded_cids
            and TOKEN_RE.fullmatch(word)
            and f"{word}s" not in w2c
            and simplemma.lemmatize(f"{word}s", lang="en") == word
        ),
        None,
    )
    assert inflected is not None, "no plural-of-a-concept-word available in this vocabulary"

    def concept_indices(encoder, text):
        """Concept blocks the text itself contributes.

        Subtracting the blocks of a text with no concept words guards against any
        block that fires regardless of the text (the mE5 prefix used to do that,
        since "passage" is a concept word; see tests/test_prefix_no_leak.py).
        """
        [out] = encoder.encode_batch_sparse([text], lang="en")
        [control] = encoder.encode_batch_sparse(["xqzzyflarn"], lang="en")
        return {i for i in out if i < BACKBONE_BASE} - {i for i in control if i < BACKBONE_BASE}

    assert not concept_indices(off, inflected)
    assert concept_indices(on, inflected)
    # "are" -> "be" is exactly the auxiliary expansion the stopword gate exists to block
    assert not concept_indices(on, "are")
