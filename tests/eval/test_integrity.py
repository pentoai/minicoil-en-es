"""Integrity guarantees for agent-driven iteration:

1. The `test` split is sealed: it can only be run with an explicit env unlock,
   so a fast-iterating agent cannot overfit to test by repeated selection.
2. Retrievers are auto-discovered from the package, so an experiment is a
   drop-in file with no edit to a central list.
"""

from typer.testing import CliRunner

from minicoil_v2.cli import app


def test_test_split_is_sealed_without_env_gate(monkeypatch):
    monkeypatch.delenv("MINICOIL_EVAL_ALLOW_TEST", raising=False)
    runner = CliRunner()
    # Nonexistent retriever: if the seal fires first (correct), we get the seal
    # message; if it doesn't, we'd fall through to "unknown retriever".
    result = runner.invoke(app, ["eval", "run", "no-such-retriever", "--split", "test"])
    assert result.exit_code != 0
    assert "MINICOIL_EVAL_ALLOW_TEST" in result.output or "sealed" in result.output.lower()


def test_test_split_allowed_with_env_gate(monkeypatch):
    monkeypatch.setenv("MINICOIL_EVAL_ALLOW_TEST", "1")
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "run", "no-such-retriever", "--split", "test"])
    # Seal is unlocked -> falls through to the normal unknown-retriever error.
    assert "unknown retriever" in result.output.lower()
    assert "sealed" not in result.output.lower()


def test_val_split_is_never_sealed(monkeypatch):
    monkeypatch.delenv("MINICOIL_EVAL_ALLOW_TEST", raising=False)
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "run", "no-such-retriever", "--split", "val"])
    # val must always be runnable for iteration; no seal regardless of env.
    assert "unknown retriever" in result.output.lower()
    assert "sealed" not in result.output.lower()


def test_discover_retriever_modules_scans_package():
    from minicoil_v2.eval.cli import _discover_retriever_modules

    found = _discover_retriever_modules()
    assert "minicoil_v2.eval.retrievers.bm25" in found
    assert "minicoil_v2.eval.retrievers.minicoil_v1" in found
    assert "minicoil_v2.eval.retrievers.translate_bm25" in found
    # private/helper modules are not retrievers and must be excluded
    assert "minicoil_v2.eval.retrievers._common" not in found
