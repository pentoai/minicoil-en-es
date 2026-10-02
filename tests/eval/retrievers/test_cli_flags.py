from typer.testing import CliRunner

from minicoil_v2.cli import app


def test_run_help_shows_rebuild_and_qdrant_url_flags():
    runner = CliRunner()
    result = runner.invoke(app, ["eval", "run", "--help"])
    assert result.exit_code == 0
    assert "--rebuild" in result.stdout
    assert "--qdrant-url" in result.stdout


def test_default_retriever_modules_includes_baselines():
    from minicoil_v2.eval.cli import DEFAULT_RETRIEVER_MODULES

    assert "minicoil_v2.eval.retrievers.bm25" in DEFAULT_RETRIEVER_MODULES
    assert "minicoil_v2.eval.retrievers.minicoil_v1" in DEFAULT_RETRIEVER_MODULES
    assert "minicoil_v2.eval.retrievers.translate_bm25" in DEFAULT_RETRIEVER_MODULES
