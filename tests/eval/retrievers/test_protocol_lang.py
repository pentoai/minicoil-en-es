from minicoil_v2.eval.retriever import BaseRetriever


def test_evaluate_passes_lang_to_index():
    """BaseRetriever.evaluate must call self.index(corpus, lang)."""
    captured: dict[str, str] = {}

    class CaptureRetriever(BaseRetriever):
        name = "capture"

        def index(self, corpus, lang):
            captured["lang"] = lang

        def search(self, query, k):
            return []

    class StubDataset:
        name = "stub"
        revision = "stub"

        def corpus(self, lang):
            return {"d1": "hola"} if lang == "es" else {"d1": "hello"}

        def queries(self, pair, split):
            return {"q1": "hi"}

        def qrels(self, pair, split):
            return {"q1": {"d1"}}

    r = CaptureRetriever()
    r.evaluate(StubDataset(), pair="eng-spa", split="dev", k=10)
    assert captured["lang"] == "es"
