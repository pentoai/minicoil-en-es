"""miniCOIL v2 encoder — text to sparse vectors with token-pooled concept embeddings.

Pipeline:
1. Prefix the text ("query: " / "passage: ") and tokenize with the mE5-small tokenizer
   (returns offset_mapping); concepts are matched on the unprefixed text
2. Forward through transformer once per text → last_hidden_state
3. For each matched concept word: pool subword tokens overlapping the word's
   char span, L2-normalize → 384D
4. Apply that concept's Linear(384, OUTPUT_DIM) + tanh → 8D, then normalize the block
   and scale it by the BM25 weight of the word it replaces
5. Build sparse vector: concept_num * OUTPUT_DIM + offset → value, plus BM25 backbone
   terms for every non-concept token (indices >= BACKBONE_BASE)

See docs/06-inference-and-scoring.md.

The token-pooled signal is what the layers were trained on — see
embed_wiki_sentences.py and token_pooling.encode_sentences_with_focals.

Usage:
    encoder = MiniCoilEncoder("data/")
    sparse = encoder.encode_sparse("The cat sat on the mat", lang="en")
    query = encoder.encode_sparse("cat on a mat", lang="en", is_query=True)
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import mmh3
import torch
from fastembed.common.utils import remove_non_alphanumeric

from minicoil_v2.concept_match import match_concepts_after_prefix
from minicoil_v2.constants import (
    HF_BIASES_KEY,
    HF_CONFIG_FILE,
    HF_WEIGHTS_FILE,
    HF_WEIGHTS_KEY,
    INPUT_ENCODER,
    OUTPUT_DIM,
    SPARSE_EPSILON,
    WORD_TO_CONCEPT_FILE,
)
from minicoil_v2.token_pooling import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    load_token_pool_model,
    pool_spans,
)
from minicoil_v2.utils import select_device

# BM25 lexical backbone. miniCOIL is BM25 with a learned semantic overlay: concept
# words become OUTPUT_DIM-value blocks, every other token still contributes a plain BM25 term so
# lexical recall doesn't collapse on out-of-vocab text. Backbone indices live in a
# range disjoint from concept indices (`concept_num*OUTPUT_DIM+offset`, ~<10**5) and
# stay int32-safe — Qdrant sparse indices must be < 2**31 (v1's max was 2_145_730_076).
#
# The backbone reuses FastEmbed's `Qdrant/bm25` pipeline (tokenize -> stopword/
# punctuation filter -> Snowball stem -> BM25 tf weight with document-length
# normalization, IDF applied by Qdrant's Modifier.IDF) — the exact implementation
# the locked bm25 baselines ran, so backbone-vs-baseline is apples-to-apples by
# construction. Like the baseline, it runs the english stemmer/stopwords for BOTH
# languages and avg_len=256. Parity is pinned by tests/test_backbone_parity.py.
BACKBONE_BASE = 1 << 27  # 134_217_728; concept indices are far below this
_BACKBONE_SPAN = (1 << 31) - 1 - BACKBONE_BASE  # hash modulus; keeps index < 2**31
BACKBONE_BM25_MODEL = "Qdrant/bm25"


def load_backbone_bm25():
    """The FastEmbed BM25 the locked baselines ran, default settings included
    (k=1.2, b=0.75, avg_len=256, english stemmer+stopwords for both languages).

    Lazy import: pulling in `fastembed.sparse.bm25` (and its model-file fetch)
    only when an encoder is actually constructed.
    """
    from fastembed.sparse.bm25 import Bm25

    return Bm25(BACKBONE_BM25_MODEL)


def load_lang_stopwords(language: str) -> set[str]:
    """Stopword set for ``language`` from the same fastembed BM25 model the backbone
    uses. The backbone runs the English stemmer + English stopwords for all text, so a
    query language's function words (Spanish "de"/"la"/"que") leak into the backbone and,
    being rare in a cross-language corpus, score as high-IDF noise. Stripping the text's
    own-language stopwords on top of the English ones removes that leak without touching
    the (English) stemmer, so legitimate cross-lingual matches (cognates, names, numbers)
    are preserved."""
    from fastembed.sparse.bm25 import Bm25

    return set(Bm25(BACKBONE_BM25_MODEL, language=language).stopwords)


def backbone_index(stem: str) -> int:
    """Deterministic, process-stable sparse index for a non-concept stem:
    fastembed's token id (abs of unsalted mmh3) remapped into the backbone's
    disjoint range — required for query/corpus index consistency."""
    return BACKBONE_BASE + abs(mmh3.hash(stem)) % _BACKBONE_SPAN


def _stem_pairs(text: str, bm25, extra_stopwords: set[str] | None = None) -> list[tuple[str, str]]:
    """(lowercased token, stem) pairs replicating fastembed ``Bm25._stem``
    byte-for-byte, but keeping the token so the concept carve can match on
    surface form. Parity is pinned by tests/test_backbone_parity.py.

    ``extra_stopwords`` (the text's own-language stopword set) is filtered ON TOP
    of fastembed's English stopwords, before the (unchanged, English) stemmer. This
    strips query-language function words that would otherwise leak into the
    cross-lingual backbone as high-IDF noise (see MiniCoilEncoder._lang_stopwords).
    Default None preserves the English backbone path byte-for-byte."""
    pairs: list[tuple[str, str]] = []
    for token in bm25.tokenizer.tokenize(remove_non_alphanumeric(text)):
        if token in bm25.punctuation:
            continue
        lower = token.lower()
        if lower in bm25.stopwords:
            continue
        if extra_stopwords is not None and lower in extra_stopwords:
            continue
        if len(token) > bm25.token_max_length:
            continue
        stem = bm25.stemmer.stem_word(lower) if bm25.stemmer else lower
        if stem:
            pairs.append((lower, stem))
    return pairs


def _bm25_doc_weight(tf: int, doc_len: int, bm25) -> float:
    """fastembed's BM25 term weight: tf saturation (k) + doc-length norm (b)."""
    return tf * (bm25.k + 1) / (tf + bm25.k * (1 - bm25.b + bm25.b * doc_len / bm25.avg_len))


def _terms_from_pairs(
    pairs: list[tuple[str, str]], concept_words: set[str], bm25, is_query: bool
) -> dict[int, float]:
    # doc_len counts every kept stem, carved concept tokens included — the
    # document really is that long; only the *term* moves to its concept block.
    doc_len = len(pairs)
    counts: dict[str, int] = {}
    for token, stem in pairs:
        if token in concept_words:
            continue
        counts[stem] = counts.get(stem, 0) + 1
    if is_query:
        # fastembed query side: distinct stems at weight 1.0, no tf/length terms.
        return {backbone_index(stem): 1.0 for stem in counts}
    return {backbone_index(s): _bm25_doc_weight(n, doc_len, bm25) for s, n in counts.items()}


def backbone_terms(
    text: str, concept_words: set[str], bm25, is_query: bool = False
) -> dict[int, float]:
    """BM25 backbone for one text: one term per distinct stem that is *not* a
    concept word (those are represented by their learned concept block, so excluding
    them here avoids double counting)."""
    return _terms_from_pairs(_stem_pairs(text, bm25), concept_words, bm25, is_query)


def load_concept_layers(
    data_dir: Path,
    model_file: str = "concept_layers.pt",
    model_path: str | Path | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor | None]:
    """Load the per-concept heads as (concept_ids, weights, biases).

    ``weights`` is (num_concepts, OUTPUT_DIM, INPUT_DIM) stacked in ``concept_ids``
    order; ``biases`` is (num_concepts, OUTPUT_DIM) or None when the checkpoint has
    none (the cosine objective trains weight-only heads).

    Two layouts are accepted, so a published model directory and a training output
    directory are both valid encoder inputs:

    - ``model.safetensors`` + ``config.json`` (published layout): concept order is
      the ``concept_ids`` list in the config, since safetensors holds one stacked
      tensor rather than a tensor per concept.
    - ``concept_models/<model_file>`` (training layout): the trainer's torch dict of
      ``{concept_id: {"weight": ..., "bias": ...}}``, keyed in sorted id order.

    An explicit ``model_path`` overrides both and is dispatched on its suffix.
    """
    if model_path is not None:
        path = Path(model_path)
    elif (data_dir / HF_WEIGHTS_FILE).exists():
        path = data_dir / HF_WEIGHTS_FILE
    else:
        path = data_dir / "concept_models" / model_file

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        config_path = path.parent / HF_CONFIG_FILE
        if not config_path.exists():
            raise FileNotFoundError(
                f"{path} needs {HF_CONFIG_FILE} beside it: the stacked tensor carries no "
                "concept ids, so the config's concept_ids list defines row order."
            )
        with open(config_path) as f:
            config = json.load(f)
        tensors = load_file(str(path))
        return list(config["concept_ids"]), tensors[HF_WEIGHTS_KEY], tensors.get(HF_BIASES_KEY)

    layers = torch.load(path, map_location="cpu", weights_only=True)
    concept_ids = sorted(layers.keys())
    weights = torch.stack([layers[cid]["weight"] for cid in concept_ids])
    biases = [layers[cid].get("bias") for cid in concept_ids]
    stacked_b = torch.stack(biases) if all(b is not None for b in biases) else None
    return concept_ids, weights, stacked_b


class MiniCoilEncoder:
    """Encodes text into miniCOIL v2 sparse vectors using token-pooled concept embeddings."""

    def __init__(
        self,
        data_dir: str | Path,
        device: str = "auto",
        model_file: str = "concept_layers.pt",
        model_path: str | Path | None = None,
        lemma_match: bool | None = None,
    ):
        data_dir = Path(data_dir)
        self.device = select_device(device)

        self.tokenizer, self.model = load_token_pool_model(INPUT_ENCODER, self.device)
        self.bm25 = load_backbone_bm25()
        # Language-aware backbone stopwords (stemmer unchanged): strip a query's own
        # language function words that the English-only backbone would otherwise leak as
        # high-IDF cross-lingual noise (applied query-side in encode_batch_sparse). Spanish
        # was the dominant spa-eng rank-11-100 failure (see constants.XL_BACKBONE_WEIGHT).
        # Non-Spanish or doc-side text gets None (no change). Toggle off for A/B
        # via MINICOIL_V2_BACKBONE_STOPWORD_FIX=0 (default on).
        if os.environ.get("MINICOIL_V2_BACKBONE_STOPWORD_FIX", "1") != "0":
            self._lang_stopwords = {"es": load_lang_stopwords("spanish")}
        else:
            self._lang_stopwords = {}

        # Lemma fallback: concept matching falls back to a lemma lookup for tokens
        # that miss the surface form ("gatos" -> "gato", "injections" -> "injection"),
        # gated on stopwords for both token and lemma (see concept_match.match_concepts).
        # On in the published config (config.json "lemma_match", applied by
        # from_pretrained); for a bare checkpoint pass lemma_match=True or set
        # MINICOIL_V2_LEMMA_MATCH=1. Applies to query AND document encoding, so eval
        # runs must pass --rebuild after toggling it.
        if lemma_match is None:
            lemma_match = os.environ.get("MINICOIL_V2_LEMMA_MATCH", "0") == "1"
        self.lemma_match = lemma_match
        if lemma_match:
            import simplemma

            def _lemmatize(token: str, lang: str) -> str:
                return simplemma.lemmatize(token, lang=lang if lang in ("en", "es") else "en")

            self._lemmatizer = _lemmatize
            en_sw = load_lang_stopwords("english")
            self._lemma_skip: dict[str, set[str]] = {
                "en": en_sw,
                "es": en_sw | load_lang_stopwords("spanish"),
            }
        else:
            self._lemmatizer = None
            self._lemma_skip = {}

        with open(data_dir / WORD_TO_CONCEPT_FILE) as f:
            self.word_to_concept: dict[str, dict[str, str]] = json.load(f)

        concept_ids, stacked_w, stacked_b = load_concept_layers(
            data_dir=data_dir, model_file=model_file, model_path=model_path
        )

        self.concept_ids = concept_ids
        self._loaded_cids = set(self.concept_ids)
        self.concept_id_to_idx = {cid: i for i, cid in enumerate(self.concept_ids)}
        self.num_concepts = len(self.concept_ids)
        self.concept_nums = [int(cid.split("-")[1]) for cid in self.concept_ids]

        self.all_weights = stacked_w.float().to(self.device)
        if stacked_b is not None:
            self.all_biases = stacked_b.float().to(self.device)
        else:
            self.all_biases = torch.zeros(self.num_concepts, OUTPUT_DIM, device=self.device)

        self.sparse_dim = self.num_concepts * OUTPUT_DIM

        # Experiment toggle (MINICOIL_V2_CONCEPT_MODE):
        #   "learned" (default) - matched concepts emit their trained 8D tanh block.
        #   "lexical"           - matched concepts emit ONE BM25-weighted term at a
        #                         shared concept-keyed index, replacing the learned
        #                         block. "perro" and "dog" map to the same concept, so
        #                         they collide on the same index and match BM25-style
        #                         across languages, independent of the learned head.
        #                         This is the cross-lingual lexical bridge test.
        self.concept_mode = os.environ.get("MINICOIL_V2_CONCEPT_MODE", "learned")

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | Path,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | None = None,
        device: str = "auto",
        **kwargs,
    ) -> "MiniCoilEncoder":
        """Load a published model, by Hugging Face repo id or local export directory.

        Inference defaults recorded at export time (currently ``lemma_match``) come
        from the repo's ``config.json``, so the downloaded model reproduces the
        published numbers without the caller setting environment variables. An
        explicit keyword still wins.
        """
        local_dir = Path(repo_id)
        if not local_dir.is_dir():
            from huggingface_hub import snapshot_download

            local_dir = Path(
                snapshot_download(
                    repo_id=str(repo_id),
                    revision=revision,
                    cache_dir=str(cache_dir) if cache_dir else None,
                    token=token,
                )
            )

        config_path = local_dir / HF_CONFIG_FILE
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        kwargs.setdefault("lemma_match", config.get("lemma_match"))
        return cls(local_dir, device=device, **kwargs)

    @torch.no_grad()
    def encode_concept_vectors(
        self,
        text: str,
        lang: str = "en",
        prefix: str = PASSAGE_PREFIX,
        max_length: int = 256,
    ) -> dict[str, torch.Tensor]:
        """Pre-Linear pooled 384D vectors per concept that fires on ``text``.

        This is exactly the representation fed into each concept's
        ``Linear(384, OUTPUT_DIM)`` at inference time — the seam the trainer must match
        byte-for-byte. Returns {concept_id: cpu tensor}.
        """
        prefixed = prefix + text
        enc = self.tokenizer(
            prefixed,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        hidden = self.model(**enc).last_hidden_state[0]
        spans = match_concepts_after_prefix(
            prefix,
            text,
            lang,
            self.word_to_concept,
            self._loaded_cids,
            lemmatizer=self._lemmatizer,
            lemma_skip_words=self._lemma_skip.get(lang),
        )
        out: dict[str, torch.Tensor] = {}
        for cid, sp in spans.items():
            vec = pool_spans(hidden, offsets, sp)
            if vec is not None:
                out[cid] = vec.cpu()
        return out

    @torch.no_grad()
    def encode_sparse(
        self,
        text: str,
        lang: str = "en",
        prefix: str | None = None,
        include_backbone: bool = True,
        is_query: bool = False,
    ) -> dict[int, float]:
        """Encode a single text into a sparse vector (see ``encode_batch_sparse``)."""
        return self.encode_batch_sparse(
            [text],
            lang=lang,
            prefix=prefix,
            include_backbone=include_backbone,
            is_query=is_query,
        )[0]

    @torch.no_grad()
    def encode_batch_sparse(
        self,
        texts: list[str],
        lang: str = "en",
        prefix: str | None = None,
        batch_size: int = 32,
        max_length: int = 256,
        include_backbone: bool = True,
        is_query: bool = False,
        backbone_weight: float = 1.0,
    ) -> list[dict[int, float]]:
        """Encode a batch of texts into sparse vectors.

        ``is_query`` mirrors fastembed's BM25 query/document asymmetry: queries
        emit flat 1.0 weights (backbone terms and concept-block scale alike),
        documents get tf-saturated, length-normalized weights. It also picks the
        mE5 instruction prefix when ``prefix`` is None: ``"query: "`` for queries,
        ``"passage: "`` for documents (the pairing the eval harness scores). The
        prefix only conditions the encoder; concepts are matched on the text alone.

        ``backbone_weight`` scales the BM25 backbone terms (indices >=
        BACKBONE_BASE). Callers that know the query and corpus languages differ
        pass < 1.0 to downweight the cross-lingual backbone noise (see
        constants.XL_BACKBONE_WEIGHT); the learned concept block is left intact.
        """
        if not texts:
            return []
        if prefix is None:
            prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX

        all_results: list[dict[int, float]] = []
        concept_nums_t = torch.tensor(self.concept_nums, dtype=torch.long, device=self.device)
        offsets = torch.arange(OUTPUT_DIM, dtype=torch.long, device=self.device)

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            prefixed = [prefix + t for t in chunk]
            enc = self.tokenizer(
                prefixed,
                return_tensors="pt",
                return_offsets_mapping=True,
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            offset_mapping = enc.pop("offset_mapping").tolist()
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            hidden = out.last_hidden_state

            for b, text in enumerate(chunk):
                # Spans index into the prefixed string (the tokenizer's offsets), but
                # are matched on the text alone, so the prefix never fires a concept.
                text_lower = prefixed[b].lower()
                concept_spans = match_concepts_after_prefix(
                    prefix,
                    text,
                    lang,
                    self.word_to_concept,
                    self._loaded_cids,
                    lemmatizer=self._lemmatizer,
                    lemma_skip_words=self._lemma_skip.get(lang),
                )

                # BM25 lexical backbone over every non-concept token, so out-of-vocab
                # words (names, numbers, rare terms) still drive retrieval. Concept
                # words are excluded — they're carried by their learned concept block below.
                # Index spaces are disjoint, so the two dicts merge without collision.
                # Edge case: a concept word that appears only past the max_length
                # truncation is matched (regex sees the full text) so it's excluded
                # here, yet yields no block (no in-window tokens) — it then contributes
                # nothing. Rare, and harmless to retrieval.
                concept_words = {
                    text_lower[s:e] for spans in concept_spans.values() for s, e in spans
                }
                # Stemmed once; doc_len feeds both backbone terms and the concept
                # blocks' displaced-term weight below.
                # Query-side only: the leak is a query emitting a spurious high-IDF
                # stopword term, so stripping the query is the validated, sufficient fix.
                # Leaving docs at the full backbone keeps doc_len/IDF identical, so the
                # other 3 pairs stay exact no-ops (doc-side stripping nicked eng-spa).
                extra_stopwords = self._lang_stopwords.get(lang) if is_query else None
                stem_pairs = _stem_pairs(text, self.bm25, extra_stopwords)
                doc_len = len(stem_pairs)
                result: dict[int, float] = (
                    _terms_from_pairs(stem_pairs, concept_words, self.bm25, is_query)
                    if include_backbone
                    else {}
                )
                # Cross-lingual backbone downweight: across languages the BM25 backbone
                # is mostly noise (cognate/number/short-token stem collisions) that
                # outscores the true concept bridge. Scale only the backbone terms and
                # let the learned concept block (added below) do the ranking.
                if backbone_weight != 1.0:
                    result = {
                        idx: (v * backbone_weight if idx >= BACKBONE_BASE else v)
                        for idx, v in result.items()
                    }

                # Cross-lingual lexical bridge: each matched concept emits ONE
                # BM25-weighted term at its concept-keyed index (offset 0), replacing
                # the learned 8D block. Two texts that share a concept (e.g. es "perro"
                # / en "dog" -> same concept) collide on the same index and match
                # BM25-style, independent of the learned head. IDF is applied by
                # Qdrant's Modifier.IDF at query time, same as the backbone terms.
                if self.concept_mode == "lexical":
                    for cid, spans in concept_spans.items():
                        cnum = self.concept_nums[self.concept_id_to_idx[cid]]
                        w = 1.0 if is_query else _bm25_doc_weight(len(spans), doc_len, self.bm25)
                        result[cnum * OUTPUT_DIM] = w
                    all_results.append(result)
                    continue

                # Pool each concept via the shared helper so training and inference
                # use byte-for-byte the same representation (see token_pooling.pool_spans).
                pooled_by_cidx: dict[int, torch.Tensor] = {}
                counts_by_cidx: dict[int, int] = {}
                for cid, spans in concept_spans.items():
                    vec = pool_spans(hidden[b], offset_mapping[b], spans)
                    if vec is None:
                        continue
                    cidx = self.concept_id_to_idx[cid]
                    pooled_by_cidx[cidx] = vec
                    counts_by_cidx[cidx] = len(spans)

                if pooled_by_cidx:
                    cidx_list = list(pooled_by_cidx)
                    pooled = torch.stack([pooled_by_cidx[c] for c in cidx_list])

                    cidx_t = torch.tensor(cidx_list, dtype=torch.long, device=self.device)
                    W = self.all_weights[cidx_t]
                    bias = self.all_biases[cidx_t]
                    output = torch.tanh(torch.einsum("nod,nd->no", W, pooled) + bias)

                    # The concept block replaces the lexical term carved out of the BM25
                    # backbone, so it must carry comparable weight. The heads are trained
                    # with a scale-invariant cosine objective, so |tanh| is arbitrary
                    # (~0.17) and far weaker than a lexical term (~1.0). Normalize each
                    # block to a unit sense-direction and scale by the same BM25 weight
                    # the displaced lexical term would have had (1.0 on the query side,
                    # tf-saturated + length-normalized on the document side): a same-
                    # sense match (cos->1) then scores like the lexical weight, opposite
                    # senses (cos<0) suppress. Cosine is normalization-invariant, so
                    # this is exactly what training optimized.
                    norms = output.norm(dim=1, keepdim=True).clamp_min(SPARSE_EPSILON)
                    tf_w = torch.tensor(
                        [
                            1.0
                            if is_query
                            else _bm25_doc_weight(counts_by_cidx[c], doc_len, self.bm25)
                            for c in cidx_list
                        ],
                        dtype=output.dtype,
                        device=self.device,
                    ).unsqueeze(1)
                    output = output / norms * tf_w

                    bases = concept_nums_t[cidx_t] * OUTPUT_DIM
                    sparse_indices = (bases.unsqueeze(1) + offsets.unsqueeze(0)).flatten()
                    flat_vals = output.flatten()
                    mask = flat_vals.abs() > SPARSE_EPSILON
                    sparse_indices_cpu = sparse_indices[mask].cpu().numpy()
                    flat_vals_cpu = flat_vals[mask].cpu().numpy()
                    result.update(
                        dict(
                            zip(
                                sparse_indices_cpu.tolist(),
                                flat_vals_cpu.tolist(),
                                strict=True,
                            )
                        )
                    )

                all_results.append(result)

        return all_results
