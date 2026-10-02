# Third-party notices

miniCOIL EN-ES builds on the data, models and libraries below. Each keeps its own
license; this file is a summary, not a substitute for those licenses.

## What determines this repository's license

| Component | Use | License |
|---|---|---|
| [MUSE bilingual dictionaries](https://github.com/facebookresearch/MUSE) (Facebook Research) | Source of every concept: `concept_vocabulary.json` and `word_to_concept.json` are derived from the EN-ES pairs | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

Because the concept vocabulary is an adaptation of the MUSE dictionaries, the vocabulary
and the trained model are released under CC BY-NC 4.0, and so is this repository
(see [LICENSE](LICENSE) and the README).

## Models used to train or run the model

| Component | Use | License |
|---|---|---|
| [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) | Input encoder at training and inference time (not redistributed) | MIT |
| [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Teacher for the published checkpoint, at embed time only | Apache-2.0 |
| [FastEmbed](https://github.com/qdrant/fastembed) `Qdrant/bm25` | BM25 backbone (tokenizer, stopwords, stemmer, weights) | Apache-2.0 |
| [simplemma](https://github.com/adbar/simplemma) | Lemma fallback | MIT (code); its lemma data comes from sources under their own licenses, listed in the simplemma repository |

## Training data

| Component | Use | License |
|---|---|---|
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (EN and ES, 2023-11-01 dumps) | Sentences for each concept, streamed at embed time; not redistributed here | CC BY-SA 4.0 (text) |

## Evaluation only

These are used by `minicoil eval` and are not part of the model.

| Component | Use | License |
|---|---|---|
| [mMARCO](https://huggingface.co/datasets/unicamp-dl/mmarco) | Benchmark; `data/eval/` stores query ids only | Apache-2.0 for the translations; derived from [MS MARCO](https://microsoft.github.io/msmarco/), whose data is for non-commercial research only |
| MLQA | Earlier benchmark | CC BY-SA 3.0 |
| [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) | `translate-bm25` baseline | CC BY-NC 4.0 |
| [`Qdrant/minicoil-v1`](https://huggingface.co/Qdrant/minicoil-v1) | `minicoil-v1` baseline | Apache-2.0 |
