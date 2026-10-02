"""Integration smoke test for the ``run_training`` batched=True wiring.

The a/b/c suites call ``train_concept_bucket``/``prepare_concept`` directly. This
test exercises the only otherwise-uncovered surface: the ``batched`` branch inside
``run_training`` (spec collection, post-loop bucket call, checkpoint save + merge,
skip/monolingual bookkeeping). It bypasses Qdrant and the encoder by monkeypatching
the two I/O boundaries and feeding a cache-derived ``batch_data``; everything from
the per-concept loop onward is the real code path.

``trim_augment_ratio=0`` so no encoder is needed (augmentation is the one path
covered only by code reuse, see the report).

Run: ``uv run python scripts/perf/validate_wiring.py``
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(__file__))
from perf_common import load_cache  # noqa: E402

import minicoil_v2.train_concept_layers as T  # noqa: E402
from minicoil_v2.constants import INPUT_DIM, OUTPUT_DIM  # noqa: E402
from minicoil_v2.settings import TrainSettings  # noqa: E402


def main() -> int:
    d = load_cache()
    cids = sorted(d["concept_indices"], key=lambda k: len(d["concept_indices"][k]))[:4]

    # batch_data restricted to the 4 target concepts (shared embedding banks kept
    # whole; concept maps restricted). This is exactly the dict _prepare_batch_inputs
    # would return for these concepts.
    batch_data = {
        "input_embs": d["input_embs"],
        "mining_embs_all": d["mining_embs_all"],
        "unique_sentences": d["unique_sentences"],
        "unique_langs": d["unique_langs"],
        "concept_indices": {c: d["concept_indices"][c] for c in cids},
        "concept_sent_langs": {c: d["concept_sent_langs"][c] for c in cids},
    }

    # Monkeypatch the I/O boundaries: Qdrant client, encoder load, and the
    # scroll+encode batch prep. Everything after is the real run_training code.
    T.get_client = lambda *a, **k: object()
    T.load_token_pool_model = lambda *a, **k: (None, None)
    T._load_or_prepare_batch = lambda *a, **k: (batch_data, 0)

    with tempfile.TemporaryDirectory() as tmp:
        st = TrainSettings(
            batched=True,
            optimizer_mode="vectorized",
            concepts=",".join(cids),
            concept_batch_size=10,  # one bucket
            epochs=2,
            train_epoch_size=256,
            val_epoch_size=256,
            sample_batch_size=256,
            dropout=0.05,
            trim_augment_ratio=0.0,
            margin_scale=0.0,
            device="cpu",
            wandb_enabled=False,
            output_dir=tmp,
        )
        T.run_training(st)

        merged_path = os.path.join(tmp, "concept_layers.pt")
        ok = os.path.exists(merged_path)
        errs = []
        if not ok:
            errs.append("no merged concept_layers.pt produced")
        else:
            merged = torch.load(merged_path, map_location="cpu", weights_only=True)
            missing = [c for c in cids if c not in merged]
            if missing:
                errs.append(f"missing concepts in merged model: {missing}")
            for c in cids:
                w = merged[c]["weight"]
                if tuple(w.shape) != (OUTPUT_DIM, INPUT_DIM):
                    errs.append(f"{c} weight shape {tuple(w.shape)} != {(OUTPUT_DIM, INPUT_DIM)}")
            print(
                f"  merged model: {len(merged)} concepts, "
                f"weight shape {tuple(merged[cids[0]]['weight'].shape)}"
            )

    ok = not errs
    print("  " + ("OK" if ok else str(errs)))
    print("\n" + ("WIRING SMOKE TEST PASSED" if ok else "WIRING SMOKE TEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
