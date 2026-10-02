"""Regression tests against the published checkpoint (``constants.PUBLISHED_MODEL_ID``).

These replace the old golden/integration tests, which pinned a local 4D checkpoint
that was never part of the repository. The model is downloaded from the Hugging
Face Hub (or read from ``MINICOIL_V2_TEST_MODEL``, a local export directory), so the
tests are marked slow.

The golden sentence has three surface forms of one concept (food/foods/feed ->
C-00122), so the all-occurrence pooling path is exercised, not just a single match.
Goldens were recorded with the prefix-leak fix in place (concepts are matched on
the text only, never on the mE5 "passage: "/"query: " prefix).
"""

import math
import os

import pytest

from minicoil_v2.constants import OUTPUT_DIM, PUBLISHED_MODEL_ID
from minicoil_v2.encoder import BACKBONE_BASE, MiniCoilEncoder, backbone_index

SENTENCE = "We understand the food and the foods we feed people"

# {concept block start index: 8 values}; blocks are C-00063 (people), C-00122
# (food), C-00126 (understand) and C-03141 (we).
EXPECTED_DOC = {
    504: [0.068478, -0.038043, 0.312127, -0.910792, -0.078598, -0.492855, -0.885853, 0.906134],
    976: [-0.425343, 0.689732, 0.525581, -0.243206, 0.982115, 0.061822, -1.378615, 0.311033],
    1008: [0.865369, 0.837117, 0.255403, 0.11062, 0.25717, -0.694858, -0.441685, 0.718925],
    25128: [-0.288041, 0.030678, -0.579438, -0.236306, 0.671098, 1.363927, -0.256148, 0.867555],
}
EXPECTED_QUERY = {
    504: [0.111289, -0.224431, 0.081331, -0.542494, -0.024989, -0.258583, -0.591209, 0.468309],
    976: [-0.230749, 0.299897, 0.369774, -0.043849, 0.485414, -0.052776, -0.677472, 0.144154],
    1008: [0.553426, 0.456639, 0.217036, 0.124021, 0.220223, -0.392585, -0.287288, 0.37089],
    25128: [-0.166357, -0.007042, -0.305226, -0.060027, 0.437255, 0.718689, -0.14437, 0.383355],
}


@pytest.fixture(scope="module")
def encoder():
    source = os.environ.get("MINICOIL_V2_TEST_MODEL", PUBLISHED_MODEL_ID)
    return MiniCoilEncoder.from_pretrained(source, device="cpu")


def _blocks(sparse: dict[int, float]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for idx, value in sparse.items():
        if idx >= BACKBONE_BASE:
            continue
        start = idx - idx % OUTPUT_DIM
        out.setdefault(start, [0.0] * OUTPUT_DIM)[idx % OUTPUT_DIM] = value
    return out


@pytest.mark.slow
@pytest.mark.parametrize(("is_query", "expected"), [(False, EXPECTED_DOC), (True, EXPECTED_QUERY)])
def test_published_golden(encoder, is_query, expected):
    out = encoder.encode_sparse(SENTENCE, lang="en", is_query=is_query)
    blocks = _blocks(out)
    assert sorted(blocks) == sorted(expected)
    for start, values in expected.items():
        assert blocks[start] == pytest.approx(values, abs=1e-4)
    # every token of the sentence is either a stopword or a concept word
    assert not [i for i in out if i >= BACKBONE_BASE]


@pytest.mark.slow
def test_query_blocks_are_unit_norm(encoder):
    # query side is flat: each concept block is a unit sense direction
    for values in _blocks(encoder.encode_sparse(SENTENCE, lang="en", is_query=True)).values():
        assert math.sqrt(sum(v * v for v in values)) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.slow
def test_oov_tokens_become_backbone_terms(encoder):
    [out] = encoder.encode_batch_sparse(["reykjavik xqzzyflarn"])
    # neither token is a concept word, so both must appear as BM25 backbone terms
    assert backbone_index("reykjavik") in out
    assert backbone_index("xqzzyflarn") in out
    [no_backbone] = encoder.encode_batch_sparse(["reykjavik xqzzyflarn"], include_backbone=False)
    assert backbone_index("reykjavik") not in no_backbone
