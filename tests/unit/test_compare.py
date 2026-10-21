from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score

from fraudgraph.data.prepare import to_canonical
from fraudgraph.data.synth import generate
from fraudgraph.models.compare import (
    Ranked,
    block_weights,
    evaluate,
    paired_difference,
    pr_band,
    topk_flags,
)


def random_case(seed: int, n: int = 2_000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(np.int8)
    # Rounded scores create many ties, the tricky case for AP.
    score = np.round(rng.random(n) + y * 0.3, 2)
    return y, score


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ap_matches_sklearn_with_ties(seed: int) -> None:
    y, score = random_case(seed)
    assert Ranked(y, score).metrics()["ap"] == pytest.approx(average_precision_score(y, score))


def test_weighted_ap_matches_sklearn_sample_weight() -> None:
    y, score = random_case(3)
    w = np.random.default_rng(4).integers(0, 4, size=len(y)).astype(float)
    ranked = Ranked(y, score)
    expected = average_precision_score(y, score, sample_weight=w)
    assert ranked.metrics(w[ranked.order])["ap"] == pytest.approx(expected)


def test_recall_at_precision_by_hand() -> None:
    # Ranked: 1, 1, 0, 1, 0  -> precision 1, 1, .67, .75, .6 ; recall .33, .67, .67, 1, 1
    y = np.array([1, 1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    m = Ranked(y, score).metrics()
    assert m["recall_at_p0.9"] == pytest.approx(2 / 3)
    assert m["recall_at_p0.5"] == pytest.approx(1.0)
    assert m["ap"] == pytest.approx((1 + 1 + 0.75) / 3)


def test_topk_flags_per_block() -> None:
    score = np.array([0.1, 0.9, 0.5, 0.8, 0.2, 0.7])
    blocks = np.array([0, 0, 0, 1, 1, 1])
    assert topk_flags(score, blocks, 1).tolist() == [False, True, False, True, False, False]


def test_block_weights_draw_whole_blocks() -> None:
    blocks = np.array([5, 5, 7, 7, 7, 9])
    index, counts = block_weights(blocks, resamples=50, seed=0)
    assert index.tolist() == [0, 0, 1, 1, 1, 2]
    assert (counts.sum(axis=1) == 3).all()  # three blocks drawn per resample


def test_evaluate_is_seeded_and_ci_brackets_point() -> None:
    y, score = random_case(5, n=5_000)
    blocks = np.arange(len(y)) // 100
    a = evaluate(y, score, blocks, resamples=200, seed=0)
    b = evaluate(y, score, blocks, resamples=200, seed=0)

    assert a == b
    lo, hi = a.ci["ap"]
    assert lo <= a.point["ap"] <= hi


def test_paired_difference_of_identical_models_is_zero() -> None:
    y, score = random_case(6)
    blocks = np.arange(len(y)) // 100
    d = paired_difference(y, score, score.copy(), blocks, resamples=100)
    assert d["delta_ap"] == 0.0
    assert d["ci95"] == [0.0, 0.0]


def test_pr_band_is_ordered() -> None:
    y, score = random_case(7, n=5_000)
    band = pr_band(y, score, np.arange(len(y)) // 100, resamples=50)
    assert np.all(band["low"] <= band["high"] + 1e-12)


# Golden numbers: a fixed score on the synthetic fixture. Any change to the metric code
# that moves these is a real behaviour change and must be deliberate.
GOLDEN = {
    "ap": 0.0313023352918724,
    "recall_at_p0.5": 0.0,
    "precision_at_k": 0.1282051282051282,
    "ap_ci95": (0.022793868411714244, 0.0420663274206825),
}


def test_golden_numbers_on_fixture() -> None:
    frame = to_canonical(generate().frame).filter(pl.col("type").is_in(["TRANSFER", "CASH_OUT"]))
    y = frame["is_fraud"].cast(pl.Int8).to_numpy()
    score = np.log1p(frame["amount"].to_numpy()) + 2.0 * (frame["type"] == "TRANSFER").to_numpy()
    blocks = (frame["step"].to_numpy() - 1) // 6
    result = evaluate(
        y, score, blocks, amount=frame["amount"].to_numpy(), alerts_per_block=5, resamples=200
    )

    assert result.point["ap"] == pytest.approx(GOLDEN["ap"], rel=1e-9)
    assert result.point["recall_at_p0.5"] == pytest.approx(GOLDEN["recall_at_p0.5"], rel=1e-9)
    assert result.point["precision_at_k"] == pytest.approx(GOLDEN["precision_at_k"], rel=1e-9)
    assert result.ci["ap"] == pytest.approx(GOLDEN["ap_ci95"], rel=1e-9)
