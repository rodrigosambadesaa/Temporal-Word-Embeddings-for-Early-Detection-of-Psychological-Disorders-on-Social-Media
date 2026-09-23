import math

import pytest

from src.metrics.erde_core import erde_score


def test_erde_true_positive_latency_and_false_positive_cost():
    score = erde_score(
        user_ids=["positive", "positive", "negative"],
        labels=[1, 1, 0],
        predictions=[0, 1, 1],
        n_texts=[2, 3, 4],
        k=5,
    )
    # Positive user is detected after 5 texts -> latency cost 0.5.
    # Negative user is a false positive -> prevalence cost 1/2.
    assert score == pytest.approx(50.0)


def test_erde_false_negative_and_true_negative():
    score = erde_score(
        user_ids=["positive", "negative"],
        labels=[1, 0],
        predictions=[0, 0],
        n_texts=[10, 10],
        k=5,
    )
    assert score == pytest.approx(50.0)


def test_erde_rejects_inconsistent_user_labels():
    with pytest.raises(ValueError, match="Inconsistent labels"):
        erde_score(
            user_ids=["same", "same"],
            labels=[0, 1],
            predictions=[0, 0],
            n_texts=[1, 1],
            k=5,
        )


def test_erde_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="equal lengths"):
        erde_score(
            user_ids=["u"],
            labels=[1],
            predictions=[],
            n_texts=[1],
            k=5,
        )


def test_erde_penalizes_later_detection_more():
    early = erde_score(["u"], [1], [1], [1], k=5)
    late = erde_score(["u"], [1], [1], [20], k=5)
    assert math.isfinite(early)
    assert late > early
