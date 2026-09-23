"""Pure implementation of the Early Risk Detection Error (ERDE) metric.

This module intentionally has no dependency on the project pipeline framework so the
metric can be unit-tested independently.
"""

from __future__ import annotations

from collections import OrderedDict
from math import exp
from typing import Hashable, Iterable


def _latency_cost(delay: int, k: int) -> float:
    """Return the ERDE latency penalty for a true-positive decision."""
    return 1.0 - (1.0 / (1.0 + exp(delay - k)))


def erde_score(
    user_ids: Iterable[Hashable],
    labels: Iterable[int],
    predictions: Iterable[int],
    n_texts: Iterable[int],
    *,
    k: int,
) -> float:
    """Compute ERDE as a percentage from chronologically ordered chunks.

    Each input item represents one chunk for one user. Chunks belonging to the same
    user must be supplied in chronological order.

    False-positive cost follows the eRisk definition and is the prevalence of
    positive users in the evaluated population. False negatives cost 1, true
    negatives cost 0, and true positives receive the latency penalty.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    users = list(user_ids)
    y_true = [int(value) for value in labels]
    y_pred = [int(value) for value in predictions]
    counts = [int(value) for value in n_texts]

    lengths = {len(users), len(y_true), len(y_pred), len(counts)}
    if len(lengths) != 1:
        raise ValueError("user_ids, labels, predictions and n_texts must have equal lengths")
    if any(value not in (0, 1) for value in y_true + y_pred):
        raise ValueError("labels and predictions must be binary (0 or 1)")
    if any(value < 0 for value in counts):
        raise ValueError("n_texts values must be non-negative")
    if not users:
        raise ValueError("ERDE cannot be computed on an empty input")

    grouped: "OrderedDict[Hashable, list[tuple[int, int, int]]]" = OrderedDict()
    for user, label, prediction, count in zip(users, y_true, y_pred, counts):
        grouped.setdefault(user, []).append((label, prediction, count))

    user_labels: dict[Hashable, int] = {}
    for user, chunks in grouped.items():
        labels_for_user = {label for label, _, _ in chunks}
        if len(labels_for_user) != 1:
            raise ValueError(f"Inconsistent labels for user {user!r}")
        user_labels[user] = chunks[0][0]

    positive_users = sum(user_labels.values())
    false_positive_cost = positive_users / len(grouped)

    costs: list[float] = []
    for user, chunks in grouped.items():
        expected = user_labels[user]
        cumulative_texts = 0
        first_positive_delay: int | None = None

        for _, prediction, count in chunks:
            cumulative_texts += count
            if prediction == 1 and first_positive_delay is None:
                first_positive_delay = cumulative_texts

        predicted_positive = first_positive_delay is not None
        if predicted_positive and expected == 0:
            costs.append(false_positive_cost)
        elif not predicted_positive and expected == 1:
            costs.append(1.0)
        elif predicted_positive and expected == 1:
            costs.append(_latency_cost(first_positive_delay, k))
        else:
            costs.append(0.0)

    return 100.0 * sum(costs) / len(costs)
