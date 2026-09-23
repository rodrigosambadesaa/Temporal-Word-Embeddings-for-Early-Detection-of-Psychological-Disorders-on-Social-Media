import numpy as np

from framework3 import BaseMetric, Container, XYData

from src.metrics.erde_core import erde_score

__all__ = ["ERDE_5", "ERDE_50"]


class ERDE(BaseMetric):
    """Early Risk Detection Error evaluated at user level."""

    def __init__(self, k: int = 5):
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> float | np.ndarray:
        if y_true is None:
            raise ValueError("y_true must be provided for evaluation")

        x_df = x_data.value.copy()
        true_values = list(y_true.value)
        predicted_values = list(y_pred.value)

        if len(x_df) != len(true_values) or len(x_df) != len(predicted_values):
            raise ValueError(
                "ERDE expects one label and one prediction per input chunk "
                f"(rows={len(x_df)}, labels={len(true_values)}, predictions={len(predicted_values)})"
            )

        x_df = x_df.assign(label=true_values, prediction=predicted_values)
        if "chunk" in x_df.columns:
            x_df = x_df.sort_values(["user", "chunk"], kind="stable")

        return erde_score(
            x_df["user"].tolist(),
            x_df["label"].tolist(),
            x_df["prediction"].tolist(),
            x_df["n_texts"].tolist(),
            k=self.k,
        )


@Container.bind()
class ERDE_5(BaseMetric):
    def __init__(self):
        self._erde = ERDE(k=5)

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> float | np.ndarray:
        return self._erde.evaluate(x_data, y_true, y_pred)


@Container.bind()
class ERDE_50(BaseMetric):
    def __init__(self):
        self._erde = ERDE(k=50)

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData
    ) -> float | np.ndarray:
        return self._erde.evaluate(x_data, y_true, y_pred)
