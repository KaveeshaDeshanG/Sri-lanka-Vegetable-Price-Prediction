"""Evaluation metrics for trained models, computed on the held-out chronological test set."""

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Raised when config.evaluation.metrics names a metric this evaluator doesn't know how to compute."""


# mean_squared_error(..., squared=False) is deprecated across sklearn
# versions in favor of a separate root_mean_squared_error function; squaring
# it manually here avoids depending on which one a given sklearn version has.
_METRIC_FUNCTIONS: dict[str, Callable[[pd.Series, np.ndarray], float]] = {
    "mae": mean_absolute_error,
    "rmse": lambda y_true, y_pred: mean_squared_error(y_true, y_pred) ** 0.5,
    "r2": r2_score,
}


class ModelEvaluator:
    """Computes config's evaluation.metrics for one or more fitted models against a held-out test set."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self._config_path = config_path
        config = ConfigLoader(config_path)
        self._metrics = config.get("evaluation.metrics", ["mae", "rmse", "r2"])

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Predict on X_test and compute each configured metric against y_test.

        Raises:
            EvaluationError: if config.evaluation.metrics names an unknown metric.
        """
        y_pred = model.predict(X_test)
        results: dict[str, float] = {}
        for metric_name in self._metrics:
            metric_function = _METRIC_FUNCTIONS.get(metric_name)
            if metric_function is None:
                raise EvaluationError(
                    f"ModelEvaluator: unknown metric '{metric_name}' in config.evaluation.metrics "
                    f"(known metrics: {sorted(_METRIC_FUNCTIONS)})"
                )
            results[metric_name] = float(metric_function(y_test, y_pred))
        logger.info("ModelEvaluator: evaluated model on %d row(s): %s", len(X_test), results)
        return results

    def evaluate_all(
        self, models: dict[str, object], X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, dict[str, float]]:
        """Evaluate every model in models against the same held-out test set."""
        return {name: self.evaluate(model, X_test, y_test) for name, model in models.items()}
