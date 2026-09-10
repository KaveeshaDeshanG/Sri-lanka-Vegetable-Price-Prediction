"""Selects the best-performing trained model according to config's evaluation.best_model_metric."""

import logging

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

_LOWER_IS_BETTER = {"mae", "rmse"}
_HIGHER_IS_BETTER = {"r2"}


class ModelSelectionError(Exception):
    """Raised when the best model cannot be determined from the given metrics."""


def select_best_model(
    metrics_by_model: dict[str, dict[str, float]],
    config_path: str = "config/config.yaml",
) -> str:
    """Return the name of the best-performing model in metrics_by_model.

    "Best" is defined by config's evaluation.best_model_metric: lower is
    better for mae/rmse (error metrics), higher is better for r2 (a fit
    quality score). The comparison direction is looked up per metric name,
    not assumed to always be "lower is better" just because that happens to
    be true for config's current default of rmse.

    Raises:
        ModelSelectionError: if metrics_by_model is empty, the configured
            metric is missing from any model's results, or the metric's
            comparison direction (lower/higher-is-better) is unknown.
    """
    if not metrics_by_model:
        raise ModelSelectionError("select_best_model: no models to select from (metrics_by_model is empty)")

    config = ConfigLoader(config_path)
    metric = config.get("evaluation.best_model_metric", "rmse")

    for model_name, metrics in metrics_by_model.items():
        if metric not in metrics:
            raise ModelSelectionError(
                f"select_best_model: model '{model_name}' has no '{metric}' metric "
                f"(available: {sorted(metrics)})"
            )

    if metric in _LOWER_IS_BETTER:
        best_model_name = min(metrics_by_model, key=lambda name: metrics_by_model[name][metric])
    elif metric in _HIGHER_IS_BETTER:
        best_model_name = max(metrics_by_model, key=lambda name: metrics_by_model[name][metric])
    else:
        raise ModelSelectionError(
            f"select_best_model: unknown comparison direction for metric '{metric}' "
            f"(known metrics: lower-is-better={sorted(_LOWER_IS_BETTER)}, "
            f"higher-is-better={sorted(_HIGHER_IS_BETTER)})"
        )

    logger.info(
        "select_best_model: '%s' selected as best model by %s=%s",
        best_model_name,
        metric,
        metrics_by_model[best_model_name][metric],
    )
    return best_model_name
