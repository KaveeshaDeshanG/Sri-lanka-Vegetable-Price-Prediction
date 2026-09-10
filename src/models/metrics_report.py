"""Persists final model-comparison metrics as a structured JSON report.

Deliberately NOT called from ModelTrainingPipeline.run() itself: that pipeline is
shared with app/services.py's Streamlit demo-training path (synthetic sample data),
and this report is meant to be the durable record of the real production training
run (scripts/train.py). Wiring it in here would let a demo retrain in the app
silently overwrite the real-data metrics report with synthetic numbers.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

DEFAULT_METRICS_FILENAME = "model_metrics.json"


def save_metrics_report(
    metrics: dict[str, dict[str, float]],
    best_model_name: str,
    best_model_metric: str,
    dataset_summary: dict[str, Any] | None = None,
    filename: str = DEFAULT_METRICS_FILENAME,
    config_path: str = "config/config.yaml",
) -> str:
    """Write every trained model's evaluation metrics plus the best-model selection to JSON.

    Written under <config's paths.metrics_dir>/<filename>, so a training run's
    MAE/RMSE/R2 and which model was selected (and why) have a durable, structured
    record instead of existing only in memory/console log output.

    Args:
        metrics: {model_name: {metric_name: value}}, exactly ModelTrainingResult.metrics.
        best_model_name: the model select_best_model() chose.
        best_model_metric: config.evaluation.best_model_metric, the metric used to choose it.
        dataset_summary: optional extra context (row counts, feature count, ...) to
            include verbatim under the report's "dataset" key.
    """
    config = ConfigLoader(config_path)
    resolved_dir = Path(config.get("paths.metrics_dir", "reports/metrics"))
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = resolved_dir / filename

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_summary or {},
        "models": metrics,
        "best_model": {
            "name": best_model_name,
            "selection_metric": best_model_metric,
            "selection_value": metrics[best_model_name][best_model_metric],
        },
    }
    resolved_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Metrics report: saved to '%s'", resolved_path)
    return str(resolved_path)
