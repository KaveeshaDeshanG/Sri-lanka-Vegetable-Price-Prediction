"""Top-level orchestrator for the modelling stage: train -> evaluate -> select -> persist."""

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator

from .evaluator import ModelEvaluator
from .model_selection import select_best_model
from .persistence import save_model
from .trainer import ModelTrainer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelTrainingResult:
    """Everything produced by one run of ModelTrainingPipeline.run()."""

    models: dict[str, BaseEstimator]
    metrics: dict[str, dict[str, float]]
    best_model_name: str
    best_model: BaseEstimator
    saved_model_path: str


class ModelTrainingPipeline:
    """Runs the full modelling stage: train every configured model, evaluate on the held-out
    chronological test set, select the best by config's evaluation.best_model_metric, and persist it.

    Consumes X_train/X_test/y_train/y_test exactly as produced by
    feature_engineering.dataset_builder.TrainTestDatasetBuilder -- this
    class does no feature engineering, encoding, or splitting of its own,
    and never fits anything on X_test/y_test.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self._config_path = config_path
        self._trainer = ModelTrainer(config_path)
        self._evaluator = ModelEvaluator(config_path)

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        best_model_filename: str = "best_model.joblib",
        output_dir: str | None = None,
    ) -> ModelTrainingResult:
        """Train every configured model on (X_train, y_train), evaluate on (X_test, y_test).

        Only the selected best model is persisted, as best_model_filename
        under output_dir (or config's paths.models_dir when output_dir is
        omitted) -- the other trained models are still returned in-memory
        (for comparison/logging) but not written to disk. output_dir lets a
        caller redirect persistence away from the production models
        directory entirely (see app/services.py::train_demo_model()).
        """
        models = self._trainer.train_all(X_train, y_train)
        metrics = self._evaluator.evaluate_all(models, X_test, y_test)
        best_model_name = select_best_model(metrics, config_path=self._config_path)
        best_model = models[best_model_name]

        saved_path = save_model(
            best_model, filename=best_model_filename, output_dir=output_dir, config_path=self._config_path
        )
        logger.info(
            "ModelTrainingPipeline: best model is '%s', saved to '%s'", best_model_name, saved_path
        )
        return ModelTrainingResult(
            models=models,
            metrics=metrics,
            best_model_name=best_model_name,
            best_model=best_model,
            saved_model_path=saved_path,
        )
