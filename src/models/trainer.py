"""Trains every model configured in config.model.models_to_train, on a training split only."""

import logging
from typing import Callable

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config.config_loader import ConfigLoader

from .baseline import build_baseline_model

logger = logging.getLogger(__name__)


class ModelTrainingError(Exception):
    """Raised when config.model.models_to_train names a model this trainer doesn't know how to build."""


def _build_random_forest(config: ConfigLoader, random_seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=config.get("model.random_forest.n_estimators", 200),
        max_depth=config.get("model.random_forest.max_depth", 10),
        random_state=random_seed,
    )


def _build_xgboost(config: ConfigLoader, random_seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=config.get("model.xgboost.n_estimators", 200),
        learning_rate=config.get("model.xgboost.learning_rate", 0.05),
        max_depth=config.get("model.xgboost.max_depth", 6),
        random_state=random_seed,
    )


# Which config keys feed which constructor is decided here, not inside each
# model's own hyperparameters -- adding a new model to config.yaml's
# models_to_train only needs one new entry in this dict.
_MODEL_FACTORIES: dict[str, Callable[[ConfigLoader, int], BaseEstimator]] = {
    "linear_regression": lambda config, random_seed: build_baseline_model(),
    "random_forest": _build_random_forest,
    "xgboost": _build_xgboost,
}


class ModelTrainer:
    """Builds and fits every model named in config.model.models_to_train, using only the training split.

    Hyperparameters (model.random_forest.*, model.xgboost.*) and
    project.random_seed are read from config, never hardcoded here --
    _MODEL_FACTORIES only decides *which* config keys feed which
    constructor argument.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self._config_path = config_path
        self._config = ConfigLoader(config_path)
        self._random_seed = self._config.get("project.random_seed", 42)
        self._models_to_train = self._config.get(
            "model.models_to_train", ["linear_regression", "random_forest", "xgboost"]
        )

    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, BaseEstimator]:
        """Fit every configured model on (X_train, y_train) only.

        There is no X_test/y_test parameter: the test split cannot leak into
        training here because this method is never given it.

        Raises:
            ModelTrainingError: if config.model.models_to_train names a
                model with no registered factory.
        """
        fitted_models: dict[str, BaseEstimator] = {}
        for model_name in self._models_to_train:
            factory = _MODEL_FACTORIES.get(model_name)
            if factory is None:
                raise ModelTrainingError(
                    f"ModelTrainer: unknown model '{model_name}' in config.model.models_to_train "
                    f"(known models: {sorted(_MODEL_FACTORIES)})"
                )
            model = factory(self._config, self._random_seed)
            model.fit(X_train, y_train)
            fitted_models[model_name] = model
            logger.info(
                "ModelTrainer: fitted '%s' on %d row(s), %d feature(s)",
                model_name,
                len(X_train),
                X_train.shape[1],
            )
        return fitted_models
