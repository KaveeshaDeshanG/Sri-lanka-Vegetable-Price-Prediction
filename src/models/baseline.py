"""Baseline model: plain linear regression, the simplest reproducible reference point."""

import logging

import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def build_baseline_model() -> LinearRegression:
    """Construct the baseline model: ordinary least squares linear regression.

    No hyperparameters and no random_state: the closed-form OLS solution
    sklearn's LinearRegression computes is deterministic given the same
    input, unlike RandomForestRegressor/XGBRegressor -- there is no
    randomness here to seed. config.yaml lists "linear_regression" first in
    model.models_to_train precisely because it needs no tuning to serve as
    the reference point every other configured model is compared against.
    """
    return LinearRegression()


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit the baseline model on the training split only."""
    model = build_baseline_model()
    model.fit(X_train, y_train)
    logger.info(
        "Baseline: fitted LinearRegression on %d row(s), %d feature(s)", len(X_train), X_train.shape[1]
    )
    return model
