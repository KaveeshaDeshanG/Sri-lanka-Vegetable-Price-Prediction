"""Forecasts a price up to 30 days ahead with a separate model for each number of days.

Predicting one day at a time and feeding each result back in makes small errors compound. Here,
each horizon gets its own model, trained directly on "the change from today's price to the price
that many days later", so a 14-day forecast uses only real data known today. Forecasts also don't
depend on future weather.

FORECAST_BANDS was chosen on validation windows before the test period, then scored once on it.
Change it only by repeating that process (scripts/model_experiments.py), or test results will no
longer be an honest estimate.
"""

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering.forecast_features import (
    FEATURE_GROUPS,
    add_horizon_features,
    add_vegetable_columns,
)
from src.models.persistence import load_model, save_model

logger = logging.getLogger(__name__)

FORECASTER_FILENAME = "direct_forecaster.joblib"
MAX_HORIZON_DAYS = 30
BOOTSTRAP_SAMPLES = 2000
RANDOM_SEED = 42


@dataclass(frozen=True)
class ForecastBand:
    first_day: int
    last_day: int
    ridge_alpha: float
    feature_groups: tuple[str, ...]


FORECAST_BANDS = (
    ForecastBand(1, 1, 1000.0, ("momentum", "level", "market", "fuel", "weather_lag_14", "other_markets")),
    ForecastBand(2, 10, 1000.0, ("momentum", "rain_windows", "weather_lag_14", "level", "other_markets")),
    ForecastBand(11, 21, 1000.0, ("rain_windows", "market", "level", "other_markets")),
    ForecastBand(22, MAX_HORIZON_DAYS, 100.0, ("momentum", "calendar", "rain_windows", "market", "level")),
)


def band_for(horizon: int) -> ForecastBand:
    return next(band for band in FORECAST_BANDS if band.first_day <= horizon <= band.last_day)


def score_predictions(actual, predicted) -> dict[str, float]:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    error = actual - predicted
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "mape": float((np.abs(error) / actual).mean() * 100),
        "r2": float(1 - (error**2).sum() / ((actual - actual.mean()) ** 2).sum()),
    }


def mae_gain_vs_holding_price(frame: pd.DataFrame, prediction_column: str = "prediction") -> dict[str, float]:
    """MAE reduction vs holding the last price (%), with a 95% interval from resampling dates.

    Resampling whole dates, not rows, because vegetables on one date tend to move together.
    """
    daily = (
        frame.assign(
            model_error=(frame["target_price"] - frame[prediction_column]).abs(),
            hold_error=(frame["target_price"] - frame["last_price"]).abs(),
        )
        .groupby("target_date")[["model_error", "hold_error"]]
        .sum()
        .to_numpy()
    )
    picks = np.random.default_rng(RANDOM_SEED).integers(0, len(daily), size=(BOOTSTRAP_SAMPLES, len(daily)))
    totals = daily[picks].sum(axis=1)
    gains = (1 - totals[:, 0] / totals[:, 1]) * 100
    return {
        "gain_pct": float((1 - daily[:, 0].sum() / daily[:, 1].sum()) * 100),
        "gain_low": float(np.percentile(gains, 2.5)),
        "gain_high": float(np.percentile(gains, 97.5)),
    }


@dataclass
class DirectForecaster:
    """A fitted Ridge model per horizon, with the vegetables and columns it was trained on."""

    vegetables: list[str]
    models: dict[int, tuple[object, list[str]]] = field(default_factory=dict)
    trained_through: pd.Timestamp | None = None

    @classmethod
    def fit(
        cls,
        origin_rows: pd.DataFrame,
        horizons: range = range(1, MAX_HORIZON_DAYS + 1),
        target_before: pd.Timestamp | None = None,
    ) -> "DirectForecaster":
        """Fit each horizon's model. target_before excludes rows whose price falls on or after it."""
        vegetables = sorted(origin_rows["vegetable"].unique())
        forecaster = cls(vegetables=vegetables)
        for horizon in horizons:
            rows = usable_rows(add_vegetable_columns(add_horizon_features(origin_rows, horizon), vegetables))
            if target_before is not None:
                rows = rows[rows["target_date"] < target_before]
            band = band_for(horizon)
            columns = _feature_columns(band, vegetables)
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=band.ridge_alpha))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # a feature can be entirely empty in early history
                model.fit(rows[columns], np.log(rows["target_price"] / rows["last_price"]))
            forecaster.models[horizon] = (model, columns)
            forecaster.trained_through = rows["target_date"].max()
        return forecaster

    def predict_rows(self, rows: pd.DataFrame, horizon: int) -> np.ndarray:
        """Predicted price in rupees for rows that already have add_vegetable_columns()."""
        model, columns = self.models[horizon]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            change = model.predict(rows[columns])
        return rows["last_price"].to_numpy() * np.exp(change)

    def forecast(self, origin_rows: pd.DataFrame, vegetable: str, origin: pd.Timestamp, horizons: list[int]) -> pd.DataFrame:
        """A forecast from one origin date: columns horizon, target_date, predicted_price, last_price.

        Raises:
            ValueError: if this vegetable or origin date has no usable current price.
        """
        selected = origin_rows[origin_rows["vegetable"] == vegetable]
        if vegetable not in self.vegetables or selected.empty:
            raise ValueError(f"The forecaster wasn't trained on '{vegetable}'.")
        results = []
        for horizon in horizons:
            rows = add_vegetable_columns(add_horizon_features(selected, horizon), self.vegetables)
            row = rows[rows["origin"] == origin]
            if row.empty or row["last_price"].isna().all():
                raise ValueError(f"'{vegetable}' has no usable price on or shortly before {origin.date()}.")
            results.append(
                {
                    "horizon": horizon,
                    "target_date": row["target_date"].iloc[0],
                    "predicted_price": float(self.predict_rows(row, horizon)[0]),
                    "last_price": float(row["last_price"].iloc[0]),
                }
            )
        return pd.DataFrame(results)

    def save(self, config_path: str = "config/config.yaml") -> str:
        return save_model(self, filename=FORECASTER_FILENAME, config_path=config_path)

    @classmethod
    def load(cls, config_path: str = "config/config.yaml") -> "DirectForecaster":
        return load_model(filename=FORECASTER_FILENAME, config_path=config_path)


def usable_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows with both a known current price and a real price on the target date."""
    return rows[rows["target_price"].notna() & rows["last_price"].notna()]


def _feature_columns(band: ForecastBand, vegetables: list[str]) -> list[str]:
    columns = [column for group in band.feature_groups for column in FEATURE_GROUPS[group]]
    return columns + [f"veg_{vegetable}" for vegetable in vegetables]
