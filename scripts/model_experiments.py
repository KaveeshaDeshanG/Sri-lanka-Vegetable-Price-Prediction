"""Compares modelling ideas on validation windows only, the way FORECAST_BANDS was chosen.

Rows whose target falls on or after forecasting.test_start are removed before anything runs, so
this script can't see the test period. Its results choose a configuration; scripts/train_forecaster.py
then scores the chosen one on the test period.

Usage:
    .venv312/Scripts/python.exe scripts/model_experiments.py
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.services import load_real_market_history  # noqa: E402
from config.config_loader import ConfigLoader  # noqa: E402
from src.feature_engineering.forecast_features import (  # noqa: E402
    FEATURE_GROUPS,
    ForecastFeatureBuilder,
    add_horizon_features,
    add_vegetable_columns,
)
from src.models.direct_forecaster import (  # noqa: E402
    RANDOM_SEED,
    mae_gain_vs_holding_price,
    score_predictions,
    usable_rows,
)

VALIDATION_WINDOW_MONTHS = 3
VALIDATION_WINDOWS = 4
HORIZONS = [1, 7, 14, 30]
MODELS = {
    "linear": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()),
    "ridge_100": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100)),
    "ridge_1000": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1000)),
    "random_forest": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=30, max_features=0.4, n_jobs=-1, random_state=RANDOM_SEED),
    ),
    "xgboost": lambda: XGBRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, reg_lambda=5.0, random_state=RANDOM_SEED,
    ),
}


def validation_windows(test_start: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Consecutive windows ending the day before the test period, oldest first."""
    ends = [test_start - pd.DateOffset(months=VALIDATION_WINDOW_MONTHS * n) for n in range(VALIDATION_WINDOWS, -1, -1)]
    return [(start, end - pd.Timedelta(days=1)) for start, end in zip(ends, ends[1:])]


def evaluate(rows: pd.DataFrame, kind: str, groups: list[str], windows: list) -> dict:
    """Each window is predicted by a model trained only on earlier targets."""
    columns = [c for group in groups for c in FEATURE_GROUPS[group]] + [c for c in rows.columns if c.startswith("veg_")]
    parts = []
    for start, end in windows:
        train = rows[rows["target_date"] < start]
        fold = rows[rows["target_date"].between(start, end)].copy()
        model = MODELS[kind]()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[columns], np.log(train["target_price"] / train["last_price"]))
            fold["prediction"] = fold["last_price"].to_numpy() * np.exp(model.predict(fold[columns]))
        fold["window"] = str(start.date())
        parts.append(fold)
    frame = pd.concat(parts, ignore_index=True)
    by_window = frame.groupby("window")[["target_price", "prediction", "last_price"]].apply(
        lambda g: (1 - (g["target_price"] - g["prediction"]).abs().mean() / (g["target_price"] - g["last_price"]).abs().mean()) * 100
    )
    return {
        **score_predictions(frame["target_price"], frame["prediction"]),
        **mae_gain_vs_holding_price(frame),
        "windows_won": int((by_window > 0).sum()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level="ERROR")
    logging.disable(logging.WARNING)

    test_start = pd.Timestamp(ConfigLoader(args.config).get_required("forecasting.test_start"))
    history = load_real_market_history(args.config)
    pre_test = history[history["date"] < test_start]
    origin_rows = ForecastFeatureBuilder(args.config).build_origin_features(pre_test)
    vegetables = sorted(origin_rows["vegetable"].unique())
    windows = validation_windows(test_start)
    groups = list(FEATURE_GROUPS)

    results = {}
    for horizon in HORIZONS:
        rows = usable_rows(add_vegetable_columns(add_horizon_features(origin_rows, horizon), vegetables))
        rows = rows[rows["target_date"] < test_start]
        print(f"\n===== {horizon} day(s) ahead: {len(rows)} rows =====", flush=True)
        models = [{"model": kind, **evaluate(rows, kind, groups, windows)} for kind in MODELS]
        table = pd.DataFrame(models).sort_values("mae")
        print(table.round(2).to_string(index=False), flush=True)

        best = table.iloc[0]["model"]
        full = table.iloc[0]["gain_pct"]
        contributions = {
            group: float(full - evaluate(rows, best, [g for g in groups if g != group], windows)["gain_pct"]) for group in groups
        }
        print(f"Gain lost by removing each group from {best}:", {k: round(v, 2) for k, v in contributions.items()}, flush=True)
        results[str(horizon)] = {"models": models, "best_model": best, "group_contributions": contributions}

    output_path = Path("reports/metrics/model_experiments_validation.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
