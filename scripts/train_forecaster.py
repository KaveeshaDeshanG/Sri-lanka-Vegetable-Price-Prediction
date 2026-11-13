"""Evaluates the direct forecaster on the held-out test period, then trains it on all data.

Writes models/direct_forecaster.joblib and reports/metrics/direct_forecaster_metrics.json. Test
scores come from models trained only on targets before forecasting.test_start. The saved model
then learns from everything, so it's current, and isn't separately scored.

Usage:
    .venv312/Scripts/python.exe scripts/train_forecaster.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.services import load_real_market_history  # noqa: E402
from config.config_loader import ConfigLoader  # noqa: E402
from scripts.backtest_forecast import DEFAULT_ORIGINS, DEFAULT_VEGETABLES  # noqa: E402
from src.feature_engineering.feature_engineer import FeatureEngineer  # noqa: E402
from src.feature_engineering.forecast_features import (  # noqa: E402
    ForecastFeatureBuilder,
    add_horizon_features,
    add_vegetable_columns,
)
from src.models.direct_forecaster import (  # noqa: E402
    MAX_HORIZON_DAYS,
    DirectForecaster,
    mae_gain_vs_holding_price,
    score_predictions,
    usable_rows,
)
from src.prediction.predictor import Predictor  # noqa: E402

logger = logging.getLogger(__name__)

REPORT_BANDS = [(1, 1), (2, 3), (4, 7), (8, 14), (15, 21), (22, 30)]
COMPARISON_BANDS = [(1, 1), (2, 3), (4, 7), (8, 14), (15, 30)]


def score_on_test_period(origin_rows: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    forecaster = DirectForecaster.fit(origin_rows, target_before=test_start)
    parts = []
    for horizon in forecaster.models:
        rows = usable_rows(add_vegetable_columns(add_horizon_features(origin_rows, horizon), forecaster.vegetables))
        rows = rows[rows["target_date"] >= test_start].copy()
        rows["prediction"] = forecaster.predict_rows(rows, horizon)
        rows["horizon"] = horizon
        parts.append(rows[["vegetable", "origin", "target_date", "target_price", "last_price", "prediction", "horizon"]])
    return pd.concat(parts, ignore_index=True)


def band_label(first: int, last: int) -> str:
    return str(first) if first == last else f"{first}-{last}"


def report_bands(scored: pd.DataFrame) -> list[dict]:
    results = []
    for first, last in REPORT_BANDS:
        band = scored[scored["horizon"].between(first, last)]
        model, hold = score_predictions(band["target_price"], band["prediction"]), score_predictions(band["target_price"], band["last_price"])
        results.append(
            {
                "days_ahead": band_label(first, last),
                "scored_rows": len(band),
                "model_mae": model["mae"],
                "model_rmse": model["rmse"],
                "model_mape": model["mape"],
                "model_r2": model["r2"],
                "hold_mae": hold["mae"],
                "hold_rmse": hold["rmse"],
                "hold_mape": hold["mape"],
                **mae_gain_vs_holding_price(band),
            }
        )
    return results


def report_vegetables(scored: pd.DataFrame) -> list[dict]:
    next_day = scored[scored["horizon"] == 1]
    return [
        {
            "vegetable": vegetable,
            "test_rows": len(group),
            "mae": float((group["target_price"] - group["prediction"]).abs().mean()),
            "mape": float(((group["target_price"] - group["prediction"]).abs() / group["target_price"]).mean() * 100),
            "hold_mae": float((group["target_price"] - group["last_price"]).abs().mean()),
        }
        for vegetable, group in next_day.groupby("vegetable")
    ]


def compare_with_earlier_model(scored: pd.DataFrame, history: pd.DataFrame, config_path: str) -> dict | None:
    """Next-day comparison on the rows the earlier 51-feature model can score, if it exists."""
    try:
        predictor = Predictor.load(config_path=config_path)
    except Exception as error:  # noqa: BLE001 -- the comparison is optional
        logger.warning("Skipping comparison with the earlier model: %s", error)
        return None
    engineer = FeatureEngineer(config_path=config_path)
    featured = engineer.build_features(history)
    engineered = engineer._engineered_column_names(featured)
    start = scored["target_date"].min()
    usable = featured[(featured["date"] >= start) & featured[engineered].notna().all(axis=1)].copy()
    features, _ = engineer.split_features_target(usable, drop_incomplete_rows=False)
    usable["earlier_model"] = predictor.predict(features).to_numpy()
    known = usable[usable["vegetable"].isin(predictor.known_vegetables())]
    joined = scored[scored["horizon"] == 1].merge(
        known[["vegetable", "date", "earlier_model"]], left_on=["vegetable", "target_date"], right_on=["vegetable", "date"]
    )
    return {
        "rows": len(joined),
        "direct_forecaster": score_predictions(joined["target_price"], joined["prediction"]),
        "earlier_model": score_predictions(joined["target_price"], joined["earlier_model"]),
        "hold_last_price": score_predictions(joined["target_price"], joined["last_price"]),
    }


def compare_with_recursive(scored: pd.DataFrame) -> list[dict]:
    """The direct forecaster on the same vegetables and start dates as scripts/backtest_forecast.py."""
    sample = scored[scored["vegetable"].isin(DEFAULT_VEGETABLES) & scored["origin"].isin(pd.to_datetime(DEFAULT_ORIGINS))]
    results = []
    for first, last in COMPARISON_BANDS:
        band = sample[sample["horizon"].between(first, last)]
        results.append(
            {
                "days_ahead": band_label(first, last),
                "scored_rows": len(band),
                "mae": float((band["target_price"] - band["prediction"]).abs().mean()),
                "mape": float(((band["target_price"] - band["prediction"]).abs() / band["target_price"]).mean() * 100),
            }
        )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level="WARNING")

    config = ConfigLoader(args.config)
    test_start = pd.Timestamp(config.get_required("forecasting.test_start"))
    history = load_real_market_history(args.config)
    origin_rows = ForecastFeatureBuilder(args.config).build_origin_features(history)

    scored = score_on_test_period(origin_rows, test_start)
    report = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "test_start": str(test_start.date()),
        "test_end": str(scored["target_date"].max().date()),
        "max_horizon_days": MAX_HORIZON_DAYS,
        "vegetables": int(origin_rows["vegetable"].nunique()),
        "bands": report_bands(scored),
        "next_day_by_vegetable": report_vegetables(scored),
        "next_day_vs_earlier_model": compare_with_earlier_model(scored, history, args.config),
        "same_sample_as_recursive_backtest": compare_with_recursive(scored),
    }
    forecaster = DirectForecaster.fit(origin_rows)
    model_path = forecaster.save(args.config)

    output_path = Path(config.get("paths.metrics_dir", "reports/metrics")) / "direct_forecaster_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(pd.DataFrame(report["bands"]).round(2).to_string(index=False))
    print(f"Saved {model_path} (trained through {forecaster.trained_through.date()}) and {output_path}")


if __name__ == "__main__":
    main()
