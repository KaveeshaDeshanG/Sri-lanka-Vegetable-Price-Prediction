"""Measures real multi-day forecast accuracy, which scripts/train.py's metrics don't cover.

Those metrics score one-day-ahead predictions, where every input is a real observed price.
A forecast is different: from day 2 on, the model's own predictions become its inputs, so
errors build up. This forecasts from past start dates using only what was known then, then
compares each day with the price that was actually recorded.

Actual past weather stands in for the forecast weather the app would use, so these results
slightly favour the model. Reads the saved model; never retrains or saves one. Writes
reports/metrics/forecast_backtest.json for the Model Performance page.

Usage:
    .venv312/Scripts/python.exe scripts/backtest_forecast.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.services import impute_missing_historical_prices, load_real_market_history  # noqa: E402
from config.config_loader import ConfigLoader  # noqa: E402
from src.data_collection.production_region_resolver import ProductionRegionResolver  # noqa: E402
from src.prediction.pipeline import PredictionPipeline  # noqa: E402

logger = logging.getLogger(__name__)

MARKET = "Dambulla"
FORECAST_DAYS = 30
HORIZON_BANDS = [(1, 1), (2, 3), (4, 7), (8, 14), (15, 30)]
DEFAULT_VEGETABLES = ["Beans", "Carrot", "Tomato", "Leeks", "Cabbage (Kandy)", "Pumpkin", "Green Chillies", "Brinjals"]
DEFAULT_ORIGINS = ["2026-05-20", "2026-06-20", "2026-07-20", "2026-08-20", "2026-09-20", "2026-10-05"]


def forecast_from_origin(
    history: pd.DataFrame,
    vegetable: str,
    origin: pd.Timestamp,
    pipeline: PredictionPipeline,
    weather: pd.DataFrame,
) -> list[dict]:
    """One recursive forecast using only rows up to origin, scored on days with a real price."""
    known = history[(history["vegetable"] == vegetable) & (history["date"] <= origin)]
    actuals = history[(history["vegetable"] == vegetable) & (history["date"] > origin)].set_index("date")["price"]
    last_price = float(known.sort_values("date")["price"].iloc[-1])
    diesel, petrol = known["diesel_price"].dropna().iloc[-1], known["petrol_price"].dropna().iloc[-1]
    working, _ = impute_missing_historical_prices(known, vegetable, MARKET)

    results = []
    for horizon in range(1, FORECAST_DAYS + 1):
        day = origin + pd.Timedelta(days=horizon)
        required = day - pd.Timedelta(days=14)
        rows = []
        if working.loc[working["date"] == required, "rainfall_mm"].notna().sum() == 0 and required in weather.index:
            rows.append({"date": required, "vegetable": vegetable, "market": MARKET, "source": "CONTEXT", "price": np.nan,
                         **weather.loc[required].to_dict(), "diesel_price": diesel, "petrol_price": petrol})
        rows.append({"date": day, "vegetable": vegetable, "market": MARKET, "source": "TARGET", "price": np.nan})
        request = pd.concat([working, pd.DataFrame(rows)], ignore_index=True)
        predicted = float(pipeline.predict(request, target_date=day, vegetable=vegetable, market=MARKET)["predicted_price"].iloc[0])
        working = pd.concat(
            [working, pd.DataFrame([{"date": day, "vegetable": vegetable, "market": MARKET, "source": "FORECAST",
                                     "price": predicted, "diesel_price": diesel, "petrol_price": petrol}])],
            ignore_index=True,
        )
        if day in actuals.index:
            results.append({"horizon": horizon, "actual": float(actuals[day]), "model": predicted, "hold_last": last_price})
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--vegetables", nargs="+", default=DEFAULT_VEGETABLES)
    parser.add_argument("--origins", nargs="+", default=DEFAULT_ORIGINS)
    args = parser.parse_args(argv)
    logging.basicConfig(level="WARNING")

    config = ConfigLoader(args.config)
    history = load_real_market_history(args.config)
    pipeline = PredictionPipeline(config_path=args.config)
    resolver = ProductionRegionResolver(config_path=args.config)
    region_weather = pd.read_csv(config.get_required("paths.raw_files.weather_data_by_region"), parse_dates=["date"])

    scored = []
    for vegetable in args.vegetables:
        location = resolver.resolve(vegetable)
        region = location.region_name if location.is_production_region else MARKET
        weather = region_weather[region_weather["region"] == region].drop(columns="region").set_index("date")
        for origin in args.origins:
            scored += forecast_from_origin(history, vegetable, pd.Timestamp(origin), pipeline, weather)
    frame = pd.DataFrame(scored)

    bands = []
    for low, high in HORIZON_BANDS:
        band = frame[frame["horizon"].between(low, high)]
        bands.append(
            {
                "days_ahead": str(low) if low == high else f"{low}-{high}",
                "scored_days": len(band),
                "model_mae": float((band["actual"] - band["model"]).abs().mean()),
                "model_mape": float(((band["actual"] - band["model"]).abs() / band["actual"]).mean() * 100),
                "hold_last_mae": float((band["actual"] - band["hold_last"]).abs().mean()),
                "hold_last_mape": float(((band["actual"] - band["hold_last"]).abs() / band["actual"]).mean() * 100),
            }
        )
    report = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "vegetables": args.vegetables,
        "forecast_start_dates": args.origins,
        "bands": bands,
    }
    output_path = Path(config.get("paths.metrics_dir", "reports/metrics")) / "forecast_backtest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(pd.DataFrame(bands).round(2).to_string(index=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
