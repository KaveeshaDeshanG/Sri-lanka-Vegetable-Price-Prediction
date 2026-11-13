"""Measure how far the direct forecaster's prices land from actual prices, for each day ahead.

Writes reports/metrics/forecast_error_bands.json: for each horizon from 1 to 30 days, quantiles of
log(actual / predicted) on the held-out test period. The Cultivation Planner turns these into a
likely range around each forecast. It reads data and refits the forecaster in memory exactly as
scripts/train_forecaster.py scores it; it never saves a model or changes an existing report.

Usage:
    .venv312/Scripts/python.exe scripts/forecast_error_bands.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.services import load_real_market_history  # noqa: E402
from config.config_loader import ConfigLoader  # noqa: E402
from scripts.train_forecaster import score_on_test_period  # noqa: E402
from src.feature_engineering.forecast_features import ForecastFeatureBuilder  # noqa: E402

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
OUTPUT_NAME = "forecast_error_bands.json"


def error_bands(scored: pd.DataFrame) -> list[dict]:
    """Quantiles of log(actual / predicted) for each horizon in a scored test set."""
    bands = []
    for horizon, rows in scored.groupby("horizon"):
        log_error = np.log(rows["target_price"] / rows["prediction"])
        bands.append(
            {
                "horizon": int(horizon),
                "rows": int(len(rows)),
                **{f"q{int(q * 100)}": float(log_error.quantile(q)) for q in QUANTILES},
                "coverage_q10_q90": float(log_error.between(log_error.quantile(0.1), log_error.quantile(0.9)).mean()),
            }
        )
    return bands


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)

    config = ConfigLoader(args.config)
    test_start = pd.Timestamp(config.get_required("forecasting.test_start"))
    history = load_real_market_history(args.config)
    origin_rows = ForecastFeatureBuilder(args.config).build_origin_features(history)
    scored = score_on_test_period(origin_rows, test_start)

    report = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "test_start": str(test_start.date()),
        "test_end": str(scored["target_date"].max().date()),
        "measure": "log(actual price / forecast price) on the held-out test period",
        "bands": error_bands(scored),
    }
    output = Path(config.get("paths.metrics_dir", "reports/metrics")) / OUTPUT_NAME
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    first, last = report["bands"][0], report["bands"][-1]
    print(f"Wrote {output}: day 1 middle 80% {np.exp(first['q10']):.3f}x to {np.exp(first['q90']):.3f}x, "
          f"day {last['horizon']} {np.exp(last['q10']):.3f}x to {np.exp(last['q90']):.3f}x")


if __name__ == "__main__":
    main()
