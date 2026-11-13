"""Chronological evaluation of the direct forecaster at 1, 7, 14 and 30 days, overall and per vegetable.

Fits the forecaster in memory on targets before forecasting.test_start only, exactly as
scripts/train_forecaster.py scores it, then scores every test-period origin. The baseline is
holding the last known price. It never saves a model.

    --name        label for the output file (reports/metrics/evaluation_<name>.json)
    --test-end    last target date to score, so two datasets can be compared on identical dates

Usage:
    .venv312/Scripts/python.exe scripts/evaluate_forecaster.py --name after_update
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

HORIZONS = (1, 7, 14, 30)


def metrics(frame: pd.DataFrame) -> dict:
    actual = frame["target_price"].to_numpy(float)
    out = {"rows": int(len(frame))}
    for name, column in (("model", "prediction"), ("hold", "last_price")):
        error = actual - frame[column].to_numpy(float)
        out[f"{name}_mae"] = float(np.abs(error).mean())
        out[f"{name}_rmse"] = float(np.sqrt((error**2).mean()))
        out[f"{name}_mape"] = float((np.abs(error) / actual).mean() * 100)
    out["mae_gain_pct"] = float((1 - out["model_mae"] / out["hold_mae"]) * 100) if out["hold_mae"] else float("nan")
    return out


def evaluate(config_path: str, test_end: str | None = None) -> dict:
    config = ConfigLoader(config_path)
    test_start = pd.Timestamp(config.get_required("forecasting.test_start"))
    history = load_real_market_history(config_path)
    origin_rows = ForecastFeatureBuilder(config_path).build_origin_features(history)
    scored = score_on_test_period(origin_rows, test_start)
    if test_end:
        scored = scored[scored["target_date"] <= pd.Timestamp(test_end)]
    training = origin_rows[origin_rows["actual_price"].notna() & (origin_rows["origin"] < test_start)]
    report = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "data_first_date": str(history["date"].min().date()),
        "data_last_date": str(history.loc[history["price"].notna(), "date"].max().date()),
        "training_targets_before": str(test_start.date()),
        "training_first_origin": str(training["origin"].min().date()),
        "test_start": str(test_start.date()),
        "test_end": str(scored["target_date"].max().date()),
        "vegetables_trained": int(origin_rows["vegetable"].nunique()),
        "by_horizon": {},
        "by_vegetable": {},
    }
    for horizon in HORIZONS:
        rows = scored[scored["horizon"] == horizon]
        report["by_horizon"][str(horizon)] = metrics(rows)
        report["by_vegetable"][str(horizon)] = {vegetable: metrics(group) for vegetable, group in rows.groupby("vegetable")}
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    report = evaluate(args.config, args.test_end)
    output_dir = Path(args.output_dir or ConfigLoader(args.config).get("paths.metrics_dir", "reports/metrics"))
    output = output_dir / f"evaluation_{args.name}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for horizon, values in report["by_horizon"].items():
        print(f"day {horizon:>2}: rows {values['rows']:5d}  MAE {values['model_mae']:7.2f} (hold {values['hold_mae']:7.2f})  "
              f"RMSE {values['model_rmse']:7.2f} (hold {values['hold_rmse']:7.2f})  MAPE {values['model_mape']:5.2f}% (hold {values['hold_mape']:5.2f}%)")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
