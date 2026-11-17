"""Retrains the direct forecaster on the current data, but only keeps it if it passes a held-out check.

Run it after new data has been imported (for example monthly, after scripts/daily_update.py). It:

1. Holds out the most recent --holdout-days of prices. A candidate forecaster is fitted only on
   targets before that window, exactly as scripts/train_forecaster.py fits its test models, and
   is scored on every forecast whose target falls inside it.
2. Accepts the candidate only if, on that window, its mean absolute error is lower than holding
   the last price, overall (with the 95% interval of the gain above zero) and in each reported
   band of days ahead. Optionally (--compare-history-start) it must also be no worse than the same
   model fitted only on data from that date, which tests whether older history helps.
3. If the saved model never saw the window (it was trained before it started), the candidate
   must also beat the saved model there.
4. If accepted: backs up the current model and metrics, then runs scripts/train_forecaster.py,
   which fits on all data and rewrites models/direct_forecaster.joblib and its metrics. If
   rejected, nothing is changed.

Every decision, with its numbers and reasons, is written to reports/metrics/retrain_decision.json
(the latest) and appended to reports/metrics/retrain_history.jsonl.

Exit codes: 0 accepted and saved; 1 rejected (the saved model is unchanged); 2 an error.

Usage:
    .venv312/Scripts/python.exe scripts/retrain_if_better.py
    .venv312/Scripts/python.exe scripts/retrain_if_better.py --compare-history-start 2024-04-25
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.services import load_real_market_history  # noqa: E402
from config.config_loader import ConfigLoader  # noqa: E402
from scripts import train_forecaster  # noqa: E402
from scripts.train_forecaster import REPORT_BANDS, band_label, score_on_test_period  # noqa: E402
from src.feature_engineering.forecast_features import ForecastFeatureBuilder, add_horizon_features, add_vegetable_columns  # noqa: E402
from src.models.direct_forecaster import FORECASTER_FILENAME, DirectForecaster, mae_gain_vs_holding_price, score_predictions, usable_rows  # noqa: E402


def _json_value(value):
    """json.dumps fallback: NumPy scalars (e.g. the numpy.bool_ from comparing two MAEs) as plain Python values."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (_PROJECT_ROOT / candidate).resolve()


def band_scores(scored: pd.DataFrame, prediction_column: str = "prediction") -> list[dict]:
    results = []
    for first, last in REPORT_BANDS:
        band = scored[scored["horizon"].between(first, last)]
        if band.empty:
            continue
        model = score_predictions(band["target_price"], band[prediction_column])
        hold = score_predictions(band["target_price"], band["last_price"])
        results.append(
            {
                "days_ahead": band_label(first, last),
                "rows": len(band),
                "model_mae": model["mae"],
                "model_rmse": model["rmse"],
                "model_mape": model["mape"],
                "hold_mae": hold["mae"],
                "hold_rmse": hold["rmse"],
                "hold_mape": hold["mape"],
                **mae_gain_vs_holding_price(band, prediction_column),
            }
        )
    return results


def score_saved_model(saved: DirectForecaster, origin_rows: pd.DataFrame, holdout_start: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for horizon in saved.models:
        rows = usable_rows(add_vegetable_columns(add_horizon_features(origin_rows, horizon), saved.vegetables))
        rows = rows[(rows["target_date"] >= holdout_start) & rows["vegetable"].isin(saved.vegetables)].copy()
        if rows.empty:
            continue
        rows["saved_prediction"] = saved.predict_rows(rows, horizon)
        rows["horizon"] = horizon
        parts.append(rows[["vegetable", "origin", "target_date", "horizon", "saved_prediction"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def decide(config_path: str, holdout_days: int, compare_history_start: str | None) -> dict:
    config = ConfigLoader(config_path)
    history = load_real_market_history(config_path)
    latest = history.loc[history["price"].notna(), "date"].max()
    holdout_start = latest - pd.Timedelta(days=holdout_days - 1)
    origin_rows = ForecastFeatureBuilder(config_path).build_origin_features(history)

    scored = score_on_test_period(origin_rows, holdout_start)
    bands = band_scores(scored)
    overall = mae_gain_vs_holding_price(scored)
    reasons, checks = [], {}

    checks["beats_holding_price_overall"] = overall["gain_low"] > 0
    if not checks["beats_holding_price_overall"]:
        reasons.append(f"overall MAE gain over holding the price is {overall['gain_pct']:.1f}% (95% interval {overall['gain_low']:.1f} to {overall['gain_high']:.1f}), not clearly above zero")
    worse_bands = [b["days_ahead"] for b in bands if b["gain_pct"] <= 0]
    checks["beats_holding_price_in_every_band"] = not worse_bands
    if worse_bands:
        reasons.append(f"MAE is not lower than holding the price for days ahead {worse_bands}")

    comparison = None
    if compare_history_start:
        start = pd.Timestamp(compare_history_start)
        short_history = history[history["date"] >= start]
        short_rows = ForecastFeatureBuilder(config_path).build_origin_features(short_history)
        short = score_on_test_period(short_rows, holdout_start)
        keys = ["vegetable", "origin", "target_date", "horizon"]
        joined = scored.merge(short[keys + ["prediction"]].rename(columns={"prediction": "short_prediction"}), on=keys)
        long_mae = (joined["target_price"] - joined["prediction"]).abs().mean()
        short_mae = (joined["target_price"] - joined["short_prediction"]).abs().mean()
        comparison = {"history_start": str(start.date()), "rows": len(joined), "candidate_mae": float(long_mae), "short_history_mae": float(short_mae),
                      "short_history_bands": band_scores(joined, "short_prediction")}
        checks["no_worse_than_shorter_history"] = long_mae <= short_mae
        if long_mae > short_mae:
            reasons.append(f"MAE {long_mae:.2f} is worse than the same model fitted on data from {start.date()} only ({short_mae:.2f})")

    saved_comparison = None
    try:
        saved = DirectForecaster.load(config_path)
    except Exception:  # noqa: BLE001 -- no usable saved model: nothing to compare with
        saved = None
    if saved is not None and saved.trained_through is not None and saved.trained_through < holdout_start:
        keys = ["vegetable", "origin", "target_date", "horizon"]
        joined = scored.merge(score_saved_model(saved, origin_rows, holdout_start), on=keys)
        candidate_mae = (joined["target_price"] - joined["prediction"]).abs().mean()
        saved_mae = (joined["target_price"] - joined["saved_prediction"]).abs().mean()
        saved_comparison = {"rows": len(joined), "candidate_mae": float(candidate_mae), "saved_mae": float(saved_mae)}
        checks["beats_saved_model_on_unseen_window"] = candidate_mae < saved_mae
        if candidate_mae >= saved_mae:
            reasons.append(f"MAE {candidate_mae:.2f} is not lower than the saved model's {saved_mae:.2f} on dates it never saw")
    elif saved is not None:
        saved_comparison = {"skipped": f"the saved model was trained through {saved.trained_through.date()}, which includes the held-out window, so comparing on it would favour the saved model unfairly"}

    return {
        "decided_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "data_first_date": str(history["date"].min().date()),
        "data_last_date": str(latest.date()),
        "holdout_start": str(holdout_start.date()),
        "holdout_end": str(scored["target_date"].max().date()),
        "vegetables": int(origin_rows["vegetable"].nunique()),
        "overall": {"rows": len(scored), **overall},
        "bands": bands,
        "compared_with_shorter_history": comparison,
        "compared_with_saved_model": saved_comparison,
        "checks": checks,
        "accepted": all(checks.values()),
        "reasons_rejected": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--holdout-days", type=int, default=60)
    parser.add_argument("--compare-history-start", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Decide and report, but never save a model.")
    args = parser.parse_args(argv)
    config = ConfigLoader(args.config)
    metrics_dir = _resolve(config.get("paths.metrics_dir", "reports/metrics"))

    decision = decide(args.config, args.holdout_days, args.compare_history_start)
    if decision["accepted"] and not args.dry_run:
        backup = _resolve(config.get_required("data_update.backup_dir")) / f"models_before_retrain_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
        backup.mkdir(parents=True)
        for path in (_resolve(config.get("paths.models_dir", "models")) / FORECASTER_FILENAME, metrics_dir / "direct_forecaster_metrics.json"):
            if path.exists():
                shutil.copy2(path, backup / path.name)
        decision["backup"] = str(backup)
        train_forecaster.main(["--config", args.config])
        decision["saved_model_trained_through"] = str(DirectForecaster.load(args.config).trained_through.date())
    decision["saved"] = bool(decision["accepted"] and not args.dry_run)

    (metrics_dir / "retrain_decision.json").write_text(json.dumps(decision, indent=2, default=_json_value), encoding="utf-8")
    with (metrics_dir / "retrain_history.jsonl").open("a", encoding="utf-8") as history:
        history.write(json.dumps(decision, default=_json_value) + "\n")
    print(json.dumps({k: decision[k] for k in ("data_first_date", "data_last_date", "holdout_start", "holdout_end", "overall", "checks", "accepted", "reasons_rejected", "saved")}, indent=2, default=_json_value))
    return 0 if decision["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
