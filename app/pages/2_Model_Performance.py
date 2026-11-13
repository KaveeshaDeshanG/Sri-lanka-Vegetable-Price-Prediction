"""Streamlit page: how the trained models performed, read from the saved training reports."""

import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)  # config and data paths are relative to the project root

import altair as alt
import pandas as pd
import streamlit as st

from app import ui
from app.catalogue import display_name
from config.config_loader import ConfigLoader

METRICS_DIR = Path(ConfigLoader("config/config.yaml").get("paths.metrics_dir", "reports/metrics"))
BASELINE_LABEL = "Yesterday's price (no model)"

st.set_page_config(page_title="Model Performance | Vegetable Price Prediction", page_icon="📊", layout="wide", initial_sidebar_state="auto")
ui.apply_theme()
ui.page_header(
    "Model performance",
    "Measured accuracy on a held-out test period, compared with simply holding the last price.",
    "Evaluation",
)


def _error_by_range_chart(bands: pd.DataFrame) -> alt.Chart:
    """Average error (MAE) for the model and for holding the last price, at each forecast range."""
    order = list(bands["days_ahead"])
    long = pd.concat(
        [
            pd.DataFrame({"Days ahead": bands["days_ahead"], "Method": "Ridge direct forecaster", "MAE (Rs)": bands["model_mae"].round(2)}),
            pd.DataFrame({"Days ahead": bands["days_ahead"], "Method": "Hold last price", "MAE (Rs)": bands["hold_mae"].round(2)}),
        ]
    )
    return (
        alt.Chart(long)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Days ahead:N", sort=order, title="Days ahead", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Method:N", sort=["Ridge direct forecaster", "Hold last price"]),
            y=alt.Y("MAE (Rs):Q", title="Average error, MAE (Rs / kg)"),
            color=alt.Color(
                "Method:N",
                scale=alt.Scale(domain=["Ridge direct forecaster", "Hold last price"], range=[ui.COLORS["primary"], "#475569"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["Days ahead", "Method", "MAE (Rs)"],
        )
        .properties(height=300)
    )


def _load_report(filename: str, script: str) -> dict | None:
    """A saved report, or None after telling the reader what's wrong and how to fix it."""
    path = METRICS_DIR / filename
    if not path.exists():
        st.warning(f"`{path.as_posix()}` is missing, so these results can't be shown. `{script}` creates it.")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        st.error(f"`{path.as_posix()}` could not be read ({error}). Restore it from a backup, or run `{script}`.")
        return None


def _show_section(render, filename: str, report: dict) -> None:
    """Render a report's section, reporting a file with unexpected contents on the page."""
    try:
        render(report)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        st.error(f"`{(METRICS_DIR / filename).as_posix()}` has unexpected contents ({type(error).__name__}: {error}).")


def _show_direct(direct: dict) -> None:
    st.header("Ridge direct forecaster (used for real data on the Predict page)")
    st.caption(
        f"One Ridge model per day ahead, up to {direct['max_horizon_days']} days, using recent prices, other markets' "
        f"published prices and observed weather. Scored on {direct['test_start']} to {direct['test_end']}, a period "
        "kept out of training and of choosing the configuration. These scores come from models trained only on "
        "earlier data; the saved models were then refit on all data, so that exact version has no separate score. "
        "This period was also used to score the earlier models below."
    )
    bands = pd.DataFrame(direct["bands"])
    next_day, furthest = bands.iloc[0], bands.iloc[-1]
    beats = sum(row["mae"] < row["hold_mae"] for row in direct["next_day_by_vegetable"])
    ui.stat_cards(
        [
            {"label": "Next-day error", "value": f"{next_day['model_mape']:.2f}%", "sub": f"MAE Rs. {next_day['model_mae']:.2f}"},
            {"label": "Next-day gain", "value": f"{next_day['gain_pct']:.1f}%", "sub": f"95% interval {next_day['gain_low']:.1f} to {next_day['gain_high']:.1f}", "accent": True},
            {"label": f"{furthest['days_ahead']} days ahead", "value": f"{furthest['model_mape']:.1f}%", "sub": f"vs {furthest['hold_mape']:.1f}% holding the price"},
            {"label": "Vegetables improved", "value": f"{beats} of {len(direct['next_day_by_vegetable'])}", "sub": "next-day, vs holding the price"},
        ]
    )
    st.subheader("Average error by forecast range")
    st.altair_chart(_error_by_range_chart(bands), width="stretch")
    st.dataframe(
        pd.DataFrame(
            {
                "Days ahead": bands["days_ahead"],
                "Rows": bands["scored_rows"],
                "Model MAE (Rs)": bands["model_mae"].round(2),
                "Model RMSE (Rs)": bands["model_rmse"].round(2),
                "Model error (%)": bands["model_mape"].round(2),
                "Hold last price MAE (Rs)": bands["hold_mae"].round(2),
                "Hold last price RMSE (Rs)": bands["hold_rmse"].round(2),
                "Hold last price error (%)": bands["hold_mape"].round(2),
                "MAE reduction (%)": bands["gain_pct"].round(1),
                "MAE reduction, 95% interval": [
                    f"{low:.1f} to {high:.1f}" for low, high in zip(bands["gain_low"], bands["gain_high"])
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "\"Error (%)\" is MAPE. MAE is the average size of an error. RMSE weighs large errors more: from 8 days ahead, "
        "the model's RMSE is about the same as holding the last price, so it doesn't reduce the largest errors."
    )
    next_day = bands.iloc[0]
    if next_day["gain_low"] > 0:
        st.success(
            f"Next-day: average error is {next_day['gain_pct']:.1f}% lower than holding the last price, and the "
            "whole 95% interval is above zero. Holding the last price means assuming the price doesn't change."
        )
    else:
        st.warning("Next-day: the improvement over holding the last price isn't statistically clear.")

    earlier = direct.get("next_day_vs_earlier_model")
    if earlier:
        st.subheader("Next-day comparison on the same rows")
        names = {
            "direct_forecaster": "Direct forecaster",
            "earlier_model": "Earlier 51-feature model",
            "hold_last_price": "Hold last price",
        }
        st.dataframe(
            pd.DataFrame(
                [
                    {"Method": label, "MAE (Rs)": earlier[key]["mae"], "RMSE (Rs)": earlier[key]["rmse"], "Error (%)": earlier[key]["mape"], "R²": earlier[key]["r2"]}
                    for key, label in names.items()
                ]
            ).round({"MAE (Rs)": 2, "RMSE (Rs)": 2, "Error (%)": 2, "R²": 4}),
            hide_index=True,
            width="stretch",
        )
        st.caption(f"{earlier['rows']} rows that every method can score.")

    st.subheader("Next-day results by vegetable")
    vegetables = pd.DataFrame(direct["next_day_by_vegetable"])
    vegetables["Beats holding the last price"] = vegetables["mae"] < vegetables["hold_mae"]
    st.caption(
        f"Beats holding the last price for {int(vegetables['Beats holding the last price'].sum())} of {len(vegetables)} "
        "vegetables with test rows. A vegetable with no Dambulla reading in the test period isn't listed."
    )
    st.dataframe(
        vegetables.sort_values("mape")
        .rename(columns={"vegetable": "Vegetable", "test_rows": "Test rows", "mae": "Model MAE (Rs)", "mape": "Model error (%)", "hold_mae": "Hold last price MAE (Rs)"})
        .round(2),
        hide_index=True,
        width="stretch",
    )


def _show_earlier_models(report: dict) -> None:
    st.header("Earlier models: Linear Regression, Random Forest and XGBoost")
    st.caption(
        "The selected model below serves uploaded data on the Predict page. For real data it's used only if the "
        "forecaster can't run and you explicitly choose it."
    )

    dataset = report.get("dataset", {})
    st.caption(
        f"Report generated {report.get('generated_at', '?')}. Test period rows: {dataset.get('test_rows', '?')}; "
        f"training rows: {dataset.get('train_rows', '?')}; features: {dataset.get('feature_count', '?')}. "
        "Each score is a one-day-ahead prediction on data the model never saw in training."
    )

    st.subheader("Model comparison")
    best_name = report["best_model"]["name"]
    rows = [
        {"Method": name.replace("_", " ").title() + (" (selected)" if name == best_name else ""), **metrics}
        for name, metrics in report["models"].items()
    ]
    baseline = report.get("naive_baseline")
    if baseline:
        rows.append({"Method": BASELINE_LABEL, **baseline})
    comparison = pd.DataFrame(rows).rename(columns={"mae": "MAE (Rs)", "rmse": "RMSE (Rs)", "r2": "R²"})
    st.dataframe(comparison.round({"MAE (Rs)": 2, "RMSE (Rs)": 2, "R²": 4}), hide_index=True, width="stretch")
    st.caption(
        f"The saved model is chosen by lowest {report['best_model']['selection_metric'].upper()} on the test period. "
        "\"Yesterday's price\" needs a reading on the exact previous day, so it scores slightly different rows from "
        "\"hold last price\" above, which can carry a reading forward."
    )

    if baseline:
        best = report["models"][best_name]
        difference = (1 - best["mae"] / baseline["mae"]) * 100
        if difference > 0:
            st.success(f"The selected model's average error (MAE) is {difference:.1f}% lower than using yesterday's price.")
        else:
            st.warning(
                f"**The best of these models does not beat using yesterday's price:** its average error (MAE) is "
                f"{-difference:.1f}% higher. The high R² mostly reflects how closely today's price follows yesterday's, "
                "and how much average price differs between vegetables."
            )

    per_vegetable = report.get("per_vegetable")
    if per_vegetable:
        st.subheader("Results by vegetable")
        table = pd.DataFrame(per_vegetable)
        table["Beats yesterday's price"] = table["mae"] < table["baseline_mae"]
        st.caption(f"The selected model beats yesterday's price for {int(table['Beats yesterday\'s price'].sum())} of {len(table)} groups.")
        st.dataframe(
            table.sort_values("mape")
            .rename(
                columns={
                    "vegetable": "Vegetable",
                    "test_rows": "Test rows",
                    "mae": "Model MAE (Rs)",
                    "mape": "Model error (%)",
                    "baseline_mae": "Yesterday's price MAE (Rs)",
                }
            )
            .round(2),
            hide_index=True,
            width="stretch",
        )


def _show_backtest(backtest: dict) -> None:
    st.subheader("Forecasting one day at a time vs directly")
    st.caption(
        f"Forecasts from {len(backtest['forecast_start_dates'])} past start dates for {len(backtest['vegetables'])} "
        "vegetables, compared with the prices actually recorded. The one-day-at-a-time approach feeds each "
        "prediction back in as the next day's input, and past weather stands in for forecast weather, which "
        "slightly favours it."
    )
    comparison = pd.DataFrame(backtest["bands"]).rename(
        columns={
            "days_ahead": "Days ahead",
            "scored_days": "Days scored",
            "model_mape": "One day at a time, error (%)",
            "hold_last_mape": "Hold last price, error (%)",
        }
    )[["Days ahead", "Days scored", "One day at a time, error (%)", "Hold last price, error (%)"]]
    if isinstance(direct, dict) and direct.get("same_sample_as_recursive_backtest"):
        same_sample = pd.DataFrame(direct["same_sample_as_recursive_backtest"])
        comparison.insert(2, "Direct forecaster, error (%)", same_sample["mape"].to_numpy())
    st.dataframe(comparison.round(1), hide_index=True, width="stretch")


def _show_evaluation(evaluation: dict) -> None:
    st.header("Accuracy at 1, 7, 14 and 30 days")
    st.caption(
        f"Models trained only on prices before {evaluation['test_start']} (data from {evaluation['training_first_origin']}), "
        f"scored on every day from {evaluation['test_start']} to {evaluation['test_end']}. Baseline: hold the last known price."
    )
    rows = []
    for horizon, values in evaluation["by_horizon"].items():
        rows.append(
            {
                "Days ahead": int(horizon),
                "Rows scored": values["rows"],
                "Model MAE (Rs)": values["model_mae"],
                "Baseline MAE (Rs)": values["hold_mae"],
                "Model RMSE (Rs)": values["model_rmse"],
                "Baseline RMSE (Rs)": values["hold_rmse"],
                "Model MAPE (%)": values["model_mape"],
                "Baseline MAPE (%)": values["hold_mape"],
            }
        )
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
    worse = [
        {"Days ahead": int(horizon), "Vegetable": display_name(vegetable), "Model MAE": values["model_mae"], "Baseline MAE": values["hold_mae"], "Rows": values["rows"]}
        for horizon, per_vegetable in evaluation["by_vegetable"].items()
        for vegetable, values in per_vegetable.items()
        if values["model_mae"] > values["hold_mae"]
    ]
    with st.expander(f"Where the model does worse than holding the last price ({len(worse)} vegetable-horizon pairs)"):
        st.dataframe(pd.DataFrame(worse).round(2) if worse else pd.DataFrame({"Result": ["None"]}), hide_index=True, width="stretch")
    with st.expander("Every vegetable at every horizon"):
        table = pd.DataFrame(
            [
                {"Days ahead": int(horizon), "Vegetable": display_name(vegetable), "Rows": values["rows"], "Model MAE": values["model_mae"],
                 "Baseline MAE": values["hold_mae"], "Model RMSE": values["model_rmse"], "Model MAPE (%)": values["model_mape"]}
                for horizon, per_vegetable in evaluation["by_vegetable"].items()
                for vegetable, values in per_vegetable.items()
            ]
        )
        st.dataframe(table.round(2), hide_index=True, width="stretch")


direct = _load_report("direct_forecaster_metrics.json", "scripts/train_forecaster.py")
if direct is not None:
    _show_section(_show_direct, "direct_forecaster_metrics.json", direct)
    st.divider()

if (METRICS_DIR / "evaluation_current_model.json").exists():
    evaluation = _load_report("evaluation_current_model.json", "scripts/evaluate_forecaster.py")
    if evaluation is not None:
        _show_section(_show_evaluation, "evaluation_current_model.json", evaluation)
        st.divider()

earlier_report = _load_report("model_metrics.json", "scripts/train.py")
if earlier_report is not None:
    _show_section(_show_earlier_models, "model_metrics.json", earlier_report)

backtest_report = _load_report("forecast_backtest.json", "scripts/backtest_forecast.py")
if backtest_report is not None:
    _show_section(_show_backtest, "forecast_backtest.json", backtest_report)
