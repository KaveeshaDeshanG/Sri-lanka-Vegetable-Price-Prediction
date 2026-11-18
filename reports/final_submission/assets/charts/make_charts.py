"""Report charts, drawn only from the project's saved metrics and stored datasets (nothing typed in by hand).

Run from the project folder:  .venv312/Scripts/python.exe reports/final_submission/assets/charts/make_charts.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
MODEL, BASE, INK, MUTED, GRID = "#2f7d4f", "#9aa5b1", "#1d2733", "#5b6773", "#e3e7eb"
SERIES = ["#2f7d4f", "#3b5f8a", "#b5651d"]  # fixed order: carrot, beans, tomato
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.edgecolor": "#b8c0c8", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


metrics = json.loads((ROOT / "reports/metrics/direct_forecaster_metrics.json").read_text(encoding="utf-8"))

# 1. MAE by forecast range: model vs holding the last price
bands = metrics["bands"]
labels = [b["days_ahead"] for b in bands]
x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(6.6, 3.3))
ax.bar(x - w / 2 - 0.01, [b["model_mae"] for b in bands], w, color=MODEL, label="Ridge direct forecaster")
ax.bar(x + w / 2 + 0.01, [b["hold_mae"] for b in bands], w, color=BASE, label="Hold the last price (baseline)")
for i, b in enumerate(bands):
    ax.text(i - w / 2, b["model_mae"] + 1, f"{b['model_mae']:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xticks(x, labels); ax.set_xlabel("Days ahead"); ax.set_ylabel("MAE (Rs per kg)")
ax.grid(axis="x", visible=False); ax.legend(loc="upper left")
save(fig, "fig_mae_by_horizon")

# 2. Next-day MAE reduction by vegetable
veg = pd.DataFrame(metrics["next_day_by_vegetable"])
veg["gain"] = (veg["hold_mae"] - veg["mae"]) / veg["hold_mae"] * 100
veg = veg.sort_values("gain")
fig, ax = plt.subplots(figsize=(6.6, 6.2))
ax.barh(veg["vegetable"], veg["gain"], color=[MODEL if g > 0 else "#b5651d" for g in veg["gain"]], height=0.65)
ax.axvline(0, color=MUTED, linewidth=0.9)
ax.set_xlabel("Next-day MAE reduction vs holding the last price (%)")
ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=8.5)
save(fig, "fig_gain_by_vegetable")

# 3. Earlier one-day models vs yesterday's price
mm = json.loads((ROOT / "reports/metrics/model_metrics.json").read_text(encoding="utf-8"))
rows = [("Linear Regression", mm["models"]["linear_regression"]), ("Random Forest", mm["models"]["random_forest"]),
        ("XGBoost", mm["models"]["xgboost"]), ("Yesterday's price", mm["naive_baseline"])]
fig, ax = plt.subplots(figsize=(6.6, 2.6))
names = [r[0] for r in rows]; maes = [r[1]["mae"] for r in rows]
ax.barh(names[::-1], maes[::-1], color=[BASE] + [MODEL] * 3, height=0.6)
for i, v in enumerate(maes[::-1]):
    ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
ax.set_xlabel("MAE on the test set (Rs per kg)"); ax.set_xlim(0, max(maes) * 1.12); ax.grid(axis="y", visible=False)
save(fig, "fig_earlier_models")

# 4. Weekly mean Dambulla wholesale prices of three vegetables, 2015-2026 (current dataset)
prices = pd.read_csv(ROOT / "data/raw/harti_prices.csv", parse_dates=["date"])
fig, ax = plt.subplots(figsize=(6.6, 3.0))
for colour, name in zip(SERIES, ["Carrot", "Beans", "Tomato"]):
    s = prices[prices["vegetable"] == name].set_index("date")["price"].resample("W").mean()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="W"))
    ax.plot(s.index, s.values, color=colour, linewidth=1.1, label=name)
ax.set_ylabel("Rs per kg (weekly mean)"); ax.legend(loc="upper left", ncol=3)
ax.set_xlim(prices["date"].min(), prices["date"].max())
save(fig, "fig_eda_long_series")

# 5. Days with a Dambulla price, by year
days = prices.groupby(prices["date"].dt.year)["date"].nunique()
fig, ax = plt.subplots(figsize=(6.6, 2.6))
ax.bar(days.index.astype(str), days.values, color=MODEL, width=0.65)
for i, v in enumerate(days.values):
    ax.text(i, v + 4, str(v), ha="center", fontsize=8, color=INK)
ax.set_ylabel("Days with a price"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 400)
save(fig, "fig_eda_coverage")

print("charts written to", OUT, "| HARTI Dambulla rows:", len(prices), "| latest:", prices["date"].max().date())
