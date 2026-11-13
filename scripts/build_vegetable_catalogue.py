"""Builds data/reference/vegetable_catalogue.csv: every vegetable in the project's sources, and what it supports.

Inputs, all read as they are:
- data/raw/harti_prices.csv and harti_all_markets.csv (HARTI daily wholesale, per market)
- data/raw/dcs_weekly_retail_prices.csv (DCS weekly retail, Colombo District)
- data/reference/doa_harvest_durations_v2.csv (DOA HORDI published durations)
- data/reference/vegetable_names.csv (Sinhala and Tamil names from DOA's own Sinhala and Tamil pages)
- models/direct_forecaster.joblib (which series the forecaster was trained on)

Support columns are a snapshot for the documented date; the app recomputes them from the data
(app/catalogue.py), so a series that goes stale stops being offered without rebuilding this file.

Usage:
    .venv312/Scripts/python.exe scripts/build_vegetable_catalogue.py
"""

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.catalogue import HARTI_SERIES, NO_DURATION_REASONS, freshness  # noqa: E402
from app.cultivation import load_crop_timings  # noqa: E402
from src.models.direct_forecaster import DirectForecaster  # noqa: E402
from src.preprocessing.transformer import DataTransformer  # noqa: E402

RAW = _PROJECT_ROOT / "data" / "raw"
REFERENCE = _PROJECT_ROOT / "data" / "reference"
# DCS item name -> DOA crop, for vegetables DCS prices that HARTI's daily bulletin doesn't.
DCS_CROPS = {
    "Ash Plantain": None, "Ash Pumpkin": None, "Beans (Butter)": "Beans", "Beans (Green)": "Beans", "Beetroot": "Beetroot",
    "Big Onion (Imported)": None, "Big Onion (Local)": None, "Bitter Gourd": "Bitter Gourd", "Brinjal": "Brinjal",
    "Cabbage": "Cabbage", "Capsicum": "Capsicum", "Carrot": "Carrot", "Cucumber": "Cucumber", "Drumstick": None,
    "Gotukola": None, "Green Chillies": None, "Kankun": None, "Kathurumurunga": None, "Knol Khol": "Knol Khol",
    "Kohila Leaves": "Kohila", "Kohila Yams": "Kohila", "Ladies Fingers": "Okra", "Leeks": "Leeks", "Lime": None,
    "Long Beans": "Long Beans (Mae)", "Luffa": "Luffa", "Mukunuwenna": None, "Nivithi": None, "Potato (Imported)": None,
    "Potato (Local)": "Potato", "Pumpkin": "Pumpkin", "Radish": "Radish", "Red Onion": None, "Sarana": None,
    "Snake Gourd": "Snake Gourd", "Thampala": None, "Tomato": "Tomato",
}


def main() -> None:
    today = pd.Timestamp.now().normalize()
    dambulla = DataTransformer().standardize_vegetable_names(pd.read_csv(RAW / "harti_prices.csv", parse_dates=["date"]))
    markets = pd.read_csv(RAW / "harti_all_markets.csv", parse_dates=["published"])
    dcs_path = RAW / "dcs_weekly_retail_prices.csv"
    dcs = pd.read_csv(dcs_path) if dcs_path.exists() else pd.DataFrame(columns=["vegetable", "period", "year", "month", "week_of_month", "item"])
    timings = load_crop_timings(REFERENCE / "doa_harvest_durations_v2.csv")
    names = pd.read_csv(REFERENCE / "vegetable_names.csv", dtype=str, keep_default_na=False).set_index("crop")
    try:
        trained = set(DirectForecaster.load(str(_PROJECT_ROOT / "config" / "config.yaml")).vegetables)
    except Exception:  # noqa: BLE001 -- the catalogue still describes the data without a model
        trained = set()
    latest_data = dambulla["date"].max()

    def harvest(crop):
        rows = [t for t in timings if t.crop == crop]
        if not rows:
            return 0, "", False, NO_DURATION_REASONS.get(crop, "No DOA HORDI duration found for this crop.")
        return len(rows), "; ".join(sorted({t.source_url for t in rows})), True, ""

    def name_cols(crop):
        if crop and crop in names.index:
            row = names.loc[crop]
            return row["sinhala_name"], row["tamil_name"], row["names_source_url"]
        return "", "", ""

    def dcs_cols(dcs_names):
        part = dcs[dcs["vegetable"].isin(dcs_names)].sort_values(["year", "month", "week_of_month"])
        if part.empty:
            return "", "", "", ""
        return "; ".join(dcs_names), part["period"].iloc[0], part["period"].iloc[-1], "; ".join(sorted(part["item"].unique()))

    records = []
    used_dcs = set()
    for series, (display, crop, raw_labels) in HARTI_SERIES.items():
        group = dambulla[dambulla["vegetable"] == series]
        panel = markets[markets["vegetable"] == series]
        if group.empty and panel.empty:
            continue
        dcs_names = [d for d, c in DCS_CROPS.items() if (c is not None and c == crop) or d == display or d == series]
        used_dcs.update(dcs_names)
        dcs_series, dcs_first, dcs_latest, dcs_items = dcs_cols(dcs_names)
        count, harvest_urls, harvest_ok, harvest_note = harvest(crop) if crop else (0, "", False, NO_DURATION_REASONS.get(series, "No DOA HORDI crop page."))
        sinhala, tamil, names_url = name_cols(crop)
        if group.empty:
            supported, note = False, "No Dambulla prices; other markets only."
        else:
            fresh = freshness(group["date"].max(), today)
            if series not in trained:
                supported, note = False, "The forecaster wasn't trained on this series."
            elif fresh.level == "stale":
                supported, note = False, f"Latest Dambulla price {group['date'].max().date()} is older than the 30-day forecast reach."
            else:
                supported, note = True, ""
        records.append(
            {
                "display_name": display,
                "internal_name": series,
                "entry_type": "HARTI daily wholesale series",
                "crop": crop or "",
                "sinhala_name": sinhala,
                "tamil_name": tamil,
                "names_source_url": names_url,
                "dataset_aliases": raw_labels + (f"; DCS: {dcs_items}" if dcs_items else ""),
                "harti_markets": "; ".join(sorted(panel["market"].unique())),
                "harti_dambulla_first": "" if group.empty else str(group["date"].min().date()),
                "harti_dambulla_latest": "" if group.empty else str(group["date"].max().date()),
                "harti_dambulla_days": int(group["date"].nunique()),
                "dcs_retail_series": dcs_series,
                "dcs_first_week": dcs_first,
                "dcs_latest_week": dcs_latest,
                "harvest_reference_rows": count,
                "harvest_source_urls": harvest_urls,
                "price_prediction_supported": supported,
                "price_prediction_note": note,
                "harvest_timing_supported": harvest_ok,
                "harvest_timing_note": harvest_note,
            }
        )
    for dcs_name, crop in DCS_CROPS.items():
        if dcs_name in used_dcs or dcs[dcs["vegetable"] == dcs_name].empty:
            continue
        count, harvest_urls, harvest_ok, harvest_note = harvest(crop) if crop else (0, "", False, "No DOA HORDI duration found for this crop.")
        sinhala, tamil, names_url = name_cols(crop)
        dcs_series, dcs_first, dcs_latest, dcs_items = dcs_cols([dcs_name])
        records.append(
            {
                "display_name": dcs_name, "internal_name": "", "entry_type": "DCS weekly retail series only", "crop": crop or "",
                "sinhala_name": sinhala, "tamil_name": tamil, "names_source_url": names_url, "dataset_aliases": f"DCS: {dcs_items}",
                "harti_markets": "", "harti_dambulla_first": "", "harti_dambulla_latest": "", "harti_dambulla_days": 0,
                "dcs_retail_series": dcs_series, "dcs_first_week": dcs_first, "dcs_latest_week": dcs_latest,
                "harvest_reference_rows": count, "harvest_source_urls": harvest_urls,
                "price_prediction_supported": False,
                "price_prediction_note": "Retail weekly averages only; the forecaster predicts HARTI Dambulla wholesale prices.",
                "harvest_timing_supported": harvest_ok, "harvest_timing_note": harvest_note,
            }
        )
    covered = {r["crop"] for r in records if r["crop"]}
    for crop in names.index:
        if crop in covered:
            continue
        count, harvest_urls, harvest_ok, harvest_note = harvest(crop)
        sinhala, tamil, names_url = name_cols(crop)
        records.append(
            {
                "display_name": crop, "internal_name": "", "entry_type": "DOA HORDI crop only", "crop": crop,
                "sinhala_name": sinhala, "tamil_name": tamil, "names_source_url": names_url, "dataset_aliases": "",
                "harti_markets": "", "harti_dambulla_first": "", "harti_dambulla_latest": "", "harti_dambulla_days": 0,
                "dcs_retail_series": "", "dcs_first_week": "", "dcs_latest_week": "",
                "harvest_reference_rows": count, "harvest_source_urls": harvest_urls,
                "price_prediction_supported": False,
                "price_prediction_note": "No price series in the HARTI daily or DCS weekly bulletins.",
                "harvest_timing_supported": harvest_ok, "harvest_timing_note": harvest_note,
            }
        )
    catalogue = pd.DataFrame(records).sort_values(["entry_type", "display_name"], ignore_index=True)
    catalogue.insert(0, "snapshot_date", str(today.date()))
    catalogue.insert(1, "latest_harti_data", str(latest_data.date()))
    catalogue.to_csv(REFERENCE / "vegetable_catalogue.csv", index=False, encoding="utf-8")
    print(catalogue.groupby("entry_type").size().to_string())
    print("price prediction supported:", int(catalogue["price_prediction_supported"].sum()))
    print("harvest timing supported:", int(catalogue["harvest_timing_supported"].sum()))


if __name__ == "__main__":
    main()
