"""DCS prints 0.00 for an item it couldn't price that week (e.g. Butter beans in 2023-03 week 3,
2024-11 week 4 and 2025-12 week 1). These tests check that the app reads such values as unavailable,
never as Rs 0, without changing the stored CSV or any genuine price."""

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from streamlit.testing.v1 import AppTest

from app import source_records as sr

DCS_CSV = Path("data/raw/dcs_weekly_retail_prices.csv")
HISTORY_PAGE = str(Path(__file__).resolve().parents[2] / "app" / "pages" / "3_Price_History.py")
BUTTER_BEANS_UNAVAILABLE = ["2023-03 week 3", "2024-11 week 4", "2025-12 week 1"]


def test_zero_and_negative_values_become_missing_and_positive_prices_are_untouched():
    stored = pd.DataFrame(
        {"period": ["a", "b", "c", "d"], "price": [0.0, 0.5, 731.03, -1.0], "range_low": [0.0, 0.5, 640.0, 0.0], "range_high": [0.0, 0.6, 800.0, 5.0]}
    )
    shown = sr.dcs_unavailable_as_missing(stored)
    assert shown["price"].isna().tolist() == [True, False, False, True]
    assert shown["price_unavailable"].tolist() == [True, False, False, True]
    assert shown.loc[1, "price"] == 0.5 and shown.loc[2, "price"] == 731.03  # small or large, real prices stay
    assert shown.loc[2, ["range_low", "range_high"]].tolist() == [640.0, 800.0]
    assert pd.isna(shown.loc[3, "range_low"]) and shown.loc[3, "range_high"] == 5.0
    assert stored["price"].tolist() == [0.0, 0.5, 731.03, -1.0]  # the input is not modified


@pytest.mark.skipif(not DCS_CSV.exists(), reason="DCS dataset not built")
def test_only_the_three_butter_bean_weeks_are_unavailable_and_the_csv_is_unchanged():
    before = hashlib.sha256(DCS_CSV.read_bytes()).hexdigest()
    stored = pd.read_csv(DCS_CSV)
    shown = sr.load_dcs_weekly()
    assert hashlib.sha256(DCS_CSV.read_bytes()).hexdigest() == before

    unavailable = shown[shown["price_unavailable"]]
    assert sorted(unavailable["period"]) == BUTTER_BEANS_UNAVAILABLE
    assert set(unavailable["vegetable"]) == {"Beans (Butter)"}
    # Every other price is exactly as stored.
    kept = ~shown["price_unavailable"]
    assert (shown.loc[kept, "price"] == stored.loc[kept, "price"]).all()
    assert len(shown) == len(stored)


@pytest.mark.skipif(not DCS_CSV.exists(), reason="DCS dataset not built")
def test_predict_page_records_show_unavailable_dcs_weeks_as_missing():
    records = sr.dcs_records()
    butter = records[records["Vegetable (as published)"] == "Beans - Butter"]
    zero_weeks = butter[butter["Observation date"].isin(BUTTER_BEANS_UNAVAILABLE)]
    assert len(zero_weeks) == 3
    assert zero_weeks[["Minimum", "Maximum", "Average / midpoint"]].isna().all().all()
    assert (records["Average / midpoint"].dropna() > 0).all()


def _chart_rows(at) -> pd.DataFrame:
    def walk(node, found):
        for child in getattr(node, "children", {}).values():
            if child.type == "vega_lite_chart":
                found.append(child)
            walk(child, found)
        return found

    charts = walk(at.main, [])
    assert charts, "no chart drawn"
    frames = [pa.ipc.open_stream(dataset.data.data).read_pandas() for dataset in charts[0].proto.datasets]
    return pd.concat(frames, ignore_index=True)


@pytest.mark.skipif(not DCS_CSV.exists(), reason="DCS dataset not built")
def test_price_history_does_not_plot_or_count_unavailable_dcs_weeks_as_zero():
    at = AppTest.from_file(HISTORY_PAGE, default_timeout=180).run()
    next(r for r in at.radio if r.label == "Data source").set_value("DCS weekly bulletins: retail, Colombo District")
    at.run()
    next(s for s in at.selectbox if s.label == "Vegetable").set_value("Beans (Butter)")
    at.run()
    assert not at.exception, at.exception

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Lowest"] != "Rs. 0"
    assert all(value not in {"Rs. 0", "Rs. 0.00"} for value in metrics.values())
    plotted = _chart_rows(at)
    assert len(plotted) > 0 and (plotted["price"] > 0).all()
    assert not plotted["period"].isin(BUTTER_BEANS_UNAVAILABLE).any()
    notice = " ".join(i.value for i in at.info)
    assert all(week in notice for week in BUTTER_BEANS_UNAVAILABLE) and "not as Rs 0" in notice
