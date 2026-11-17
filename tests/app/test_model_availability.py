"""The dashboard must say when a dataset or model is unusable, and never quietly use another model."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.services import (
    HistoryInputError,
    _local_weather_reading,
    break_long_gaps,
    forecaster_problems,
    historical_data_available,
    historical_dataset_problems,
    missing_historical_datasets,
    prepare_uploaded_history,
)
from config.config_loader import ConfigLoader

APP = Path(__file__).resolve().parents[2] / "app"
PREDICT_PAGE = str(APP / "pages" / "1_Predict.py")
PROBLEM = "`data/raw/harti_all_markets.csv` is missing."


def _project(tmp_path, weather=True, panel=True, model=True) -> str:
    """A config over temporary files, with each file optionally left out."""
    weather_file, panel_file = tmp_path / "weather.csv", tmp_path / "panel.csv"
    if weather:
        weather_file.write_text("date,region,rainfall_mm,temperature_c,humidity_pct\n2025-01-01,Dambulla,1,2,3\n", encoding="utf-8")
    if panel:
        panel_file.write_text("published,market,vegetable,price\n2025-01-01,Kandy,Tomato,100\n", encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    if model:
        (models / "direct_forecaster.joblib").write_bytes(b"x")
    config = tmp_path / "config.yaml"
    config.write_text(
        f'paths:\n  models_dir: "{models.as_posix()}"\n  raw_files:\n'
        f'    weather_data_by_region: "{weather_file.as_posix()}"\n'
        f'    harti_all_markets: "{panel_file.as_posix()}"\n',
        encoding="utf-8",
    )
    return str(config)


def test_no_problems_when_every_file_is_present(tmp_path):
    assert forecaster_problems(_project(tmp_path)) == []


@pytest.mark.parametrize(
    ("missing", "named"),
    [({"weather": False}, "weather.csv"), ({"panel": False}, "panel.csv"), ({"model": False}, "direct_forecaster.joblib")],
)
def test_each_missing_file_is_named(tmp_path, missing, named):
    problems = forecaster_problems(_project(tmp_path, **missing))
    assert len(problems) == 1
    assert named in problems[0]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("a,b\n1,2\n", "missing required column"),
        ("published,market,vegetable,price\n", "no data rows"),
        ("published,market,vegetable,price\n2025-01-01,Kandy,Tomato,abc\n", "non-numeric"),
        ("", "could not be read"),
        (b"\xff\xfe\x00garbage\n\xff", "could not be read"),
    ],
)
def test_a_damaged_dataset_is_reported_not_used(tmp_path, content, expected):
    config = _project(tmp_path)
    panel = tmp_path / "panel.csv"
    panel.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    problems = forecaster_problems(config)
    assert len(problems) == 1
    assert expected in problems[0]


def test_missing_datasets_are_listed_individually(tmp_path):
    present = tmp_path / "prices.csv"
    present.write_text("x\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f'paths:\n  raw_files:\n    harti_prices: "{present.as_posix()}"\n'
        f'    weather_data: "{(tmp_path / "w.csv").as_posix()}"\n    fuel_prices: "{(tmp_path / "f.csv").as_posix()}"\n',
        encoding="utf-8",
    )
    missing = missing_historical_datasets(str(config))
    assert [Path(path).name for path in missing] == ["w.csv", "f.csv"]
    assert not historical_data_available(str(config))


def test_a_damaged_historical_dataset_is_reported(tmp_path):
    good = tmp_path / "weather.csv"
    good.write_text("date,rainfall_mm,temperature_c,humidity_pct\n2025-01-01,1,2,3\n", encoding="utf-8")
    bad = tmp_path / "prices.csv"
    bad.write_text("this is not valid content\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f'paths:\n  raw_files:\n    harti_prices: "{bad.as_posix()}"\n    weather_data: "{good.as_posix()}"\n'
        f'    fuel_prices: "{(tmp_path / "none.csv").as_posix()}"\n',
        encoding="utf-8",
    )
    problems = historical_dataset_problems(str(config))
    assert len(problems) == 2
    assert "missing required column" in problems[0]
    assert "none.csv" in problems[1]


def test_home_reports_a_damaged_price_file_instead_of_saying_the_forecaster_is_ready():
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=60)
    with patch("app.services.historical_dataset_problems", return_value=["`data/raw/harti_prices.csv` could not be read."]):
        at.run()
    assert not at.exception
    assert "harti_prices.csv" in _text(at.error)
    assert not at.info


@pytest.mark.parametrize("page", ["1_Predict.py", "3_Price_History.py"])
def test_a_damaged_regional_weather_file_gives_a_clear_error_not_a_crash(page):
    """Found in browser testing: a header-only file raised an unhandled TypeError on both pages."""
    st.cache_data.clear()
    at = AppTest.from_file(str(APP / "pages" / page), default_timeout=90)
    with patch("app.services.apply_production_region_weather", side_effect=TypeError("Invalid value")):
        at.run()
    st.cache_data.clear()
    assert not at.exception
    assert "weather_data_by_region.csv" in _text(at.error)


def _upload(**columns) -> pd.DataFrame:
    base = {"date": ["2026-01-01", "2026-01-02"], "vegetable": ["Carrot", "Carrot"], "market": ["Dambulla", "Dambulla"], "price": [100.0, 110.0]}
    return pd.DataFrame({**base, **columns})


def test_exact_duplicate_upload_rows_are_dropped_and_reported():
    raw = pd.concat([_upload(), _upload()], ignore_index=True)
    cleaned, notes = prepare_uploaded_history(raw)
    assert len(cleaned) == 2
    assert any("2 exact duplicate" in note for note in notes)


def test_rows_differing_only_by_name_spacing_count_as_duplicates():
    raw = pd.concat([_upload(), _upload(vegetable=["carrot ", "carrot "], market=[" dambulla", " dambulla"])], ignore_index=True)
    cleaned, _ = prepare_uploaded_history(raw)
    assert len(cleaned) == 2


def test_future_dated_upload_rows_are_flagged_and_kept():
    cleaned, notes = prepare_uploaded_history(_upload(date=["2026-01-01", "2099-01-01"]))
    assert len(cleaned) == 2
    assert any("dated after today" in note for note in notes)


def test_different_prices_for_the_same_day_are_flagged_and_kept():
    raw = pd.concat([_upload(), _upload(price=[105.0, 110.0])], ignore_index=True)
    cleaned, notes = prepare_uploaded_history(raw)
    assert len(cleaned) == 3
    assert any("1 date, vegetable and market combination(s) have more than one price" in note for note in notes)


def test_a_clean_upload_has_no_notes():
    assert prepare_uploaded_history(_upload())[1] == []


def test_a_missing_column_message_lists_what_was_found():
    with pytest.raises(HistoryInputError, match=r"Columns found: \['DATE', 'vegetable', 'market'\]"):
        prepare_uploaded_history(pd.DataFrame({"DATE": ["2026-01-01"], "vegetable": ["Carrot"], "market": ["Dambulla"]}))


def test_a_damaged_regional_weather_file_falls_through_to_the_provider(tmp_path):
    damaged = tmp_path / "weather.csv"
    damaged.write_text("not,the,right,columns\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(f'paths:\n  raw_files:\n    weather_data_by_region: "{damaged.as_posix()}"\n', encoding="utf-8")
    location = SimpleNamespace(is_production_region=False, region_name=None)
    assert _local_weather_reading(ConfigLoader(str(config)), location, "Dambulla", pd.Timestamp("2025-01-01")) is None


def test_chart_line_breaks_only_across_long_gaps():
    dates = pd.to_datetime(["2025-01-01", "2025-01-03", "2025-02-01", "2025-02-02"])
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=dates, name="price")
    result = break_long_gaps(series, max_gap_days=7)
    assert result.isna().sum() == 1
    assert pd.isna(result.loc["2025-01-31"])
    assert result.dropna().equals(series)


def _labels(at) -> list[str]:
    return [button.label for button in at.button]


def _run_predict(problems):
    at = AppTest.from_file(PREDICT_PAGE, default_timeout=90)
    with patch("app.services.forecaster_problems", return_value=problems):
        at.run()
    return at


def _text(elements) -> str:
    return " ".join(element.value for element in elements)


def test_real_data_uses_the_forecaster_and_says_so():
    at = _run_predict([])
    assert not at.exception
    # Latest Real Data Prediction names its model in the prediction section (since 2026-11-15).
    assert "**Model:** Ridge direct forecaster" in _text(at.caption)
    assert not at.checkbox

    next(button for button in at.button if button.label == "Predict").click()
    with patch("app.services.forecaster_problems", return_value=[]):
        at.run()
    assert not at.exception
    assert "Model used:** Ridge direct forecaster" in _text(at.caption)


def test_an_unavailable_forecaster_stops_rather_than_switching_model():
    at = _run_predict([PROBLEM])
    assert not at.exception
    assert PROBLEM in _text(at.error)
    assert "Predict" not in _labels(at)
    assert "Earlier comparison model loaded" not in _text(at.success)


def test_the_earlier_model_is_used_only_when_explicitly_chosen():
    at = _run_predict([PROBLEM])
    at.checkbox(key="use_earlier_model_for_real_data").check()
    with patch("app.services.forecaster_problems", return_value=[PROBLEM]):
        at.run()
    assert not at.exception
    assert PROBLEM in _text(at.error)
    assert "You chose the earlier comparison model" in _text(at.warning)

    next(button for button in at.button if button.label == "Predict").click()
    with patch("app.services.forecaster_problems", return_value=[PROBLEM]):
        at.run()
    assert not at.exception
    assert "Model used:** Earlier comparison model" in _text(at.caption)


def test_an_unreadable_forecaster_file_is_reported_and_stops():
    st.cache_resource.clear()
    with patch("src.models.direct_forecaster.DirectForecaster.load", side_effect=IndexError("pop from empty list")):
        at = _run_predict([])
    st.cache_resource.clear()
    assert not at.exception
    assert "may be damaged" in _text(at.error)
    assert "Predict" not in _labels(at)


def test_home_reports_an_unusable_forecaster_instead_of_claiming_a_real_model():
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=60)
    with patch("app.services.forecaster_problems", return_value=[PROBLEM]):
        at.run()
    assert not at.exception
    assert PROBLEM in _text(at.error)
    assert not at.info


def test_home_describes_the_ready_forecaster():
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=60).run()
    assert not at.exception
    assert "Ridge direct forecaster" in _text(at.info)
    assert "one Ridge model per day ahead" in _text(at.markdown)


@pytest.mark.parametrize(
    ("direct", "expected_kind", "expected_text"),
    [
        (None, "warning", "is missing"),
        ("{not json", "error", "could not be read"),
        ("{}", "error", "unexpected contents"),
    ],
)
def test_performance_page_explains_each_unusable_report(tmp_path, direct, expected_kind, expected_text):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    if direct is not None:
        (metrics / "direct_forecaster_metrics.json").write_text(direct, encoding="utf-8")
    real_get = ConfigLoader.get

    def get(self, key, default=None):
        return str(metrics) if key == "paths.metrics_dir" else real_get(self, key, default)

    at = AppTest.from_file(str(APP / "pages" / "2_Model_Performance.py"), default_timeout=60)
    with patch.object(ConfigLoader, "get", get):
        at.run()
    assert not at.exception
    messages = getattr(at, expected_kind)
    assert any("direct_forecaster_metrics.json" in m.value and expected_text in m.value for m in messages)
    # The other reports are absent here, and each says so rather than ending the page early.
    assert any("forecast_backtest.json" in m.value for m in at.warning)
