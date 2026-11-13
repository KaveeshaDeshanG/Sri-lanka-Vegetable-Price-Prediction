"""Regression tests: one for each defect found in the project audit, so none can return."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.forecasting import forecast_future_price, max_supported_forecast_date
from app.services import (
    ARCHIVE_MIN_AGE_DAYS,
    HistoryInputError,
    build_live_prediction_input,
    merge_harti_live_price,
    prepare_uploaded_history,
)
from src.data_collection.harti_live_client import (
    HartiLiveClient,
    HartiLiveSnapshot,
    parse_price_range,
    parse_reporting_date,
    resolve_vegetable,
)
from src.data_collection.live_fuel_price_client import CeypetcoLiveFuelPriceClient
from src.data_collection.live_weather_client import LiveWeatherError, OpenMeteoWeatherClient
from src.feature_engineering.categorical_features import CategoricalEncoder
from src.prediction.pipeline import PredictionPipeline
from src.prediction.predictor import InferenceArtifact, Predictor


def _config(tmp_path) -> str:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "weather_api:\n"
        "  enabled: true\n"
        "  latitude: 7.8567\n"
        "  longitude: 80.6517\n"
        '  timezone: "Asia/Colombo"\n'
        "  timeout_seconds: 10\n"
        "paths:\n"
        f'  crop_production_regions_mapping: "{(tmp_path / "none.yaml").as_posix()}"\n',
        encoding="utf-8",
    )
    return str(config_path)


def _weather(dates) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(list(dates)), "rainfall_mm": 9.0, "temperature_c": 31.0, "humidity_pct": 85.0})


def _rows(vegetable, dates) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "vegetable": vegetable,
            "market": "Dambulla",
            "source": "HARTI",
            "price": 100.0,
            "rainfall_mm": 5.0,
            "temperature_c": 28.0,
            "humidity_pct": 70.0,
        }
    )


# C3: PDF extraction -------------------------------------------------------------------


def test_real_bulletin_extraction_reads_date_labels_and_prices():
    with open("tests/fixtures/harti_sample_bulletin.pdf", "rb") as file:
        pdf_bytes = file.read()
    client = HartiLiveClient()
    df = client._parse_table_rows(client._extract_vegetable_table(pdf_bytes))

    assert (df["date"] == pd.Timestamp("2026-08-18")).all()
    # Labels whose spaces Table.extract() strips must still be recognized.
    assert {"Bitter Gourd", "Ladies Fingers", "Green Chillies", "Snake Gourd"} <= set(df["vegetable"])
    assert df.loc[df["vegetable"] == "Beans", "price"].iloc[0] == 375.0  # "300 - 450" in the PDF


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2026.11.06", "2026-11-06"),
        ("2024-04-25", "2024-04-25"),
        ("10/9/2024", "2024-09-10"),  # day first
        ("28/07/2024", "2024-07-28"),
        ("not-a-date", None),
        (None, None),
    ],
)
def test_parse_reporting_date_accepts_every_format_harti_has_used(text, expected):
    result = parse_reporting_date(text)
    assert result == (pd.Timestamp(expected) if expected else None)


def test_price_range_with_a_lost_dash_is_still_read():
    assert parse_price_range("200 250") == (200.0, 250.0)


def test_label_spacing_variants_resolve_to_the_same_vegetable():
    assert resolve_vegetable("Beet Root(N'Eliya)") == "Beetroot (Nuwara Eliya)"
    assert resolve_vegetable("Bitter Gourd") == resolve_vegetable("BitterGourd") == "Bitter Gourd"


# M6: archive years; M5: repeated downloads --------------------------------------------


@patch("src.data_collection.harti_live_client.requests.get")
def test_archive_reads_every_year_table(mock_get):
    html = """
    <table><tr><td>2026-01-02</td><td>English</td><td><a href="a.pdf">x</a></td></tr></table>
    <table><tr><td>2025-01-02</td><td>English</td><td><a href="b.pdf">x</a></td></tr></table>
    """
    mock_get.return_value = MagicMock(status_code=200, text=html)
    entries = HartiLiveClient()._fetch_archive_entries()
    assert sorted(entry.date.year for entry in entries) == [2025, 2026]


@patch("src.data_collection.harti_live_client.HartiLiveClient._parse_table_rows")
@patch("src.data_collection.harti_live_client.HartiLiveClient._extract_vegetable_table")
@patch("src.data_collection.harti_live_client.requests.get")
def test_a_past_bulletin_is_only_downloaded_once_per_client(mock_get, _extract, mock_parse):
    html = '<table><tr><td>2026-01-02</td><td>English</td><td><a href="a.pdf">x</a></td></tr></table>'
    mock_get.return_value = MagicMock(status_code=200, text=html, content=b"%PDF")
    mock_parse.return_value = pd.DataFrame(
        {"date": [pd.Timestamp("2026-01-02")], "vegetable": ["Carrot"], "market": ["Dambulla"], "price": [100.0], "source": ["HARTI_LIVE"]}
    )
    client = HartiLiveClient()
    client.fetch_prices_for_date("2026-01-02")
    client.fetch_prices_for_date("2026-01-02")
    assert mock_get.call_count == 2  # one archive page + one PDF, not four requests


# H1: weather coverage per vegetable ---------------------------------------------------


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_another_vegetables_weather_does_not_count_as_coverage(mock_for_coordinates, tmp_path):
    today = pd.Timestamp.now().normalize()
    required = today - pd.Timedelta(days=3)
    target = required + pd.Timedelta(days=14)
    days = pd.date_range(end=today, periods=30)
    history = pd.concat([_rows("Tomato", days), _rows("Carrot", days.drop(required))], ignore_index=True)
    mock_for_coordinates.return_value.fetch_recent_daily_weather.return_value = _weather([required])

    result = build_live_prediction_input(history, target, "Carrot", "Dambulla", _config(tmp_path))

    assert mock_for_coordinates.return_value.fetch_recent_daily_weather.called
    context = result[(result["date"] == required) & (result["vegetable"] == "Carrot")]
    assert context["rainfall_mm"].iloc[0] == 9.0


# H2: old dates come from the archive --------------------------------------------------


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_an_old_required_date_uses_the_archive_not_the_forecast_endpoint(mock_for_coordinates, tmp_path):
    today = pd.Timestamp.now().normalize()
    required = today - pd.Timedelta(days=60)  # the real forecast endpoint returns nulls this far back
    target = required + pd.Timedelta(days=14)
    history = _rows("Carrot", pd.date_range(end=required - pd.Timedelta(days=1), periods=5))
    client = mock_for_coordinates.return_value
    client.fetch_historical_daily_weather.return_value = _weather([required])

    result = build_live_prediction_input(history, target, "Carrot", "Dambulla", _config(tmp_path))

    client.fetch_recent_daily_weather.assert_not_called()
    assert (result["source"] == "ARCHIVE_WEATHER").sum() == 1


def test_archive_threshold_stays_within_the_forecast_endpoints_reliable_window():
    assert 2 < ARCHIVE_MIN_AGE_DAYS < 56


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_saved_regional_weather_is_used_before_any_network_request(mock_for_coordinates, tmp_path):
    required = pd.Timestamp("2026-10-25")
    weather_file = tmp_path / "regions.csv"
    pd.DataFrame({"date": [required], "region": ["Dambulla"], "rainfall_mm": [7.0], "temperature_c": [26.0], "humidity_pct": [80.0]}).to_csv(weather_file, index=False)
    config_path = _config(tmp_path)
    with open(config_path, "a", encoding="utf-8") as file:
        file.write(f'  raw_files:\n    weather_data_by_region: "{weather_file.as_posix()}"\n')
    history = _rows("Tomato", pd.date_range("2026-10-26", periods=5))  # unmapped, so market weather applies

    result = build_live_prediction_input(history, required + pd.Timedelta(days=14), "Tomato", "Dambulla", config_path)

    mock_for_coordinates.assert_not_called()
    assert result.loc[result["source"] == "LOCAL_WEATHER", "rainfall_mm"].iloc[0] == 7.0


# M4: retry and caching ----------------------------------------------------------------


@patch("src.data_collection.live_weather_client.time.sleep")
@patch("src.data_collection.live_weather_client.requests.get")
def test_a_brief_service_overload_is_retried(mock_get, _sleep):
    overloaded = MagicMock(status_code=503, text="overloaded")
    ok = MagicMock(status_code=200)
    ok.json.return_value = {
        "daily": {"time": ["2026-11-01"], "precipitation_sum": [1.0], "temperature_2m_mean": [25.0], "relative_humidity_2m_mean": [80.0]}
    }
    mock_get.side_effect = [overloaded, ok]
    assert len(OpenMeteoWeatherClient(latitude=1, longitude=1).fetch_recent_daily_weather(0, 1)) == 1


@patch("src.data_collection.live_weather_client.time.sleep")
@patch("src.data_collection.live_weather_client.requests.get")
def test_a_lasting_failure_still_raises_after_retrying(mock_get, _sleep):
    mock_get.return_value = MagicMock(status_code=503, text="overloaded")
    with pytest.raises(LiveWeatherError, match="HTTP 503"):
        OpenMeteoWeatherClient(latitude=1, longitude=1).fetch_recent_daily_weather(0, 1)
    assert mock_get.call_count == 3


@patch("src.data_collection.live_weather_client.requests.get")
def test_one_null_day_is_dropped_without_losing_the_rest_of_the_window(mock_get):
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "daily": {
            "time": ["2026-11-21", "2026-11-22"],
            "precipitation_sum": [1.0, None],
            "temperature_2m_mean": [25.0, None],
            "relative_humidity_2m_mean": [80.0, None],
        }
    }
    mock_get.return_value = response
    weather = OpenMeteoWeatherClient(latitude=1, longitude=1).fetch_recent_daily_weather(0, 2)
    assert list(weather["date"]) == [pd.Timestamp("2026-11-21")]


@patch("app.services.OpenMeteoWeatherClient.for_coordinates")
def test_weather_cache_serves_later_days_without_another_request(mock_for_coordinates, tmp_path):
    today = pd.Timestamp.now().normalize()
    history = _rows("Carrot", pd.date_range(end=today, periods=5))
    client = mock_for_coordinates.return_value
    client.fetch_recent_daily_weather.return_value = _weather(pd.date_range(today, periods=16))
    cache: dict = {}
    config_path = _config(tmp_path)

    for offset in (1, 2, 3):
        build_live_prediction_input(history, today + pd.Timedelta(days=14 + offset), "Carrot", "Dambulla", config_path, weather_cache=cache)

    assert client.fetch_recent_daily_weather.call_count == 1


def test_furthest_date_hint_reflects_the_seasonal_endpoint():
    today = pd.Timestamp.now().normalize()
    assert max_supported_forecast_date(today) == today + pd.Timedelta(days=45 + 14)


# M2: CEYPETCO's JavaScript page -------------------------------------------------------


@patch("src.data_collection.live_fuel_price_client.requests.get")
def test_fuel_prices_come_from_the_public_api(mock_get):
    api = MagicMock(status_code=200)
    api.json.return_value = {
        "data": [
            {"product": "Lanka Auto Diesel", "price": 397, "effectiveDate": "2026-11-01T00:00:00.000Z"},
            {"product": "Lanka Petrol 92 Octane", "price": 424, "effectiveDate": "2026-11-01T00:00:00.000Z"},
        ]
    }
    mock_get.return_value = api
    snapshot = CeypetcoLiveFuelPriceClient().fetch_latest_prices()
    assert snapshot.prices == {"diesel_price": 397.0, "petrol_price": 424.0}
    assert snapshot.effective_dates["diesel_price"] == pd.Timestamp("2026-11-01")


@patch("src.data_collection.live_fuel_price_client.requests.get")
def test_an_empty_price_api_falls_back_to_the_html_page(mock_get):
    empty_api = MagicMock(status_code=200)
    empty_api.json.return_value = {"data": []}
    with open("tests/fixtures/ceypetco_sample_page.html", encoding="utf-8") as file:
        page = MagicMock(status_code=200, text=file.read())
    mock_get.side_effect = [empty_api, page]
    assert CeypetcoLiveFuelPriceClient().fetch_latest_prices().prices["diesel_price"] == 382.0


# M7: Eggplant / Brinjal; duplicate live rows ------------------------------------------


def _snapshot(vegetable, date="2026-11-06"):
    data = pd.DataFrame({"date": [pd.Timestamp(date)], "vegetable": [vegetable], "market": ["Dambulla"], "price": [150.0], "source": ["HARTI_LIVE"]})
    return HartiLiveSnapshot(data=data, bulletin_date=pd.Timestamp(date), retrieved_at=pd.Timestamp.now(), source_url="x")


def test_brinjal_finds_its_eggplant_row_in_a_live_bulletin():
    history = _rows("Brinjal", pd.date_range("2026-11-01", "2026-11-05"))
    merged = merge_harti_live_price(history, _snapshot("Eggplant"), "Brinjal", "Dambulla")
    assert merged.iloc[-1]["vegetable"] == "Brinjal" and merged.iloc[-1]["price"] == 150.0


def test_a_bulletin_the_history_already_contains_is_not_added_twice():
    history = _rows("Carrot", pd.date_range("2026-11-01", "2026-11-06"))
    assert len(merge_harti_live_price(history, _snapshot("Carrot"), "Carrot", "Dambulla")) == len(history)


# M1, M10: uploaded data ---------------------------------------------------------------


@pytest.mark.parametrize(
    "frame,message",
    [
        (pd.DataFrame(columns=["date", "vegetable", "market", "price"]), "no data rows"),
        (pd.DataFrame({"date": ["x"], "vegetable": ["Carrot"], "market": ["Dambulla"], "price": [1]}), "No date"),
        (pd.DataFrame({"date": ["2026-01-01"], "vegetable": [None], "market": ["Dambulla"], "price": [1]}), "valid date, vegetable"),
        (pd.DataFrame({"date": ["2026-01-01"], "vegetable": ["Carrot"], "market": ["Dambulla"], "price": [None]}), "valid price"),
        (pd.DataFrame({"date": ["2026-01-01"], "vegetable": ["Carrot"], "market": ["Dambulla"]}), "price"),
    ],
)
def test_unusable_uploads_give_a_clear_error(frame, message):
    with pytest.raises(HistoryInputError, match=message):
        prepare_uploaded_history(frame)


def test_upload_is_normalized_the_same_way_as_training_data():
    frame = pd.DataFrame({"date": ["2026-01-02", "2026-01-01"], "vegetable": [" carrot ", "EGGPLANT"], "market": ["dambulla", "Dambulla"], "price": ["10", "20"]})
    cleaned, notes = prepare_uploaded_history(frame)
    assert list(cleaned["vegetable"]) == ["Brinjal", "Carrot"]
    assert set(cleaned["market"]) == {"Dambulla"}
    assert cleaned["rainfall_mm"].isna().all()  # missing weather can be fetched live
    assert notes == []


def test_upload_reports_rows_it_dropped():
    frame = pd.DataFrame({"date": ["2026-01-01", "bad"], "vegetable": ["Carrot", "Carrot"], "market": ["Dambulla", "Dambulla"], "price": [10, 20]})
    cleaned, notes = prepare_uploaded_history(frame)
    assert len(cleaned) == 1 and "1 row" in notes[0]


# M3: honest source labels -------------------------------------------------------------


def test_forecast_from_synthetic_sample_is_not_labelled_as_harti(tmp_path):
    days = pd.date_range("2024-01-01", periods=40)
    history = _rows("Carrot", days).assign(source="SAMPLE")
    pipeline = MagicMock()
    pipeline.predict.return_value = pd.DataFrame({"predicted_price": [100.0]})
    result = forecast_future_price(history, "Carrot", "Dambulla", days[-1] + pd.Timedelta(days=1), pipeline, _config(tmp_path))
    assert not any("HARTI" in source for source in result.data_sources)
    assert any("synthetic" in source for source in result.data_sources)


# M9, M10: unseen categories and name normalization ------------------------------------


def _predictor() -> Predictor:
    encoder = CategoricalEncoder().fit(pd.DataFrame({"vegetable": ["Carrot"], "market": ["Dambulla"]}))
    return Predictor(MagicMock(), InferenceArtifact(encoder, ["vegetable_Carrot", "market_Dambulla"]))


def test_predictor_reports_categories_training_never_saw():
    predictor = _predictor()
    assert predictor.unseen_categories(vegetable="Carrot", market="Dambulla") == []
    assert predictor.unseen_categories(vegetable="Drumstick", market="Pettah") == ["Drumstick", "Pettah"]


def test_requested_names_are_normalized_like_the_data():
    pipeline = PredictionPipeline(predictor=_predictor())
    assert pipeline._normalize_identity(" carrot", "dambulla dec") == ("Carrot", "Dambulla")
    assert pipeline._normalize_identity("Eggplant", "Dambulla") == ("Brinjal", "Dambulla")
