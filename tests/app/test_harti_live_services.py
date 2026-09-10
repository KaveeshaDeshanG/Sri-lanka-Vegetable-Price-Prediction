"""Unit tests for app.services.load_real_market_history / fetch_harti_latest_snapshot /
merge_harti_live_price -- the "Latest HARTI Data" prediction option's service layer.

HartiLiveClient itself is mocked out here (its own network/parsing behavior is covered
by tests/data_collection/test_harti_live_client.py) -- these tests only check the
composition logic: config gating, row selection, and the fuel-price forward-fill.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from app.services import (
    IMPUTED_HISTORICAL_SOURCE_LABEL,
    backfill_harti_price_gap,
    backfill_internal_harti_gaps,
    build_prediction_input,
    fetch_harti_latest_snapshot,
    fetch_live_fuel_price,
    impute_missing_historical_prices,
    load_real_market_history,
    merge_harti_live_price,
)
from src.data_collection.harti_live_client import HartiLiveDataError, HartiLiveSnapshot
from src.data_collection.live_fuel_price_client import LiveFuelPriceError, LiveFuelPriceSnapshot
from src.prediction.pipeline import PredictionPipeline


def _fuel_snapshot(prices, effective_dates, retrieved_at="2026-08-21 09:00:00") -> LiveFuelPriceSnapshot:
    return LiveFuelPriceSnapshot(
        prices=prices,
        effective_dates={k: pd.Timestamp(v) for k, v in effective_dates.items()},
        retrieved_at=pd.Timestamp(retrieved_at),
        source_url="https://ceypetco.gov.lk/marketing-sales/",
    )


def _snapshot(rows, bulletin_date="2026-08-18", retrieved_at="2026-08-18 10:00:00") -> HartiLiveSnapshot:
    df = pd.DataFrame(rows)
    return HartiLiveSnapshot(
        data=df,
        bulletin_date=pd.Timestamp(bulletin_date),
        retrieved_at=pd.Timestamp(retrieved_at),
        source_url="https://www.harti.gov.lk/some.pdf",
    )


def _history(n_days=40, vegetable="Tomato", market="Dambulla", with_fuel=True):
    dates = pd.date_range("2026-06-01", periods=n_days, freq="D")
    data = {
        "date": dates,
        "vegetable": [vegetable] * n_days,
        "market": [market] * n_days,
        "source": ["HARTI"] * n_days,
        "price": [200.0 + i for i in range(n_days)],
        "rainfall_mm": [5.0] * n_days,
        "temperature_c": [28.0] * n_days,
        "humidity_pct": [70.0] * n_days,
    }
    if with_fuel:
        # fuel series deliberately stops 3 days before the price history's last date --
        # the realistic gap merge_harti_live_price's forward-fill must bridge.
        data["diesel_price"] = [363.0] * (n_days - 3) + [float("nan")] * 3
        data["petrol_price"] = [371.0] * (n_days - 3) + [float("nan")] * 3
    return pd.DataFrame(data)


# 1. fetch_harti_latest_snapshot -----------------------------------------------------


def test_fetch_harti_latest_snapshot_raises_when_disabled(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("harti:\n  latest_data_enabled: false\n", encoding="utf-8")

    with pytest.raises(HartiLiveDataError, match="disabled"):
        fetch_harti_latest_snapshot(str(config_path))


@patch("app.services.HartiLiveClient.from_config")
def test_fetch_harti_latest_snapshot_delegates_to_client_when_enabled(mock_from_config, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("harti:\n  latest_data_enabled: true\n", encoding="utf-8")
    expected = _snapshot([{"date": pd.Timestamp("2026-08-18"), "vegetable": "Carrot", "market": "Dambulla", "price": 175.0, "source": "HARTI_LIVE"}])
    mock_from_config.return_value.fetch_latest_prices.return_value = expected

    result = fetch_harti_latest_snapshot(str(config_path))

    assert result is expected
    mock_from_config.assert_called_once_with(str(config_path))


@patch("app.services.HartiLiveClient.from_config")
def test_fetch_harti_latest_snapshot_defaults_to_enabled(mock_from_config, tmp_path):
    """No harti section at all -- must not silently disable the feature."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project:\n  name: x\n", encoding="utf-8")
    mock_from_config.return_value.fetch_latest_prices.return_value = _snapshot(
        [{"date": pd.Timestamp("2026-08-18"), "vegetable": "Carrot", "market": "Dambulla", "price": 175.0, "source": "HARTI_LIVE"}]
    )

    fetch_harti_latest_snapshot(str(config_path))  # must not raise

    mock_from_config.return_value.fetch_latest_prices.assert_called_once()


# 1b. fetch_live_fuel_price ------------------------------------------------------------


def test_fetch_live_fuel_price_raises_when_disabled(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fuel_api:\n  enabled: false\n", encoding="utf-8")

    with pytest.raises(LiveFuelPriceError, match="disabled"):
        fetch_live_fuel_price(str(config_path))


@patch("app.services.CeypetcoLiveFuelPriceClient.from_config")
def test_fetch_live_fuel_price_delegates_to_client_when_enabled(mock_from_config, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fuel_api:\n  enabled: true\n", encoding="utf-8")
    expected = _fuel_snapshot({"diesel_price": 382.0}, {"diesel_price": "2026-06-29"})
    mock_from_config.return_value.fetch_latest_prices.return_value = expected

    result = fetch_live_fuel_price(str(config_path))

    assert result is expected
    mock_from_config.assert_called_once_with(str(config_path))


@patch("app.services.CeypetcoLiveFuelPriceClient.from_config")
def test_fetch_live_fuel_price_defaults_to_enabled(mock_from_config, tmp_path):
    """No fuel_api section at all -- must not silently disable the feature."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project:\n  name: x\n", encoding="utf-8")
    mock_from_config.return_value.fetch_latest_prices.return_value = _fuel_snapshot(
        {"diesel_price": 382.0}, {"diesel_price": "2026-06-29"}
    )

    fetch_live_fuel_price(str(config_path))  # must not raise

    mock_from_config.return_value.fetch_latest_prices.assert_called_once()


# 2. merge_harti_live_price -----------------------------------------------------------


def test_merge_harti_live_price_appends_the_matching_row():
    history = _history()
    snapshot = _snapshot(
        [
            {"date": pd.Timestamp("2026-08-18"), "vegetable": "Tomato", "market": "Dambulla", "price": 275.0, "source": "HARTI_LIVE"},
            {"date": pd.Timestamp("2026-08-18"), "vegetable": "Carrot", "market": "Dambulla", "price": 175.0, "source": "HARTI_LIVE"},
        ]
    )

    result = merge_harti_live_price(history, snapshot, "Tomato", "Dambulla")

    assert len(result) == len(history) + 1
    new_row = result.iloc[-1]
    assert new_row["date"] == pd.Timestamp("2026-08-18")
    assert new_row["price"] == 275.0
    assert new_row["source"] == "HARTI_LIVE"


def test_merge_harti_live_price_forward_fills_fuel_price_gap():
    history = _history(with_fuel=True)  # fuel stops 3 days before price history's end
    snapshot = _snapshot(
        [{"date": pd.Timestamp("2026-08-18"), "vegetable": "Tomato", "market": "Dambulla", "price": 275.0, "source": "HARTI_LIVE"}]
    )

    result = merge_harti_live_price(history, snapshot, "Tomato", "Dambulla")

    new_row = result.iloc[-1]
    assert new_row["diesel_price"] == 363.0  # last known real value, carried forward
    assert new_row["petrol_price"] == 371.0


def test_merge_harti_live_price_prefers_live_fuel_price_when_it_covers_the_row_date():
    history = _history(with_fuel=True)  # last known real diesel/petrol: 363.0 / 371.0
    snapshot = _snapshot(
        [{"date": pd.Timestamp("2026-08-18"), "vegetable": "Tomato", "market": "Dambulla", "price": 275.0, "source": "HARTI_LIVE"}]
    )
    live_fuel = _fuel_snapshot(
        {"diesel_price": 382.0, "petrol_price": 414.0},
        {"diesel_price": "2026-06-29", "petrol_price": "2026-06-29"},  # effective well before 2026-08-18
    )

    result = merge_harti_live_price(history, snapshot, "Tomato", "Dambulla", live_fuel_snapshot=live_fuel)

    new_row = result.iloc[-1]
    assert new_row["diesel_price"] == 382.0  # live CEYPETCO reading, not the forward-filled 363.0
    assert new_row["petrol_price"] == 414.0


def test_merge_harti_live_price_falls_back_to_forward_fill_when_live_fuel_predates_row():
    history = _history(with_fuel=True)
    snapshot = _snapshot(
        [{"date": pd.Timestamp("2026-08-18"), "vegetable": "Tomato", "market": "Dambulla", "price": 275.0, "source": "HARTI_LIVE"}]
    )
    live_fuel = _fuel_snapshot(
        {"diesel_price": 382.0, "petrol_price": 414.0},
        {"diesel_price": "2026-09-01", "petrol_price": "2026-09-01"},  # effective AFTER this row's date
    )

    result = merge_harti_live_price(history, snapshot, "Tomato", "Dambulla", live_fuel_snapshot=live_fuel)

    new_row = result.iloc[-1]
    # using the live price here would apply a future revision to an earlier date -- must not happen
    assert new_row["diesel_price"] == 363.0
    assert new_row["petrol_price"] == 371.0


def test_merge_harti_live_price_fills_missing_weather_columns_with_nan():
    history = _history()
    snapshot = _snapshot(
        [{"date": pd.Timestamp("2026-08-18"), "vegetable": "Tomato", "market": "Dambulla", "price": 275.0, "source": "HARTI_LIVE"}]
    )

    result = merge_harti_live_price(history, snapshot, "Tomato", "Dambulla")

    new_row = result.iloc[-1]
    assert pd.isna(new_row["rainfall_mm"])
    assert pd.isna(new_row["temperature_c"])
    assert pd.isna(new_row["humidity_pct"])


def test_merge_harti_live_price_raises_for_unsupported_vegetable():
    history = _history()
    snapshot = _snapshot(
        [{"date": pd.Timestamp("2026-08-18"), "vegetable": "Carrot", "market": "Dambulla", "price": 175.0, "source": "HARTI_LIVE"}]
    )

    with pytest.raises(HartiLiveDataError, match="no 'Dambulla' price for 'Tomato'"):
        merge_harti_live_price(history, snapshot, "Tomato", "Dambulla")


def test_merge_harti_live_price_error_lists_which_vegetables_were_available():
    history = _history()
    snapshot = _snapshot(
        [{"date": pd.Timestamp("2026-08-18"), "vegetable": "Carrot", "market": "Dambulla", "price": 175.0, "source": "HARTI_LIVE"}]
    )

    with pytest.raises(HartiLiveDataError, match="Carrot"):
        merge_harti_live_price(history, snapshot, "Tomato", "Dambulla")


def test_merge_harti_live_price_does_not_mutate_history():
    history = _history()
    original = history.copy()
    snapshot = _snapshot(
        [{"date": pd.Timestamp("2026-08-18"), "vegetable": "Tomato", "market": "Dambulla", "price": 275.0, "source": "HARTI_LIVE"}]
    )

    merge_harti_live_price(history, snapshot, "Tomato", "Dambulla")

    pd.testing.assert_frame_equal(history, original)


# 2b. backfill_harti_price_gap ---------------------------------------------------------


@patch("app.services.HartiLiveClient.from_config")
def test_backfill_harti_price_gap_fills_missing_days_with_real_bulletins(mock_from_config):
    history = _history(n_days=10, with_fuel=False)  # 2026-06-01 .. 2026-06-10
    last_known = history["date"].max()
    bulletin_date = last_known + pd.Timedelta(days=4)  # gap: 06-11, 06-12, 06-13

    def fetch(day):
        day = pd.Timestamp(day)
        return _snapshot(
            [{"date": day, "vegetable": "Tomato", "market": "Dambulla", "price": 300.0 + day.day,
              "source": "HARTI_LIVE"}],
            bulletin_date=str(day.date()),
        )

    mock_from_config.return_value.fetch_prices_for_date.side_effect = fetch
    snapshot = _snapshot(
        [{"date": bulletin_date, "vegetable": "Tomato", "market": "Dambulla", "price": 304.0, "source": "HARTI_LIVE"}],
        bulletin_date=str(bulletin_date.date()),
    )

    result = backfill_harti_price_gap(history, snapshot, "Tomato", "Dambulla")

    filled = result.loc[result["date"] > last_known].sort_values("date")
    assert filled["date"].tolist() == [
        pd.Timestamp("2026-06-11"), pd.Timestamp("2026-06-12"), pd.Timestamp("2026-06-13")
    ]
    assert filled["price"].tolist() == [311.0, 312.0, 313.0]
    assert filled["source"].tolist() == ["HARTI_LIVE"] * 3
    # the live bulletin's own date is NOT part of this function's job -- merge_harti_live_price adds it
    assert bulletin_date not in result["date"].tolist()


@patch("app.services.HartiLiveClient.from_config")
def test_backfill_harti_price_gap_applies_live_fuel_price_to_backfilled_rows(mock_from_config):
    history = _history(n_days=10, with_fuel=True)  # 2026-06-01 .. 2026-06-10, fuel stops 06-08
    last_known = history["date"].max()
    bulletin_date = last_known + pd.Timedelta(days=4)  # gap: 06-11, 06-12, 06-13

    def fetch(day):
        day = pd.Timestamp(day)
        return _snapshot(
            [{"date": day, "vegetable": "Tomato", "market": "Dambulla", "price": 300.0 + day.day,
              "source": "HARTI_LIVE"}],
            bulletin_date=str(day.date()),
        )

    mock_from_config.return_value.fetch_prices_for_date.side_effect = fetch
    snapshot = _snapshot(
        [{"date": bulletin_date, "vegetable": "Tomato", "market": "Dambulla", "price": 304.0, "source": "HARTI_LIVE"}],
        bulletin_date=str(bulletin_date.date()),
    )
    live_fuel = _fuel_snapshot(
        {"diesel_price": 382.0, "petrol_price": 414.0},
        {"diesel_price": "2026-06-01", "petrol_price": "2026-06-01"},  # effective before every gap day
    )

    result = backfill_harti_price_gap(
        history, snapshot, "Tomato", "Dambulla", live_fuel_snapshot=live_fuel
    )

    filled = result.loc[result["date"] > last_known].sort_values("date")
    assert filled["diesel_price"].tolist() == [382.0, 382.0, 382.0]
    assert filled["petrol_price"].tolist() == [414.0, 414.0, 414.0]


@patch("app.services.HartiLiveClient.from_config")
def test_backfill_harti_price_gap_skips_days_with_no_bulletin(mock_from_config):
    history = _history(n_days=5, with_fuel=False)  # ends 2026-06-05
    last_known = history["date"].max()
    bulletin_date = last_known + pd.Timedelta(days=3)  # gap: 06-06, 06-07

    def fetch(day):
        day = pd.Timestamp(day)
        if day == last_known + pd.Timedelta(days=1):
            raise HartiLiveDataError("HARTI did not publish a bulletin that day")
        return _snapshot(
            [{"date": day, "vegetable": "Tomato", "market": "Dambulla", "price": 500.0, "source": "HARTI_LIVE"}],
            bulletin_date=str(day.date()),
        )

    mock_from_config.return_value.fetch_prices_for_date.side_effect = fetch
    snapshot = _snapshot(
        [{"date": bulletin_date, "vegetable": "Tomato", "market": "Dambulla", "price": 501.0, "source": "HARTI_LIVE"}],
        bulletin_date=str(bulletin_date.date()),
    )

    result = backfill_harti_price_gap(history, snapshot, "Tomato", "Dambulla")

    filled_dates = sorted(result.loc[result["date"] > last_known, "date"].tolist())
    assert filled_dates == [last_known + pd.Timedelta(days=2)]  # day 1 skipped, never fabricated


@patch("app.services.HartiLiveClient.from_config")
def test_backfill_harti_price_gap_skips_days_with_no_reading_for_the_vegetable(mock_from_config):
    history = _history(n_days=5, with_fuel=False)
    last_known = history["date"].max()
    bulletin_date = last_known + pd.Timedelta(days=2)  # gap: one day

    mock_from_config.return_value.fetch_prices_for_date.return_value = _snapshot(
        [{"date": last_known + pd.Timedelta(days=1), "vegetable": "Carrot", "market": "Dambulla",
          "price": 175.0, "source": "HARTI_LIVE"}],
        bulletin_date=str((last_known + pd.Timedelta(days=1)).date()),
    )
    snapshot = _snapshot(
        [{"date": bulletin_date, "vegetable": "Tomato", "market": "Dambulla", "price": 501.0, "source": "HARTI_LIVE"}],
        bulletin_date=str(bulletin_date.date()),
    )

    result = backfill_harti_price_gap(history, snapshot, "Tomato", "Dambulla")

    pd.testing.assert_frame_equal(result, history)  # never fabricates Tomato's missing day


@patch("app.services.HartiLiveClient.from_config")
def test_backfill_harti_price_gap_noop_when_bulletin_is_the_very_next_day(mock_from_config):
    history = _history(n_days=5, with_fuel=False)
    last_known = history["date"].max()
    bulletin_date = last_known + pd.Timedelta(days=1)  # no gap at all
    snapshot = _snapshot(
        [{"date": bulletin_date, "vegetable": "Tomato", "market": "Dambulla", "price": 400.0, "source": "HARTI_LIVE"}],
        bulletin_date=str(bulletin_date.date()),
    )

    result = backfill_harti_price_gap(history, snapshot, "Tomato", "Dambulla")

    pd.testing.assert_frame_equal(result, history)
    mock_from_config.return_value.fetch_prices_for_date.assert_not_called()


@patch("app.services.HartiLiveClient.from_config")
def test_backfill_harti_price_gap_caps_lookback_to_max_configured_window(mock_from_config):
    history = _history(n_days=5, with_fuel=False)  # ends 2026-06-05
    last_known = history["date"].max()
    bulletin_date = last_known + pd.Timedelta(days=40)  # gap far larger than any configured window

    mock_from_config.return_value.fetch_prices_for_date.side_effect = HartiLiveDataError("no bulletin")
    snapshot = _snapshot(
        [{"date": bulletin_date, "vegetable": "Tomato", "market": "Dambulla", "price": 400.0, "source": "HARTI_LIVE"}],
        bulletin_date=str(bulletin_date.date()),
    )

    backfill_harti_price_gap(history, snapshot, "Tomato", "Dambulla")

    calls = [c.args[0] for c in mock_from_config.return_value.fetch_prices_for_date.call_args_list]
    # config/config.yaml's feature_engineering.{lag_periods_days,rolling_windows_days} max out at 30
    assert len(calls) == 30
    assert min(calls) == bulletin_date - pd.Timedelta(days=30)
    assert max(calls) == bulletin_date - pd.Timedelta(days=1)


# 2c. backfill_internal_harti_gaps ------------------------------------------------------


@patch("app.services.HartiLiveClient.from_config")
def test_backfill_internal_harti_gaps_fills_an_isolated_hole_in_the_trailing_window(mock_from_config):
    # dense 40-day history with ONE isolated missing day in the middle -- e.g. HARTI's
    # own bulletin never covered this market/vegetable that specific day.
    dates = pd.date_range("2026-06-01", periods=40, freq="D")
    missing_day = pd.Timestamp("2026-06-20")
    history = pd.DataFrame(
        {
            "date": [d for d in dates if d != missing_day],
            "vegetable": "Tomato",
            "market": "Dambulla",
            "source": "HARTI",
            "price": [200.0 + i for i, d in enumerate(dates) if d != missing_day],
        }
    )

    def fetch(day):
        day = pd.Timestamp(day)
        if day != missing_day:
            raise HartiLiveDataError(f"no bulletin for {day.date()}")
        return _snapshot(
            [{"date": day, "vegetable": "Tomato", "market": "Dambulla", "price": 999.0, "source": "HARTI_LIVE"}],
            bulletin_date=str(day.date()),
        )

    mock_from_config.return_value.fetch_prices_for_date.side_effect = fetch

    result = backfill_internal_harti_gaps(history, "Tomato", "Dambulla")

    filled = result.loc[result["date"] == missing_day]
    assert len(filled) == 1
    assert filled["price"].iloc[0] == 999.0
    assert filled["source"].iloc[0] == "HARTI_LIVE"


@patch("app.services.HartiLiveClient.from_config")
def test_backfill_internal_harti_gaps_skips_a_day_harti_never_published(mock_from_config):
    dates = pd.date_range("2026-06-01", periods=40, freq="D")
    missing_day = pd.Timestamp("2026-06-20")
    history = pd.DataFrame(
        {
            "date": [d for d in dates if d != missing_day],
            "vegetable": "Tomato",
            "market": "Dambulla",
            "source": "HARTI",
            "price": [200.0 + i for i, d in enumerate(dates) if d != missing_day],
        }
    )
    mock_from_config.return_value.fetch_prices_for_date.side_effect = HartiLiveDataError("no bulletin")

    result = backfill_internal_harti_gaps(history, "Tomato", "Dambulla")

    # never fabricated -- the day genuinely has no real bulletin, so it stays missing
    assert result.loc[result["date"] == missing_day].empty
    pd.testing.assert_frame_equal(result.reset_index(drop=True), history.reset_index(drop=True))


@patch("app.services.HartiLiveClient.from_config")
def test_backfill_internal_harti_gaps_noop_when_trailing_window_is_already_dense(mock_from_config):
    history = _history(n_days=40, with_fuel=False)  # dense, no gaps

    result = backfill_internal_harti_gaps(history, "Tomato", "Dambulla")

    pd.testing.assert_frame_equal(result, history)
    mock_from_config.return_value.fetch_prices_for_date.assert_not_called()


@patch("app.services.HartiLiveClient.from_config")
def test_backfill_internal_harti_gaps_only_checks_the_trailing_configured_window(mock_from_config):
    # a hole far outside the trailing max(lag+rolling)=30-day window must not be probed
    dates = pd.date_range("2026-01-01", periods=200, freq="D")
    old_missing_day = dates[10]  # ~190 days before the group's last date -- irrelevant
    history = pd.DataFrame(
        {
            "date": [d for d in dates if d != old_missing_day],
            "vegetable": "Tomato",
            "market": "Dambulla",
            "source": "HARTI",
            "price": [200.0] * (len(dates) - 1),
        }
    )
    mock_from_config.return_value.fetch_prices_for_date.side_effect = HartiLiveDataError("no bulletin")

    backfill_internal_harti_gaps(history, "Tomato", "Dambulla")

    called_days = [c.args[0] for c in mock_from_config.return_value.fetch_prices_for_date.call_args_list]
    assert pd.Timestamp(old_missing_day) not in called_days
    assert len(called_days) == 0  # the trailing 30-day window itself has no holes


# 2d. impute_missing_historical_prices --------------------------------------------------


def _history_with_gap(missing_day, n_days=40, vegetable="Tomato", market="Dambulla", start="2026-06-01"):
    dates = [d for d in pd.date_range(start, periods=n_days, freq="D") if d != pd.Timestamp(missing_day)]
    return pd.DataFrame(
        {
            "date": dates,
            "vegetable": vegetable,
            "market": market,
            "source": "HARTI",
            "price": [200.0 + i for i in range(len(dates))],
        }
    )


def test_impute_missing_historical_prices_linearly_interpolates_an_isolated_gap():
    missing_day = pd.Timestamp("2026-06-20")
    history = _history_with_gap(missing_day)
    before_price = history.loc[history["date"] == missing_day - pd.Timedelta(days=1), "price"].iloc[0]
    after_price = history.loc[history["date"] == missing_day + pd.Timedelta(days=1), "price"].iloc[0]

    result, imputed_dates = impute_missing_historical_prices(history, "Tomato", "Dambulla")

    assert imputed_dates == [missing_day]
    imputed_row = result.loc[result["date"] == missing_day]
    assert len(imputed_row) == 1
    assert imputed_row["source"].iloc[0] == IMPUTED_HISTORICAL_SOURCE_LABEL
    assert imputed_row["price"].iloc[0] == pytest.approx((before_price + after_price) / 2)


def test_impute_missing_historical_prices_carries_the_earlier_real_value_forward_at_the_leading_edge():
    # a group whose OWN history starts a few days into the trailing lookback window --
    # nothing real exists before its first day, so the leading gap must be filled by
    # carrying its first real value backward ("only next exists"), never invented.
    dates = pd.date_range("2026-06-05", periods=10, freq="D")  # starts mid-window
    history = pd.DataFrame(
        {
            "date": dates, "vegetable": "Tomato", "market": "Dambulla", "source": "HARTI",
            "price": [300.0 + i for i in range(len(dates))],
        }
    )
    first_real_price = history.loc[history["date"] == dates[0], "price"].iloc[0]

    result, imputed_dates = impute_missing_historical_prices(history, "Tomato", "Dambulla")

    assert pd.Timestamp("2026-06-01") in imputed_dates  # inside the default 30-day window, before dates[0]
    leading_row = result.loc[result["date"] == pd.Timestamp("2026-06-01")]
    assert leading_row["price"].iloc[0] == pytest.approx(first_real_price)
    assert leading_row["source"].iloc[0] == IMPUTED_HISTORICAL_SOURCE_LABEL


def test_impute_missing_historical_prices_never_overwrites_a_real_observation():
    missing_day = pd.Timestamp("2026-06-20")
    history = _history_with_gap(missing_day)
    real_snapshot = history.copy(deep=True)

    result, _ = impute_missing_historical_prices(history, "Tomato", "Dambulla")

    # every original real row survives, completely unchanged, at its original price
    merged_back = result.merge(real_snapshot, on=["date", "vegetable", "market"], suffixes=("", "_orig"))
    assert (merged_back["price"] == merged_back["price_orig"]).all()
    assert (merged_back["source"] == merged_back["source_orig"]).all()
    # and the original DataFrame object itself was never mutated
    pd.testing.assert_frame_equal(history, real_snapshot)


def test_impute_missing_historical_prices_is_runtime_only_never_written_to_disk():
    missing_day = pd.Timestamp("2026-06-20")
    history = _history_with_gap(missing_day)

    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        impute_missing_historical_prices(history, "Tomato", "Dambulla")
        mock_to_csv.assert_not_called()


def test_impute_missing_historical_prices_noop_when_nothing_is_missing():
    history = _history_with_gap(missing_day=pd.Timestamp("1999-01-01"))  # not actually missing anything real
    result, imputed_dates = impute_missing_historical_prices(history, "Tomato", "Dambulla")
    assert imputed_dates == []
    pd.testing.assert_frame_equal(result, history)


def test_backfill_internal_harti_gaps_then_impute_only_fills_what_archive_backfill_missed():
    """Priority ordering: a date real HARTI archive data covers must be filled by
    backfill_internal_harti_gaps() (real data), and impute_missing_historical_prices()
    must then leave it alone -- imputation only ever touches what is STILL missing."""
    missing_day = pd.Timestamp("2026-06-20")
    history = _history_with_gap(missing_day)

    with patch("app.services.HartiLiveClient.from_config") as mock_from_config:
        mock_from_config.return_value.fetch_prices_for_date.return_value = _snapshot(
            [{"date": missing_day, "vegetable": "Tomato", "market": "Dambulla", "price": 777.0,
              "source": "HARTI_LIVE"}],
            bulletin_date=str(missing_day.date()),
        )
        after_archive_backfill = backfill_internal_harti_gaps(history, "Tomato", "Dambulla")

    after_imputation, imputed_dates = impute_missing_historical_prices(
        after_archive_backfill, "Tomato", "Dambulla"
    )

    assert imputed_dates == []  # nothing left for imputation to do
    filled_row = after_imputation.loc[after_imputation["date"] == missing_day]
    assert filled_row["price"].iloc[0] == 777.0  # the REAL archive value, not an interpolated estimate
    assert filled_row["source"].iloc[0] == "HARTI_LIVE"


# 3. End-to-end: merged history reaches PredictionPipeline and produces a real prediction -


def test_merged_history_produces_a_real_prediction_through_the_full_pipeline(tmp_path):
    from src.feature_engineering.dataset_builder import TrainTestDatasetBuilder
    from src.models.baseline import train_baseline_model
    from src.models.persistence import save_model
    from src.prediction.predictor import InferenceArtifact, save_inference_artifact

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n', encoding="utf-8")

    history = _history(n_days=60, with_fuel=False)
    builder = TrainTestDatasetBuilder(config_path=str(config_path))
    X_train, _X_test, y_train, _y_test = builder.build(history)
    model = train_baseline_model(X_train, y_train)
    save_model(model, filename="best_model.joblib", config_path=str(config_path))
    save_inference_artifact(
        InferenceArtifact(builder.fitted_encoder, list(X_train.columns)), config_path=str(config_path)
    )

    snapshot = _snapshot(
        [
            {
                "date": history["date"].max() + pd.Timedelta(days=1),
                "vegetable": "Tomato",
                "market": "Dambulla",
                "price": 300.0,
                "source": "HARTI_LIVE",
            }
        ],
        bulletin_date=str((history["date"].max() + pd.Timedelta(days=1)).date()),
    )
    merged = merge_harti_live_price(history, snapshot, "Tomato", "Dambulla")
    target_date = snapshot.bulletin_date + pd.Timedelta(days=1)
    input_df = build_prediction_input(merged, target_date, "Tomato", "Dambulla")

    pipeline = PredictionPipeline(config_path=str(config_path))
    result = pipeline.predict(input_df, target_date=target_date, vegetable="Tomato", market="Dambulla")

    assert len(result) == 1
    assert result.loc[0, "date"] == target_date
    assert pd.notna(result.loc[0, "predicted_price"])


# 4. Regression: a stale historical CSV + live bulletin gap must not silently fail ------


def test_stale_history_with_live_bulletin_gap_reproduces_and_is_fixed_by_backfill(tmp_path):
    """Regression test for the reported failure: "Predictor: cannot predict for row(s)
    ... required feature(s) ['price_lag_7d', 'price_rolling_std_7d'] are missing or NaN".

    Root cause: the historical CSV had gone stale relative to "Latest HARTI Data"'s
    live bulletin (its last known date for this vegetable/market was more than 7 days
    before the bulletin's date). merge_harti_live_price() only ever adds the bulletin's
    single row, so price_lag_7d had no exact match 7 days before the target date, and
    price_rolling_std_7d's trailing 7-day window held only that one new point -- too
    few observations for a defined standard deviation. Both are demonstrated below
    using the unmodified, pre-fix code path (merge_harti_live_price() alone).

    Fix: backfill_harti_price_gap() closes the gap with real (here, mocked) HARTI
    bulletins for the missing days before merge_harti_live_price() runs, giving both
    features genuine underlying observations again.
    """
    from src.feature_engineering.dataset_builder import TrainTestDatasetBuilder
    from src.models.baseline import train_baseline_model
    from src.models.persistence import save_model
    from src.prediction.predictor import FeatureAlignmentError, InferenceArtifact, save_inference_artifact

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n', encoding="utf-8")

    history = _history(n_days=60, with_fuel=False)  # dense daily 2026-06-01 .. 2026-07-30
    builder = TrainTestDatasetBuilder(config_path=str(config_path))
    X_train, _X_test, y_train, _y_test = builder.build(history)
    model = train_baseline_model(X_train, y_train)
    save_model(model, filename="best_model.joblib", config_path=str(config_path))
    save_inference_artifact(
        InferenceArtifact(builder.fitted_encoder, list(X_train.columns)), config_path=str(config_path)
    )

    last_known = history["date"].max()
    bulletin_date = last_known + pd.Timedelta(days=8)  # an 8-day stale gap -- longer than the 7-day window
    live_price = 500.0
    snapshot = _snapshot(
        [{"date": bulletin_date, "vegetable": "Tomato", "market": "Dambulla", "price": live_price,
          "source": "HARTI_LIVE"}],
        bulletin_date=str(bulletin_date.date()),
    )
    target_date = bulletin_date + pd.Timedelta(days=1)
    pipeline = PredictionPipeline(config_path=str(config_path))

    # Pre-fix behavior: merging only the live row leaves the gap open.
    merged_without_backfill = merge_harti_live_price(history, snapshot, "Tomato", "Dambulla")
    input_without_backfill = build_prediction_input(merged_without_backfill, target_date, "Tomato", "Dambulla")
    with pytest.raises(FeatureAlignmentError, match="price_lag_7d"):
        pipeline.predict(input_without_backfill, target_date=target_date, vegetable="Tomato", market="Dambulla")

    # Fix: backfill the real (mocked) HARTI bulletins for the gap days first.
    with patch("app.services.HartiLiveClient.from_config") as mock_from_config:
        def fetch(day):
            day = pd.Timestamp(day)
            price = live_price - (bulletin_date - day).days
            return _snapshot(
                [{"date": day, "vegetable": "Tomato", "market": "Dambulla", "price": price,
                  "source": "HARTI_LIVE"}],
                bulletin_date=str(day.date()),
            )

        mock_from_config.return_value.fetch_prices_for_date.side_effect = fetch
        backfilled = backfill_harti_price_gap(
            history, snapshot, "Tomato", "Dambulla", config_path=str(config_path)
        )

    merged_with_backfill = merge_harti_live_price(backfilled, snapshot, "Tomato", "Dambulla")
    input_with_backfill = build_prediction_input(merged_with_backfill, target_date, "Tomato", "Dambulla")

    normalized = pipeline._normalize(input_with_backfill)
    featured = pipeline._feature_engineer.build_features(normalized)
    target_features = featured.loc[
        (featured["vegetable"] == "Tomato")
        & (featured["market"] == "Dambulla")
        & (featured["date"] == target_date)
    ].iloc[0]
    assert pd.notna(target_features["price_lag_7d"])
    assert pd.notna(target_features["price_rolling_std_7d"])

    result = pipeline.predict(
        input_with_backfill, target_date=target_date, vegetable="Tomato", market="Dambulla"
    )
    assert len(result) == 1
    assert pd.notna(result.loc[0, "predicted_price"])


# 5. Regression: fuel_price_lag_1d must reflect the LIVE CEYPETCO price, not just a
#    forward-filled guess, when a live reading is available and covers the needed date --

def test_live_fuel_price_reaches_diesel_price_lag_1d_not_just_the_stale_csv_value(tmp_path):
    """The historical fuel CSV can go stale exactly like the price CSV (see
    test_stale_history_with_live_bulletin_gap_...) -- forward-filling its last known
    value is a legitimate, non-fabricated fallback (see finalize_live_row()'s
    docstring), but if CEYPETCO's own currently-in-effect price has actually changed
    since that stale CSV's last row, forward-fill alone would silently use a
    now-wrong price. This proves the live reading -- not the stale forward-filled one
    -- is what ends up in diesel_price_lag_1d/petrol_price_lag_1d when it's available
    and covers the needed date.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'paths:\n  models_dir: "{tmp_path.as_posix()}"\n', encoding="utf-8")

    history = _history(n_days=60, with_fuel=True)  # fuel stops 3 days before price history's end
    stale_fuel_price = 363.0  # history's own last known (forward-fillable) diesel price
    live_diesel_price = 382.0  # CEYPETCO's real, currently-in-effect price -- different from the stale one

    last_known = history["date"].max()
    bulletin_date = last_known + pd.Timedelta(days=1)  # no price gap -- isolates the fuel behavior
    target_date = bulletin_date + pd.Timedelta(days=1)

    snapshot = _snapshot(
        [{"date": bulletin_date, "vegetable": "Tomato", "market": "Dambulla", "price": 300.0,
          "source": "HARTI_LIVE"}],
        bulletin_date=str(bulletin_date.date()),
    )
    live_fuel = _fuel_snapshot(
        {"diesel_price": live_diesel_price, "petrol_price": 414.0},
        {"diesel_price": "2026-06-29", "petrol_price": "2026-06-29"},  # effective well before bulletin_date
    )

    merged = merge_harti_live_price(history, snapshot, "Tomato", "Dambulla", live_fuel_snapshot=live_fuel)
    input_df = build_prediction_input(merged, target_date, "Tomato", "Dambulla")

    pipeline = PredictionPipeline(config_path=str(config_path))
    normalized = pipeline._normalize(input_df)
    featured = pipeline._feature_engineer.build_features(normalized)
    target_features = featured.loc[
        (featured["vegetable"] == "Tomato")
        & (featured["market"] == "Dambulla")
        & (featured["date"] == target_date)
    ].iloc[0]

    assert target_features["diesel_price_lag_1d"] == live_diesel_price
    assert target_features["diesel_price_lag_1d"] != stale_fuel_price


# 6. load_real_market_history ----------------------------------------------------------


def test_load_real_market_history_runs_the_real_preprocessing_pipeline(tmp_path):
    """Uses the project's small hermetic sample CSVs (same ones
    tests/preprocessing/test_pipeline.py relies on) -- proves this function is a
    genuine PreprocessingPipeline + apply_production_region_weather composition, not
    a reimplementation. weather_data_by_region/crop_production_regions_mapping are
    deliberately pointed at nonexistent paths (mirroring test_live_weather_services.py's
    own _config() helper) so this test stays hermetic against the real project's
    data/raw/weather_data_by_region.csv, which would otherwise be picked up by
    apply_production_region_weather()'s own config default."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "paths:\n"
        '  raw_files:\n'
        '    harti_prices: "tests/fixtures/market_sample.csv"\n'
        '    weather_data: "tests/fixtures/weather_sample.csv"\n'
        '    fuel_prices: "tests/fixtures/fuel_price_sample.csv"\n'
        f'    weather_data_by_region: "{(tmp_path / "nonexistent_region_weather.csv").as_posix()}"\n'
        f'  crop_production_regions_mapping: "{(tmp_path / "nonexistent_mapping.yaml").as_posix()}"\n'
        "data_schema:\n"
        '  date_column: "date"\n'
        '  vegetable_column: "vegetable"\n'
        '  market_column: "market"\n'
        '  price_column: "price"\n'
        "  weather_columns:\n"
        '    - "rainfall_mm"\n'
        '    - "temperature_c"\n'
        '    - "humidity_pct"\n'
        "  fuel_price_columns:\n"
        '    - "diesel_price"\n'
        '    - "petrol_price"\n',
        encoding="utf-8",
    )

    result = load_real_market_history(config_path=str(config_path))

    assert len(result) > 0
    assert "price" in result.columns
    assert "rainfall_mm" in result.columns
    assert "diesel_price" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
