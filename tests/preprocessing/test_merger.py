"""Unit tests for src.preprocessing.merger.DataMerger."""

import pandas as pd
import pytest

from src.preprocessing.merger import DataMerger


@pytest.fixture
def merger() -> DataMerger:
    return DataMerger()


def test_merge_keeps_all_market_rows_with_nan_weather_when_missing(merger):
    market_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "vegetable": ["Carrot", "Carrot"],
            "market": ["Dambulla", "Dambulla"],
            "price": [100.0, 105.0],
        }
    )
    weather_df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "rainfall_mm": [5.0],
            "temperature_c": [28.0],
        }
    )
    merged = merger.merge(market_df, weather_df)
    assert len(merged) == 2  # both market rows kept
    missing_row = merged[merged["date"] == "2024-01-02"].iloc[0]
    assert pd.isna(missing_row["rainfall_mm"])


def test_merge_excludes_weather_only_dates(merger):
    market_df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "vegetable": ["Carrot"],
            "market": ["Dambulla"],
            "price": [100.0],
        }
    )
    weather_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "rainfall_mm": [5.0, 10.0],
            "temperature_c": [28.0, 29.0],
        }
    )
    merged = merger.merge(market_df, weather_df)
    assert len(merged) == 1
    assert "2024-01-02" not in merged["date"].values


def test_merge_deduplicates_weather_dates_by_averaging(merger):
    market_df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "vegetable": ["Carrot"],
            "market": ["Dambulla"],
            "price": [100.0],
        }
    )
    weather_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "rainfall_mm": [4.0, 6.0],
            "temperature_c": [28.0, 30.0],
        }
    )
    merged = merger.merge(market_df, weather_df)
    assert len(merged) == 1
    assert merged.iloc[0]["rainfall_mm"] == pytest.approx(5.0)
    assert merged.iloc[0]["temperature_c"] == pytest.approx(29.0)


def test_merge_fuel_prices_returns_df_unchanged_when_fuel_df_is_none(merger):
    """Optional behavior: fuel not configured must not alter the market+weather merge at all."""
    df = pd.DataFrame({"date": ["2024-01-01"], "vegetable": ["Carrot"], "price": [100.0]})
    result = merger.merge_fuel_prices(df, None)
    pd.testing.assert_frame_equal(result, df)


def test_merge_fuel_prices_attaches_same_date_price_to_every_matching_row(merger):
    """Date-only join: two vegetables sharing a date both get the same fuel price -- no manual
    per-vegetable duplication, just the natural many-to-one join."""
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "vegetable": ["Carrot", "Tomato"],
            "price": [100.0, 80.0],
        }
    )
    fuel_df = pd.DataFrame({"date": ["2024-01-01"], "diesel_price": [317.0], "petrol_price": [344.0]})

    merged = merger.merge_fuel_prices(df, fuel_df)

    assert len(merged) == 2  # no fan-out
    assert (merged["diesel_price"] == 317.0).all()
    assert (merged["petrol_price"] == 344.0).all()


def test_merge_fuel_prices_keeps_market_rows_with_nan_fuel_when_missing(merger):
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "vegetable": ["Carrot", "Carrot"],
            "price": [100.0, 105.0],
        }
    )
    fuel_df = pd.DataFrame({"date": ["2024-01-01"], "diesel_price": [317.0], "petrol_price": [344.0]})

    merged = merger.merge_fuel_prices(df, fuel_df)

    assert len(merged) == 2  # both market rows kept
    missing_row = merged[merged["date"] == "2024-01-02"].iloc[0]
    assert pd.isna(missing_row["diesel_price"])


def test_merge_fuel_prices_deduplicates_fuel_dates_by_averaging(merger):
    df = pd.DataFrame({"date": ["2024-01-01"], "vegetable": ["Carrot"], "price": [100.0]})
    fuel_df = pd.DataFrame(
        {"date": ["2024-01-01", "2024-01-01"], "diesel_price": [310.0, 320.0], "petrol_price": [340.0, 360.0]}
    )

    merged = merger.merge_fuel_prices(df, fuel_df)

    assert len(merged) == 1  # no fan-out from the duplicate fuel row
    assert merged.iloc[0]["diesel_price"] == pytest.approx(315.0)
