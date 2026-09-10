"""Unit tests for src.preprocessing.cleaner.DataCleaner."""

import pandas as pd
import pytest

from src.preprocessing.cleaner import DataCleaner


@pytest.fixture
def cleaner() -> DataCleaner:
    return DataCleaner()


def test_detect_missing_values_reports_counts(cleaner):
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    counts = cleaner.detect_missing_values(df, "Test")
    assert counts["a"] == 1
    assert "b" not in counts


def test_clean_market_data_drops_missing_critical_fields(cleaner):
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", None, "2024-01-03"],
            "vegetable": ["Carrot", "Tomato", "Beans"],
            "market": ["Dambulla", "Dambulla", "Dambulla"],
            "price": [100.0, 90.0, 80.0],
        }
    )
    cleaned = cleaner.clean_market_data(df)
    assert len(cleaned) == 2
    assert cleaned["date"].isna().sum() == 0


def test_clean_market_data_removes_non_positive_price(cleaner):
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "vegetable": ["Carrot", "Tomato"],
            "market": ["Dambulla", "Dambulla"],
            "price": [100.0, -5.0],
        }
    )
    cleaned = cleaner.clean_market_data(df)
    assert len(cleaned) == 1
    assert (cleaned["price"] > 0).all()


def test_clean_market_data_removes_exact_duplicates(cleaner):
    row = {"date": "2024-01-01", "vegetable": "Carrot", "market": "Dambulla", "price": 100.0}
    df = pd.DataFrame([row, row, row])
    cleaned = cleaner.clean_market_data(df)
    assert len(cleaned) == 1


def test_clean_market_data_title_cases_text(cleaner):
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "vegetable": ["  carrot  "],
            "market": ["DAMBULLA"],
            "price": [100.0],
        }
    )
    cleaned = cleaner.clean_market_data(df)
    assert cleaned.loc[0, "vegetable"] == "Carrot"
    assert cleaned.loc[0, "market"] == "Dambulla"


def test_clean_text_columns_is_public_and_directly_callable(cleaner):
    """src/prediction reuses this directly (skipping the row-dropping steps around it in
    clean_market_data(), which would delete a row whose price is intentionally NaN)."""
    df = pd.DataFrame({"vegetable": ["  carrot  "], "market": ["DAMBULLA"]})
    cleaned = cleaner.clean_text_columns(df, ["vegetable", "market"])
    assert cleaned.loc[0, "vegetable"] == "Carrot"
    assert cleaned.loc[0, "market"] == "Dambulla"


def test_clean_market_data_drops_implausible_dates(cleaner):
    df = pd.DataFrame(
        {
            "date": ["1990-01-01", "2024-01-01", "2999-01-01"],
            "vegetable": ["Carrot", "Tomato", "Beans"],
            "market": ["Dambulla", "Dambulla", "Dambulla"],
            "price": [100.0, 90.0, 80.0],
        }
    )
    cleaned = cleaner.clean_market_data(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["vegetable"] == "Tomato"


def test_clean_weather_data_interpolates_missing_values(cleaner):
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "rainfall_mm": [0.0, None, 10.0],
            "temperature_c": [28.0, 29.0, 30.0],
            "humidity_pct": [70.0, 75.0, 80.0],
        }
    )
    cleaned = cleaner.clean_weather_data(df)
    assert len(cleaned) == 3
    interpolated_row = cleaned.loc[cleaned["date"] == "2024-01-02"].iloc[0]
    assert interpolated_row["rainfall_mm"] == pytest.approx(5.0)


def test_clean_weather_data_removes_negative_rainfall(cleaner):
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "rainfall_mm": [10.0, -1.0],
            "temperature_c": [28.0, 29.0],
            "humidity_pct": [70.0, 75.0],
        }
    )
    cleaned = cleaner.clean_weather_data(df)
    assert len(cleaned) == 1
    assert (cleaned["rainfall_mm"] >= 0).all()


def test_clean_weather_data_removes_out_of_range_temperature(cleaner):
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "rainfall_mm": [10.0, 5.0],
            "temperature_c": [28.0, 90.0],
            "humidity_pct": [70.0, 75.0],
        }
    )
    cleaned = cleaner.clean_weather_data(df)
    assert len(cleaned) == 1


def test_clean_fuel_price_data_forward_fills_missing_values(cleaner):
    """A gap must be carried forward from the last known price, NOT linearly interpolated."""
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "diesel_price": [300.0, None, 340.0],
            "petrol_price": [350.0, 360.0, 370.0],
        }
    )
    cleaned = cleaner.clean_fuel_price_data(df)
    assert len(cleaned) == 3
    filled_row = cleaned.loc[cleaned["date"] == "2024-01-02"].iloc[0]
    # Forward-filled from 2024-01-01's 300.0 -- NOT the interpolated midpoint (320.0).
    assert filled_row["diesel_price"] == pytest.approx(300.0)


def test_clean_fuel_price_data_does_not_backward_fill_leading_gap(cleaner):
    """A gap with no earlier known price can't be forward-filled and must be dropped, not guessed."""
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "diesel_price": [None, 310.0, 310.0],
            "petrol_price": [350.0, 360.0, 370.0],
        }
    )
    cleaned = cleaner.clean_fuel_price_data(df)
    assert len(cleaned) == 2
    assert "2024-01-01" not in cleaned["date"].values
    # Must never have been back-filled from the later 310.0 value.
    assert not (cleaned["date"] == "2024-01-01").any()


def test_clean_fuel_price_data_removes_non_positive_price(cleaner):
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "diesel_price": [300.0, 0.0],
            "petrol_price": [350.0, 360.0],
        }
    )
    cleaned = cleaner.clean_fuel_price_data(df)
    assert len(cleaned) == 1
    assert (cleaned["diesel_price"] > 0).all()


def test_clean_fuel_price_data_drops_missing_date(cleaner):
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", None],
            "diesel_price": [300.0, 310.0],
            "petrol_price": [350.0, 360.0],
        }
    )
    cleaned = cleaner.clean_fuel_price_data(df)
    assert len(cleaned) == 1
    assert cleaned["date"].isna().sum() == 0
