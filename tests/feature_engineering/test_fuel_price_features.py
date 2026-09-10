"""Unit tests for src.feature_engineering.fuel_price_features.FuelPriceFeatureGenerator."""

import pandas as pd
import pytest

from src.feature_engineering.fuel_price_features import FuelPriceFeatureGenerator


@pytest.fixture
def generator() -> FuelPriceFeatureGenerator:
    return FuelPriceFeatureGenerator(fuel_price_columns=["diesel_price", "petrol_price"], fuel_price_lag_days=1)


def test_fuel_price_lag_matches_the_reading_from_1_day_earlier(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "vegetable": ["Carrot"] * 2,
            "market": ["Dambulla"] * 2,
            "price": [100.0, 110.0],
            "diesel_price": [300.0, 310.0],
            "petrol_price": [350.0, 360.0],
        }
    )
    result = generator.add_fuel_price_lag_features(df)
    lagged_diesel = result.loc[result["date"] == "2024-01-02", "diesel_price_lag_1d"].iloc[0]
    lagged_petrol = result.loc[result["date"] == "2024-01-02", "petrol_price_lag_1d"].iloc[0]
    assert lagged_diesel == pytest.approx(300.0)
    assert lagged_petrol == pytest.approx(350.0)


def test_fuel_price_lag_is_nan_without_a_matching_earlier_date(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "vegetable": ["Carrot"],
            "market": ["Dambulla"],
            "price": [100.0],
            "diesel_price": [300.0],
            "petrol_price": [350.0],
        }
    )
    result = generator.add_fuel_price_lag_features(df)
    assert pd.isna(result.loc[0, "diesel_price_lag_1d"])
    assert pd.isna(result.loc[0, "petrol_price_lag_1d"])


def test_fuel_price_lag_never_uses_same_day_or_future_value(generator):
    """No same-day/future leakage: date D's lag_1d must equal exactly D-1's value, never D's own
    value, and must be unaffected by what happens on or after D."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "vegetable": ["Carrot"] * 3,
            "market": ["Dambulla"] * 3,
            "price": [100.0, 110.0, 120.0],
            "diesel_price": [300.0, 310.0, 999.0],  # 999.0 on 01-03 must never leak backward
            "petrol_price": [350.0, 360.0, 999.0],
        }
    )
    result = generator.add_fuel_price_lag_features(df)

    lag_on_02 = result.loc[result["date"] == "2024-01-02", "diesel_price_lag_1d"].iloc[0]
    lag_on_03 = result.loc[result["date"] == "2024-01-03", "diesel_price_lag_1d"].iloc[0]
    assert lag_on_02 == pytest.approx(300.0)  # 01-01's value, not 01-02's own 310.0
    assert lag_on_03 == pytest.approx(310.0)  # 01-02's value, not 01-03's own 999.0
    assert lag_on_03 != pytest.approx(999.0)


def test_fuel_price_lag_is_shared_across_vegetables_and_markets_for_the_same_date(generator):
    """Fuel price is a single national series -- it is not grouped like price lag/rolling features."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "vegetable": ["Carrot", "Tomato", "Carrot", "Tomato"],
            "market": ["Dambulla", "Pettah", "Dambulla", "Pettah"],
            "price": [100.0, 80.0, 110.0, 85.0],
            "diesel_price": [300.0, 300.0, 310.0, 310.0],
            "petrol_price": [350.0, 350.0, 360.0, 360.0],
        }
    )
    result = generator.add_fuel_price_lag_features(df)
    lagged_values = result.loc[result["date"] == "2024-01-02", "diesel_price_lag_1d"]
    assert len(result) == 4  # no fan-out despite two rows sharing each date
    assert lagged_values.tolist() == pytest.approx([300.0, 300.0])


def test_fuel_price_lag_skips_missing_fuel_price_columns_without_error():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "vegetable": ["Carrot"],
            "market": ["Dambulla"],
            "price": [100.0],
        }
    )
    generator = FuelPriceFeatureGenerator(fuel_price_columns=["diesel_price", "petrol_price"])
    result = generator.add_fuel_price_lag_features(df)
    assert "diesel_price_lag_1d" not in result.columns
    assert list(result["price"]) == [100.0]


def test_fuel_price_lag_builds_only_for_partially_available_columns(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "vegetable": ["Carrot"] * 2,
            "market": ["Dambulla"] * 2,
            "price": [100.0, 110.0],
            "diesel_price": [300.0, 310.0],
            # petrol_price intentionally absent
        }
    )
    result = generator.add_fuel_price_lag_features(df)
    assert "diesel_price_lag_1d" in result.columns
    assert "petrol_price_lag_1d" not in result.columns
