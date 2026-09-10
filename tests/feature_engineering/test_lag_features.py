"""Unit tests for src.feature_engineering.lag_features.LagFeatureGenerator."""

import pandas as pd
import pytest

from src.feature_engineering.lag_features import LagFeatureGenerator


@pytest.fixture
def generator() -> LagFeatureGenerator:
    return LagFeatureGenerator(lag_periods_days=[1, 7])


def test_lag_feature_uses_exact_calendar_day_match(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-08"]),
            "vegetable": ["Carrot"] * 3,
            "market": ["Dambulla"] * 3,
            "price": [100.0, 102.0, 110.0],
        }
    )
    result = generator.add_lag_features(df)

    lag1_jan2 = result.loc[result["date"] == "2024-01-02", "price_lag_1d"].iloc[0]
    assert lag1_jan2 == pytest.approx(100.0)

    lag7_jan8 = result.loc[result["date"] == "2024-01-08", "price_lag_7d"].iloc[0]
    assert lag7_jan8 == pytest.approx(100.0)  # 2024-01-08 minus 7 days is 2024-01-01 (price 100)


def test_lag_feature_is_nan_when_no_row_exists_that_many_days_earlier(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "vegetable": ["Carrot"] * 2,
            "market": ["Dambulla"] * 2,
            "price": [100.0, 110.0],
        }
    )
    result = generator.add_lag_features(df)
    lag1_jan8 = result.loc[result["date"] == "2024-01-08", "price_lag_1d"].iloc[0]
    assert pd.isna(lag1_jan8)


def test_lag_feature_does_not_mix_different_vegetables(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02"]),
            "vegetable": ["Carrot", "Carrot", "Tomato"],
            "market": ["Dambulla"] * 3,
            "price": [100.0, 105.0, 999.0],
        }
    )
    result = generator.add_lag_features(df)
    carrot_lag1 = result.loc[
        (result["vegetable"] == "Carrot") & (result["date"] == "2024-01-02"), "price_lag_1d"
    ].iloc[0]
    tomato_lag1 = result.loc[
        (result["vegetable"] == "Tomato") & (result["date"] == "2024-01-02"), "price_lag_1d"
    ].iloc[0]
    assert carrot_lag1 == pytest.approx(100.0)
    assert pd.isna(tomato_lag1)  # no prior Tomato row at all, must not borrow Carrot's price


def test_lag_feature_does_not_mix_different_markets(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02"]),
            "vegetable": ["Carrot"] * 3,
            "market": ["Dambulla", "Dambulla", "Pettah"],
            "price": [100.0, 105.0, 999.0],
        }
    )
    result = generator.add_lag_features(df)
    pettah_lag1 = result.loc[result["market"] == "Pettah", "price_lag_1d"].iloc[0]
    assert pd.isna(pettah_lag1)  # no prior Pettah row, must not borrow Dambulla's price


def test_lag_feature_does_not_leak_future_prices(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "vegetable": ["Carrot"] * 3,
            "market": ["Dambulla"] * 3,
            "price": [100.0, 101.0, 999999.0],  # future spike on the last day
        }
    )
    result = generator.add_lag_features(df)
    lag1_jan2 = result.loc[result["date"] == "2024-01-02", "price_lag_1d"].iloc[0]
    assert lag1_jan2 == pytest.approx(100.0)  # never sees the day-3 spike


def test_lag_feature_insufficient_history_is_nan_not_zero(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "vegetable": ["Carrot"],
            "market": ["Dambulla"],
            "price": [100.0],
        }
    )
    result = generator.add_lag_features(df)
    assert pd.isna(result.loc[0, "price_lag_1d"])
    assert pd.isna(result.loc[0, "price_lag_7d"])


def test_lag_feature_does_not_fan_out_duplicate_source_rows():
    """Two sources reporting the same (vegetable, market, date) must not multiply row count."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "vegetable": ["Carrot"] * 4,
            "market": ["Dambulla"] * 4,
            "source": ["HARTI", "Dambulla", "HARTI", "Dambulla"],
            "price": [100.0, 104.0, 105.0, 109.0],
        }
    )
    generator = LagFeatureGenerator(lag_periods_days=[1])
    result = generator.add_lag_features(df)
    assert len(result) == 4  # no fan-out
    lag1_values = result.loc[result["date"] == "2024-01-02", "price_lag_1d"]
    assert lag1_values.tolist() == pytest.approx([102.0, 102.0])  # average of 100.0 and 104.0


def test_lag_feature_empty_input(generator):
    df = pd.DataFrame(columns=["date", "vegetable", "market", "price"])
    df["date"] = pd.to_datetime(df["date"])
    result = generator.add_lag_features(df)
    assert len(result) == 0
    assert "price_lag_1d" in result.columns
