"""Unit tests for src.feature_engineering.rolling_features.RollingFeatureGenerator."""

import pandas as pd
import pytest

from src.feature_engineering.rolling_features import RollingFeatureGenerator


@pytest.fixture
def generator() -> RollingFeatureGenerator:
    return RollingFeatureGenerator(rolling_windows_days=[3])


def test_rolling_mean_excludes_the_current_row_itself(generator):
    """The classic leakage bug: a naive rolling window would include today's own price."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "vegetable": ["Carrot"] * 2,
            "market": ["Dambulla"] * 2,
            "price": [100.0, 999999.0],  # huge value on the second day
        }
    )
    result = generator.add_rolling_features(df)
    mean_day2 = result.loc[result["date"] == "2024-01-02", "price_rolling_mean_3d"].iloc[0]
    assert mean_day2 == pytest.approx(100.0)  # only day 1, never blended with day 2's own price


def test_rolling_mean_and_std_over_multiple_points(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "vegetable": ["Carrot"] * 3,
            "market": ["Dambulla"] * 3,
            "price": [100.0, 102.0, 101.0],
        }
    )
    result = generator.add_rolling_features(df)
    mean_day3 = result.loc[result["date"] == "2024-01-03", "price_rolling_mean_3d"].iloc[0]
    std_day3 = result.loc[result["date"] == "2024-01-03", "price_rolling_std_3d"].iloc[0]
    assert mean_day3 == pytest.approx(101.0)  # mean of day1 (100) and day2 (102)
    assert std_day3 == pytest.approx(1.4142135, rel=1e-4)


def test_rolling_window_excludes_points_outside_the_window(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-10"]),
            "vegetable": ["Carrot"] * 2,
            "market": ["Dambulla"] * 2,
            "price": [100.0, 200.0],
        }
    )
    result = generator.add_rolling_features(df)
    mean_day10 = result.loc[result["date"] == "2024-01-10", "price_rolling_mean_3d"].iloc[0]
    assert pd.isna(mean_day10)  # day 1 is 9 days earlier, outside the 3-day window


def test_rolling_features_do_not_mix_different_vegetables_or_markets(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "vegetable": ["Carrot", "Carrot", "Tomato", "Tomato"],
            "market": ["Dambulla"] * 4,
            "price": [100.0, 102.0, 500.0, 505.0],
        }
    )
    result = generator.add_rolling_features(df)
    carrot_mean = result.loc[
        (result["vegetable"] == "Carrot") & (result["date"] == "2024-01-02"), "price_rolling_mean_3d"
    ].iloc[0]
    tomato_mean = result.loc[
        (result["vegetable"] == "Tomato") & (result["date"] == "2024-01-02"), "price_rolling_mean_3d"
    ].iloc[0]
    assert carrot_mean == pytest.approx(100.0)
    assert tomato_mean == pytest.approx(500.0)


def test_rolling_std_is_nan_with_a_single_point_in_window(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "vegetable": ["Carrot"] * 2,
            "market": ["Dambulla"] * 2,
            "price": [100.0, 102.0],
        }
    )
    result = generator.add_rolling_features(df)
    std_day2 = result.loc[result["date"] == "2024-01-02", "price_rolling_std_3d"].iloc[0]
    assert pd.isna(std_day2)  # only 1 prior point; sample std needs at least 2


def test_rolling_features_insufficient_history_first_row_is_nan(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "vegetable": ["Carrot"],
            "market": ["Dambulla"],
            "price": [100.0],
        }
    )
    result = generator.add_rolling_features(df)
    assert pd.isna(result.loc[0, "price_rolling_mean_3d"])
    assert pd.isna(result.loc[0, "price_rolling_std_3d"])


def test_rolling_features_empty_input(generator):
    df = pd.DataFrame(columns=["date", "vegetable", "market", "price"])
    df["date"] = pd.to_datetime(df["date"])
    result = generator.add_rolling_features(df)
    assert len(result) == 0
    assert "price_rolling_mean_3d" in result.columns
    assert "price_rolling_std_3d" in result.columns
