"""Unit tests for src.feature_engineering.weather_features.WeatherFeatureGenerator."""

import pandas as pd
import pytest

from src.feature_engineering.weather_features import WeatherFeatureGenerator


@pytest.fixture
def generator() -> WeatherFeatureGenerator:
    return WeatherFeatureGenerator(weather_columns=["rainfall_mm", "temperature_c"], weather_lag_days=14)


def test_weather_lag_matches_the_reading_from_n_days_earlier(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-15"]),
            "vegetable": ["Carrot"] * 2,
            "market": ["Dambulla"] * 2,
            "price": [100.0, 110.0],
            "rainfall_mm": [5.0, 20.0],
            "temperature_c": [28.0, 30.0],
        }
    )
    result = generator.add_weather_lag_features(df)
    lagged_rainfall = result.loc[result["date"] == "2024-01-15", "rainfall_mm_lag14d"].iloc[0]
    lagged_temp = result.loc[result["date"] == "2024-01-15", "temperature_c_lag14d"].iloc[0]
    assert lagged_rainfall == pytest.approx(5.0)
    assert lagged_temp == pytest.approx(28.0)


def test_weather_lag_is_nan_without_a_matching_earlier_date(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "vegetable": ["Carrot"],
            "market": ["Dambulla"],
            "price": [100.0],
            "rainfall_mm": [5.0],
            "temperature_c": [28.0],
        }
    )
    result = generator.add_weather_lag_features(df)
    assert pd.isna(result.loc[0, "rainfall_mm_lag14d"])


def test_weather_lag_is_shared_across_vegetables_and_markets_by_default(generator):
    """Default (no group_columns): weather is treated as a single national series, not
    grouped like price lag/rolling features -- this is the pre-production-region-mapping
    behavior, still correct for a caller that never applies vegetable-specific weather."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-15", "2024-01-15"]),
            "vegetable": ["Carrot", "Tomato", "Carrot", "Tomato"],
            "market": ["Dambulla", "Pettah", "Dambulla", "Pettah"],
            "price": [100.0, 80.0, 110.0, 85.0],
            "rainfall_mm": [5.0, 5.0, 20.0, 20.0],
            "temperature_c": [28.0, 28.0, 30.0, 30.0],
        }
    )
    result = generator.add_weather_lag_features(df)
    lagged_values = result.loc[result["date"] == "2024-01-15", "rainfall_mm_lag14d"]
    assert len(result) == 4  # no fan-out despite two rows sharing each date
    assert lagged_values.tolist() == pytest.approx([5.0, 5.0])


# --- group_columns=["vegetable"]: production-region-aware mode --------------------
# Mirrors how FeatureEngineer actually constructs WeatherFeatureGenerator once
# apply_production_region_weather() may have given different vegetables different
# same-day readings (see feature_engineer.py).


@pytest.fixture
def grouped_generator() -> WeatherFeatureGenerator:
    return WeatherFeatureGenerator(
        weather_columns=["rainfall_mm", "temperature_c"],
        weather_lag_days=14,
        group_columns=["vegetable"],
    )


def test_weather_lag_case_a_different_vegetables_keep_different_weather(grouped_generator):
    """Leeks (Nuwara Eliya) and Beans (Badulla) share a calendar date but have distinct
    production-region weather -- their lag features must stay distinct, not be averaged
    into one blended value the way the ungrouped default would."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-15", "2024-01-15"]),
            "vegetable": ["Leeks", "Beans", "Leeks", "Beans"],
            "market": ["Dambulla", "Dambulla", "Dambulla", "Dambulla"],
            "price": [100.0, 80.0, 110.0, 85.0],
            "rainfall_mm": [2.6, 1.1, 20.0, 20.0],  # Nuwara Eliya vs Badulla same-day readings
            "temperature_c": [18.7, 25.4, 30.0, 30.0],
        }
    )
    result = grouped_generator.add_weather_lag_features(df)
    lagged = result.loc[result["date"] == "2024-01-15"].set_index("vegetable")
    assert lagged.loc["Leeks", "rainfall_mm_lag14d"] == pytest.approx(2.6)
    assert lagged.loc["Beans", "rainfall_mm_lag14d"] == pytest.approx(1.1)
    assert lagged.loc["Leeks", "rainfall_mm_lag14d"] != lagged.loc["Beans", "rainfall_mm_lag14d"]
    assert len(result) == 4  # still no fan-out


def test_weather_lag_case_b_uses_the_same_vegetables_own_history(grouped_generator):
    """A vegetable's lag feature is looked up from ITS OWN earlier rows -- unaffected by
    another vegetable's reading on the exact lag date, even when that other vegetable's
    own history is denser/different."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-08", "2024-01-15", "2024-01-15"]
            ),
            "vegetable": ["Carrot", "Tomato", "Tomato", "Carrot", "Tomato"],
            "market": ["Dambulla"] * 5,
            "price": [100.0, 80.0, 81.0, 110.0, 85.0],
            "rainfall_mm": [2.6, 1.7, 9.9, 20.0, 20.0],
            "temperature_c": [18.7, 29.8, 24.0, 30.0, 30.0],
        }
    )
    result = grouped_generator.add_weather_lag_features(df)
    lagged = result.loc[result["date"] == "2024-01-15"].set_index("vegetable")
    # Carrot's 2024-01-15 lag must come from Carrot's own 2024-01-01 row (2.6), never
    # from Tomato's 2024-01-01 (1.7) or 2024-01-08 (9.9) rows.
    assert lagged.loc["Carrot", "rainfall_mm_lag14d"] == pytest.approx(2.6)
    assert lagged.loc["Tomato", "rainfall_mm_lag14d"] == pytest.approx(1.7)


def test_weather_lag_case_c_missing_weather_still_nan_when_grouped(grouped_generator):
    """No matching earlier date within the SAME vegetable's group -> NaN, same missing-data
    contract as the ungrouped default, not fabricated from another vegetable's reading."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "vegetable": ["Carrot", "Tomato"],
            "market": ["Dambulla", "Dambulla"],
            "price": [100.0, 80.0],
            "rainfall_mm": [2.6, 1.7],
            "temperature_c": [18.7, 29.8],
        }
    )
    result = grouped_generator.add_weather_lag_features(df)
    assert pd.isna(result.loc[result["vegetable"] == "Carrot", "rainfall_mm_lag14d"]).all()
    assert pd.isna(result.loc[result["vegetable"] == "Tomato", "rainfall_mm_lag14d"]).all()


def test_weather_lag_falls_back_to_date_only_when_group_column_absent():
    """A configured group column missing from the input degrades to the ungrouped
    (date-only) lookup -- logged, not fatal -- exactly like LagFeatureGenerator's own
    missing-grouping-column fallback."""
    generator = WeatherFeatureGenerator(
        weather_columns=["rainfall_mm"], weather_lag_days=14, group_columns=["vegetable"]
    )
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-15"]),
            "price": [100.0, 110.0],
            "rainfall_mm": [5.0, 20.0],
        }
    )
    result = generator.add_weather_lag_features(df)
    assert result.loc[result["date"] == "2024-01-15", "rainfall_mm_lag14d"].iloc[0] == pytest.approx(5.0)


def test_weather_lag_skips_missing_weather_columns_without_error():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "vegetable": ["Carrot"],
            "market": ["Dambulla"],
            "price": [100.0],
        }
    )
    generator = WeatherFeatureGenerator(weather_columns=["rainfall_mm", "temperature_c"])
    result = generator.add_weather_lag_features(df)
    assert "rainfall_mm_lag14d" not in result.columns
    assert list(result["price"]) == [100.0]


def test_weather_lag_builds_only_for_partially_available_columns(generator):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-15"]),
            "vegetable": ["Carrot"] * 2,
            "market": ["Dambulla"] * 2,
            "price": [100.0, 110.0],
            "rainfall_mm": [5.0, 20.0],
            # temperature_c intentionally absent
        }
    )
    result = generator.add_weather_lag_features(df)
    assert "rainfall_mm_lag14d" in result.columns
    assert "temperature_c_lag14d" not in result.columns
