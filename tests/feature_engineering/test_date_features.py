"""Unit tests for src.feature_engineering.date_features.DateFeatureGenerator."""

import pandas as pd
import pytest

from src.feature_engineering.date_features import DateFeatureGenerator


@pytest.fixture
def generator() -> DateFeatureGenerator:
    return DateFeatureGenerator()


def test_add_date_features_values_are_correct(generator):
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-03-15"])})
    result = generator.add_date_features(df)

    assert result.loc[0, "day_of_week"] == 0  # Monday
    assert result.loc[0, "day_of_month"] == 1
    assert result.loc[0, "week_of_year"] == 1
    assert result.loc[0, "month"] == 1
    assert result.loc[0, "quarter"] == 1

    assert result.loc[1, "day_of_week"] == 4  # Friday
    assert result.loc[1, "day_of_month"] == 15
    assert result.loc[1, "month"] == 3
    assert result.loc[1, "quarter"] == 1


def test_add_date_features_coerces_string_dates(generator):
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"]})
    result = generator.add_date_features(df)
    assert result.loc[0, "day_of_week"] == 0
    assert result.loc[1, "day_of_week"] == 1


def test_add_date_features_does_not_mutate_input(generator):
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])})
    generator.add_date_features(df)
    assert "day_of_week" not in df.columns


def test_add_date_features_missing_date_column_is_skipped_not_fatal(generator):
    df = pd.DataFrame({"price": [100.0, 200.0]})
    result = generator.add_date_features(df)
    assert "day_of_week" not in result.columns
    assert list(result["price"]) == [100.0, 200.0]


def test_add_date_features_empty_input(generator):
    df = pd.DataFrame({"date": pd.to_datetime([])})
    result = generator.add_date_features(df)
    assert len(result) == 0
    assert "day_of_week" in result.columns
