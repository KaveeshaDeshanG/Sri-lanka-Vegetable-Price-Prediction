"""Unit tests for src.preprocessing.transformer.DataTransformer."""

import pandas as pd
import pytest

from src.preprocessing.transformer import DataTransformer


@pytest.fixture
def transformer() -> DataTransformer:
    return DataTransformer()


def test_convert_date_column_dtype(transformer):
    df = pd.DataFrame({"date": ["2024-01-02", "2024-01-01"]})
    converted = transformer.convert_date_column(df)
    assert pd.api.types.is_datetime64_any_dtype(converted["date"])


def test_sort_chronologically_orders_rows(transformer):
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-01"])})
    sorted_df = transformer.sort_chronologically(df)
    assert list(sorted_df["date"]) == list(pd.to_datetime(["2024-01-01", "2024-01-02"]))


def test_standardize_vegetable_names_maps_alias(transformer):
    df = pd.DataFrame({"vegetable": ["Eggplant", "Carrot"]})
    standardized = transformer.standardize_vegetable_names(df)
    assert standardized.loc[0, "vegetable"] == "Brinjal"
    assert standardized.loc[1, "vegetable"] == "Carrot"


def test_standardize_market_names_maps_alias(transformer):
    df = pd.DataFrame({"market": ["Colombo", "Dambulla"]})
    standardized = transformer.standardize_market_names(df)
    assert standardized.loc[0, "market"] == "Pettah"
    assert standardized.loc[1, "market"] == "Dambulla"


def test_convert_numeric_columns(transformer):
    df = pd.DataFrame({"price": ["100.5", "200.0"]})
    converted = transformer.convert_numeric_columns(df, ["price"])
    assert pd.api.types.is_float_dtype(converted["price"])
