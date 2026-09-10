"""Unit tests for src.data_collection.fuel_price_loader.FuelPriceDataLoader."""

import pandas as pd
import pytest

from src.data_collection.data_validator import DataValidationError
from src.data_collection.fuel_price_loader import FuelPriceDataLoader

FIXTURE = "tests/fixtures/fuel_price_sample.csv"


@pytest.fixture
def loader() -> FuelPriceDataLoader:
    return FuelPriceDataLoader()


def test_load_valid_fuel_csv_returns_all_rows_and_columns(loader):
    df = loader.load(FIXTURE)
    original = pd.read_csv(FIXTURE)
    assert len(df) == len(original)
    assert "diesel_price" in df.columns
    assert "petrol_price" in df.columns


def test_load_raises_if_file_missing(loader):
    with pytest.raises(DataValidationError):
        loader.load("tests/fixtures/does_not_exist.csv")


def test_load_raises_if_diesel_price_column_missing(loader, tmp_path):
    path = tmp_path / "missing_diesel.csv"
    path.write_text("date,petrol_price\n2024-01-01,344.0\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="diesel_price"):
        loader.load(str(path))


def test_load_raises_if_petrol_price_column_missing(loader, tmp_path):
    path = tmp_path / "missing_petrol.csv"
    path.write_text("date,diesel_price\n2024-01-01,317.0\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="petrol_price"):
        loader.load(str(path))


def test_load_raises_if_date_column_missing(loader, tmp_path):
    path = tmp_path / "missing_date.csv"
    path.write_text("diesel_price,petrol_price\n317.0,344.0\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="date"):
        loader.load(str(path))


def test_load_raises_if_diesel_price_non_positive(loader, tmp_path):
    path = tmp_path / "non_positive_diesel.csv"
    path.write_text("date,diesel_price,petrol_price\n2024-01-01,0.0,344.0\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="diesel_price"):
        loader.load(str(path))


def test_load_raises_if_petrol_price_non_positive(loader, tmp_path):
    path = tmp_path / "non_positive_petrol.csv"
    path.write_text("date,diesel_price,petrol_price\n2024-01-01,317.0,-5.0\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="petrol_price"):
        loader.load(str(path))


def test_load_raises_if_price_column_non_numeric(loader, tmp_path):
    path = tmp_path / "non_numeric.csv"
    path.write_text("date,diesel_price,petrol_price\n2024-01-01,abc,344.0\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="diesel_price"):
        loader.load(str(path))


def test_load_raises_on_duplicate_dates(loader, tmp_path):
    path = tmp_path / "duplicate_dates.csv"
    path.write_text(
        "date,diesel_price,petrol_price\n"
        "2024-01-01,317.0,344.0\n"
        "2024-01-01,320.0,345.0\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="duplicate"):
        loader.load(str(path))


def test_load_raises_if_file_is_empty(loader, tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError):
        loader.load(str(path))


def test_loader_respects_configured_fuel_price_columns(tmp_path):
    """A diesel-only deployment (no petrol_price at all) must work when configured accordingly."""
    path = tmp_path / "diesel_only.csv"
    path.write_text("date,diesel_price\n2024-01-01,317.0\n2024-01-02,317.0\n", encoding="utf-8")
    diesel_only_loader = FuelPriceDataLoader(fuel_price_columns=["diesel_price"])

    df = diesel_only_loader.load(str(path))

    assert len(df) == 2
    assert "diesel_price" in df.columns
