"""Unit tests for DataLoader's market-data facade: HARTI required, second source optional."""

import pytest

from config.config_loader import ConfigKeyError
from src.data_collection.data_loader import DataLoader

SINGLE_SOURCE_CONFIG = "tests/fixtures/test_config.yaml"
TWO_SOURCE_CONFIG = "tests/fixtures/test_config_two_source.yaml"
FUEL_CONFIG = "tests/fixtures/test_config_with_fuel.yaml"


def test_load_market_data_succeeds_with_harti_only_config():
    loader = DataLoader(config_path=SINGLE_SOURCE_CONFIG)
    df = loader.load_market_data()
    assert len(df) > 0
    assert set(df["source"].unique()) == {"HARTI"}


def test_load_market_data_still_supports_a_configured_second_source():
    """Backward-compatible: if dambulla_prices IS set in config, it's still loaded and combined."""
    loader = DataLoader(config_path=TWO_SOURCE_CONFIG)
    df = loader.load_market_data()
    assert set(df["source"].unique()) == {"HARTI", "Dambulla"}


def test_load_market_data_raises_if_harti_prices_missing_from_config(tmp_path):
    config_path = tmp_path / "config_missing_harti.yaml"
    config_path.write_text(
        """
paths:
  raw_files:
    weather_data: "tests/fixtures/weather_sample.csv"
data_schema:
  date_column: "date"
  vegetable_column: "vegetable"
  market_column: "market"
  price_column: "price"
  weather_columns: ["rainfall_mm", "temperature_c", "humidity_pct"]
""",
        encoding="utf-8",
    )
    loader = DataLoader(config_path=str(config_path))
    with pytest.raises(ConfigKeyError):
        loader.load_market_data()


def test_load_all_returns_market_and_weather_with_harti_only_source():
    loader = DataLoader(config_path=SINGLE_SOURCE_CONFIG)
    data = loader.load_all()
    assert set(data.keys()) == {"market", "weather", "fuel"}
    assert set(data["market"]["source"].unique()) == {"HARTI"}
    assert len(data["weather"]) > 0


def test_load_fuel_price_data_returns_none_when_not_configured():
    """paths.raw_files.fuel_prices is unset in SINGLE_SOURCE_CONFIG -- must not raise."""
    loader = DataLoader(config_path=SINGLE_SOURCE_CONFIG)
    assert loader.load_fuel_price_data() is None


def test_load_all_fuel_key_is_none_when_not_configured():
    loader = DataLoader(config_path=SINGLE_SOURCE_CONFIG)
    data = loader.load_all()
    assert data["fuel"] is None


def test_load_fuel_price_data_returns_dataframe_when_configured():
    loader = DataLoader(config_path=FUEL_CONFIG)
    df = loader.load_fuel_price_data()
    assert df is not None
    assert len(df) > 0
    assert "diesel_price" in df.columns
    assert "petrol_price" in df.columns


def test_load_all_includes_fuel_dataframe_when_configured():
    loader = DataLoader(config_path=FUEL_CONFIG)
    data = loader.load_all()
    assert set(data.keys()) == {"market", "weather", "fuel"}
    assert data["fuel"] is not None
    assert len(data["fuel"]) > 0
