"""Unit tests for MarketDataLoader: HARTI is required, a second source is optional."""

import pandas as pd
import pytest

from src.data_collection.data_validator import DataValidationError
from src.data_collection.market_loader import MarketDataLoader

FIXTURE = "tests/fixtures/market_sample.csv"


@pytest.fixture
def loader():
    return MarketDataLoader()


def test_load_harti_prices_tags_source_harti(loader):
    df = loader.load_harti_prices(FIXTURE)
    assert set(df["source"].unique()) == {"HARTI"}


def test_load_dambulla_prices_tags_source_dambulla(loader):
    """Kept for when a genuine second source exists -- not used by default ingestion."""
    df = loader.load_dambulla_prices(FIXTURE)
    assert set(df["source"].unique()) == {"Dambulla"}


def test_load_all_without_dambulla_path_returns_harti_only(loader):
    df = loader.load_all(FIXTURE, dambulla_path=None)
    original = pd.read_csv(FIXTURE)
    assert len(df) == len(original)
    assert set(df["source"].unique()) == {"HARTI"}


def test_load_all_dambulla_path_defaults_to_none(loader):
    """Omitting dambulla_path entirely must behave identically to passing None."""
    df = loader.load_all(FIXTURE)
    assert set(df["source"].unique()) == {"HARTI"}


def test_load_all_does_not_fabricate_a_second_source(loader):
    """Single-source load must never introduce a source="Dambulla" label."""
    df = loader.load_all(FIXTURE)
    assert "Dambulla" not in set(df["source"].unique())


def test_load_all_with_both_paths_still_supports_a_second_source(loader):
    """Backward-compatible: when a second source IS configured, it's still combined."""
    df = loader.load_all(FIXTURE, dambulla_path=FIXTURE)
    original = pd.read_csv(FIXTURE)
    assert len(df) == 2 * len(original)
    assert set(df["source"].unique()) == {"HARTI", "Dambulla"}


def test_load_all_raises_if_harti_path_missing(loader):
    with pytest.raises(DataValidationError):
        loader.load_all("tests/fixtures/does_not_exist.csv")
