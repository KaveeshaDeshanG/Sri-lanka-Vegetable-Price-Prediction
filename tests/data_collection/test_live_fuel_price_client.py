"""Unit tests for src.data_collection.live_fuel_price_client.CeypetcoLiveFuelPriceClient.

Network calls are mocked except for the "real structure" test at the bottom, which
parses tests/fixtures/ceypetco_sample_page.html -- a real excerpt of CEYPETCO's own
Marketing & Sales page (captured 2026-08) -- through the real _parse_cards(), proving
the parsing logic works against genuine page structure, not just hand-built fixtures.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data_collection.live_fuel_price_client import (
    DEFAULT_URL,
    CeypetcoLiveFuelPriceClient,
    LiveFuelPriceError,
    LiveFuelPriceSnapshot,
)

FIXTURE_PAGE_PATH = Path("tests/fixtures/ceypetco_sample_page.html")


def _client(**overrides) -> CeypetcoLiveFuelPriceClient:
    kwargs = {"url": DEFAULT_URL, "timeout_seconds": 5.0}
    kwargs.update(overrides)
    return CeypetcoLiveFuelPriceClient(**kwargs)


def _fake_response(status_code: int = 200, text: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    return response


def _price_card(name: str, price: str, effective_date: str) -> str:
    return f"""
    <div class="price-card">
        <div class="price-header">
            <h3 class="fuel-name">{name}</h3>
            <span class="fuel-category">White Oil</span>
        </div>
        <div class="price-body">
            <div class="price-value">
                <span class="price-currency">Rs.</span>
                {price}
                <span class="price-unit">per Ltr</span>
            </div>
            <div class="effective-date">
                <i>\U0001F4C5</i> Effect from: {effective_date}
            </div>
        </div>
    </div>
    """


def _page(cards: list[str]) -> str:
    return f'<html><body><div class="pricing-container">{"".join(cards)}</div></body></html>'


# 1. _parse_cards --------------------------------------------------------------------


def test_parse_cards_extracts_name_price_and_effective_date():
    html = _page(
        [
            _price_card("Lanka Auto Diesel", "382.00", "29-06-2026 12.00 Midnight"),
            _price_card("Lanka Petrol 92 Octane", "414.00", "29-06-2026 12.00 Midnight"),
        ]
    )
    cards = _client()._parse_cards(html)

    assert cards["Lanka Auto Diesel"] == (382.0, pd.Timestamp("2026-06-29"))
    assert cards["Lanka Petrol 92 Octane"] == (414.0, pd.Timestamp("2026-06-29"))


def test_parse_cards_handles_effective_date_without_a_time_component():
    html = _page([_price_card("Lanka Kerosene", "285.00", "30-05-2026")])
    cards = _client()._parse_cards(html)
    assert cards["Lanka Kerosene"] == (285.0, pd.Timestamp("2026-05-30"))


def test_parse_cards_raises_when_no_price_card_found():
    with pytest.raises(LiveFuelPriceError, match="no price card found"):
        _client()._parse_cards("<html><body>nothing here</body></html>")


def test_parse_cards_raises_on_unparseable_price():
    html = """
    <div class="price-card">
        <div class="price-header"><h3 class="fuel-name">Lanka Auto Diesel</h3></div>
        <div class="price-body">
            <div class="price-value">Rs. TBD per Ltr</div>
            <div class="effective-date">Effect from: 29-06-2026</div>
        </div>
    </div>
    """
    with pytest.raises(LiveFuelPriceError, match="could not parse a price"):
        _client()._parse_cards(html)


def test_parse_cards_raises_on_unparseable_effective_date():
    html = """
    <div class="price-card">
        <div class="price-header"><h3 class="fuel-name">Lanka Auto Diesel</h3></div>
        <div class="price-body">
            <div class="price-value">Rs. 382.00 per Ltr</div>
            <div class="effective-date">Effect from: sometime soon</div>
        </div>
    </div>
    """
    with pytest.raises(LiveFuelPriceError, match="could not parse an effective date"):
        _client()._parse_cards(html)


# 2. fetch_latest_prices() -- full flow ------------------------------------------------


@patch("src.data_collection.live_fuel_price_client.requests.get")
def test_fetch_latest_prices_returns_configured_columns_only(mock_get):
    mock_get.return_value = _fake_response(
        text=_page(
            [
                _price_card("Lanka Auto Diesel", "382.00", "29-06-2026 12.00 Midnight"),
                _price_card("Lanka Petrol 92 Octane", "414.00", "29-06-2026 12.00 Midnight"),
                _price_card("Lanka Kerosene", "285.00", "30-05-2026"),
            ]
        )
    )

    snapshot = _client().fetch_latest_prices()

    assert snapshot.prices == {"diesel_price": 382.0, "petrol_price": 414.0}
    assert snapshot.effective_dates == {
        "diesel_price": pd.Timestamp("2026-06-29"),
        "petrol_price": pd.Timestamp("2026-06-29"),
    }
    assert snapshot.source_url == DEFAULT_URL
    assert isinstance(snapshot.retrieved_at, pd.Timestamp)


@patch("src.data_collection.live_fuel_price_client.requests.get")
def test_fetch_latest_prices_raises_when_a_configured_column_has_no_matching_card(mock_get):
    mock_get.return_value = _fake_response(
        text=_page([_price_card("Lanka Kerosene", "285.00", "30-05-2026")])
    )
    with pytest.raises(LiveFuelPriceError, match="diesel_price"):
        _client().fetch_latest_prices()


@patch("src.data_collection.live_fuel_price_client.requests.get")
def test_fetch_latest_prices_raises_on_http_error(mock_get):
    mock_get.return_value = _fake_response(status_code=500)
    with pytest.raises(LiveFuelPriceError, match="HTTP 500"):
        _client().fetch_latest_prices()


@patch("src.data_collection.live_fuel_price_client.requests.get")
def test_fetch_latest_prices_raises_on_network_failure(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError("no route to host")
    with pytest.raises(LiveFuelPriceError, match="request failed"):
        _client().fetch_latest_prices()


@patch("src.data_collection.live_fuel_price_client.requests.get")
def test_fetch_latest_prices_raises_on_timeout(mock_get):
    mock_get.side_effect = requests.exceptions.Timeout("timed out")
    with pytest.raises(LiveFuelPriceError, match="timed out"):
        _client().fetch_latest_prices()


# 3. LiveFuelPriceSnapshot.covers() ----------------------------------------------------


def test_snapshot_covers_true_on_or_after_effective_date():
    snapshot = LiveFuelPriceSnapshot(
        prices={"diesel_price": 382.0},
        effective_dates={"diesel_price": pd.Timestamp("2026-06-29")},
        retrieved_at=pd.Timestamp("2026-08-21"),
        source_url=DEFAULT_URL,
    )
    assert snapshot.covers("diesel_price", pd.Timestamp("2026-06-29")) is True
    assert snapshot.covers("diesel_price", pd.Timestamp("2026-08-20")) is True


def test_snapshot_covers_false_before_effective_date():
    snapshot = LiveFuelPriceSnapshot(
        prices={"diesel_price": 382.0},
        effective_dates={"diesel_price": pd.Timestamp("2026-06-29")},
        retrieved_at=pd.Timestamp("2026-08-21"),
        source_url=DEFAULT_URL,
    )
    assert snapshot.covers("diesel_price", pd.Timestamp("2026-06-28")) is False


def test_snapshot_covers_false_for_unknown_column():
    snapshot = LiveFuelPriceSnapshot(
        prices={"diesel_price": 382.0},
        effective_dates={"diesel_price": pd.Timestamp("2026-06-29")},
        retrieved_at=pd.Timestamp("2026-08-21"),
        source_url=DEFAULT_URL,
    )
    assert snapshot.covers("petrol_price", pd.Timestamp("2026-08-20")) is False


# 4. from_config ------------------------------------------------------------------------


def test_from_config_reads_fuel_api_section(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "fuel_api:\n"
        '  url: "https://example.test/marketing-sales/"\n'
        "  timeout_seconds: 7\n"
        "data_schema:\n"
        "  fuel_price_columns:\n"
        '    - "diesel_price"\n'
        '    - "petrol_price"\n',
        encoding="utf-8",
    )
    client = CeypetcoLiveFuelPriceClient.from_config(str(config_path))
    assert client._url == "https://example.test/marketing-sales/"
    assert client._timeout_seconds == 7
    assert client._label_map == {
        "diesel_price": "Lanka Auto Diesel",
        "petrol_price": "Lanka Petrol 92 Octane",
    }


def test_from_config_defaults_when_fuel_api_section_absent(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project:\n  name: x\n", encoding="utf-8")
    client = CeypetcoLiveFuelPriceClient.from_config(str(config_path))
    assert client._url == DEFAULT_URL


# 5. Real page structure (no mocking) --------------------------------------------------


def test_parse_cards_against_real_captured_page_structure():
    html = FIXTURE_PAGE_PATH.read_text(encoding="utf-8")
    cards = _client()._parse_cards(html)

    assert cards["Lanka Auto Diesel"] == (382.0, pd.Timestamp("2026-06-29"))
    assert cards["Lanka Petrol 92 Octane"] == (414.0, pd.Timestamp("2026-06-29"))
    assert cards["Lanka Kerosene"] == (285.0, pd.Timestamp("2026-05-30"))


def test_fetch_latest_prices_against_real_captured_page_structure():
    html = FIXTURE_PAGE_PATH.read_text(encoding="utf-8")
    with patch("src.data_collection.live_fuel_price_client.requests.get") as mock_get:
        mock_get.return_value = _fake_response(text=html)
        snapshot = _client().fetch_latest_prices()

    assert snapshot.prices == {"diesel_price": 382.0, "petrol_price": 414.0}
