"""Live client for CEYPETCO's own published retail fuel prices.

Investigated directly against the real site (2026-08): CEYPETCO (ceypetco.gov.lk) has
no API or CSV/Excel export either, but its "Marketing & Sales" page
(`fuel_api.url`, default `marketing-sales/`) renders each product's current retail
price as a "price card" -- product name, price, and the date that price took effect
-- verified against the real, unmocked page (see tests/fixtures/ceypetco_sample_page.html,
a real excerpt of that page). Fuel price is a step-valued administrative price (see
fuel_price_features.py's docstring): it stays exactly constant between CEYPETCO
revisions, so unlike HARTI's daily bulletin or Open-Meteo's daily weather, there is no
"today's reading" to fetch -- only "whatever price is currently in effect, and since
when". That effective date matters: a price card's value is only valid for a date on
or after its own effective date, never for an earlier date (using it there would mean
a later revision governing a date before it took effect -- the same forward-looking
leak this project's feature generators are designed to avoid elsewhere).

Mirrors harti_live_client.py's/live_weather_client.py's design: one exception type for
every failure mode, and output normalized into this project's existing fuel-price
schema (diesel_price, petrol_price) -- a drop-in alternative fuel-price *source*, not a
special case downstream code needs to branch on.
"""

import logging
import re
from dataclasses import dataclass

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://ceypetco.gov.lk/marketing-sales/"
# CEYPETCO's page became a JavaScript app (2026-11): its HTML no longer contains prices, and
# the page itself loads them from this public endpoint. Tried first; the HTML parser remains
# as a fallback in case the site changes back.
DEFAULT_API_URL = "https://api.ceypetco.gov.lk/api/admin/fuel-prices/active"
_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; veggie-price-prediction/1.0)"}

# This project's fuel_price_columns -> the exact product name CEYPETCO's page uses for
# it (verified against the real page, 2026-08). "Lanka Auto Diesel" is the plain diesel
# grade Dambulla DEC's wholesale transport actually runs on (not "Lanka Super Diesel 4
# Star Euro 4", a premium grade); "Lanka Petrol 92 Octane" is the base petrol grade
# (not "Lanka Petrol 95 Octane Euro 4") -- matching data/raw/fuel_prices.csv's own
# historical values exactly (382.00 / 414.00 on the date this was verified).
DEFAULT_LABEL_MAP: dict[str, str] = {
    "diesel_price": "Lanka Auto Diesel",
    "petrol_price": "Lanka Petrol 92 Octane",
}

_PRICE_PATTERN = re.compile(r"([\d,]+\.\d+)")
_EFFECTIVE_DATE_PATTERN = re.compile(r"Effect from:\s*(\d{2}-\d{2}-\d{4})")


class LiveFuelPriceError(Exception):
    """Raised when CEYPETCO's current fuel prices cannot be fetched, parsed, or validated.

    Covers network failure, timeout, non-200 HTTP status, a changed page structure, a
    configured fuel-price column with no matching product card, an unparseable price,
    or an unparseable effective date. Never partially returns a result -- either a
    fully validated LiveFuelPriceSnapshot comes back, or this is raised.
    """


@dataclass(frozen=True)
class LiveFuelPriceSnapshot:
    """CEYPETCO's currently-in-effect price for each configured fuel-price column.

    prices: {fuel_price_column: current price}, e.g. {"diesel_price": 382.0,
        "petrol_price": 414.0}.
    effective_dates: {fuel_price_column: the date this price took effect}. A reading
        is only valid for a row dated on or after its own effective date -- see this
        module's docstring.
    retrieved_at: wall-clock time this snapshot was fetched.
    source_url: the exact URL fetched, for traceability.
    """

    prices: dict[str, float]
    effective_dates: dict[str, pd.Timestamp]
    retrieved_at: pd.Timestamp
    source_url: str

    def covers(self, column: str, date: "pd.Timestamp") -> bool:
        """Whether this snapshot's reading for `column` is valid for `date`.

        False for a column this snapshot has no reading for, or for a date before
        that reading's own effective date.
        """
        effective_date = self.effective_dates.get(column)
        return effective_date is not None and pd.Timestamp(date) >= effective_date


class CeypetcoLiveFuelPriceClient:
    """Fetches CEYPETCO's current retail fuel prices from its Marketing & Sales page."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        timeout_seconds: float = 15.0,
        label_map: dict[str, str] | None = None,
        api_url: str = DEFAULT_API_URL,
    ) -> None:
        self._url = url
        self._api_url = api_url
        self._timeout_seconds = timeout_seconds
        self._label_map = dict(label_map) if label_map else dict(DEFAULT_LABEL_MAP)

    @classmethod
    def from_config(cls, config_path: str = "config/config.yaml") -> "CeypetcoLiveFuelPriceClient":
        """Build a client from config.yaml's fuel_api section.

        Every value has a safe default (mirrors HartiLiveClient.from_config()) since
        fuel_api.enabled is checked by callers, not this class.
        """
        config = ConfigLoader(config_path)
        fuel_price_columns = config.get(
            "data_schema.fuel_price_columns", ["diesel_price", "petrol_price"]
        )
        label_map = {
            column: DEFAULT_LABEL_MAP[column] for column in fuel_price_columns if column in DEFAULT_LABEL_MAP
        }
        return cls(
            url=config.get("fuel_api.url", DEFAULT_URL),
            timeout_seconds=config.get("fuel_api.timeout_seconds", 15.0),
            label_map=label_map or None,
            api_url=config.get("fuel_api.api_url", DEFAULT_API_URL),
        )

    def fetch_latest_prices(self) -> LiveFuelPriceSnapshot:
        """Fetch CEYPETCO's currently-in-effect price for every configured fuel column.

        Raises:
            LiveFuelPriceError: on network failure, timeout, non-200 HTTP response, a
                page structure that no longer matches (no price cards found), a
                configured column whose product label isn't present on the page, or
                an unparseable price/effective-date for it.
        """
        source_url = self._api_url
        try:
            cards = self._fetch_cards_from_api()
        except LiveFuelPriceError as error:
            logger.info("CEYPETCO price API unavailable (%s); trying the HTML page", error)
            source_url = self._url
            cards = self._parse_cards(self._download())

        prices: dict[str, float] = {}
        effective_dates: dict[str, pd.Timestamp] = {}
        for column, label in self._label_map.items():
            if label not in cards:
                available = sorted(cards.keys())
                message = (
                    f"CEYPETCO page structure has changed: no product labeled '{label}' "
                    f"(needed for '{column}') was found. Products found: {available}"
                )
                logger.error(message)
                raise LiveFuelPriceError(message)
            price, effective_date = cards[label]
            prices[column] = price
            effective_dates[column] = effective_date

        logger.info(
            "CeypetcoLiveFuelPriceClient: fetched %s from %s", prices, source_url
        )
        return LiveFuelPriceSnapshot(
            prices=prices,
            effective_dates=effective_dates,
            retrieved_at=pd.Timestamp.now(),
            source_url=source_url,
        )

    def _fetch_cards_from_api(self) -> dict[str, tuple[float, "pd.Timestamp"]]:
        """{product name: (price, effective_date)} from CEYPETCO's public price endpoint."""
        try:
            response = requests.get(self._api_url, timeout=self._timeout_seconds, headers=_REQUEST_HEADERS)
            if response.status_code != 200:
                raise LiveFuelPriceError(f"price API returned HTTP {response.status_code}")
            records = response.json()["data"]
            cards = {
                record["product"]: (
                    float(record["price"]),
                    pd.Timestamp(record["effectiveDate"]).tz_localize(None).normalize(),
                )
                for record in records
                if record.get("product") and record.get("price") is not None and record.get("effectiveDate")
            }
        except (requests.exceptions.RequestException, ValueError, KeyError, TypeError, AttributeError) as error:
            raise LiveFuelPriceError(f"price API request failed: {error}") from error
        if not cards:
            raise LiveFuelPriceError("price API returned no products")
        return cards

    def _download(self) -> str:
        try:
            response = requests.get(self._url, timeout=self._timeout_seconds, headers=_REQUEST_HEADERS)
        except requests.exceptions.Timeout as error:
            message = f"CEYPETCO request timed out after {self._timeout_seconds}s"
            logger.error(message)
            raise LiveFuelPriceError(message) from error
        except requests.exceptions.RequestException as error:
            message = f"CEYPETCO request failed: {error}"
            logger.error(message)
            raise LiveFuelPriceError(message) from error

        if response.status_code != 200:
            message = f"CEYPETCO returned HTTP {response.status_code}"
            logger.error(message)
            raise LiveFuelPriceError(message)
        return response.text

    def _parse_cards(self, html: str) -> dict[str, tuple[float, "pd.Timestamp"]]:
        """Parse every "price card" on the page into {product name: (price, effective_date)}.

        Pure parsing (no I/O) -- kept separate from _download() so it can be
        unit-tested directly against a real captured page
        (tests/fixtures/ceypetco_sample_page.html), not just hand-built fixtures.
        """
        soup = BeautifulSoup(html, "html.parser")
        cards = soup.find_all("div", class_="price-card")
        if not cards:
            message = "CEYPETCO page structure has changed: no price card found on the page"
            logger.error(message)
            raise LiveFuelPriceError(message)

        parsed: dict[str, tuple[float, pd.Timestamp]] = {}
        for card in cards:
            name_element = card.find(class_="fuel-name")
            if name_element is None:
                continue
            name = name_element.get_text(strip=True)

            price_element = card.find(class_="price-value")
            price_match = _PRICE_PATTERN.search(price_element.get_text(" ", strip=True)) if price_element else None
            if price_match is None:
                message = f"CEYPETCO: could not parse a price for product '{name}'"
                logger.error(message)
                raise LiveFuelPriceError(message)
            price = float(price_match.group(1).replace(",", ""))

            date_element = card.find(class_="effective-date")
            date_match = (
                _EFFECTIVE_DATE_PATTERN.search(date_element.get_text(" ", strip=True)) if date_element else None
            )
            if date_match is None:
                message = f"CEYPETCO: could not parse an effective date for product '{name}'"
                logger.error(message)
                raise LiveFuelPriceError(message)
            effective_date = pd.to_datetime(date_match.group(1), format="%d-%m-%Y")

            parsed[name] = (price, effective_date)

        return parsed
