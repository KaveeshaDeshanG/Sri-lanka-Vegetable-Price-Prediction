"""Live client for HARTI's own Daily Food Commodities Bulletin archive.

Investigated directly against the real site (2026-08): HARTI (harti.gov.lk) has no API,
no CSV/Excel export, and no queryable form for "today's" prices. What actually exists is
a static archive page (`harti.archive_url`, default `daily-price.php`) listing one row per
calendar day -- date, language, and a link to that day's PDF bulletin -- going back to
2015. Each PDF's vegetable-price table has one column per collection market (Peliyagoda,
Kandy, Dambulla, Meegoda, Norochchole, Thambuththegama, Keppetipola, Nuwaraeliya,
Bandarawela, Veyangoda), each showing its OWN reporting date (a market that didn't report
"today" simply shows its last reported date instead) and a "min - max" wholesale price
range, or "-" when that market has no reading for a vegetable that day.

This is therefore "latest available HARTI bulletin data", not real-time and not an API --
HARTI itself typically publishes 1-2 days behind, and not every market reports every day.
Callers must not describe this as real-time.

Mirrors live_weather_client.py's design: one exception type for every failure mode
(network, timeout, HTTP error, changed page/PDF structure, no data), a DataFrame output
in the project's existing market-price schema (date, vegetable, market, price, source),
and validation via the same DataValidator used everywhere else -- so this is a drop-in
alternative market-price *source*, not a special case downstream code needs to branch on.
"""

import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin

import pandas as pd
import pymupdf
import requests
from bs4 import BeautifulSoup

from config.config_loader import ConfigLoader

from .data_validator import DataValidationError, DataValidator

logger = logging.getLogger(__name__)

DEFAULT_ARCHIVE_URL = "https://www.harti.gov.lk/daily-price.php"
DEFAULT_TARGET_MARKET = "Dambulla"
_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; veggie-price-prediction/1.0)"}

# Verified against a real, unmocked bulletin (2026-08-18): every one of this project's
# 31 real vegetable categories (see data/raw/harti_prices.csv) appears in HARTI's own
# live table under one of these exact labels. Keys are normalized (see _normalize_label)
# -- lowercased, whitespace-collapsed -- so "Beet root (N Eliya)" and "beet root (n
# eliya)" resolve identically. Values are the CURRENT real harti_prices.csv vocabulary,
# not the (partly stale) 30-name list in README.md -- e.g. HARTI's live bulletin lists
# "Eggplant" and "Brinjals" as two separate rows, matching the raw CSV's real vocabulary,
# not the README's "Brinjal, Brinjals" pairing (see reports/PROJECT_REPORT.md's audit
# history for that discrepancy). Expand this table if HARTI's own labels change.
HARTI_LIVE_VEGETABLE_ALIASES: dict[str, str] = {
    "beans": "Beans",
    "carrot": "Carrot",
    "leeks": "Leeks",
    "beet root": "Beetroot",
    "beetroot": "Beetroot",
    "beet root (n eliya)": "Beetroot (Nuwara Eliya)",
    "beet root (n'eliya)": "Beetroot (Nuwara Eliya)",
    "beetroot (nuwara eliya)": "Beetroot (Nuwara Eliya)",
    "knolkhol": "Knol Khol",
    "knol khol": "Knol Khol",
    "raddish": "Radish",
    "radish": "Radish",
    "cabbage (n'eliya)": "Cabbage (Nuwara Eliya)",
    "cabbage (nuwara eliya)": "Cabbage (Nuwara Eliya)",
    "cabbage (kandy)": "Cabbage (Kandy)",
    "tomato": "Tomato",
    "ladies fingers": "Ladies Fingers",
    "brinjals": "Brinjals",
    "brinjal": "Brinjals",
    "capsicum": "Capsicum",
    "pumpkin": "Pumpkin",
    "cucumber": "Cucumber",
    "bitter gourd": "Bitter Gourd",
    "snake gourd": "Snake Gourd",
    "drumstick": "Drumstick",
    "luffa": "Luffa",
    "long beans": "Long Beans",
    "ash plantains": "Ash Plantain",
    "ash plantain": "Ash Plantain",
    "green chillies": "Green Chillies",
    "lime": "Lime",
    "sweet potatoe": "Sweet Potato",
    "sweet potato": "Sweet Potato",
    "manioc": "Manioc",
    "eggplant": "Eggplant",
    "potato(imported)": "Potato (Imported)",
    "potato (imported)": "Potato (Imported)",
    "potato (welimada)": "Potato (Welimada)",
    "potato (nuwaraeliya)": "Potato (Nuwara Eliya)",
    "potato (nuwara eliya)": "Potato (Nuwara Eliya)",
    "b'onion imported": "Big Onion (Imported)",
    "big onion imported": "Big Onion (Imported)",
    "big-onion local": "Big Onion (Local)",
    "big onion local": "Big Onion (Local)",
}

_RANGE_PATTERN = re.compile(r"^\s*([\d,]+(?:\.\d+)?)\s*-\s*([\d,]+(?:\.\d+)?)\s*$")
_SINGLE_VALUE_PATTERN = re.compile(r"^\s*([\d,]+(?:\.\d+)?)\s*$")


class HartiLiveDataError(Exception):
    """Raised when HARTI's latest bulletin cannot be found, fetched, parsed, or validated.

    Covers network failure, timeout, non-200 HTTP status, a changed archive-page or
    PDF-table structure, a missing target-market column, an unparseable reporting date,
    or a bulletin with no usable price for any tracked vegetable. Never partially
    returns a result -- either a fully validated HartiLiveSnapshot comes back, or this
    is raised, exactly like LiveWeatherError's contract for live weather.
    """


@dataclass(frozen=True)
class _ArchiveEntry:
    date: pd.Timestamp
    medium: str
    pdf_url: str


@dataclass(frozen=True)
class HartiLiveSnapshot:
    """One fetched HARTI bulletin's prices for the configured target market.

    data: columns [date, vegetable, market, price, "source"] (source="HARTI_LIVE"),
        one row per vegetable the bulletin reported a price for in `market` --
        vegetables with no reading that day ("-" in the PDF) are simply absent,
        never filled with a fabricated value.
    bulletin_date: the reporting date HARTI's own table lists for `market`'s column
        specifically -- NOT necessarily "today", since a market that didn't report
        keeps showing its last reported date, and not necessarily the bulletin
        PDF's own cover date, since different markets can lag by different amounts.
    retrieved_at: wall-clock time this snapshot was fetched, for UI/caching display.
    source_url: the exact PDF URL fetched, for traceability.
    """

    data: pd.DataFrame
    bulletin_date: pd.Timestamp
    retrieved_at: pd.Timestamp
    source_url: str


class HartiLiveClient:
    """Fetches the latest (or a specific) HARTI daily bulletin's prices for one market.

    Never trusts a hardcoded page number or filename pattern -- HARTI's own filename
    convention has already changed at least twice historically (see the archive's
    older `daily_DD-MM-YYYY.pdf` entries vs. today's differently-named ones), so every
    fetch re-reads the archive's own table for the exact current link, and searches
    every page of the downloaded PDF for whichever table has the configured target
    market's column, rather than assuming a fixed page index.
    """

    def __init__(
        self,
        archive_url: str = DEFAULT_ARCHIVE_URL,
        timeout_seconds: float = 15.0,
        target_market: str = DEFAULT_TARGET_MARKET,
        date_column: str = "date",
        vegetable_column: str = "vegetable",
        market_column: str = "market",
        price_column: str = "price",
    ) -> None:
        self._archive_url = archive_url
        self._timeout_seconds = timeout_seconds
        self._target_market = target_market
        self._date_column = date_column
        self._vegetable_column = vegetable_column
        self._market_column = market_column
        self._price_column = price_column
        self._validator = DataValidator()

    @classmethod
    def from_config(cls, config_path: str = "config/config.yaml") -> "HartiLiveClient":
        """Build a client from config.yaml's harti section.

        Every value has a safe default (see DEFAULT_ARCHIVE_URL/DEFAULT_TARGET_MARKET)
        since, unlike weather_api's coordinates, there is no meaningful "must be
        explicitly configured" value here -- harti.latest_data_enabled is checked by
        callers (app/services.py), not this class, mirroring how weather_api.enabled
        is checked by build_live_prediction_input rather than inside
        OpenMeteoWeatherClient itself.
        """
        config = ConfigLoader(config_path)
        return cls(
            archive_url=config.get("harti.archive_url", DEFAULT_ARCHIVE_URL),
            timeout_seconds=config.get("harti.request_timeout_seconds", 15.0),
            target_market=config.get("harti.target_market", DEFAULT_TARGET_MARKET),
            date_column=config.get("data_schema.date_column", "date"),
            vegetable_column=config.get("data_schema.vegetable_column", "vegetable"),
            market_column=config.get("data_schema.market_column", "market"),
            price_column=config.get("data_schema.price_column", "price"),
        )

    def fetch_latest_prices(self) -> HartiLiveSnapshot:
        """Fetch the most recently published HARTI bulletin's prices for the target market."""
        return self._fetch(target_date=None)

    def fetch_prices_for_date(self, target_date: "pd.Timestamp | str") -> HartiLiveSnapshot:
        """Fetch a specific date's HARTI bulletin's prices for the target market.

        Raises:
            HartiLiveDataError: if no bulletin was published for that exact date --
                a legitimate, expected outcome (HARTI does not publish every calendar
                day), not necessarily a sign the archive's structure has changed.
        """
        return self._fetch(target_date=pd.Timestamp(target_date))

    # ------------------------------------------------------------------
    # Archive index (daily-price.php): one row per published bulletin day
    # ------------------------------------------------------------------

    def _fetch(self, target_date: "pd.Timestamp | None") -> HartiLiveSnapshot:
        entries = self._fetch_archive_entries()
        english_entries = [e for e in entries if e.medium.strip().lower() == "english"]
        if not english_entries:
            raise HartiLiveDataError(
                "HARTI archive contained no English-language bulletin entries"
            )

        if target_date is None:
            entry = max(english_entries, key=lambda e: e.date)
        else:
            matches = [e for e in english_entries if e.date == target_date]
            if not matches:
                raise HartiLiveDataError(
                    f"No HARTI bulletin found for {target_date.date()} -- HARTI may not "
                    "have published one for that day"
                )
            entry = matches[0]

        pdf_bytes = self._download_pdf(entry.pdf_url)
        table_data = self._extract_vegetable_table(pdf_bytes)
        df = self._parse_table_rows(table_data)
        self._validate(df)

        bulletin_date = df[self._date_column].iloc[0]
        retrieved_at = pd.Timestamp.now()
        logger.info(
            "HartiLiveClient: fetched %d vegetable price(s) for '%s' dated %s from %s",
            len(df),
            self._target_market,
            bulletin_date.date(),
            entry.pdf_url,
        )
        return HartiLiveSnapshot(
            data=df, bulletin_date=bulletin_date, retrieved_at=retrieved_at, source_url=entry.pdf_url
        )

    def _fetch_archive_entries(self) -> list[_ArchiveEntry]:
        try:
            response = requests.get(
                self._archive_url, timeout=self._timeout_seconds, headers=_REQUEST_HEADERS
            )
        except requests.exceptions.Timeout as error:
            message = f"HARTI archive request timed out after {self._timeout_seconds}s"
            logger.error(message)
            raise HartiLiveDataError(message) from error
        except requests.exceptions.RequestException as error:
            message = f"HARTI archive request failed: {error}"
            logger.error(message)
            raise HartiLiveDataError(message) from error

        if response.status_code != 200:
            message = f"HARTI archive returned HTTP {response.status_code}"
            logger.error(message)
            raise HartiLiveDataError(message)

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        if table is None:
            message = "HARTI archive page structure has changed: no bulletin table found"
            logger.error(message)
            raise HartiLiveDataError(message)

        entries: list[_ArchiveEntry] = []
        body = table.find("tbody") or table
        for row in body.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            date_text = cells[0].get_text(strip=True)
            medium_text = cells[1].get_text(strip=True)
            link = cells[2].find("a")
            if link is None or not link.get("href"):
                continue
            parsed_date = pd.to_datetime(date_text, format="%Y-%m-%d", errors="coerce")
            if pd.isna(parsed_date):
                continue  # skip an unparseable row rather than failing the whole archive
            entries.append(
                _ArchiveEntry(
                    date=parsed_date,
                    medium=medium_text,
                    pdf_url=urljoin(self._archive_url, link["href"]),
                )
            )

        if not entries:
            message = "HARTI archive page contained no usable bulletin entries"
            logger.error(message)
            raise HartiLiveDataError(message)
        return entries

    # ------------------------------------------------------------------
    # PDF bulletin: download + locate the target market's table
    # ------------------------------------------------------------------

    def _download_pdf(self, url: str) -> bytes:
        try:
            response = requests.get(url, timeout=self._timeout_seconds, headers=_REQUEST_HEADERS)
        except requests.exceptions.Timeout as error:
            message = f"HARTI bulletin PDF request timed out after {self._timeout_seconds}s"
            logger.error(message)
            raise HartiLiveDataError(message) from error
        except requests.exceptions.RequestException as error:
            message = f"HARTI bulletin PDF request failed: {error}"
            logger.error(message)
            raise HartiLiveDataError(message) from error

        if response.status_code != 200:
            message = f"HARTI bulletin PDF returned HTTP {response.status_code} for '{url}'"
            logger.error(message)
            raise HartiLiveDataError(message)
        if not response.content:
            message = f"HARTI bulletin PDF at '{url}' was empty"
            logger.error(message)
            raise HartiLiveDataError(message)
        return response.content

    def _extract_vegetable_table(self, pdf_bytes: bytes) -> list[list]:
        """Search every page for whichever table has a column headed by the target market.

        Never assumes a fixed page index: HARTI's bulletin has changed its page count/
        section order before (rice/subsidiary crops vs. vegetables vs. fruits are
        separate pages, in an order this client does not rely on).
        """
        try:
            document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as error:  # pymupdf's own exception types vary by corruption mode
            message = f"Could not open HARTI bulletin PDF: {error}"
            logger.error(message)
            raise HartiLiveDataError(message) from error

        target_normalized = self._normalize_market_header(self._target_market)
        for page in document:
            try:
                found_tables = page.find_tables()
            except Exception as error:  # noqa: BLE001 -- one page's detection failing shouldn't abort the search
                logger.debug("HartiLiveClient: table detection failed on one page: %s", error)
                continue
            for table in found_tables.tables:
                data = table.extract()
                if len(data) < 3:
                    continue
                market_row = data[1]
                if any(
                    cell and self._normalize_market_header(cell) == target_normalized
                    for cell in market_row
                ):
                    return data

        message = (
            f"HARTI bulletin structure has changed: no table with a '{self._target_market}' "
            "market column was found on any page"
        )
        logger.error(message)
        raise HartiLiveDataError(message)

    def _parse_table_rows(self, table_data: list[list]) -> pd.DataFrame:
        """Pure parsing step (no I/O) -- kept separate from PDF/network access so it can
        be unit-tested directly against hand-built fixture rows, not just a real PDF."""
        date_row, market_row = table_data[0], table_data[1]
        target_normalized = self._normalize_market_header(self._target_market)
        column_index = next(
            (
                i
                for i, cell in enumerate(market_row)
                if cell and self._normalize_market_header(cell) == target_normalized
            ),
            None,
        )
        if column_index is None:
            message = f"No '{self._target_market}' market column found in the extracted table"
            logger.error(message)
            raise HartiLiveDataError(message)

        column_date_text = date_row[column_index] if column_index < len(date_row) else None
        column_date = (
            pd.to_datetime(column_date_text, format="%Y.%m.%d", errors="coerce")
            if column_date_text
            else pd.NaT
        )
        if pd.isna(column_date):
            message = (
                f"Could not parse a reporting date for the '{self._target_market}' column "
                f"(raw value: {column_date_text!r})"
            )
            logger.error(message)
            raise HartiLiveDataError(message)

        records = []
        for row in table_data[2:]:
            if not row or row[0] is None:
                continue
            canonical = HARTI_LIVE_VEGETABLE_ALIASES.get(self._normalize_label(row[0]))
            if canonical is None:
                continue  # not one of this project's tracked vegetables (fruit/rice/section header/etc.)
            cell_value = row[column_index] if column_index < len(row) else None
            price = self._parse_price_cell(cell_value)
            if price is None:
                continue  # no reading for this vegetable today -- never fabricated
            records.append({self._vegetable_column: canonical, self._price_column: price})

        if not records:
            message = (
                f"HARTI bulletin for {column_date.date()} contained no usable "
                f"'{self._target_market}' price for any tracked vegetable"
            )
            logger.error(message)
            raise HartiLiveDataError(message)

        df = pd.DataFrame.from_records(records)
        df[self._date_column] = column_date
        df[self._market_column] = self._target_market
        df["source"] = "HARTI_LIVE"
        return df[
            [self._date_column, self._vegetable_column, self._market_column, self._price_column, "source"]
        ]

    def _validate(self, df: pd.DataFrame) -> None:
        try:
            self._validator.validate_positive(df, self._price_column, "HartiLive")
            self._validator.validate_date_not_missing(df, self._date_column, "HartiLive")
        except DataValidationError as error:
            message = f"HARTI live data failed validation: {error}"
            logger.error(message)
            raise HartiLiveDataError(message) from error

        duplicate_mask = df[self._vegetable_column].duplicated()
        if duplicate_mask.any():
            duplicates = sorted(df.loc[duplicate_mask, self._vegetable_column].unique().tolist())
            message = f"HARTI bulletin unexpectedly listed duplicate vegetable row(s): {duplicates}"
            logger.error(message)
            raise HartiLiveDataError(message)

    @staticmethod
    def _normalize_label(value: object) -> str:
        return " ".join(str(value).replace("\n", " ").split()).strip().lower()

    @classmethod
    def _normalize_market_header(cls, value: object) -> str:
        normalized = cls._normalize_label(value)
        return normalized.removesuffix(" market")

    @staticmethod
    def _parse_price_cell(raw: object) -> float | None:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text or text == "-":
            return None

        range_match = _RANGE_PATTERN.match(text)
        if range_match:
            low, high = (float(group.replace(",", "")) for group in range_match.groups())
            return (low + high) / 2

        single_match = _SINGLE_VALUE_PATTERN.match(text)
        if single_match:
            return float(single_match.group(1).replace(",", ""))

        logger.warning("HartiLiveClient: could not parse price cell %r; skipping", raw)
        return None
