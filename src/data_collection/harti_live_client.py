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
    # Rows printed only in older bulletins (2015 to 2023). Each is its own HARTI row, so each
    # stays its own series: none is merged into a later label without evidence they're the same.
    "brinjals (other)": "Brinjals (Other)",
    "brinjals (village)": "Brinjals (Village)",
    "bitter gourd (other)": "Bitter Gourd (Other)",
    "bitter gourd (village)": "Bitter Gourd (Village)",
    "dambala (wing beans)": "Wing Beans",
    "thithbatu": "Thibbatu",
}

_RANGE_PATTERN = re.compile(r"^\s*([\d,]+(?:\.\d+)?)\s*-\s*([\d,]+(?:\.\d+)?)\s*$")
# Some cells lose their dash in text extraction ("200 250").
_DASHLESS_RANGE_PATTERN = re.compile(r"^\s*([\d,]+(?:\.\d+)?)\s+([\d,]+(?:\.\d+)?)\s*$")
_SCRAMBLED_RANGE_PATTERN = re.compile(r"^\s*-\s*([\d,]+(?:\.\d+)?)\s+([\d,]+(?:\.\d+)?)\s*$|^\s*([\d,]+(?:\.\d+)?)\s+([\d,]+(?:\.\d+)?)\s*-\s*$")
_SINGLE_VALUE_PATTERN = re.compile(r"^\s*([\d,]+(?:\.\d+)?)\s*$")
_COMPACT_ALIASES = {key.replace(" ", ""): value for key, value in HARTI_LIVE_VEGETABLE_ALIASES.items()}
# Formats HARTI has used for a market's reporting date. Slash dates are day-first.
_REPORTING_DATE_FORMATS = ("%Y.%m.%d", "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y")
# A market that didn't report keeps showing its last reporting date, so its date may trail
# the bulletin slightly. Anything further apart is a parsing error, not data.
MAX_REPORTING_LAG_DAYS = 7
MARKET_NAMES = (
    "Peliyagoda", "Kandy", "Dambulla", "Meegoda", "Norochchole",
    "Thambuththegama", "Keppetipola", "Nuwaraeliya", "Bandarawela", "Veyangoda", "Pettah",
)
# Spellings older bulletins use for the same market. Pettah (Colombo's former Manning Market) is
# kept as its own market: HARTI lists Peliyagoda separately once the market moved there in 2021.
_MARKET_SPELLINGS = {"kappetipola": "Keppetipola", "t'thegama": "Thambuththegama", "petha": "Pettah"}


def parse_reporting_date(text: object) -> "pd.Timestamp | None":
    """A market column's reporting date, in any format HARTI has used; None if unparseable."""
    cleaned = str(text).strip() if text else ""
    for date_format in _REPORTING_DATE_FORMATS:
        parsed = pd.to_datetime(cleaned, format=date_format, errors="coerce")
        if pd.notna(parsed):
            return parsed
    return None


def parse_price_range(raw: object) -> tuple[float, float] | None:
    """(low, high) from a "min - max" or single-value cell; None when there is no reading."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text == "-":
        return None
    match = _RANGE_PATTERN.match(text) or _DASHLESS_RANGE_PATTERN.match(text)
    if match:
        low, high = (float(group.replace(",", "")) for group in match.groups())
        return low, high
    # Older bulletins wrap a range over two lines, so text extraction can return "- 115.00 110.00":
    # one dash and two numbers is still exactly one min-max range, whichever order they come out in.
    scrambled = _SCRAMBLED_RANGE_PATTERN.match(text)
    if scrambled:
        first, second = (float(group.replace(",", "")) for group in scrambled.groups() if group is not None)
        return min(first, second), max(first, second)
    single = _SINGLE_VALUE_PATTERN.match(text)
    if single:
        value = float(single.group(1).replace(",", ""))
        return value, value
    logger.warning("HartiLiveClient: could not parse price cell %r; skipping", raw)
    return None


def resolve_vegetable(label: object) -> str | None:
    """Map a bulletin row label to the project's vocabulary, ignoring spacing differences
    (HARTI prints both "Beet Root (N'Eliya)" and "Beet Root(N'Eliya)")."""
    return _COMPACT_ALIASES.get(HartiLiveClient._normalize_label(label).replace(" ", ""))


def canonical_market_name(header: object) -> str | None:
    """A bulletin's market header as its real name. Extraction can truncate a header
    ("Peliyagod") or leave part of a neighbouring cell in it ("Kandy 2024.11.13")."""
    text = str(header or "").lower()
    if "hambuth" in text:  # the leading "T" can land in the neighbouring column
        return "Thambuththegama"
    for spelling, name in _MARKET_SPELLINGS.items():
        if spelling in text.replace(" market", "").split():
            return name
    first_word = text.split(" ")[0] if text else ""
    if len(first_word) < 4:
        return None
    for name in MARKET_NAMES:
        if name.lower().startswith(first_word[:6]) or first_word.startswith(name.lower()[:6]):
            return name
    return None


def parse_all_markets(table_data: list[list], bulletin_date: pd.Timestamp) -> pd.DataFrame:
    """Every market's readings from one bulletin's table.

    `published` is the bulletin's date: the day a reading actually became known. Some
    markets are always reported a day late, so a reading's own `reported` date can't be used
    to decide what a forecast made on a given day could have seen.
    """
    date_row, market_row = table_data[0], table_data[1]
    records = []
    for column, header in enumerate(market_row):
        market = canonical_market_name(header)
        reported = parse_reporting_date(date_row[column]) if market and column < len(date_row) else None
        if reported is None or not 0 <= (bulletin_date - reported).days <= MAX_REPORTING_LAG_DAYS:
            continue
        for row in table_data[2:]:
            vegetable = resolve_vegetable(row[0]) if row and row[0] else None
            price_range = parse_price_range(row[column]) if vegetable and column < len(row) else None
            if price_range:
                records.append(
                    {
                        "published": bulletin_date,
                        "reported": reported,
                        "market": market,
                        "vegetable": vegetable,
                        "price": sum(price_range) / 2,
                    }
                )
    return pd.DataFrame(records, columns=["published", "reported", "market", "vegetable", "price"])


def cell_text(words: list, cell_box: tuple | None) -> str:
    """A table cell's text in reading order, from only the words centred inside it.

    Used instead of pymupdf's Table.extract(), which scrambles this PDF's date cells
    ("20261106\\n. .") and price cells ("380 500\\n-") and strips spaces from row labels
    ("BitterGourd"). `words` is page.get_text("words").
    """
    if cell_box is None:
        return ""
    x0, y0, x1, y1 = cell_box
    inside = [
        word
        for word in words
        if x0 <= (word[0] + word[2]) / 2 <= x1 and y0 <= (word[1] + word[3]) / 2 <= y1
    ]
    return " ".join(word[4] for word in sorted(inside, key=lambda word: (round(word[1] / 3), word[0])))


def split_merged_label_rows(rows: list[list], words: list, table, first_row: int) -> list[list]:
    """Give each row its own label where one label cell spans several rows.

    2015-2020 bulletins print "Gingerly / Soya / Potato (Welimada) / Potato (Nuwaraeliya) /
    B'Onion Imported / Big-onion Local" as one tall label cell beside six separately priced rows.
    Each label line is matched to the price row at the same height on the page. Rows whose label
    already reads as one known vegetable are left unchanged, so every other bulletin is unaffected.
    """
    for offset, row in enumerate(rows):
        label = row[0] if row else None
        if not label or resolve_vegetable(label) is not None:
            continue
        box = table.rows[first_row + offset].cells[0]
        if box is None:
            continue
        x0, y0, x1, y1 = box
        inside = [w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
        lines: list[list] = []
        for word in sorted(inside, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
            middle = (word[1] + word[3]) / 2
            if lines and abs(lines[-1][0] - middle) <= 2.5:
                lines[-1][1].append(word)
            else:
                lines.append([middle, [word]])
        if len(lines) < 2 or sum(resolve_vegetable(" ".join(w[4] for w in ws)) is not None for _, ws in lines) < 2:
            continue
        # The spanning row's own box covers every line, so a line goes to the unlabelled row at
        # its height first, and to the spanning row only when no such row exists (its top line).
        own_label = None
        for middle, line_words in lines:
            text = " ".join(w[4] for w in sorted(line_words, key=lambda w: w[0]))
            target = next(
                (
                    other
                    for other in range(offset + 1, len(rows))
                    if not rows[other][0] and table.rows[first_row + other].bbox[1] <= middle <= table.rows[first_row + other].bbox[3]
                ),
                None,
            )
            if target is not None:
                rows[target][0] = text
            elif own_label is None:
                own_label = text
        rows[offset][0] = own_label
    return rows


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
    all_markets: pd.DataFrame | None = None  # parse_all_markets() output for this bulletin


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
        # A long-lived client (the app caches one) avoids re-downloading the 2.8 MB archive
        # page and any past bulletin it has already read. A past bulletin never changes.
        self._archive_entries: list[_ArchiveEntry] | None = None
        self._snapshots_by_date: dict[pd.Timestamp, HartiLiveSnapshot] = {}

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
        if target_date is not None and target_date in self._snapshots_by_date:
            return self._snapshots_by_date[target_date]

        # "Latest" must always see a freshly published bulletin; a past date can reuse the index.
        if target_date is None or self._archive_entries is None:
            self._archive_entries = self._fetch_archive_entries()
        entries = self._archive_entries
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
        if not 0 <= (entry.date - bulletin_date).days <= MAX_REPORTING_LAG_DAYS:
            message = (
                f"HARTI bulletin listed for {entry.date.date()} reports '{self._target_market}' "
                f"as {bulletin_date.date()} -- too far apart to trust"
            )
            logger.error(message)
            raise HartiLiveDataError(message)
        retrieved_at = pd.Timestamp.now()
        logger.info(
            "HartiLiveClient: fetched %d vegetable price(s) for '%s' dated %s from %s",
            len(df),
            self._target_market,
            bulletin_date.date(),
            entry.pdf_url,
        )
        snapshot = HartiLiveSnapshot(
            data=df,
            bulletin_date=bulletin_date,
            retrieved_at=retrieved_at,
            source_url=entry.pdf_url,
            all_markets=parse_all_markets(table_data, entry.date),
        )
        if target_date is not None:
            self._snapshots_by_date[target_date] = snapshot
        return snapshot

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
        tables = soup.find_all("table")  # one table per year, newest first
        if not tables:
            message = "HARTI archive page structure has changed: no bulletin table found"
            logger.error(message)
            raise HartiLiveDataError(message)

        entries: list[_ArchiveEntry] = []
        for table in tables:
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
        separate pages, in an order this client does not rely on). Nor a fixed header
        row: older bulletins have a title row above it.

        Returns rows shaped [date row, market row, data rows...], with each cell read
        by word position -- see cell_text() for why not Table.extract()'s own text.
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
                header_index = next(
                    (
                        index
                        for index, row in enumerate(data[:4])
                        if any(cell and self._normalize_market_header(cell) == target_normalized for cell in row)
                    ),
                    None,
                )
                # A header on row 0 has no date row above it, so it can't be this table.
                if not header_index or len(data) < header_index + 2:
                    continue
                words = page.get_text("words")
                rows = [
                    [cell_text(words, box) or None for box in table.rows[row_index].cells]
                    for row_index in range(header_index - 1, len(data))
                ]
                return split_merged_label_rows(rows, words, table, header_index - 1)

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
        column_date = parse_reporting_date(column_date_text)
        if column_date is None:
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
            canonical = resolve_vegetable(row[0])
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
        price_range = parse_price_range(raw)
        return None if price_range is None else sum(price_range) / 2
