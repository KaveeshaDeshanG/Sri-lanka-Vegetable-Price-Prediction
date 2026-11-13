"""Reads the Central Bank of Sri Lanka's Daily Price Report PDF.

The report (https://www.cbsl.gov.lk/en/statistics/economic-indicators/price-report) is published on
working days as `price_report_YYYYMMDD_e.pdf`. Its "Wholesale and Retail Prices: Selected Food
Commodities" table has, for each market and price type, two columns: an earlier day ("Yesterday" or
"Last Friday") and "Today". Checked against reports from 2021-08-04, 2024-11-08, 2026-11-06 and
2026-11-09: wholesale for Pettah and Dambulla, retail for Pettah, Dambulla and Narahenpita.

Only the "Today" column is kept, dated with the report's own date, so each report adds one reading
per market, price type and item, and no reading is ever counted twice. Columns are found from the
printed headers, never from fixed positions; a report whose headers can't be matched is rejected
with a reason instead of being guessed at. "n.a." stays missing.

These prices are collected by CBSL, separately from HARTI's. They're kept as their own dataset and
are never merged into a HARTI series.
"""

import re
from dataclasses import dataclass, field

import pandas as pd
import pymupdf

TITLE_WORDS = ("Wholesale", "and", "Retail")
MARKETS = ("Pettah", "Dambulla", "Narahenpita", "Marandagahamula")
PRICE_TYPES = {"Wholesale": "wholesale", "Retail": "retail"}
SECTION_HEADINGS = {"VEGETABLES", "FISH", "OTHERS", "RICE", "FRUITS"}
_NUMBER = re.compile(r"^\d{1,3}(,\d{3})*(\.\d+)?$|^\d+(\.\d+)?$")
_DATE_IN_TITLE = re.compile(r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})")
_FILE_DATE = re.compile(r"price_report_(\d{8})")

# CBSL item label -> the crop it describes (None: not a vegetable). Labels as printed.
CBSL_VEGETABLE_ITEMS = {
    "Beans": "Beans",
    "Carrot": "Carrot",
    "Cabbage": "Cabbage",
    "Tomato": "Tomato",
    "Brinjal": "Brinjal",
    "Pumpkin": "Pumpkin",
    "Snake gourd": "Snake Gourd",
    "Green Chilli": "Green Chilli",
    "Lime": "Lime",
    "Red Onion (Local)": "Red Onion",
    "Red Onion (Imp)": "Red Onion",
    "Big Onion (Local)": "Big Onion",
    "Big Onion (Imp)": "Big Onion",
    "Potato (Local)": "Potato",
    "Potato (Imp)": "Potato",
}


@dataclass
class CbslReport:
    report_date: pd.Timestamp | None = None
    rows: list[dict] = field(default_factory=list)
    error: str | None = None


def normalise_item(label: str) -> str:
    """The item name with known misprints corrected: some reports print "(lmp)" for "(Imp)"."""
    return " ".join(label.replace("(lmp)", "(Imp)").split())


def report_date_from_name(file_name: str) -> pd.Timestamp | None:
    match = _FILE_DATE.search(file_name)
    return pd.Timestamp(match.group(1)) if match else None


def _same_line(words, y, tolerance=3.0):
    return [w for w in words if abs(w[1] - y) <= tolerance]


def parse_cbsl_price_report(pdf_bytes: bytes, file_name: str) -> CbslReport:
    """One report's "Today" prices, or an error saying why it couldn't be read. Never raises."""
    expected = report_date_from_name(file_name)
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:  # noqa: BLE001 -- pymupdf's exception types vary by corruption mode
        return CbslReport(error=f"could not open PDF: {error}")

    for page in document:
        words = page.get_text("words")
        title = next(
            (
                w for i, w in enumerate(words)
                if w[4] == "Wholesale" and i + 2 < len(words) and (words[i + 1][4], words[i + 2][4]) == TITLE_WORDS[1:]
            ),
            None,
        )
        if title is None:
            continue
        return _read_table(page, words, title, expected)
    return CbslReport(error="no 'Wholesale and Retail Prices' table found")


def _read_table(page, words, title, expected: pd.Timestamp | None) -> CbslReport:
    match = _DATE_IN_TITLE.search(" ".join(w[4] for w in _same_line(words, title[1])))
    if not match:
        return CbslReport(error="the table title has no date")
    report_date = pd.Timestamp(f"{match.group(1)} {match.group(2)} {match.group(3)}")
    if expected is not None and report_date != expected:
        return CbslReport(error=f"the table is dated {report_date.date()}, but the file name says {expected.date()}")

    todays = sorted((w for w in words if w[4] == "Today" and title[1] < w[1] < title[1] + 120), key=lambda w: w[0])
    if not todays:
        return CbslReport(report_date, error="no 'Today' column headings")
    header_bottom = max(w[3] for w in todays)
    groups = [w for w in words if w[4] in PRICE_TYPES and title[3] < w[1] < todays[0][1]]
    markets = [w for w in words if w[4] in MARKETS and title[3] < w[1] < todays[0][1]]
    if not groups or not markets:
        return CbslReport(report_date, error="no wholesale/retail or market headings above the 'Today' columns")

    columns = []
    previous_x = -1.0
    for today in todays:
        label = max((m for m in markets if previous_x < m[0] < today[0]), key=lambda m: m[0], default=None)
        if label is None:
            return CbslReport(report_date, error=f"no market heading for the 'Today' column at x={today[0]:.0f}")
        centre = (today[0] + today[2]) / 2
        group = min(groups, key=lambda g: abs((g[0] + g[2]) / 2 - centre))
        columns.append({"market": label[4], "price_type": PRICE_TYPES[group[4]], "x0": today[0] - 25, "x1": today[2] + 25})
        previous_x = today[0]
    if len({(c["market"], c["price_type"]) for c in columns}) != len(columns):
        return CbslReport(report_date, error="two 'Today' columns resolved to the same market and price type")

    rows = []
    units = sorted((w for w in words if w[4].lower().startswith("rs./") and w[1] > header_bottom), key=lambda w: w[1])
    for unit in units:
        line = sorted(_same_line(words, unit[1]), key=lambda w: w[0])
        printed = " ".join(w[4] for w in line if w[2] <= unit[0] + 1 and w is not unit).strip()
        if not printed or printed.upper() in SECTION_HEADINGS:
            continue
        item = normalise_item(printed)
        for column in columns:
            cell = [w[4] for w in line if column["x0"] <= w[0] and w[2] <= column["x1"] and w[0] > unit[2]]
            # The band around a 'Today' heading excludes the earlier day's column to its left.
            if not cell:
                continue  # this item isn't reported for this market and price type at all
            value = cell[-1]
            if value.lower() == "n.a.":
                price = None  # printed as not available: kept as missing, never filled in
            elif _NUMBER.match(value):
                price = float(value.replace(",", ""))
            else:
                continue
            rows.append(
                {
                    "date": report_date,
                    "item": item,
                    "item_as_printed": printed,
                    "vegetable": CBSL_VEGETABLE_ITEMS.get(item),
                    "unit": unit[4],
                    "market": column["market"],
                    "price_type": column["price_type"],
                    "price": price,
                }
            )
    if not any(r["price"] is not None for r in rows):
        return CbslReport(report_date, error="table found, but no prices could be read")
    return CbslReport(report_date, rows)
