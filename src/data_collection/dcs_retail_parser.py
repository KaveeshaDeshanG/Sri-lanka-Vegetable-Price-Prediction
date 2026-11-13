"""Reads the Department of Census and Statistics' weekly retail price bulletins (DCSB-WRP PDFs).

Each bulletin is one table: "OPEN MARKET WEEKLY AVERAGE RETAIL PRICES - <n>th Week of <Month>
<Year>", main markets in Colombo District. Per item it prints the average retail price for the
same week a year earlier, the previous week and this week, two percentage changes, and this
week's price range. Only this week's average and range are kept, so overlapping bulletins never
produce two values for one week. These are RETAIL prices for Colombo District, a different price
type and place from HARTI's wholesale market prices, and are never merged with them.

DCS publishes the week only as "<n>th week of <month>", not as dates, so the week is kept as
published (year, month, week_of_month). No calendar dates are invented for it.
"""

import re
from dataclasses import dataclass, field

import pandas as pd
import pymupdf

TITLE_PATTERN = re.compile(
    r"RETAIL\s+PRICES\s*-\s*(\d)\s*(?:st|nd|rd|th)\s+Week\s+of\s+([A-Za-z]+)\.?\s*(\d{4})", re.IGNORECASE
)
FILE_PATTERN = re.compile(r"DCSB-WRP-(\d{4})-(\d{2})-W(\d)", re.IGNORECASE)
MONTHS = {name: number for number, name in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], start=1
)}
MONTHS["otober"] = 10  # a misspelling in the 3rd week of October 2020 bulletin's own title
_NUMBER = re.compile(r"^-?\d[\d,]*\.\d+$|^-?\d[\d,]*$")
_PERCENT = re.compile(r"^-?\d[\d,]*(?:\.\d+)?%$")
COLUMNS = ["last_year_avg", "previous_week_avg", "this_week_avg", "change_vs_last_year", "change_vs_previous_week", "range_low", "range_high"]
# Raw DCS item text (after joining sub-items to their group) -> project vegetable name. Items not
# listed here (rice, fish, spices, ...) aren't vegetables and are left out.
DCS_VEGETABLE_ITEMS = {
    "ash plantain": "Ash Plantain",
    "ash pumpkin": "Ash Pumpkin",
    "bandakka": "Ladies Fingers",
    "brinjal": "Brinjal",
    "bitter guard": "Bitter Gourd",
    "bitter gourd": "Bitter Gourd",
    "cucumber": "Cucumber",
    "red pumpkin": "Pumpkin",
    "snake gourd": "Snake Gourd",
    "gotukola": "Gotukola",
    "kankun": "Kankun",
    "kathurumurunga": "Kathurumurunga",
    "kohila leaves": "Kohila Leaves",
    "kohila yams": "Kohila Yams",
    "vetakolu": "Luffa",
    "green chillies": "Green Chillies",
    "capsicum": "Capsicum",
    "leeks": "Leeks",
    "limes": "Lime",
    "potatoes - local": "Potato (Local)",
    "potatoes - imported": "Potato (Imported)",
    "mukunuwenna": "Mukunuwenna",
    "nivithi": "Nivithi",
    "sarana": "Sarana",
    "thampala": "Thampala",
    "beans - butter": "Beans (Butter)",
    "beans - green": "Beans (Green)",
    "long beans": "Long Beans",
    "beetroot": "Beetroot",
    "beet root": "Beetroot",
    "cabbagge seed": "Cabbage",
    "cabbage seed": "Cabbage",
    "cabbage": "Cabbage",
    "carrot": "Carrot",
    "drumstick": "Drumstick",
    "drumsticks": "Drumstick",
    "knol khol": "Knol Khol",
    "raddish": "Radish",
    "radish": "Radish",
    "tomatoe - no 1.": "Tomato",
    "tomatoe - no 1": "Tomato",
    "tomato - no 1.": "Tomato",
    "tomatoes - no 1.": "Tomato",
    "red onions - average": "Red Onion",
    "b.onions - local": "Big Onion (Local)",
    "b.onions - imported": "Big Onion (Imported)",
}


@dataclass
class DcsBulletin:
    year: int | None = None
    month: int | None = None
    week_of_month: int | None = None
    rows: list[dict] = field(default_factory=list)
    error: str | None = None


def _lines(page) -> list[list]:
    lines: list[list] = []
    for word in sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        middle = (word[1] + word[3]) / 2
        if lines and abs(lines[-1][0] - middle) <= 3:
            lines[-1][1].append(word)
        else:
            lines.append([middle, [word]])
    return [sorted(words, key=lambda w: w[0]) for _, words in lines]


def _to_float(token: str) -> float:
    return float(token.replace(",", "").rstrip("%"))


def parse_dcs_weekly_pdf(pdf_bytes: bytes, file_name: str = "") -> DcsBulletin:
    """Every priced item of one DCS weekly retail bulletin. Never raises for a bad PDF: reports it."""
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:  # noqa: BLE001 -- pymupdf's exception types vary by corruption mode
        return DcsBulletin(error=f"could not open PDF: {error}")
    first_text = " ".join(document[0].get_text().split()) if len(document) else ""
    title = TITLE_PATTERN.search(first_text)
    if not title or title.group(2).lower() not in MONTHS:
        return DcsBulletin(error="no 'RETAIL PRICES - <n> Week of <Month> <Year>' title")
    result = DcsBulletin(year=int(title.group(3)), month=MONTHS[title.group(2).lower()], week_of_month=int(title.group(1)))
    named = FILE_PATTERN.search(file_name)
    if named and (int(named.group(1)), int(named.group(2)), int(named.group(3))) != (result.year, result.month, result.week_of_month):
        result.error = f"title week {result.year}-{result.month:02d} W{result.week_of_month} doesn't match the file name {file_name}"
        return result

    for page in document:
        group = ""
        lines = _lines(page)
        # Column positions differ between years, so they are read from this page's own headings:
        # the title's first word marks the left margin, and the "Unit" heading the unit column.
        margin = next((w[0] for words in lines for w in words if w[4] == "OPEN"), None)
        unit_x = next((w[0] for words in lines for w in words if w[4] == "Unit"), None)
        if margin is None or unit_x is None:
            continue
        has_range = any(w[4].lower() == "range" for words in lines[:12] for w in words)
        for words in lines:
            unit_start = next((i for i, w in enumerate(words) if unit_x - 22 <= w[0] <= unit_x + 6), None)
            if unit_start is None or unit_start == 0 or words[0][0] > unit_x - 22:
                continue
            item_words = words[:unit_start]
            values = [w for w in words[unit_start:] if _NUMBER.match(w[4]) or _PERCENT.match(w[4])]
            unit_words = [w for w in words[unit_start:] if w[0] < unit_x + 30]
            item = " ".join(w[4] for w in item_words).strip()
            if not item:
                continue
            if item_words[0][0] < margin + 10:  # a top-level item; an indented one continues the group above
                group = item.split(" -")[0].strip() if " -" in item else item
                full_item = item
            else:
                full_item = f"{group} - {item}"
            numbers = [w for w in values if w[0] >= unit_x + 30]
            # Some bulletins (for example during 2020's lockdowns) print no price-range columns.
            expected = 7 if has_range else 5
            if len(numbers) != expected:
                continue  # no price this week, or a row that doesn't read cleanly: skipped, not guessed
            parsed = dict(zip(COLUMNS, (_to_float(w[4]) for w in numbers), strict=False))
            parsed.setdefault("range_low", None)
            parsed.setdefault("range_high", None)
            if not (_PERCENT.match(numbers[3][4]) and _PERCENT.match(numbers[4][4])):
                continue
            result.rows.append(
                {
                    "item": full_item,
                    "unit": " ".join(w[4] for w in unit_words),
                    "vegetable": DCS_VEGETABLE_ITEMS.get(" ".join(full_item.lower().split())),
                    **parsed,
                }
            )
    if not result.rows:
        result.error = "no priced rows found"
    return result
