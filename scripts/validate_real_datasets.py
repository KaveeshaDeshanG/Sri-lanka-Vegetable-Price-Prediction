"""Validates the rebuilt raw datasets and writes reports/DATA_QUALITY_REPORT.md.

Reads data only: it never changes a dataset, and it never trains or saves a model. It
reports every issue it finds and leaves the decision to the reader.

The PDF cross-check takes a fixed-seed random sample of harti_prices_detail.csv and tests
each record two ways that do not use the build script's table detection:
  1. Re-reads the value from the PDF by page position alone (market header x-range and row
     label y-position).
  2. Saves image crops of the original PDF cells to reports/figures/ for reading by eye.
     The crops show no extracted value, so the extracted number can't influence the reader.

Usage:
    .venv312/Scripts/python.exe scripts/validate_real_datasets.py --cache-dir <pdf cache folder>
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
from PIL import Image, ImageDraw

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.build_harti_prices import (  # noqa: E402
    MAX_REPORTING_LAG_DAYS,
    extract_market_rows,
    hash_url,
    parse_price_range,
    resolve_vegetable,
)
from src.data_collection.harti_live_client import (  # noqa: E402
    HARTI_LIVE_VEGETABLE_ALIASES,
    MARKET_NAMES,
    canonical_market_name,
)

MARKET = "Dambulla"
REGIONS_PER_FILE = 8
SAMPLE_PER_YEAR = 14
SAMPLE_SEED = 20261106
STRIPS_PER_SHEET = 14
JUMP_RATIO = 3.0
WIDE_RANGE_RATIO = 3.0
OUTLIER_RATIO = 5.0
MARKET_SAMPLE_SIZE = 4
_TRAINING_NAMES = {"Eggplant": "Brinjal"}


def pdf_path(cache_dir: Path, bulletin_date: pd.Timestamp, url: str) -> Path:
    return cache_dir / f"{bulletin_date:%Y-%m-%d}_{abs(hash_url(url)):016x}.pdf"


def table_markdown(frame: pd.DataFrame) -> str:
    """A GitHub-flavored table, without needing the optional tabulate dependency."""
    if frame.empty:
        return "_None._"
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def longest_gap_days(dates: pd.Series) -> int:
    ordered = pd.Series(sorted(dates.unique()))
    return int(ordered.diff().dt.days.max() - 1) if len(ordered) > 1 else 0


def check_prices(detail: pd.DataFrame) -> dict:
    d = detail.sort_values(["vegetable", "date"]).copy()
    d["previous_price"] = d.groupby("vegetable")["price"].shift()
    d["previous_date"] = d.groupby("vegetable")["date"].shift()
    consecutive = (d["date"] - d["previous_date"]).dt.days == 1
    ratio = d["price"] / d["previous_price"]
    jumps = d[consecutive & ((ratio >= JUMP_RATIO) | (ratio <= 1 / JUMP_RATIO))]
    midpoint_ok = np.isclose(d["price"], (d["low"] + d["high"]) / 2)

    by_day = detail.pivot_table(index="date", columns="vegetable", values="raw_cell", aggfunc="first")
    consecutive_days = by_day.index.to_series().diff().dt.days == 1
    identical = ((by_day == by_day.shift()) | (by_day.isna() & by_day.shift().isna())).all(axis=1) & consecutive_days

    return {
        "non_positive": int((d["price"] <= 0).sum()),
        "missing": int(d["price"].isna().sum()),
        "low_above_high": d[d["low"] > d["high"]],
        "midpoint_mismatch": int((~midpoint_ok).sum()),
        "wide_range": d[d["high"] / d["low"] >= WIDE_RANGE_RATIO],
        "jumps": jumps,
        "identical_days": [str(x.date()) for x in identical[identical].index],
        "summary": d.groupby("vegetable")["price"].agg(rows="count", minimum="min", median="median", maximum="max"),
    }


def recheck_by_position(page: pymupdf.Page, vegetable: str, market: str = MARKET) -> str | None:
    """Read the cell using only word positions: no find_tables(), no table grid."""
    words = page.get_text("words")
    market_words = sorted((w for w in words if w[4].lower() == "market"), key=lambda w: w[0])
    headers = [w for w in words if canonical_market_name(w[4]) == market]
    target = min(headers, key=lambda w: w[1], default=None)
    if target is None or len(market_words) < 2:
        return None
    centres = [(w[0] + w[2]) / 2 for w in market_words]
    index = int(np.argmin([abs(c - (target[0] + target[2]) / 2) for c in centres]))
    half_width = np.median(np.diff(centres)) / 2
    x0, x1 = centres[index] - half_width, centres[index] + half_width
    first_column_end = centres[0] - half_width

    lines: dict[int, list] = {}
    for word in words:
        if word[1] > target[3] and (word[0] + word[2]) / 2 < first_column_end:
            lines.setdefault(round(word[1] / 3), []).append(word)
    for line in lines.values():
        label = " ".join(w[4] for w in sorted(line, key=lambda w: w[0]))
        if resolve_vegetable(label) == vegetable:
            y0, y1 = min(w[1] for w in line) - 1, max(w[3] for w in line) + 1
            inside = [w for w in words if y0 <= (w[1] + w[3]) / 2 <= y1 and x0 <= (w[0] + w[2]) / 2 <= x1]
            return " ".join(w[4] for w in sorted(inside, key=lambda w: w[0]))
    return None


def locate_cells(page: pymupdf.Page, vegetable: str, market: str = MARKET):
    """Rectangles for (row label, reporting date, market header, price) for an image crop."""
    for table in page.find_tables().tables:
        extracted = table.extract()
        header = next(
            (i for i, row in enumerate(extracted[:4]) if any(c and canonical_market_name(c) == market for c in row)), None
        )
        if not header:
            continue
        column = next(i for i, c in enumerate(extracted[header]) if c and canonical_market_name(c) == market)
        for row_index in range(header + 1, len(extracted)):
            box = table.rows[row_index].cells[0]
            if box is None:
                continue
            label = " ".join(w[4] for w in page.get_text("words", clip=pymupdf.Rect(box)))
            if resolve_vegetable(label) == vegetable:
                return (
                    box,
                    table.rows[header - 1].cells[column],
                    table.rows[header].cells[column],
                    table.rows[row_index].cells[column],
                )
    return None


def crop_image(page: pymupdf.Page, box) -> Image.Image:
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=pymupdf.Rect(box))
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def cross_check(detail: pd.DataFrame, cache_dir: Path, figures_dir: Path) -> pd.DataFrame:
    """Automated re-read of the sample, plus image sheets for checking by eye."""
    sample = (
        detail.assign(_year=detail["date"].dt.year)
        .groupby("_year", group_keys=False)
        .sample(n=SAMPLE_PER_YEAR, random_state=SAMPLE_SEED)
        .sort_values("date", ignore_index=True)
    )
    results, strips = [], []
    for number, record in enumerate(sample.itertuples(index=False), start=1):
        document = pymupdf.open(pdf_path(cache_dir, record.bulletin_date, record.source_pdf))
        page = next(p for p in document if MARKET.lower() in p.get_text().lower())
        position_text = recheck_by_position(page, record.vegetable)
        position_range = parse_price_range(position_text) if position_text else None
        results.append(
            {
                "id": number,
                "date": record.date.date(),
                "vegetable": record.vegetable,
                "csv_price": record.price,
                "csv_raw_cell": record.raw_cell,
                "position_reread": position_text,
                "position_match": position_range == (record.low, record.high),
                "source_pdf": record.source_pdf,
            }
        )
        cells = locate_cells(page, record.vegetable)
        if cells is None:
            continue
        crops = [crop_image(page, box) for box in cells]
        height = max(c.height for c in crops) + 6
        strip = Image.new("RGB", (640 + sum(c.width + 12 for c in crops), height), "white")
        ImageDraw.Draw(strip).text((4, 8), f"#{number:02d} {record.date.date()} {record.vegetable}", fill="black", font_size=20)
        x = 400
        for crop in crops:
            strip.paste(crop, (x, 3))
            x += crop.width + 12
        strips.append(strip)

    figures_dir.mkdir(parents=True, exist_ok=True)
    for sheet_number, start in enumerate(range(0, len(strips), STRIPS_PER_SHEET), start=1):
        batch = strips[start : start + STRIPS_PER_SHEET]
        sheet = Image.new("RGB", (max(s.width for s in batch), sum(s.height + 4 for s in batch)), "white")
        y = 0
        for strip in batch:
            sheet.paste(strip, (0, y))
            y += strip.height + 4
        sheet.save(figures_dir / f"pdf_crosscheck_sheet_{sheet_number}.png")
    return pd.DataFrame(results)


def check_duplicate_bulletins(detail: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """Dates HARTI listed more than one PDF for, and whether the PDFs disagree."""
    rows = []
    by_date: dict[str, list[Path]] = {}
    for path in sorted(cache_dir.glob("*.pdf")):
        by_date.setdefault(path.name[:10], []).append(path)
    for date, paths in by_date.items():
        if len(paths) < 2:
            continue
        extracts = [extract_market_rows(p.read_bytes(), pd.Timestamp(date), MARKET) for p in paths]
        values = [{r["vegetable"]: r["raw_cell"] for r in e.rows} for e in extracts]
        differing = sorted(k for k in set().union(*values) if len({v.get(k) for v in values}) > 1)
        rows.append({"date": date, "pdf_count": len(paths), "vegetables_differing": len(differing), "which": ", ".join(differing[:6])})
    return pd.DataFrame(rows)


def read_prices_from_pdf_table(path: Path) -> list[tuple]:
    """Every market's (published, market, vegetable, price) by a second route: the digits in
    pymupdf's Table.extract() cells, not the word-position reading the dataset was built with."""
    published = pd.Timestamp(path.name[:10])
    document = pymupdf.open(path)
    for page in document:
        for table in page.find_tables().tables:
            extracted = table.extract()
            header = next(
                (i for i, row in enumerate(extracted[:4]) if any(c and canonical_market_name(c) == MARKET for c in row)), None
            )
            if not header:
                continue
            columns = {i: canonical_market_name(c) for i, c in enumerate(extracted[header]) if c and canonical_market_name(c)}
            cells = []
            seen = set()
            for row in extracted[header + 1 :]:
                vegetable = resolve_vegetable(row[0]) if row and row[0] else None
                if vegetable is None or vegetable in seen:
                    continue
                seen.add(vegetable)
                for column, market in columns.items():
                    numbers = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", row[column] or "")]
                    if numbers:
                        cells.append((published, market, _TRAINING_NAMES.get(vegetable, vegetable), sum(numbers) / len(numbers), len(numbers)))
            return cells
    return []


def compare_panel_with_pdfs(panel: pd.DataFrame, cache_dir: Path) -> dict:
    """Compare every reading in the dataset with an independent read of its source PDF."""
    paths = sorted(cache_dir.glob("*.pdf"))
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as pool:
        cells = [cell for result in pool.map(read_prices_from_pdf_table, paths, chunksize=8) for cell in result]
    from_pdf = pd.DataFrame(cells, columns=["published", "market", "vegetable", "pdf_price", "numbers_in_cell"])
    key = ["published", "market", "vegetable"]
    joined = panel.merge(from_pdf, on=key, how="outer", indicator=True)
    both = joined[joined["_merge"] == "both"]
    differing = both[~np.isclose(both["price"], both["pdf_price"])]
    return {
        "pdfs_read": len(paths),
        "cells_in_pdfs": len(from_pdf),
        "readings_matching": int(len(both) - len(differing)),
        "price_differs": differing[[*key, "price", "pdf_price", "numbers_in_cell"]],
        "only_in_dataset": joined[joined["_merge"] == "left_only"][[*key, "price"]],
        "only_in_pdfs": joined[joined["_merge"] == "right_only"][[*key, "pdf_price", "numbers_in_cell"]],
    }


def market_sample_check(panel: pd.DataFrame, cache_dir: Path, figures_dir: Path) -> pd.DataFrame:
    """A random sample per market, re-read by position and saved as image crops for reading by eye."""
    sample = (
        panel.groupby("market", group_keys=False)
        .sample(n=MARKET_SAMPLE_SIZE, random_state=SAMPLE_SEED)
        .sort_values(["market", "published"], ignore_index=True)
    )
    results, strips = [], []
    for number, record in enumerate(sample.itertuples(index=False), start=1):
        path = next(cache_dir.glob(f"{record.published:%Y-%m-%d}_*.pdf"))
        page = next(p for p in pymupdf.open(path) if MARKET.lower() in p.get_text().lower())
        # A label on the page uses HARTI's name: "Brinjal" in this dataset is its "Eggplant" row.
        pdf_vegetable = "Eggplant" if record.vegetable == "Brinjal" else record.vegetable
        text = recheck_by_position(page, pdf_vegetable, record.market)
        price_range = parse_price_range(text) if text else None
        results.append(
            {
                "id": number,
                "market": record.market,
                "published": record.published.date(),
                "reported": record.reported.date(),
                "vegetable": record.vegetable,
                "csv_price": record.price,
                "position_reread": text,
                "position_match": price_range is not None and np.isclose(sum(price_range) / 2, record.price),
            }
        )
        cells = locate_cells(page, pdf_vegetable, record.market)
        if cells is None:
            continue
        crops = [crop_image(page, box) for box in cells]
        strip = Image.new("RGB", (640 + sum(c.width + 12 for c in crops), max(c.height for c in crops) + 6), "white")
        label = f"#{number:02d} {record.market[:11]} {record.published.date()} {pdf_vegetable}"
        ImageDraw.Draw(strip).text((4, 8), label, fill="black", font_size=18)
        x = 430
        for crop in crops:
            strip.paste(crop, (x, 3))
            x += crop.width + 12
        strips.append(strip)

    figures_dir.mkdir(parents=True, exist_ok=True)
    for sheet_number, start in enumerate(range(0, len(strips), STRIPS_PER_SHEET), start=1):
        batch = strips[start : start + STRIPS_PER_SHEET]
        sheet = Image.new("RGB", (max(s.width for s in batch), sum(s.height + 4 for s in batch)), "white")
        y = 0
        for strip in batch:
            sheet.paste(strip, (0, y))
            y += strip.height + 4
        sheet.save(figures_dir / f"market_crosscheck_sheet_{sheet_number}.png")
    return pd.DataFrame(results)


def all_markets_section(data_dir: Path, cache_dir: Path, figures_dir: Path, metrics_dir: Path, prices, fuel, weather) -> list[str]:
    panel = pd.read_csv(data_dir / "harti_all_markets.csv", parse_dates=["published", "reported"])
    key = ["published", "market", "vegetable"]
    bulletins = {pd.Timestamp(p.name[:10]) for p in cache_dir.glob("*.pdf")}
    published_dates = set(panel["published"])
    lag = (panel["published"] - panel["reported"]).dt.days

    per_market = (
        panel.assign(lag=lag)
        .groupby("market")
        .agg(
            rows=("price", "size"),
            bulletins=("published", "nunique"),
            first=("published", "min"),
            last=("published", "max"),
            vegetables=("vegetable", "nunique"),
            lag_min=("lag", "min"),
            lag_max=("lag", "max"),
            median_price=("price", "median"),
        )
        .reset_index()
    )
    per_market["bulletin coverage %"] = (per_market["bulletins"] / len(bulletins) * 100).round(1)
    per_market["first"], per_market["last"] = per_market["first"].dt.date, per_market["last"].dt.date

    unknown_markets = sorted(set(panel["market"]) - set(MARKET_NAMES))
    expected_vegetables = {_TRAINING_NAMES.get(v, v) for v in HARTI_LIVE_VEGETABLE_ALIASES.values()}
    unknown_vegetables = sorted(set(panel["vegetable"]) - expected_vegetables)
    per_vegetable = panel.pivot_table(index="vegetable", columns="market", values="price", aggfunc="size", fill_value=0)

    medians = panel.groupby(["market", "vegetable"])["price"].transform("median")
    ratio = panel["price"] / medians
    far_from_median = panel[(ratio >= OUTLIER_RATIO) | (ratio <= 1 / OUTLIER_RATIO)]
    ordered = panel.sort_values(["market", "vegetable", "published"])
    previous = ordered.groupby(["market", "vegetable"])["price"].shift()
    consecutive = ordered.groupby(["market", "vegetable"])["published"].diff().dt.days == 1
    change = ordered["price"] / previous
    jumps = ordered[consecutive & ((change >= JUMP_RATIO) | (change <= 1 / JUMP_RATIO))]

    dambulla = panel[panel["market"] == MARKET]
    # harti_prices.csv keeps HARTI's "Eggplant"; this dataset uses training's name for that row.
    same_names = prices.assign(vegetable=prices["vegetable"].replace(_TRAINING_NAMES))
    joined = dambulla.merge(same_names, left_on=["published", "vegetable"], right_on=["date", "vegetable"], how="outer", suffixes=("", "_csv"), indicator=True)
    both = joined[joined["_merge"] == "both"]
    dambulla_differs = int((~np.isclose(both["price"], both["price_csv"])).sum())

    checks = [
        ("Rows", len(panel)),
        ("Duplicate (published, market, vegetable)", int(panel.duplicated(key).sum())),
        ("Repeated readings, same (reported, market, vegetable) with a later bulletin", int(panel.duplicated(["reported", "market", "vegetable"]).sum())),
        ("Missing values", int(panel.isna().sum().sum())),
        ("Non-positive prices", int((panel["price"] <= 0).sum())),
        ("Market names outside the expected 10", ", ".join(unknown_markets) or 0),
        ("Vegetable names outside the project's vocabulary", ", ".join(unknown_vegetables) or 0),
        ("Reported date after its bulletin", int((lag < 0).sum())),
        ("Reported date more than 7 days before its bulletin", int((lag > 7).sum())),
        ("Published dates with no cached source PDF", len(published_dates - bulletins)),
        ("Cached PDFs contributing no reading", len(bulletins - published_dates)),
        ("Published dates outside fuel_prices.csv", len(published_dates - set(fuel["date"]))),
        ("Published dates outside weather_data.csv", len(published_dates - set(weather["date"]))),
        ("Dambulla rows missing from harti_prices.csv", int((joined["_merge"] == "left_only").sum())),
        ("harti_prices.csv rows missing from Dambulla rows", int((joined["_merge"] == "right_only").sum())),
        ("Dambulla prices that differ from harti_prices.csv", dambulla_differs),
        (f"Prices at least {OUTLIER_RATIO:g}x from their market and vegetable median (needs review)", len(far_from_median)),
        (f"Day-over-day change of at least {JUMP_RATIO:g}x (needs review)", len(jumps)),
    ]

    comparison = compare_panel_with_pdfs(panel, cache_dir)
    sample = market_sample_check(panel, cache_dir, figures_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(metrics_dir / "market_crosscheck_sample.csv", index=False)
    show_outliers = far_from_median.sort_values("price", ascending=False)[["published", "market", "vegetable", "price"]]

    return [
        "## 8. All-markets dataset (harti_all_markets.csv)",
        "",
        table_markdown(pd.DataFrame(checks, columns=["Check", "Result"])),
        "",
        "### Coverage by market",
        "",
        "`lag` is the days between a market's reported date and the bulletin that published it.",
        "",
        table_markdown(per_market),
        "",
        "### Readings per vegetable and market",
        "",
        table_markdown(per_vegetable.reset_index()),
        "",
        "### Comparison with the source PDFs",
        "",
        f"Every reading compared with the digits in `Table.extract()`'s cells, a separate route from the word-position "
        f"reading used to build the dataset. PDFs read: {comparison['pdfs_read']}. Cells found: {comparison['cells_in_pdfs']}. "
        f"Matching: **{comparison['readings_matching']}**. Price differs: **{len(comparison['price_differs'])}**. "
        f"Only in the dataset: **{len(comparison['only_in_dataset'])}**. Only in the PDFs: **{len(comparison['only_in_pdfs'])}**.",
        "",
        "Price differs (up to 40):",
        "",
        table_markdown(comparison["price_differs"].head(40)),
        "",
        "Only in the dataset (up to 40):",
        "",
        table_markdown(comparison["only_in_dataset"].head(40)),
        "",
        "Only in the PDFs (up to 40):",
        "",
        table_markdown(comparison["only_in_pdfs"].head(40)),
        "",
        f"### Random sample ({MARKET_SAMPLE_SIZE} per market, seed {SAMPLE_SEED})",
        "",
        f"Position-based re-read matched: **{int(sample['position_match'].sum())} of {len(sample)}**. "
        "Image crops: `reports/figures/market_crosscheck_sheet_*.png`.",
        "",
        table_markdown(sample),
        "",
        f"### Prices at least {OUTLIER_RATIO:g}x from their median (up to 40, highest first)",
        "",
        table_markdown(show_outliers.head(40)),
        "",
    ]


def build_report(data_dir: Path, cache_dir: Path, figures_dir: Path, metrics_dir: Path) -> str:
    detail = pd.read_csv(data_dir / "harti_prices_detail.csv", parse_dates=["date", "bulletin_date"], keep_default_na=False, na_values=[""])
    prices = pd.read_csv(data_dir / "harti_prices.csv", parse_dates=["date"])
    build = json.loads((data_dir / "harti_prices_build_report.json").read_text(encoding="utf-8"))
    fuel = pd.read_csv(data_dir / "fuel_prices.csv", parse_dates=["date"])
    weather = pd.read_csv(data_dir / "weather_data.csv", parse_dates=["date"])
    regions = pd.read_csv(data_dir / "weather_data_by_region.csv", parse_dates=["date"])

    lines = [
        "# Data Quality Report: Rebuilt Real Datasets",
        "",
        f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} by `scripts/validate_real_datasets.py`. See `reports/DATA_SOURCES.md` for sources.",
        "",
    ]

    # ---- coverage
    span = pd.date_range(prices["date"].min(), prices["date"].max())
    missing_days = span.difference(pd.DatetimeIndex(prices["date"].unique()))
    lines += [
        "## 1. Coverage",
        "",
        table_markdown(
            pd.DataFrame(
                [
                    {"File": "harti_prices.csv", "Rows": len(prices), "First": prices["date"].min().date(), "Last": prices["date"].max().date(), "Distinct dates": prices["date"].nunique()},
                    {"File": "fuel_prices.csv", "Rows": len(fuel), "First": fuel["date"].min().date(), "Last": fuel["date"].max().date(), "Distinct dates": fuel["date"].nunique()},
                    {"File": "weather_data.csv", "Rows": len(weather), "First": weather["date"].min().date(), "Last": weather["date"].max().date(), "Distinct dates": weather["date"].nunique()},
                    {"File": "weather_data_by_region.csv", "Rows": len(regions), "First": regions["date"].min().date(), "Last": regions["date"].max().date(), "Distinct dates": regions["date"].nunique()},
                ]
            )
        ),
        "",
        f"HARTI bulletins listed for the window: **{build['bulletins_listed']}**. Yielding no prices: **{len(build['bulletins_skipped'])}** (see the reason for each below).",
        f"Calendar days in the price span: {len(span)}. Days with no Dambulla price at all: **{len(missing_days)}** ({len(missing_days) / len(span) * 100:.1f}%).",
        f"Bulletins whose Dambulla date trailed the bulletin date (allowed up to {MAX_REPORTING_LAG_DAYS} days): {build['bulletins_where_market_date_trailed_bulletin']}.",
        "",
        "Skipped bulletins:",
        "",
        table_markdown(pd.DataFrame(build["bulletins_skipped"])),
        "",
        "Days with no Dambulla price: " + (", ".join(str(d.date()) for d in missing_days) or "none") + ".",
        "",
    ]

    # ---- per vegetable
    check = check_prices(detail)
    per_veg = check["summary"].reset_index()
    per_veg["coverage %"] = (per_veg["rows"] / len(span) * 100).round(1)
    per_veg["longest gap (days)"] = [longest_gap_days(detail.loc[detail["vegetable"] == v, "date"]) for v in per_veg["vegetable"]]
    per_veg["first"] = [detail.loc[detail["vegetable"] == v, "date"].min().date() for v in per_veg["vegetable"]]
    per_veg["last"] = [detail.loc[detail["vegetable"] == v, "date"].max().date() for v in per_veg["vegetable"]]
    per_veg = per_veg.sort_values("rows")
    lines += ["## 2. Vegetable names and coverage (Rs/kg midpoint)", "", f"{len(per_veg)} distinct labels.", "", table_markdown(per_veg), ""]
    lines += [
        "Row labels in the bulletins not mapped to a tracked vegetable (fruit, section headings, or a label to review):",
        "",
        table_markdown(pd.DataFrame([{"label": k, "bulletins": v} for k, v in build["unknown_row_labels"].items()])),
        "",
    ]

    # ---- record integrity
    key_dupes = int(prices.duplicated(["date", "vegetable", "market"]).sum())
    wide, jumps = check["wide_range"], check["jumps"]
    show = ["date", "vegetable", "raw_cell", "previous_price", "price", "source_pdf"]
    lines += [
        "## 3. Record integrity",
        "",
        table_markdown(
            pd.DataFrame(
                [
                    {"Check": "Duplicate (date, vegetable, market) in harti_prices.csv", "Result": key_dupes},
                    {"Check": "Exact duplicate rows", "Result": int(prices.duplicated().sum())},
                    {"Check": "Missing values in harti_prices.csv", "Result": int(prices.isna().sum().sum())},
                    {"Check": "Non-positive prices", "Result": check["non_positive"]},
                    {"Check": "Low above high", "Result": len(check["low_above_high"])},
                    {"Check": "Price differs from (low+high)/2", "Result": check["midpoint_mismatch"]},
                    {"Check": "Market values", "Result": ", ".join(sorted(prices["market"].unique()))},
                    {"Check": "Build cross-check mismatches (word vs table digits)", "Result": build["cross_check_mismatch_count"]},
                    {"Check": f"High/low range at least {WIDE_RANGE_RATIO:g}x (needs review)", "Result": len(wide)},
                    {"Check": f"Day-over-day price change at least {JUMP_RATIO:g}x (needs review)", "Result": len(jumps)},
                    {"Check": "Consecutive days with every vegetable's cell identical", "Result": len(check["identical_days"])},
                ]
            )
        ),
        "",
        "Identical consecutive days: " + (", ".join(check["identical_days"]) or "none") + ".",
        "",
        f"### Day-over-day changes of at least {JUMP_RATIO:g}x",
        "",
        table_markdown(jumps[show].assign(date=jumps["date"].dt.date).head(60)),
        "",
        f"### High/low ranges of at least {WIDE_RANGE_RATIO:g}x",
        "",
        table_markdown(wide[["date", "vegetable", "raw_cell", "source_pdf"]].assign(date=wide["date"].dt.date).head(60)),
        "",
        "Rows with a low-above-high cell:",
        "",
        table_markdown(check["low_above_high"][["date", "vegetable", "raw_cell", "source_pdf"]]),
        "",
    ]

    # ---- fuel & weather
    fuel_gaps = pd.date_range(fuel["date"].min(), fuel["date"].max()).difference(fuel["date"])
    weather_gaps = pd.date_range(weather["date"].min(), weather["date"].max()).difference(weather["date"])
    region_counts = regions.groupby("region")["date"].agg(rows="count", first="min", last="max", dupes=lambda s: int(s.duplicated().sum())).reset_index()
    price_dates = pd.DatetimeIndex(prices["date"].unique())
    no_fuel = price_dates.difference(fuel["date"])
    no_weather = price_dates.difference(weather["date"])
    no_lag = (price_dates - pd.Timedelta(days=14)).difference(weather["date"])
    no_lag = no_lag[no_lag >= weather["date"].min()]
    no_fuel_lag = (price_dates - pd.Timedelta(days=1)).difference(fuel["date"])
    no_fuel_lag = no_fuel_lag[no_fuel_lag >= fuel["date"].min()]
    lines += [
        "## 4. Fuel and weather",
        "",
        table_markdown(
            pd.DataFrame(
                [
                    {"Check": "Fuel: missing calendar days", "Result": len(fuel_gaps)},
                    {"Check": "Fuel: duplicate dates", "Result": int(fuel["date"].duplicated().sum())},
                    {"Check": "Fuel: missing values", "Result": int(fuel.isna().sum().sum())},
                    {"Check": "Fuel: non-positive prices", "Result": int((fuel[["diesel_price", "petrol_price"]] <= 0).sum().sum())},
                    {"Check": "Fuel: diesel range", "Result": f"{fuel['diesel_price'].min():g} to {fuel['diesel_price'].max():g}"},
                    {"Check": "Fuel: petrol range", "Result": f"{fuel['petrol_price'].min():g} to {fuel['petrol_price'].max():g}"},
                    {"Check": "Weather: missing calendar days", "Result": len(weather_gaps)},
                    {"Check": "Weather: duplicate dates", "Result": int(weather["date"].duplicated().sum())},
                    {"Check": "Weather: missing values", "Result": int(weather.isna().sum().sum())},
                    {"Check": "Weather: rainfall range (mm)", "Result": f"{weather['rainfall_mm'].min():g} to {weather['rainfall_mm'].max():g}"},
                    {"Check": "Weather: temperature range (C)", "Result": f"{weather['temperature_c'].min():g} to {weather['temperature_c'].max():g}"},
                    {"Check": "Weather: humidity range (%)", "Result": f"{weather['humidity_pct'].min():g} to {weather['humidity_pct'].max():g}"},
                ]
            )
        ),
        "",
        "Regional weather series:",
        "",
        table_markdown(region_counts.assign(first=region_counts["first"].dt.date, last=region_counts["last"].dt.date)),
        "",
        "## 5. Date alignment with the price dataset",
        "",
        table_markdown(
            pd.DataFrame(
                [
                    {"Check": "Price dates with no same-day fuel row", "Count": len(no_fuel), "Dates": ", ".join(str(d.date()) for d in no_fuel[:12])},
                    {"Check": "Price dates with no same-day weather row", "Count": len(no_weather), "Dates": ", ".join(str(d.date()) for d in no_weather[:12])},
                    {"Check": "Price dates whose day-before fuel row is missing", "Count": len(no_fuel_lag), "Dates": ", ".join(str(d.date()) for d in no_fuel_lag[:12])},
                    {"Check": "Price dates whose 14-day-earlier weather row is missing", "Count": len(no_lag), "Dates": ", ".join(str(d.date()) for d in no_lag[:12])},
                ]
            )
        ),
        "",
    ]

    # ---- duplicate bulletins
    duplicates = check_duplicate_bulletins(detail, cache_dir)
    lines += ["## 6. Dates with more than one published PDF", "", "The build keeps the first PDF for such a date.", "", table_markdown(duplicates), ""]

    # ---- pdf sample
    sample = cross_check(detail, cache_dir, figures_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(metrics_dir / "pdf_crosscheck_sample.csv", index=False)
    lines += [
        "## 7. Cross-check against the original PDFs",
        "",
        f"Random sample of {len(sample)} records (seed {SAMPLE_SEED}, {SAMPLE_PER_YEAR} per calendar year). Position-based re-read matched: "
        f"**{int(sample['position_match'].sum())} of {len(sample)}**. Image crops: `reports/figures/pdf_crosscheck_sheet_*.png`.",
        "",
        table_markdown(sample.drop(columns="source_pdf")),
        "",
    ]
    lines += all_markets_section(data_dir, cache_dir, figures_dir, metrics_dir, prices, fuel, weather)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--report", default="reports/DATA_QUALITY_REPORT.md")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--metrics-dir", default="reports/metrics")
    args = parser.parse_args(argv)

    report = build_report(Path(args.data_dir), Path(args.cache_dir), Path(args.figures_dir), Path(args.metrics_dir))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
