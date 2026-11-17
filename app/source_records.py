"""The official price records behind the Predict page, one source at a time, with full traceability.

Streamlit-free. Each source keeps its own records: HARTI (wholesale, by market), CBSL (wholesale and
retail, kept apart) and DCS (weekly retail averages). They are never combined into one series:
their collectors, markets, price types, units and periods differ (reports/DATA_SOURCES.md). Only
HARTI's Dambulla series is used by the forecaster; the others are shown for inspection.

A field a source doesn't publish or this project didn't store is left empty (shown as "not
published" / "not recorded"), never filled with an estimate or zero.
"""

import re
from pathlib import Path

import pandas as pd

from config.config_loader import ConfigLoader
from src.preprocessing.transformer import DataTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ITEM_MAP_PATH = PROJECT_ROOT / "data" / "reference" / "source_item_map.csv"

SOURCE_HARTI_DAMBULLA = "HARTI daily bulletin, Dambulla"
SOURCE_HARTI_OTHER = "HARTI daily bulletin, other markets"
SOURCE_CBSL = "CBSL Daily Price Report"
SOURCE_DCS = "DCS weekly retail prices"

COLUMNS = [
    "Source",
    "Observation date",
    "Published",
    "Vegetable (as published)",
    "Market / location",
    "Price type",
    "Unit",
    "Minimum",
    "Maximum",
    "Average / midpoint",
    "Source document",
    "Collected",
]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _raw(config: ConfigLoader, key: str) -> Path | None:
    value = config.get(f"paths.raw_files.{key}")
    return _resolve(value) if value else None


def source_item_map(path: str | Path = ITEM_MAP_PATH) -> pd.DataFrame:
    """Hand-reviewed HARTI series -> CBSL items / DCS vegetables. Unmapped means no clear match."""
    table = pd.read_csv(path, dtype=str).fillna("")
    for column in ("cbsl_items", "dcs_vegetables"):
        table[column] = table[column].map(lambda text: [part.strip() for part in text.split(";") if part.strip()])
    return table.set_index("harti_series")


def dcs_unit(raw: object) -> str:
    """DCS spells its units several ways ("1 kg.", "1Kg.", "1 Kg"). Same unit -> one label;
    different units (500 g, bunch) stay different. Nothing is converted."""
    text = str(raw or "").strip()
    compact = re.sub(r"[\s.]", "", text.lower())
    labels = {"1kg": "Rs per 1 kg", "500g": "Rs per 500 g", "bunch": "Rs per bunch"}
    return labels.get(compact, f"Rs per {text}" if text else "")


def harti_collection_times(config_path: str = "config/config.yaml") -> pd.Series:
    """When each HARTI bulletin (by its date) was downloaded, from the import log or the saved PDF.

    The saved copies live outside the project (data_update.harti_pdf_cache_dir); without them, only
    bulletins imported by scripts/daily_update.py have a recorded time.
    """
    config = ConfigLoader(_resolve(config_path))
    times: dict[pd.Timestamp, pd.Timestamp] = {}
    cache = config.get("data_update.harti_pdf_cache_dir")
    if cache and _resolve(cache).exists():
        for pdf in _resolve(cache).glob("????-??-??_*.pdf"):
            day = pd.Timestamp(pdf.name[:10])
            stamp = pd.Timestamp(pdf.stat().st_mtime, unit="s", tz="UTC").tz_convert("Asia/Colombo").tz_localize(None)
            times[day] = min(times.get(day, stamp), stamp)
    log_path = _resolve(config.get("paths.raw_data_dir", "data/raw")) / "harti_import_log.csv"
    if log_path.exists():
        log = pd.read_csv(log_path, parse_dates=["bulletin_date", "retrieved_at"])
        for row in log[log["status"] == "imported"].itertuples(index=False):
            times[row.bulletin_date] = min(times.get(row.bulletin_date, row.retrieved_at), row.retrieved_at)
    return pd.Series(times, dtype="datetime64[ns]").sort_index()


def harti_dambulla_records(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Every stored HARTI Dambulla reading: the forecaster's own series."""
    config = ConfigLoader(_resolve(config_path))
    detail_path = _resolve(config.get("paths.raw_data_dir", "data/raw")) / "harti_prices_detail.csv"
    detail = DataTransformer().standardize_vegetable_names(pd.read_csv(detail_path, parse_dates=["date", "bulletin_date"]))
    collected = harti_collection_times(config_path)
    return pd.DataFrame(
        {
            "Source": SOURCE_HARTI_DAMBULLA,
            "Observation date": detail["date"],
            "Published": detail["bulletin_date"],
            "series": detail["vegetable"],
            "Vegetable (as published)": detail["vegetable"],
            "Market / location": detail["market"],
            "Price type": "Wholesale",
            "Unit": "Rs per kg",
            "Minimum": detail["low"],
            "Maximum": detail["high"],
            "Average / midpoint": detail["price"],
            "Source document": detail["source_pdf"],
            "Collected": detail["bulletin_date"].map(collected),
        }
    )


def harti_other_market_records(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Every other market's HARTI readings (features of the forecaster, not forecast themselves)."""
    config = ConfigLoader(_resolve(config_path))
    panel_path = _raw(config, "harti_all_markets")
    if panel_path is None or not panel_path.exists():
        return pd.DataFrame(columns=["series", *COLUMNS])
    panel = pd.read_csv(panel_path, parse_dates=["published", "reported"])
    panel = panel[panel["market"] != "Dambulla"]
    ranges_path = panel_path.with_name("harti_all_markets_ranges.csv")
    if ranges_path.exists():
        ranges = pd.read_csv(ranges_path, parse_dates=["published", "reported"])
        panel = panel.merge(ranges[["published", "reported", "market", "vegetable", "low", "high"]], on=["published", "reported", "market", "vegetable"], how="left")
    else:
        panel = panel.assign(low=float("nan"), high=float("nan"))
    detail_path = _resolve(config.get("paths.raw_data_dir", "data/raw")) / "harti_prices_detail.csv"
    links = pd.read_csv(detail_path, usecols=["bulletin_date", "source_pdf"], parse_dates=["bulletin_date"]).drop_duplicates("bulletin_date").set_index("bulletin_date")["source_pdf"]
    collected = harti_collection_times(config_path)
    return pd.DataFrame(
        {
            "Source": SOURCE_HARTI_OTHER,
            "Observation date": panel["reported"],
            "Published": panel["published"],
            "series": panel["vegetable"],
            "Vegetable (as published)": panel["vegetable"],
            "Market / location": panel["market"],
            "Price type": "Wholesale",
            "Unit": "Rs per kg",
            "Minimum": panel["low"],
            "Maximum": panel["high"],
            "Average / midpoint": panel["price"],
            "Source document": panel["published"].map(links),
            "Collected": panel["published"].map(collected),
        }
    )


def cbsl_records(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """CBSL's 'Today' prices: wholesale and retail rows stay separate rows."""
    path = _raw(ConfigLoader(_resolve(config_path)), "cbsl_daily_prices")
    if path is None or not path.exists():
        return pd.DataFrame(columns=["item", *COLUMNS])
    table = pd.read_csv(path, parse_dates=["date", "retrieved_at"])
    return pd.DataFrame(
        {
            "Source": SOURCE_CBSL,
            "Observation date": table["date"],
            "Published": table["date"],
            "item": table["item"],
            "Vegetable (as published)": table["item_as_printed"],
            "Market / location": table["market"],
            "Price type": table["price_type"].str.capitalize(),
            "Unit": table["unit"].str.replace("Rs./", "Rs per ", regex=False),
            "Minimum": float("nan"),  # CBSL prints one price, no range
            "Maximum": float("nan"),
            "Average / midpoint": table["price"],
            "Source document": table["source_url"],
            "Collected": table["retrieved_at"],
        }
    )


DCS_PRICE_COLUMNS = ("price", "range_low", "range_high")


def dcs_unavailable_as_missing(table: pd.DataFrame) -> pd.DataFrame:
    """A display copy of DCS's weekly table in which a printed 0.00 is "not available", not Rs 0.

    DCS prints 0.00 (with a -100% change) for an item it couldn't price that week, e.g. Butter beans
    in 2023-03 week 3, 2024-11 week 4 and 2025-12 week 1. No real retail price is zero or negative,
    so such values become missing here. The stored CSV is not changed, and every positive price,
    however small, is kept exactly as stored. `price_unavailable` marks the rows whose weekly
    average was not published.
    """
    shown = table.copy()
    for column in DCS_PRICE_COLUMNS:
        if column in shown:
            values = pd.to_numeric(shown[column], errors="coerce")
            shown[column] = values.where(values > 0)
    shown["price_unavailable"] = shown["price"].isna() if "price" in shown else False
    return shown


def load_dcs_weekly(config_path: str = "config/config.yaml") -> pd.DataFrame | None:
    """dcs_weekly_retail_prices.csv as stored, with DCS's 0.00 "not available" values as missing."""
    path = _raw(ConfigLoader(_resolve(config_path)), "dcs_weekly_retail")
    if path is None or not path.exists():
        return None
    return dcs_unavailable_as_missing(pd.read_csv(path))


def dcs_records(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """DCS weekly retail averages for Colombo District. A week, not a date, is what DCS publishes."""
    config = ConfigLoader(_resolve(config_path))
    table = load_dcs_weekly(config_path)
    if table is None:
        return pd.DataFrame(columns=["vegetable", *COLUMNS])
    cache = config.get("data_update.dcs_pdf_cache_dir")
    downloaded = {}
    if cache and _resolve(cache).exists():
        for name in table["source_file"].unique():
            pdf = _resolve(cache) / name
            if pdf.exists():
                downloaded[name] = pd.Timestamp(pdf.stat().st_mtime, unit="s", tz="UTC").tz_convert("Asia/Colombo").tz_localize(None)
    return pd.DataFrame(
        {
            "Source": SOURCE_DCS,
            "Observation date": table["period"],
            "Published": table["period"],
            "sort_key": table["year"] * 100 + table["month"] + table["week_of_month"] / 10,
            "vegetable": table["vegetable"],
            "Vegetable (as published)": table["item"],
            "Market / location": table["market"],
            "Price type": "Retail (weekly average)",
            "Unit": table["unit"].map(dcs_unit),
            "Minimum": table["range_low"],
            "Maximum": table["range_high"],
            "Average / midpoint": table["price"],
            "Source document": table["source_url"],
            "Collected": table["source_file"].map(downloaded),
        }
    )


def records_for_series(series: str, sources: dict[str, pd.DataFrame], latest_only: int | None = None) -> dict[str, pd.DataFrame]:
    """Each source's records for one HARTI series, newest first, using the reviewed name map only.

    sources: {"harti_dambulla": ..., "harti_other": ..., "cbsl": ..., "dcs": ...} as returned above.
    latest_only: keep only the records from that many most recent observation periods per source.
    """
    mapping = source_item_map()
    cbsl_items = mapping.loc[series, "cbsl_items"] if series in mapping.index else []
    dcs_vegetables = mapping.loc[series, "dcs_vegetables"] if series in mapping.index else []
    selected = {
        "harti_dambulla": sources["harti_dambulla"][sources["harti_dambulla"]["series"] == series],
        "harti_other": sources["harti_other"][sources["harti_other"]["series"] == series],
        "cbsl": sources["cbsl"][sources["cbsl"]["item"].isin(cbsl_items)],
        "dcs": sources["dcs"][sources["dcs"]["vegetable"].isin(dcs_vegetables)],
    }
    result = {}
    for name, frame in selected.items():
        order = "sort_key" if name == "dcs" else "Observation date"
        frame = frame.sort_values([order, "Market / location"], ascending=[False, True])
        if latest_only is not None and not frame.empty:
            keep = frame[order].drop_duplicates().head(latest_only)
            frame = frame[frame[order].isin(keep)]
        result[name] = frame[COLUMNS].reset_index(drop=True)
    return result


def mapping_note(series: str) -> str:
    mapping = source_item_map()
    return mapping.loc[series, "note"] if series in mapping.index else ""


def latest_dambulla_overview(dambulla: pd.DataFrame) -> pd.DataFrame:
    """The newest HARTI Dambulla reading of every series: what a forecast would start from."""
    newest = dambulla.sort_values("Observation date").groupby("series", as_index=False).tail(1)
    return newest.sort_values("series")[COLUMNS].reset_index(drop=True)
