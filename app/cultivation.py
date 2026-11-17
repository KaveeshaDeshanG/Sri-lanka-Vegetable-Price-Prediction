"""Cultivation planning: harvest window, price around it and a suggested selling period.

Streamlit-free, so it can be tested with plain pytest. Three sources, each labelled for what it is:

- Harvest timing comes from the Department of Agriculture's published guidance
  (data/reference/doa_harvest_durations_v2.csv), one row per crop guidance or released variety.
  It's a reference range, not a model output: no farm-level planting and harvest records exist to
  train one (reports/CULTIVATION_DATA_PLAN.md).
- Prices up to 30 days after the latest recorded price come from the Ridge direct forecaster, with
  a likely range from its measured test-period errors (reports/metrics/forecast_error_bands.json).
- Beyond that, there is no forecast. The planner shows recorded prices from the same weeks of
  earlier years instead (the harvest window widened by SEASONAL_PAD_DAYS on each side), and says
  that it is history, not a prediction. select_price() picks the one price the plan leads with,
  in a fixed order of priority, and labels what kind of price it is.

Durations are kept in the unit the source uses and converted only when a date is calculated:
days and weeks exactly (a week is 7 days), months as calendar months. No month is assumed to be
30 days. A half month ("4 ½ months") is the day halfway between the two whole calendar months
around it, rounded down. See add_duration().
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.direct_forecaster import MAX_HORIZON_DAYS, DirectForecaster

REFERENCE_COLUMNS = (
    "crop", "variety", "price_labels", "min_value", "max_value", "unit", "start_stage", "event",
    "repeat_harvest", "scope", "source_url", "source_text",
)
UNITS = ("days", "weeks", "months")
# The date the user must enter, for each starting point a source uses. Never interchangeable.
START_STAGES = {
    "planting": ("Planting date", "The source counts from planting."),
    "seeding": ("Seeding (sowing) date", "The source counts from seeding, the day the seed is sown."),
    "transplanting": ("Transplanting date", "The source counts from transplanting seedlings to the field."),
    "field planting": ("Field planting date", "The source counts from field planting."),
    "nursery sowing": ("Nursery sowing date", "The source counts from sowing in the nursery, not from transplanting."),
    "flowering": ("Flowering date", "The source counts from flowering, not from planting."),
    "not stated": (
        "Start date (the source doesn't say which)",
        "The source gives a duration but doesn't say whether it counts from sowing, planting or transplanting.",
    ),
}
EVENTS = {
    "first harvest": "first harvest",
    "harvest": "harvest",
    "maturity": "maturity",
    "maturity for consumption": "maturity for consumption",
    "harvesting period": "harvesting period",
}
SAME_SEASON_YEARS = 12
# "The same weeks" of an earlier year: the harvest window widened by a week on each side. A one-day
# harvest window on its own matches at most one bulletin per year (none on a weekend or holiday),
# which left the earlier-years reference with only 2-10 prices in all.
SEASONAL_PAD_DAYS = 7
# The headline seasonal reference uses the most recent years that have prices in those weeks, so a
# decade-old price level (before the 2022 price rises) doesn't pull it down. Older years stay in the table.
SEASONAL_REFERENCE_YEARS = 5
# Fewer than this, and the seasonal reference is shown with its count but doesn't lead the plan.
MIN_SEASONAL_YEARS = 3
MIN_SEASONAL_OBSERVATIONS = 10

PRICE_SOURCE = "HARTI daily bulletin"
PRICE_MARKET = "Dambulla"
PRICE_TYPE = "Wholesale"
PRICE_UNIT = "Rs/kg"
PRICE_KINDS = {
    "model_forecast": "Model Forecast",
    "recorded_actual": "Actual Recorded Price – Harvest Window Already Passed",
    "seasonal_reference": "Historical Seasonal Price Reference – Not a Forecast",
    "latest_actual": "Latest Available Actual Price – Not a Future Prediction",
    "unavailable": "No Verified Price Available",
}


class CultivationReferenceError(Exception):
    """The harvest reference file is missing or malformed."""


@dataclass(frozen=True)
class CropTiming:
    crop: str
    variety: str
    price_labels: list[str]
    min_value: float
    max_value: float
    unit: str
    start_stage: str
    event: str
    repeat_harvest: str
    scope: str
    source_url: str
    source_text: str

    @property
    def date_label(self) -> str:
        return START_STAGES[self.start_stage][0]

    @property
    def stage_note(self) -> str:
        return START_STAGES[self.start_stage][1]

    @property
    def duration_text(self) -> str:
        """The duration as published, e.g. "60–65 days", "10–12 weeks", "4½ months"."""
        low, high = _format_value(self.min_value), _format_value(self.max_value)
        unit = self.unit if not (self.min_value == self.max_value == 1) else self.unit.rstrip("s")
        return f"{low} {unit}" if low == high else f"{low}–{high} {unit}"

    @property
    def calculation_basis(self) -> str:
        if self.unit == "days":
            rule = "days are added to the date"
        elif self.unit == "weeks":
            rule = "each week is 7 days"
        else:
            rule = "months are calendar months; a half month is halfway between the two whole months around it"
        return f"{self.duration_text} after the {self.date_label.split(' (')[0].lower()}, to {self.event}; {rule}."


def _format_value(value: float) -> str:
    whole = int(value)
    if value == whole:
        return str(whole)
    return f"{whole}½" if value - whole == 0.5 else f"{value:g}"


def add_duration(start, value: float, unit: str) -> pd.Timestamp:
    """The date `value` units after `start`.

    days/weeks: exact (a week is 7 days); the result must be a whole number of days.
    months: calendar months, so 1 month after 31 January is 28 (or 29) February, the last day of
    that month. A fractional month adds that fraction of the days between the two whole months
    around it, rounded down: 4½ months after 10 January is 25 May (10 May plus 15 of the 31 days
    to 10 June).
    """
    start = pd.Timestamp(start).normalize()
    if value <= 0:
        raise ValueError("A duration must be positive.")
    if unit == "days" or unit == "weeks":
        days = value * (7 if unit == "weeks" else 1)
        if days != int(days):
            raise ValueError(f"{value} {unit} is not a whole number of days.")
        return start + pd.Timedelta(days=int(days))
    if unit == "months":
        whole = int(value)
        base = start + pd.DateOffset(months=whole)
        fraction = value - whole
        if fraction == 0:
            return base
        following = start + pd.DateOffset(months=whole + 1)
        return base + pd.Timedelta(days=math.floor(fraction * (following - base).days))
    raise ValueError(f"Unknown duration unit '{unit}'.")


def load_crop_timings(path: str | Path) -> list[CropTiming]:
    """Read and check the published harvest timings."""
    path = Path(path)
    if not path.exists():
        raise CultivationReferenceError(f"`{path.as_posix()}` is missing.")
    try:
        table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except (ValueError, OSError, UnicodeDecodeError) as error:
        raise CultivationReferenceError(f"`{path.as_posix()}` could not be read ({error}).") from error
    missing = [column for column in REFERENCE_COLUMNS if column not in table.columns]
    if missing:
        raise CultivationReferenceError(f"`{path.as_posix()}` is missing column(s): {', '.join(missing)}.")
    timings, seen = [], set()
    for number, row in enumerate(table.itertuples(index=False), start=2):
        where = f"`{path.as_posix()}` line {number} ({row.crop}, {row.variety})"
        try:
            low, high = float(row.min_value), float(row.max_value)
        except ValueError as error:
            raise CultivationReferenceError(f"{where} has a non-numeric duration.") from error
        if not (0 < low <= high):
            raise CultivationReferenceError(f"{where} has an invalid duration range.")
        if row.unit not in UNITS:
            raise CultivationReferenceError(f"{where} has an unknown unit '{row.unit}'.")
        if row.unit in ("days", "weeks") and (low != int(low) or high != int(high)):
            raise CultivationReferenceError(f"{where} has a fractional number of {row.unit}.")
        if row.unit == "months" and any((value * 2) != int(value * 2) for value in (low, high)):
            raise CultivationReferenceError(f"{where}: only whole and half months are supported.")
        if row.start_stage not in START_STAGES:
            raise CultivationReferenceError(f"{where} has an unknown start stage '{row.start_stage}'.")
        if row.event not in EVENTS:
            raise CultivationReferenceError(f"{where} has an unknown event '{row.event}'.")
        if not row.source_url.startswith("https://doa.gov.lk/") or not row.source_text.strip():
            raise CultivationReferenceError(f"{where} has no Department of Agriculture source.")
        if (row.crop, row.variety) in seen:
            raise CultivationReferenceError(f"{where} is listed twice.")
        seen.add((row.crop, row.variety))
        timings.append(
            CropTiming(
                crop=row.crop,
                variety=row.variety,
                price_labels=[label.strip() for label in row.price_labels.split(";") if label.strip()],
                min_value=low,
                max_value=high,
                unit=row.unit,
                start_stage=row.start_stage,
                event=row.event,
                repeat_harvest=row.repeat_harvest.strip(),
                scope=row.scope,
                source_url=row.source_url,
                source_text=row.source_text.strip(),
            )
        )
    return timings


def harvest_window(timing: CropTiming, start_date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last day of the published range, counted from start_date (the source's own stage)."""
    return add_duration(start_date, timing.min_value, timing.unit), add_duration(start_date, timing.max_value, timing.unit)


def latest_start_in_reach(timing: CropTiming, forecast_end) -> pd.Timestamp:
    """The latest start date whose earliest harvest day still falls on or before forecast_end.

    Searched day by day backwards (at most one year plus a month), so it uses add_duration()'s own
    calendar rules rather than an approximation of them.
    """
    forecast_end = pd.Timestamp(forecast_end).normalize()
    start = forecast_end
    for _ in range(400):
        if add_duration(start, timing.min_value, timing.unit) <= forecast_end:
            return start
        start -= pd.Timedelta(days=1)
    raise ValueError(f"No start date reaches {forecast_end.date()} for {timing.duration_text}.")


def load_error_bands(path: str | Path) -> dict[int, dict] | None:
    """Per-horizon quantiles of log(actual / forecast), or None if the report isn't usable."""
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        return {int(band["horizon"]): band for band in report["bands"]}
    except (OSError, ValueError, KeyError, TypeError):
        return None


@dataclass
class CultivationPlan:
    timing: CropTiming
    price_label: str | None
    start_date: pd.Timestamp
    harvest_start: pd.Timestamp
    harvest_end: pd.Timestamp
    latest_price_date: pd.Timestamp | None
    forecast_end: pd.Timestamp | None
    status: str  # "passed", "in_reach", "partly_in_reach", "beyond" or "no_prices"
    forecast: pd.DataFrame | None = None  # date, horizon, price, low, high
    window_forecast: pd.DataFrame | None = None  # forecast rows inside the harvest window
    recorded_in_window: pd.DataFrame | None = None  # recorded prices inside the harvest window
    best_day: dict | None = None
    clear_advantage: bool = False
    same_season: pd.DataFrame | None = None  # per earlier year, the same weeks (see SEASONAL_PAD_DAYS)
    seasonal: dict | None = None  # seasonal_reference() summary
    warnings: list[str] = field(default_factory=list)
    stale: bool = False  # the series' latest price is too old to forecast from

    @property
    def window_days(self) -> int:
        """Days in the harvest window, both ends included."""
        return (self.harvest_end - self.harvest_start).days + 1

    @property
    def forecast_days(self) -> int:
        """Harvest-window days the forecast actually covers."""
        return 0 if self.window_forecast is None else len(self.window_forecast)

    @property
    def fully_forecast(self) -> bool:
        return self.forecast_days > 0 and self.forecast_days == self.window_days


def forecast_with_range(
    forecaster: DirectForecaster, origin_rows: pd.DataFrame, price_label: str, latest: pd.Timestamp, bands: dict | None
) -> pd.DataFrame:
    """The forecast for every day up to MAX_HORIZON_DAYS, with a likely range where measured."""
    path = forecaster.forecast(origin_rows, price_label, latest, list(range(1, MAX_HORIZON_DAYS + 1)))
    frame = pd.DataFrame({"date": path["target_date"], "horizon": path["horizon"], "price": path["predicted_price"]})
    if bands:
        frame["low"] = [price * np.exp(bands[h]["q10"]) if h in bands else np.nan for h, price in zip(frame["horizon"], frame["price"], strict=True)]
        frame["high"] = [price * np.exp(bands[h]["q90"]) if h in bands else np.nan for h, price in zip(frame["horizon"], frame["price"], strict=True)]
    else:
        frame["low"] = frame["high"] = np.nan
    return frame


def suggest_selling_day(window_forecast: pd.DataFrame) -> tuple[dict | None, bool]:
    """The forecast's highest-priced day in the window, and whether it beats the first day clearly.

    "Clearly" means the best day's likely low still exceeds the first day's forecast. Otherwise the
    difference is within normal forecast error, and selling as soon as it's harvested is as good.
    """
    if window_forecast is None or window_forecast.empty:
        return None, False
    best = window_forecast.loc[window_forecast["price"].idxmax()]
    first = window_forecast.iloc[0]
    clear = bool(pd.notna(best["low"]) and best["date"] != first["date"] and best["low"] > first["price"])
    return {"date": best["date"], "price": float(best["price"]), "low": best["low"], "high": best["high"]}, clear


def _valid(actual: pd.DataFrame) -> pd.DataFrame:
    """Rows with a real price: a missing or zero-coded value is not a price of Rs 0."""
    prices = pd.to_numeric(actual["price"], errors="coerce")
    return actual.assign(price=prices)[prices > 0]


def same_season_history(
    actual: pd.DataFrame, harvest_start: pd.Timestamp, harvest_end: pd.Timestamp, pad_days: int = 0
) -> pd.DataFrame:
    """Recorded prices in the same calendar window of each earlier year that has any.

    pad_days widens the window on each side, so a one-day harvest is compared with the same weeks.
    """
    actual = _valid(actual)
    rows = []
    for years_back in range(1, SAME_SEASON_YEARS + 1):
        start = harvest_start - pd.DateOffset(years=years_back) - pd.Timedelta(days=pad_days)
        end = harvest_end - pd.DateOffset(years=years_back) + pd.Timedelta(days=pad_days)
        prices = actual.loc[actual["date"].between(start, end), "price"]
        if prices.empty:
            continue
        rows.append(
            {
                "Period": f"{start.date()} to {end.date()}",
                "Days recorded": int(len(prices)),
                "Lowest (Rs/kg)": float(prices.min()),
                "Median (Rs/kg)": float(prices.median()),
                "Highest (Rs/kg)": float(prices.max()),
            }
        )
    return pd.DataFrame(rows, columns=["Period", "Days recorded", "Lowest (Rs/kg)", "Median (Rs/kg)", "Highest (Rs/kg)"])


def seasonal_reference(
    actual: pd.DataFrame, harvest_start: pd.Timestamp, harvest_end: pd.Timestamp, pad_days: int = SEASONAL_PAD_DAYS
) -> dict:
    """A summary of recorded prices in the same weeks of earlier years.

    Uses the SEASONAL_REFERENCE_YEARS most recent earlier years that have prices in those weeks.
    `sufficient` says whether that is enough (MIN_SEASONAL_YEARS years and MIN_SEASONAL_OBSERVATIONS
    prices) for the reference to lead the plan.
    """
    actual = _valid(actual)
    per_year = []
    for years_back in range(1, SAME_SEASON_YEARS + 1):
        start = harvest_start - pd.DateOffset(years=years_back) - pd.Timedelta(days=pad_days)
        end = harvest_end - pd.DateOffset(years=years_back) + pd.Timedelta(days=pad_days)
        rows = actual[actual["date"].between(start, end)]
        if not rows.empty:
            per_year.append((start, end, rows))
    used = per_year[:SEASONAL_REFERENCE_YEARS]
    recorded = pd.concat([r for _, _, r in used]) if used else actual.iloc[0:0]
    prices = recorded["price"]
    years = sorted(recorded["date"].dt.year.unique().tolist())
    summary = {
        "observations": len(prices),
        "years": years,
        "years_available": len(per_year),
        "periods": [f"{start.date()} to {end.date()}" for start, end, _ in used],
        "pad_days": pad_days,
        "median": float(prices.median()) if len(prices) else None,
        "mean": float(prices.mean()) if len(prices) else None,
        "low": float(prices.min()) if len(prices) else None,
        "high": float(prices.max()) if len(prices) else None,
    }
    summary["seasons"] = len(used)
    summary["sufficient"] = len(used) >= MIN_SEASONAL_YEARS and len(prices) >= MIN_SEASONAL_OBSERVATIONS
    return summary


@dataclass
class PriceResult:
    """The one price a plan leads with, and exactly what kind of price it is."""

    kind: str  # a key of PRICE_KINDS
    price: float | None = None  # Rs/kg; None only for "unavailable"
    low: float | None = None
    high: float | None = None
    median: float | None = None
    mean: float | None = None
    date: pd.Timestamp | None = None  # forecast date, or the actual observation date
    horizon: int | None = None  # days after the latest recorded price (forecasts only)
    period: str | None = None  # the dates a historical figure covers
    observations: int | None = None
    years: list[int] = field(default_factory=list)
    age_days: int | None = None  # how old an actual observation is
    explanation: str = ""
    limitation: str | None = None
    source: str = PRICE_SOURCE
    market: str = PRICE_MARKET
    price_type: str = PRICE_TYPE
    unit: str = PRICE_UNIT

    @property
    def label(self) -> str:
        return PRICE_KINDS[self.kind]


def select_price(plan: "CultivationPlan", actual: pd.DataFrame | None, today=None, latest_record: dict | None = None) -> PriceResult:
    """The price the plan leads with, by priority. Never a made-up or zero price.

    1. Model forecast: the harvest falls within the forecaster's reach (the first harvest day it covers).
       A harvest window that has already passed shows the prices actually recorded in it instead.
    2. Historical seasonal reference: the same weeks of earlier years, when there are enough prices.
    3. Latest available actual price: the series' newest recorded price, with its age.
    Otherwise "unavailable", with the reason.

    latest_record: the newest HARTI Dambulla record of this series (keys date, price, low, high), used
    for the latest price's range; without it only the price itself is shown.
    """
    today = pd.Timestamp(today or pd.Timestamp.now()).normalize()
    if plan.status == "no_prices" or actual is None or _valid(actual).empty:
        return PriceResult(
            "unavailable",
            explanation=(
                f"No verified price can be shown for {plan.timing.crop}: HARTI's daily bulletin has no price series for it, "
                "so the project's HARTI Dambulla data (data/raw/harti_prices.csv) holds none, and no other "
                "vegetable's prices are substituted for it."
            ),
        )
    actual = _valid(actual)

    if plan.window_forecast is not None and not plan.window_forecast.empty:
        first = plan.window_forecast.iloc[0]
        coverage = (
            None
            if plan.fully_forecast
            else f"The forecast covers {plan.forecast_days} of the {plan.window_days} harvest days; "
            "the rest are beyond its 30-day reach."
        )
        return PriceResult(
            "model_forecast",
            price=float(first["price"]),
            low=None if pd.isna(first["low"]) else float(first["low"]),
            high=None if pd.isna(first["high"]) else float(first["high"]),
            date=pd.Timestamp(first["date"]),
            horizon=int(first["horizon"]),
            explanation=(
                f"The Ridge direct forecaster's prediction for {first['date']:%Y-%m-%d}, {int(first['horizon'])} day(s) after "
                f"the latest recorded price ({plan.latest_price_date:%Y-%m-%d}). The range is the middle 80% of the "
                "forecaster's test-period errors for all vegetables together at that many days ahead."
            ),
            limitation=coverage,
        )

    if plan.status == "passed" and plan.recorded_in_window is not None and not plan.recorded_in_window.empty:
        prices = plan.recorded_in_window["price"]
        return PriceResult(
            "recorded_actual",
            price=float(prices.median()),
            low=float(prices.min()),
            high=float(prices.max()),
            median=float(prices.median()),
            mean=float(prices.mean()),
            period=f"{plan.harvest_start.date()} to {plan.harvest_end.date()}",
            observations=len(prices),
            years=sorted(plan.recorded_in_window["date"].dt.year.unique().tolist()),
            explanation="The harvest window has already passed, so these are the prices actually recorded in it (median shown).",
        )

    seasonal = plan.seasonal
    reason = _no_forecast_reason(plan)
    if seasonal and seasonal["sufficient"]:
        return PriceResult(
            "seasonal_reference",
            price=seasonal["median"],
            low=seasonal["low"],
            high=seasonal["high"],
            median=seasonal["median"],
            mean=seasonal["mean"],
            period="; ".join(seasonal["periods"]),
            observations=seasonal["observations"],
            years=seasonal["years"],
            explanation=(
                f"{reason} This is the median of the {seasonal['observations']} prices recorded in the same weeks "
                f"(the harvest dates ± {seasonal['pad_days']} days) of the {seasonal['seasons']} most recent earlier years "
                f"that have prices then (dates from {seasonal['years'][0]} to {seasonal['years'][-1]}): "
                "what the price has been at this time of year, not a prediction of this year's price."
            ),
            limitation=(
                "Earlier years' prices are not adjusted for inflation, and a year can differ a lot from the next."
                + (
                    f" The most recent of these prices is from {seasonal['years'][-1]}, so they may be a poor guide to today's level."
                    if plan.harvest_start.year - seasonal["years"][-1] > 2
                    else ""
                )
            ),
        )

    last = actual.sort_values("date").iloc[-1]
    record = latest_record or {}
    observed = pd.Timestamp(record.get("date", last["date"]))
    thin = ""
    if seasonal and seasonal["observations"]:
        thin = (
            f" Only {seasonal['observations']} price(s) in {seasonal['seasons']} earlier year(s) fall in the same weeks, "
            f"too few for a seasonal reference (at least {MIN_SEASONAL_OBSERVATIONS} prices in {MIN_SEASONAL_YEARS} years are needed)."
        )
    elif seasonal is not None:
        thin = " No price was recorded in the same weeks of any earlier year."
    return PriceResult(
        "latest_actual",
        price=float(record.get("price", last["price"])),
        low=None if pd.isna(record.get("low", float("nan"))) else float(record["low"]),
        high=None if pd.isna(record.get("high", float("nan"))) else float(record["high"]),
        date=observed,
        age_days=int((today - observed).days),
        explanation=f"{reason}{thin} This is the newest price actually recorded, shown for reference.",
        limitation="Today's price says little about the price at a harvest weeks or months away.",
    )


def _no_forecast_reason(plan: "CultivationPlan") -> str:
    if plan.stale:
        return (
            f"No forecast: the latest recorded price ({plan.latest_price_date:%Y-%m-%d}) is more than "
            f"{MAX_HORIZON_DAYS} days old."
        )
    if plan.status in ("beyond", "partly_in_reach"):
        return (
            f"No forecast: the harvest is after {plan.forecast_end:%Y-%m-%d}, the furthest date the model forecasts "
            f"({MAX_HORIZON_DAYS} days after the latest recorded price)."
        )
    if plan.status == "passed":
        return "The harvest window has passed and no price was recorded in it."
    return "No forecast is available."


def build_plan(
    timing: CropTiming,
    price_label: str | None,
    start_date,
    actual: pd.DataFrame | None,
    forecaster: DirectForecaster | None,
    origin_rows: pd.DataFrame | None,
    bands: dict | None,
    today: pd.Timestamp | None = None,
) -> CultivationPlan:
    """Everything the Cultivation Planner shows for one crop or variety and start date.

    actual: this price series' recorded prices (columns date, price), or None when the crop has no
    price series. Forecasts are requested only for dates within MAX_HORIZON_DAYS of the latest price.
    """
    if start_date is None or pd.isna(pd.Timestamp(start_date)):
        raise ValueError("A start date is required.")
    start = pd.Timestamp(start_date).normalize()
    harvest_start, harvest_end = harvest_window(timing, start)
    today = (today or pd.Timestamp.now()).normalize()

    if actual is None or actual.empty or price_label is None:
        plan = CultivationPlan(timing, price_label, start, harvest_start, harvest_end, None, None, "no_prices")
        plan.warnings.append(
            f"No market price series exists for {timing.crop} in the HARTI data, so only the harvest dates are shown."
        )
        _add_timing_warnings(plan)
        return plan

    latest = pd.Timestamp(actual["date"].max())
    forecast_end = latest + pd.Timedelta(days=MAX_HORIZON_DAYS)
    if harvest_end <= latest:
        status = "passed"
    elif harvest_start > forecast_end:
        status = "beyond"
    elif harvest_end > forecast_end or harvest_start <= latest:
        status = "partly_in_reach"
    else:
        status = "in_reach"

    plan = CultivationPlan(timing, price_label, start, harvest_start, harvest_end, latest, forecast_end, status)
    _add_timing_warnings(plan)
    # A forecast from a price that is already older than the forecaster's reach would describe dates
    # that have passed, from data the farmer can't act on. Show recorded prices instead.
    plan.stale = (today - latest).days > MAX_HORIZON_DAYS
    if plan.stale:
        plan.warnings.append(
            f"The latest recorded price for {price_label} is from {latest.date()}, {(today - latest).days} days ago, "
            "so no forecast is shown for it. Recorded prices from earlier years are shown instead."
        )

    if status in ("in_reach", "partly_in_reach") and not plan.stale and forecaster is not None and origin_rows is not None:
        plan.forecast = forecast_with_range(forecaster, origin_rows, price_label, latest, bands)
        inside = plan.forecast["date"].between(harvest_start, harvest_end)
        plan.window_forecast = plan.forecast[inside].reset_index(drop=True)
        # A selling day is suggested only when the forecast covers the whole window: the best of the
        # days it does cover says nothing about the days it doesn't.
        if plan.fully_forecast:
            plan.best_day, plan.clear_advantage = suggest_selling_day(plan.window_forecast)
        if status == "partly_in_reach":
            plan.warnings.append(
                f"The forecast covers only {plan.forecast_days} of the {plan.window_days} days in the harvest window "
                f"(its reach ends {forecast_end.date()}), so no selling day is suggested."
            )
    if status in ("passed", "partly_in_reach"):
        recorded = actual[actual["date"].between(harvest_start, harvest_end)]
        plan.recorded_in_window = recorded.reset_index(drop=True) if not recorded.empty else None
    if plan.stale or status in ("beyond", "partly_in_reach", "passed") or plan.forecast is None:
        plan.same_season = same_season_history(actual, harvest_start, harvest_end, pad_days=SEASONAL_PAD_DAYS)
        plan.seasonal = seasonal_reference(actual, harvest_start, harvest_end)
    return plan


def _add_timing_warnings(plan: CultivationPlan) -> None:
    timing = plan.timing
    if timing.start_stage == "not stated":
        plan.warnings.append(
            "The source doesn't say whether this duration counts from sowing, planting or transplanting. "
            "The dates are counted from the date you entered, so treat them as approximate."
        )
    if timing.start_stage == "flowering":
        plan.warnings.append("This duration counts from flowering, so enter the date the crop flowered, not the planting date.")
    if timing.event == "first harvest" and timing.repeat_harvest:
        plan.warnings.append(f"The dates are for the first harvest. Later harvests, as published: {timing.repeat_harvest}")
