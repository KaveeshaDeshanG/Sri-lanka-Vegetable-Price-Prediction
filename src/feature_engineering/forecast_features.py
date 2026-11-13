"""Features for the direct forecaster: what was known on an origin date, for a price some days later.

Separate from FeatureEngineer's row-level features, because the question differs. Those describe
"this row's date, given exact earlier dates". These describe "standing on the origin date, with
whatever has been published so far, what will the price be `horizon` days later?". A gap in the
data therefore doesn't remove a row: the last reading simply carries forward for up to
MAX_CARRY_DAYS, and days_since_reading tells the model how old it is.

Every value is built only from data available on the origin date. Other markets are keyed by the
bulletin that published them (see parse_all_markets), because a reading dated a given day may not
be public until the next day. tests/feature_engineering/test_forecast_features.py enforces both.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config.config_loader import ConfigLoader
from src.data_collection.production_region_resolver import ProductionRegionResolver
from src.preprocessing.transformer import DataTransformer

TARGET_MARKET = "Dambulla"
MAX_CARRY_DAYS = 7
OTHER_MARKET_CARRY_DAYS = 3
MIN_ROWS_PER_VEGETABLE = 60
COMPARISON_MARKETS = ("Peliyagoda", "Kandy", "Keppetipola", "Nuwaraeliya", "Bandarawela", "Meegoda")

FEATURE_GROUPS = {
    "level": ["log_last"],
    "momentum": [
        "log_change_1", "log_change_3", "log_change_7", "log_change_14", "log_change_30",
        "vs_mean_7", "vs_mean_14", "vs_mean_30", "volatility_7", "volatility_30",
        "vs_max_30", "vs_min_30", "days_since_reading",
    ],
    "calendar": ["target_day_of_week", "target_year_sin", "target_year_cos"],
    "rain_windows": ["rain_sum_3", "rain_sum_7", "rain_sum_14", "rain_sum_30", "temp_mean_7", "temp_mean_30", "humidity_mean_7"],
    "weather_lag_14": ["rain_lag_14", "temp_lag_14", "humidity_lag_14"],
    "market": ["market_change_1", "market_change_7", "market_vs_mean_30"],
    "fuel": ["diesel_change_30", "petrol_change_30"],
    "other_markets": [
        "other_premium", "other_change_1", "other_change_3", "other_dispersion",
        *[f"premium_{market.lower()}" for market in COMPARISON_MARKETS],
    ],
}


class ForecastFeatureBuilder:
    """Builds forecast features from price history, regional weather and other markets' prices."""

    def __init__(self, config_path: str = "config/config.yaml", extra_market_rows: pd.DataFrame | None = None) -> None:
        config = ConfigLoader(config_path)
        self._resolver = ProductionRegionResolver(config_path=config_path)
        self._weather = pd.read_csv(config.get_required("paths.raw_files.weather_data_by_region"), parse_dates=["date"])
        panel_path = config.get("paths.raw_files.harti_all_markets")
        panels = []
        if panel_path and Path(panel_path).exists():
            panels.append(pd.read_csv(panel_path, parse_dates=["published"]))
        if extra_market_rows is not None and not extra_market_rows.empty:
            # A live bulletin keeps HARTI's "Eggplant"; the saved data and training call it "Brinjal".
            panels.append(DataTransformer().standardize_vegetable_names(extra_market_rows))
        columns = ["published", "market", "vegetable", "price"]
        self._panel = pd.concat([p[columns] for p in panels], ignore_index=True) if panels else pd.DataFrame(columns=columns)

    def build_origin_features(self, history: pd.DataFrame) -> pd.DataFrame:
        """One row per (vegetable, calendar day) with everything that doesn't depend on horizon."""
        calendar = pd.date_range(history["date"].min(), history["date"].max(), freq="D")
        wide = history.pivot_table(index="date", columns="vegetable", values="price").reindex(calendar)
        wide = wide.loc[:, wide.notna().sum() >= MIN_ROWS_PER_VEGETABLE]
        held = np.log(wide.ffill(limit=MAX_CARRY_DAYS))

        market_daily = held.diff().median(axis=1)
        market_level = held.sub(held.rolling(30, min_periods=10).mean()).median(axis=1)
        fuel = pd.DataFrame(index=calendar)
        if {"diesel_price", "petrol_price"} <= set(history.columns):
            fuel = history.groupby("date")[["diesel_price", "petrol_price"]].first().reindex(calendar).ffill()
        other = self._other_market_features(calendar)

        frames = []
        for vegetable in wide.columns:
            frame = self._vegetable_frame(vegetable, wide[vegetable], held[vegetable], calendar)
            frame["market_change_1"] = market_daily
            frame["market_change_7"] = market_daily.rolling(7, min_periods=3).sum()
            frame["market_vs_mean_30"] = market_level
            for name in ("diesel", "petrol"):
                column = f"{name}_price"
                frame[f"{name}_change_30"] = np.log(fuel[column]).diff(30) if column in fuel else np.nan
            frames.append(frame)
        rows = pd.concat(frames, ignore_index=True).merge(other, on=["vegetable", "origin"], how="left")

        rows["other_premium"] = rows["other_level"] - rows["log_last"]
        for market in COMPARISON_MARKETS:
            rows[f"premium_{market.lower()}"] = rows[f"level_{market.lower()}"] - rows["log_last"]
        return rows

    def _vegetable_frame(self, vegetable, price, last, calendar) -> pd.DataFrame:
        location = self._resolver.resolve(vegetable)
        region = location.region_name if location.is_production_region else TARGET_MARKET
        weather = self._weather[self._weather["region"] == region].set_index("date").reindex(calendar)

        frame = pd.DataFrame(index=calendar)
        frame["vegetable"] = vegetable
        frame["origin"] = calendar
        frame["last_price"] = np.exp(last)
        frame["log_last"] = last
        frame["actual_price"] = price
        for days in (1, 3, 7, 14, 30):
            frame[f"log_change_{days}"] = last - last.shift(days)
        for days in (7, 14, 30):
            frame[f"vs_mean_{days}"] = last - last.rolling(days, min_periods=2).mean()
        frame["volatility_7"] = last.diff().rolling(7, min_periods=3).std()
        frame["volatility_30"] = last.diff().rolling(30, min_periods=10).std()
        frame["vs_max_30"] = last - last.rolling(30, min_periods=5).max()
        frame["vs_min_30"] = last - last.rolling(30, min_periods=5).min()
        positions = np.arange(len(price))
        last_seen = pd.Series(np.where(price.notna(), positions, np.nan), index=calendar).ffill()
        frame["days_since_reading"] = positions - last_seen

        rain = weather["rainfall_mm"]
        for days in (3, 7, 14, 30):
            frame[f"rain_sum_{days}"] = rain.rolling(days, min_periods=max(2, days // 2)).sum()
        frame["temp_mean_7"] = weather["temperature_c"].rolling(7, min_periods=3).mean()
        frame["temp_mean_30"] = weather["temperature_c"].rolling(30, min_periods=10).mean()
        frame["humidity_mean_7"] = weather["humidity_pct"].rolling(7, min_periods=3).mean()
        frame["raw_rain"] = rain
        frame["raw_temp"] = weather["temperature_c"]
        frame["raw_humidity"] = weather["humidity_pct"]
        return frame

    def _other_market_features(self, calendar: pd.DatetimeIndex) -> pd.DataFrame:
        """Other markets' prices, keyed by the bulletin that published them."""
        base = self._panel[self._panel["market"] != TARGET_MARKET]
        frames = []
        for vegetable, group in base.groupby("vegetable"):
            wide = np.log(
                group.pivot_table(index="published", columns="market", values="price")
                .reindex(calendar)
                .ffill(limit=OTHER_MARKET_CARRY_DAYS)
            )
            median = wide.median(axis=1)
            frame = pd.DataFrame(
                {
                    "vegetable": vegetable,
                    "origin": calendar,
                    "other_level": median.to_numpy(),
                    "other_change_1": median.diff(1).to_numpy(),
                    "other_change_3": median.diff(3).to_numpy(),
                    "other_dispersion": wide.std(axis=1).to_numpy(),
                }
            )
            for market in COMPARISON_MARKETS:
                frame[f"level_{market.lower()}"] = wide[market].to_numpy() if market in wide else np.nan
            frames.append(frame)
        if frames:
            return pd.concat(frames, ignore_index=True)
        columns = ["vegetable", "origin", "other_level", "other_change_1", "other_change_3", "other_dispersion"]
        columns += [f"level_{market.lower()}" for market in COMPARISON_MARKETS]
        empty = pd.DataFrame(columns=columns)
        empty["vegetable"] = empty["vegetable"].astype(str)
        empty["origin"] = pd.to_datetime(empty["origin"])
        return empty.astype({c: float for c in columns[2:]})


def add_horizon_features(origin_rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add the horizon-specific columns: target date and calendar, 14-day-before-target weather
    (or the newest reading known on the origin date), and the actual price `horizon` days on."""
    rows = origin_rows.copy()
    target_dates = rows["origin"] + pd.Timedelta(days=horizon)
    rows["target_date"] = target_dates
    rows["target_day_of_week"] = target_dates.dt.day_of_week
    rows["target_year_sin"] = np.sin(2 * np.pi * target_dates.dt.day_of_year / 365.25)
    rows["target_year_cos"] = np.cos(2 * np.pi * target_dates.dt.day_of_year / 365.25)

    groups = rows.groupby("vegetable", sort=False)
    weather_shift = max(14 - horizon, 0)
    rows["rain_lag_14"] = groups["raw_rain"].shift(weather_shift)
    rows["temp_lag_14"] = groups["raw_temp"].shift(weather_shift)
    rows["humidity_lag_14"] = groups["raw_humidity"].shift(weather_shift)
    rows["target_price"] = groups["actual_price"].shift(-horizon)
    return rows


def add_vegetable_columns(rows: pd.DataFrame, vegetables: list[str]) -> pd.DataFrame:
    """One-hot vegetable identity, in a fixed order."""
    identity = pd.DataFrame(
        {f"veg_{vegetable}": (rows["vegetable"] == vegetable).astype(float) for vegetable in vegetables},
        index=rows.index,
    )
    return pd.concat([rows, identity], axis=1)
