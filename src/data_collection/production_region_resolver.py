"""Resolves the evidence-based production-region coordinates for a vegetable, per
config/crop_production_regions.yaml -- falling back to the market's own configured
weather location (weather_api.latitude/longitude) when a vegetable has no sufficiently
evidenced production region.

This is the ONE place that decides "which coordinates does this vegetable's weather
come from" -- both the historical dataset rebuild (scripts/build_historical_weather_by_region.py)
and the live prediction path (app/services.py) call this rather than each re-reading
crop_production_regions.yaml themselves.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

DEFAULT_MAPPING_PATH = "config/crop_production_regions.yaml"


@dataclass(frozen=True)
class ResolvedWeatherLocation:
    """Where to fetch/read weather from for one vegetable.

    is_production_region=False means no sufficiently evidenced production region
    exists for this vegetable (or the mapping file has no entry for it) -- the
    market's own configured coordinates are used instead, identical to this
    project's original (pre-crop-specific-weather) behavior. This is never a
    guessed location: absence of an entry in crop_production_regions.yaml is a
    deliberate, documented "insufficient evidence" outcome, not an oversight.
    """

    region_name: str
    latitude: float
    longitude: float
    is_production_region: bool
    evidence: str | None = None
    confidence: str | None = None


class ProductionRegionResolver:
    """Loads config/crop_production_regions.yaml once and resolves vegetable -> location."""

    def __init__(
        self,
        mapping_path: str = DEFAULT_MAPPING_PATH,
        config_path: str = "config/config.yaml",
    ) -> None:
        self._mapping_path = Path(mapping_path)
        self._market_config = ConfigLoader(config_path)
        self._regions: dict[str, dict] = {}
        self._vegetable_regions: dict[str, dict] = {}
        if self._mapping_path.exists():
            with self._mapping_path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
            self._regions = data.get("regions", {}) or {}
            self._vegetable_regions = data.get("vegetable_production_regions", {}) or {}
            # Keyed by normalized (casefolded, whitespace-collapsed) name so a caller
            # that hasn't yet run DataTransformer.standardize_vegetable_names() (e.g.
            # raw prediction input) still resolves correctly -- same tolerance
            # DataCleaner.clean_text_columns()/DataTransformer already apply elsewhere.
            self._normalized_vegetable_regions = {
                self._normalize_name(name): entry for name, entry in self._vegetable_regions.items()
            }
        else:
            logger.warning(
                "ProductionRegionResolver: mapping file '%s' not found; every vegetable "
                "will fall back to market-location weather.",
                self._mapping_path,
            )
            self._normalized_vegetable_regions = {}

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(str(name).split()).casefold()

    def resolve(self, vegetable: str) -> ResolvedWeatherLocation:
        """Return the production-region location for `vegetable`, or the market's
        configured location if no sufficiently evidenced region is mapped for it.

        Matching is case/whitespace-tolerant (e.g. "carrot" resolves the same as
        "Carrot"), so this works whether `vegetable` has already been through
        DataTransformer.standardize_vegetable_names() or not.

        Never raises for an unmapped/unknown vegetable -- falling back to market
        weather is the correct, documented behavior for those, not an error.
        """
        entry = self._normalized_vegetable_regions.get(self._normalize_name(vegetable))
        if entry is not None:
            region_key = entry.get("region")
            region = self._regions.get(region_key)
            if region is not None:
                return ResolvedWeatherLocation(
                    region_name=region["name"],
                    latitude=region["latitude"],
                    longitude=region["longitude"],
                    is_production_region=True,
                    evidence=entry.get("evidence"),
                    confidence=entry.get("confidence"),
                )
            logger.warning(
                "ProductionRegionResolver: vegetable '%s' maps to unknown region key '%s'; "
                "falling back to market-location weather.",
                vegetable,
                region_key,
            )

        logger.info(
            "ProductionRegionResolver: '%s' has no configured production-region mapping in '%s'; "
            "falling back to market-location weather (this is a deliberate, documented fallback, "
            "not a guess at the vegetable's actual cultivation location).",
            vegetable,
            self._mapping_path,
        )
        market_latitude = self._market_config.get_required("weather_api.latitude")
        market_longitude = self._market_config.get_required("weather_api.longitude")
        return ResolvedWeatherLocation(
            region_name="market location (no sufficiently evidenced production region)",
            latitude=market_latitude,
            longitude=market_longitude,
            is_production_region=False,
        )
