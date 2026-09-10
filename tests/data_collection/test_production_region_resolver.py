"""Unit tests for src.data_collection.production_region_resolver.ProductionRegionResolver."""

from pathlib import Path

from src.data_collection.production_region_resolver import ProductionRegionResolver


def _mapping_yaml(tmp_path: Path) -> Path:
    mapping_path = tmp_path / "crop_production_regions.yaml"
    mapping_path.write_text(
        "regions:\n"
        "  nuwara_eliya:\n"
        '    name: "Nuwara Eliya"\n'
        "    latitude: 6.97078\n"
        "    longitude: 80.78286\n"
        "  jaffna:\n"
        '    name: "Jaffna"\n'
        "    latitude: 9.66845\n"
        "    longitude: 80.00742\n"
        "vegetable_production_regions:\n"
        "  Carrot:\n"
        "    region: nuwara_eliya\n"
        '    evidence: "test evidence"\n'
        "    confidence: high\n"
        "  Big Onion (Local):\n"
        "    region: jaffna\n"
        '    evidence: "test evidence"\n'
        "    confidence: medium\n",
        encoding="utf-8",
    )
    return mapping_path


def _market_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "weather_api:\n  latitude: 7.8567\n  longitude: 80.6517\n", encoding="utf-8"
    )
    return config_path


# 1. Mapped vegetable resolves to its production region, with correct coordinates ----


def test_resolve_returns_production_region_for_mapped_vegetable(tmp_path):
    resolver = ProductionRegionResolver(
        mapping_path=str(_mapping_yaml(tmp_path)), config_path=str(_market_config(tmp_path))
    )
    result = resolver.resolve("Carrot")

    assert result.is_production_region is True
    assert result.region_name == "Nuwara Eliya"
    assert result.latitude == 6.97078
    assert result.longitude == 80.78286
    assert result.confidence == "high"
    assert result.evidence == "test evidence"


def test_resolve_returns_correct_coordinates_for_a_different_mapped_vegetable(tmp_path):
    resolver = ProductionRegionResolver(
        mapping_path=str(_mapping_yaml(tmp_path)), config_path=str(_market_config(tmp_path))
    )
    result = resolver.resolve("Big Onion (Local)")

    assert result.region_name == "Jaffna"
    assert result.latitude == 9.66845
    assert result.longitude == 80.00742


# 2. Case/whitespace-tolerant matching ------------------------------------------------


def test_resolve_is_case_and_whitespace_tolerant(tmp_path):
    resolver = ProductionRegionResolver(
        mapping_path=str(_mapping_yaml(tmp_path)), config_path=str(_market_config(tmp_path))
    )
    result = resolver.resolve("  carrot  ")

    assert result.is_production_region is True
    assert result.region_name == "Nuwara Eliya"


# 3. Unmapped/unknown vegetable falls back to market location, never a guess ---------


def test_resolve_falls_back_to_market_location_for_unmapped_vegetable(tmp_path):
    resolver = ProductionRegionResolver(
        mapping_path=str(_mapping_yaml(tmp_path)), config_path=str(_market_config(tmp_path))
    )
    result = resolver.resolve("Tomato")

    assert result.is_production_region is False
    assert result.latitude == 7.8567
    assert result.longitude == 80.6517
    assert result.evidence is None
    assert result.confidence is None


def test_resolve_falls_back_to_market_location_for_unknown_vegetable_name(tmp_path):
    resolver = ProductionRegionResolver(
        mapping_path=str(_mapping_yaml(tmp_path)), config_path=str(_market_config(tmp_path))
    )
    result = resolver.resolve("Not A Real Vegetable")

    assert result.is_production_region is False
    assert result.latitude == 7.8567
    assert result.longitude == 80.6517


# 4. Missing mapping file: every vegetable falls back, doesn't raise -----------------


def test_resolve_falls_back_for_every_vegetable_when_mapping_file_is_missing(tmp_path):
    resolver = ProductionRegionResolver(
        mapping_path=str(tmp_path / "does_not_exist.yaml"), config_path=str(_market_config(tmp_path))
    )
    result = resolver.resolve("Carrot")  # would be mapped if the file existed

    assert result.is_production_region is False
    assert result.latitude == 7.8567
    assert result.longitude == 80.6517


# 5. Market location is independent of which market is asked about ------------------
#    (resolve() takes no market argument at all -- structurally cannot depend on it)


def test_resolve_signature_has_no_market_parameter():
    import inspect

    signature = inspect.signature(ProductionRegionResolver.resolve)
    assert "market" not in signature.parameters
