"""Unit tests for the data-provider abstraction (issue #75).

sentle can read from Planetary Computer (default) or the Copernicus Data Space
Ecosystem (CDSE). The providers differ in asset naming, href preparation and
which item properties are available; these offline tests pin that per-provider
logic (no network).
"""

import contextlib

import pytest

from sentle.stac import (
    CDSEProvider,
    PlanetaryComputerProvider,
    get_provider,
)


class _Asset:
    def __init__(self, href):
        self.href = href


class _Item:
    def __init__(self, item_id="", properties=None, assets=None):
        self.id = item_id
        self.properties = properties or {}
        self.assets = assets or {}


CDSE_ID = "S2A_MSIL2A_20230615T102031_N0510_R065_T32TPS_20240912T065622"


class TestGetProvider:
    def test_default_and_names(self):
        assert isinstance(get_provider("planetary_computer"),
                          PlanetaryComputerProvider)
        assert isinstance(get_provider("cdse"), CDSEProvider)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown provider"):
            get_provider("earthsearch")


class TestPlanetaryComputer:
    def setup_method(self):
        self.p = PlanetaryComputerProvider()

    def test_asset_key_is_bare_band(self):
        assert self.p.s2_asset_key("B02") == "B02"
        assert self.p.s2_asset_key("B8A") == "B8A"

    def test_mgrs_tile_from_property(self):
        item = _Item(properties={"s2:mgrs_tile": "32TPS"})
        assert self.p.s2_mgrs_tile(item) == "32TPS"

    def test_baseline_from_property(self):
        item = _Item(properties={"s2:processing_baseline": "05.10"})
        assert self.p.s2_processing_baseline(item) == 5.10

    def test_rasterio_env_is_noop(self):
        assert isinstance(self.p.rasterio_env(),
                          contextlib.nullcontext)

    def test_supports_sentinel1(self):
        assert self.p.supports_sentinel1 is True
        assert self.p.s1_collection == "sentinel-1-rtc"


class TestCDSE:
    def setup_method(self):
        self.p = CDSEProvider()

    @pytest.mark.parametrize("band,expected", [
        ("B02", "B02_10m"),   # 10 m
        ("B05", "B05_20m"),   # 20 m
        ("B01", "B01_60m"),   # 60 m
        ("B8A", "B8A_20m"),
    ])
    def test_asset_key_has_resolution_suffix(self, band, expected):
        assert self.p.s2_asset_key(band) == expected

    def test_mgrs_tile_parsed_from_id(self):
        assert self.p.s2_mgrs_tile(_Item(item_id=CDSE_ID)) == "32TPS"

    def test_baseline_parsed_from_id(self):
        # N0510 -> 5.10
        assert self.p.s2_processing_baseline(_Item(item_id=CDSE_ID)) == 5.10

    def test_bad_id_raises(self):
        with pytest.raises(ValueError, match="MGRS tile"):
            self.p.s2_mgrs_tile(_Item(item_id="not-a-product-id"))

    def test_prepare_href_s3_to_vsis3(self):
        assert self.p.prepare_href("s3://eodata/a/b.jp2") == "/vsis3/eodata/a/b.jp2"

    def test_prepare_href_passthrough_non_s3(self):
        assert self.p.prepare_href("/local/x.jp2") == "/local/x.jp2"

    def test_no_sentinel1(self):
        assert self.p.supports_sentinel1 is False
        assert self.p.s1_collection is None

    def test_granule_metadata_key_is_underscore(self):
        item = _Item(assets={"granule_metadata": _Asset("s3://eodata/MTD.xml")})
        assert self.p.granule_metadata_href(item) == "s3://eodata/MTD.xml"
