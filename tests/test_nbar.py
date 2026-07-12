"""Tests for the NBAR (sen2nbar) integration fixes (issues #53, #59).

sen2nbar 2024.6.0 reads the source EPSG from ``item.properties["proj:epsg"]``
and fetches the ``granule-metadata`` XML over plain HTTP. Both break against the
current Planetary Computer catalog:

* PC dropped ``proj:epsg`` in favour of ``proj:code`` -> ``KeyError: 'proj:epsg'``
* the granule-metadata blob needs a signed (SAS) URL.

``sentle.nbar`` now prepares the item (injects ``proj:epsg`` from the known tile
CRS, signs the metadata href) before calling sen2nbar, and applies NBAR on a
best-effort basis (warn + continue) so a bad scene doesn't abort the whole run.
"""

import numpy as np
import pytest
import xarray as xr
from rasterio.crs import CRS

from sentle import nbar
from sentle.const import S2_NBAR_BANDS


class _FakeAsset:
    def __init__(self, href):
        self.href = href


class _FakeItem:
    def __init__(self, item_id="S2_TEST", properties=None, metadata_href=None):
        self.id = item_id
        self.properties = properties if properties is not None else {}
        self.assets = {}
        if metadata_href is not None:
            self.assets["granule-metadata"] = _FakeAsset(metadata_href)


class TestPrepareItem:
    def test_injects_proj_epsg_from_crs(self):
        item = _FakeItem(properties={})  # no proj:epsg (like current PC)
        nbar._prepare_item_for_sen2nbar(item, CRS.from_epsg(32632))
        assert item.properties["proj:epsg"] == 32632

    def test_does_not_overwrite_existing_proj_epsg(self):
        item = _FakeItem(properties={"proj:epsg": 4326})
        nbar._prepare_item_for_sen2nbar(item, CRS.from_epsg(32632))
        assert item.properties["proj:epsg"] == 4326

    def test_signs_granule_metadata_href(self, monkeypatch):
        monkeypatch.setattr(nbar, "refresh_sas_token",
                            lambda url: url + "?sig=SIGNED")
        item = _FakeItem(metadata_href="https://host/MTD_TL.xml")
        nbar._prepare_item_for_sen2nbar(item, CRS.from_epsg(32632))
        assert item.assets["granule-metadata"].href.endswith("?sig=SIGNED")

    def test_missing_metadata_asset_is_tolerated(self):
        item = _FakeItem(metadata_href=None)  # no granule-metadata asset
        # must not raise
        nbar._prepare_item_for_sen2nbar(item, CRS.from_epsg(32632))

    def test_falls_back_to_proj_code_when_crs_has_no_epsg(self, monkeypatch):
        # a CRS whose to_epsg() is None -> use proj:code from the item
        item = _FakeItem(properties={"proj:code": "EPSG:32633"})

        class _NoEpsgCRS:
            def to_epsg(self):
                return None

        nbar._prepare_item_for_sen2nbar(item, _NoEpsgCRS())
        assert item.properties["proj:epsg"] == 32633


class TestGetCFactorValue:
    def _fake_c(self):
        # a coarse c-factor grid over a UTM subtile, one value per NBAR band
        y = np.arange(5100000 + 7320, 5100000, -120.0)
        x = np.arange(600000, 600000 + 7320, 120.0)
        data = np.ones((len(S2_NBAR_BANDS), len(y), len(x)), np.float32) * 1.03
        return xr.DataArray(data, dims=("band", "y", "x"),
                            coords={"band": list(S2_NBAR_BANDS), "y": y, "x": x})

    def test_prepares_item_and_interpolates(self, monkeypatch):
        nbar.c_factor_cache.clear()
        captured = {}

        def fake_c_factor_from_item(item, to_epsg):
            # the item must have been prepared (proj:epsg injected)
            captured["proj_epsg"] = item.properties.get("proj:epsg")
            captured["to_epsg"] = to_epsg
            return self._fake_c()

        monkeypatch.setattr(nbar, "refresh_sas_token", lambda u: u)
        monkeypatch.setattr(nbar.sen2nbar.c_factor, "c_factor_from_item",
                            fake_c_factor_from_item)

        item = _FakeItem(metadata_href="https://host/MTD_TL.xml")
        bounds = (600000, 5100000, 600000 + 7320, 5100000 + 7320)
        out = nbar.get_c_factor_value(item, CRS.from_epsg(32632), bounds)

        assert captured["proj_epsg"] == 32632
        assert captured["to_epsg"] == "EPSG:32632"
        # interpolated onto the 732x732 subtile grid
        assert out.shape == (len(S2_NBAR_BANDS), 732, 732)
        assert np.allclose(out, 1.03, atol=1e-3)

    def test_result_is_cached_per_item(self, monkeypatch):
        nbar.c_factor_cache.clear()
        calls = {"n": 0}

        def fake_c_factor_from_item(item, to_epsg):
            calls["n"] += 1
            return self._fake_c()

        monkeypatch.setattr(nbar, "refresh_sas_token", lambda u: u)
        monkeypatch.setattr(nbar.sen2nbar.c_factor, "c_factor_from_item",
                            fake_c_factor_from_item)

        item = _FakeItem(metadata_href="https://host/MTD_TL.xml")
        bounds = (600000, 5100000, 600000 + 7320, 5100000 + 7320)
        nbar.get_c_factor_value(item, CRS.from_epsg(32632), bounds)
        nbar.get_c_factor_value(item, CRS.from_epsg(32632), bounds)
        # sen2nbar is the expensive call -> only invoked once (cached by id)
        assert calls["n"] == 1


class TestNbarRobustness:
    """Issue #59: an NBAR failure for one scene must warn and continue, not
    abort the whole run."""

    def test_nbar_failure_is_non_fatal(self, monkeypatch):
        import numpy as np
        from rasterio import transform, windows
        from rasterio.crs import CRS as RioCRS
        from rasterio.enums import Resampling
        from sentle import sentinel2
        from sentle.const import S2_RAW_BANDS

        SIZE = 60
        # shrink the subtile so the test arrays/reproject stay tiny
        monkeypatch.setattr(sentinel2, "S2_subtile_size", SIZE)

        tile_tf = transform.from_origin(600000, 5100000, 10, 10)

        class _FakeReader:
            crs = RioCRS.from_epsg(32632)
            transform = tile_tf

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def close(self):
                pass

            def read(self, indexes, window, out_shape, out_dtype, **kw):
                return np.full(out_shape, 5000, dtype=out_dtype)

        monkeypatch.setattr(sentinel2.rasterio, "open",
                            lambda *a, **k: _FakeReader())
        # make NBAR blow up the way a missing/unreadable metadata scene would
        def boom(*a, **k):
            raise RuntimeError("granule metadata unavailable")
        monkeypatch.setattr(sentinel2, "get_c_factor_value", boom)

        assets = {b: _FakeAsset(f"https://host/{b}.tif") for b in S2_RAW_BANDS}

        class _Item:
            id = "S2_BADMETA"
            properties = {"s2:processing_baseline": "5.0"}

        item = _Item()
        item.assets = assets

        win = windows.Window(0, 0, SIZE, SIZE)
        ptile_tf = transform.from_origin(600000, 5100000, 10, 10)

        with pytest.warns(UserWarning, match="nbar_failure"):
            arr, write_win, bands = sentinel2.process_S2_subtile(
                intersecting_windows=win,
                stac_item=item,
                timestamp=0,
                target_crs=RioCRS.from_epsg(32632),
                target_resolution=10,
                ptile_transform=ptile_tf,
                ptile_width=SIZE,
                ptile_height=SIZE,
                S2_mask_snow=False,
                S2_cloud_classification=False,
                S2_cloud_classification_device="cpu",
                S2_nbar=True,
                cloud_request_queue=None,
                cloud_response_queue=None,
                resampling_method=Resampling.nearest,
            )

        # NBAR was skipped, but the subtile still came through with reflectance
        assert arr is not None and write_win is not None
        assert bands == list(S2_RAW_BANDS)
        assert np.nanmax(arr) > 0
