"""Test the open-dataset reuse cache in process_S2_subtile (issue #75 / TLM).

For CDSE JP2s the first windowed read on a dataset is expensive (openjpeg
discovers the tile structure), so sentle keeps band datasets open and reuses
them across subtiles of the same tile. This test drives ``process_S2_subtile``
twice with a shared cache and a counting fake ``rasterio.open`` to confirm each
band file is opened once, not once per subtile.
"""

import numpy as np
import pytest
from rasterio import transform, windows
from rasterio.crs import CRS
from rasterio.enums import Resampling

from sentle import sentinel2
from sentle.const import S2_RAW_BANDS


class _FakeReader:
    def __init__(self, crs, tf):
        self.crs = crs
        self.transform = tf

    def read(self, indexes, window, out_shape, out_dtype, **kw):
        return np.full(out_shape, 5000, dtype=out_dtype)

    def close(self):
        pass


class _Asset:
    def __init__(self, href):
        self.href = href


class _Item:
    id = "S2A_MSIL2A_20230615T102031_N0510_R065_T32TPS_20240912T065622"
    properties = {"s2:processing_baseline": "05.10", "s2:mgrs_tile": "32TPS"}

    def __init__(self):
        self.assets = {b: _Asset(f"https://host/{b}.tif") for b in S2_RAW_BANDS}


def _run(monkeypatch, ds_cache):
    SIZE = 60
    monkeypatch.setattr(sentinel2, "S2_subtile_size", SIZE)
    tf = transform.from_origin(600000, 5100000, 10, 10)
    crs = CRS.from_epsg(32632)

    opens = {"n": 0}

    def fake_open(href, *a, **k):
        opens["n"] += 1
        return _FakeReader(crs, tf)

    monkeypatch.setattr(sentinel2.rasterio, "open", fake_open)

    def call():
        return sentinel2.process_S2_subtile(
            intersecting_windows=windows.Window(0, 0, SIZE, SIZE),
            stac_item=_Item(), timestamp=0, target_crs=crs,
            target_resolution=10, ptile_transform=transform.from_origin(
                600000, 5100000, 10, 10),
            ptile_width=SIZE, ptile_height=SIZE, S2_mask_snow=False,
            S2_cloud_classification=False, S2_cloud_classification_device="cpu",
            S2_nbar=False, cloud_request_queue=None, cloud_response_queue=None,
            resampling_method=Resampling.nearest, ds_cache=ds_cache)

    call()
    call()
    return opens["n"]


def test_reuse_opens_each_band_once(monkeypatch):
    cache = {}
    n_opens = _run(monkeypatch, ds_cache=cache)
    # two subtiles, 12 bands, but with a shared cache each file opens once
    assert n_opens == len(S2_RAW_BANDS)
    assert len(cache) == len(S2_RAW_BANDS)


def test_no_cache_opens_each_time(monkeypatch):
    n_opens = _run(monkeypatch, ds_cache=None)
    # without a cache every subtile re-opens every band
    assert n_opens == 2 * len(S2_RAW_BANDS)
