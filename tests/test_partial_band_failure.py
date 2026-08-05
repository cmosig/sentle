"""Tests for subtiles whose band downloads partly failed (issue #87).

``process_S2_subtile`` allocated its output with ``np.empty`` and filled it band
by band. A band whose read raised ``RasterioIOError`` was only warned about, so
its plane kept whatever was in that heap memory -- and the guard further down
(``s2_tile_transform``/``s2_crs``) is satisfied by a successful B02 alone, so the
garbage was fed to the cloud classifier and written to the zarr store as
reflectance.

That path used to be nearly unreachable because there were no HTTP timeouts at
all: a stalled read hung instead of raising. Adding the timeouts turns those
hangs into caught ``RasterioIOError``s, which is exactly what makes this matter.

Offline: ``rasterio.open`` is faked, the hrefs are never opened.
"""

import numpy as np
import pytest
import rasterio
from rasterio import transform, windows
from rasterio.crs import CRS
from rasterio.enums import Resampling

from sentle import sentinel2
from sentle.const import S2_RAW_BANDS

SIZE = 32
POISON = np.float32(1234.5)


class _FakeReader:

    def __init__(self, crs, tf):
        self.crs = crs
        self.transform = tf

    def read(self, indexes, window, out_shape, out_dtype, **kwargs):
        return np.full(out_shape, 5000, dtype=out_dtype)

    def close(self):
        pass


class _Asset:

    def __init__(self, href):
        self.href = href


class _Item:
    id = "S2A_MSIL2A_20230615T102031_N0510_R065_T32TPS_20240912T065622"
    # baseline < 4.0 -> no harmonization, so the values below stay exact
    properties = {"s2:processing_baseline": "02.14", "s2:mgrs_tile": "32TPS"}

    def __init__(self):
        self.assets = {b: _Asset(f"https://host/{b}.tif") for b in S2_RAW_BANDS}


def _poison_heap():
    # free a few blocks of the right size filled with a plausible reflectance,
    # so an uninitialised allocation is likely to hand them straight back
    for _ in range(6):
        block = np.full((len(S2_RAW_BANDS), SIZE, SIZE), POISON,
                        dtype=np.float32)
        del block


def _run(monkeypatch, failing_band):
    monkeypatch.setattr(sentinel2, "S2_subtile_size", SIZE)
    tf = transform.from_origin(600000, 5100000, 10, 10)
    crs = CRS.from_epsg(32632)

    def fake_open(href, *args, **kwargs):
        if failing_band is not None and f"/{failing_band}." in href:
            raise rasterio.errors.RasterioIOError(f"cannot open {href}")
        return _FakeReader(crs, tf)

    monkeypatch.setattr(sentinel2.rasterio, "open", fake_open)
    _poison_heap()

    return sentinel2.process_S2_subtile(
        intersecting_windows=windows.Window(0, 0, SIZE, SIZE),
        stac_item=_Item(),
        timestamp=0,
        target_crs=crs,
        target_resolution=10,
        ptile_transform=transform.from_origin(600000, 5100000, 10, 10),
        ptile_width=SIZE,
        ptile_height=SIZE,
        S2_mask_snow=False,
        S2_cloud_classification=False,
        S2_cloud_classification_device="cpu",
        S2_nbar=False,
        cloud_request_queue=None,
        cloud_response_queue=None,
        resampling_method=Resampling.nearest,
    )


def test_download_buffer_is_never_uninitialised(monkeypatch):
    # defense in depth behind the drop guard below: the buffer used to be
    # np.empty, so a band whose read failed kept whatever was in that heap
    # memory. A run that stops at the guard never reaches the reprojection, so
    # any np.empty here would be that buffer.
    shapes = []
    real_empty = np.empty

    def spy_empty(shape, *args, **kwargs):
        shapes.append(shape)
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(sentinel2.np, "empty", spy_empty)

    with pytest.warns(UserWarning):
        array, _, _ = _run(monkeypatch, "B03")

    assert array is None
    assert shapes == []


def test_incomplete_subtile_never_reaches_the_cloud_classifier(monkeypatch):
    # the classifier consumes all 12 bands, so a blank band would silently
    # corrupt the cloud mask of the bands that did load
    calls = []
    monkeypatch.setattr(sentinel2, "worker_get_cloud_mask",
                        lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sentinel2, "S2_subtile_size", SIZE)

    tf = transform.from_origin(600000, 5100000, 10, 10)
    crs = CRS.from_epsg(32632)

    def fake_open(href, *args, **kwargs):
        if "/B03." in href:
            raise rasterio.errors.RasterioIOError(f"cannot open {href}")
        return _FakeReader(crs, tf)

    monkeypatch.setattr(sentinel2.rasterio, "open", fake_open)

    with pytest.warns(UserWarning, match="incomplete_band_set"):
        sentinel2.process_S2_subtile(
            intersecting_windows=windows.Window(0, 0, SIZE, SIZE),
            stac_item=_Item(),
            timestamp=0,
            target_crs=crs,
            target_resolution=10,
            ptile_transform=tf,
            ptile_width=SIZE,
            ptile_height=SIZE,
            S2_mask_snow=False,
            S2_cloud_classification=True,
            S2_cloud_classification_device="cpu",
            S2_nbar=False,
            cloud_request_queue=None,
            cloud_response_queue=None,
            resampling_method=Resampling.nearest,
        )

    assert calls == []


@pytest.mark.parametrize("failing_band", ["B03", "B01", "B12"])
def test_failed_band_drops_the_subtile(monkeypatch, failing_band):
    with pytest.warns(UserWarning, match="incomplete_band_set"):
        array, write_win, band_names = _run(monkeypatch, failing_band)

    assert (array, write_win, band_names) == (None, None, None)


def test_missing_b02_still_drops_the_subtile(monkeypatch):
    # pre-existing behaviour: without B02 there is no tile transform at all
    with pytest.warns(UserWarning, match="stac_read_failure"):
        array, write_win, band_names = _run(monkeypatch, "B02")

    assert (array, write_win, band_names) == (None, None, None)


def test_all_bands_ok_still_returns_data(monkeypatch):
    array, write_win, band_names = _run(monkeypatch, None)

    assert array is not None and write_win is not None
    assert band_names == S2_RAW_BANDS
    assert array.shape[0] == len(S2_RAW_BANDS)
    # no harmonization at this baseline, so every valid pixel is the raw 5000
    assert set(np.unique(array)) <= {0.0, 5000.0}
    assert (array == 5000.0).any()
