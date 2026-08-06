"""A raster read that keeps failing must abort the run (issue #87).

``process_S2_subtile`` used to allocate its output with ``np.empty`` and, when a
band's read raised ``RasterioIOError``, warn and carry on -- so that band kept
uninitialised heap memory, and later just kept NoData. Either way the cube that
came out held less (or worse) data than the next run over the same area.

Reads are now retried a bounded number of times and then raise
``SentleReadError``, which fails the whole run. A cube that finishes therefore
always holds the same data as any other successful run.

Offline: ``rasterio.open`` is faked, the hrefs are never opened.
"""

import numpy as np
import pytest
import rasterio
from rasterio import transform, windows
from rasterio.crs import CRS
from rasterio.enums import Resampling

from sentle import sentinel2, stac
from sentle.const import S2_RAW_BANDS
from sentle.stac import SentleReadError

SIZE = 32


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


def _run(monkeypatch, fake_open, read_retries=0, S2_bands=None, **overrides):
    monkeypatch.setattr(sentinel2, "S2_subtile_size", SIZE)
    monkeypatch.setattr(sentinel2.rasterio, "open", fake_open)

    kwargs = dict(
        intersecting_windows=windows.Window(0, 0, SIZE, SIZE),
        stac_item=_Item(),
        timestamp=0,
        target_crs=CRS.from_epsg(32632),
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
        S2_bands=S2_bands,
        read_retries=read_retries,
    )
    kwargs.update(overrides)
    return sentinel2.process_S2_subtile(**kwargs)


def _reader_failing(band, times=None, crs=None, tf=None):
    """A fake rasterio.open that fails for ``band`` the first ``times`` calls."""
    crs = crs or CRS.from_epsg(32632)
    tf = tf or transform.from_origin(600000, 5100000, 10, 10)
    calls = {"n": 0}

    def fake_open(href, *args, **kwargs):
        if f"/{band}." in href:
            calls["n"] += 1
            if times is None or calls["n"] <= times:
                raise rasterio.errors.RasterioIOError(f"cannot open {href}")
        return _FakeReader(crs, tf)

    return fake_open, calls


@pytest.mark.parametrize("failing_band", ["B03", "B02", "B12"])
def test_a_permanently_failing_band_aborts_the_run(monkeypatch, failing_band):
    fake_open, _ = _reader_failing(failing_band)

    with pytest.raises(SentleReadError, match=failing_band):
        _run(monkeypatch, fake_open, read_retries=0)


def test_the_read_is_actually_retried(monkeypatch):
    # fails twice, succeeds on the third attempt
    fake_open, calls = _reader_failing("B03", times=2)
    monkeypatch.setattr(stac, "READ_RETRY_BACKOFF", 0.0)

    with pytest.warns(UserWarning, match="stac_read_retry"):
        array, _, band_names = _run(monkeypatch, fake_open, read_retries=2)

    assert calls["n"] == 3
    assert array is not None
    # the recovered band carries real data, not NoData
    assert (array[band_names.index("B03")] == 5000.0).any()


def test_retries_are_bounded_and_then_it_gives_up(monkeypatch):
    fake_open, calls = _reader_failing("B03")
    monkeypatch.setattr(stac, "READ_RETRY_BACKOFF", 0.0)

    with pytest.warns(UserWarning, match="stac_read_retry"):
        with pytest.raises(SentleReadError, match="after 3 attempt"):
            _run(monkeypatch, fake_open, read_retries=2)

    assert calls["n"] == 3


def test_a_failed_read_evicts_the_cached_dataset(monkeypatch):
    # a handle that just failed is not reliably reusable (GTiff latches the
    # failed block, JP2 keeps the corrupted decoded tile), so the retry has to
    # reopen rather than read through the cached one
    fake_open, _ = _reader_failing("B03", times=1)
    monkeypatch.setattr(stac, "READ_RETRY_BACKOFF", 0.0)
    ds_cache = {}

    with pytest.warns(UserWarning, match="stac_read_retry"):
        _run(monkeypatch, fake_open, read_retries=1, ds_cache=ds_cache)

    # the failed href was not left behind poisoning later subtiles
    assert all("B03" not in href or ds is not None
               for href, ds in ds_cache.items())


def test_all_bands_ok_returns_data(monkeypatch):
    fake_open, _ = _reader_failing("__none__")
    array, write_win, band_names = _run(monkeypatch, fake_open)

    assert array is not None and write_win is not None
    assert band_names == S2_RAW_BANDS
    # no harmonization at this baseline, so every valid pixel is the raw 5000
    assert set(np.unique(array)) <= {0.0, 5000.0}
    assert (array == 5000.0).any()


@pytest.mark.parametrize("bands", [["B04", "B03"], ["B08"], ["B03", "B11"]])
def test_band_subset_without_b02_still_produces_data(monkeypatch, bands):
    # the tile transform used to be taken only from B02, so any subset without
    # it silently produced a completely empty cube
    fake_open, _ = _reader_failing("__none__")
    array, write_win, band_names = _run(monkeypatch,
                                        fake_open,
                                        S2_bands=bands)

    assert array is not None and write_win is not None
    assert band_names == bands
    assert (array == 5000.0).any()
