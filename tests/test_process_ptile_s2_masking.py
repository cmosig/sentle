"""Integration tests for the Sentinel-2 masking / compositing in
``process_ptile_S2_dispatcher``.

The dispatcher is where per-acquisition arrays get their snow and cloud masks
applied and, in temporal-composite mode, where the mask/classification bands are
dropped from the saved output. These tests drive the *real* dispatcher with
monkeypatched per-timestamp arrays (so no download / no model runs) and assert
the masking and band-bookkeeping behaviour.

This complements ``test_nodata_zero_reproject.py`` (which focuses on the ``== 0``
NoData/cloud masks) and ``test_empty_cube_regression.py`` (which focuses on the
"last acquisition is None" sentinel bug).
"""

import numpy as np
import pytest

from sentle import sentinel2
from sentle.cloud_mask import S2_cloud_mask_band
from sentle.const import S2_RAW_BANDS
from sentle.snow_mask import S2_snow_mask_band

PTILE_H = PTILE_W = 4


class _FakeItem:
    def __init__(self, tile, ts):
        self.properties = {"s2:mgrs_tile": tile}
        self.datetime = ts


def _run(monkeypatch, arrays, per_ts_bands, *, saved_bands, apply_snow=False,
         apply_cloud=False, freq=None):
    """Drive the dispatcher.

    ``arrays`` / ``per_ts_bands`` are what the (faked) ``process_ptile_S2``
    returns for each acquisition -- i.e. *with* the snow/cloud bands present.
    ``saved_bands`` is what ``process()`` would allocate the output for (raw
    bands only when compositing, since the mask bands are dropped).
    """
    item_list = [_FakeItem(f"T{i}", i) for i in range(len(arrays))]
    by_ts = {i: arrays[i] for i in range(len(arrays))}

    def fake_process_ptile_S2(*, timestamp, **kwargs):
        return by_ts[timestamp].copy(), list(per_ts_bands)

    monkeypatch.setattr(sentinel2, "process_ptile_S2", fake_process_ptile_S2)

    return sentinel2.process_ptile_S2_dispatcher(
        target_crs=None,
        target_resolution=10.0,
        S2_cloud_classification_device="cpu",
        time_composite_freq=freq,
        S2_apply_snow_mask=apply_snow,
        S2_apply_cloud_mask=apply_cloud,
        S2_bands_to_save=list(saved_bands),
        ptile_height=PTILE_H,
        ptile_width=PTILE_W,
        ptile_transform=None,
        item_list=item_list,
        ts=None,
        bound_left=0, bound_right=0, bound_bottom=0, bound_top=0,
        S2_mask_snow=apply_snow,
        S2_cloud_classification=apply_cloud,
        S2_return_cloud_probabilities=False,
        S2_nbar=False,
        S2_subtiles=None,
        cloud_request_queue=None,
        cloud_response_queue=None,
        resampling_method=None,
    )


def _raw(fill=500.0):
    return np.full((len(S2_RAW_BANDS), PTILE_H, PTILE_W), fill, dtype=np.float32)


def test_single_acquisition_passes_data_through_unchanged(monkeypatch):
    """No masks, one acquisition -> the reflectance is returned verbatim."""
    out = _run(monkeypatch, [_raw(500.0)], S2_RAW_BANDS,
               saved_bands=S2_RAW_BANDS)
    assert out.shape == (len(S2_RAW_BANDS), PTILE_H, PTILE_W)
    assert np.allclose(out, 500.0)
    assert not np.isnan(out).any()


def test_snow_pixel_is_masked_to_nan(monkeypatch):
    """A pixel flagged as snow (snow band == 0) is zeroed and becomes NaN."""
    bands = list(S2_RAW_BANDS) + [S2_snow_mask_band]
    arr = np.full((len(bands), PTILE_H, PTILE_W), 500.0, dtype=np.float32)
    arr[-1, :, :] = 1.0        # 1 == clear ...
    arr[-1, 2, 2] = 0.0        # ... 0 == snow at one pixel

    out = _run(monkeypatch, [arr], bands, saved_bands=bands, apply_snow=True)

    assert np.all(np.isnan(out[:, 2, 2]))          # snow pixel dropped
    assert np.allclose(out[:len(S2_RAW_BANDS), 0, 0], 500.0)  # clear kept


def test_temporal_composite_drops_snow_and_cloud_bands(monkeypatch):
    """In composite mode the snow/cloud bands are stripped from the output, so
    the saved cube carries only the raw reflectance bands."""
    per_ts_bands = list(S2_RAW_BANDS) + [S2_cloud_mask_band, S2_snow_mask_band]
    arr = np.full((len(per_ts_bands), PTILE_H, PTILE_W), 500.0, dtype=np.float32)
    arr[len(S2_RAW_BANDS), :, :] = 0.0           # cloud band: all clear
    arr[-1, :, :] = 1.0                          # snow band: all clear

    out = _run(monkeypatch, [arr], per_ts_bands,
               saved_bands=list(S2_RAW_BANDS),  # process() allocates raw-only
               apply_snow=True, apply_cloud=True, freq="2W")

    assert out.shape[0] == len(S2_RAW_BANDS)
    assert np.allclose(out, 500.0)


def test_temporal_composite_masks_cloudy_pixel(monkeypatch):
    """A cloudy pixel (cloud band != 0) in composite mode is dropped to NaN."""
    per_ts_bands = list(S2_RAW_BANDS) + [S2_cloud_mask_band, S2_snow_mask_band]
    arr = np.full((len(per_ts_bands), PTILE_H, PTILE_W), 500.0, dtype=np.float32)
    cloud_idx = len(S2_RAW_BANDS)
    arr[cloud_idx, :, :] = 0.0        # clear ...
    arr[cloud_idx, 1, 1] = 2.0        # ... thick cloud at one pixel
    arr[-1, :, :] = 1.0              # snow band all clear

    out = _run(monkeypatch, [arr], per_ts_bands,
               saved_bands=list(S2_RAW_BANDS),
               apply_snow=True, apply_cloud=True, freq="2W")

    assert np.all(np.isnan(out[:, 1, 1]))            # cloudy pixel dropped
    assert np.allclose(out[:, 0, 0], 500.0)          # clear pixel kept


def test_no_valid_acquisitions_returns_none(monkeypatch):
    """If every acquisition returns None, the whole ptile is None (nothing to
    write) rather than an all-zero array."""
    item_list = [_FakeItem("T0", 0)]

    def fake_none(*, timestamp, **kwargs):
        return None, None

    monkeypatch.setattr(sentinel2, "process_ptile_S2", fake_none)

    out = sentinel2.process_ptile_S2_dispatcher(
        target_crs=None, target_resolution=10.0,
        S2_cloud_classification_device="cpu", time_composite_freq=None,
        S2_apply_snow_mask=False, S2_apply_cloud_mask=False,
        S2_bands_to_save=list(S2_RAW_BANDS), ptile_height=PTILE_H,
        ptile_width=PTILE_W, ptile_transform=None, item_list=item_list, ts=None,
        bound_left=0, bound_right=0, bound_bottom=0, bound_top=0,
        S2_mask_snow=False, S2_cloud_classification=False,
        S2_return_cloud_probabilities=False, S2_nbar=False, S2_subtiles=None,
        cloud_request_queue=None, cloud_response_queue=None,
        resampling_method=None)

    assert out is None
