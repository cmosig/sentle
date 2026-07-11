"""Tests for the Sentinel-2 band-subset feature (issue #7).

When cloud detection is not requested, a user may download only a few raw
bands (e.g. B04/B03/B02 for an RGB cube). These tests check that the subset
flows through ``process_ptile_S2_dispatcher`` and that the NoData mask keys off
the raw bands *actually present* rather than the full ``S2_RAW_BANDS`` set (a
KeyError trap if the subset were ignored).
"""

import numpy as np
import pytest

from sentle import sentinel2
from sentle.const import S2_RAW_BANDS

PTILE_H = PTILE_W = 4
RGB = ["B02", "B03", "B04"]


class _FakeItem:
    def __init__(self, tile, ts):
        self.properties = {"s2:mgrs_tile": tile}
        self.datetime = ts


def _run(monkeypatch, arr, bands, *, freq=None):
    item_list = [_FakeItem("T0", 0)]

    captured = {}

    def fake_process_ptile_S2(*, timestamp, S2_bands, **kwargs):
        captured["S2_bands"] = S2_bands
        return arr.copy(), list(bands)

    monkeypatch.setattr(sentinel2, "process_ptile_S2", fake_process_ptile_S2)

    out = sentinel2.process_ptile_S2_dispatcher(
        target_crs=None, target_resolution=10.0,
        S2_cloud_classification_device="cpu", time_composite_freq=freq,
        S2_apply_snow_mask=False, S2_apply_cloud_mask=False,
        S2_bands_to_save=list(bands), ptile_height=PTILE_H, ptile_width=PTILE_W,
        ptile_transform=None, item_list=item_list, ts=None,
        bound_left=0, bound_right=0, bound_bottom=0, bound_top=0,
        S2_mask_snow=False, S2_cloud_classification=False,
        S2_return_cloud_probabilities=False, S2_nbar=False, S2_subtiles=None,
        cloud_request_queue=None, cloud_response_queue=None,
        resampling_method=None, S2_bands=bands)
    return out, captured


def test_subset_is_forwarded_to_process_ptile_S2(monkeypatch):
    arr = np.full((len(RGB), PTILE_H, PTILE_W), 500.0, dtype=np.float32)
    _, captured = _run(monkeypatch, arr, RGB)
    assert captured["S2_bands"] == RGB


def test_output_has_only_the_requested_bands(monkeypatch):
    arr = np.full((len(RGB), PTILE_H, PTILE_W), 500.0, dtype=np.float32)
    out, _ = _run(monkeypatch, arr, RGB)
    assert out.shape == (len(RGB), PTILE_H, PTILE_W)
    assert np.allclose(out, 500.0)


def test_nodata_mask_uses_only_present_raw_bands(monkeypatch):
    # a zero in one of the (few) requested bands still masks the whole pixel;
    # crucially this must not KeyError trying to index absent S2_RAW_BANDS
    arr = np.full((len(RGB), PTILE_H, PTILE_W), 500.0, dtype=np.float32)
    arr[0, 1, 1] = 0.0  # B02 NoData at (1, 1)
    out, _ = _run(monkeypatch, arr, RGB)
    assert np.all(np.isnan(out[:, 1, 1]))
    # everything else stays valid
    keep = np.delete(out.reshape(len(RGB), -1), 1 * PTILE_W + 1, axis=1)
    assert not np.isnan(keep).any()


def test_single_band_subset_works(monkeypatch):
    one = ["B08"]
    arr = np.full((1, PTILE_H, PTILE_W), 1234.0, dtype=np.float32)
    out, captured = _run(monkeypatch, arr, one)
    assert captured["S2_bands"] == one
    assert out.shape == (1, PTILE_H, PTILE_W)
    assert np.allclose(out, 1234.0)
