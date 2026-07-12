"""Tests for the temporal-composite aggregation methods (issue #57).

``time_composite_method`` selects how the acquisitions within one
``time_composite_freq`` window are aggregated: ``mean`` (default, streaming),
or ``median`` / ``min`` / ``max`` (buffered, NoData-aware). These tests drive
the real ``process_ptile_S2_dispatcher`` with monkeypatched per-timestamp
arrays so the aggregation math runs without any download.
"""

import numpy as np
import pytest

from sentle import sentinel2
from sentle.const import S2_RAW_BANDS

PTILE_H = PTILE_W = 2


class _FakeItem:
    def __init__(self, tile, ts):
        self.properties = {"s2:mgrs_tile": tile}
        self.datetime = ts


def _arr(fill):
    return np.full((len(S2_RAW_BANDS), PTILE_H, PTILE_W), fill, dtype=np.float32)


def _run(monkeypatch, per_ts_values, method):
    # one acquisition per value, at distinct timestamps
    arrays = []
    for v in per_ts_values:
        a = _arr(v)
        a[:, 1, 1] = 0.0  # (1,1) is NoData in every acquisition
        arrays.append(a)
    item_list = [_FakeItem(f"T{i}", i) for i in range(len(arrays))]
    by_ts = {i: arrays[i] for i in range(len(arrays))}

    def fake_process_ptile_S2(*, timestamp, **kwargs):
        return by_ts[timestamp].copy(), list(S2_RAW_BANDS)

    monkeypatch.setattr(sentinel2, "process_ptile_S2", fake_process_ptile_S2)

    return sentinel2.process_ptile_S2_dispatcher(
        target_crs=None, target_resolution=10.0,
        S2_cloud_classification_device="cpu", time_composite_freq="7D",
        S2_apply_snow_mask=False, S2_apply_cloud_mask=False,
        S2_bands_to_save=list(S2_RAW_BANDS), ptile_height=PTILE_H,
        ptile_width=PTILE_W, ptile_transform=None, item_list=item_list, ts=None,
        bound_left=0, bound_right=0, bound_bottom=0, bound_top=0,
        S2_mask_snow=False, S2_cloud_classification=False,
        S2_return_cloud_probabilities=False, S2_nbar=False, S2_subtiles=None,
        cloud_request_queue=None, cloud_response_queue=None,
        resampling_method=None, time_composite_method=method)


# values chosen so mean != median != min != max
VALUES = [100.0, 100.0, 400.0]  # mean 200, median 100, min 100, max 400
EXPECTED = {"mean": 200.0, "median": 100.0, "min": 100.0, "max": 400.0}


@pytest.mark.parametrize("method,expected", list(EXPECTED.items()))
def test_composite_method_reduces_correctly(monkeypatch, method, expected):
    out = _run(monkeypatch, VALUES, method)
    # a valid pixel is reduced with the requested method
    assert np.allclose(out[:, 0, 0], expected), (
        f"{method}: expected {expected}, got {out[0, 0, 0]}")


@pytest.mark.parametrize("method", list(EXPECTED))
def test_all_nodata_pixel_is_nan_for_every_method(monkeypatch, method):
    out = _run(monkeypatch, VALUES, method)
    # (1,1) was NoData (0) in every acquisition -> NaN regardless of method
    assert np.all(np.isnan(out[:, 1, 1]))


def test_median_ignores_nodata_per_pixel(monkeypatch):
    # [100, NoData, 400] -> median over the two valid values = 250
    out = _run(monkeypatch, [100.0, 0.0, 400.0], "median")
    assert np.allclose(out[:, 0, 0], 250.0)


def test_min_ignores_nodata_per_pixel(monkeypatch):
    # [300, NoData, 500] -> min over valid = 300 (NoData/0 not treated as min)
    out = _run(monkeypatch, [300.0, 0.0, 500.0], "min")
    assert np.allclose(out[:, 0, 0], 300.0)
