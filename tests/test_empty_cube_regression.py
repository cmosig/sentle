"""Regression tests for the "empty cube" bug.

Root cause (see EMPTY_CUBE_BUG_REPORT.md): in ``process_ptile_S2`` and
``process_ptile_S2_dispatcher`` the band-list variable was used both as the
per-iteration unpack target *and* as the post-loop "did anything succeed?"
sentinel. When the **last** iterated subtile / acquisition returned ``None``
(e.g. an out-of-bounds edge subtile at a large ``processing_spatial_chunk_size``
where a block straddles a UTM-zone boundary), it clobbered the sentinel back to
``None`` and the whole composite was silently discarded -- even though many
earlier subtiles / acquisitions held valid data.

These tests reproduce exactly that ordering (valid items first, an invalid item
last) and assert that the composite is still produced. They fail against the
pre-fix code and pass after decoupling the unpack target from the sentinel.
"""

import numpy as np
import pandas as pd
import pytest

from sentle import sentinel2
from sentle.const import S2_RAW_BANDS

PTILE_H = 4
PTILE_W = 4


def _full_window():
    from rasterio import windows
    return windows.Window(col_off=0, row_off=0, width=PTILE_W, height=PTILE_H)


def test_process_ptile_S2_keeps_data_when_last_subtile_out_of_bounds(monkeypatch):
    """A valid composite must survive even if the *last* subtile is out of
    bounds (returns ``(None, None, None)``)."""

    # three subtiles iterated in this order; the last one is the offending
    # out-of-bounds edge tile.
    subtiles = pd.DataFrame({
        "name": ["T_valid_1", "T_valid_2", "T_edge_oob"],
        "intersecting_windows": [None, None, None],
    })
    # every subtile has a matching STAC item for this timestamp
    items = pd.DataFrame({
        "tile": ["T_valid_1", "T_valid_2", "T_edge_oob"],
        "item": ["item1", "item2", "item3"],
        "ts": [0, 0, 0],
    })

    def fake_process_S2_subtile(*, intersecting_windows, stac_item, **kwargs):
        # the edge tile maps entirely outside the ptile -> no valid window
        if stac_item == "item3":
            return None, None, None
        arr = np.ones((len(S2_RAW_BANDS), PTILE_H, PTILE_W), dtype=np.float32)
        return arr, _full_window(), list(S2_RAW_BANDS)

    monkeypatch.setattr(sentinel2, "process_S2_subtile", fake_process_S2_subtile)

    result_array, result_bands = sentinel2.process_ptile_S2(
        timestamp=0,
        target_crs=None,
        target_resolution=10.0,
        S2_cloud_classification=False,
        S2_cloud_classification_device="cpu",
        S2_mask_snow=False,
        S2_return_cloud_probabilities=False,
        S2_nbar=False,
        subtiles=subtiles,
        ptile_transform=None,
        ptile_width=PTILE_W,
        ptile_height=PTILE_H,
        items=items,
        cloud_request_queue=None,
        cloud_response_queue=None,
        resampling_method=None,
    )

    assert result_array is not None, (
        "composite was discarded because the last subtile was out of bounds")
    assert result_bands == list(S2_RAW_BANDS)
    # the two valid subtiles contributed real (non-zero) data
    assert np.nansum(result_array) > 0


def test_process_ptile_S2_dispatcher_keeps_data_when_last_ts_invalid(monkeypatch):
    """The weekly composite must survive even if the *last* acquisition in the
    window returns ``None`` from ``process_ptile_S2``."""

    class FakeItem:
        def __init__(self, tile, ts):
            self.properties = {"s2:mgrs_tile": tile}
            self.datetime = ts

    # two acquisitions at distinct timestamps; the second (last) one is invalid
    item_list = [FakeItem("T1", 0), FakeItem("T2", 1)]

    def fake_process_ptile_S2(*, timestamp, **kwargs):
        if timestamp == 1:  # last acquisition fails
            return None, None
        arr = np.ones((len(S2_RAW_BANDS), PTILE_H, PTILE_W), dtype=np.float32)
        return arr, list(S2_RAW_BANDS)

    monkeypatch.setattr(sentinel2, "process_ptile_S2", fake_process_ptile_S2)

    result = sentinel2.process_ptile_S2_dispatcher(
        target_crs=None,
        target_resolution=10.0,
        S2_cloud_classification_device="cpu",
        time_composite_freq=None,
        S2_apply_snow_mask=False,
        S2_apply_cloud_mask=False,
        S2_bands_to_save=list(S2_RAW_BANDS),
        ptile_height=PTILE_H,
        ptile_width=PTILE_W,
        ptile_transform=None,
        item_list=item_list,
        ts=None,
        bound_left=0,
        bound_right=0,
        bound_bottom=0,
        bound_top=0,
        S2_mask_snow=False,
        S2_cloud_classification=False,
        S2_return_cloud_probabilities=False,
        S2_nbar=False,
        S2_subtiles=None,
        cloud_request_queue=None,
        cloud_response_queue=None,
        resampling_method=None,
    )

    assert result is not None, (
        "weekly composite was discarded because the last acquisition was None")
    # the first valid acquisition contributed real (non-zero) data
    assert np.nansum(result) > 0
