"""Tests for ``setup_zarr_storage`` -- the on-disk cube skeleton.

Before any tile is downloaded, sentle lays out the full Zarr store: the 4-D
``sentle`` data array plus the ``time``/``band``/``y``/``x`` coordinate arrays
and the CRS attribute that let xarray open the result as a georeferenced cube.
These tests build a small store offline and assert its shape, coordinates,
dtype/fill and metadata -- the contract every downstream reader depends on.
"""

import numpy as np
import pandas as pd
import pytest
import zarr
from rasterio.crs import CRS

from sentle.sentle import setup_zarr_storage

RES = 10.0
LEFT, BOTTOM, RIGHT, TOP = 300000.0, 5000000.0, 300500.0, 5000300.0
HEIGHT, WIDTH = 30, 50  # (TOP-BOTTOM)/RES, (RIGHT-LEFT)/RES
BANDS = ["B02", "B03", "B04"]


def _timestamps():
    return [
        {"collection": "sentinel-2-l2a", "ts": pd.Timestamp("2020-06-15",
                                                            tz="UTC")},
        {"collection": "sentinel-2-l2a", "ts": pd.Timestamp("2020-06-20",
                                                            tz="UTC")},
    ]


def _build(tmp_path, **overrides):
    kwargs = dict(
        zarr_store=str(tmp_path / "cube.zarr"),
        timestamp_list=_timestamps(),
        height=HEIGHT,
        width=WIDTH,
        bound_left=LEFT,
        bound_right=RIGHT,
        bound_top=TOP,
        bound_bottom=BOTTOM,
        target_resolution=RES,
        processing_spatial_chunk_size=4000,
        zarr_store_chunk_size={"time": 10, "y": 250, "x": 250},
        S2_bands_to_save=BANDS,
        total_bands_to_save=BANDS,
        target_crs=CRS.from_epsg(32633),
        consolidate_metadata=True,
        coord_save_mode="top-left",
        save_as_uint16=False,
    )
    kwargs.update(overrides)
    store_path = kwargs["zarr_store"]
    setup_zarr_storage(**kwargs)
    return zarr.open(store_path, mode="r")


def test_data_array_shape_and_dims(tmp_path):
    root = _build(tmp_path)
    data = root["sentle"]
    assert data.shape == (2, len(BANDS), HEIGHT, WIDTH)
    assert data.dtype == np.float32


def test_nodata_fill_is_nan_for_float(tmp_path):
    root = _build(tmp_path)
    # freshly-created chunks read back as the fill value (NaN)
    assert np.isnan(root["sentle"][0, 0, 0, 0])


def test_crs_stored_as_wkt_attr(tmp_path):
    root = _build(tmp_path)
    assert root.attrs["crs_wkt"] == CRS.from_epsg(32633).to_wkt()


def test_band_coordinate_matches_input(tmp_path):
    root = _build(tmp_path)
    assert list(root["band"][:]) == BANDS


def test_x_y_coordinates_top_left(tmp_path):
    root = _build(tmp_path)
    x = root["x"][:]
    y = root["y"][:]
    assert len(x) == WIDTH and len(y) == HEIGHT
    # top-left convention: first pixel edge sits exactly on the bound
    assert x[0] == pytest.approx(LEFT)
    assert y[0] == pytest.approx(TOP)
    # x increases, y decreases, both at the target resolution
    assert np.allclose(np.diff(x), RES)
    assert np.allclose(np.diff(y), -RES)


def test_x_y_coordinates_center_mode_shift_half_pixel(tmp_path):
    root = _build(tmp_path, coord_save_mode="center")
    x = root["x"][:]
    y = root["y"][:]
    # center convention: shifted inward by half a pixel
    assert x[0] == pytest.approx(LEFT + RES / 2)
    assert y[0] == pytest.approx(TOP - RES / 2)
    assert root["x"].attrs["coord_save_mode"] == "center"


def test_time_coordinate_values_and_order(tmp_path):
    root = _build(tmp_path)
    time = root["time"][:]
    # unique timestamps, most recent first (reverse-sorted)
    expected = [
        int((pd.Timestamp("2020-06-20") - pd.Timestamp(0)).total_seconds()),
        int((pd.Timestamp("2020-06-15") - pd.Timestamp(0)).total_seconds()),
    ]
    assert list(time) == expected
    # xarray needs these to decode the axis as datetimes
    assert root["time"].attrs["units"] == "seconds since 1970-01-01 00:00:00"


def test_duplicate_timestamps_are_collapsed(tmp_path):
    ts = pd.Timestamp("2020-06-15", tz="UTC")
    root = _build(tmp_path, timestamp_list=[
        {"collection": "sentinel-2-l2a", "ts": ts},
        {"collection": "sentinel-1-rtc", "ts": ts},  # same instant
    ])
    assert root["sentle"].shape[0] == 1
    assert len(root["time"][:]) == 1


def test_uint16_mode_dtype_and_zero_fill(tmp_path):
    root = _build(tmp_path, save_as_uint16=True)
    data = root["sentle"]
    assert data.dtype == np.uint16
    # uint16 cubes use 0 as the fill/nodata sentinel, not NaN
    assert data[0, 0, 0, 0] == 0


def test_band_chunk_equals_number_of_s2_bands(tmp_path):
    # S1 and S2 share the band axis; the chunk spans exactly the S2 bands so
    # the two collections can be written independently without chunk conflicts
    root = _build(tmp_path, S2_bands_to_save=BANDS,
                  total_bands_to_save=BANDS + ["vv_asc", "vh_asc"])
    assert root["sentle"].chunks[1] == len(BANDS)
