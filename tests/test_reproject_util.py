"""Unit tests for the geometry / windowing helpers in ``reproject_util``.

These functions are the deterministic backbone of sentle's reprojection and
tiling: they turn requested bounds + a target resolution into pixel grids,
transforms and (clipped) write windows. They are pure -- no network, no GDAL
warp -- so they are cheap to test exhaustively, and getting them wrong silently
mis-aligns or drops data.
"""

import warnings

import numpy as np
import pytest
from affine import Affine
from rasterio import transform, windows
from rasterio.crs import CRS

from sentle.reproject_util import (
    bounds_from_transform_height_width_res,
    calculate_aligned_transform,
    check_and_round_bounds,
    height_width_from_bounds_res,
    pixel_count,
    recrop_write_window,
    transform_height_width_from_bounds_res,
    window_overlaps_bounds,
)


class TestFractionalResolution:
    """Issue #4: fractional resolutions (e.g. 0.1 degrees for EPSG:4326) are
    not exactly representable in floating point, so the pixel-count helpers
    must round tolerantly instead of demanding an exact remainder of 0."""

    def test_pixel_count_fractional_degrees(self):
        # 11.05 - 11.00 = 0.05 (with float error) -> exactly 50 pixels at 0.001
        assert pixel_count(11.05 - 11.00, 0.001) == 50
        assert pixel_count(0.1, 0.1) == 1
        assert pixel_count(0.5, 0.05) == 10

    def test_pixel_count_rejects_non_integer_multiple(self):
        with pytest.raises(AssertionError):
            pixel_count(0.0525, 0.001)  # 52.5 pixels

    def test_height_width_fractional_degrees_returns_ints(self):
        h, w = height_width_from_bounds_res(11.0, 46.0, 11.05, 46.05, 0.001)
        assert (h, w) == (50, 50)
        assert isinstance(h, int) and isinstance(w, int)

    def test_transform_height_width_fractional_degrees(self):
        tf, h, w = transform_height_width_from_bounds_res(
            11.0, 46.0, 11.05, 46.05, 0.001)
        assert (h, w) == (50, 50)
        assert tf.a == pytest.approx(0.001)
        assert tf.e == pytest.approx(-0.001)
        assert tf.c == pytest.approx(11.0)
        assert tf.f == pytest.approx(46.05)

    def test_check_and_round_fractional_divisible_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            left, bottom, right, top = check_and_round_bounds(
                11.0, 46.0, 11.05, 46.05, 0.001)
        # a whole number of pixels -> nothing trimmed
        assert pixel_count(right - left, 0.001) == 50
        assert pixel_count(top - bottom, 0.001) == 50

    def test_check_and_round_fractional_non_divisible_warns_and_trims(self):
        with pytest.warns(UserWarning):
            left, _, right, _ = check_and_round_bounds(
                11.0, 46.0, 11.0525, 46.05, 0.001)
        # trimmed down to a whole number of pixels
        assert pixel_count(right - left, 0.001) == 52


class TestCheckAndRoundBounds:
    def test_exactly_divisible_bounds_are_unchanged_and_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            out = check_and_round_bounds(0, 0, 100, 100, 10)
        assert out == (0, 0, 100, 100)

    def test_non_divisible_width_is_rounded_down_with_warning(self):
        with pytest.warns(UserWarning):
            left, bottom, right, top = check_and_round_bounds(0, 0, 105, 100, 10)
        # 105 % 10 == 5 -> right trimmed to 100; the span becomes a whole
        # number of pixels
        assert right == 100
        assert (right - left) % 10 == 0

    def test_non_divisible_height_is_rounded_down_with_warning(self):
        with pytest.warns(UserWarning):
            left, bottom, right, top = check_and_round_bounds(0, 0, 100, 107, 10)
        assert top == 100
        assert (top - bottom) % 10 == 0

    def test_rounding_uses_resolution_not_integers(self):
        # a fractional resolution must round down to a multiple of that
        # resolution, not to a whole number
        with pytest.warns(UserWarning):
            _, _, right, _ = check_and_round_bounds(0, 0, 11.0, 5.0, 2.5)
        # 11.0 % 2.5 == 1.0 -> trimmed to 10.0 (a multiple of 2.5)
        assert right == pytest.approx(10.0)


class TestHeightWidthFromBoundsRes:
    def test_basic_grid(self):
        h, w = height_width_from_bounds_res(0, 0, 100, 50, 10)
        assert (h, w) == (5, 10)

    def test_asserts_on_non_divisible_bounds(self):
        with pytest.raises(AssertionError):
            height_width_from_bounds_res(0, 0, 105, 50, 10)


class TestTransformHeightWidthFromBoundsRes:
    def test_returns_int_dims_and_consistent_transform(self):
        tf, h, w = transform_height_width_from_bounds_res(500000, 4000000,
                                                          500500, 4000300, 10)
        assert (h, w) == (30, 50)
        assert isinstance(h, int) and isinstance(w, int)
        # top-left corner of the transform is (left, top)
        assert tf.c == pytest.approx(500000)
        assert tf.f == pytest.approx(4000300)
        # pixel size matches the requested resolution
        assert tf.a == pytest.approx(10)
        assert tf.e == pytest.approx(-10)

    def test_asserts_on_non_divisible_bounds(self):
        with pytest.raises(AssertionError):
            transform_height_width_from_bounds_res(0, 0, 100, 55, 10)

    def test_roundtrips_with_bounds_from_transform(self):
        left, bottom, right, top, res = 300000, 5000000, 300800, 5000600, 10
        tf, h, w = transform_height_width_from_bounds_res(left, bottom, right,
                                                         top, res)
        recovered = bounds_from_transform_height_width_res(tf, h, w, res)
        assert recovered == pytest.approx((left, bottom, right, top))


class TestWindowOverlapsBounds:
    @pytest.mark.parametrize("win,expected", [
        (windows.Window(0, 0, 10, 10), True),        # fully inside
        (windows.Window(-5, -5, 10, 10), True),      # straddles top-left corner
        (windows.Window(95, 95, 10, 10), True),      # straddles bottom-right
        (windows.Window(-20, 0, 10, 10), False),     # entirely left
        (windows.Window(0, -20, 10, 10), False),     # entirely above
        (windows.Window(100, 0, 10, 10), False),     # touching right edge -> no area
        (windows.Window(0, 100, 10, 10), False),     # touching bottom edge -> no area
    ])
    def test_overlap(self, win, expected):
        assert window_overlaps_bounds(win, 100, 100) is expected


class TestRecropWriteWindow:
    def test_window_fully_inside_is_unchanged(self):
        win = windows.Window(col_off=10, row_off=20, width=30, height=40)
        write_win, local_win = recrop_write_window(win, 100, 100)
        assert (write_win.col_off, write_win.row_off) == (10, 20)
        assert (write_win.width, write_win.height) == (30, 40)
        # nothing cropped locally
        assert (local_win.col_off, local_win.row_off) == (0, 0)
        assert (local_win.width, local_win.height) == (30, 40)

    def test_overlap_left(self):
        win = windows.Window(col_off=-5, row_off=10, width=20, height=20)
        write_win, local_win = recrop_write_window(win, 100, 100)
        # write clamps to column 0, dropping the 5 out-of-bounds columns
        assert write_win.col_off == 0
        assert write_win.width == 15
        # locally we skip the first 5 columns of the source array
        assert local_win.col_off == 5
        assert local_win.width == 15

    def test_overlap_bottom(self):
        # row_off < 0 is the "bottom" (array-top) overlap branch
        win = windows.Window(col_off=10, row_off=-8, width=20, height=20)
        write_win, local_win = recrop_write_window(win, 100, 100)
        assert write_win.row_off == 0
        assert write_win.height == 12
        assert local_win.row_off == 8
        assert local_win.height == 12

    def test_overlap_right(self):
        win = windows.Window(col_off=90, row_off=10, width=20, height=20)
        write_win, local_win = recrop_write_window(win, 100, 100)
        # only 10 columns fit before the right edge
        assert write_win.col_off == 90
        assert write_win.width == 10
        assert local_win.width == 10

    def test_overlap_top(self):
        win = windows.Window(col_off=10, row_off=90, width=20, height=20)
        write_win, local_win = recrop_write_window(win, 100, 100)
        assert write_win.height == 10
        assert local_win.height == 10

    def test_local_and_write_extents_always_agree(self):
        # the invariant the function asserts internally: the cropped source
        # extent equals the clamped destination extent
        win = windows.Window(col_off=-5, row_off=-5, width=200, height=200)
        write_win, local_win = recrop_write_window(win, 100, 100)
        assert write_win.width == local_win.width
        assert write_win.height == local_win.height
        # a window larger than the bounds is clipped to the bounds
        assert write_win.width == 100
        assert write_win.height == 100


class TestBoundsFromTransform:
    def test_matches_manual_computation(self):
        tf = transform.from_bounds(west=1000, south=2000, east=1500,
                                   north=2300, width=50, height=30)
        left, bottom, right, top = bounds_from_transform_height_width_res(
            tf, height=30, width=50, resolution=10)
        assert (left, bottom, right, top) == pytest.approx((1000, 2000, 1500,
                                                            2300))


class TestCalculateAlignedTransform:
    def test_transform_is_aligned_to_resolution_grid(self):
        crs = CRS.from_epsg(32633)
        tres = 10
        tf, h, w = calculate_aligned_transform(
            src_crs=crs, dst_crs=crs, height=1000, width=1000,
            left=300123, bottom=5000456, right=310123, top=5010456, tres=tres)
        # both origin coordinates snap onto the resolution grid
        assert tf.c % tres == 0
        assert tf.f % tres == 0
        # one extra pixel is added on each axis (rounding-down compensation)
        assert isinstance(h, int) and isinstance(w, int)
        assert h > 0 and w > 0
        # pixel size is preserved
        assert tf.a == pytest.approx(tres)
        assert tf.e == pytest.approx(-tres)

    def test_raises_when_default_transform_degenerate(self, monkeypatch):
        import sentle.reproject_util as ru

        def fake_default(*args, **kwargs):
            return Affine.identity(), None, None

        monkeypatch.setattr(ru.warp, "calculate_default_transform",
                            fake_default)
        with pytest.raises(ValueError):
            calculate_aligned_transform(
                src_crs=CRS.from_epsg(32633), dst_crs=CRS.from_epsg(4326),
                height=10, width=10, left=0, bottom=0, right=10, top=10,
                tres=10)
