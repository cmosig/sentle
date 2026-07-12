"""Tests for the configurable Sentinel-2 subtile size (issue #30).

The subtile size is the side length (in 10 m pixels) each Sentinel-2 tile is
split into for download. It defaults to 732 but can be made smaller (down to
366) so small cubes are cheaper to generate. Valid sizes are divisors of 10980
that are multiples of 6 (so the 20 m/60 m bands read on an integer grid). These
tests cover the two places the size matters:

* ``obtain_subtiles`` -- the geometry that enumerates download windows, and
* ``cloud_mask.compute_cloud_mask`` -- which must pad any subtile size up to a
  multiple of 32 for the cloudsen U-Net and crop back afterwards.
"""

import numpy as np
import pytest
from rasterio.crs import CRS

from sentle.sentinel2 import obtain_subtiles

# an area spanning parts of several MGRS tiles, in UTM 33N
CRS_STR = "EPSG:32633"
BOUNDS = (300000, 5000000, 360000, 5060000)


@pytest.mark.parametrize("size", [366, 732, 1098])
def test_obtain_subtiles_windows_have_requested_size(s2grid, size):
    kept = obtain_subtiles(CRS.from_user_input(CRS_STR), *BOUNDS, s2grid.copy(),
                           subtile_size=size)
    assert len(kept) > 0
    for win in kept["intersecting_windows"]:
        assert win.width == size
        assert win.height == size
        # windows never run past the 10980-pixel tile edge
        assert win.col_off + win.width <= 10980
        assert win.row_off + win.height <= 10980


def test_smaller_subtiles_produce_more_windows(s2grid):
    big = obtain_subtiles(CRS.from_user_input(CRS_STR), *BOUNDS, s2grid.copy(),
                          subtile_size=732)
    small = obtain_subtiles(CRS.from_user_input(CRS_STR), *BOUNDS,
                            s2grid.copy(), subtile_size=366)
    # finer tiling covers the same area with more (smaller) windows
    assert len(small) > len(big)


def test_default_size_matches_explicit_732(s2grid):
    default = obtain_subtiles(CRS.from_user_input(CRS_STR), *BOUNDS,
                              s2grid.copy())
    explicit = obtain_subtiles(CRS.from_user_input(CRS_STR), *BOUNDS,
                               s2grid.copy(), subtile_size=732)
    assert len(default) == len(explicit)


def test_non_divisor_size_asserts(s2grid):
    with pytest.raises(AssertionError):
        obtain_subtiles(CRS.from_user_input(CRS_STR), *BOUNDS, s2grid.copy(),
                        subtile_size=500)


# --------------------------------------------------------------------------- #
# Cloud-mask padding: real model, no network (model ships with the package).
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def cloud_model():
    from sentle.cloud_mask import load_cloudsen_model
    return load_cloudsen_model("cpu")


@pytest.mark.parametrize("size", [366, 732, 1098])
def test_compute_cloud_mask_handles_subtile_size(cloud_model, size):
    from sentle.cloud_mask import compute_cloud_mask
    rng = np.random.default_rng(0)
    arr = rng.uniform(0, 4000, size=(12, size, size)).astype(np.float32)
    out = compute_cloud_mask(arr, cloud_model, "cpu")
    # four class probabilities, cropped back to the original subtile extent
    assert out.shape == (4, size, size)
    # softmax over the four classes sums to one per pixel
    assert np.allclose(out.sum(axis=0), 1.0, atol=1e-4)
