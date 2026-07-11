"""End-to-end smoke test: a real (tiny) download through the full ``process``.

This exercises the entire pipeline against live Planetary Computer data --
catalog search, subtile selection, download, harmonization, reprojection and
Zarr writing -- and checks that the resulting cube is well-formed and carries
plausible Sentinel-2 surface-reflectance values.

It is **opt-in**: it needs network access and takes a while, so it only runs
when the ``SENTLE_RUN_E2E`` environment variable is set (e.g.
``SENTLE_RUN_E2E=1 pytest -m e2e``). It is skipped by default so the offline
unit suite stays fast and deterministic.

The issue that motivated the test suite (#47) suggested comparing produced
cubes against committed sample cubes. Rather than commit large binary fixtures,
this test pins the structural + value contract of the cube, which is the part
that actually regresses when the pipeline breaks.
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.e2e

if not os.environ.get("SENTLE_RUN_E2E"):
    pytest.skip("set SENTLE_RUN_E2E=1 to run end-to-end network tests",
                allow_module_level=True)

import xarray as xr
from rasterio.enums import Resampling

from sentle.const import S2_RAW_BANDS
from sentle.sentle import process

# a tiny area (~1 km2) in the Italian Alps (UTM 32N / MGRS T32T*) with a short
# window; kept minimal to keep the download light. Integer UTM bounds on the
# resolution grid avoid floating-point divisibility issues.
TARGET_CRS = "EPSG:32632"
RES = 10  # integer resolution + integer bounds -> whole-pixel grid
LEFT, BOTTOM, RIGHT, TOP = 654000, 5095000, 655000, 5096000
DATETIME = "2023-06-01/2023-06-20"


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    store = str(tmp_path_factory.mktemp("e2e") / "cube.zarr")
    process(
        target_crs=TARGET_CRS,
        target_resolution=RES,
        bound_left=LEFT,
        bound_bottom=BOTTOM,
        bound_right=RIGHT,
        bound_top=TOP,
        datetime=DATETIME,
        zarr_store=store,
        S1_assets=None,               # Sentinel-2 only -> no S1 download
        S2_cloud_classification=False,  # skip the model/service
        S2_mask_snow=False,
        num_workers=1,
        resampling_method=Resampling.nearest,
    )
    return xr.open_zarr(store)


def test_cube_has_expected_structure(cube):
    assert set(cube["sentle"].dims) == {"time", "band", "y", "x"}
    assert list(cube["band"].values) == S2_RAW_BANDS
    # spatial extent matches the request (100 x 100 pixels at ~11 m)
    assert cube.sizes["x"] == round((RIGHT - LEFT) / RES)
    assert cube.sizes["y"] == round((TOP - BOTTOM) / RES)
    assert cube.sizes["time"] >= 1


def test_cube_is_georeferenced(cube):
    assert "crs_wkt" in cube.attrs
    # coordinates fall within the requested bounds
    assert cube["x"].min() >= LEFT - RES
    assert cube["x"].max() <= RIGHT + RES
    assert cube["y"].min() >= BOTTOM - RES
    assert cube["y"].max() <= TOP + RES


def test_cube_contains_plausible_reflectance(cube):
    data = cube["sentle"].values
    finite = data[np.isfinite(data)]
    # there must be *some* real data
    assert finite.size > 0
    # harmonized Sentinel-2 surface reflectance is non-negative and well within
    # the 16-bit range (leaving headroom for bright targets)
    assert finite.min() >= 0
    assert np.nanmedian(finite) > 0
    assert finite.max() < 20000


def test_small_subtile_size_with_cloud_classification(tmp_path_factory):
    """Issue #30: a smaller subtile size (244) must work end-to-end, including
    the cloud model whose input is padded to a multiple of 32."""
    store = str(tmp_path_factory.mktemp("e2e_sub") / "sub.zarr")
    process(
        target_crs=TARGET_CRS,
        target_resolution=RES,
        bound_left=LEFT,
        bound_bottom=BOTTOM,
        bound_right=RIGHT,
        bound_top=TOP,
        datetime=DATETIME,
        zarr_store=store,
        S1_assets=None,
        S2_subtile_size=244,
        S2_cloud_classification=True,
        S2_cloud_classification_device="cpu",
        num_workers=1,
        resampling_method=Resampling.nearest,
    )
    ds = xr.open_zarr(store)
    assert "S2_cloud_classification" in list(ds["band"].values)
    cc = ds.sel(band="S2_cloud_classification")["sentle"].values
    classes = np.unique(cc[np.isfinite(cc)])
    # cloud classes are a subset of {clear, thick, thin, shadow}
    assert set(classes).issubset({0.0, 1.0, 2.0, 3.0})
    # reflectance still looks sane
    b02 = ds.sel(band="B02")["sentle"].values
    b02 = b02[np.isfinite(b02)]
    assert b02.size > 0 and b02.min() >= 0
