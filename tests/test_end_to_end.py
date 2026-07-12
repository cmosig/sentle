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


def test_wgs84_fractional_degree_resolution(tmp_path_factory):
    """Issue #4: EPSG:4326 with a fractional-degree resolution."""
    store = str(tmp_path_factory.mktemp("e2e_wgs84") / "wgs84.zarr")
    res = 0.001  # ~111 m
    left, bottom, right, top = 11.00, 46.00, 11.05, 46.05
    process(
        target_crs="EPSG:4326",
        target_resolution=res,
        bound_left=left,
        bound_bottom=bottom,
        bound_right=right,
        bound_top=top,
        datetime=DATETIME,
        zarr_store=store,
        S1_assets=None,
        S2_cloud_classification=False,
        S2_mask_snow=False,
        num_workers=1,
        resampling_method=Resampling.nearest,
    )
    ds = xr.open_zarr(store)
    assert ds.sizes["x"] == 50
    assert ds.sizes["y"] == 50
    # coordinates are in degrees, aligned to the requested bounds
    assert float(ds["x"][0]) == pytest.approx(left, abs=1e-4)
    assert float(ds["y"][0]) == pytest.approx(top, abs=1e-4)
    assert float(ds["x"].max()) < right
    finite = ds["sentle"].values[np.isfinite(ds["sentle"].values)]
    assert finite.size > 0 and finite.min() >= 0


def test_band_subset_downloads_only_requested_bands(tmp_path_factory):
    """Issue #7: with cloud detection off, request only an RGB band subset."""
    store = str(tmp_path_factory.mktemp("e2e_rgb") / "rgb.zarr")
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
        S2_bands=["B04", "B03", "B02"],   # requested in RGB order
        S2_cloud_classification=False,
        S2_mask_snow=False,
        num_workers=1,
        resampling_method=Resampling.nearest,
    )
    ds = xr.open_zarr(store)
    # output band order follows S2_RAW_BANDS, not the requested order
    assert list(ds["band"].values) == ["B02", "B03", "B04"]
    assert ds.sizes["band"] == 3
    finite = ds["sentle"].values[np.isfinite(ds["sentle"].values)]
    assert finite.size > 0 and finite.min() >= 0


def test_sentinel1_only_run(tmp_path_factory):
    """Issue #64: download a Sentinel-1-only cube via S2_bands=[]."""
    store = str(tmp_path_factory.mktemp("e2e_s1") / "s1.zarr")
    process(
        target_crs=TARGET_CRS,
        target_resolution=RES,
        bound_left=LEFT,
        bound_bottom=BOTTOM,
        bound_right=RIGHT,
        bound_top=TOP,
        datetime=DATETIME,
        zarr_store=store,
        S2_bands=[],                       # Sentinel-2 disabled
        S1_assets=["vv_asc", "vh_asc"],
        num_workers=1,
        resampling_method=Resampling.nearest,
    )
    ds = xr.open_zarr(store)
    assert list(ds["band"].values) == ["vv_asc", "vh_asc"]
    assert set(ds["sentle"].dims) == {"time", "band", "y", "x"}
    finite = ds["sentle"].values[np.isfinite(ds["sentle"].values)]
    # Sentinel-1 RTC gamma0 backscatter is a small non-negative linear value
    assert finite.size > 0
    assert finite.min() >= 0
