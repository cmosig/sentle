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


def test_median_composite_differs_from_mean(tmp_path_factory):
    """Issue #57: a median composite runs end-to-end and differs from mean."""
    common = dict(
        target_crs=TARGET_CRS, target_resolution=RES,
        bound_left=LEFT, bound_bottom=BOTTOM, bound_right=RIGHT, bound_top=TOP,
        datetime="2023-06-01/2023-06-21", time_composite_freq="7D",
        S1_assets=None, S2_cloud_classification=False, S2_mask_snow=False,
        num_workers=1, resampling_method=Resampling.nearest)

    def _run(store, method):
        process(zarr_store=store, time_composite_method=method, **common)
        return xr.open_zarr(store)["sentle"].values

    mean = _run(str(tmp_path_factory.mktemp("mean") / "m.zarr"), "mean")
    median = _run(str(tmp_path_factory.mktemp("median") / "d.zarr"), "median")

    assert mean.shape == median.shape
    # same NoData footprint, different values (median != mean for skewed data)
    assert np.array_equal(np.isnan(mean), np.isnan(median))
    both = np.isfinite(mean) & np.isfinite(median)
    assert both.sum() > 0
    assert not np.allclose(mean[both], median[both])
    # median reflectance stays in a sane range
    assert np.nanmin(median[both]) >= 0 and np.nanmax(median[both]) < 20000


def test_sentinel1_median_composite_differs_from_mean(tmp_path_factory):
    """Issue #57: the aggregation methods apply to Sentinel-1 too. A wide
    window (45 days) is used so several S1 acquisitions fall in one composite
    (S1 revisit is ~12 days), making median != mean."""
    common = dict(
        target_crs=TARGET_CRS, target_resolution=RES,
        bound_left=LEFT, bound_bottom=BOTTOM, bound_right=RIGHT, bound_top=TOP,
        datetime="2023-06-01/2023-07-16", time_composite_freq="45D",
        S2_bands=[], S1_assets=["vv_asc", "vh_asc"],
        num_workers=1, resampling_method=Resampling.nearest)

    def _run(store, method):
        process(zarr_store=store, time_composite_method=method, **common)
        return xr.open_zarr(store)["sentle"].values

    mean = _run(str(tmp_path_factory.mktemp("s1mean") / "m.zarr"), "mean")
    median = _run(str(tmp_path_factory.mktemp("s1med") / "d.zarr"), "median")

    assert mean.shape == median.shape
    assert np.array_equal(np.isnan(mean), np.isnan(median))
    both = np.isfinite(mean) & np.isfinite(median)
    assert both.sum() > 0
    assert not np.allclose(mean[both], median[both])
    # S1 RTC gamma0 backscatter stays non-negative
    assert np.nanmin(median[both]) >= 0


def test_nbar_runs_and_changes_reflectance(tmp_path_factory):
    """Issues #53/#59: NBAR (sen2nbar) must actually run against the current
    Planetary Computer catalog (it used to crash with KeyError: 'proj:epsg')
    and produce a plausibly BRDF-corrected cube."""
    from sentle.const import S2_NBAR_BANDS

    def _run(store, nbar):
        process(
            target_crs=TARGET_CRS, target_resolution=RES,
            bound_left=LEFT, bound_bottom=BOTTOM,
            bound_right=RIGHT, bound_top=TOP,
            datetime=DATETIME, zarr_store=store,
            S1_assets=None, S2_cloud_classification=False, S2_mask_snow=False,
            S2_nbar=nbar, num_workers=1,
            resampling_method=Resampling.nearest)
        return xr.open_zarr(store)

    base = _run(str(tmp_path_factory.mktemp("e2e_plain") / "p.zarr"), False)
    nbar = _run(str(tmp_path_factory.mktemp("e2e_nbar") / "n.zarr"), True)

    # same shape/bands; NBAR does not add or drop bands
    assert list(nbar["band"].values) == list(base["band"].values)

    # the NBAR-corrected bands differ from the raw ones (c-factor applied),
    # but only modestly (BRDF correction is a small multiplicative factor)
    b = base.sel(band=list(S2_NBAR_BANDS))["sentle"].values
    n = nbar.sel(band=list(S2_NBAR_BANDS))["sentle"].values
    both = np.isfinite(b) & np.isfinite(n) & (b > 0)
    assert both.sum() > 0
    ratio = n[both] / b[both]
    assert not np.allclose(ratio, 1.0)          # NBAR actually did something
    assert 0.8 < np.nanmedian(ratio) < 1.2      # ... but stayed sane


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


@pytest.mark.skipif(not os.environ.get("SENTLE_RUN_CDSE_E2E"),
                    reason="set SENTLE_RUN_CDSE_E2E=1 (needs CDSE S3 credentials, "
                           "e.g. AWS_PROFILE=cdse)")
def test_cdse_matches_planetary_computer(tmp_path_factory):
    """Issue #75: the CDSE provider must produce the same reflectances as
    Planetary Computer (both serve the same ESA L2A product)."""
    common = dict(
        target_crs=TARGET_CRS, target_resolution=RES,
        bound_left=LEFT, bound_bottom=BOTTOM, bound_right=RIGHT, bound_top=TOP,
        datetime="2023-06-13/2023-06-17", S1_assets=None,
        S2_cloud_classification=False, S2_mask_snow=False, num_workers=1,
        resampling_method=Resampling.nearest)

    def _run(store, prov):
        process(zarr_store=store, provider=prov, **common)
        return xr.open_zarr(store)

    pc = _run(str(tmp_path_factory.mktemp("pc") / "pc.zarr"),
              "planetary_computer")
    cdse = _run(str(tmp_path_factory.mktemp("cdse") / "cdse.zarr"), "cdse")

    assert list(cdse["band"].values) == list(pc["band"].values)
    a, b = pc["sentle"].values, cdse["sentle"].values
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 0
    # byte-identical: same ESA product, just COG (PC) vs JP2 (CDSE) containers
    assert np.array_equal(a[both], b[both])
