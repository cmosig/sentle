"""End-to-end regression tests: current code vs. the previous release.

These pin sentle's *numerical output* against reference cubes produced by the
last PyPI release (2026.6.4, git tag ``2026.06.04``) for a few representative
scenarios:

* **compositing** -- a weekly temporal composite (mean) of a small area.
* **utm_overlap** -- a small area straddling three overlapping MGRS tiles, so
  the tile-overlap de-duplication and multi-tile stitching are exercised.
* **reprojection** -- the same kind of area requested in EPSG:3857, so the
  reprojection from the native UTM data to a different CRS is exercised.
* **nbar** -- a small area with NBAR (sen2nbar) enabled. Its reference is a
  *current-code* golden rather than a previous-release one: NBAR crashed in the
  release (``KeyError: 'proj:epsg'``, see #53/#59) so the release could not
  produce a baseline. This scenario therefore pins NBAR output going forward.

Each reference cube (``tests/data/regression/<scenario>.npz``) holds the
Sentinel-2 reflectance array (the *previous release* produced it, except for
``nbar`` -- see above). The test runs the *current* code for the identical
request and asserts the reflectances match.
Because the resampling is nearest-neighbour and the Planetary Computer L2A data
is static per scene, the two are expected to agree to within float tolerance --
so this catches any accidental change in the download / tiling / reprojection /
compositing math (e.g. from a refactor).

Opt-in (needs network): set ``SENTLE_RUN_E2E=1``.

Regenerating the references (after an *intentional* output change): check out
the release tag and re-run the generator used to create them::

    git checkout 2026.06.04
    # run each scenario below through sentle.process and np.savez_compressed
    # data=<sentle values>, bands=<band names> into tests/data/regression/
    git checkout main
"""

import os
import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.e2e

if not os.environ.get("SENTLE_RUN_E2E"):
    pytest.skip("set SENTLE_RUN_E2E=1 to run end-to-end network tests",
                allow_module_level=True)

import xarray as xr
from rasterio.enums import Resampling

from sentle.sentle import process

REF_DIR = pathlib.Path(__file__).parent / "data" / "regression"

# NOTE: these must stay identical to the parameters the committed reference
# cubes were generated with (see the module docstring).
SCENARIOS = {
    "compositing": dict(
        target_crs="EPSG:32632", target_resolution=10,
        bound_left=654000, bound_bottom=5095000,
        bound_right=654500, bound_top=5095500,
        datetime="2023-06-01/2023-06-21", time_composite_freq="7D"),
    "utm_overlap": dict(
        target_crs="EPSG:32632", target_resolution=10,
        bound_left=699600, bound_bottom=5100000,
        bound_right=700100, bound_top=5100500,
        datetime="2023-06-05/2023-06-15"),
    "reprojection": dict(
        target_crs="EPSG:3857", target_resolution=10,
        bound_left=1224500, bound_bottom=5780300,
        bound_right=1225000, bound_top=5780800,
        datetime="2023-06-05/2023-06-15"),
    # NBAR reference is a current-code golden (release crashed on NBAR)
    "nbar": dict(
        target_crs="EPSG:32632", target_resolution=10,
        bound_left=654000, bound_bottom=5095000,
        bound_right=654500, bound_top=5095500,
        datetime="2023-06-05/2023-06-15", S2_nbar=True),
}


def _run_current(tmp_path, name):
    store = str(tmp_path / f"{name}.zarr")
    process(zarr_store=store, S1_assets=None, S2_cloud_classification=False,
            S2_mask_snow=False, num_workers=1,
            resampling_method=Resampling.nearest, **SCENARIOS[name])
    ds = xr.open_zarr(store)
    return ds["sentle"].values.astype(np.float32), [str(b) for b in
                                                    ds["band"].values]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_reflectances_match_previous_release(tmp_path, name):
    ref = np.load(REF_DIR / f"{name}.npz")
    ref_data = ref["data"].astype(np.float32)
    ref_bands = [str(b) for b in ref["bands"]]

    cur_data, cur_bands = _run_current(tmp_path, name)

    # same cube geometry and bands
    assert cur_bands == ref_bands, f"{name}: band mismatch"
    assert cur_data.shape == ref_data.shape, (
        f"{name}: shape {cur_data.shape} != reference {ref_data.shape}")

    # NoData (NaN) footprint must be identical
    assert np.array_equal(np.isnan(cur_data), np.isnan(ref_data)), (
        f"{name}: NaN/NoData mask changed vs the previous release")

    # reflectances must match where valid (nearest resampling -> ~exact)
    finite = np.isfinite(ref_data)
    np.testing.assert_allclose(
        cur_data[finite], ref_data[finite], rtol=0, atol=1e-2,
        err_msg=f"{name}: reflectances differ from the previous release")
