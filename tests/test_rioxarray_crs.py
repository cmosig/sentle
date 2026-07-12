"""Regression tests for issue #58: ``ds.rio.crs`` must resolve.

sentle stored the CRS only as a root-group ``crs_wkt`` attribute, which
rioxarray does not consult, so ``xarray.open_zarr(store).rio.crs`` came back
``None`` and the cube was effectively un-georeferenced for the rioxarray
ecosystem (reprojection, clipping, writing GeoTIFFs).

The fix writes a CF grid-mapping: a scalar ``spatial_ref`` variable holding the
WKT plus ``grid_mapping``/``coordinates`` attributes on the data array, so a
plain ``xr.open_zarr(store)`` (as shown in the README) exposes the CRS.
"""

import pandas as pd
import pytest
import xarray as xr
from rasterio.crs import CRS

from sentle.sentle import setup_zarr_storage

RES = 10.0
LEFT, BOTTOM, RIGHT, TOP = 300000.0, 5000000.0, 300500.0, 5000300.0
HEIGHT, WIDTH = 30, 50
BANDS = ["B02", "B03", "B04"]


def _build(tmp_path, crs, **overrides):
    store_path = str(tmp_path / "cube.zarr")
    kwargs = dict(
        zarr_store=store_path,
        timestamp_list=[{"collection": "sentinel-2-l2a",
                         "ts": pd.Timestamp("2020-06-15", tz="UTC")}],
        height=HEIGHT, width=WIDTH,
        bound_left=LEFT, bound_right=RIGHT, bound_top=TOP, bound_bottom=BOTTOM,
        target_resolution=RES, processing_spatial_chunk_size=4000,
        zarr_store_chunk_size={"time": 10, "y": 250, "x": 250},
        S2_bands_to_save=BANDS, total_bands_to_save=BANDS,
        target_crs=crs, consolidate_metadata=True,
    )
    kwargs.update(overrides)
    setup_zarr_storage(**kwargs)
    return store_path


@pytest.mark.parametrize("epsg", [32633, 4326, 3857, 32720])
def test_rio_crs_resolves_on_plain_open_zarr(tmp_path, epsg):
    crs = CRS.from_epsg(epsg)
    store_path = _build(tmp_path, crs)

    # exactly the pattern from the README -- no decode_coords="all"
    ds = xr.open_zarr(store_path)

    assert ds.rio.crs is not None
    assert ds.rio.crs.to_epsg() == epsg
    # the CRS is also reachable from the data variable itself
    assert ds["sentle"].rio.crs.to_epsg() == epsg


def test_spatial_ref_is_a_coordinate_not_a_data_var(tmp_path):
    store_path = _build(tmp_path, CRS.from_epsg(32633))
    ds = xr.open_zarr(store_path)
    assert "spatial_ref" in ds.coords
    assert "spatial_ref" not in ds.data_vars


def test_legacy_root_crs_wkt_attr_is_preserved(tmp_path):
    # keep the original attribute for backwards compatibility
    store_path = _build(tmp_path, CRS.from_epsg(32633))
    ds = xr.open_zarr(store_path)
    assert ds.attrs["crs_wkt"] == CRS.from_epsg(32633).to_wkt()
