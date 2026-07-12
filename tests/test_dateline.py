"""Regression tests for the antimeridian (date-line) crash in ``obtain_subtiles``
(issue #60).

Near +/-180 longitude ``rasterio.warp.transform_geom`` returns a MultiPolygon
(the geometry is cut at the antimeridian). The old code built the footprint with
``Polygon(*transform_geom(...)["coordinates"])``, which cannot parse a
MultiPolygon and raised ``TypeError: float() argument must be a string or a real
number, not 'tuple'``. ``obtain_subtiles`` now uses ``shapely.geometry.shape``,
which handles Polygon and MultiPolygon alike.

Pure geometry on the bundled MGRS grid -- no network.
"""

import itertools

import numpy as np
import pytest
from rasterio.crs import CRS

from sentle.const import S2_subtile_size
from sentle.sentinel2 import obtain_subtiles

# AOIs straddling / adjacent to the antimeridian:
#  - EPSG:32660 (UTM zone 60N) sits just west of +180
#  - EPSG:32601 (UTM zone  1N) sits just east of -180
DATELINE_CASES = [
    ("utm60N_near_+180", "EPSG:32660", (720000, 6000000, 740000, 6020000)),
    ("utm1N_near_-180", "EPSG:32601", (200000, 6000000, 220000, 6020000)),
]


@pytest.mark.parametrize("name,crs,bounds", DATELINE_CASES,
                         ids=[c[0] for c in DATELINE_CASES])
def test_obtain_subtiles_handles_antimeridian(s2grid, name, crs, bounds):
    # must not raise (this is the #60 regression) and must find real subtiles
    kept = obtain_subtiles(CRS.from_user_input(crs), *bounds, s2grid.copy())
    assert len(kept) > 0, f"{name}: no subtiles found near the dateline"

    # every kept tile is a real MGRS tile from the grid
    grid_names = set(s2grid["name"])
    assert set(kept["name"]).issubset(grid_names)

    # windows are valid and never run past the 10980-pixel tile edge
    for win in kept["intersecting_windows"]:
        assert win.width == S2_subtile_size and win.height == S2_subtile_size
        assert 0 <= win.col_off <= 10980 - S2_subtile_size
        assert 0 <= win.row_off <= 10980 - S2_subtile_size


def test_dateline_area_uses_tiles_on_both_sides(s2grid):
    # the UTM-1N AOI is covered by zone-60 tiles (names starting "60") on the
    # far side of the antimeridian too -- proving cross-dateline tiles are kept
    kept = obtain_subtiles(CRS.from_user_input("EPSG:32601"),
                           200000, 6000000, 220000, 6020000, s2grid.copy())
    prefixes = {name[:2] for name in kept["name"].unique()}
    assert "60" in prefixes or "01" in prefixes


def test_no_duplicate_subtiles_near_dateline(s2grid):
    kept = obtain_subtiles(CRS.from_user_input("EPSG:32660"),
                           720000, 6000000, 740000, 6020000, s2grid.copy())
    keys = [(r.name, r.intersecting_windows.col_off, r.intersecting_windows.row_off)
            for r in kept.itertuples(index=False)]
    assert len(keys) == len(set(keys))
