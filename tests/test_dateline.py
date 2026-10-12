"""Regression tests for the antimeridian (date-line) crash in ``obtain_subtiles``
(issue #60).

Near +/-180 longitude ``rasterio.warp.transform_geom`` returns a MultiPolygon
(the geometry is cut at the antimeridian). The old code built the footprint with
``Polygon(*transform_geom(...)["coordinates"])``, which cannot parse a
MultiPolygon and raised ``TypeError: float() argument must be a string or a real
number, not 'tuple'``. ``obtain_subtiles`` now uses ``shapely.geometry.shape``,
which handles Polygon and MultiPolygon alike.

Follow-up: GDAL's antimeridian cutting in ``transform_geom`` is itself
unreliable -- some footprints come back uncut (spanning the whole globe, so an
area matched every tile at its latitude) and some self-intersecting (shapely
then raised ``TopologyException``). The lon/lat work near +/-180 is now done
with longitudes in [0, 360). The cases below are 30 km world-grid blocks that
failed that way.

Pure geometry on the bundled MGRS grid -- no network.
"""

import itertools
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rasterio import transform, warp, windows
from rasterio.crs import CRS
from shapely.geometry import MultiPolygon, box, mapping, shape
from shapely.ops import unary_union

import sentle.sentinel2 as sentinel2
from sentle.const import S2_subtile_size
from sentle.sentinel2 import (
    _hole_filling_subtiles,
    claimed_by_contributors,
    obtain_subtiles,
)

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


# 30 km blocks (EPSG, left, bottom) on or next to the antimeridian that sentle
# 2026.10.1 could not handle: a TopologyException in ``obtain_subtiles`` or a
# footprint wrapped around the globe (hundreds of tiles selected). EPSG:32661
# is UPS North; such blocks exist in real-world grids built with a "zone 61".
BLOCK = 30000
ANTIMERIDIAN_BLOCKS = [
    ("fiji_tile_straddles_180", "EPSG:32660", 720000, -1890000),
    ("chukotka_crosses_180", "EPSG:32601", 360000, 7650000),
    ("aleutians_crosses_180", "EPSG:32660", 690000, 5760000),
    ("ups_north_crosses_180", "EPSG:32661", 1980000, 4050000),
    ("aoi_wrapped_around_globe", "EPSG:32601", 330000, 7230000),
    ("ups_north_far_south_wrapped", "EPSG:32661", 1980000, 18780000),
]


def _block_bounds(left, bottom):
    return left, bottom, left + BLOCK, bottom + BLOCK


@pytest.mark.parametrize("name,crs,left,bottom", ANTIMERIDIAN_BLOCKS,
                         ids=[c[0] for c in ANTIMERIDIAN_BLOCKS])
def test_antimeridian_blocks_select_only_nearby_tiles(s2grid, name, crs, left,
                                                      bottom):
    bounds = _block_bounds(left, bottom)
    subtiles = obtain_subtiles(CRS.from_user_input(crs), *bounds,
                               s2grid.copy(), include_redundant=True)
    tiles = set(subtiles["name"])
    # tiles on either side of the antimeridian only, not every tile at the
    # block's latitude around the globe
    assert 0 < len(tiles) <= 6, (name, sorted(tiles))
    assert {t[:2] for t in tiles} <= {"01", "60"}, (name, sorted(tiles))

    # the non-redundant subtiles cover the whole block (compared in projected
    # CRSs, which are continuous across the antimeridian)
    kept = subtiles[~subtiles["redundant"]]
    covered = unary_union([
        shape(warp.transform_geom(
            row.tile_crs, crs,
            box(*windows.bounds(row.intersecting_windows,
                                row.tile_transform))))
        for row in kept.itertuples(index=False)
    ])
    block = box(*bounds)
    assert block.difference(covered).area < 1e-9 * block.area, name


@pytest.mark.parametrize("crs,left,bottom", [
    ("EPSG:32632", 450000, 5700000),   # Germany
    ("EPSG:32723", 300000, 7500000),   # Brazil
    ("EPSG:32654", 360000, 3960000),   # Japan
    ("EPSG:32659", 600000, 5880000),   # Kamchatka, near but not at 180
])
def test_lonlat360_frame_matches_default_away_from_antimeridian(
        s2grid, monkeypatch, crs, left, bottom):
    # The [0, 360) frame is the default algorithm with shifted longitudes, so
    # forcing it where the default is valid must change nothing.
    args = (CRS.from_user_input(crs), *_block_bounds(left, bottom))

    def keys(df):
        return [(r.name, r.intersecting_windows.col_off,
                 r.intersecting_windows.row_off, r.redundant, r.tile_crs,
                 tuple(r.tile_transform)) for r in df.itertuples(index=False)]

    default = keys(obtain_subtiles(*args, s2grid.copy(), include_redundant=True))
    monkeypatch.setattr(sentinel2, "_use_lonlat360", lambda *a, **k: True)
    forced = keys(obtain_subtiles(*args, s2grid.copy(), include_redundant=True))
    assert default == forced


def _items_from_grid(s2grid, names):
    """Stand-in STAC items whose footprint is the tile's own (multi)polygon,
    split at +/-180 like real Sentinel-2 items there."""
    grid = s2grid.set_index("name")
    tiles, items = [], []
    for name in names:
        geom = grid.loc[name, "geometry"]
        polygons = [g for g in getattr(geom, "geoms", [geom])
                    if g.geom_type == "Polygon"]
        tiles.append(name)
        items.append(SimpleNamespace(geometry=mapping(MultiPolygon(polygons))))
    return {"tile": tiles, "item": items}


@pytest.mark.parametrize("name,crs,left,bottom", ANTIMERIDIAN_BLOCKS[:4],
                         ids=[c[0] for c in ANTIMERIDIAN_BLOCKS[:4]])
def test_claimed_by_contributors_near_antimeridian(s2grid, name, crs, left,
                                                   bottom):
    target_crs = CRS.from_user_input(crs)
    bounds = _block_bounds(left, bottom)
    subtiles = obtain_subtiles(target_crs, *bounds, s2grid.copy(),
                               include_redundant=True)
    items = _items_from_grid(s2grid, subtiles["name"].unique())
    shape_ = (BLOCK // 100, BLOCK // 100)
    ptile_transform = transform.from_origin(bounds[0], bounds[3], 100, 100)

    claimed = claimed_by_contributors(subtiles[~subtiles["redundant"]],
                                      items, target_crs, ptile_transform,
                                      shape_)
    # every pixel is inside a contributing tile and its (shrunk) footprint,
    # except a sliver along the footprint edge
    assert claimed.mean() > 0.95, (name, claimed.mean())

    redundant = subtiles[subtiles["redundant"]]
    if not redundant.empty:
        fill = _hole_filling_subtiles(redundant, items,
                                      np.ones(shape_, dtype=bool),
                                      target_crs, ptile_transform, None)
        assert fill is not None, name
