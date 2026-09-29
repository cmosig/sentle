"""``S2_skip_redundant_reads``: per acquisition, read each location of a
spatial chunk from one MGRS tile only, and nothing outside the chunk or outside
an item's data footprint.

Pure geometry on the bundled grid (offline). The end-to-end read path is
exercised by benchmarks/request_count.py.
"""

import numpy as np
import pytest
from rasterio import transform, warp, windows
from rasterio.crs import CRS
from shapely.geometry import box, mapping, shape

from sentle.sentinel2 import (needed_tile_windows, obtain_subtiles,
                              read_subtile_from_tile_window)

UTM32 = CRS.from_epsg(32632)
# 30 km chunk across the overlap of two MGRS tiles in one UTM zone
TWO_TILES = (UTM32, (596590, 4157140, 626590, 4187140))
# 30 km chunk well inside 32TPT
ONE_TILE = (UTM32, (620000, 5250000, 650000, 5280000))


def _area(crs, bounds):
    return shape(warp.transform_geom(crs, "EPSG:4326",
                                     mapping(box(*bounds).segmentize(1000))))


def _subtiles(s2grid, case, drop_redundant=False):
    crs, bounds = case
    return obtain_subtiles(crs, *bounds, s2grid.copy(),
                           drop_redundant=drop_redundant)


def _window_bounds(subtiles, name, win):
    tile = subtiles[subtiles["name"] == name].iloc[0]
    local = box(*windows.bounds(win, tile.tile_transform)).segmentize(100)
    return shape(warp.transform_geom(tile.tile_crs, "EPSG:4326",
                                     mapping(local)))


def test_single_tile_reads_only_the_chunk_plus_margin(s2grid):
    subtiles = _subtiles(s2grid, ONE_TILE)
    (name,) = subtiles["name"].unique()
    needed = needed_tile_windows(subtiles, {name: None}, _area(*ONE_TILE))

    win = needed[name]
    # 3000 px chunk + 36 px margin each side, rounded out to 6 px
    assert 3000 < win.width <= 3000 + 2 * 36 + 12
    assert 3000 < win.height <= 3000 + 2 * 36 + 12
    # the union of its subtiles is much bigger
    union = windows.union(*subtiles["intersecting_windows"])
    assert win.width * win.height < 0.8 * union.width * union.height


def test_every_window_is_aligned_to_the_60m_grid(s2grid):
    subtiles = _subtiles(s2grid, TWO_TILES)
    needed = needed_tile_windows(subtiles,
                                 dict.fromkeys(subtiles["name"]),
                                 _area(*TWO_TILES))
    for win in needed.values():
        assert win.col_off % 6 == win.row_off % 6 == 0
        assert win.width % 6 == win.height % 6 == 0


def test_overlap_is_read_from_one_tile(s2grid):
    subtiles = _subtiles(s2grid, TWO_TILES)
    first, second = subtiles["name"].unique()  # priority order
    needed = needed_tile_windows(subtiles, {first: None, second: None},
                                 _area(*TWO_TILES))

    a = _window_bounds(subtiles, first, needed[first])
    b = _window_bounds(subtiles, second, needed[second])
    area = _area(*TWO_TILES)
    # together they cover the chunk ...
    assert area.difference(a.union(b)).area == pytest.approx(0, abs=1e-9)
    # ... and overlap only in a strip about as wide as the margin (~360 m),
    # not in the ~9 km MGRS overlap
    overlap_km = a.intersection(b).intersection(area).area / area.area * 30
    assert overlap_km < 1.5


def test_missing_tile_is_filled_by_the_overlapping_one(s2grid):
    subtiles = _subtiles(s2grid, TWO_TILES)
    first, second = subtiles["name"].unique()
    area = _area(*TWO_TILES)

    both = needed_tile_windows(subtiles, {first: None, second: None}, area)
    alone = needed_tile_windows(subtiles, {second: None}, area)

    assert first not in alone
    covered = _window_bounds(subtiles, second, alone[second])
    # everything the second tile has pixels for in the chunk is read from it
    tile_pixels = _window_bounds(subtiles, second,
                                 windows.Window(0, 0, 10980, 10980))
    assert area.intersection(tile_pixels).difference(
        covered).area == pytest.approx(0, abs=1e-9)
    assert alone[second].width > both[second].width


def test_static_elimination_would_have_left_that_gap(s2grid):
    # the default (drop_redundant=True) path cannot fill in: the second
    # tile's subtiles under the first tile's claim are already gone
    kept = _subtiles(s2grid, TWO_TILES, drop_redundant=True)
    every = _subtiles(s2grid, TWO_TILES, drop_redundant=False)
    assert len(kept) < len(every)


def test_item_footprint_limits_the_read(s2grid):
    subtiles = _subtiles(s2grid, ONE_TILE)
    (name,) = subtiles["name"].unique()
    area = _area(*ONE_TILE)
    # the item has data only in the western third of the chunk (swath edge)
    west, south, east, north = area.bounds
    footprint = box(west - 1, south - 1, west + (east - west) / 3, north + 1)

    full = needed_tile_windows(subtiles, {name: None}, area)[name]
    part = needed_tile_windows(subtiles, {name: footprint}, area)[name]
    # (heights differ by a few px: a lon/lat box is not a UTM rectangle)
    assert abs(part.height - full.height) <= 12
    assert part.width < 0.5 * full.width


def test_tile_without_data_in_the_chunk_is_not_read(s2grid):
    subtiles = _subtiles(s2grid, ONE_TILE)
    (name,) = subtiles["name"].unique()
    far_away = box(100, 10, 101, 11)
    assert needed_tile_windows(subtiles, {name: far_away},
                               _area(*ONE_TILE)) == {}


def test_subtile_partly_outside_the_window_is_zero_filled(tmp_path):
    import rasterio
    path = str(tmp_path / "t.tif")
    data = (np.arange(1464 * 1464, dtype=np.uint32) % 5000 + 1).reshape(
        1464, 1464).astype("uint16")
    with rasterio.open(path, "w", driver="GTiff", height=1464, width=1464,
                       count=1, dtype="uint16", crs=UTM32,
                       transform=transform.from_origin(0, 14640, 10,
                                                       10)) as dst:
        dst.write(data, 1)

    window = windows.Window(600, 0, 864, 1464)
    got, _, _ = read_subtile_from_tile_window(
        path, windows.Window(0, 0, 732, 732), 1, window, {})

    assert (got[:, :600] == 0).all()
    np.testing.assert_array_equal(got[:, 600:], data[:732, 600:732])
