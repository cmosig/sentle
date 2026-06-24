"""Tests for the MGRS-tile-overlap redundancy elimination in ``obtain_subtiles``.

The Sentinel-2 products are distributed on the UTM/MGRS tiling grid, whose
tiles overlap: a single location on the ground is covered by up to six tiles
(Bauer-Marschallinger & Falkner, 2023, "Wasting petabytes: A survey of the
Sentinel-2 UTM tiling grid and its spatial overhead"). Naively downloading
every (tile, subtile-window) that intersects the requested area therefore
fetches the same ground location several times.

``obtain_subtiles`` removes that redundancy by keeping, for every location,
only the subtile of the highest-priority tile covering it. These tests assert
the two properties that make this safe:

1. **Coverage is preserved** -- the kept subtiles together still cover every
   location that any candidate subtile covered (no data is lost).
2. **What is skipped is genuinely redundant** -- every dropped subtile's
   ground footprint is already covered by the kept subtiles.

All tests are pure geometry (no network / no downloads).
"""

import itertools

import geopandas as gpd
import numpy as np
import pkg_resources
import pytest
from rasterio import transform, warp, windows
from rasterio.crs import CRS
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from sentle.const import S2_subtile_size
from sentle.sentinel2 import obtain_subtiles


@pytest.fixture(scope="module")
def s2grid():
    return gpd.read_file(
        pkg_resources.resource_filename(
            "sentle", "data/sentinel2_grid_stripped_with_epsg.gpkg"))


def _window_footprint(tile_crs, tile_transform, win, dst_crs):
    """Ground footprint of a subtile window, expressed in ``dst_crs``."""
    return Polygon(*warp.transform_geom(
        src_crs=tile_crs,
        dst_crs=dst_crs,
        geom=box(*windows.bounds(win, tile_transform)))["coordinates"])


def _all_candidate_footprints(s2grid, target_crs, left, bottom, right, top):
    """Replicates the pre-deduplication behaviour: footprints of *every*
    (tile, subtile-window) intersecting the requested bounds, keyed by
    (tile_name, col_off, row_off)."""
    target_crs = CRS.from_user_input(target_crs)
    transformed_bounds = Polygon(*warp.transform_geom(
        src_crs=target_crs,
        dst_crs=s2grid.crs,
        geom=box(left, bottom, right, top))["coordinates"])

    grid = s2grid[s2grid["geometry"].intersects(transformed_bounds)].copy()
    grid["fp_utm"] = grid[["geometry", "crs"]].apply(
        lambda s: Polygon(*warp.transform_geom(
            src_crs=s2grid.crs, dst_crs=s["crs"],
            geom=s["geometry"].geoms[0])["coordinates"]),
        axis=1)
    grid["tf"] = grid["fp_utm"].apply(
        lambda x: transform.from_bounds(*x.bounds, width=10980, height=10980))

    general_windows = [
        windows.Window(c, r, S2_subtile_size, S2_subtile_size)
        for c, r in itertools.product(np.arange(0, 10980, S2_subtile_size),
                                      np.arange(0, 10980, S2_subtile_size))
    ]

    candidates = {}
    for row in grid.itertuples(index=False):
        for win in general_windows:
            fp = _window_footprint(row.crs, row.tf, win, s2grid.crs)
            if transformed_bounds.intersects(fp):
                candidates[(row.name, win.col_off, win.row_off)] = fp
    return candidates, transformed_bounds


def _kept_footprints(s2grid, target_crs, left, bottom, right, top):
    """Footprints of the subtiles actually returned by ``obtain_subtiles``."""
    kept = obtain_subtiles(CRS.from_user_input(target_crs), left, bottom,
                           right, top, s2grid.copy())
    # rebuild per-tile transforms to turn the kept windows into footprints
    grid = s2grid[s2grid["name"].isin(kept["name"].unique())].copy()
    grid["fp_utm"] = grid[["geometry", "crs"]].apply(
        lambda s: Polygon(*warp.transform_geom(
            src_crs=s2grid.crs, dst_crs=s["crs"],
            geom=s["geometry"].geoms[0])["coordinates"]),
        axis=1)
    grid["tf"] = grid["fp_utm"].apply(
        lambda x: transform.from_bounds(*x.bounds, width=10980, height=10980))
    tf_by_name = dict(zip(grid["name"], grid["tf"]))
    crs_by_name = dict(zip(grid["name"], grid["crs"]))

    out = {}
    for row in kept.itertuples(index=False):
        win = row.intersecting_windows
        fp = _window_footprint(crs_by_name[row.name], tf_by_name[row.name],
                               win, s2grid.crs)
        out[(row.name, win.col_off, win.row_off)] = fp
    return kept, out


# (name, target_crs, bounds) -- areas chosen to span multiple MGRS tiles,
# including a same-UTM-zone case and a case straddling the UTM 34N/35N border.
MULTI_TILE_CASES = [
    ("single_zone_multi_tile", "EPSG:32635", (300000, 5000000, 360000, 5060000)),
    ("cross_utm_zone_border", "EPSG:32635", (270000, 4560000, 300000, 4590000)),
    ("large_single_zone", "EPSG:32632", (400000, 5200000, 500000, 5300000)),
    ("latlon_request", "EPSG:4326", (10.0, 45.0, 10.5, 45.5)),
]


@pytest.mark.parametrize("name,crs,bounds", MULTI_TILE_CASES,
                         ids=[c[0] for c in MULTI_TILE_CASES])
def test_skipped_subtiles_are_redundant(s2grid, name, crs, bounds):
    """Every subtile that is dropped must be fully covered by the kept
    subtiles -- i.e. the skipped download carried no unique ground data."""
    candidates, _ = _all_candidate_footprints(s2grid, crs, *bounds)
    kept_df, kept = _kept_footprints(s2grid, crs, *bounds)

    # the kept set must be a strict subset of the candidate set
    assert set(kept).issubset(set(candidates))

    skipped_keys = set(candidates) - set(kept)
    assert skipped_keys, f"{name}: expected some redundant subtiles to drop"

    kept_union = unary_union(list(kept.values()))
    for key in skipped_keys:
        fp = candidates[key]
        # area of this skipped footprint that is NOT covered by kept subtiles
        uncovered = fp.difference(kept_union).area
        frac = uncovered / fp.area
        assert frac < 1e-6, (
            f"{name}: skipped subtile {key} carries {frac:.4%} unique ground "
            f"area -- it is NOT redundant and dropping it loses data")


@pytest.mark.parametrize("name,crs,bounds", MULTI_TILE_CASES,
                         ids=[c[0] for c in MULTI_TILE_CASES])
def test_coverage_is_preserved(s2grid, name, crs, bounds):
    """The kept subtiles must cover every location that any candidate subtile
    covered -- the optimisation removes redundancy but never creates gaps."""
    candidates, aoi = _all_candidate_footprints(s2grid, crs, *bounds)
    _, kept = _kept_footprints(s2grid, crs, *bounds)

    candidate_union = unary_union(list(candidates.values()))
    kept_union = unary_union(list(kept.values()))

    # restrict to the requested area; anything any tile could provide there
    # must still be provided after deduplication
    providable = candidate_union.intersection(aoi)
    lost = providable.difference(kept_union).area
    assert lost / providable.area < 1e-6, (
        f"{name}: deduplication lost {lost / providable.area:.4%} of coverage")


@pytest.mark.parametrize("name,crs,bounds", MULTI_TILE_CASES,
                         ids=[c[0] for c in MULTI_TILE_CASES])
def test_overlap_redundancy_is_reduced(s2grid, name, crs, bounds):
    """Sanity check that deduplication actually removes redundant downloads
    for overlapping multi-tile areas."""
    candidates, _ = _all_candidate_footprints(s2grid, crs, *bounds)
    kept_df, _ = _kept_footprints(s2grid, crs, *bounds)
    assert len(kept_df) < len(candidates), (
        f"{name}: expected fewer downloads after deduplication "
        f"({len(kept_df)} vs {len(candidates)})")


def test_single_tile_request_keeps_everything(s2grid):
    """An area well inside a single MGRS tile has no overlap, so nothing may
    be dropped (no false-positive redundancy removal)."""
    crs, bounds = "EPSG:32632", (410000, 5210000, 420000, 5220000)
    candidates, _ = _all_candidate_footprints(s2grid, crs, *bounds)
    kept_df, _ = _kept_footprints(s2grid, crs, *bounds)
    assert len(kept_df) == len(candidates)


def test_no_duplicate_subtiles(s2grid):
    """``obtain_subtiles`` must never return the same (tile, window) twice."""
    crs, bounds = "EPSG:32635", (300000, 5000000, 360000, 5060000)
    kept = obtain_subtiles(CRS.from_user_input(crs), *bounds, s2grid.copy())
    keys = [(r.name, r.intersecting_windows.col_off,
             r.intersecting_windows.row_off)
            for r in kept.itertuples(index=False)]
    assert len(keys) == len(set(keys))
