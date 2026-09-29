"""Sentinel-2 reads: each band of a tile is read once per ptile, cropped to the
ptile, and pixels without data are filled from redundant subtiles.

None of that may change a pixel that the previous algorithm -- one windowed
GDAL read per subtile, resampled to 10 m by ``out_shape``, over the
non-redundant subtiles only -- had data for. The tests here run the full ptile
path both ways on synthetic granules placed on the real MGRS grid (two tiles
of one UTM zone, and two tiles of neighbouring zones), for several resampling
methods and target grids, and compare bit for bit.

Offline: the granules are small-on-disk (highly compressible) GeoTIFFs.
"""

import numpy as np
import pandas as pd
import pystac
import pytest
import rasterio
from affine import Affine
from rasterio import windows
from rasterio.crs import CRS
from rasterio.enums import Resampling

from sentle import sentinel2
from sentle.const import S2_RAW_BAND_RESOLUTION, S2_subtile_size
from sentle.reproject_util import transform_height_width_from_bounds_res
from sentle.stac import PlanetaryComputerProvider

UTM32 = CRS.from_epsg(32632)
# 30 km chunk across the overlap of 32SPG and 32SNG, with redundant subtiles
SAME_ZONE = (UTM32, (596590, 4157140, 626590, 4187140))
# 4 km chunk across the UTM 32/33 boundary (32UQU and 33UUP)
TWO_ZONES = (UTM32, (746000, 5376000, 750000, 5380000))
# one band per native resolution keeps the tests fast
BANDS = ["B02", "B05", "B01"]


def _write(path, crs, tf, n, value, nodata_box=None):
    rows = (np.arange(n, dtype=np.uint32) // 3)[:, None]
    cols = (np.arange(n, dtype=np.uint32) // 5)[None, :]
    data = ((rows * 7 + cols * 13 + value) % 9000 + 1200).astype("uint16")
    if nodata_box is not None:
        r0, r1, c0, c1 = nodata_box
        data[r0:r1, c0:c1] = 0
    profile = dict(driver="GTiff", height=n, width=n, count=1,
                   dtype="uint16", crs=crs, transform=tf, tiled=True,
                   blockxsize=512, blockysize=512, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


@pytest.fixture(scope="module")
def granules(tmp_path_factory, s2grid):
    """{(tile, band): path} for every tile of both cases, plus a
    ``(tile, band, "nodata")`` variant of the first tile of SAME_ZONE. Values
    differ per
    tile, so averaging two tiles and taking one would not agree. The first
    tile of SAME_ZONE has NoData where the second one's subtiles are
    redundant (a swath edge)."""
    root = tmp_path_factory.mktemp("granules")
    paths = {}
    for case_no, (crs, bounds) in enumerate((SAME_ZONE, TWO_ZONES)):
        subtiles = sentinel2.obtain_subtiles(crs, *bounds, s2grid.copy(),
                                             include_redundant=True)
        for tile_no, (name, group) in enumerate(
                subtiles.groupby("name", sort=False)):
            tile = group.iloc[0]
            for band in BANDS:
                factor = S2_RAW_BAND_RESOLUTION[band] // 10
                n = 10980 // factor
                # real granules sit on whole metres, exactly 10 m apart
                tf = Affine(10 * factor, 0, round(tile.tile_transform.c), 0,
                            -10 * factor, round(tile.tile_transform.f))
                nodata = None
                if case_no == 0 and tile_no == 0:
                    # NoData (a swath edge) exactly where the other tile's
                    # subtiles are redundant: the previous reads drop those
                    # subtiles and leave a hole
                    red = subtiles[subtiles["redundant"]]
                    other = red.iloc[0]
                    x0, y0, x1, y1 = windows.bounds(
                        windows.union(*red["intersecting_windows"]),
                        other.tile_transform)
                    win = windows.from_bounds(
                        x0, y0, x1, y1, transform=tile.tile_transform)
                    win = win.round_offsets().round_lengths().intersection(
                        windows.Window(0, 0, 10980, 10980))
                    nodata = tuple(v // factor for v in (
                        win.row_off, win.row_off + win.height,
                        win.col_off, win.col_off + win.width))
                path = str(root / f"{name}_{band}.tif")
                _write(path, tile.tile_crs, tf, n, 3001 * tile_no)
                paths[(name, band)] = path
                if nodata is not None:
                    path = str(root / f"{name}_{band}_nodata.tif")
                    _write(path, tile.tile_crs, tf, n, 3001 * tile_no,
                           nodata)
                    paths[(name, band, "nodata")] = path
    return paths


class _Provider(PlanetaryComputerProvider):

    def prepare_href(self, href):
        return href

    def rasterio_env(self):
        return rasterio.Env()


def _items(granules, tiles, nodata=False):
    out = []
    for tile in tiles:
        item = pystac.Item(
            id=f"S2B_MSIL2A_20230601T101559_N0509_R065_T{tile}_20230601T1",
            geometry=None, bbox=None,
            datetime=pd.Timestamp("2023-06-01T10:15:59Z").to_pydatetime(),
            properties={"s2:mgrs_tile": tile,
                        "s2:processing_baseline": "05.09"},
            collection="sentinel-2-l2a")
        for band in BANDS:
            href = granules.get((tile, band, "nodata")) if nodata else None
            item.add_asset(band, pystac.Asset(
                href=href or granules[(tile, band)]))
        out.append(item)
    return out


def _direct_read(href, read_window, factor, tile_window, window_cache,
                 key=None):
    """The previous read: one GDAL read per subtile, upsampled by out_shape."""
    with rasterio.open(href) as dr:
        return dr.read(indexes=1, window=read_window,
                       out_shape=(S2_subtile_size, S2_subtile_size),
                       out_dtype=np.float32), dr.crs, dr.transform


def _run(case, items, s2grid, target_crs=None, res=10,
         resampling=Resampling.nearest, previous=False, monkeypatch=None):
    crs, bounds = case
    target_crs = target_crs or crs
    if target_crs != crs:
        bounds = rasterio.warp.transform_bounds(crs, target_crs, *bounds)
    # whole pixels of the target grid
    bounds = (bounds[0], bounds[1],
              bounds[0] + (bounds[2] - bounds[0]) // res * res,
              bounds[1] + (bounds[3] - bounds[1]) // res * res)
    left, bottom, right, top = bounds
    ptile_transform, height, width = transform_height_width_from_bounds_res(
        left, bottom, right, top, res)
    subtiles = sentinel2.obtain_subtiles(target_crs, *bounds, s2grid.copy(),
                                         include_redundant=not previous)
    if previous:
        monkeypatch.setattr(sentinel2, "read_subtile_from_tile_window",
                            _direct_read)
    try:
        return sentinel2.process_ptile_S2_dispatcher(
            target_crs=target_crs, target_resolution=res,
            S2_cloud_classification_device="cpu", time_composite_freq=None,
            S2_apply_snow_mask=False, S2_apply_cloud_mask=False,
            S2_bands_to_save=list(BANDS), ptile_height=height,
            ptile_width=width, ptile_transform=ptile_transform,
            item_list=items, ts=items[0].datetime, bound_left=left,
            bound_right=right, bound_bottom=bottom, bound_top=top,
            S2_mask_snow=False, S2_cloud_classification=False,
            S2_return_cloud_probabilities=False, S2_nbar=False,
            S2_subtiles=subtiles, cloud_request_queue=None,
            cloud_response_queue=None, resampling_method=resampling,
            S2_bands=list(BANDS), provider=_Provider())
    finally:
        if previous:
            monkeypatch.undo()


def _tiles(case, s2grid):
    return list(sentinel2.obtain_subtiles(
        case[0], *case[1], s2grid.copy())["name"].unique())


@pytest.mark.parametrize("case", [SAME_ZONE, TWO_ZONES],
                         ids=["same_zone", "two_zones"])
@pytest.mark.parametrize("target_crs,res,resampling", [
    (None, 10, Resampling.nearest),
    (None, 10, Resampling.bilinear),
    (None, 10, Resampling.cubic),
    (None, 30, Resampling.average),
    (None, 30, Resampling.lanczos),
    (CRS.from_epsg(3035), 20, Resampling.nearest),
    (CRS.from_epsg(3035), 20, Resampling.cubic),
    (CRS.from_epsg(4326), 0.0002, Resampling.bilinear),
], ids=["10m-nearest", "10m-bilinear", "10m-cubic", "30m-average",
        "30m-lanczos", "3035-nearest", "3035-cubic", "4326-bilinear"])
def test_identical_where_the_previous_reads_had_data(
        granules, s2grid, monkeypatch, case, target_crs, res, resampling):
    items = _items(granules, _tiles(case, s2grid))
    before = _run(case, items, s2grid, target_crs, res, resampling,
                  previous=True, monkeypatch=monkeypatch)
    now = _run(case, items, s2grid, target_crs, res, resampling)

    assert before.shape == now.shape
    had_data = ~np.isnan(before)
    assert had_data.mean() > 0.5
    np.testing.assert_array_equal(now[had_data], before[had_data])
    # nothing that had data lost it, and nothing new appeared except by
    # filling pixels the previous reads left empty
    assert not np.isnan(now[had_data]).any()


def test_a_missing_tile_is_filled_from_the_redundant_subtiles(
        granules, s2grid, monkeypatch):
    # only the lower-priority tile has an item: the previous reads dropped
    # its subtiles that the missing tile covers, and left a hole there
    tiles = _tiles(SAME_ZONE, s2grid)
    items = _items(granules, tiles[1:])
    before = _run(SAME_ZONE, items, s2grid, previous=True,
                  monkeypatch=monkeypatch)
    now = _run(SAME_ZONE, items, s2grid)

    had_data = ~np.isnan(before)
    np.testing.assert_array_equal(now[had_data], before[had_data])
    filled = np.isnan(before) & ~np.isnan(now)
    assert filled.sum() > 100_000
    # every hole the second tile covers is filled
    everything = _run(SAME_ZONE, items, s2grid.copy())
    assert np.isnan(now).sum() == np.isnan(everything).sum()


def test_nodata_in_the_priority_tile_is_filled_from_the_other(
        granules, s2grid, monkeypatch):
    items = _items(granules, _tiles(SAME_ZONE, s2grid), nodata=True)
    before = _run(SAME_ZONE, items, s2grid, previous=True,
                  monkeypatch=monkeypatch)
    now = _run(SAME_ZONE, items, s2grid)

    had_data = ~np.isnan(before)
    np.testing.assert_array_equal(now[had_data], before[had_data])
    assert (np.isnan(before) & ~np.isnan(now)).sum() > 100_000
    assert not np.isnan(now).any()


def test_redundant_subtiles_are_not_read_when_there_are_no_holes(
        granules, s2grid, monkeypatch):
    reads = []
    real = sentinel2.read_subtile_from_tile_window

    def counting(href, *args, **kwargs):
        reads.append(href)
        return real(href, *args, **kwargs)

    monkeypatch.setattr(sentinel2, "read_subtile_from_tile_window", counting)
    items = _items(granules, _tiles(SAME_ZONE, s2grid))
    _run(SAME_ZONE, items, s2grid)

    subtiles = sentinel2.obtain_subtiles(SAME_ZONE[0], *SAME_ZONE[1],
                                         s2grid.copy())
    # one call per non-redundant subtile and band, none for redundant ones
    assert len(reads) == len(subtiles) * len(BANDS)


def test_each_band_of_a_tile_is_opened_once_per_ptile(granules, s2grid,
                                                      monkeypatch):
    opened = []
    real_open = rasterio.open
    monkeypatch.setattr(sentinel2.rasterio, "open",
                        lambda path, *a, **k: opened.append(path) or
                        real_open(path, *a, **k))
    tiles = _tiles(SAME_ZONE, s2grid)
    _run(SAME_ZONE, _items(granules, tiles), s2grid)

    assert sorted(opened) == sorted(set(opened))
    assert len(opened) == len(tiles) * len(BANDS)


def test_window_is_the_chunk_plus_margin(s2grid):
    crs, bounds = SAME_ZONE
    subtiles = sentinel2.obtain_subtiles(crs, *bounds, s2grid.copy())
    cropped = sentinel2.tile_read_windows(subtiles, crs, bounds, 10,
                                          Resampling.nearest)
    whole = sentinel2.tile_read_windows(subtiles, crs, bounds, 10,
                                        Resampling.nearest, crop=False)
    for name, (win, _, union) in cropped.items():
        assert whole[name][0] == union
        # inside the union of the tile's subtiles, and aligned to the 60 m
        # grid so the 20 m and 60 m windows are whole pixels
        assert windows.intersect(win, union)
        assert win.intersection(union) == win
        for v in (win.col_off, win.row_off, win.width, win.height):
            assert v % 6 == 0
    # a 3000 px chunk needs at most 3000 px + margin per axis from any tile
    for win, _, _ in cropped.values():
        assert win.width <= 3000 + 2 * 12 and win.height <= 3000 + 2 * 12
    # and reads markedly less than the whole subtiles
    area = sum(w.width * w.height for w, _, _ in cropped.values())
    whole_area = sum(w.width * w.height for w, _, _ in whole.values())
    assert area < 0.8 * whole_area


@pytest.mark.parametrize("band", BANDS)
@pytest.mark.parametrize("col_off,row_off", [(0, 0), (732, 2196),
                                             (7320, 8784)])
def test_slice_equals_direct_gdal_read(granules, band, col_off, row_off):
    href = granules[("32SPG", band)]
    factor = S2_RAW_BAND_RESOLUTION[band] // 10
    read_window = windows.Window(col_off // factor, row_off // factor,
                                 S2_subtile_size // factor,
                                 S2_subtile_size // factor)
    tile_window = windows.Window(0, 0, 10980, 10980)
    got, _, _ = sentinel2.read_subtile_from_tile_window(
        href, read_window, factor, (tile_window, None, tile_window), {})
    want, _, _ = _direct_read(href, read_window, factor, None, None)
    np.testing.assert_array_equal(got, want)


def test_hole_mask_covers_the_swath_edge_fringe_but_not_lone_dark_pixels():
    count = np.ones((3, 100, 100), dtype=np.uint8)
    count[:, :, :40] = 0          # no data at all (outside the swath)
    count[2, :, 40:44] = 0        # one band ends a few pixels later
    count[1, 70, 90] = 0          # a dark pixel far away, one band at 0
    hole = sentinel2.hole_mask(count)

    assert hole[:, :44].all()
    assert not hole[:, 44:].any()


def test_hole_mask_is_empty_without_pixels_missing_every_band():
    count = np.ones((3, 50, 50), dtype=np.uint8)
    count[1, 10, 10] = 0
    assert not sentinel2.hole_mask(count).any()


def _nodata_lonlat(s2grid):
    """The NoData block of the first SAME_ZONE tile, as a lon/lat polygon."""
    from shapely.geometry import box, mapping, shape
    crs, bounds = SAME_ZONE
    subtiles = sentinel2.obtain_subtiles(crs, *bounds, s2grid.copy(),
                                         include_redundant=True)
    red = subtiles[subtiles["redundant"]]
    other = red.iloc[0]
    block = box(*windows.bounds(windows.union(*red["intersecting_windows"]),
                                other.tile_transform))
    return shape(rasterio.warp.transform_geom(other.tile_crs, "EPSG:4326",
                                              mapping(block)))


def _with_footprints(items, s2grid, cut=None):
    """Give each item its MGRS tile outline as footprint, the first one
    minus ``cut`` (lon/lat)."""
    from shapely.geometry import mapping
    grid = s2grid.set_index("name")
    for n, item in enumerate(items):
        outline = grid.loc[item.properties["s2:mgrs_tile"], "geometry"]
        if n == 0 and cut is not None:
            outline = outline.difference(cut)
        item.geometry = mapping(outline)
    return items


def test_a_gap_the_tile_claims_data_for_is_not_read_again(
        granules, s2grid, monkeypatch):
    # the NoData block lies deep inside the first tile's footprint: a gap in
    # the scene itself, which the overlapping tile shares -- not read
    reads = []
    real = sentinel2.read_tile_window
    monkeypatch.setattr(sentinel2, "read_tile_window",
                        lambda href, *a, **k: reads.append(href) or
                        real(href, *a, **k))
    tiles = _tiles(SAME_ZONE, s2grid)
    items = _with_footprints(_items(granules, tiles, nodata=True), s2grid)
    now = _run(SAME_ZONE, items, s2grid)

    assert len(reads) == len(tiles) * len(BANDS)
    assert np.isnan(now).any()


def test_a_gap_outside_the_footprint_is_filled(granules, s2grid,
                                               monkeypatch):
    # the same block outside the first tile's footprint: a swath edge
    tiles = _tiles(SAME_ZONE, s2grid)
    items = _with_footprints(
        _items(granules, tiles, nodata=True), s2grid,
        cut=_nodata_lonlat(s2grid).buffer(0.02))
    before = _run(SAME_ZONE, items, s2grid, previous=True,
                  monkeypatch=monkeypatch)
    now = _run(SAME_ZONE, items, s2grid)

    had_data = ~np.isnan(before)
    np.testing.assert_array_equal(now[had_data], before[had_data])
    assert (np.isnan(before) & ~np.isnan(now)).sum() > 100_000
