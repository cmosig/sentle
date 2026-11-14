"""An acquisition published as two products of one tile must not lose a piece.

When an acquisition reaches the ground in two pieces, ESA publishes one L2A
product per piece for each tile on the cut: same sensing time, different
processing time, each covering only part of the tile (e.g. 32UPU on
2026-07-11, where the newer product covers a quarter of the tile and the
older one the rest). sentle used only the newest product of a tile, so the
older piece was missing from the cube.

Older products of a tile now fill the pixels that are still empty, where their
footprint reaches past the newer one. A reprocessing covers the same area, so
it is still never read.

Offline: the granules are sparse local GeoTIFFs of 32TPS.
"""

import numpy as np
import pandas as pd
import pystac
import pytest
import rasterio
from affine import Affine
from rasterio import features, warp, windows
from rasterio.crs import CRS
from rasterio.enums import Resampling
from shapely.geometry import box, mapping

from sentle import sentinel2
from sentle.const import S2_RAW_BAND_RESOLUTION
from sentle.reproject_util import transform_height_width_from_bounds_res
from sentle.stac import PlanetaryComputerProvider

UTM32 = CRS.from_epsg(32632)
# 2 km in the middle of MGRS tile 32TPS, far from any tile overlap
BOUNDS = (654000, 5144000, 656000, 5146000)
# the cut between the two pieces runs north-south through BOUNDS, on the
# 60 m grid of the tile so every band ends there
CUT = 655200
# one band per native resolution keeps the tests fast
BANDS = ["B02", "B05", "B01"]

WEST, EAST = (600000, CUT), (654600, 710000)
SCENES = {
    # name: (processing time, x range with data, pixel value)
    "newer_west": ("20230611T180000", WEST, 3000),
    "older_east": ("20230611T120000", EAST, 5000),
    # a reprocessing: same area as newer_west, processed earlier
    "older_west": ("20230611T130000", WEST, 7000),
    # another piece, processed even earlier, that covers nothing new
    "oldest_west": ("20230611T100000", WEST, 9000),
}


class _Provider(PlanetaryComputerProvider):

    def prepare_href(self, href):
        return href

    def rasterio_env(self):
        return rasterio.Env()


@pytest.fixture(scope="module")
def subtiles(s2grid):
    out = sentinel2.obtain_subtiles(UTM32, *BOUNDS, s2grid.copy(),
                                    include_redundant=True)
    assert list(out["name"].unique()) == ["32TPS"]
    return out


@pytest.fixture(scope="module")
def granules(tmp_path_factory, subtiles):
    """{(scene, band): path}: full-size but sparse granules of 32TPS that hold
    the scene's value where its x range and the subtiles covering BOUNDS
    meet."""
    root = tmp_path_factory.mktemp("granules")
    tile = subtiles.iloc[0]
    union = windows.union(*subtiles["intersecting_windows"])
    paths = {}
    for scene, (_, (x0, x1), value) in SCENES.items():
        for band in BANDS:
            factor = S2_RAW_BAND_RESOLUTION[band] // 10
            n = 10980 // factor
            res = 10 * factor
            left, top = round(tile.tile_transform.c), round(tile.tile_transform.f)
            tf = Affine(res, 0, left, 0, -res, top)
            path = str(root / f"{scene}_{band}.tif")
            profile = dict(driver="GTiff", height=n, width=n, count=1,
                           dtype="uint16", crs=tile.tile_crs, transform=tf,
                           tiled=True, blockxsize=512, blockysize=512,
                           compress="deflate", sparse_ok=True)
            row0 = union.row_off // factor
            col0 = max(union.col_off // factor, (x0 - left) // res)
            col1 = min((union.col_off + union.width) // factor,
                       (x1 - left) // res)
            win = windows.Window(col0, row0, col1 - col0,
                                 union.height // factor)
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(np.full((int(win.height), int(win.width)), value,
                                  dtype="uint16"), 1, window=win)
            paths[(scene, band)] = path
    return paths


def _item(granules, scene):
    processed, (x0, x1), _ = SCENES[scene]
    footprint = warp.transform_geom(
        UTM32, "EPSG:4326",
        mapping(box(x0, 5100000, x1, 5200000).segmentize(1000)))
    item = pystac.Item(
        id=f"S2A_MSIL2A_20230611T101601_N0214_R065_T32TPS_{processed}",
        geometry=footprint,
        bbox=list(features.bounds(footprint)),
        datetime=pd.Timestamp("2023-06-11T10:16:01Z").to_pydatetime(),
        # baseline < 4.0 -> no harmonization, so the values stay exact
        properties={"s2:mgrs_tile": "32TPS",
                    "s2:processing_baseline": "02.14"},
        collection="sentinel-2-l2a")
    for band in BANDS:
        item.add_asset(band, pystac.Asset(href=granules[(scene, band)]))
    return item


def _opened(monkeypatch):
    """The hrefs ``rasterio.open`` is asked to open, in order."""
    real_open = rasterio.open
    opened = []

    def fake_open(href, *args, **kwargs):
        opened.append(href)
        return real_open(href, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", fake_open)
    return opened


def _dispatch(items, subtiles):
    left, bottom, right, top = BOUNDS
    ptile_transform, height, width = transform_height_width_from_bounds_res(
        left, bottom, right, top, 10)
    return sentinel2.process_ptile_S2_dispatcher(
        target_crs=UTM32, target_resolution=10,
        S2_cloud_classification_device="cpu", time_composite_freq=None,
        S2_apply_snow_mask=False, S2_apply_cloud_mask=False,
        S2_bands_to_save=list(BANDS), ptile_height=height,
        ptile_width=width, ptile_transform=ptile_transform,
        item_list=items, ts=items[0].datetime, bound_left=left,
        bound_right=right, bound_bottom=bottom, bound_top=top,
        S2_mask_snow=False, S2_cloud_classification=False,
        S2_return_cloud_probabilities=False, S2_nbar=False,
        S2_subtiles=subtiles, cloud_request_queue=None,
        cloud_response_queue=None, resampling_method=Resampling.nearest,
        S2_bands=list(BANDS), provider=_Provider(), missing_items=[])


def _columns_west_of_cut():
    return (CUT - BOUNDS[0]) // 10


def test_the_older_piece_fills_the_rest_of_the_tile(granules, subtiles):
    out = _dispatch([_item(granules, "newer_west"),
                     _item(granules, "older_east")], subtiles)
    cut = _columns_west_of_cut()
    assert not np.isnan(out).any()
    # where both pieces have data the newer one is kept
    np.testing.assert_array_equal(out[:, :, :cut], 3000)
    np.testing.assert_array_equal(out[:, :, cut:], 5000)


def test_the_newer_piece_alone_leaves_the_rest_empty(granules, subtiles):
    out = _dispatch([_item(granules, "newer_west")], subtiles)
    cut = _columns_west_of_cut()
    np.testing.assert_array_equal(out[:, :, :cut], 3000)
    assert np.isnan(out[:, :, cut:]).all()


def test_a_reprocessing_is_never_read(granules, subtiles, monkeypatch):
    opened = _opened(monkeypatch)
    out = _dispatch([_item(granules, "newer_west"),
                     _item(granules, "older_west")], subtiles)
    cut = _columns_west_of_cut()
    np.testing.assert_array_equal(out[:, :, :cut], 3000)
    assert np.isnan(out[:, :, cut:]).all()
    assert not any("older_west" in href for href in opened)


def test_an_older_piece_is_found_past_a_reprocessing(granules, subtiles,
                                                     monkeypatch):
    opened = _opened(monkeypatch)
    out = _dispatch([
        _item(granules, "newer_west"),
        _item(granules, "older_west"),
        _item(granules, "older_east"),
        _item(granules, "oldest_west"),
    ], subtiles)
    cut = _columns_west_of_cut()
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(out[:, :, :cut], 3000)
    np.testing.assert_array_equal(out[:, :, cut:], 5000)
    assert not any("older_west" in href or "oldest_west" in href
                   for href in opened)
