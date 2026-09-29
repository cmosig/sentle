"""``read_ptile_windows``: one read per band of the whole ptile window instead
of one per subtile and band, to cut the HTTP requests a Sentinel-2 cube makes.

It must not change a single pixel. The subtile is now sliced out of a native
resolution window and repeated up to 10 m with ``np.repeat`` instead of being
resampled by GDAL's ``out_shape`` read, so the tests compare against exactly
that direct GDAL read, for every band resolution, and run the full ptile path
both ways.

Offline: the "tile" is three small-on-disk (highly compressible) GeoTIFFs.
"""

import numpy as np
import pandas as pd
import pystac
import pytest
import rasterio
from rasterio import windows
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from sentle import sentinel2
from sentle.const import S2_RAW_BAND_RESOLUTION, S2_RAW_BANDS, S2_subtile_size
from sentle.reproject_util import transform_height_width_from_bounds_res
from sentle.stac import PlanetaryComputerProvider

TILE = "32TPT"
LEFT, TOP = 620000, 5280000  # inside 32TPT (UTM 32N)


@pytest.fixture(scope="module")
def tile_files(tmp_path_factory):
    """One COG-like file per resolution, values that differ per pixel block so
    a misplaced slice cannot go unnoticed."""
    root = tmp_path_factory.mktemp("tile")
    paths = {}
    for res, n in ((10, 10980), (20, 5490), (60, 1830)):
        path = str(root / f"res{res}.tif")
        profile = dict(driver="GTiff", height=n, width=n, count=1,
                       dtype="uint16", crs=CRS.from_epsg(32632),
                       transform=from_origin(600000, 5300040, res, res),
                       tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate")
        rows = (np.arange(n, dtype=np.uint32) // 3)[:, None]
        cols = (np.arange(n, dtype=np.uint32) // 5)[None, :]
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(((rows * 7 + cols * 13) % 9000 + 1200).astype("uint16"),
                      1)
        paths[res] = path
    return paths


def _direct_read(path, read_window):
    with rasterio.open(path) as dr:
        return dr.read(indexes=1, window=read_window,
                       out_shape=(S2_subtile_size, S2_subtile_size),
                       out_dtype=np.float32)


@pytest.mark.parametrize("res", [10, 20, 60])
@pytest.mark.parametrize("col_off,row_off", [(0, 0), (732, 2196), (7320, 8784)])
def test_slice_equals_direct_gdal_read(tile_files, res, col_off, row_off):
    factor = res // 10
    # a 3x3 subtile ptile window, the subtile asked for sits inside it
    tile_window = windows.Window(732 * (col_off // 732 - (col_off > 0)),
                                 732 * (row_off // 732 - (row_off > 0)),
                                 732 * 3, 732 * 3)
    subtile = windows.Window(col_off, row_off, 732, 732)
    read_window = windows.Window(subtile.col_off // factor,
                                 subtile.row_off // factor,
                                 subtile.width // factor,
                                 subtile.height // factor)

    got, crs, tf = sentinel2.read_subtile_from_tile_window(
        tile_files[res], read_window, factor, tile_window, {})

    np.testing.assert_array_equal(got, _direct_read(tile_files[res],
                                                    read_window))
    assert got.dtype == np.float32
    with rasterio.open(tile_files[res]) as dr:
        assert (crs, tf) == (dr.crs, dr.transform)


def test_one_read_per_band_however_many_subtiles(tile_files, monkeypatch):
    reads = []
    real_open = rasterio.open

    class _Counting:

        def __init__(self, ds):
            self._ds = ds

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._ds.close()

        def read(self, *args, **kwargs):
            reads.append(kwargs.get("window"))
            return self._ds.read(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._ds, name)

    monkeypatch.setattr(sentinel2.rasterio, "open",
                        lambda path, *a, **k: _Counting(real_open(path)))

    tile_window = windows.Window(0, 0, 732 * 4, 732 * 4)
    cache = {}
    for col in range(4):
        for row in range(4):
            sentinel2.read_subtile_from_tile_window(
                tile_files[10],
                windows.Window(732 * col, 732 * row, 732, 732), 1,
                tile_window, cache)

    assert len(reads) == 1


class _Provider(PlanetaryComputerProvider):

    def prepare_href(self, href):
        return href

    def rasterio_env(self):
        return rasterio.Env()


def _dispatch(tile_files, s2grid, read_ptile_windows):
    size = 2000  # px at 10 m -> 3x3 subtiles
    right, bottom = LEFT + size * 10, TOP - size * 10
    ptile_transform, height, width = transform_height_width_from_bounds_res(
        LEFT, bottom, right, TOP, 10)
    subtiles = sentinel2.obtain_subtiles(CRS.from_epsg(32632), LEFT, bottom,
                                         right, TOP, s2grid.copy())
    assert set(subtiles["name"]) == {TILE}

    item = pystac.Item(
        id=f"S2B_MSIL2A_20230601T101559_N0509_R065_T{TILE}_20230601T140000",
        geometry=None, bbox=None,
        datetime=pd.Timestamp("2023-06-01T10:15:59Z").to_pydatetime(),
        properties={"s2:mgrs_tile": TILE, "s2:processing_baseline": "05.09"},
        collection="sentinel-2-l2a")
    for band, res in S2_RAW_BAND_RESOLUTION.items():
        item.add_asset(band, pystac.Asset(href=tile_files[res]))

    return sentinel2.process_ptile_S2_dispatcher(
        target_crs=CRS.from_epsg(32632), target_resolution=10,
        S2_cloud_classification_device="cpu", time_composite_freq=None,
        S2_apply_snow_mask=False, S2_apply_cloud_mask=False,
        S2_bands_to_save=list(S2_RAW_BANDS), ptile_height=height,
        ptile_width=width, ptile_transform=ptile_transform,
        item_list=[item], ts=item.datetime, bound_left=LEFT,
        bound_right=right, bound_bottom=bottom, bound_top=TOP,
        S2_mask_snow=False, S2_cloud_classification=False,
        S2_return_cloud_probabilities=False, S2_nbar=False,
        S2_subtiles=subtiles, cloud_request_queue=None,
        cloud_response_queue=None, resampling_method=Resampling.nearest,
        S2_bands=list(S2_RAW_BANDS), provider=_Provider(),
        read_ptile_windows=read_ptile_windows)


def test_ptile_is_identical_with_and_without_window_reads(tile_files, s2grid):
    plain = _dispatch(tile_files, s2grid, read_ptile_windows=False)
    windowed = _dispatch(tile_files, s2grid, read_ptile_windows=True)

    assert plain.shape == (12, 2000, 2000)
    assert not np.isnan(plain).all()
    np.testing.assert_array_equal(plain, windowed)


def test_union_windows_cover_every_subtile_of_a_tile():
    wins = [windows.Window(732 * c, 732 * r, 732, 732)
            for c in (2, 3) for r in (1, 2, 3)]
    subtiles = pd.DataFrame({"name": ["A"] * len(wins) + ["B"],
                             "intersecting_windows":
                             wins + [windows.Window(0, 0, 732, 732)]})
    union = sentinel2.union_tile_windows(subtiles)

    assert union["A"] == windows.Window(732 * 2, 732, 732 * 2, 732 * 3)
    assert union["B"] == windows.Window(0, 0, 732, 732)
    # 2 and 6 divide the union offsets, so 20 m / 60 m windows are whole pixels
    assert union["A"].col_off % 6 == 0 and union["A"].row_off % 6 == 0
    assert union["A"].width % 6 == 0 and union["A"].height % 6 == 0
