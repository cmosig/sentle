"""A scene whose data is gone from the provider's storage must not kill the run.

Planetary Computer's catalog can list an item whose blobs no longer exist: a
single band of one 2022 scene of T35HLD answered HTTP 404, ``retry_read``
turned that into a ``SentleReadError`` after three attempts, and an 11-year
cube could not be built -- by that run or by any retry of it.

With ``skip_missing_assets`` (default) such a scene is left out as a whole,
with a warning in the worker and one summary warning when the run ends. Only a
clear "does not exist" counts as missing: every other read failure keeps the
retry-then-abort behaviour of ``test_read_failure_aborts.py``.

Offline: the granules are sparse local GeoTIFFs, and ``rasterio.open`` is
wrapped to fail for the hrefs that are meant to be gone.
"""

import warnings

import numpy as np
import pandas as pd
import pystac
import pytest
import rasterio
import xarray as xr
from affine import Affine
from rasterio import transform, windows
from rasterio.crs import CRS
from rasterio.enums import Resampling

from sentle import sentinel1, sentinel2, stac
from sentle import sentle as sentle_mod
from sentle.const import S2_RAW_BAND_RESOLUTION
from sentle.reproject_util import transform_height_width_from_bounds_res
from sentle.stac import (
    PlanetaryComputerProvider,
    SentleMissingAssetError,
    SentleReadError,
    is_missing_asset_error,
    retry_read,
)
from sentle.utils import PtileResult, release_job_queues

UTM32 = CRS.from_epsg(32632)
# 2 km in the middle of MGRS tile 32TPS, far from any tile overlap
BOUNDS = (654000, 5144000, 656000, 5146000)
# one band per native resolution keeps the tests fast
BANDS = ["B02", "B05", "B01"]

# what GDAL reports, as seen against Planetary Computer and S3
HTTP_404 = "HTTP response code: 404"
S3_NO_SUCH_KEY = "ObjectNotFound: The specified key does not exist."

NEW, OLD, NEW_REPROCESSED_EARLIER = "new", "old", "new_v0"
SCENES = {
    # name: (id, datetime, pixel value)
    NEW: ("S2A_MSIL2A_20230611T101601_N0214_R065_T32TPS_20230611T180000",
          "2023-06-11T10:16:01Z", 3000),
    OLD: ("S2B_MSIL2A_20230606T101559_N0214_R065_T32TPS_20230606T180000",
          "2023-06-06T10:15:59Z", 5000),
    # same acquisition as NEW, processed earlier (sorts after it)
    NEW_REPROCESSED_EARLIER:
    ("S2A_MSIL2A_20230611T101601_N0214_R065_T32TPS_20230611T120000",
     "2023-06-11T10:16:01Z", 7000),
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
    """{(scene, band): path}: full-size but sparse granules of 32TPS that
    hold the scene's value in the subtiles covering BOUNDS."""
    root = tmp_path_factory.mktemp("granules")
    tile = subtiles.iloc[0]
    union = windows.union(*subtiles["intersecting_windows"])
    paths = {}
    for scene, (_, _, value) in SCENES.items():
        for band in BANDS:
            factor = S2_RAW_BAND_RESOLUTION[band] // 10
            n = 10980 // factor
            tf = Affine(10 * factor, 0, round(tile.tile_transform.c), 0,
                        -10 * factor, round(tile.tile_transform.f))
            path = str(root / f"{scene}_{band}.tif")
            profile = dict(driver="GTiff", height=n, width=n, count=1,
                           dtype="uint16", crs=tile.tile_crs, transform=tf,
                           tiled=True, blockxsize=512, blockysize=512,
                           compress="deflate", sparse_ok=True)
            win = windows.Window(union.col_off // factor,
                                 union.row_off // factor,
                                 union.width // factor,
                                 union.height // factor)
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(np.full((int(win.height), int(win.width)), value,
                                  dtype="uint16"), 1, window=win)
            paths[(scene, band)] = path
    return paths


def _item(granules, scene):
    item_id, when, _ = SCENES[scene]
    west, south, east, north = 9.5, 46.0, 11.5, 47.0
    item = pystac.Item(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north],
                             [west, north], [west, south]]],
        },
        bbox=[west, south, east, north],
        datetime=pd.Timestamp(when).to_pydatetime(),
        # baseline < 4.0 -> no harmonization, so the values stay exact
        properties={"s2:mgrs_tile": "32TPS",
                    "s2:processing_baseline": "02.14"},
        collection="sentinel-2-l2a")
    for band in BANDS:
        item.add_asset(band, pystac.Asset(href=granules[(scene, band)]))
    return item


def _break(monkeypatch, hrefs, message=HTTP_404):
    """Make ``rasterio.open`` fail with ``message`` for ``hrefs``; returns the
    hrefs it was asked to open, in order."""
    real_open = rasterio.open
    opened = []

    def fake_open(href, *args, **kwargs):
        opened.append(href)
        if href in hrefs:
            raise rasterio.errors.RasterioIOError(message)
        return real_open(href, *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", fake_open)
    monkeypatch.setattr(stac, "READ_RETRY_BACKOFF", 0.0)
    return opened


def _dispatch(items, subtiles, missing_items, time_composite_freq=None,
              read_retries=2):
    left, bottom, right, top = BOUNDS
    ptile_transform, height, width = transform_height_width_from_bounds_res(
        left, bottom, right, top, 10)
    return sentinel2.process_ptile_S2_dispatcher(
        target_crs=UTM32, target_resolution=10,
        S2_cloud_classification_device="cpu",
        time_composite_freq=time_composite_freq,
        S2_apply_snow_mask=False, S2_apply_cloud_mask=False,
        S2_bands_to_save=list(BANDS), ptile_height=height,
        ptile_width=width, ptile_transform=ptile_transform,
        item_list=items, ts=items[0].datetime, bound_left=left,
        bound_right=right, bound_bottom=bottom, bound_top=top,
        S2_mask_snow=False, S2_cloud_classification=False,
        S2_return_cloud_probabilities=False, S2_nbar=False,
        S2_subtiles=subtiles, cloud_request_queue=None,
        cloud_response_queue=None, resampling_method=Resampling.nearest,
        S2_bands=list(BANDS), provider=_Provider(),
        read_retries=read_retries, missing_items=missing_items)


# --------------------------------------------------------------------------- #
# what counts as missing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("message", [
    HTTP_404,
    S3_NO_SUCH_KEY,
    "'/vsicurl/https://host/B01.tif' not recognized: HTTP response code: 404",
])
def test_a_clear_does_not_exist_is_missing(message):
    assert is_missing_asset_error(rasterio.errors.RasterioIOError(message))


@pytest.mark.parametrize("message", [
    "HTTP response code: 403",  # expired SAS token
    "HTTP response code: 409",  # Planetary Computer, unsigned request
    "HTTP response code: 429",
    "HTTP response code: 500",
    "HTTP response code: 503",
    "HTTP response code: 4040",
    "Operation timed out after 30000 milliseconds with 0 bytes received",
    "Could not resolve host: sentinel2l2a01.blob.core.windows.net",
    "The specified bucket does not exist",
    # GDAL says this for anything it could not open, whatever the reason
    "'/vsis3/eodata/B01.jp2' does not exist in the file system, and is not "
    "recognized as a supported dataset name.",
    "Read failed. See previous exception for details.",
])
def test_everything_else_is_not_missing(message):
    assert not is_missing_asset_error(rasterio.errors.RasterioIOError(message))


def test_a_failed_read_is_missing_if_its_cause_is():
    # a read that fails after the open succeeded only says "Read failed"; the
    # HTTP status is on the GDAL error it was raised from
    def read():
        try:
            raise RuntimeError(HTTP_404)
        except RuntimeError as cause:
            raise rasterio.errors.RasterioIOError(
                "Read failed. See previous exception for details.") from cause

    with pytest.raises(rasterio.errors.RasterioIOError) as excinfo:
        read()
    assert is_missing_asset_error(excinfo.value)


def test_a_missing_asset_is_not_retried():
    calls = []

    def operation():
        calls.append(1)
        raise rasterio.errors.RasterioIOError(HTTP_404)

    with warnings.catch_warnings():
        # no stac_read_retry warning either
        warnings.simplefilter("error")
        with pytest.raises(SentleMissingAssetError,
                           match="asset=x does not exist"):
            retry_read(operation, "asset=x", retries=2)
    assert len(calls) == 1


def test_a_missing_asset_error_is_a_read_error():
    # so code that catches SentleReadError keeps catching it
    assert issubclass(SentleMissingAssetError, SentleReadError)


@pytest.mark.parametrize("message", ["HTTP response code: 403",
                                     "HTTP response code: 503"])
def test_a_transient_failure_is_still_retried_then_raised(monkeypatch,
                                                          message):
    monkeypatch.setattr(stac, "READ_RETRY_BACKOFF", 0.0)
    calls = []

    def operation():
        calls.append(1)
        raise rasterio.errors.RasterioIOError(message)

    with pytest.warns(UserWarning, match="stac_read_retry"):
        with pytest.raises(SentleReadError,
                           match="after 3 attempt") as excinfo:
            retry_read(operation, "asset=x", retries=2)
    assert len(calls) == 3
    assert not isinstance(excinfo.value, SentleMissingAssetError)


# --------------------------------------------------------------------------- #
# Sentinel-2
# --------------------------------------------------------------------------- #


def test_all_scenes_present_is_the_baseline(granules, subtiles):
    missing = []
    out = _dispatch([_item(granules, NEW), _item(granules, OLD)], subtiles,
                    missing, time_composite_freq="7D")

    assert missing == []
    # the mean of both scenes
    assert np.array_equal(out, np.full_like(out, 4000.0))


@pytest.mark.parametrize("gone_band", BANDS)
def test_a_missing_scene_is_left_out_of_the_composite(
        granules, subtiles, monkeypatch, gone_band):
    gone = granules[(NEW, gone_band)]
    _break(monkeypatch, {gone})
    missing = []

    with pytest.warns(UserWarning, match="missing_asset_skip") as record:
        out = _dispatch([_item(granules, NEW), _item(granules, OLD)],
                        subtiles, missing, time_composite_freq="7D")

    # the other scene alone, in every band: the bands of the missing scene
    # that could have been read are not used either
    assert np.array_equal(out, np.full_like(out, 5000.0))
    assert [m["item"] for m in missing] == [SCENES[NEW][0]]
    assert gone in missing[0]["reason"]
    message = str(record[0].message)
    assert SCENES[NEW][0] in message and gone in message


def test_a_missing_scene_on_its_own_yields_no_data(granules, subtiles,
                                                   monkeypatch):
    _break(monkeypatch, {granules[(NEW, "B01")]})
    missing = []

    with pytest.warns(UserWarning, match="missing_asset_skip"):
        out = _dispatch([_item(granules, NEW)], subtiles, missing)

    # what the dispatcher returns for an acquisition without any data
    assert out is None
    assert [m["item"] for m in missing] == [SCENES[NEW][0]]


def test_an_older_processing_of_the_scene_takes_its_place(
        granules, subtiles, monkeypatch):
    # an acquisition can be listed more than once (reprocessed); the newest is
    # used. If its data is gone, that is as if it were not listed.
    _break(monkeypatch, {granules[(NEW, "B05")]})
    missing = []

    with pytest.warns(UserWarning, match="missing_asset_skip"):
        out = _dispatch(stac.sort_items([
            _item(granules, NEW_REPROCESSED_EARLIER),
            _item(granules, NEW)
        ]), subtiles, missing)

    assert np.array_equal(out, np.full_like(out, 7000.0))
    assert [m["item"] for m in missing] == [SCENES[NEW][0]]


def test_a_missing_scene_raises_without_skipping(granules, subtiles,
                                                 monkeypatch):
    opened = _break(monkeypatch, {granules[(NEW, "B01")]})

    with pytest.raises(SentleMissingAssetError, match="B01"):
        _dispatch([_item(granules, NEW), _item(granules, OLD)], subtiles,
                  missing_items=None)

    assert opened.count(granules[(NEW, "B01")]) == 1


@pytest.mark.parametrize("message", ["HTTP response code: 403",
                                     "HTTP response code: 503"])
def test_a_transient_failure_still_aborts_when_skipping(
        granules, subtiles, monkeypatch, message):
    opened = _break(monkeypatch, {granules[(NEW, "B01")]}, message)
    missing = []

    with pytest.warns(UserWarning, match="stac_read_retry"):
        with pytest.raises(SentleReadError,
                           match="after 3 attempt") as excinfo:
            _dispatch([_item(granules, NEW), _item(granules, OLD)], subtiles,
                      missing)

    assert not isinstance(excinfo.value, SentleMissingAssetError)
    assert opened.count(granules[(NEW, "B01")]) == 3
    assert missing == []


# --------------------------------------------------------------------------- #
# Sentinel-1
# --------------------------------------------------------------------------- #


class _Asset:

    def __init__(self, href):
        self.href = href


class _S1Item:
    properties = {"sat:orbit_state": "ascending"}

    def __init__(self, item_id, vv, vh):
        self.id = item_id
        self.assets = {"vv": _Asset(vv), "vh": _Asset(vh)}


@pytest.fixture(scope="module")
def s1_items(tmp_path_factory):
    root = tmp_path_factory.mktemp("s1")
    items = []
    for name, value in (("first", 2.0), ("second", 6.0)):
        hrefs = []
        for asset in ("vv", "vh"):
            path = str(root / f"{name}_{asset}.tif")
            with rasterio.open(
                    path, "w", driver="GTiff", height=40, width=40, count=1,
                    dtype="float32", crs=UTM32, nodata=-32768,
                    transform=transform.from_origin(599900, 5100100, 10,
                                                    10)) as dst:
                dst.write(np.full((40, 40), value, dtype="float32"), 1)
            hrefs.append(path)
        items.append(_S1Item(f"S1A_IW_GRDH_1SDV_{name}", *hrefs))
    return items


def _run_s1(items, missing_items, monkeypatch):
    monkeypatch.setattr(sentinel1, "refresh_sas_token", lambda href: href)
    return sentinel1.process_ptile_S1(
        target_crs=UTM32, target_resolution=10, time_composite_freq="7D",
        bound_left=600000, bound_right=600100, bound_bottom=5099900,
        bound_top=5100000, ts=None, S1_assets=["vv_asc", "vh_asc"],
        ptile_height=10, ptile_width=10,
        ptile_transform=transform.from_origin(600000, 5100000, 10, 10),
        item_list=items, resampling_method=Resampling.nearest,
        read_retries=2, missing_items=missing_items)


def test_sentinel1_baseline(s1_items, monkeypatch):
    missing = []
    out = _run_s1(s1_items, missing, monkeypatch)

    assert missing == []
    assert np.array_equal(out, np.full_like(out, 4.0))


def test_sentinel1_item_with_a_missing_asset_is_left_out(s1_items,
                                                         monkeypatch):
    # vh is read after vv: the item's vv must not be used either
    first, _ = s1_items
    _break(monkeypatch, {first.assets["vh"].href})
    missing = []

    with pytest.warns(UserWarning, match="missing_asset_skip"):
        out = _run_s1(s1_items, missing, monkeypatch)

    assert np.array_equal(out, np.full_like(out, 6.0))
    assert [m["item"] for m in missing] == [first.id]


def test_sentinel1_missing_asset_raises_without_skipping(s1_items,
                                                         monkeypatch):
    _break(monkeypatch, {s1_items[0].assets["vh"].href})

    with pytest.raises(SentleMissingAssetError):
        _run_s1(s1_items, None, monkeypatch)


def test_sentinel1_transient_failure_still_aborts(s1_items, monkeypatch):
    _break(monkeypatch, {s1_items[0].assets["vh"].href},
           "HTTP response code: 503")
    missing = []

    with pytest.warns(UserWarning, match="stac_read_retry"):
        with pytest.raises(SentleReadError) as excinfo:
            _run_s1(s1_items, missing, monkeypatch)

    assert not isinstance(excinfo.value, SentleMissingAssetError)
    assert missing == []


# --------------------------------------------------------------------------- #
# process()
# --------------------------------------------------------------------------- #


def _process(monkeypatch, tmp_path, granules, **overrides):
    items = [_item(granules, NEW), _item(granules, OLD)]
    monkeypatch.setattr(sentle_mod, "search_items",
                        lambda provider, collections, datetime, bbox: items)
    monkeypatch.setattr(sentle_mod, "get_provider", lambda name: _Provider())

    store = str(tmp_path / "cube.zarr")
    kwargs = dict(
        target_crs=UTM32, target_resolution=10, bound_left=BOUNDS[0],
        bound_bottom=BOUNDS[1], bound_right=BOUNDS[2], bound_top=BOUNDS[3],
        datetime="2023-06-01/2023-06-15", zarr_store=store, S1_assets=None,
        S2_bands=list(BANDS), processing_spatial_chunk_size=1000,
        # in this process, so the stand-ins above apply
        num_workers=1)
    kwargs.update(overrides)
    sentle_mod.process(**kwargs)
    return xr.open_zarr(store)["sentle"]


def test_process_completes_and_reports_the_skipped_scene(
        granules, monkeypatch, tmp_path):
    gone = granules[(NEW, "B01")]
    opened = _break(monkeypatch, {gone})

    with pytest.warns(UserWarning) as record:
        cube = _process(monkeypatch, tmp_path, granules)

    # the run finished: the other scene is there, the missing one is NoData
    # in every band (also the ones whose files exist)
    assert sorted(cube.band.values) == sorted(BANDS)
    old = cube.sel(time="2023-06-06").values
    new = cube.sel(time="2023-06-11").values
    assert np.array_equal(old, np.full_like(old, 5000.0))
    assert np.isnan(new).all()
    # not retried
    assert opened.count(gone) == 1

    messages = [str(w.message) for w in record]
    # once where it happened, once as the summary of the run
    assert any(m.startswith("missing_asset_skip") for m in messages)
    summary = [m for m in messages if m.startswith("missing_assets_skipped")]
    assert len(summary) == 1
    assert "count=1" in summary[0]
    assert SCENES[NEW][0] in summary[0] and gone in summary[0]
    assert SCENES[OLD][0] not in summary[0]


def test_the_summary_reaches_the_caller_from_worker_processes(
        granules, monkeypatch, tmp_path):
    # with a pool the workers' own warnings stay in the workers; the skipped
    # scene has to travel back with the ptile's result
    _break(monkeypatch, {granules[(NEW, "B01")]})

    with pytest.warns(UserWarning, match="missing_assets_skipped count=1"):
        cube = _process(monkeypatch, tmp_path, granules, num_workers=2)

    assert np.isnan(cube.sel(time="2023-06-11").values).all()
    assert (cube.sel(time="2023-06-06").values == 5000.0).all()


def test_process_without_missing_scenes_reports_nothing(
        granules, monkeypatch, tmp_path):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        cube = _process(monkeypatch, tmp_path, granules)

    assert not [w for w in record if "missing_asset" in str(w.message)]
    new = cube.sel(time="2023-06-11").values
    assert np.array_equal(new, np.full_like(new, 3000.0))


def test_process_aborts_on_a_missing_scene_when_told_to(
        granules, monkeypatch, tmp_path):
    _break(monkeypatch, {granules[(NEW, "B01")]})

    with pytest.raises(SentleMissingAssetError, match="B01"):
        _process(monkeypatch, tmp_path, granules, skip_missing_assets=False)


def test_process_still_aborts_on_a_transient_failure(
        granules, monkeypatch, tmp_path):
    _break(monkeypatch, {granules[(NEW, "B01")]}, "HTTP response code: 503")

    with pytest.warns(UserWarning, match="stac_read_retry"):
        with pytest.raises(SentleReadError,
                           match="after 3 attempt") as excinfo:
            _process(monkeypatch, tmp_path, granules)

    assert not isinstance(excinfo.value, SentleMissingAssetError)


def test_report_lists_each_scene_once():
    # a scene is skipped in every ptile it touches
    entry = {"item": "A", "reason": "asset=a does not exist"}
    results = [PtileResult(None, (entry,)), PtileResult(None, (entry,)),
               PtileResult(None, ({"item": "B", "reason": "asset=b"},)),
               PtileResult(None)]

    with pytest.warns(UserWarning, match="count=2") as record:
        missing = sentle_mod.report_missing_items(results)

    assert missing == {"A": "asset=a does not exist", "B": "asset=b"}
    assert len(record) == 1


def test_queue_cleanup_reads_the_job_id_off_the_result(monkeypatch):
    from sentle import utils
    monkeypatch.setitem(utils.GLOBAL_QUEUES, 7, object())
    monkeypatch.setitem(utils.GLOBAL_QUEUES, 8, object())

    release_job_queues([PtileResult(7), 8, PtileResult(None)])

    assert 7 not in utils.GLOBAL_QUEUES and 8 not in utils.GLOBAL_QUEUES
