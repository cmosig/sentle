"""``process()`` must search the catalog once for the whole run, not once per
ptile: at least two Planetary Computer requests per ptile (open the catalog,
search, plus a page per 10 items) is what ran into the rate limit on larger
cubes.

Offline: ``search_items`` and ``Parallel`` are stand-ins; the dispatched jobs
are recorded instead of run.
"""

import pandas as pd
import pystac
import pytest

from sentle import sentle as sentle_mod
from sentle import stac


def _item(item_id, when, bbox, collection="sentinel-2-l2a"):
    west, south, east, north = bbox
    return pystac.Item(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north],
                             [west, north], [west, south]]],
        },
        bbox=list(bbox),
        datetime=pd.Timestamp(when, tz="UTC").to_pydatetime(),
        properties={},
        collection=collection,
    )


class _CollectingParallel:
    jobs = []

    def __init__(self, **kwargs):
        pass

    def __call__(self, jobs):
        _CollectingParallel.jobs = [job[2] for job in jobs]
        return []


def _run(monkeypatch, tmp_path, items, **overrides):
    searches = []

    def fake_search(provider, collections, datetime, bbox):
        searches.append(dict(collections=list(collections),
                             datetime=datetime, bbox=list(bbox)))
        return items

    monkeypatch.setattr(sentle_mod, "search_items", fake_search)
    monkeypatch.setattr(sentle_mod, "Parallel", _CollectingParallel)
    monkeypatch.setattr(sentle_mod.gpd, "read_file", lambda *a, **k: None)
    monkeypatch.setattr(sentle_mod, "obtain_subtiles", lambda **kw: None)

    kwargs = dict(
        target_crs="EPSG:32632",
        target_resolution=10,
        # 2 x 1 spatial chunks of 1000 px
        bound_left=600000,
        bound_bottom=5100000,
        bound_right=620000,
        bound_top=5110000,
        datetime="2023-06-01/2023-06-10",
        zarr_store=str(tmp_path / "cube.zarr"),
        S1_assets=None,
        processing_spatial_chunk_size=1000,
        num_workers=2,
    )
    kwargs.update(overrides)
    sentle_mod.process(**kwargs)
    return searches, _CollectingParallel.jobs


# west of / east of the middle of the requested area: UTM 32N x=600000..620000
# is ~10.29E..10.55E, the boundary between the two 1000 px chunks ~10.42E
WEST = (10.2, 45.9, 10.38, 46.2)
EAST = (10.46, 45.9, 10.7, 46.2)
BOTH = (10.2, 45.9, 10.7, 46.2)


def test_one_search_however_many_ptiles(monkeypatch, tmp_path):
    items = [_item(f"i{n}", f"2023-06-0{n}T10:00:00", BOTH) for n in (1, 2, 3)]
    searches, jobs = _run(monkeypatch, tmp_path, items)

    assert len(searches) == 1
    # 3 timestamps x 2 spatial chunks
    assert len(jobs) == 6


def test_ptiles_without_items_are_not_dispatched(monkeypatch, tmp_path):
    items = [
        _item("both", "2023-06-01T10:00:00", BOTH),
        _item("west", "2023-06-02T10:00:00", WEST),
    ]
    _, jobs = _run(monkeypatch, tmp_path, items)

    assert len(jobs) == 3
    for job in jobs:
        assert job["item_list"], "a job without items should never be sent"
    west_only = [j for j in jobs if j["item_list"][0].id == "west"]
    assert len(west_only) == 1
    assert west_only[0]["zarr_save_slice"]["x"].start == 0


def test_each_ptile_gets_only_its_own_items(monkeypatch, tmp_path):
    items = [
        _item("west", "2023-06-01T10:00:00", WEST),
        _item("east", "2023-06-01T10:00:00", EAST),
    ]
    _, jobs = _run(monkeypatch, tmp_path, items)

    by_x = {j["zarr_save_slice"]["x"].start: j["item_list"] for j in jobs}
    assert [i.id for i in by_x[0]] == ["west"]
    assert [i.id for i in by_x[1000]] == ["east"]


def test_time_index_skips_empty_timestamps_but_keeps_position(
        monkeypatch, tmp_path):
    # the zarr time position comes from the timestamp list, so a timestamp
    # that is empty in a chunk must not shift the ones after it
    items = [
        _item("a", "2023-06-01T10:00:00", BOTH),
        _item("b", "2023-06-02T10:00:00", EAST),
        _item("c", "2023-06-03T10:00:00", BOTH),
    ]
    _, jobs = _run(monkeypatch, tmp_path, items)

    west = sorted(j["zarr_save_slice"]["time"] for j in jobs
                  if j["zarr_save_slice"]["x"].start == 0)
    # newest first: c=0, b=1, a=2 -- b is absent from the west chunk
    assert west == [0, 2]


def test_composite_searches_once_over_the_whole_window_range(
        monkeypatch, tmp_path):
    items = [_item("a", "2023-06-04T10:00:00", BOTH)]
    searches, jobs = _run(monkeypatch, tmp_path, items,
                          time_composite_freq="7D")

    assert len(searches) == 1
    start, end = searches[0]["datetime"]
    # the first and last window reach half a period past their timestamps
    assert start < pd.Timestamp("2023-06-01", tz="UTC")
    assert end > pd.Timestamp("2023-06-10", tz="UTC")
    assert all(j["item_list"] for j in jobs)


def test_search_bbox_is_padded(monkeypatch, tmp_path):
    searches, _ = _run(monkeypatch, tmp_path,
                       [_item("a", "2023-06-01T10:00:00", BOTH)])
    area = sentle_mod.lonlat_bounds("EPSG:32632", 600000, 5100000, 620000,
                                    5110000)
    west, south, east, north = searches[0]["bbox"]
    assert west < area[0] and south < area[1]
    assert east > area[2] and north > area[3]


def test_429_is_retried():
    adapter = stac.get_stac_api_io().session.get_adapter("https://x.invalid")
    assert 429 in adapter.max_retries.status_forcelist


def _asset_item(*hrefs):
    item = _item("a", "2023-06-01T10:00:00", (10, 45, 11, 46))
    for n, href in enumerate(hrefs):
        item.add_asset(f"b{n}", pystac.Asset(href=href))
    return item


def test_sas_tokens_are_fetched_once_per_storage_container(monkeypatch):
    signed = []
    monkeypatch.setattr(stac, "refresh_sas_token",
                        lambda href: signed.append(href))
    item = _asset_item(
        "https://acct.blob.core.windows.net/cont1/a/B02.tif",
        "https://acct.blob.core.windows.net/cont1/b/B03.tif",
        "https://acct.blob.core.windows.net/cont2/c/B04.tif",
        "https://elsewhere.example/d/B05.tif",
    )
    stac.PlanetaryComputerProvider().prefetch_sas_tokens([item, item])

    assert len(signed) == 2


def test_failed_token_prefetch_is_not_an_error(monkeypatch):

    def boom(href):
        raise RuntimeError("token endpoint down")

    monkeypatch.setattr(stac, "refresh_sas_token", boom)
    item = _asset_item("https://acct.blob.core.windows.net/cont1/a/B02.tif")
    # workers still sign for themselves, so this must not abort the run
    stac.PlanetaryComputerProvider().prefetch_sas_tokens([item])


def test_gdal_read_options_skip_sidecar_probing_unless_user_set(monkeypatch):
    monkeypatch.delenv("GDAL_DISABLE_READDIR_ON_OPEN", raising=False)
    assert stac.gdal_read_options()["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
    assert "GDAL_HTTP_LOW_SPEED_LIMIT" in stac.gdal_read_options()

    monkeypatch.setenv("GDAL_DISABLE_READDIR_ON_OPEN", "FALSE")
    assert "GDAL_DISABLE_READDIR_ON_OPEN" not in stac.gdal_read_options()
