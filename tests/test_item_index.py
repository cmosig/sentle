"""``ItemIndex`` must answer a ptile's item query exactly as the catalog search
it replaced would have: one search per run instead of one (or more) per ptile.

Offline: items are built in memory.
"""

import datetime as dt

import pystac
import pytest

from sentle.stac import ItemIndex, item_time_range, lonlat_bbox, pad_lonlat_bbox

UTC = dt.timezone.utc


def _at(day, hour=10):
    return dt.datetime(2023, 6, day, hour, tzinfo=UTC)


def _item(item_id, when, bbox=(8, 45, 10, 47), collection="sentinel-2-l2a",
          start=None, end=None):
    west, south, east, north = bbox
    properties = {}
    if start is not None:
        properties = {
            "start_datetime": start.isoformat(),
            "end_datetime": end.isoformat(),
        }
    return pystac.Item(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": [[[west, south], [east, south], [east, north],
                             [west, north], [west, south]]],
        },
        bbox=list(bbox),
        datetime=when,
        properties=properties,
        collection=collection,
    )


AREA = lonlat_bbox([9.0, 46.0, 9.1, 46.1])


def _ids(index, collection, start, end, area=AREA):
    return [i.id for i in index.query(collection, start, end, area)]


def test_instant_query_matches_only_that_timestamp():
    index = ItemIndex([_item("a", _at(1)), _item("b", _at(2))])
    assert _ids(index, "sentinel-2-l2a", _at(2), _at(2)) == ["b"]


def test_range_query_is_inclusive_on_both_ends():
    index = ItemIndex([_item("a", _at(1)), _item("b", _at(2)),
                       _item("c", _at(3))])
    assert _ids(index, "sentinel-2-l2a", _at(1), _at(3)) == ["c", "b", "a"]
    assert _ids(index, "sentinel-2-l2a", _at(2), _at(3)) == ["c", "b"]


def test_collections_are_kept_apart():
    index = ItemIndex([
        _item("s2", _at(1)),
        _item("s1", _at(1), collection="sentinel-1-rtc"),
    ])
    assert _ids(index, "sentinel-1-rtc", _at(1), _at(1)) == ["s1"]
    assert _ids(index, "sentinel-2-l2a", _at(1), _at(1)) == ["s2"]
    assert _ids(index, "landsat", _at(1), _at(1)) == []


def test_items_outside_the_area_are_dropped():
    index = ItemIndex([
        _item("near", _at(1)),
        _item("far", _at(1), bbox=(20, 45, 21, 47)),
    ])
    assert _ids(index, "sentinel-2-l2a", _at(1), _at(1)) == ["near"]


def test_item_with_a_time_range_overlaps_rather_than_contains():
    # Sentinel-1 RTC items carry start/end_datetime; a search matches those
    # that overlap the queried range, not just ones starting inside it
    index = ItemIndex([
        _item("rtc", _at(2), collection="sentinel-1-rtc",
              start=_at(1, 23), end=_at(2, 1)),
    ])
    assert _ids(index, "sentinel-1-rtc", _at(2, 0), _at(3)) == ["rtc"]
    assert _ids(index, "sentinel-1-rtc", _at(1), _at(1, 22)) == []
    assert _ids(index, "sentinel-1-rtc", _at(2), _at(4)) == []


def test_items_are_returned_newest_first():
    index = ItemIndex([_item("a", _at(1)), _item("c", _at(3)),
                       _item("b", _at(2))])
    assert _ids(index, "sentinel-2-l2a", _at(1), _at(3)) == ["c", "b", "a"]


def test_item_without_geometry_never_matches():
    item = _item("a", _at(1))
    item.geometry = None
    assert _ids(ItemIndex([item]), "sentinel-2-l2a", _at(1), _at(1)) == []


def test_empty_index():
    assert _ids(ItemIndex([]), "sentinel-2-l2a", _at(1), _at(2)) == []


def test_item_time_range_falls_back_to_datetime():
    assert item_time_range(_item("a", _at(1))) == (_at(1), _at(1))


def test_antimeridian_bbox_is_split():
    area = lonlat_bbox([179.9, 10, -179.9, 11])
    assert area.intersects(lonlat_bbox([179.95, 10.2, 179.99, 10.3]))
    assert area.intersects(lonlat_bbox([-179.99, 10.2, -179.95, 10.3]))
    assert not area.intersects(lonlat_bbox([0, 10.2, 1, 10.3]))


def test_padding_grows_every_side_and_stays_valid():
    west, south, east, north = pad_lonlat_bbox([9, 46, 10, 47], pad=0.5)
    assert (west, south, east, north) == (8.5, 45.5, 10.5, 47.5)
    assert pad_lonlat_bbox([-180, -90, 180, 90], pad=1) == [-180, -90, 180, 90]
    # crossing the antimeridian keeps crossing it
    west, _, east, _ = pad_lonlat_bbox([179.9, 10, -179.9, 11], pad=0.05)
    assert west > east
