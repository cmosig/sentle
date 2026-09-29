"""The per-ptile STAC item order must be deterministic AND prefer the newest
reprocessing (issue #87).

The catalog promises no order, but the order is load-bearing twice over:
``process_ptile_S2`` takes ``subdf["item"].iloc[0]`` for a tile, so it decides
which product is downloaded when an acquisition has been reprocessed; and it
decides the order a mean composite accumulates float32 in.

Planetary Computer holds both the original and the reprocessed product for many
acquisitions, and the last field of a Sentinel product id is its processing
timestamp. Sorting ascending by id therefore selects the *older*, superseded
baseline -- which changed real reflectances by up to 812 DN on 99% of pixels
and broke every cube in tests/test_regression_vs_release.py. The sort is
descending so the newest reprocessing wins.

Offline: the catalog is a stand-in, nothing is downloaded.
"""

import datetime as dt

import pystac
import pytest

from sentle.stac import ItemIndex, lonlat_bbox

# same acquisition, processed in 2023 and reprocessed in 2024
ORIGINAL = "S2A_MSIL2A_20230612T100601_R022_T32TPS_20230612T173551"
REPROCESSED = "S2A_MSIL2A_20230612T100601_R022_T32TPS_20240911T004021"

UTC = dt.timezone.utc
EARLIER = dt.datetime(2023, 6, 10, 10, 6, 1, tzinfo=UTC)
LATER = dt.datetime(2023, 6, 12, 10, 6, 1, tzinfo=UTC)

AREA = lonlat_bbox([9.0, 46.0, 9.1, 46.1])


def _item(item_id, when):
    return pystac.Item(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": [[[8, 45], [10, 45], [10, 47], [8, 47], [8, 45]]],
        },
        bbox=[8, 45, 10, 47],
        datetime=when,
        properties={},
        collection="sentinel-2-l2a",
    )


def _search(items):
    return ItemIndex(items).query("sentinel-2-l2a", EARLIER, LATER, AREA)


@pytest.mark.parametrize("catalog_order", [
    [ORIGINAL, REPROCESSED],
    [REPROCESSED, ORIGINAL],
])
def test_newest_reprocessing_is_selected(catalog_order):
    # whichever way the catalog lists them, the reprocessed product wins --
    # ascending order silently downloaded the superseded baseline instead
    items = [_item(item_id, LATER) for item_id in catalog_order]

    assert _search(items)[0].id == REPROCESSED


def test_order_is_independent_of_what_the_catalog_returned():
    forwards = [_item(ORIGINAL, LATER), _item(REPROCESSED, LATER)]
    backwards = [_item(REPROCESSED, LATER), _item(ORIGINAL, LATER)]

    assert [i.id for i in _search(forwards)] == [i.id
                                                 for i in _search(backwards)]


def test_timestamps_are_ordered_newest_first():
    # pins the accumulation order a mean composite sums float32 in
    items = [_item("b", EARLIER), _item("a", LATER)]

    assert [i.datetime for i in _search(items)] == [LATER, EARLIER]
