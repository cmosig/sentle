"""Tests for ``retrieve_timestamps`` in its temporal-composite mode.

When ``time_composite_freq`` is set, the time axis is generated purely from the
requested ``datetime`` range and the frequency -- no catalog query -- so that
every composite time series is aligned to the same grid regardless of the exact
start/end the user asked for. This branch is fully offline and is what these
tests exercise. (The no-composite branch queries Planetary Computer and is left
to the end-to-end tests.)
"""

import pandas as pd
import pytest
from rasterio.crs import CRS

from sentle.sentle import retrieve_timestamps


def _call(freq, datetime, collections=("sentinel-2-l2a",)):
    return retrieve_timestamps(
        time_composite_freq=freq,
        datetime=datetime,
        bound_left=0.0,
        bound_bottom=0.0,
        bound_right=1000.0,
        bound_top=1000.0,
        target_crs=CRS.from_epsg(32633),
        collections=list(collections),
    )


def test_daily_range_is_descending_and_complete():
    out = _call("1D", "2020-01-05/2020-01-08")
    ts = [row["ts"] for row in out]
    assert ts == [
        pd.Timestamp("2020-01-08", tz="UTC"),
        pd.Timestamp("2020-01-07", tz="UTC"),
        pd.Timestamp("2020-01-06", tz="UTC"),
        pd.Timestamp("2020-01-05", tz="UTC"),
    ]


def test_all_entries_carry_the_single_collection():
    out = _call("1D", "2020-01-05/2020-01-06")
    assert {row["collection"] for row in out} == {"sentinel-2-l2a"}


def test_multiple_collections_expand_each_timestamp():
    out = _call("1D", "2020-01-05/2020-01-06",
                collections=("sentinel-2-l2a", "sentinel-1-rtc"))
    # 2 timestamps x 2 collections
    assert len(out) == 4
    # each timestamp appears once per collection
    per_ts = {}
    for row in out:
        per_ts.setdefault(row["ts"], set()).add(row["collection"])
    for cols in per_ts.values():
        assert cols == {"sentinel-2-l2a", "sentinel-1-rtc"}


def test_single_datetime_yields_one_timestamp():
    out = _call("1D", "2020-01-05")
    assert [row["ts"] for row in out] == [pd.Timestamp("2020-01-05", tz="UTC")]


def test_timestamps_are_evenly_spaced_by_freq():
    out = _call("2D", "2020-01-01/2020-01-09")
    ts = [row["ts"] for row in out]
    diffs = {a - b for a, b in zip(ts[:-1], ts[1:])}
    assert diffs == {pd.Timedelta("2D")}


def test_start_is_rounded_to_frequency_grid():
    # start rounds to the freq grid so series with different requested starts
    # still align; 7D grid anchors on the pandas epoch week boundary
    out = _call("7D", "2020-01-03/2020-01-20")
    start = min(row["ts"] for row in out)
    assert start == pd.Timestamp("2020-01-03", tz="UTC").round("7D")
