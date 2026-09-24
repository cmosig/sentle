"""Tests for ``process(..., append=True)`` -- extending a cube in time.

The whole file runs offline. ``retrieve_timestamps`` is replaced by a canned
list of timestamps (except in the composite tests, where the real function is
already network-free) and ``process_ptile`` by a stand-in that writes a
value derived from the timestamp into the slice it was handed. That exercises
the parts an append actually changes -- the config manifest, the compatibility
checks, the time-axis resize, the timestamp-to-index mapping and the
commit/rollback bookkeeping -- without downloading anything.
"""

import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from rasterio.enums import Resampling

import sentle.sentle as sentle_mod
from sentle.append import SENTLE_CONFIG_ATTR
from sentle.const import S2_RAW_BANDS
from sentle.sentle import process

# the real, network-free composite timestamp generator, captured before any
# test monkeypatches the module
REAL_RETRIEVE_TIMESTAMPS = sentle_mod.retrieve_timestamps

CRS = "EPSG:32633"
RES = 10
LEFT, BOTTOM, RIGHT, TOP = 300000.0, 5000000.0, 300200.0, 5000200.0
# the full band list, so the Sentinel-2 options under test (snow/cloud masks,
# NBAR) are all valid requests rather than band-selection errors
BANDS = list(S2_RAW_BANDS)

T1 = pd.Timestamp("2023-06-01", tz="UTC")
T2 = pd.Timestamp("2023-06-05", tz="UTC")
T3 = pd.Timestamp("2023-07-01", tz="UTC")
T4 = pd.Timestamp("2023-07-05", tz="UTC")
T_OLD = pd.Timestamp("2023-01-01", tz="UTC")


def _value_for(ts):
    """A deterministic per-timestep value so writes can be traced back."""
    return float(pd.Timestamp(ts).dayofyear)


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replace the network- and raster-touching parts of ``process``.

    Yields a mutable ``state`` whose ``timestamps`` the caller sets before each
    run and whose ``written`` records every timestamp a worker was asked to
    process.
    """
    state = types.SimpleNamespace(timestamps=[], written=[], fail_on=None,
                                  silent=False, value_offset=0.0)

    def fake_retrieve_timestamps(**kwargs):
        return [{
            "collection": "sentinel-2-l2a",
            "ts": ts
        } for ts in sorted(state.timestamps, reverse=True)]

    def fake_process_ptile(**kwargs):
        ts = kwargs["ts"]
        if state.fail_on is not None and ts == state.fail_on:
            raise RuntimeError(f"simulated download failure at {ts}")
        state.written.append(ts)
        if state.silent:
            # a tile with no data: the worker returns without writing
            return kwargs["job_id"]
        slices = kwargs["zarr_save_slice"]
        data = zarr.open(kwargs["zarr_store"])["sentle"]
        data[slices["time"], slices["band"], slices["y"],
             slices["x"]] = _value_for(ts) + state.value_offset
        return kwargs["job_id"]

    monkeypatch.setattr(sentle_mod, "retrieve_timestamps",
                        fake_retrieve_timestamps)
    monkeypatch.setattr(sentle_mod, "process_ptile", fake_process_ptile)
    # the S2 grid intersection is pure geometry but irrelevant here
    monkeypatch.setattr(sentle_mod, "obtain_subtiles",
                        lambda **kwargs: "subtiles")
    monkeypatch.setattr(sentle_mod, "gpd",
                        types.SimpleNamespace(read_file=lambda *a, **k: None))
    return state


def _run(store, **overrides):
    kwargs = dict(
        target_crs=CRS,
        target_resolution=RES,
        bound_left=LEFT,
        bound_bottom=BOTTOM,
        bound_right=RIGHT,
        bound_top=TOP,
        datetime="2023-01-01/2023-12-31",
        zarr_store=str(store),
        S1_assets=None,
        S2_bands=list(BANDS),
        num_workers=1,
        resampling_method=Resampling.nearest,
    )
    kwargs.update(overrides)
    return process(**kwargs)


@pytest.fixture
def cube(tmp_path, fake_pipeline):
    """A two-timestep cube (T1, T2) to append to."""
    store = tmp_path / "cube.zarr"
    fake_pipeline.timestamps = [T1, T2]
    _run(store)
    fake_pipeline.written.clear()
    return store


def _open(store):
    return zarr.open(str(store), mode="r")


def _times(store):
    return [
        pd.Timestamp(int(s), unit="s") for s in _open(store)["time"][:]
    ]


def _manifest(store):
    return _open(store).attrs[SENTLE_CONFIG_ATTR]


def _at(store, ts):
    """The data slice stored for ``ts`` -- looked up, not assumed positional."""
    index = _times(store).index(pd.Timestamp(ts).tz_localize(None))
    return _open(store)["sentle"][index]


# ---------------------------------------------------------------- happy path


def test_fresh_run_records_a_config_manifest(cube):
    manifest = _manifest(cube)
    assert manifest["target_resolution"] == RES
    assert manifest["S2_bands"] == BANDS
    assert manifest["S1_assets"] is None
    assert manifest["time_composite_freq"] is None
    assert manifest["provider"] == "planetary_computer"
    # a fresh cube declares all of its timesteps up front
    assert manifest["time_committed"] == 2
    assert manifest["append_in_progress"] is None


def test_append_adds_the_new_timesteps(cube, fake_pipeline):
    fake_pipeline.timestamps = [T3, T4]
    _run(cube, append=True)

    assert _open(cube)["sentle"].shape[0] == 4
    # newest first, and the new bins are merged into place rather than tacked
    # onto the end
    assert _times(cube) == [
        T4.tz_localize(None),
        T3.tz_localize(None),
        T2.tz_localize(None),
        T1.tz_localize(None),
    ]
    assert sorted(fake_pipeline.written) == [T3, T4]


def test_append_keeps_the_time_axis_sorted(cube, fake_pipeline):
    # the whole point: a cube stays readable with .sel(time=slice(...))
    for batch in ([T3], [T_OLD], [T4], [pd.Timestamp("2023-06-03", tz="UTC")]):
        fake_pipeline.timestamps = batch
        _run(cube, append=True)
        times = _times(cube)
        assert times == sorted(times, reverse=True), f"unsorted after {batch}"

    ds = xr.open_zarr(str(cube))
    assert list(pd.to_datetime(ds.time.values)) == sorted(
        pd.to_datetime(ds.time.values), reverse=True)
    # ascending slice selection works without the caller sorting first
    window = ds.sortby("time").sel(time=slice("2023-06-01", "2023-07-01"))
    assert window.sizes["time"] == 4


def test_append_preserves_the_data_already_stored(cube, fake_pipeline):
    before = _open(cube)["sentle"][:]
    fake_pipeline.timestamps = [T3, T4]
    _run(cube, append=True)

    # the stored bins moved down the axis to make room, values unchanged
    after = _open(cube)["sentle"][:]
    assert np.array_equal(before, after[2:])


def test_append_writes_each_timestep_at_its_own_index(cube, fake_pipeline):
    fake_pipeline.timestamps = [T3, T4]
    _run(cube, append=True)

    data = _open(cube)["sentle"][:]
    for index, ts in enumerate([T4, T3, T2, T1]):
        assert np.all(data[index] == _value_for(ts)), f"wrong data at {ts}"


def test_append_skips_timestamps_already_stored(cube, fake_pipeline):
    # the requested range overlaps the stored one; only T3 is genuinely new
    fake_pipeline.timestamps = [T1, T2, T3]
    _run(cube, append=True)

    assert fake_pipeline.written == [T3]
    assert _open(cube)["sentle"].shape[0] == 3
    assert len(set(_times(cube))) == 3


def test_append_with_nothing_new_leaves_the_cube_untouched(cube,
                                                           fake_pipeline):
    fake_pipeline.timestamps = [T1, T2]
    with pytest.warns(UserWarning, match="Nothing to append"):
        _run(cube, append=True)

    assert fake_pipeline.written == []
    assert _open(cube)["sentle"].shape[0] == 2
    assert _manifest(cube)["time_committed"] == 2


def test_append_can_backfill_older_timesteps(cube, fake_pipeline):
    fake_pipeline.timestamps = [T_OLD]
    _run(cube, append=True)

    # appending strictly older data keeps the newest-first axis monotonic
    times = _times(cube)
    assert times == sorted(times, reverse=True)
    assert times[-1] == T_OLD.tz_localize(None)


def test_append_of_newer_data_keeps_the_axis_sorted(cube, fake_pipeline):
    fake_pipeline.timestamps = [T3]
    _run(cube, append=True)

    times = _times(cube)
    assert times == [T3.tz_localize(None), T2.tz_localize(None),
                     T1.tz_localize(None)]
    assert np.all(_open(cube)["sentle"][0] == _value_for(T3))


def test_append_can_fill_a_gap_between_stored_timesteps(cube, fake_pipeline):
    middle = pd.Timestamp("2023-06-03", tz="UTC")
    fake_pipeline.timestamps = [middle]
    _run(cube, append=True)

    cube_ds = xr.open_zarr(str(cube))
    # inserted between T1 and T2 without the caller having to sort
    assert list(cube_ds.time.values) == [
        np.datetime64(ts.tz_localize(None)) for ts in (T2, middle, T1)
    ]
    # the gap-filled timestep carries its own data, at its own place
    assert np.all(
        cube_ds["sentle"].isel(time=1).values == _value_for(middle))


def test_appended_cube_is_readable_by_xarray(cube, fake_pipeline):
    fake_pipeline.timestamps = [T3, T4]
    _run(cube, append=True)

    ds = xr.open_zarr(str(cube))
    assert ds.sizes["time"] == 4
    assert ds.sizes["x"] == 20 and ds.sizes["y"] == 20
    assert list(ds.band.values) == BANDS
    # still newest-first, and still sorted
    assert list(pd.to_datetime(ds.time.values)) == [
        ts.tz_localize(None) for ts in (T4, T3, T2, T1)
    ]


def test_append_refreshes_consolidated_metadata(cube, fake_pipeline):
    # the workers re-open the store per tile and read the shape from the
    # consolidated metadata; a stale copy makes them write out of bounds
    fake_pipeline.timestamps = [T3, T4]
    _run(cube, append=True)

    root = zarr.open_group(str(cube), mode="r")
    assert root.metadata.consolidated_metadata is not None
    assert root["sentle"].shape[0] == 4
    assert root["time"].shape[0] == 4


def test_repeated_appends_keep_extending(cube, fake_pipeline):
    for timestamps in ([T3], [T4]):
        fake_pipeline.timestamps = timestamps
        _run(cube, append=True)

    assert _open(cube)["sentle"].shape[0] == 4
    assert _manifest(cube)["time_committed"] == 4


def test_append_updates_the_committed_length(cube, fake_pipeline):
    fake_pipeline.timestamps = [T3, T4]
    _run(cube, append=True)

    manifest = _manifest(cube)
    assert manifest["time_committed"] == 4
    assert manifest["append_in_progress"] is None


# ------------------------------------------------------------- API behaviour


def test_default_behaviour_is_unchanged(cube, fake_pipeline):
    # without append (and without overwrite) an existing store still fails
    fake_pipeline.timestamps = [T3]
    with pytest.raises(Exception):
        _run(cube)


def test_overwrite_still_replaces_the_cube(cube, fake_pipeline):
    fake_pipeline.timestamps = [T3]
    _run(cube, overwrite=True)
    assert _open(cube)["sentle"].shape[0] == 1


def test_append_and_overwrite_are_mutually_exclusive(cube, fake_pipeline):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _run(cube, append=True, overwrite=True)


def test_append_requires_an_existing_cube(tmp_path, fake_pipeline):
    fake_pipeline.timestamps = [T1]
    with pytest.raises(ValueError, match="requires an existing sentle cube"):
        _run(tmp_path / "missing.zarr", append=True)


def test_allow_missing_config_requires_append(cube, fake_pipeline):
    with pytest.raises(ValueError, match="only has an effect"):
        _run(cube, append_allow_missing_config=True)


# --------------------------------------------------- rejected configurations

REJECTED = [
    (dict(target_crs="EPSG:32632"), "target_crs"),
    (dict(target_resolution=20), "target_resolution"),
    (dict(bound_right=300400.0), "bound_right"),
    (dict(bound_left=300100.0, bound_right=300300.0), "bound_left"),
    (dict(bound_top=5000400.0), "bound_top"),
    (dict(bound_bottom=4999800.0), "bound_bottom"),
    (dict(S2_bands=["B02", "B03", "B04"]), "S2_bands"),
    (dict(S1_assets=["vv_asc"]), "S1_assets"),
    (dict(time_composite_freq="7D"), "time_composite_freq"),
    (dict(S2_mask_snow=True), "S2_mask_snow"),
    (dict(S2_mask_snow=True, S2_apply_snow_mask=True), "S2_apply_snow_mask"),
    (dict(S2_cloud_classification=True), "S2_cloud_classification"),
    (dict(S2_return_cloud_probabilities=True),
     "S2_return_cloud_probabilities"),
    (dict(S2_nbar=True), "S2_nbar"),
    (dict(save_as_uint16=True), "save_as_uint16"),
    (dict(provider="cdse"), "provider"),
    (dict(zarr_store_chunk_size={
        "time": 5,
        "x": 250,
        "y": 250
    }), "zarr_store_chunk_size"),
    (dict(resampling_method=Resampling.bilinear), "resampling_method"),
    (dict(coord_save_mode="center"), "coord_save_mode"),
]


@pytest.mark.parametrize("override,parameter",
                         REJECTED,
                         ids=[name for _, name in REJECTED])
def test_incompatible_append_is_rejected(cube, fake_pipeline, override,
                                         parameter):
    fake_pipeline.timestamps = [T3]
    with pytest.raises(ValueError, match="Cannot append") as excinfo:
        _run(cube, append=True, **override)

    # the error has to name the parameter that disagrees
    assert parameter in str(excinfo.value)
    # ... and nothing may have been written
    assert fake_pipeline.written == []
    assert _open(cube)["sentle"].shape[0] == 2
    assert _manifest(cube)["time_committed"] == 2


def test_rejection_lists_the_stored_and_requested_value(cube, fake_pipeline):
    fake_pipeline.timestamps = [T3]
    with pytest.raises(ValueError) as excinfo:
        _run(cube, append=True, time_composite_freq="7D")

    message = str(excinfo.value)
    assert "time_composite_freq: stored None, requested '7D'" in message


def test_rejection_reports_every_mismatch_at_once(cube, fake_pipeline):
    fake_pipeline.timestamps = [T3]
    with pytest.raises(ValueError) as excinfo:
        _run(cube,
             append=True,
             S2_nbar=True,
             resampling_method=Resampling.bilinear)

    message = str(excinfo.value)
    # one error, both parameters -- not a fail-on-first-mismatch
    assert "S2_nbar" in message and "resampling_method" in message


def test_shifted_grid_is_rejected_even_with_matching_extent(
        cube, fake_pipeline):
    # same size and resolution, origin moved by half a pixel: the extent check
    # alone would let this through, the coordinate arrays must not
    fake_pipeline.timestamps = [T3]
    with pytest.raises(ValueError, match="grid alignment"):
        _run(cube,
             append=True,
             bound_left=LEFT + 5,
             bound_right=RIGHT + 5,
             bound_bottom=BOTTOM + 5,
             bound_top=TOP + 5)


def test_uint16_cube_rejects_a_float_append(tmp_path, fake_pipeline):
    store = tmp_path / "uint16.zarr"
    fake_pipeline.timestamps = [T1]
    _run(store, save_as_uint16=True)
    fake_pipeline.written.clear()

    fake_pipeline.timestamps = [T3]
    with pytest.raises(ValueError, match="save_as_uint16"):
        _run(store, append=True, save_as_uint16=False)


# ------------------------------------------------- cubes without a manifest


def _strip_manifest(store):
    """Turn a cube into one written before the manifest existed."""
    root = zarr.open_group(str(store), mode="a", use_consolidated=False)
    del root.attrs[SENTLE_CONFIG_ATTR]
    zarr.consolidate_metadata(zarr.storage.LocalStore(str(store)))


def test_append_to_a_cube_without_a_manifest_is_refused(cube, fake_pipeline):
    _strip_manifest(cube)
    fake_pipeline.timestamps = [T3]

    with pytest.raises(ValueError, match="Cannot verify") as excinfo:
        _run(cube, append=True)

    message = str(excinfo.value)
    assert "append_allow_missing_config=True" in message
    assert "S2_nbar" in message
    assert fake_pipeline.written == []
    assert _open(cube)["sentle"].shape[0] == 2


def test_append_to_a_cube_without_a_manifest_can_be_forced(
        cube, fake_pipeline):
    _strip_manifest(cube)
    fake_pipeline.timestamps = [T3]

    with pytest.warns(UserWarning, match="could not be verified"):
        _run(cube, append=True, append_allow_missing_config=True)

    assert _open(cube)["sentle"].shape[0] == 3
    # the append writes a manifest, so the next one is fully checked
    assert _manifest(cube)["time_committed"] == 3


def test_forcing_still_checks_what_the_store_records(cube, fake_pipeline):
    # append_allow_missing_config waives only the unverifiable parameters --
    # grid, bands, dtype and chunking are still enforced
    _strip_manifest(cube)
    fake_pipeline.timestamps = [T3]

    with pytest.raises(ValueError, match="band list"):
        _run(cube,
             append=True,
             append_allow_missing_config=True,
             S2_bands=["B02", "B03", "B04"])
    assert fake_pipeline.written == []


# ------------------------------------------------------ temporal composites


@pytest.fixture
def composite_cube(tmp_path, fake_pipeline, monkeypatch):
    """A 7-day-composite cube built through the real timestamp generator.

    ``retrieve_timestamps`` needs no network when ``time_composite_freq`` is
    set, so the bin grid under test is the real one.
    """
    monkeypatch.setattr(sentle_mod, "retrieve_timestamps",
                        REAL_RETRIEVE_TIMESTAMPS)
    store = tmp_path / "composite.zarr"
    _run(store,
         datetime="2023-06-01/2023-06-22",
         time_composite_freq="7D",
         time_composite_method="mean")
    fake_pipeline.written.clear()
    return store


def test_composite_append_continues_the_same_bin_grid(composite_cube):
    before = _times(composite_cube)
    _run(composite_cube,
         append=True,
         datetime="2023-06-22/2023-07-13",
         time_composite_freq="7D",
         time_composite_method="mean")

    after = _times(composite_cube)
    # every stored bin survives, and the axis is still sorted
    assert set(before) <= set(after)
    assert after == sorted(after, reverse=True)
    # and every bin, old or new, sits on the same 7-day lattice
    offsets = {(ts - after[0]).total_seconds() % (7 * 86400) for ts in after}
    assert offsets == {0.0}


def test_composite_append_from_mid_bin_does_not_split_a_bin(composite_cube):
    before = set(_times(composite_cube))
    # start the new range in the middle of a bin that is already stored
    _run(composite_cube,
         append=True,
         datetime="2023-06-18/2023-07-06",
         time_composite_freq="7D",
         time_composite_method="mean")

    after = _times(composite_cube)
    # no bin is recomputed and no bin appears twice
    assert len(after) == len(set(after))
    assert before <= set(after)


def test_composite_append_only_processes_the_new_bins(composite_cube,
                                                      fake_pipeline):
    stored = set(_times(composite_cube))
    _run(composite_cube,
         append=True,
         datetime="2023-06-18/2023-07-06",
         time_composite_freq="7D",
         time_composite_method="mean")

    processed = {pd.Timestamp(ts).tz_localize(None)
                 for ts in fake_pipeline.written}
    assert processed and not (processed & stored)


def test_composite_method_mismatch_is_rejected(composite_cube):
    with pytest.raises(ValueError, match="time_composite_method"):
        _run(composite_cube,
             append=True,
             datetime="2023-07-01/2023-07-13",
             time_composite_freq="7D",
             time_composite_method="median")


def test_composite_append_rejects_off_grid_stored_bins(composite_cube):
    # nudge one stored bin off the 7-day lattice, as a cube built with a
    # different frequency would be, and make sure the append refuses it
    root = zarr.open_group(str(composite_cube), mode="a",
                           use_consolidated=False)
    root["time"][0] = int(root["time"][0]) + 3600
    zarr.consolidate_metadata(zarr.storage.LocalStore(str(composite_cube)))

    with pytest.raises(ValueError, match="not spaced on a"):
        _run(composite_cube,
             append=True,
             datetime="2023-07-01/2023-07-13",
             time_composite_freq="7D",
             time_composite_method="mean")


# ----------------------------------------------------- interrupted appends


def test_failed_append_rolls_the_cube_back(cube, fake_pipeline):
    before = _open(cube)["sentle"][:]
    fake_pipeline.timestamps = [T3, T4]
    fake_pipeline.fail_on = T4

    with pytest.raises(Exception):
        _run(cube, append=True)

    # the cube must not claim timesteps that were never fully written
    assert _open(cube)["sentle"].shape[0] == 2
    assert _open(cube)["time"].shape[0] == 2
    assert _manifest(cube)["time_committed"] == 2
    assert np.array_equal(_open(cube)["sentle"][:], before)


def test_failed_append_leaves_a_readable_cube(cube, fake_pipeline):
    fake_pipeline.timestamps = [T3]
    fake_pipeline.fail_on = T3
    with pytest.raises(Exception):
        _run(cube, append=True)

    ds = xr.open_zarr(str(cube))
    assert ds.sizes["time"] == 2


def test_a_cube_can_be_appended_to_after_a_failed_append(cube, fake_pipeline):
    fake_pipeline.timestamps = [T3, T4]
    fake_pipeline.fail_on = T4
    with pytest.raises(Exception):
        _run(cube, append=True)

    fake_pipeline.fail_on = None
    fake_pipeline.written.clear()
    fake_pipeline.timestamps = [T3, T4]
    _run(cube, append=True)

    assert _open(cube)["sentle"].shape[0] == 4
    assert sorted(fake_pipeline.written) == [T3, T4]


def _simulate_hard_kill(store, pending_seconds):
    """Grow the arrays past ``time_committed``, as a killed append leaves them."""
    root = zarr.open_group(str(store), mode="a", use_consolidated=False)
    data, time = root["sentle"], root["time"]
    start = time.shape[0]
    data.resize((start + len(pending_seconds), ) + tuple(data.shape[1:]))
    time.resize((start + len(pending_seconds), ))
    time[start:] = pending_seconds
    # partial output from the killed run
    data[start:] = 999.0
    zarr.consolidate_metadata(zarr.storage.LocalStore(str(store)))
    return start


def test_killed_append_is_rolled_back_on_the_next_append(cube, fake_pipeline):
    pending = [
        int(T3.tz_localize(None).timestamp()),
        int(T4.tz_localize(None).timestamp())
    ]
    _simulate_hard_kill(cube, pending)
    assert _open(cube)["sentle"].shape[0] == 4  # the cube over-claims

    fake_pipeline.timestamps = [T3]
    with pytest.warns(UserWarning, match="did not finish"):
        _run(cube, append=True)

    # rolled back to 2, then extended by the one requested timestep
    assert _open(cube)["sentle"].shape[0] == 3
    assert _manifest(cube)["time_committed"] == 3
    assert fake_pipeline.written == [T3]


def test_rollback_clears_partial_data_from_the_shared_time_chunk(
        cube, fake_pipeline):
    # time chunk is 10 and the cube holds 2 timesteps, so a rolled-back append
    # leaves its bytes inside a chunk that resize keeps. A later timestep must
    # not inherit them.
    assert _open(cube)["sentle"].chunks[0] == 10
    _simulate_hard_kill(cube, [
        int(T3.tz_localize(None).timestamp()),
        int(T4.tz_localize(None).timestamp())
    ])

    # the next append finds no data for T_OLD, so nothing is written there
    fake_pipeline.timestamps = [T_OLD]
    fake_pipeline.silent = True
    with pytest.warns(UserWarning, match="did not finish"):
        _run(cube, append=True)

    data = _open(cube)["sentle"][:]
    assert data.shape[0] == 3
    # index 2 was never written by this run, so it must read as NoData rather
    # than as the killed run's leftovers
    assert np.isnan(data[2]).all()


def test_inconsistent_committed_length_is_refused(cube, fake_pipeline):
    root = zarr.open_group(str(cube), mode="a", use_consolidated=False)
    config = dict(root.attrs[SENTLE_CONFIG_ATTR])
    config["time_committed"] = 99
    root.attrs[SENTLE_CONFIG_ATTR] = config
    zarr.consolidate_metadata(zarr.storage.LocalStore(str(cube)))

    fake_pipeline.timestamps = [T3]
    with pytest.raises(ValueError, match="inconsistent"):
        _run(cube, append=True)


# ------------------------------------------------ recomputing stored bins


def test_recompute_trailing_overwrites_the_newest_stored_timestep(
        cube, fake_pipeline):
    # the cube holds T2 (newest) at index 0 and T1 at index 1
    fake_pipeline.timestamps = [T1, T2, T3]
    fake_pipeline.value_offset = 1000.0
    _run(cube, append=True, append_recompute_trailing=1)

    assert _open(cube)["sentle"].shape[0] == 3
    # T2 was recomputed with the new run's value, wherever it now sits
    assert np.all(_at(cube, T2) == _value_for(T2) + 1000.0)
    # T1 was left exactly as it was
    assert np.all(_at(cube, T1) == _value_for(T1))
    # T3 was inserted
    assert np.all(_at(cube, T3) == _value_for(T3) + 1000.0)
    assert sorted(fake_pipeline.written) == [T2, T3]


def test_recompute_trailing_does_not_change_the_time_axis(cube,
                                                          fake_pipeline):
    before = _times(cube)
    fake_pipeline.timestamps = [T1, T2]      # nothing new, only a recompute
    fake_pipeline.value_offset = 7.0
    _run(cube, append=True, append_recompute_trailing=1)

    assert _times(cube) == before
    assert _open(cube)["sentle"].shape[0] == 2
    assert _manifest(cube)["time_committed"] == 2
    assert np.all(_open(cube)["sentle"][0] == _value_for(T2) + 7.0)


def test_recompute_trailing_clears_the_timestep_first(cube, fake_pipeline):
    # a composite writes only where it has data; if the recomputed run finds
    # none, the bin must end up NoData rather than keeping the old pixels
    fake_pipeline.timestamps = [T1, T2]
    fake_pipeline.silent = True
    _run(cube, append=True, append_recompute_trailing=1)

    data = _open(cube)["sentle"][:]
    assert np.isnan(data[0]).all(), "stale pixels survived the recompute"
    assert np.all(data[1] == _value_for(T1)), "the other timestep was touched"


def test_recompute_trailing_picks_the_newest_by_value_not_position(
        cube, fake_pipeline):
    # T3 is the newest timestamp but it is not the newest *stored* one until
    # it is merged in; after this append it sits at index 0
    fake_pipeline.timestamps = [T3]
    _run(cube, append=True)
    assert _times(cube)[0] == T3.tz_localize(None)

    fake_pipeline.written.clear()
    fake_pipeline.timestamps = [T1, T2, T3]
    fake_pipeline.value_offset = 500.0
    _run(cube, append=True, append_recompute_trailing=1)

    assert fake_pipeline.written == [T3]
    assert np.all(_at(cube, T3) == _value_for(T3) + 500.0)
    assert np.all(_at(cube, T2) == _value_for(T2))
    assert np.all(_at(cube, T1) == _value_for(T1))


def test_recompute_trailing_beyond_the_requested_range_warns(cube,
                                                             fake_pipeline):
    # only T2 is inside the requested range, so only one can be recomputed
    fake_pipeline.timestamps = [T2, T3]
    with pytest.warns(UserWarning, match="only 1 of the newest stored"):
        _run(cube, append=True, append_recompute_trailing=2)

    assert _open(cube)["sentle"].shape[0] == 3


def test_recompute_trailing_requires_append(cube, fake_pipeline):
    with pytest.raises(ValueError, match="only has an effect"):
        _run(cube, append_recompute_trailing=1)


@pytest.mark.parametrize("bad", [-1, 1.5, "1", True])
def test_recompute_trailing_rejects_bad_values(cube, fake_pipeline, bad):
    with pytest.raises(ValueError, match="append_recompute_trailing"):
        _run(cube, append=True, append_recompute_trailing=bad)


def test_failed_recompute_says_the_timesteps_were_cleared(cube,
                                                          fake_pipeline):
    fake_pipeline.timestamps = [T2, T3]
    fake_pipeline.fail_on = T3

    with pytest.warns(UserWarning, match="are now NoData"):
        with pytest.raises(Exception):
            _run(cube, append=True, append_recompute_trailing=1)

    # the appended timestep is rolled back; the recomputed one stays cleared
    assert _open(cube)["sentle"].shape[0] == 2
    assert _manifest(cube)["time_committed"] == 2
    assert np.isnan(_open(cube)["sentle"][0]).all()


def test_recompute_trailing_of_several_timesteps(cube, fake_pipeline):
    fake_pipeline.timestamps = [T1, T2]
    fake_pipeline.value_offset = 3.0
    _run(cube, append=True, append_recompute_trailing=2)

    data = _open(cube)["sentle"][:]
    assert np.all(data[0] == _value_for(T2) + 3.0)
    assert np.all(data[1] == _value_for(T1) + 3.0)
    assert _open(cube)["sentle"].shape[0] == 2


# ------------------------------------------- the sorted-axis invariant


def test_stored_data_moves_down_to_make_room(cube, fake_pipeline):
    before = _open(cube)["sentle"][:]
    before_times = _times(cube)
    fake_pipeline.timestamps = [T3, T4]
    _run(cube, append=True)

    after = _open(cube)["sentle"][:]
    after_times = _times(cube)
    # the two stored bins are now at the bottom of the axis, values intact
    assert after_times[2:] == before_times
    assert np.array_equal(after[2:], before)


def test_backfill_leaves_stored_bins_where_they_were(cube, fake_pipeline):
    before = _open(cube)["sentle"][:]
    fake_pipeline.timestamps = [T_OLD]
    _run(cube, append=True)

    # a pure backfill sorts after everything stored, so nothing has to move
    after = _open(cube)["sentle"][:]
    assert np.array_equal(after[:2], before)
    assert np.all(_at(cube, T_OLD) == _value_for(T_OLD))
    assert _times(cube) == sorted(_times(cube), reverse=True)


def test_inserted_slot_does_not_inherit_shifted_data(cube, fake_pipeline):
    # the shift copies rather than moves, so a slot the run finds no data for
    # must read back as NoData, not as whatever used to sit at that index
    fake_pipeline.timestamps = [T3]
    fake_pipeline.silent = True
    _run(cube, append=True)

    assert np.isnan(_at(cube, T3)).all()
    assert np.all(_at(cube, T2) == _value_for(T2))
    assert np.all(_at(cube, T1) == _value_for(T1))


def test_failed_append_restores_the_original_layout(cube, fake_pipeline):
    before = _open(cube)["sentle"][:]
    before_times = _times(cube)
    fake_pipeline.timestamps = [T3, T4]
    fake_pipeline.fail_on = T4

    with pytest.raises(Exception):
        _run(cube, append=True)

    # the shift has to be undone, not just the axis truncated
    assert _open(cube)["sentle"].shape[0] == 2
    assert _times(cube) == before_times
    assert np.array_equal(_open(cube)["sentle"][:], before)


def test_killed_append_mid_shift_is_repaired(cube, fake_pipeline):
    before = _open(cube)["sentle"][:]
    before_times = _times(cube)

    # stop the run right after the shift, before any data is written
    # (insert_time_slots clears the new slots, so that is the seam)
    import sentle.append as append_mod
    real_clear = append_mod.clear_timesteps

    def boom(*a, **k):
        real_clear(*a, **k)
        raise KeyboardInterrupt("killed mid-append")

    fake_pipeline.timestamps = [T3, T4]
    with pytest.raises(KeyboardInterrupt):
        append_mod.clear_timesteps = boom
        try:
            _run(cube, append=True)
        finally:
            append_mod.clear_timesteps = real_clear

    # the cube is left claiming 4 timesteps; the next append must repair it
    fake_pipeline.timestamps = [T3]
    with pytest.warns(UserWarning, match="did not finish"):
        _run(cube, append=True)

    assert _times(cube) == [T3.tz_localize(None)] + before_times
    assert np.array_equal(_open(cube)["sentle"][1:], before)
    assert np.all(_at(cube, T3) == _value_for(T3))


def test_unsorted_cube_is_refused_and_can_be_repaired(cube, fake_pipeline):
    from sentle.append import repair_time_order

    # forge the layout the old append produced: sorted block + newer block
    root = zarr.open_group(str(cube), mode="a", use_consolidated=False)
    data, time = root["sentle"], root["time"]
    data.resize((3, ) + tuple(data.shape[1:]))
    time.resize((3, ))
    time[2] = int(T3.tz_localize(None).timestamp())
    data[2] = _value_for(T3)
    config = dict(root.attrs[SENTLE_CONFIG_ATTR])
    config["time_committed"] = 3
    root.attrs[SENTLE_CONFIG_ATTR] = config
    zarr.consolidate_metadata(zarr.storage.LocalStore(str(cube)))
    assert _times(cube) != sorted(_times(cube), reverse=True)

    fake_pipeline.timestamps = [T4]
    with pytest.raises(ValueError, match="repair_time_order"):
        _run(cube, append=True)

    assert repair_time_order(str(cube)) is True
    assert _times(cube) == [T3.tz_localize(None), T2.tz_localize(None),
                            T1.tz_localize(None)]
    assert np.all(_at(cube, T3) == _value_for(T3))
    assert np.all(_at(cube, T2) == _value_for(T2))
    assert np.all(_at(cube, T1) == _value_for(T1))

    # and it is appendable again
    _run(cube, append=True)
    assert _times(cube) == sorted(_times(cube), reverse=True)


def test_repair_is_a_no_op_on_a_sorted_cube(cube):
    from sentle.append import repair_time_order
    before = _times(cube)
    assert repair_time_order(str(cube)) is False
    assert _times(cube) == before
