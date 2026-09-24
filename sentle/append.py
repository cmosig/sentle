"""Extending an existing sentle cube along the time axis (``append=True``).

A sentle cube is a Zarr store whose ``sentle`` array is indexed
``(time, band, y, x)``. Everything except ``time`` is fixed when the cube is
created, so "appending" means exactly one thing: grow the ``time`` dimension of
``sentle`` and of the ``time`` coordinate array, and write the timesteps that
are not stored yet.

Two things make that harder than it sounds.

**The cube has to agree with the new call.** A cube built at 10 m in EPSG:32633
with 7-day composites cannot absorb timesteps produced at 20 m, in a different
CRS, or without compositing -- the result would be a cube whose pixels mean
different things at different times, with nothing on disk saying so. Up to now
the store recorded only the CRS, the band names and the x/y coordinates, which
is not enough to tell. Cubes created from this version on therefore carry a
config manifest in the root attributes (:data:`SENTLE_CONFIG_ATTR`) listing
every parameter that shapes the pixels. Cubes written before it exists are
still checked against everything derivable from the store itself (grid, bands,
dtype, chunking, composite bin alignment), and the remaining parameters are
refused rather than assumed -- see
:func:`check_append_compatible`.

**An interrupted append must not leave the cube claiming timesteps it never
wrote.** Growing the ``time`` coordinate and growing ``sentle`` are two
separate writes, and the processing happens after both. The manifest therefore
also carries ``time_committed``: the number of timesteps that were fully
processed. ``process()`` rolls the cube back to that length if the work raises,
and :func:`recover_interrupted_append` performs the same rollback at the start
of the next append if the process was killed outright.
"""

import os
import shutil
import warnings

import numpy as np
import pandas as pd
import zarr
from rasterio.crs import CRS
from rasterio.enums import Resampling

# root attribute holding the config manifest of the cube
SENTLE_CONFIG_ATTR = "sentle_config"

# bumped when the manifest layout changes in a way older sentle cannot read
SENTLE_CONFIG_VERSION = 1

# bounds/resolution are floats that survive a JSON round-trip exactly, but the
# user may re-type them slightly differently between runs; compare them at a
# fraction of a pixel rather than bit-for-bit. Grid alignment proper is checked
# on the stored x/y coordinate arrays, which is the stricter test.
_COORD_TOL = 1e-6

# manifest entries compared as plain floats (scaled by target_resolution)
_FLOAT_KEYS = ("target_resolution", "bound_left", "bound_bottom",
               "bound_right", "bound_top")

# Parameters that shape the pixels but leave no trace in the store itself, so
# they can only be verified against the manifest. Order is the order they are
# reported in.
MANIFEST_ONLY_PARAMS = (
    "time_composite_freq",
    "time_composite_method",
    "S2_mask_snow",
    "S2_apply_snow_mask",
    "S2_cloud_classification",
    "S2_apply_cloud_mask",
    "S2_return_cloud_probabilities",
    "S2_nbar",
    "resampling_method",
    "provider",
)

# Parameters checked against the manifest *and* independently against the store
# (bands, dtype, chunking and the x/y grid are all derivable from the arrays).
_STORE_BACKED_PARAMS = (
    "target_crs",
    "target_resolution",
    "bound_left",
    "bound_bottom",
    "bound_right",
    "bound_top",
    "S2_bands",
    "S1_assets",
    "save_as_uint16",
    "zarr_store_chunk_size",
    # shifts the x/y coordinates by half a pixel, so the grid check catches it
    "coord_save_mode",
)


def build_store_config(
    *,
    target_crs,
    target_resolution: float,
    bound_left: float,
    bound_bottom: float,
    bound_right: float,
    bound_top: float,
    S2_bands,
    S2_bands_to_save,
    S1_assets,
    total_bands_to_save,
    time_composite_freq,
    time_composite_method: str,
    S2_mask_snow: bool,
    S2_apply_snow_mask: bool,
    S2_cloud_classification: bool,
    S2_apply_cloud_mask: bool,
    S2_return_cloud_probabilities: bool,
    S2_nbar: bool,
    save_as_uint16: bool,
    provider: str,
    coord_save_mode: str,
    resampling_method,
    zarr_store_chunk_size: dict,
) -> dict:
    """Build the JSON-serializable config manifest stored in the cube."""
    return {
        "version": SENTLE_CONFIG_VERSION,
        "target_crs": CRS.from_user_input(target_crs).to_wkt(),
        "target_resolution": float(target_resolution),
        "bound_left": float(bound_left),
        "bound_bottom": float(bound_bottom),
        "bound_right": float(bound_right),
        "bound_top": float(bound_top),
        "S2_bands": list(S2_bands) if S2_bands else None,
        "S2_bands_to_save": list(S2_bands_to_save),
        "S1_assets": list(S1_assets) if S1_assets else None,
        "bands": list(total_bands_to_save),
        "time_composite_freq": time_composite_freq,
        "time_composite_method": time_composite_method,
        "S2_mask_snow": bool(S2_mask_snow),
        "S2_apply_snow_mask": bool(S2_apply_snow_mask),
        "S2_cloud_classification": bool(S2_cloud_classification),
        "S2_apply_cloud_mask": bool(S2_apply_cloud_mask),
        "S2_return_cloud_probabilities": bool(S2_return_cloud_probabilities),
        "S2_nbar": bool(S2_nbar),
        "save_as_uint16": bool(save_as_uint16),
        "provider": str(provider),
        "coord_save_mode": str(coord_save_mode),
        "resampling_method": Resampling(resampling_method).name,
        "zarr_store_chunk_size": {
            k: int(zarr_store_chunk_size[k])
            for k in ("time", "y", "x")
        },
    }


def read_store_config(root) -> dict | None:
    """Return the config manifest of an opened cube, or None if it has none."""
    config = root.attrs.get(SENTLE_CONFIG_ATTR)
    if not isinstance(config, dict):
        return None
    return dict(config)


def write_store_config(root, config: dict) -> None:
    """Persist the config manifest into the cube's root attributes."""
    root.attrs[SENTLE_CONFIG_ATTR] = config


def _update_store_config(root, **updates) -> dict:
    """Merge ``updates`` into the stored manifest and write it back."""
    config = read_store_config(root) or {}
    config.update(updates)
    write_store_config(root, config)
    return config


def _format(value) -> str:
    if isinstance(value, str) and value.startswith(("PROJCS", "GEOGCS",
                                                    "PROJCRS", "GEOGCRS")):
        # a full WKT is unreadable in an error message
        try:
            return str(CRS.from_wkt(value))
        except Exception:
            pass
    return repr(value)


def _values_equal(key, stored, requested, target_resolution) -> bool:
    if key == "target_crs":
        try:
            return CRS.from_wkt(stored) == CRS.from_wkt(requested)
        except Exception:
            return stored == requested
    if key in _FLOAT_KEYS:
        try:
            return abs(float(stored) - float(requested)) <= (
                _COORD_TOL * abs(float(target_resolution)))
        except (TypeError, ValueError):
            return stored == requested
    return stored == requested


def check_append_compatible(
    *,
    stored_config: dict | None,
    requested_config: dict,
    stored_bands: list[str],
    stored_chunks: tuple,
    stored_dtype,
    stored_x: np.ndarray,
    stored_y: np.ndarray,
    requested_x: np.ndarray,
    requested_y: np.ndarray,
    requested_band_chunk: int,
    allow_missing_config: bool,
) -> None:
    """Refuse an append whose configuration disagrees with the cube on disk.

    Raises a single ``ValueError`` listing *every* mismatch it found, each
    naming the parameter, the stored value and the requested one. Nothing is
    written before this returns.
    """
    resolution = requested_config["target_resolution"]
    mismatches = []

    def report(name, stored, requested):
        mismatches.append(
            f"  - {name}: stored {_format(stored)}, requested "
            f"{_format(requested)}")

    # ------------------------------------------------------------------
    # checks against the store itself -- these work for every cube, with or
    # without a manifest
    if list(stored_bands) != list(requested_config["bands"]):
        report("band list (S2_bands/S1_assets/mask layers)", list(stored_bands),
               list(requested_config["bands"]))

    requested_dtype = np.dtype(
        np.uint16 if requested_config["save_as_uint16"] else np.float32)
    if np.dtype(stored_dtype) != requested_dtype:
        report("save_as_uint16 (stored dtype)", str(np.dtype(stored_dtype)),
               str(requested_dtype))

    requested_chunks = requested_config["zarr_store_chunk_size"]
    stored_chunk_dict = {
        "time": int(stored_chunks[0]),
        "y": int(stored_chunks[2]),
        "x": int(stored_chunks[3]),
    }
    if stored_chunk_dict != requested_chunks:
        report("zarr_store_chunk_size", stored_chunk_dict, requested_chunks)
    if int(stored_chunks[1]) != int(requested_band_chunk):
        report("band chunk size", int(stored_chunks[1]),
               int(requested_band_chunk))

    # the authoritative grid check: the coordinates this call would have
    # written must be exactly the ones already on disk. Catches a shifted
    # origin, a different extent and a different resolution in one go.
    for axis, stored_coord, requested_coord in (("x", stored_x, requested_x),
                                                ("y", stored_y, requested_y)):
        if len(stored_coord) != len(requested_coord):
            report(f"{axis} coordinate length (bounds/target_resolution)",
                   len(stored_coord), len(requested_coord))
        elif not np.array_equal(stored_coord, requested_coord):
            bad = int(np.argmax(stored_coord != requested_coord))
            report(
                f"{axis} coordinates (grid alignment, first difference at "
                f"index {bad})", float(stored_coord[bad]),
                float(requested_coord[bad]))

    # ------------------------------------------------------------------
    # checks against the manifest
    if stored_config is None:
        # Everything derivable from the store was checked above; the rest
        # cannot be verified. Do not assume it matches.
        if not allow_missing_config:
            raise ValueError(
                "Cannot verify that this append matches the existing cube: it "
                f"was created by a sentle version that did not write the "
                f"{SENTLE_CONFIG_ATTR!r} manifest, so the parameters "
                + ", ".join(MANIFEST_ONLY_PARAMS) +
                " are not recorded anywhere and a mismatch would silently "
                "produce a cube whose timesteps mean different things.\n"
                "Either recreate the cube with overwrite=True (recommended), "
                "or, if you are certain the parameters of this call are "
                "identical to the original run, pass "
                "append_allow_missing_config=True to append anyway." +
                ("\nThe following also differ:\n" + "\n".join(mismatches)
                 if mismatches else ""))
        warnings.warn(
            "Appending to a cube without a "
            f"{SENTLE_CONFIG_ATTR!r} manifest (append_allow_missing_config="
            "True): " + ", ".join(MANIFEST_ONLY_PARAMS) +
            " could not be verified against the original run.")
    else:
        for key in _STORE_BACKED_PARAMS + MANIFEST_ONLY_PARAMS:
            if key not in stored_config:
                # a newer parameter against an older (but present) manifest
                continue
            if key == "time_composite_method" and (
                    requested_config["time_composite_freq"] is None):
                # irrelevant without compositing
                continue
            if not _values_equal(key, stored_config[key],
                                 requested_config[key], resolution):
                report(key, stored_config[key], requested_config[key])

    if mismatches:
        raise ValueError(
            "Cannot append to the existing cube: the requested configuration "
            "does not match the cube on disk.\n" + "\n".join(mismatches) +
            "\nPass parameters identical to the original run, or use "
            "overwrite=True to replace the cube.")


def timestamps_to_stored_seconds(timestamps) -> np.ndarray:
    """Convert timestamps to the int64 epoch seconds the cube stores.

    The ``time`` array is ``int64`` seconds since 1970-01-01, so this is the
    only representation in which a new timestamp can be compared against what
    is already stored.
    """
    def to_seconds(ts):
        ts = pd.Timestamp(ts)
        if ts.tz is not None:
            ts = ts.tz_localize(tz=None)
        return int((ts - pd.Timestamp(0, tz=None)).total_seconds())

    return np.array([to_seconds(ts) for ts in timestamps], dtype="int64")


def _fixed_freq_seconds(time_composite_freq) -> float | None:
    """Length of ``time_composite_freq`` in seconds, or None if not fixed."""
    if time_composite_freq is None:
        return None
    try:
        seconds = pd.Timedelta(time_composite_freq).total_seconds()
    except (ValueError, TypeError):
        # non-fixed offsets (month/year ends) have no constant length;
        # retrieve_timestamps cannot round to them either, so this cannot
        # normally happen -- skip the check rather than guess.
        return None
    return seconds if seconds > 0 else None


def check_composite_grid_alignment(existing_seconds: np.ndarray,
                                   new_seconds: np.ndarray,
                                   time_composite_freq) -> None:
    """Refuse an append whose composite bins do not sit on the stored grid.

    With ``time_composite_freq`` set, a timestep is the aggregate of every
    acquisition in ``[ts - freq/2, ts + freq/2]``. ``retrieve_timestamps``
    rounds the requested start onto the epoch-aligned ``freq`` lattice, so two
    runs with the same frequency always produce the same bin boundaries no
    matter where in a bin their date ranges start -- a bin is either already
    stored (and skipped whole) or entirely new, never half-recomputed.

    This verifies that invariant instead of trusting it, which also gives cubes
    without a manifest a real check on ``time_composite_freq``.
    """
    step = _fixed_freq_seconds(time_composite_freq)
    if step is None or len(existing_seconds) == 0:
        return

    reference = int(existing_seconds[0])
    off_grid = [
        int(s) for s in existing_seconds if (int(s) - reference) % step != 0
    ]
    if off_grid:
        raise ValueError(
            f"Cannot append: the stored timesteps are not spaced on a "
            f"{time_composite_freq!r} grid (e.g. "
            f"{pd.Timestamp(off_grid[0], unit='s')} is not a whole number of "
            f"{time_composite_freq!r} bins from "
            f"{pd.Timestamp(reference, unit='s')}), so the existing cube was "
            f"not built with time_composite_freq={time_composite_freq!r}.")

    off_grid = [int(s) for s in new_seconds if (int(s) - reference) % step != 0]
    if off_grid:
        raise ValueError(
            f"Cannot append: the new composite bin centred on "
            f"{pd.Timestamp(off_grid[0], unit='s')} does not fall on the same "
            f"{time_composite_freq!r} bin grid as the stored data (anchored at "
            f"{pd.Timestamp(reference, unit='s')}). Appending it would split a "
            f"bin across two timesteps.")


def open_cube_for_append(zarr_store):
    """Open an existing cube for appending.

    Returns ``(store, root, was_consolidated)``. The group is opened with
    ``use_consolidated=False``: consolidated metadata caches the array shapes,
    which is exactly what an append changes, so every read and write here must
    go to the real metadata documents.
    """
    if isinstance(zarr_store, str):
        store = zarr.storage.LocalStore(zarr_store)
    else:
        store = zarr_store

    try:
        probe = zarr.open_group(store=store, mode="r")
    except Exception as e:
        raise ValueError(
            f"append=True requires an existing sentle cube at {zarr_store!r}, "
            f"but it could not be opened ({e}). Run once with append=False to "
            f"create it.") from e

    was_consolidated = getattr(probe.metadata, "consolidated_metadata",
                               None) is not None

    root = zarr.open_group(store=store, mode="a", use_consolidated=False)
    for array_name in ("sentle", "time", "band", "x", "y"):
        if array_name not in root:
            raise ValueError(
                f"append=True requires an existing sentle cube at "
                f"{zarr_store!r}, but the store has no {array_name!r} array. "
                f"Run once with append=False to create it.")

    return store, root, was_consolidated


def consolidate(store) -> None:
    """Refresh consolidated metadata, suppressing zarr's v3 warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zarr.consolidate_metadata(store)


def _clear_shared_time_chunk(data, committed: int) -> None:
    """Reset the part of ``committed``'s time chunk that lies past it.

    ``resize`` deletes chunks that fall entirely outside the new shape, but the
    chunk straddling ``committed`` stays -- it still holds committed data. Its
    tail keeps whatever an aborted append wrote there, and a later append that
    grows the array back would expose those bytes as if they belonged to the
    new timestamps. Overwrite that tail with the fill value so the region is
    guaranteed to read as NoData when it reappears.
    """
    chunk_time = int(data.chunks[0])
    length = int(data.shape[0])
    if committed >= length or chunk_time <= 1 or committed % chunk_time == 0:
        # nothing kept: resize drops whole chunks on its own
        return

    boundary_end = min(length, -(-committed // chunk_time) * chunk_time)
    fill = data.fill_value
    # walk the spatial chunk grid so the write stays chunk-sized in memory
    _, _, chunk_y, chunk_x = (int(c) for c in data.chunks)
    for y0 in range(0, int(data.shape[2]), chunk_y):
        for x0 in range(0, int(data.shape[3]), chunk_x):
            data[committed:boundary_end, :, y0:y0 + chunk_y,
                 x0:x0 + chunk_x] = fill


def clear_timesteps(root, indices) -> None:
    """Reset stored timesteps to NoData, ready to be recomputed.

    A composite writes only where it found data, so recomputing a bin straight
    on top of its old contents would leave the previous run's pixels wherever
    the new one has a gap -- one timestep holding two different runs. Clearing
    first makes the bin unambiguously the product of the new run.
    """
    data = root["sentle"]
    fill = data.fill_value
    _, _, chunk_y, chunk_x = (int(c) for c in data.chunks)
    for index in indices:
        # walk the spatial chunk grid so the write stays chunk-sized in memory
        for y0 in range(0, int(data.shape[2]), chunk_y):
            for x0 in range(0, int(data.shape[3]), chunk_x):
                data[int(index), :, y0:y0 + chunk_y, x0:x0 + chunk_x] = fill


def _truncate_time_axis(root, length: int) -> None:
    """Shrink ``sentle`` and ``time`` back to ``length`` timesteps."""
    data = root["sentle"]
    _clear_shared_time_chunk(data, length)
    data.resize((length, ) + tuple(data.shape[1:]))
    root["time"].resize((length, ))


def recover_interrupted_append(root, store, zarr_store,
                               reconsolidate: bool) -> int:
    """Roll a cube back to its last committed length, if it was interrupted.

    Returns the number of timesteps the cube holds afterwards. A cube that was
    never interrupted (or predates ``time_committed``) is left untouched.
    """
    stored_length = int(root["time"].shape[0])
    config = read_store_config(root)
    if config is None or "time_committed" not in config:
        return stored_length

    committed = int(config["time_committed"])
    if committed == stored_length:
        return stored_length
    if committed > stored_length:
        raise ValueError(
            f"The cube is inconsistent: its manifest records "
            f"{committed} committed timesteps but the time axis only has "
            f"{stored_length}. Refusing to append to it.")

    warnings.warn(
        f"A previous append to this cube did not finish: the time axis has "
        f"{stored_length} timesteps but only {committed} were committed. "
        f"Rolling the cube back to {committed} timesteps and discarding the "
        f"{stored_length - committed} partial ones.")
    _undo_extension(root, zarr_store, committed,
                    config.get("append_in_progress"))
    _truncate_time_axis(root, committed)
    _update_store_config(root, time_committed=committed,
                         append_in_progress=None)
    if reconsolidate:
        consolidate(store)
    return committed


def time_axis_order(seconds: np.ndarray) -> str | None:
    """``"descending"``, ``"ascending"``, or None if the axis is not sorted."""
    seconds = np.asarray(seconds)
    if len(seconds) < 2:
        return "descending"
    diffs = np.diff(seconds)
    if (diffs < 0).all():
        return "descending"
    if (diffs > 0).all():
        return "ascending"
    return None


def local_store_path(zarr_store) -> str | None:
    """Filesystem root of ``zarr_store``, or None if it is not a local store."""
    if isinstance(zarr_store, str):
        return zarr_store
    root = getattr(zarr_store, "root", None)
    return str(root) if root is not None else None


def plan_extension(existing_seconds: np.ndarray, new_seconds: np.ndarray,
                   chunk_time: int, reserved: dict | None = None) -> dict:
    """Work out where new timesteps go without disturbing the stored ones.

    The cube is newest-first and zarr only grows an array at its end, so newer
    timesteps have to be *prepended*. That is done by renaming whole time-chunk
    keys (see :func:`prepend_time_chunks`), which requires the number of freed
    slots to be a multiple of ``chunk_time``; the surplus becomes reserved
    slots -- real lattice timestamps, newer than everything else, carrying no
    data until a later append fills them.

    Returns a plan with:
      ``fill``            {timestamp: index} -- reserved slots being filled now
      ``prepend``         {timestamp: index} -- new bins newer than the cube
      ``tail``            {timestamp: index} -- new bins older than the cube
      ``prepend_chunks``  time-chunks to free at the front
      ``length``          final length of the time axis
      ``reserved``        {index: seconds} left reserved afterwards
      ``surplus``         {index: seconds} newly reserved by this call

    Raises if a new timestamp falls *between* stored ones, which cannot be done
    without moving stored data.
    """
    reserved = {int(k): int(v) for k, v in (reserved or {}).items()}
    existing = np.asarray(existing_seconds, dtype="int64")
    new = sorted({int(v) for v in np.asarray(new_seconds, dtype="int64")},
                 reverse=True)
    count = len(existing)

    # reserved slots already carry their timestamp, so a new bin landing on one
    # is a fill, not an insertion
    by_seconds = {seconds: index for index, seconds in reserved.items()}
    fill = {seconds: by_seconds[seconds] for seconds in new
            if seconds in by_seconds}

    outstanding = [seconds for seconds in new if seconds not in by_seconds]
    if count:
        newest, oldest = int(existing.max()), int(existing.min())
        newer = [v for v in outstanding if v > newest]
        older = [v for v in outstanding if v < oldest]
        between = [v for v in outstanding if oldest < v < newest]
        if between:
            raise ValueError(
                f"Cannot append {len(between)} timestep(s) that fall between "
                f"timesteps the cube already holds (e.g. "
                f"{pd.Timestamp(between[0], unit='s')}). Making room for them "
                f"would mean moving stored data, which append does not do. "
                f"Extend the cube at either end instead, or rebuild it with "
                f"overwrite=True.")
    else:
        newer, older = outstanding, []

    chunk_time = int(chunk_time)
    prepend_chunks = -(-len(newer) // chunk_time) if newer else 0
    freed = prepend_chunks * chunk_time
    shift = freed

    # everything that already exists slides up by the freed slots
    plan_reserved = {index + shift: seconds
                     for index, seconds in reserved.items()}
    for seconds, index in list(fill.items()):
        fill[seconds] = index + shift

    # the freed block is filled newest-first: surplus future bins, then the new
    # ones. The surplus sits in front because it is newer still.
    surplus_count = freed - len(newer)
    step = 0
    if surplus_count:
        if len(newer) >= 2:
            step = newer[0] - newer[1]
        elif count >= 2:
            ordered = np.sort(existing)[::-1]
            step = int(ordered[0] - ordered[1])
        if step <= 0:
            raise ValueError(
                "Cannot work out the timestep spacing needed to reserve "
                f"{surplus_count} slot(s) while prepending; the cube has too "
                "few timesteps to infer it.")
    surplus = {
        index: newer[0] + (surplus_count - index) * step
        for index in range(surplus_count)
    }
    prepend = {seconds: surplus_count + offset
               for offset, seconds in enumerate(newer)}

    length = freed + count + len(older)
    tail = {seconds: freed + count + offset
            for offset, seconds in enumerate(older)}

    plan_reserved.update(surplus)
    for seconds in fill:
        plan_reserved.pop(fill[seconds], None)

    return {
        "fill": fill,
        "prepend": prepend,
        "tail": tail,
        "prepend_chunks": prepend_chunks,
        "length": length,
        "reserved": plan_reserved,
        "surplus": surplus,
        "shift": shift,
    }


def prepend_time_chunks(store_path: str, chunks: int, chunk_count: int) -> None:
    """Free ``chunks`` time-chunks at the front by renaming chunk keys.

    A zarr chunk is addressed by its key, so shifting every time-chunk index up
    by ``chunks`` moves the whole cube along the time axis without reading or
    rewriting a single byte of it. Renames go from the highest index down so a
    destination is never an index still waiting to be moved.
    """
    if chunks <= 0:
        return
    chunk_root = os.path.join(store_path, "sentle", "c")
    if not os.path.isdir(chunk_root):
        return
    for index in range(chunk_count - 1, -1, -1):
        source = os.path.join(chunk_root, str(index))
        if not os.path.exists(source):
            continue
        os.rename(source, os.path.join(chunk_root, str(index + chunks)))


def _unprepend_time_chunks(store_path: str, chunks: int,
                           chunk_count: int) -> None:
    """Undo :func:`prepend_time_chunks`, lowest index first."""
    if chunks <= 0:
        return
    chunk_root = os.path.join(store_path, "sentle", "c")
    if not os.path.isdir(chunk_root):
        return
    for index in range(chunk_count):
        source = os.path.join(chunk_root, str(index + chunks))
        if not os.path.exists(source):
            # already moved back by an earlier, interrupted undo
            continue
        destination = os.path.join(chunk_root, str(index))
        if os.path.exists(destination):
            # the freed block only ever holds output from the run being undone,
            # so whatever is sitting in the way is discardable by definition
            shutil.rmtree(destination)
        os.rename(source, destination)


def apply_extension(root, store, zarr_store, plan: dict, order: str,
                    reconsolidate: bool) -> None:
    """Make room for the planned timesteps and write the time coordinate.

    Existing data is never read or rewritten: the front block is freed by
    renaming time-chunk keys, and only the (metadata-sized) time coordinate is
    rewritten. The manifest is marked before anything moves so a hard kill
    leaves a cube the next append repairs.
    """
    data = root["sentle"]
    time = root["time"]
    count = int(time.shape[0])
    existing_seconds = np.asarray(time[:], dtype="int64")
    chunk_time = int(data.chunks[0])
    chunk_count = -(-count // chunk_time)

    store_path = local_store_path(zarr_store)
    if plan["prepend_chunks"] and store_path is None:
        raise ValueError(
            "Prepending timesteps renames chunk keys, which sentle only does "
            "on a local zarr store. Append to a local copy and upload it, or "
            "pass a filesystem path as zarr_store.")

    _update_store_config(root,
                         time_committed=count,
                         time_order=order,
                         append_in_progress={
                             "committed": count,
                             "pending": plan["length"],
                             "order": order,
                             "prepend_chunks": plan["prepend_chunks"],
                             "chunk_count": chunk_count,
                             "shift": plan["shift"],
                         })

    data.resize((plan["length"], ) + tuple(data.shape[1:]))
    time.resize((plan["length"], ))

    prepend_time_chunks(store_path, plan["prepend_chunks"], chunk_count)

    merged = np.zeros(plan["length"], dtype="int64")
    merged[plan["shift"]:plan["shift"] + count] = existing_seconds
    for seconds, index in plan["prepend"].items():
        merged[index] = seconds
    for index, seconds in plan["surplus"].items():
        merged[index] = seconds
    for seconds, index in plan["tail"].items():
        merged[index] = seconds
    time[:] = merged

    _update_store_config(root,
                         reserved_slots={
                             str(index): int(seconds)
                             for index, seconds in plan["reserved"].items()
                         },
                         append_in_progress={
                             "committed": count,
                             "pending": plan["length"],
                             "order": order,
                             "prepend_chunks": plan["prepend_chunks"],
                             "chunk_count": chunk_count,
                             "shift": plan["shift"],
                             "time_merged": True,
                         })

    if reconsolidate:
        # the workers re-open the store per tile and would otherwise read the
        # pre-resize shape from the consolidated metadata and fail out of bounds
        consolidate(store)


def _undo_extension(root, zarr_store, committed: int,
                    in_progress: dict | None) -> None:
    """Put the cube back the way it was before an interrupted extension."""
    in_progress = in_progress or {}
    chunks = int(in_progress.get("prepend_chunks") or 0)
    chunk_count = int(in_progress.get("chunk_count") or 0)
    shift = int(in_progress.get("shift") or 0)
    if not chunks:
        return
    store_path = local_store_path(zarr_store)
    if store_path is None:
        raise ValueError(
            "Cannot undo an interrupted prepend on a non-local zarr store.")

    time = root["time"]
    seconds = np.asarray(time[:], dtype="int64")
    _unprepend_time_chunks(store_path, chunks, chunk_count)
    if in_progress.get("time_merged"):
        time[:committed] = seconds[shift:shift + committed]


def repair_time_order(zarr_store, consolidate_metadata: bool = True) -> bool:
    """Sort the time axis of a cube whose timesteps are out of order.

    Cubes written by sentle always keep their time axis sorted. This repairs
    one that does not -- a cube appended to by a sentle version that grew the
    axis at the end regardless of where the new timestamps belonged. Returns
    True if the cube was reordered, False if it was already sorted.

    The reordered data is staged in a sibling array and swapped in only once it
    is complete, so an interrupted repair leaves the original untouched.
    """
    store, root, was_consolidated = open_cube_for_append(zarr_store)
    reconsolidate = was_consolidated or consolidate_metadata
    try:
        seconds = np.asarray(root["time"][:], dtype="int64")
        if time_axis_order(seconds) is not None:
            return False

        config = read_store_config(root) or {}
        order = config.get("time_order", "descending")
        permutation = np.argsort(seconds, kind="stable")
        if order == "descending":
            permutation = permutation[::-1]

        data = root["sentle"]
        staging_path = "sentle__resort"
        if staging_path in root:
            del root[staging_path]
        staging = zarr.create(
            shape=data.shape,
            chunks=data.chunks,
            dtype=data.dtype,
            fill_value=data.fill_value,
            store=store,
            path=f"/{staging_path}",
            overwrite=True,
            config=dict(write_empty_chunks=False),
            dimension_names=["time", "band", "y", "x"],
        )
        for new_index, old_index in enumerate(permutation):
            staging[new_index] = data[int(old_index)]

        for key, value in dict(data.attrs).items():
            staging.attrs[key] = value

        store_root = str(zarr_store) if isinstance(zarr_store, str) else None
        if store_root is None:
            raise ValueError(
                "repair_time_order currently supports local zarr stores only")

        # swap by renaming, never by deleting first: two atomic renames leave
        # at most a microsecond window, whereas removing the array up front
        # would lose the cube outright if the process died right there
        live = os.path.join(store_root, "sentle")
        staged = os.path.join(store_root, staging_path)
        superseded = os.path.join(store_root, "sentle__superseded")
        root["time"][:] = seconds[permutation]
        os.rename(live, superseded)
        os.rename(staged, live)
        shutil.rmtree(superseded, ignore_errors=True)
        _update_store_config(root, time_order=order)
        if reconsolidate:
            consolidate(store)
        return True
    finally:
        store.close()


def commit_append(root, store, length: int, reconsolidate: bool) -> None:
    """Mark ``length`` timesteps as fully written."""
    _update_store_config(root, time_committed=int(length),
                         append_in_progress=None)
    if reconsolidate:
        consolidate(store)


def rollback_append(root, store, zarr_store, committed: int,
                    reconsolidate: bool, recomputed_indices=None) -> None:
    """Undo an append that failed, returning the cube to ``committed``.

    Timesteps that were being *recomputed* sit below ``committed`` and are not
    truncated. They were cleared before the run started, so they cannot be
    restored here -- say so loudly rather than leave the caller to discover
    empty timesteps later.
    """
    config = read_store_config(root) or {}
    _undo_extension(root, zarr_store, int(committed),
                    config.get("append_in_progress"))
    _truncate_time_axis(root, committed)
    _update_store_config(root, time_committed=int(committed),
                         reserved_slots=config.get("reserved_slots_before",
                                                   {}) or {},
                         append_in_progress=None)
    if reconsolidate:
        consolidate(store)

    if recomputed_indices:
        times = np.asarray(root["time"][:], dtype="int64")
        dates = ", ".join(
            str(pd.Timestamp(int(times[i]), unit="s").date())
            for i in sorted(recomputed_indices))
        warnings.warn(
            f"The append failed after clearing {len(recomputed_indices)} "
            f"timestep(s) for recomputation ({dates}); those timesteps are now "
            f"NoData and their previous contents are gone. Re-run the append "
            f"with append_recompute_trailing set to restore them.")
