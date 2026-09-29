import itertools
import math
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import threading
import warnings

import numpy as np
import pandas as pd
import rasterio
import scipy.ndimage as sc
from rasterio import features, transform, warp, windows
from rasterio.crs import CRS
from rasterio.enums import Resampling
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from .cloud_mask import S2_cloud_mask_band, S2_cloud_prob_bands, worker_get_cloud_mask
from .const import (
    DEFAULT_READ_RETRIES,
    S2_FOOTPRINT_BUFFER_DEG,
    S2_HOLE_FRINGE_PX,
    S2_NBAR_BANDS,
    S2_RAW_BAND_RESOLUTION,
    S2_RAW_BANDS,
    S2_READ_THREADS,
    S2_subtile_size,
)
from .nbar import get_c_factor_value
from .reproject_util import (
    bounds_from_transform_height_width_res,
    calculate_aligned_transform,
    recrop_write_window,
    reproject_nodata_zero,
    window_overlaps_bounds,
)
from .snow_mask import S2_snow_mask_band, compute_potential_snow_layer
from .stac import PlanetaryComputerProvider, retry_read


def obtain_subtiles(target_crs: CRS, left: float, bottom: float, right: float,
                    top: float, s2grid, include_redundant: bool = False):
    """Retrieves the sentinel subtiles that intersect the with the specified
    bounds. The bounds are interpreted based on the given target_crs.

    Subtiles that higher-priority tiles fully cover are redundant and left
    out. With ``include_redundant=True`` they are returned too, flagged in the
    ``redundant`` column: ``process_ptile_S2`` reads them only to fill pixels
    that no other subtile has data for in an acquisition (a higher-priority
    tile without an item, or with NoData there).
    """

    # TODO make it possible to not only use naive bounds but also MultiPolygons

    # check if supplied sub_tile_width makes sense
    assert (S2_subtile_size >= 16) and (
        S2_subtile_size
        <= 10980), "S2_subtile_size needs to within 16 and 10980"
    assert (
        10980 %
        S2_subtile_size) == 0, "S2_subtile_size needs to be a divisor of 10980"

    # convert bounds to sentinel grid crs. ``shape`` (rather than
    # ``Polygon(*coordinates)``) is used throughout obtain_subtiles because a
    # geometry that crosses the antimeridian is returned by ``transform_geom``
    # as a (cut) MultiPolygon, which ``Polygon(*...)`` cannot parse -- see
    # issue #60. ``shape`` handles Polygon and MultiPolygon alike, and the
    # downstream intersects/intersection/union/area operations work on both.
    transformed_bounds = shape(
        warp.transform_geom(src_crs=target_crs,
                            dst_crs=s2grid.crs,
                            geom=box(left, bottom, right, top)))

    # extract overlapping sentinel tiles
    s2grid = s2grid[s2grid["geometry"].intersects(transformed_bounds)].copy()

    general_subtile_windows = [
        windows.Window(col_off=col_off,
                       row_off=row_off,
                       width=S2_subtile_size,
                       height=S2_subtile_size)
        for col_off, row_off in itertools.product(
            np.arange(0, 10980, S2_subtile_size),
            np.arange(0, 10980, S2_subtile_size))
    ]

    # reproject s2 footprint to local utm footprint. Transform the whole tile
    # geometry (not just its first part): a tile straddling the antimeridian is
    # stored as a MultiPolygon split at +/-180 in lat/lon, but is contiguous in
    # its local UTM, so using the full geometry yields the correct extent.
    s2grid["s2_footprint_utm"] = s2grid[[
        "geometry", "crs"
    ]].apply(lambda ser: shape(warp.transform_geom(
        src_crs=s2grid.crs, dst_crs=ser["crs"], geom=ser["geometry"])),
             axis=1)

    # obtain transform of each sentinel 2 tile in local utm crs
    s2grid["tile_transform"] = s2grid["s2_footprint_utm"].apply(
        lambda x: transform.from_bounds(*x.bounds, width=10980, height=10980))

    # For each tile, compute the subtile windows intersecting the bounds
    # together with their geographic footprint (in s2grid.crs). The footprint
    # is reused below to eliminate downloads that are redundant because of the
    # Sentinel-2 MGRS tile overlap (see Bauer-Marschallinger & Falkner, 2023:
    # adjacent UTM/MGRS tiles overlap, so a single location is covered by up to
    # 6 tiles -- downloading it from more than one is wasted bandwidth).
    def _intersecting_window_footprints(ser):
        out = []
        for win_subtile in general_subtile_windows:
            footprint = shape(warp.transform_geom(
                src_crs=ser["crs"],
                dst_crs=s2grid.crs,
                geom=box(*windows.bounds(win_subtile, ser["tile_transform"]))))
            if transformed_bounds.intersects(footprint):
                out.append((win_subtile, footprint))
        return out

    s2grid["window_footprints"] = s2grid[["tile_transform", "crs"]].apply(
        _intersecting_window_footprints, axis=1)

    # drop tiles whose windows do not intersect the bounds (edge cases)
    s2grid = s2grid[s2grid["window_footprints"].apply(len) > 0]

    # ------------------------------------------------------------------
    # redundancy elimination across overlapping MGRS tiles
    #
    # We greedily assign geography to tiles, processing the tiles that cover
    # the largest share of the requested area first (this keeps the number of
    # partially-overlapping boundary subtiles minimal). A subtile window is
    # only kept if it contributes geography that is not already covered by a
    # higher-priority tile. This preserves full coverage (every location that
    # is covered by at least one tile stays covered by the highest-priority
    # tile covering it) while never downloading the same location twice.
    # ------------------------------------------------------------------

    # the area each tile contributes within the requested bounds; used both as
    # the priority key and (clipped) as the "claimed" region
    s2grid["aoi_footprint"] = s2grid["window_footprints"].apply(
        lambda wf: unary_union([fp for _, fp in wf]).intersection(
            transformed_bounds))
    s2grid["aoi_area"] = s2grid["aoi_footprint"].apply(lambda g: g.area)

    # process tiles covering the most area first; name as deterministic tiebreak
    s2grid = s2grid.sort_values(["aoi_area", "name"],
                                ascending=[False, True])

    # fraction of a subtile footprint that must be newly covered for the
    # subtile to be worth downloading; small enough to only drop fully
    # redundant subtiles (and numerical slivers), never genuine sub-pixel data
    keep_frac_threshold = 1e-6

    claimed = None
    kept_names = []
    kept_windows = []
    kept_rows = []
    kept_redundant = []
    for row in s2grid.itertuples(index=False):
        for win_subtile, footprint in row.window_footprints:
            if claimed is None:
                keep = True
            else:
                covered_area = footprint.intersection(claimed).area
                keep = (footprint.area -
                        covered_area) > keep_frac_threshold * footprint.area
            if keep or include_redundant:
                kept_names.append(row.name)
                kept_windows.append(win_subtile)
                kept_rows.append(row)
                kept_redundant.append(not keep)
        claimed = (row.aoi_footprint
                   if claimed is None else claimed.union(row.aoi_footprint))

    # each row is one subtile of a sentinel tile to download and process,
    # grouped by tile in priority order. The tile's CRS and 10 m transform
    # ride along for ``tile_read_windows``.
    return pd.DataFrame({
        "name": kept_names,
        "intersecting_windows": kept_windows,
        "redundant": kept_redundant,
        "tile_crs": [r.crs for r in kept_rows],
        "tile_transform": [r.tile_transform for r in kept_rows],
    })


# Radius, in source pixels, of GDAL's warp kernel per resampling method; the
# kernel grows by the downsampling factor. Methods not listed get the largest.
_RESAMPLING_RADIUS = {
    Resampling.nearest: 1,
    Resampling.bilinear: 1,
    Resampling.cubic: 2,
    Resampling.cubic_spline: 2,
    Resampling.lanczos: 3,
}


def tile_read_windows(subtiles, target_crs, bounds, target_resolution,
                      resampling, crop=True):
    """The 10 m pixel window to read per MGRS tile: ``{name: (window, tile
    transform, union)}``, with ``union`` the uncropped window.

    Without ``crop`` that is the union of the tile's ``subtiles``. With
    ``crop`` it is cut down to ``bounds`` (in ``target_crs``) plus the pixels
    the resampling kernel reaches from inside it. Everything left out would
    only have been reprojected onto pixels outside ``bounds``, so the result
    within ``bounds`` is unchanged -- the parts of the subtiles outside the
    window are read as 0 (NoData), exactly what GDAL sees beyond a tile edge.
    Windows are aligned to 6 px so they are whole pixels in the 20 m and 60 m
    bands too.
    """
    area = box(*bounds)
    area = area.segmentize(max(area.length / 400, 1e-9))
    radius = _RESAMPLING_RADIUS.get(resampling, 3)
    out = {}
    for name, group in subtiles.groupby("name", sort=False):
        union = windows.union(*group["intersecting_windows"])
        tile = group.iloc[0]
        if not crop:
            out[name] = (union, tile.tile_transform, union)
            continue
        local = shape(warp.transform_geom(target_crs, tile.tile_crs,
                                          mapping(area)))
        float_win = windows.from_bounds(*local.bounds,
                                        transform=tile.tile_transform)
        # source pixels per target pixel: the kernel scales with it
        scale = max(
            1.0, float_win.width / ((bounds[2] - bounds[0]) / target_resolution),
            float_win.height / ((bounds[3] - bounds[1]) / target_resolution))
        # +2: the chunk outline is only densified, and the grid's pixel
        # centres need not line up with the target's
        margin = math.ceil((radius + 1) * scale) + 2
        col0 = (math.floor(float_win.col_off) - margin) // 6 * 6
        row0 = (math.floor(float_win.row_off) - margin) // 6 * 6
        col1 = -(-(math.ceil(float_win.col_off + float_win.width) + margin)
                 // 6) * 6
        row1 = -(-(math.ceil(float_win.row_off + float_win.height) + margin)
                 // 6) * 6
        win = windows.Window(col0, row0, col1 - col0, row1 - row0)
        if windows.intersect(win, union):
            out[name] = (win.intersection(union), tile.tile_transform, union)
    return out


def read_tile_window(href, factor, tile_window):
    """Read ``tile_window`` (see ``tile_read_windows``) of one band file:
    ``(data, native window, crs, transform)``, in the band's own resolution
    and dtype."""
    window, grid_transform, union = tile_window
    dr = rasterio.open(href)
    try:
        # the window was placed with the tile grid's transform; should the
        # file's grid ever differ, read everything rather than guess
        if grid_transform is not None and (
                abs(dr.transform.c - grid_transform.c) > 1
                or abs(dr.transform.f - grid_transform.f) > 1):
            window = union
        native = windows.Window(window.col_off // factor,
                                window.row_off // factor,
                                window.width // factor,
                                window.height // factor)
        return (dr.read(indexes=1, window=native), native, dr.crs,
                dr.transform)
    finally:
        dr.close()


def prefetch_tile_windows(stac_item, bands, tile_window, provider,
                          window_cache, read_retries):
    """Read every band window of one tile into ``window_cache``,
    ``S2_READ_THREADS`` at a time. Same requests as reading them one by one
    from ``process_S2_subtile``, without waiting on each round trip in turn.
    A read that still fails after its retries raises, like it would there.

    The provider's GDAL environment is entered here, once, and not in the
    reading threads: rasterio applies an ``Env`` process-wide only from the
    main thread and thread-locally from any other, and GDAL reads with threads
    of its own (the JP2 decoder does) that would not see thread-local options
    -- for CDSE they then fell back to the default AWS profile and got 403. So
    the reads only run concurrently when this is the main thread (as in a
    worker process), and one by one otherwise.
    """

    def fetch(band):
        asset_href = stac_item.assets[provider.s2_asset_key(band)].href
        if asset_href in window_cache:
            return
        factor = S2_RAW_BAND_RESOLUTION[band] // 10
        window_cache[asset_href] = retry_read(
            lambda: read_tile_window(provider.prepare_href(asset_href), factor,
                                     tile_window),
            f"asset={asset_href} band={band}", retries=read_retries)

    threads = (S2_READ_THREADS
               if threading.current_thread() is threading.main_thread() else 1)
    with provider.rasterio_env(), ThreadPoolExecutor(threads) as pool:
        # list() re-raises the first failure
        list(pool.map(fetch, bands))


def read_subtile_from_tile_window(href, read_window, factor, tile_window,
                                  window_cache, key=None):
    """Return one subtile band exactly as a direct windowed read of it would,
    served from a single read per band of ``tile_window`` (see
    ``tile_read_windows``).

    Reading every subtile on its own costs an HTTP request per COG block row
    per read, and blocks shared by neighbouring subtiles are fetched again. One
    read of the tile's window lets GDAL fetch each block once, one request per
    block row (see ``_GDAL_READ_OPTIONS`` in ``stac.py``). The window is kept
    in the band's native dtype and resolution and upsampled per subtile with
    ``np.repeat`` -- for an integer factor that is the pixel GDAL's
    nearest-neighbour ``out_shape`` read picks. Parts of the subtile outside
    the window are 0 (NoData).

    ``read_window`` is in the band's own pixel grid, like the direct read.
    ``window_cache`` holds the reads by ``key`` (default ``href``); pass the
    unsigned href so a re-signed URL does not read the window again.
    """
    key = href if key is None else key
    entry = window_cache.get(key)
    if entry is None:
        entry = window_cache[key] = read_tile_window(href, factor,
                                                     tile_window)
    data, native, crs, tf = entry

    subtile = np.zeros((int(read_window.height), int(read_window.width)),
                       dtype=np.float32)
    if windows.intersect(read_window, native):
        part = read_window.intersection(native)
        src_row = int(part.row_off - native.row_off)
        src_col = int(part.col_off - native.col_off)
        dst_row = int(part.row_off - read_window.row_off)
        dst_col = int(part.col_off - read_window.col_off)
        h, w = int(part.height), int(part.width)
        subtile[dst_row:dst_row + h, dst_col:dst_col + w] = \
            data[src_row:src_row + h, src_col:src_col + w]
    if factor > 1:
        subtile = subtile.repeat(factor, axis=0).repeat(factor, axis=1)
    return subtile, crs, tf


def process_S2_subtile(
    intersecting_windows,
    stac_item,
    timestamp,
    target_crs: CRS,
    target_resolution: float,
    ptile_transform,
    ptile_width: int,
    ptile_height: int,
    S2_mask_snow: bool,
    S2_cloud_classification: bool,
    S2_cloud_classification_device: str,
    S2_nbar: bool,
    cloud_request_queue: mp.Queue,
    cloud_response_queue: mp.Queue,
    resampling_method: Resampling,
    S2_bands: list = None,
    provider=None,
    read_retries: int = DEFAULT_READ_RETRIES,
    tile_window=None,
    window_cache=None,
):
    """Processes a single sentinel 2 subtile. This includes downloading the
    data, reprojecting it to the target_crs and target_resolution, applying
    cloud and snow masks and computing NBAR if requested. The function returns
    the reprojected subtile, the write window and the band names of the
    reprojected subtile.

    ``S2_bands`` selects which raw reflectance bands to download (a subset of
    ``S2_RAW_BANDS``); defaults to all of them. Cloud classification always
    requires the full set (enforced upstream). ``provider`` supplies the
    catalog-specific asset keys / href preparation / GDAL env (default:
    Planetary Computer).

    ``tile_window`` is this tile's entry of ``tile_read_windows``: bands are
    read once per ptile over it and cached in ``window_cache``, see
    ``read_subtile_from_tile_window``.
    """
    if provider is None:
        provider = PlanetaryComputerProvider()
    if tile_window is None:
        # on its own: read just this subtile
        tile_window = (intersecting_windows, None, intersecting_windows)
    if window_cache is None:
        window_cache = {}

    # which raw bands to actually download for this subtile
    if S2_bands is None:
        S2_bands = S2_RAW_BANDS
    download_bands = list(S2_bands)

    # init array that needs to be filled. Zero-filled rather than np.empty: 0 is
    # sentle's NoData sentinel everywhere downstream, so a band whose download
    # fails below stays NoData instead of carrying uninitialised heap memory
    # into the cloud classifier and the zarr store (issue #87).
    subtile_array = np.zeros(
        (len(download_bands), S2_subtile_size, S2_subtile_size),
        dtype=np.float32)
    band_names = download_bands.copy()

    # save CRS of downloaded sentinel tiles
    s2_crs = None
    # save transformation of sentinel tile for later processing
    s2_tile_transform = None
    # per-scene processing baseline decides whether harmonization is applied
    apply_harmonization = provider.s2_processing_baseline(stac_item) >= 4.0

    # retrieve each band for the subtile, within the provider's GDAL
    # environment (e.g. CDSE S3 credentials). Each band of the tile is read
    # once per ptile, over ``tile_window``, and the subtile is sliced out of
    # that read (``read_subtile_from_tile_window``).
    with provider.rasterio_env():
        for i, band in enumerate(download_bands):
            asset_href = stac_item.assets[provider.s2_asset_key(band)].href
            factor = S2_RAW_BAND_RESOLUTION[band] // 10
            orig_win = intersecting_windows
            # convert read window respective to tile resolution
            # (lower resolution -> fewer pixels for same area)
            read_window = windows.Window(orig_win.col_off // factor,
                                         orig_win.row_off // factor,
                                         orig_win.width // factor,
                                         orig_win.height // factor)

            def attempt(asset_href=asset_href, read_window=read_window,
                        factor=factor):
                # re-signed on every attempt: an expired SAS token is a
                # plausible cause of a failed read, and re-signing is cheap
                return read_subtile_from_tile_window(
                    provider.prepare_href(asset_href), read_window, factor,
                    tile_window, window_cache, key=asset_href)

            read_data, band_crs, band_transform = retry_read(
                attempt,
                f"asset={asset_href} band={band}",
                retries=read_retries)

            # harmonization
            if apply_harmonization:
                # clip values to minimum 1000; done here instead of
                # clipping to zero later to avoid integer underflow
                # when using a uint16 potentially later on
                read_data[read_data < 1000] = 1000
                # adjust reflectance for non-zero values
                read_data[read_data != 0] -= 1000

            # save
            subtile_array[i] = read_data

            # save and validate epsg
            assert (s2_crs is None) or (
                s2_crs == band_crs), "CRS mismatch within one sentinel tile"
            s2_crs = band_crs

            # save the transform of a 10m band tile. Keyed on the resolution,
            # not on B02: an S2_bands subset without B02 used to leave this None
            # and silently drop every subtile, i.e. write an empty cube.
            if s2_tile_transform is None and factor == 1:
                s2_tile_transform = band_transform

    # in this case we have no data for this subtile, or the tile has no CRS
    if s2_tile_transform is None or not s2_crs:
        return None, None, None

    # determine bounds based on subtile window and tile transform
    subtile_bounds_utm = windows.bounds(intersecting_windows,
                                        s2_tile_transform)
    assert (
        subtile_bounds_utm[2] - subtile_bounds_utm[0]
    ) // 10 == S2_subtile_size, "mismatch between subtile size and bounds on x-axis"
    assert (
        subtile_bounds_utm[3] - subtile_bounds_utm[1]
    ) // 10 == S2_subtile_size, "mismatch between subtile size and bounds on y-axis"

    if S2_cloud_classification:
        # this waits for the cloud mask to be computed in the service
        result_probs = worker_get_cloud_mask(
            array=subtile_array,
            request_queue=cloud_request_queue,
            response_queue=cloud_response_queue)
        band_names += S2_cloud_prob_bands
        subtile_array = np.concatenate([subtile_array, result_probs])

    if S2_nbar:
        # needs to happen at a per-item level after after clouds were detected.
        # NBAR relies on the per-scene granule metadata, which is occasionally
        # missing/unreadable for some scenes (see issue #59); in that case warn
        # and continue with un-corrected reflectance rather than aborting the
        # whole (potentially multi-hour) job.
        try:
            c = get_c_factor_value(stac_item, s2_crs, subtile_bounds_utm)

            # apply c-factor to array; indices are relative to the downloaded
            # band subset (which is guaranteed to contain every NBAR band
            # upstream) and kept in NBAR-band order so they line up with the
            # c-factor bands
            nbar_indices = [download_bands.index(b) for b in S2_NBAR_BANDS]
            subtile_array[nbar_indices] *= c
        except Exception as e:
            warnings.warn(
                f"nbar_failure item={stac_item.id} "
                f"exception_type={type(e).__name__} message={e} "
                f"note=skipping_NBAR_for_this_subtile")

    # 3 reproject to target_crs for each band
    # determine transform --> round to target resolution so that reprojected
    # subtiles align across subtiles
    subtile_repr_transform, subtile_repr_height, subtile_repr_width = calculate_aligned_transform(
        src_crs=s2_crs,
        dst_crs=target_crs,
        width=subtile_array.shape[1],
        height=subtile_array.shape[0],
        left=subtile_bounds_utm[0],
        bottom=subtile_bounds_utm[1],
        right=subtile_bounds_utm[2],
        top=subtile_bounds_utm[3],
        tres=target_resolution)

    # billinear reprojection for everything
    subtile_array_repr = np.empty(
        (len(band_names), subtile_repr_height, subtile_repr_width),
        dtype=np.float32)
    reproject_nodata_zero(source=subtile_array,
                          destination=subtile_array_repr,
                          src_transform=transform.from_bounds(
                              *subtile_bounds_utm,
                              width=S2_subtile_size,
                              height=S2_subtile_size),
                          src_crs=s2_crs,
                          dst_crs=target_crs,
                          dst_transform=subtile_repr_transform,
                          resampling=resampling_method)
    # explicit clear
    del subtile_array

    # compute bounds in target crs based on rounded transform
    subtile_bounds_tcrs = bounds_from_transform_height_width_res(
        tf=subtile_repr_transform,
        height=subtile_repr_height,
        width=subtile_repr_width,
        resolution=target_resolution)

    # figure out where to write the subtile within the overall bounds
    write_win = windows.from_bounds(
        *subtile_bounds_tcrs,
        transform=ptile_transform).round_offsets().round_lengths()

    if not window_overlaps_bounds(write_win, ptile_height, ptile_width):
        return None, None, None

    write_win, local_win = recrop_write_window(write_win, ptile_height,
                                               ptile_width)

    # crop subtile_array based on computed local win because it could overlap
    # with the overall bounds
    subtile_array_repr = subtile_array_repr[:, local_win.
                                            row_off:local_win.height +
                                            local_win.row_off, local_win.
                                            col_off:local_win.col_off +
                                            local_win.width]

    return subtile_array_repr, write_win, band_names


def process_ptile_S2_dispatcher(
    target_crs: CRS,
    target_resolution: float,
    S2_cloud_classification_device: str,
    time_composite_freq: str,
    S2_apply_snow_mask: bool,
    S2_apply_cloud_mask: bool,
    S2_bands_to_save,
    ptile_height,
    ptile_width,
    ptile_transform,
    item_list,
    ts,
    bound_left,
    bound_right,
    bound_bottom,
    bound_top,
    S2_mask_snow: bool,
    S2_cloud_classification: bool,
    S2_return_cloud_probabilities: bool,
    S2_nbar: bool,
    S2_subtiles,
    cloud_request_queue: mp.Queue,
    cloud_response_queue: mp.Queue,
    resampling_method: Resampling,
    S2_bands: list = None,
    time_composite_method: str = "mean",
    provider=None,
    read_retries: int = DEFAULT_READ_RETRIES,
):
    if provider is None:
        provider = PlanetaryComputerProvider()

    # the raw reflectance bands to download/save (subset of S2_RAW_BANDS)
    if S2_bands is None:
        S2_bands = S2_RAW_BANDS
    S2_bands = list(S2_bands)

    items = pd.DataFrame()
    items["item"] = item_list
    items["tile"] = items["item"].apply(provider.s2_mgrs_tile)
    items["ts"] = items["item"].apply(lambda x: x.datetime)

    # intiate one array representing the entire subtile for that timestamp
    ptile_array = np.full(shape=(len(S2_bands_to_save), ptile_height,
                                 ptile_width),
                          fill_value=0,
                          dtype=np.float32)

    # also dont need to perform aggreation if we only have one item
    perform_aggregation = (time_composite_freq is not None) and (len(item_list)
                                                                 > 1)

    # "mean" aggregates by streaming sum/count (memory-cheap); the other
    # methods need the individual acquisitions, so they are buffered and
    # reduced at the end. Only a handful of acquisitions fall in one composite
    # window, so the buffer stays small.
    use_buffer = perform_aggregation and time_composite_method != "mean"
    buffered_stamps = [] if use_buffer else None

    if perform_aggregation and not use_buffer:
        # count how many values we add per pixel to compute mean later
        ptile_array_count = np.full(shape=(len(S2_bands_to_save), ptile_height,
                                           ptile_width),
                                    fill_value=0,
                                    dtype=np.uint8)

    ptile_array_bands = None
    timestamps_it = items["ts"].drop_duplicates().tolist()
    for ts in timestamps_it:
        ptile_timestamp, ret_bands = process_ptile_S2(
            timestamp=ts,
            target_crs=target_crs,
            target_resolution=target_resolution,
            S2_cloud_classification=S2_cloud_classification,
            S2_cloud_classification_device=S2_cloud_classification_device,
            S2_mask_snow=S2_mask_snow,
            S2_return_cloud_probabilities=S2_return_cloud_probabilities,
            S2_nbar=S2_nbar,
            subtiles=S2_subtiles,
            ptile_transform=ptile_transform,
            ptile_width=ptile_width,
            ptile_height=ptile_height,
            items=items[items["ts"] == ts],
            cloud_request_queue=cloud_request_queue,
            cloud_response_queue=cloud_response_queue,
            resampling_method=resampling_method,
            S2_bands=S2_bands,
            provider=provider,
            read_retries=read_retries,
        )

        # this happens when the href is not available in subtile -> planetary
        # computer issue
        if ptile_timestamp is None:
            continue

        # only assign the sentinel/band accumulator for valid timestamps, so a
        # last acquisition returning None cannot clobber it and discard the
        # composite assembled from earlier valid acquisitions
        ptile_array_bands = ret_bands

        # replace nans with zero, to that sum works properly
        ptile_timestamp = np.nan_to_num(ptile_timestamp, 0)

        # apply masks and drop classification layers if doing temporal aggregation
        if S2_apply_snow_mask:
            snow_index = ptile_array_bands.index(S2_snow_mask_band)
            ptile_timestamp *= ptile_timestamp[snow_index]

            if time_composite_freq is not None:
                ptile_timestamp = np.delete(ptile_timestamp,
                                            snow_index,
                                            axis=0)

        if S2_apply_cloud_mask:
            cloud_index = ptile_array_bands.index(S2_cloud_mask_band)
            ptile_timestamp *= (ptile_timestamp[cloud_index] == 0)

            if time_composite_freq is not None:
                ptile_timestamp = np.delete(ptile_timestamp,
                                            cloud_index,
                                            axis=0)

        # save new data
        if use_buffer:
            # keep this acquisition; encode nodata/masked (0) as NaN so the
            # nan-aware reducer ignores it per band and pixel
            stamp = ptile_timestamp.astype(np.float32, copy=True)
            stamp[stamp == 0] = np.nan
            buffered_stamps.append(stamp)
        else:
            ptile_array += ptile_timestamp

            if perform_aggregation:
                # count where we added data
                ptile_array_count += ptile_timestamp != 0

    if ptile_array_bands is None:
        return None

    if time_composite_freq is not None:
        if S2_snow_mask_band in ptile_array_bands:
            ptile_array_bands.remove(S2_snow_mask_band)
        if S2_cloud_mask_band in ptile_array_bands:
            ptile_array_bands.remove(S2_cloud_mask_band)

    if use_buffer:
        # reduce the buffered acquisitions with the requested method, ignoring
        # NoData (NaN); then restore the 0-based NoData sentinel
        reducer = {
            "median": np.nanmedian,
            "min": np.nanmin,
            "max": np.nanmax,
        }[time_composite_method]
        with warnings.catch_warnings():
            # an all-NoData pixel reduces to NaN -> expected, handled below
            warnings.simplefilter("ignore")
            ptile_array = reducer(np.stack(buffered_stamps, axis=0),
                                  axis=0).astype(np.float32)
        ptile_array = np.nan_to_num(ptile_array, nan=0.0)
    elif perform_aggregation:
        # compute mean based on sum and count for each pixel
        with warnings.catch_warnings():
            # filter out divide by zero warning, this is expected here
            warnings.simplefilter("ignore")
            ptile_array /= ptile_array_count

    # ... and set all such pixels to nan (of which some are already nan because
    # of divide by zero)
    # determine nodata mask based on where values are zero -> mean nodata for S2...
    # (need to do this here, because after computing mean there will be nans
    # from divide by zero)
    # only the raw reflectance bands actually present drive the NoData mask
    raw_bands_present = [b for b in ptile_array_bands if b in S2_RAW_BANDS]
    ptile_array[:,
                np.any(ptile_array[
                    [ptile_array_bands.index(band)
                     for band in raw_bands_present]] == 0,
                       axis=0)] = np.nan

    return ptile_array


def hole_mask(raw_count):
    """Pixels to fill, from the per-band count of values (bands x rows x
    cols): those without any band, and within ``S2_HOLE_FRINGE_PX`` of one
    those that miss some band."""
    empty = ~raw_count.any(axis=0)
    if not empty.any():
        return empty
    near = sc.maximum_filter(empty, size=2 * S2_HOLE_FRINGE_PX + 1)
    return near & ~raw_count.all(axis=0)


def claimed_by_contributors(subtiles, items, target_crs, ptile_transform,
                            shape_):
    """Pixels of the ptile (``shape_``) that a tile which contributed to it
    claims to have data for: inside both one of its non-redundant
    ``subtiles`` and its item's footprint, shrunk by
    ``S2_FOOTPRINT_BUFFER_DEG``.

    A pixel there without data is a gap in the scene itself (L2A NoData for a
    defective or saturated pixel). Overlapping tiles of one acquisition are
    processed from the same L1C data and have the same gap, so reading them
    for it would be wasted. Swath edges and missing tiles lie outside every
    such area.
    """
    first_item = {}
    for tile, item in zip(items["tile"], items["item"]):
        first_item.setdefault(tile, item)
    claims = []
    for name, group in subtiles.groupby("name", sort=False):
        item = first_item.get(name)
        if item is None or item.geometry is None:
            continue
        inner = shape(item.geometry).buffer(-S2_FOOTPRINT_BUFFER_DEG)
        if inner.is_empty:
            continue
        tile = group.iloc[0]
        covered = unary_union([
            shape(warp.transform_geom(
                tile.tile_crs, "EPSG:4326",
                mapping(box(*windows.bounds(w, tile.tile_transform)).segmentize(
                    1000)))) for w in group["intersecting_windows"]
        ])
        claim = inner.intersection(covered)
        if not claim.is_empty:
            claims.append(warp.transform_geom("EPSG:4326", target_crs,
                                              mapping(claim)))
    if not claims:
        return np.zeros(shape_, dtype=bool)
    return features.geometry_mask(claims, out_shape=shape_,
                                  transform=ptile_transform, invert=True)


def _hole_filling_subtiles(redundant, items, hole, target_crs,
                           ptile_transform, provider):
    """The redundant subtiles worth reading to fill ``hole`` (a ptile-shaped
    mask of pixels without data), and the target-CRS bounds of the holes they
    cover; None if there are none.

    A candidate needs an item in this acquisition, has to reproject onto a
    hole, and its item's footprint (grown by ``S2_FOOTPRINT_BUFFER_DEG``) has
    to reach that hole -- otherwise it has nothing to add there.
    """
    if redundant.empty or not hole.any():
        return None
    first_item = {}
    for tile, item in zip(items["tile"], items["item"]):
        first_item.setdefault(tile, item)
    footprints = {}
    keep = []
    boxes = []
    height, width = hole.shape
    for idx, st in enumerate(redundant.itertuples(index=False)):
        item = first_item.get(st.name)
        if item is None:
            continue
        outline = warp.transform_geom(
            st.tile_crs, target_crs,
            mapping(box(*windows.bounds(st.intersecting_windows,
                                        st.tile_transform)).segmentize(1000)))
        win = windows.from_bounds(*shape(outline).bounds,
                                  transform=ptile_transform)
        r0 = max(int(math.floor(win.row_off)), 0)
        c0 = max(int(math.floor(win.col_off)), 0)
        r1 = min(int(math.ceil(win.row_off + win.height)), height)
        c1 = min(int(math.ceil(win.col_off + win.width)), width)
        if r0 >= r1 or c0 >= c1:
            continue
        rows, cols = np.nonzero(hole[r0:r1, c0:c1])
        if rows.size == 0:
            continue
        hole_box = windows.bounds(
            windows.Window(c0 + cols.min(), r0 + rows.min(),
                           cols.max() - cols.min() + 1,
                           rows.max() - rows.min() + 1), ptile_transform)
        if item.geometry is not None:
            if st.name not in footprints:
                footprints[st.name] = shape(item.geometry).buffer(
                    S2_FOOTPRINT_BUFFER_DEG)
            hole_lonlat = shape(warp.transform_geom(
                target_crs, "EPSG:4326", mapping(box(*hole_box))))
            if not footprints[st.name].intersects(hole_lonlat):
                continue
        keep.append(idx)
        boxes.append(hole_box)
    if not keep:
        return None
    bounds = (min(b[0] for b in boxes), min(b[1] for b in boxes),
              max(b[2] for b in boxes), max(b[3] for b in boxes))
    return redundant.iloc[keep], bounds


def process_ptile_S2(
    timestamp,
    target_crs: CRS,
    target_resolution: float,
    S2_cloud_classification_device: str,
    subtiles,
    ptile_transform,
    ptile_height,
    ptile_width,
    cloud_request_queue,
    cloud_response_queue,
    items,
    S2_mask_snow: bool,
    S2_cloud_classification: bool,
    S2_return_cloud_probabilities: bool,
    S2_nbar: bool,
    resampling_method: Resampling,
    S2_bands: list = None,
    provider=None,
    read_retries: int = DEFAULT_READ_RETRIES,
):
    """One acquisition of one ptile: the mean of every subtile that has data.

    The subtiles that are not ``redundant`` (see ``obtain_subtiles``) are
    reprojected and averaged. Each band of a tile is read once, cropped to the
    ptile (see ``tile_read_windows``) unless the cloud model runs, which needs
    whole subtiles. Pixels no subtile has data for are then filled from the
    redundant subtiles of tiles that do have an item in this acquisition --
    those are read only for such holes, and only where the item's footprint
    can have data.
    """
    if provider is None:
        provider = PlanetaryComputerProvider()

    if S2_bands is None:
        S2_bands = S2_RAW_BANDS
    S2_bands = list(S2_bands)

    num_bands = len(S2_bands)

    # add cloud probability bands if requested
    if S2_cloud_classification:
        num_bands += 4

    def accumulate(subset, bounds):
        """Sum and count of the non-zero values ``subset`` reprojects onto the
        ptile, and the band names (None if nothing was read)."""
        total = np.zeros((num_bands, ptile_height, ptile_width),
                         dtype=np.float32)
        count = np.zeros((num_bands, ptile_height, ptile_width),
                         dtype=np.uint8)
        band_names = None
        tile_windows = tile_read_windows(subset, target_crs, bounds,
                                         target_resolution, resampling_method,
                                         crop=not S2_cloud_classification)
        window_cache, window_cache_tile = {}, None
        for st in subset.itertuples(index=False, name="subtile"):
            # get the item for the tile of this subtile
            subdf = items[items["tile"] == st.name]

            if subdf.empty or st.name not in tile_windows:
                continue

            # the item list is ordered newest first (see ``sort_items``), so
            # the first entry is the most recent reprocessing of the tile
            stac_item = subdf["item"].iloc[0]

            # subtiles come grouped by tile: keep one tile's reads at a time,
            # and fetch its band windows together up front
            if st.name != window_cache_tile:
                window_cache.clear()
                window_cache_tile = st.name
                prefetch_tile_windows(stac_item, S2_bands,
                                      tile_windows[st.name], provider,
                                      window_cache, read_retries)

            subtile_array_ret, write_win, ret_bands = process_S2_subtile(
                intersecting_windows=st.intersecting_windows,
                stac_item=stac_item,
                timestamp=timestamp,
                target_crs=target_crs,
                target_resolution=target_resolution,
                ptile_transform=ptile_transform,
                ptile_width=ptile_width,
                ptile_height=ptile_height,
                S2_mask_snow=S2_mask_snow,
                S2_cloud_classification=S2_cloud_classification,
                S2_cloud_classification_device=S2_cloud_classification_device,
                S2_nbar=S2_nbar,
                cloud_response_queue=cloud_response_queue,
                cloud_request_queue=cloud_request_queue,
                resampling_method=resampling_method,
                S2_bands=S2_bands,
                provider=provider,
                read_retries=read_retries,
                tile_window=tile_windows[st.name],
                window_cache=window_cache,
            )

            # this happens when the href is not available
            # or the subtile does not overlap with the ptile
            if (subtile_array_ret is None or write_win is None
                    or ret_bands is None):
                continue

            band_names = ret_bands
            rows = slice(write_win.row_off, write_win.row_off + write_win.height)
            cols = slice(write_win.col_off, write_win.col_off + write_win.width)
            total[:, rows, cols] += subtile_array_ret
            count[:, rows, cols] += ~(subtile_array_ret == 0)
        return total, count, band_names

    ptile_bounds = transform.array_bounds(ptile_height, ptile_width,
                                          ptile_transform)
    redundant = (subtiles["redundant"] if "redundant" in subtiles else
                 pd.Series(False, index=subtiles.index))
    subtile_array, subtile_array_count, subtile_array_bands = accumulate(
        subtiles[~redundant.values], ptile_bounds)

    # holes: pixels without any raw band, plus the pixels around them that
    # miss some band (see S2_HOLE_FRINGE_PX), minus gaps in the scene itself
    # (see claimed_by_contributors). Their missing bands are filled from
    # redundant subtiles; a band that has data is never touched.
    hole = hole_mask(subtile_array_count[:len(S2_bands)])
    if hole.any():
        hole &= ~claimed_by_contributors(subtiles[~redundant.values], items,
                                         target_crs, ptile_transform,
                                         hole.shape)
    fill = _hole_filling_subtiles(subtiles[redundant.values], items, hole,
                                  target_crs, ptile_transform, provider)
    if fill is not None:
        fill_subtiles, fill_bounds = fill
        fill_sum, fill_count, fill_bands = accumulate(fill_subtiles,
                                                      fill_bounds)
        take = hole[None] & (subtile_array_count == 0) & (fill_count > 0)
        subtile_array[take] = fill_sum[take]
        subtile_array_count[take] = fill_count[take]
        if subtile_array_bands is None and take.any():
            subtile_array_bands = fill_bands

    # no subtile had data for this acquisition
    if subtile_array_bands is None:
        return None, None

    with warnings.catch_warnings():
        # filter out divide by zero warning, this is expected here
        warnings.simplefilter("ignore")
        # TODO I feel like this is not necessary becauset there should not be
        # more then one subtile in one area
        subtile_array /= subtile_array_count

    # compute cloud classification layer
    if S2_cloud_classification:

        cloud_prob_indices = [
            subtile_array_bands.index(band) for band in S2_cloud_prob_bands
        ]

        # select cloud class based on maximum probability
        cloud_class = np.argmax(subtile_array[cloud_prob_indices],
                                axis=0,
                                keepdims=True)

        # apply max filter on cloud clases to dilate invalid pixels
        cloud_class = sc.maximum_filter(cloud_class,
                                        size=(1, 7, 7),
                                        mode="nearest").astype(np.float32)

        # save cloud classes layer
        subtile_array = np.concatenate([subtile_array, cloud_class], axis=0)
        subtile_array_bands.append(S2_cloud_mask_band)

        if not S2_return_cloud_probabilities:
            subtile_array = np.delete(subtile_array,
                                      cloud_prob_indices,
                                      axis=0)
            for band in S2_cloud_prob_bands:
                del subtile_array_bands[subtile_array_bands.index(band)]

    if S2_mask_snow:
        subtile_array = np.concatenate([
            subtile_array,
            np.expand_dims(compute_potential_snow_layer(
                B03=subtile_array[subtile_array_bands.index("B03")],
                B11=subtile_array[subtile_array_bands.index("B11")],
                B08=subtile_array[subtile_array_bands.index("B08")]),
                           axis=0)
        ])
        subtile_array_bands.append(S2_snow_mask_band)

    return subtile_array, subtile_array_bands
