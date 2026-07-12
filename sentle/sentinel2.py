import itertools
import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd
import rasterio
import scipy.ndimage as sc
from rasterio import transform, warp, windows
from rasterio.crs import CRS
from rasterio.enums import Resampling
from shapely.geometry import box, shape
from shapely.ops import unary_union

from .cloud_mask import S2_cloud_mask_band, S2_cloud_prob_bands, worker_get_cloud_mask
from .const import (
    S2_NBAR_BANDS,
    S2_RAW_BAND_RESOLUTION,
    S2_RAW_BANDS,
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
from .stac import refresh_sas_token


def obtain_subtiles(target_crs: CRS, left: float, bottom: float, right: float,
                    top: float, s2grid):
    """Retrieves the sentinel subtiles that intersect the with the specified
    bounds. The bounds are interpreted based on the given target_crs.
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
    for row in s2grid.itertuples(index=False):
        for win_subtile, footprint in row.window_footprints:
            if claimed is None:
                keep = True
            else:
                covered_area = footprint.intersection(claimed).area
                keep = (footprint.area -
                        covered_area) > keep_frac_threshold * footprint.area
            if keep:
                kept_names.append(row.name)
                kept_windows.append(win_subtile)
        claimed = (row.aoi_footprint
                   if claimed is None else claimed.union(row.aoi_footprint))

    # each row is one subtile of a sentinel tile to download and process
    return pd.DataFrame({
        "name": kept_names,
        "intersecting_windows": kept_windows
    })


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
):
    """Processes a single sentinel 2 subtile. This includes downloading the
    data, reprojecting it to the target_crs and target_resolution, applying
    cloud and snow masks and computing NBAR if requested. The function returns
    the reprojected subtile, the write window and the band names of the
    reprojected subtile.

    ``S2_bands`` selects which raw reflectance bands to download (a subset of
    ``S2_RAW_BANDS``); defaults to all of them. Cloud classification always
    requires the full set (enforced upstream).
    """

    # which raw bands to actually download for this subtile
    if S2_bands is None:
        S2_bands = S2_RAW_BANDS
    download_bands = list(S2_bands)

    # init array that needs to be filled
    subtile_array = np.empty(
        (len(download_bands), S2_subtile_size, S2_subtile_size),
        dtype=np.float32)
    band_names = download_bands.copy()

    # save CRS of downloaded sentinel tiles
    s2_crs = None
    # save transformation of sentinel tile for later processing
    s2_tile_transform = None
    # retrieve each band for subtile in sentinel tile
    for i, band in enumerate(download_bands):
        href = refresh_sas_token(stac_item.assets[band].href)
        try:
            with rasterio.open(href) as dr:

                # convert read window respective to tile resolution
                # (lower resolution -> fewer pixels for same area)
                factor = S2_RAW_BAND_RESOLUTION[band] // 10
                orig_win = intersecting_windows
                read_window = windows.Window(orig_win.col_off // factor,
                                             orig_win.row_off // factor,
                                             orig_win.width // factor,
                                             orig_win.height // factor)

                # read subtile and directly upsample to 10m resolution using
                # nearest-neighbor (default)
                read_data = dr.read(indexes=1,
                                    window=read_window,
                                    out_shape=(S2_subtile_size,
                                               S2_subtile_size),
                                    out_dtype=np.float32)

                # harmonization
                if float(
                        stac_item.properties["s2:processing_baseline"]) >= 4.0:

                    # clip values to minimum 1000
                    # we do this here instead of clipping to zero later to avoid
                    # integer underflow when using a uint16 potentially later on
                    read_data[read_data < 1000] = 1000

                    # adjust reflectance for non-zero values
                    read_data[read_data != 0] -= 1000

                # save
                subtile_array[i] = read_data

                # save and validate epsg
                assert (s2_crs is None) or (
                    s2_crs == dr.crs), "CRS mismatch within one sentinel tile"
                s2_crs = dr.crs

                # save transform for a 10m band tile
                if band == "B02":
                    s2_tile_transform = dr.transform
        except rasterio.errors.RasterioIOError as e:
            warnings.warn(
                f"stac_read_failure asset={href} band={band} exception_type={type(e).__name__} message={e} note=planetary_computer_issue"
            )

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
):

    # the raw reflectance bands to download/save (subset of S2_RAW_BANDS)
    if S2_bands is None:
        S2_bands = S2_RAW_BANDS
    S2_bands = list(S2_bands)

    items = pd.DataFrame()
    items["item"] = item_list
    items["tile"] = items["item"].apply(lambda x: x.properties["s2:mgrs_tile"])
    items["ts"] = items["item"].apply(lambda x: x.datetime)

    # intiate one array representing the entire subtile for that timestamp
    ptile_array = np.full(shape=(len(S2_bands_to_save), ptile_height,
                                 ptile_width),
                          fill_value=0,
                          dtype=np.float32)

    # also dont need to perform aggreation if we only have one item
    perform_aggregation = (time_composite_freq is not None) and (len(item_list)
                                                                 > 1)

    if perform_aggregation:
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

    if perform_aggregation:
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
):
    # the raw reflectance bands to download/save (subset of S2_RAW_BANDS)
    if S2_bands is None:
        S2_bands = S2_RAW_BANDS
    S2_bands = list(S2_bands)

    # cloud classification layer and snow mask is added later
    num_bands = len(S2_bands)

    if S2_cloud_classification:
        # we need the probs here, will remove when returning
        num_bands += 4

    # intiate one array representing the entire subtile for that timestamp
    subtile_array = np.full(shape=(num_bands, ptile_height, ptile_width),
                            fill_value=0,
                            dtype=np.float32)

    # count how many values we add per pixel to compute mean later
    subtile_array_count = np.full(shape=(num_bands, ptile_height, ptile_width),
                                  fill_value=0,
                                  dtype=np.uint8)

    subtile_array_bands = None
    for st in subtiles.itertuples(index=False, name="subtile"):
        # filter items by sentinel tile name
        subdf = items[items["tile"] == st.name]

        if subdf.empty:
            # can happen with orbit edges, because we filter stac with bounds
            continue

        # NOTE it is possible that multiple items are returned for one
        # timestamp and a sentinel tile. These are duplicates and a bug in
        # sentinel2 repository
        stac_item = subdf["item"].iloc[0]

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
        )

        # this happens when the href is not available
        # -> planetary computer issue
        if subtile_array_ret is None or write_win is None or ret_bands is None:
            continue

        # only assign the sentinel/band accumulator for valid subtiles, so an
        # out-of-bounds last subtile cannot clobber it back to None and discard
        # data accumulated from earlier valid subtiles
        subtile_array_bands = ret_bands

        # also replace nan with 0 so that the mean computation works
        # (this is reverted later)
        subtile_array[:,
                      write_win.row_off:write_win.row_off + write_win.height,
                      write_win.col_off:write_win.col_off +
                      write_win.width] += subtile_array_ret

        subtile_array_count[:, write_win.row_off:write_win.row_off +
                            write_win.height,
                            write_win.col_off:write_win.col_off +
                            write_win.width] += ~(subtile_array_ret == 0)

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
