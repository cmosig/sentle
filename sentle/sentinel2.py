import itertools
import multiprocessing as mp

import numpy as np
import pandas as pd
import rasterio
import scipy.ndimage as sc
from rasterio import transform, warp, windows
from rasterio.crs import CRS
from rasterio.enums import Resampling
from shapely.geometry import Polygon, box

from .cloud_mask import (S2_cloud_mask_band, S2_cloud_prob_bands,
                         worker_get_cloud_mask)
from .const import *
from .reproject_util import *
from .snow_mask import S2_snow_mask_band, compute_potential_snow_layer


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

    # convert bounds to sentinel grid crs
    transformed_bounds = Polygon(*warp.transform_geom(
        src_crs=target_crs,
        dst_crs=s2grid.crs,
        geom=box(left, bottom, right, top))["coordinates"])

    # extract overlapping sentinel tiles
    s2grid = s2grid[s2grid["geometry"].intersects(transformed_bounds)].copy()

    general_subtile_windows = [
        windows.Window(col_off, row_off, S2_subtile_size, S2_subtile_size)
        for col_off, row_off in itertools.product(
            np.arange(0, 10980, S2_subtile_size),
            np.arange(0, 10980, S2_subtile_size))
    ]

    # reproject s2 footprint to local utm footprint
    s2grid["s2_footprint_utm"] = s2grid[[
        "geometry", "crs"
    ]].apply(lambda ser: Polygon(*warp.transform_geom(
        src_crs=s2grid.crs, dst_crs=ser["crs"], geom=ser["geometry"].geoms[0])[
            "coordinates"]),
             axis=1)

    # obtain transform of each sentinel 2 tile in local utm crs
    s2grid["tile_transform"] = s2grid["s2_footprint_utm"].apply(
        lambda x: transform.from_bounds(*x.bounds, width=10980, height=10980))

    # convert read window to polygon in S2 local CRS, then transform to
    # s2grid.crs and check overlap with transformed bounds
    s2grid["intersecting_windows"] = s2grid[["tile_transform", "crs"]].apply(
        lambda ser: [
            win_subtile for win_subtile in general_subtile_windows
            if transformed_bounds.intersects(
                Polygon(*warp.transform_geom(
                    src_crs=ser["crs"],
                    dst_crs=s2grid.crs,
                    geom=box(*windows.bounds(win_subtile, ser["tile_transform"]
                                             )))["coordinates"]))
        ],
        axis=1)

    # each line contains one subtile of a sentinel that we want to download and
    # process because it intersects the specified bounds to download
    s2grid = s2grid.explode("intersecting_windows")

    # remove sentinel tiles that do not intersect with the bounds
    # this can happen in edge cases
    s2grid = s2grid.dropna(subset=["intersecting_windows"])

    # only keep columns that we also need later
    return s2grid[['name', 'intersecting_windows']]


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
    cloud_request_queue: mp.Queue,
    cloud_response_queue: mp.Queue,
):
    """Processes a single sentinel 2 subtile. This includes downloading the
    data, reprojecting it to the target_crs and target_resolution, applying
    cloud and snow masks and computing NBAR if requested. The function returns
    the reprojected subtile, the write window and the band names of the
    reprojected subtile.
    """

    # init array that needs to be filled
    subtile_array = np.empty(
        (len(S2_RAW_BANDS), S2_subtile_size, S2_subtile_size),
        dtype=np.float32)
    band_names = S2_RAW_BANDS.copy()

    # save CRS of downloaded sentinel tiles
    s2_crs = None
    # save transformation of sentinel tile for later processing
    s2_tile_transform = None
    # retrieve each band for subtile in sentinel tile
    for i, band in enumerate(S2_RAW_BANDS):
        href = stac_item.assets[band].href
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
            print("Failed to read from stac repository.", type(e))
            print("This is a planetary computer issue, not a sentle issue")
            print("Asset", band, href)

    # in this case we have no data for this subtile
    if s2_tile_transform is None:
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
    warp.reproject(source=subtile_array,
                   destination=subtile_array_repr,
                   src_transform=transform.from_bounds(*subtile_bounds_utm,
                                                       width=S2_subtile_size,
                                                       height=S2_subtile_size),
                   src_crs=s2_crs,
                   dst_crs=target_crs,
                   src_nodata=0,
                   dst_nodata=0,
                   dst_transform=subtile_repr_transform,
                   resampling=Resampling.bilinear)
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
    S2_subtiles,
    cloud_request_queue: mp.Queue,
    cloud_response_queue: mp.Queue,
):

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
        ptile_timestamp, ptile_array_bands = process_ptile_S2(
            timestamp=ts,
            target_crs=target_crs,
            target_resolution=target_resolution,
            S2_cloud_classification=S2_cloud_classification,
            S2_cloud_classification_device=S2_cloud_classification_device,
            S2_mask_snow=S2_mask_snow,
            S2_return_cloud_probabilities=S2_return_cloud_probabilities,
            subtiles=S2_subtiles,
            ptile_transform=ptile_transform,
            ptile_width=ptile_width,
            ptile_height=ptile_height,
            items=items[items["ts"] == ts],
            cloud_request_queue=cloud_request_queue,
            cloud_response_queue=cloud_response_queue,
        )

        # this happens when the href is not available in subtile -> planetary
        # computer issue
        if ptile_timestamp is None:
            continue

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
    ptile_array[:,
                np.any(ptile_array[
                    [ptile_array_bands.index(band)
                     for band in S2_RAW_BANDS]] == 0,
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
):
    # cloud classification layer and snow mask is added later
    num_bands = len(S2_RAW_BANDS)

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

        subtile_array_ret, write_win, subtile_array_bands = process_S2_subtile(
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
            cloud_response_queue=cloud_response_queue,
            cloud_request_queue=cloud_request_queue)

        # this happens when the href is not available
        # -> planetary computer issue
        if subtile_array_ret is None:
            continue

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
