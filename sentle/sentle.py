import dask.array
import logging
import rioxarray as rxr
from affine import Affine
import pandas as pd
from rasterio.enums import Resampling
from atenea import atenea
import itertools
import numpy as np
from rasterio import warp, windows, transform
from shapely.geometry import box, Polygon
from rasterio.crs import CRS
import zarr
import rasterio.warp
from typing import Optional, Union
from utils import *
from pystac_client.item_search import DatetimeLike
import geopandas as gpd
import pkg_resources
import pystac_client
import planetary_computer
import xarray as xr
import warnings
from tqdm import tqdm
import os
from dask.distributed import Client, Variable, LocalCluster
from termcolor import colored
import matplotlib.pyplot as plt
from numcodecs import Blosc
import scipy.ndimage as sc


def recrop_write_window(win, overall_height, overall_width):
    """ Determine write window based on overlap with actual bounds and also
    return how the array that will be written needs to be cropped. """

    # global
    grow = win.row_off
    gcol = win.col_off
    gwidth = win.width
    gheight = win.height

    # local
    lrow = 0
    lcol = 0
    lwidth = win.width
    lheight = win.height

    # if overlapping to the left
    if gcol < 0:
        lcol = abs(gcol)
        gwidth -= abs(gcol)
        lwidth -= abs(gcol)
        gcol = 0

    # if overlapping on the bottom
    if grow < 0:
        lrow = abs(grow)
        gheight -= abs(grow)
        lheight -= abs(grow)
        grow = 0

    # if overlapping to the right
    if overall_width < (gcol + gwidth):
        difwidth = (gcol + gwidth) - overall_width
        gwidth -= difwidth
        lwidth -= difwidth

    # if overlapping to the top
    if overall_height < (grow + gheight):
        difheight = (grow + gheight) - overall_height
        gheight -= difheight
        lheight -= difheight

    assert gcol >= 0
    assert grow >= 0
    assert gwidth <= win.width
    assert gheight <= win.height
    assert overall_height >= (grow + gheight)
    assert overall_width >= (gcol + gwidth)
    assert lcol >= 0
    assert lrow >= 0
    assert lwidth <= win.width
    assert lheight <= win.height
    assert lwidth == gwidth
    assert lheight == lheight
    assert all([
        x % 1 == 0
        for x in [grow, gcol, gwidth, gheight, lrow, lcol, lwidth, lheight]
    ])

    return windows.Window(
        row_off=grow, col_off=gcol, height=gheight,
        width=gwidth).round_offsets().round_lengths(), windows.Window(
            row_off=lrow, col_off=lcol, height=lheight,
            width=lwidth).round_offsets().round_lengths()


def obtain_subtiles(target_crs: CRS, left: float, bottom: float, right: float,
                    top: float, subtile_size: int):
    """Retrieves the sentinel subtiles that intersect the with the specified
    bounds. The bounds are interpreted based on the given target_crs.
    """

    # TODO make it possible to not only use naive bounds but also MultiPolygons

    # check if supplied sub_tile_width makes sense
    assert (subtile_size
            >= 16) and (subtile_size
                        <= 10980), "subtile_size needs to within 16 and 10980"
    assert (10980 %
            subtile_size) == 0, "subtile_size needs to be a divisor of 10980"

    # load sentinel grid
    s2grid = Variable("s2gridfile").get()
    assert s2grid.crs == "EPSG:4326"

    # convert bounds to sentinel grid crs
    transformed_bounds = Polygon(*warp.transform_geom(
        src_crs=target_crs,
        dst_crs=s2grid.crs,
        geom=box(left, bottom, right, top))["coordinates"])

    # extract overlapping sentinel tiles
    s2grid = s2grid[s2grid["geometry"].intersects(transformed_bounds)]

    general_subtile_windows = [
        windows.Window(col_off, row_off, subtile_size, subtile_size)
        for col_off, row_off in itertools.product(
            np.arange(0, 10980, subtile_size), np.arange(
                0, 10980, subtile_size))
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
        lambda x: rasterio.transform.from_bounds(
            *x.bounds, width=10980, height=10980))

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

    return s2grid


def process_subtile(intersecting_windows, stac_item, timestamp,
                    subtile_size: int, target_crs: CRS,
                    target_resolution: float, stac_endpoint: str,
                    ptile_transform, ptile_width: int, ptile_height: int,
                    mask_snow: bool, mask_clouds: bool,
                    return_cloud_classification_layer: bool,
                    return_cloud_probabilities: bool, compute_nbar: bool,
                    mask_clouds_device: str):

    # TODO entirely remove skip atenea

    # TODO can we completely stick to uint16 here to save memory?

    # init array that needs to be filled
    # TODO why is there no x/y coordinate here?
    subtile_array = xr.DataArray(data=np.empty(
        (1, len(S2_RAW_BANDS), subtile_size, subtile_size), dtype=np.float32),
                                 dims=["time", "band", "y", "x"],
                                 coords=dict(band=S2_RAW_BANDS,
                                             time=[timestamp],
                                             id=("time", [stac_item.id])),
                                 attrs=dict(stac=stac_endpoint,
                                            collection="sentinel-2-l2a"))

    # save CRS of downloaded sentinel tiles
    crs = None
    # save transformation of sentinel tile for later processing
    transform = None
    # retrieve each band for subtile in sentinel tile
    for band in S2_RAW_BANDS:
        href = stac_item.assets[band].href
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
                                out_shape=(subtile_size, subtile_size),
                                out_dtype=np.float32)

            # replace 0 with NaN -> important for merging
            # 0 means in sentinel2 nodata
            read_data[read_data == 0] = np.nan

            subtile_array.loc[dict(band=band)] = read_data

            # save and validate epsg
            assert (crs is None) or (
                crs == dr.crs), "CRS mismatch within one sentinel tile"
            crs = dr.crs

            # save transform for a 10m band tile
            if band == "B02":
                transform = dr.transform

    # this is required for atenea
    if compute_nbar:
        subtile_array.attrs["epsg"] = crs.to_epsg()
    # this is required for rioxarray to figure out the crs
    subtile_array.attrs["crs"] = crs
    # set stac_item, required for harmonization
    subtile_array.attrs["stac_item"] = stac_item

    # determine bounds based on subtile window and tile transform
    subtile_bounds_utm = windows.bounds(intersecting_windows, transform)
    assert (
        subtile_bounds_utm[2] - subtile_bounds_utm[0]
    ) // 10 == subtile_size, "mismatch between subtile size and bounds on x-axis"
    assert (
        subtile_bounds_utm[3] - subtile_bounds_utm[1]
    ) // 10 == subtile_size, "mismatch between subtile size and bounds on y-axis"

    # set array coordindates based on bounds and standard resolution of 10m
    xs_utm = np.arange(start=subtile_bounds_utm[0],
                       stop=subtile_bounds_utm[2],
                       step=10)
    ys_utm = np.arange(start=subtile_bounds_utm[3],
                       stop=subtile_bounds_utm[1],
                       step=-10)
    subtile_array = subtile_array.assign_coords(dict(x=xs_utm, y=ys_utm))

    if mask_clouds:
        assert subtile_size == 732, "cloud masking only works with subtile size of 732 at the moment"
        # add padding of 2 pixels around the edge
        subtile_array = subtile_array.pad(pad_width=dict(x=(2, 2), y=(2, 2)))
        assert subtile_array.x.shape == (736, ) and subtile_array.y.shape == (
            736, ), "unexpected shape after padding"

    # 2 push that tile through atenea
    subtile_array = atenea.process(
        subtile_array,
        source="sentle",
        reduce_time=False,
        mask_clouds=mask_clouds,
        # we compute classification ourselves based on the returned probabilities
        return_cloud_probabilities=True,
        return_cloud_classification_layer=False,
        return_clear_sky_mask=False,
        stac=stac_endpoint,
        quiet=True,
        mask_clouds_device=mask_clouds_device,
        nbar=compute_nbar,
        mask_snow=mask_snow,
    )

    # drop all attributes --> only needed to atenea
    if compute_nbar:
        del subtile_array.attrs["epsg"]
    del subtile_array.attrs["stac_item"]

    if mask_clouds:

        # crop subtile array to original shape
        assert subtile_array.x.shape == (736, ) and subtile_array.y.shape == (
            736,
        ), f"unexpected shape before padding removal {subtile_array.sizes}"
        subtile_array = subtile_array[:, :, 2:-2, 2:-2]
        assert subtile_array.x.shape == (732, ) and subtile_array.y.shape == (
            732,
        ), f"unexpected shape after padding removal{subtile_array.sizes}"

    # remove time dimension, only needed for ateana
    subtile_array = subtile_array.loc[dict(time=timestamp)]

    # make sure that x and y are the correct spatial resolutions
    subtile_array = subtile_array.rio.set_spatial_dims(x_dim="x", y_dim="y")

    # 3 reproject to target_crs for each band
    # determine transform --> round to target resolution so that reprojected
    # subtiles align across subtiles
    subtile_repr_transform, subtile_repr_width, subtile_repr_height = rasterio.warp.calculate_default_transform(
        src_crs=crs,
        dst_crs=target_crs,
        width=subtile_array.sizes["x"],
        height=subtile_array.sizes["y"],
        left=subtile_bounds_utm[0],
        bottom=subtile_bounds_utm[1],
        right=subtile_bounds_utm[2],
        top=subtile_bounds_utm[3],
        resolution=target_resolution)
    subtile_repr_transform = Affine(
        subtile_repr_transform.a,
        subtile_repr_transform.b,
        subtile_repr_transform.c -
        (subtile_repr_transform.c % target_resolution),
        subtile_repr_transform.d,
        subtile_repr_transform.e,
        # + target_resolution because upper left corner
        subtile_repr_transform.f -
        (subtile_repr_transform.f % target_resolution) + target_resolution,
        subtile_repr_transform.g,
        subtile_repr_transform.h,
        subtile_repr_transform.i,
    )

    # include one more pixel because rounding down
    subtile_repr_width += 1
    subtile_repr_height += 1

    # compute bounds in target crs based on rounded transform
    subtile_bounds_tcrs = bounds_from_transform_height_width_res(
        transform=subtile_repr_transform,
        height=subtile_repr_height,
        width=subtile_repr_width,
        resolution=target_resolution)

    NN_BANDS = ["snow_mask"]
    # billinear reprojection for everything but the NN bands
    subtile_array_BL = subtile_array.sel(
        band=[x for x in subtile_array.band.data
              if x not in NN_BANDS]).rio.reproject(
                  dst_crs=target_crs,
                  transform=subtile_repr_transform,
                  shape=(subtile_repr_height, subtile_repr_width),
                  nodata=np.nan,
                  resampling=Resampling.bilinear)

    # TODO generalize maybe
    if mask_snow:
        # nearest neighbor reprojection
        subtile_array_NN = subtile_array.sel(band=NN_BANDS).rio.reproject(
            dst_crs=target_crs,
            transform=subtile_repr_transform,
            shape=(subtile_repr_height, subtile_repr_width),
            nodata=np.nan,
            resampling=Resampling.nearest)

        # merge again
        subtile_array = xr.concat([subtile_array_BL, subtile_array_NN],
                                  dim="band")
    else:
        subtile_array = subtile_array_BL

    # change center to coordinates to top-left coords (rioxarray caveat)
    # TODO file an issue with rioxarray to it a parameter how coords are
    # represented
    subtile_array = subtile_array.assign_coords(
        dict(x=subtile_array.x.data - (target_resolution / 2),
             y=subtile_array.y.data + (target_resolution / 2)))

    # figure out where to write the subtile within the overall bounds
    write_win = windows.from_bounds(
        *subtile_bounds_tcrs,
        transform=ptile_transform).round_offsets().round_lengths()

    write_win, local_win = recrop_write_window(write_win, ptile_height,
                                               ptile_width)

    # crop subtile_array based on computed local win because it could overlap
    # with the overall bounds
    subtile_array = subtile_array[:, local_win.row_off:local_win.height +
                                  local_win.row_off,
                                  local_win.col_off:local_win.col_off +
                                  local_win.width]

    return subtile_array, write_win


def process_ptile(
    da: xr.DataArray,
    target_crs: CRS,
    target_resolution: float,
    catalog,
    stac_endpoint: str,
    mask_clouds_device: str,
    subtile_size: int = 732,
    mask_snow: bool = False,
    mask_clouds: bool = False,
    return_cloud_classification_layer: bool = False,
    return_cloud_probabilities: bool = False,
    compute_nbar: bool = False,
):
    # compute bounds of ptile
    # (add target resolution to miny and maxx because we are using top-left
    # coordinates)
    bound_left = da.x.min().item()
    bound_bottom = da.y.min().item() - target_resolution
    bound_right = da.x.max().item() + target_resolution
    bound_top = da.y.max().item()

    # obtain sub-sentinel tiles based on supplied bounds and CRS
    subtiles = obtain_subtiles(target_crs,
                               bound_left,
                               bound_bottom,
                               bound_right,
                               bound_top,
                               subtile_size=subtile_size)

    # extract the timestamp we are processing. there should only be one
    timestamp = da.time.data
    assert timestamp.shape == (1, )
    timestamp = timestamp[0]

    # retrieve items (possible across multiple sentinel tile) for specified
    # timestamp
    item_list = list(
        catalog.search(collections=["sentinel-2-l2a"],
                       datetime=timestamp,
                       bbox=rasterio.warp.transform_bounds(
                           src_crs=target_crs,
                           dst_crs="EPSG:4326",
                           left=bound_left,
                           bottom=bound_bottom,
                           right=bound_right,
                           top=bound_top)).item_collection())

    if len(item_list) == 0:
        # if there is nothing within the bounds and for that timestamp return.
        # possible and normal
        print(colored("empty ptile, returning input da", "green"))
        return da

    items = pd.DataFrame()
    items["item"] = item_list
    items["tile"] = items["item"].apply(lambda x: x.properties["s2:mgrs_tile"])

    # determine ptile transform from bounds
    ptile_width = (bound_right - bound_left) / target_resolution
    ptile_height = (bound_top - bound_bottom) / target_resolution
    ptile_transform = transform.from_bounds(west=bound_left,
                                            south=bound_bottom,
                                            east=bound_right,
                                            north=bound_top,
                                            width=ptile_width,
                                            height=ptile_height)

    # cloud classification layer is added later
    num_bands = da.shape[1]
    if return_cloud_classification_layer:
        num_bands -= 1

    # intiate one array representing the entire subtile for that timestamp
    subtile_array = np.full(shape=(num_bands, da.shape[2], da.shape[3]),
                            fill_value=0,
                            dtype=np.float32)

    # count how many values we add per pixel to compute mean later
    subtile_array_count = np.full(shape=(num_bands, da.shape[2], da.shape[3]),
                                  fill_value=0,
                                  dtype=np.uint8)

    subtile_array_bands = None
    for i, st in enumerate(subtiles.itertuples(index=False, name="subtile")):
        subdf = items[items["tile"] == st.name]
        # there should only be one S2 tile for the timestamp
        assert subdf.shape[
            0] <= 1, f"unexpected number of items:\n {subdf} \n {items} \n {st.name}"

        if subdf.empty:
            # can happen with orbit edges, because we filter stac with bounds
            continue

        stac_item = subdf["item"].iloc[0]

        subtile_array_xr, write_win = process_subtile(
            intersecting_windows=st.intersecting_windows,
            stac_item=stac_item,
            timestamp=timestamp,
            subtile_size=subtile_size,
            target_crs=target_crs,
            target_resolution=target_resolution,
            stac_endpoint=stac_endpoint,
            ptile_transform=ptile_transform,
            ptile_width=ptile_width,
            ptile_height=ptile_height,
            mask_snow=mask_snow,
            mask_clouds=mask_clouds,
            return_cloud_classification_layer=return_cloud_classification_layer,
            return_cloud_probabilities=return_cloud_probabilities,
            compute_nbar=compute_nbar,
            mask_clouds_device=mask_clouds_device)

        # save band order
        subtile_array_bands = list(subtile_array_xr.band.data)

        # also replace nan with 0 so that the mean computation works
        # (this is reverted later)
        subtile_array[:,
                      write_win.row_off:write_win.row_off + write_win.height,
                      write_win.col_off:write_win.col_off +
                      write_win.width] += subtile_array_xr.fillna(0).data

        subtile_array_count[:, write_win.row_off:write_win.row_off +
                            write_win.height,
                            write_win.col_off:write_win.col_off +
                            write_win.width] += ~np.isnan(
                                subtile_array_xr.data)

    # TODO do snow mask computation outside atenea too -> easier merging?

    # determine nodata mask based on where values are zero -> mean nodata for S2...
    # (need to do this here, because after computing mean there will be nans
    # from divide by zero)
    nodata_mask_S2_raw = np.any(subtile_array[[
        subtile_array_bands.index(band) for band in S2_RAW_BANDS
    ]] == 0,
                                axis=0)

    with warnings.catch_warnings():
        # filter out divide by zero warning, this is expected here
        warnings.simplefilter("ignore")
        subtile_array /= subtile_array_count

    if return_cloud_classification_layer:

        # compute cloud classification layer
        cloud_prob_bands = [
            "clear_sky_probability", "thick_cloud_probability",
            "thin_cloud_probability", "shadow_probability"
        ]

        # select cloud class based on maximum probability
        cloud_class = np.argmax(subtile_array[[
            subtile_array_bands.index(band) for band in cloud_prob_bands
        ]],
                                axis=0,
                                keepdims=True)

        # apply max filter on cloud clases to dilate invalid pixels
        cloud_class = sc.maximum_filter(cloud_class,
                                        size=(1, 7, 7),
                                        mode="nearest").astype(np.float32)

        # save cloud classes layer
        subtile_array = np.concatenate([subtile_array, cloud_class], axis=0)
        subtile_array_bands.append("cloud_classification_layer")

    # ... and set all such pixels to nan (of which some are already nan because
    # of divide by zero)
    subtile_array[:, nodata_mask_S2_raw] = np.nan

    # expand dimensions -> one timestep
    subtile_array = np.expand_dims(subtile_array, axis=0)

    # wrap numpy array into xarray again
    out_array = xr.DataArray(data=subtile_array,
                             dims=["time", "band", "y", "x"],
                             coords=dict(time=[timestamp],
                                         band=subtile_array_bands,
                                         x=da.x,
                                         y=da.y))

    # only return bands that have been requested
    return out_array.sel(band=da.band)


def process(
    zarr_path: str,
    target_crs: CRS,
    target_resolution: float,
    bound_left: float,
    bound_bottom: float,
    bound_right: float,
    bound_top: float,
    datetime: DatetimeLike,
    processing_tile_size: int,
    num_workers: int = 1,
    threads_per_worker: int = 1,
    subtile_size: int = 732,
    memory_limit_per_worker: str = "4GB",
    mask_snow: bool = False,
    mask_clouds: bool = False,
    mask_clouds_device="cuda",
    return_cloud_classification_layer: bool = False,
    return_cloud_probabilities: bool = False,
    compute_nbar: bool = False,
):
    """
    Parameters
    ----------

    target_crs: CRS
        Specifies the target CRS that all data will be reprojected to.
    target_resolution: float
        Determines the resolution that all data is reprojected to in the `target_crs`.
    bound_left: float
        Left bound of area that is supposed to be covered. Unit is in `target_crs`.
    bound_bottom: float
        Bottom bound of area that is supposed to be covered. Unit is in `target_crs`.
    bound_right: float
        Right bound of area that is supposed to be covered. Unit is in `target_crs`.
    bound_top: float
        Top bound of area that is supposed to be covered. Unit is in `target_crs`.
    datetime: DatetimeLike
        Specifies time range of data to be downloaded. This is forwarded to the respective stac interface.
    subtile_size: int, default=732
        Specifies the size of each subtile. The maximum is the size of a sentinel tile (10980). If cloud filtering is enabled the minimum tilesize is 256, otherwise 16. It also needs to be a divisor of 10980, so that each sentinel tile can be segmented without overlaps.
    num_cores: int, default = 1
        Number of CPU cores across which subtile processing is supposed to be distributed.
    zarr_path: str
        Path where zarr storage is supposed to be created.
    """

    assert subtile_size == 732, "Unsupported subtile size."

    # TODO update docstring
    # TODO move out dask client init and zarr store and return lazy dask array
    # -> also make this into class with to_zarr function ?
    # TODO provide function to aggregate by timeperiod and before that filter clouds

    # derive bands to save from arguments
    bands_to_save = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]
    if mask_snow:
        bands_to_save.append("snow_mask")
    if compute_nbar:
        bands_to_save += [
            "NBAR_B02",
            "NBAR_B03",
            "NBAR_B04",
            "NBAR_B05",
            "NBAR_B06",
            "NBAR_B07",
            "NBAR_B08",
            "NBAR_B11",
            "NBAR_B12",
        ]
    if return_cloud_classification_layer:
        bands_to_save.append("cloud_classification_layer")
    if return_cloud_probabilities:
        bands_to_save += [
            "clear_sky_probability",
            "thick_cloud_probability",
            "thin_cloud_probability",
            "shadow_probability",
        ]

    cluster = LocalCluster(dashboard_address="127.0.0.1:9988",
                           n_workers=num_workers,
                           threads_per_worker=threads_per_worker,
                           memory_limit=memory_limit_per_worker)
    client = Client(cluster)
    print(client.dashboard_link)

    # load Sentinel 2 grid
    Variable("s2gridfile").set(
        gpd.read_file(
            pkg_resources.resource_filename(
                __name__, "data/sentinel2_grid_stripped_with_epsg.gpkg")))

    # setup zarr storage
    # determine width and height based on bounds and resolution
    width, w_rem = divmod(abs(bound_right - bound_left), target_resolution)
    height, h_rem = divmod(abs(bound_top - bound_bottom), target_resolution)
    if h_rem > 0:
        warnings.warn(
            "Specified top/bottom bounds are not perfectly divisable by specified target_resolution. The resulting coverage will be slightly cropped"
        )
    if w_rem > 0:
        warnings.warn(
            "Specified left/right bounds are not perfectly divisable by specified target_resolution. The resulting coverage will be slightly cropped"
        )

    # sign into planetary computer
    stac_endpoint = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystac_client.Client.open(
        stac_endpoint,
        modifier=planetary_computer.sign_inplace,
    )

    # get all items within date range and area
    search = catalog.search(collections=["sentinel-2-l2a"],
                            datetime=datetime,
                            bbox=rasterio.warp.transform_bounds(
                                src_crs=target_crs,
                                dst_crs="EPSG:4326",
                                left=bound_left,
                                bottom=bound_bottom,
                                right=bound_right,
                                top=bound_top))

    timesteps = sorted(
        list(set([i.datetime for i in search.item_collection()])))

    # chunks with one per timestep -> many empty timesteps for specific areas,
    # because we have all the timesteps for Germany
    out_array = xr.DataArray(
        data=dask.array.full(
            shape=(len(timesteps), len(bands_to_save), height, width),
            chunks=(1, len(bands_to_save), processing_tile_size,
                    processing_tile_size),
            # needs to be float in order to store NaNs
            dtype=np.float32,
            fill_value=np.nan),
        dims=["time", "band", "y", "x"],
        coords=dict(
            time=timesteps,
            band=bands_to_save,
            x=np.arange(bound_left, bound_right,
                        target_resolution).astype(np.float32),
            # we do y-axis in reverse: top-left coordinate
            y=np.arange(bound_top, bound_bottom,
                        -target_resolution).astype(np.float32)))

    out_array = out_array.map_blocks(
        process_ptile,
        kwargs=dict(
            target_crs=target_crs,
            target_resolution=target_resolution,
            subtile_size=subtile_size,
            catalog=catalog,
            stac_endpoint=stac_endpoint,
            mask_snow=mask_snow,
            mask_clouds=mask_clouds,
            return_cloud_classification_layer=return_cloud_classification_layer,
            return_cloud_probabilities=return_cloud_probabilities,
            compute_nbar=compute_nbar,
            mask_clouds_device=mask_clouds_device,
        ),
        template=out_array)

    # TODO maybe cast to uint16 at the end again

    # convert Timestamp object to UTC timestamp float so that it can be stored in zarr
    out_array = out_array.assign_coords(
        dict(time=[int(t.timestamp()) for t in out_array.time.data]))

    store = zarr.storage.DirectoryStore(zarr_path, dimension_separator=".")

    # NOTE the compression may not be optimal, need to benchmark
    out_array.rename("S2").to_zarr(store=store,
                                   mode="w-",
                                   compute=True,
                                   encoding={
                                       "S2": {
                                           "write_empty_chunks": False,
                                           "compressor": Blosc(cname="zstd"),
                                       }
                                   })


if __name__ == "__main__":

    print("------------------------------------")
    print("------------------------------------")
    print("------------------------------------")

    # x = process(
    #     target_crs=CRS.from_string("EPSG:8857"),
    #     bound_left=767300,
    #     bound_bottom=7290000,
    #     bound_right=776000,
    #     bound_top=7315000,
    #     datetime="2023-11-16",
    #     # datetime="2023-11-11/2023-12-01",
    #     # datetime="2023-11",
    #     # datetime="2020/2023",
    #     processing_tile_size=4000,
    #     target_resolution=10,
    #     zarr_path="bigout_parallel_test_5.zarr",
    #     num_workers=1,
    #     threads_per_worker=1,
    #     # less then 3GB per worker will likely not work
    #     memory_limit_per_worker="8GB",
    #     mask_clouds=True,
    #     mask_snow=True,
    #     return_cloud_probabilities=True,
    #     return_cloud_classification_layer=True,
    #     compute_nbar=False,
    #     mask_clouds_device="cuda")

    x = process(
        target_crs=CRS.from_string("EPSG:8857"),
        bound_left=921070,
        bound_bottom=6101250,
        bound_right=977630,
        bound_top=6144550,
        datetime="2023-06-10",
        # datetime="2023-06-01/2023-12-01",
        # datetime="2023-11",
        # datetime="2020/2023",
        processing_tile_size=4000,
        target_resolution=10,
        zarr_path="/net/scratch/cmosig/halle_leipzig_5.zarr",
        num_workers=50,
        threads_per_worker=1,
        # less then 3GB per worker will likely not work
        memory_limit_per_worker="8GB",
        mask_clouds=False,
        mask_snow=False,
        return_cloud_probabilities=False,
        return_cloud_classification_layer=False,
        compute_nbar=False,
        mask_clouds_device="cuda")

    # x = process(
    #     target_crs=CRS.from_string("EPSG:8857"),
    #     bound_left=564670,
    #     bound_bottom=5718050,
    #     bound_right=1084500,
    #     bound_top=6409170,
    #     # datetime="2023-11-11/2023-12-01",
    #     datetime="2023",
    #     processing_tile_size=4000,
    #     target_resolution=10,
    #     zarr_path="bigout_oneday.zarr")
