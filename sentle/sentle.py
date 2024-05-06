import dask.array
import rioxarray as rxr
from affine import Affine
import pandas as pd
from rasterio.enums import Resampling
from atenea import atenea
import itertools
import numpy as np
from rasterio import warp, windows, transform
from shapely.geometry import box
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
from dask.distributed import Client
from termcolor import colored
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt


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
    # TODO this should be loaded overall just once
    s2grid = gpd.read_file(
        pkg_resources.resource_filename(__name__,
                                        "data/sentinel2_grid_stripped.gpkg"))

    # convert box to sentinel grid crs
    transformed_bounds = box(
        *rasterio.warp.transform_bounds(src_crs=target_crs,
                                        dst_crs=s2grid.crs,
                                        left=left,
                                        bottom=bottom,
                                        right=right,
                                        top=top))

    # extract overlapping sentinel tiles
    s2grid = s2grid[s2grid["geometry"].intersects(transformed_bounds)]

    general_subtile_windows = [
        windows.Window(col_off, row_off, subtile_size, subtile_size)
        for col_off, row_off in itertools.product(
            np.arange(0, 10980, subtile_size), np.arange(
                0, 10980, subtile_size))
    ]

    # get polygon in respective row-column domain of sentinel tile assuming 10m
    # resolution
    s2grid["bounds_rowcol"] = s2grid["geometry"].apply(
        lambda geom: windows.from_bounds(
            *transformed_bounds.bounds,
            transform.from_bounds(*geom.bounds, 10980, 10980)))

    # iterate through subtiles of each sentinel tile and determine whether they
    # overlap with the bounds
    s2grid["intersecting_windows"] = s2grid["bounds_rowcol"].apply(
        lambda win_bound: [
            win_subtile for win_subtile in general_subtile_windows
            if windows.intersect([win_bound, win_subtile])
        ])

    # each line contains one subtile of a sentinel that we want to download and
    # process because it intersects the specified bounds to download
    s2grid = s2grid.explode("intersecting_windows")

    return s2grid


def process_subtile(subtile, timestamp, atenea_args: dict, subtile_size: int,
                    target_crs: CRS, target_resolution: float,
                    df: pd.DataFrame, stac_endpoint: str, ptile_transform,
                    ptile_width: int, ptile_height: int):

    # TODO clean up and push through as proper parameters
    s2_name = subtile.name
    intersecting_windows = subtile.intersecting_windows
    df = df[df["tile"] == s2_name]

    # init array that needs to be filled
    subtile_array = xr.DataArray(
        data=np.empty((len(BANDS), subtile_size, subtile_size),
                      dtype=np.float32),
        dims=["band", "y", "x"],
        coords=dict(band=BANDS),
        attrs=dict(
            stac=stac_endpoint,
            collection="sentinel-2-l2a",
            # TODO transform upstream
            id=df["id"].iloc[0]))

    # save CRS of downloaded sentinel tiles
    crs = None
    # save transformation of sentinel tile for later processing
    transform = None
    # TODO transform to one item upstream
    item = df["item"].iloc[0]
    # retrieve each band for subtile in sentinel tile
    for band in BANDS:
        href = item.assets[band].href
        with rasterio.open(href) as dr:

            # convert read window respective to tile resolution
            # (lower resolution -> fewer pixels for same area)
            factor = BAND_RESOLUTION[band] // 10
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
    subtile_array.attrs["epsg"] = crs.to_epsg()
    # this is required for rioxarray to figure out the crs
    subtile_array.attrs["crs"] = crs

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
    ys_utm = np.arange(start=subtile_bounds_utm[1],
                       stop=subtile_bounds_utm[3],
                       step=10)

    subtile_array = subtile_array.assign_coords(dict(x=xs_utm, y=ys_utm))

    # 2 push that tile through atenea
    # TODO add atenea kwargs
    # subtile_array = atenea.process(
    #     subtile_array,
    #     source="cubo",
    #     # TODO need to add padding and then reactivate cloud filtering
    #     mask_clouds=False,
    #     # dont reduce time otherwise timesteps will be broken
    #     reduce_time=False,
    #     return_cloud_classification_layer=True,
    #     # chunksize=(len(items), subtile_size),
    #     stac=stac_endpoint)

    # make sure that x and y are the correct spatial resolutions
    # TODO is that actually needed?
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

    # TODO using billinear resampling for spectral bands and nearest neighbor
    # resampling for everything else

    # take only sentinel bands for now
    subtile_array = subtile_array.sel(band=BANDS)
    subtile_array = subtile_array.rio.reproject(
        dst_crs=target_crs,
        transform=subtile_repr_transform,
        shape=(subtile_repr_height, subtile_repr_width),
        nodata=np.nan)

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
        subtile_size: int = 732,
        kwargs_atenea: dict = dict(),
):

    # TODO think about what to do with clouds and if to optionally filter them like atenea
    # --> mask_Clouds, drop_cloudy, clear_sky_threshold, mask_snow)

    # compute bounds of ptile
    # (add target resolution to miny and maxx because we are using top-left
    # coordinates)
    bound_left = da.x.min().item()
    bound_bottom = da.y.min().item() - target_resolution
    bound_right = da.x.max().item() + target_resolution
    bound_top = da.y.max().item()

    # TODO we do obtain_subtiles for each timestamp right now, maybe this could be moved upstream
    # figure out all the sentinel 2 subtiles
    #
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

    # TODO create geopandas df in process function and somehow pass filtered
    # version of this --> less network I/O
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
    items["id"] = items["item"].apply(lambda x: x.id)

    # determine ptile transform from bounds
    ptile_width = (bound_right - bound_left) / target_resolution
    ptile_height = (bound_top - bound_bottom) / target_resolution
    ptile_transform = transform.from_bounds(west=bound_left,
                                            south=bound_bottom,
                                            east=bound_right,
                                            north=bound_top,
                                            width=ptile_width,
                                            height=ptile_height)

    # intiate one array representing the entire subtile for that timestamp
    subtile_array = np.full(shape=(da.shape[1], da.shape[2], da.shape[3]),
                            fill_value=0,
                            dtype=np.float32)

    # count how many values we add per pixel to compute mean later
    subtile_array_count = np.full(shape=(da.shape[1], da.shape[2],
                                         da.shape[3]),
                                  fill_value=0,
                                  dtype=np.uint8)

    for i, st in enumerate(subtiles.itertuples(index=False, name="subtile")):

        subtile_array_xr, write_win = process_subtile(
            subtile=st,
            timestamp=timestamp,
            atenea_args=kwargs_atenea,
            subtile_size=subtile_size,
            target_crs=target_crs,
            target_resolution=target_resolution,
            # df=items[items["tile"] == st.name],
            df=items,
            stac_endpoint=stac_endpoint,
            ptile_transform=ptile_transform,
            ptile_width=ptile_width,
            ptile_height=ptile_height)

        # TODO move fillna upstream, -> set nodata = 0 instead of np.nan earlier
        subtile_array[:,
                      write_win.row_off:write_win.row_off + write_win.height,
                      write_win.col_off:write_win.col_off +
                      write_win.width] += subtile_array_xr.fillna(0).data

        subtile_array_count[:, write_win.row_off:write_win.row_off +
                            write_win.height,
                            write_win.col_off:write_win.col_off +
                            write_win.width] += ~np.isnan(
                                subtile_array_xr.data)

    # compute mean based on the number of overlapping pixels
    # TODO suppress warning of invalid values encountered in divide, this is epxected
    subtile_array /= subtile_array_count

    # replace zeros with nans again
    subtile_array[subtile_array == 0] = np.nan

    # expand dimensions -> one timestep
    subtile_array = np.expand_dims(subtile_array, axis=0)

    # wrap numpy array into xarray again
    out_array = xr.DataArray(data=subtile_array,
                             dims=["time", "band", "y", "x"],
                             coords=dict(time=[timestamp],
                                         band=BANDS,
                                         x=da.x,
                                         y=da.y))

    return out_array


def process(zarr_path: str,
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
            kwargs_atenea: dict = dict(),
            memory_limit_per_worker: str = "2GB"):
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
    kwargs_atenea: dict, default = None
        Arguments passed to atenea specifying processing steps that are applied to each tile.
    zarr_path: str
        Path where zarr storage is supposed to be created.
    """
    # TODO update docstring

    client = Client(n_workers=num_workers,
                    threads_per_worker=threads_per_worker,
                    memory_limit=memory_limit_per_worker)
    print(client.dashboard_link)

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

    timesteps = [i.datetime for i in search.item_collection()]

    # chunks with one per timestep -> many empty timesteps for specific areas,
    # because we have all the timesteps for Germany
    out_array = xr.DataArray(
        data=dask.array.full(
            shape=(len(timesteps), len(BANDS), height, width),
            chunks=(1, 12, processing_tile_size, processing_tile_size),
            # needs to be float in order to store NaNs
            dtype=np.float32,
            fill_value=np.nan),
        dims=["time", "band", "y", "x"],
        coords=dict(
            time=timesteps,
            band=BANDS,
            x=np.arange(bound_left, bound_right,
                        target_resolution).astype(np.float32),
            # we do y-axis in reverse: top-left coordinate
            y=np.arange(bound_top, bound_bottom,
                        -target_resolution).astype(np.float32)))

    out_array = out_array.map_blocks(process_ptile,
                                     kwargs=dict(
                                         target_crs=target_crs,
                                         target_resolution=target_resolution,
                                         subtile_size=subtile_size,
                                         kwargs_atenea=kwargs_atenea,
                                         catalog=catalog,
                                         stac_endpoint=stac_endpoint),
                                     template=out_array)

    # TODO maybe cast to uint16 at the end again

    # convert Timestamp object to UTC timestamp float so that it can be stored in zarr
    out_array = out_array.assign_coords(
        dict(time=[int(t.timestamp()) for t in out_array.time.data]))

    store = zarr.storage.DirectoryStore(zarr_path, dimension_separator=".")
    with ProgressBar():
        out_array.rename("S2").to_zarr(
            store=store,
            mode="w-",
            compute=True,
            encoding={"S2": {
                "write_empty_chunks": False
            }})


if __name__ == "__main__":

    print("------------------------------------")
    print("------------------------------------")
    print("------------------------------------")

    x = process(
        target_crs=CRS.from_string("EPSG:8857"),
        bound_left=767300,
        bound_bottom=7290000,
        bound_right=776000,
        bound_top=7315000,
        # datetime="2023-11-16",
        # datetime="2023-11-11/2023-12-01",
        datetime="2023-11",
        # datetime="2020/2023",
        processing_tile_size=4000,
        target_resolution=10,
        zarr_path="bigout_parallel_test_3.zarr",
        num_workers=5,
        threads_per_worker=1,
        # less then 2GB per worker will likely not work
        memory_limit_per_worker="3GB")

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
