from dask.distributed import Client
import dask.array as da
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


def recrop_write_window(win, overall_height, overall_width):

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

    return windows.Window(row_off=grow,
                          col_off=gcol,
                          height=gheight,
                          width=gwidth), windows.Window(row_off=lrow,
                                                        col_off=lcol,
                                                        height=lheight,
                                                        width=lwidth)


def obtain_subtiles(target_crs: CRS, left: float, bottom: float, right: float,
                    top: float, subtile_size: int):
    """Retrieves the sentinel subtiles that intersect the with the specified
    bounds. The bounds are interpreted based on the given target_crs.
    """

    # TODO make it possible to not only use naive bounds but also MultiPolygons

    # 0 check if supplied sub_tile_width makes sense
    assert (subtile_size
            >= 16) and (subtile_size
                        <= 10980), "subtile_size needs to within 16 and 10980"
    assert (10980 %
            subtile_size) == 0, "subtile_size needs to be a divisor of 10980"

    # 1 load sentinel grid
    s2grid = gpd.read_file(
        pkg_resources.resource_filename(__name__,
                                        "data/sentinel2_grid_stripped.gpkg"))

    # 2 convert box to sentinel grid crs
    transformed_bounds = box(
        *rasterio.warp.transform_bounds(src_crs=target_crs,
                                        dst_crs=s2grid.crs,
                                        left=left,
                                        bottom=bottom,
                                        right=right,
                                        top=top))

    # 3 extract overlapping sentinel tiles
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


def bounds_from_transform_height_width_res(transform, height, width,
                                           resolution):
    # minx, miny, maxx, maxy
    return (transform.c, transform.f - (height * resolution),
            transform.c + (width * resolution), transform.f)


def process_subtile(subtile, out_array, atenea_args: dict, subtile_size: int,
                    target_crs: CRS, target_resolution: float,
                    df: pd.DataFrame, stac_endpoint: str, overall_transform,
                    overall_width: int, overall_height: int):

    subtile_array = xr.DataArray(data=np.empty(
        (df.shape[0], len(BANDS), subtile_size, subtile_size),
        dtype=np.uint16),
                                 dims=["time", "band", "y", "x"],
                                 coords=dict(time=df["datetime"].tolist(),
                                             band=BANDS,
                                             id=("time", df["id"].tolist())),
                                 attrs=dict(stac=stac_endpoint,
                                            collection="sentinel-2-l2a"))

    # iterate through timestamps
    crs = None
    transform = None
    for item in df["item"]:
        for band in BANDS:
            href = item.assets[band].href
            with rasterio.open(href) as dr:
                # convert read window respective to tile resolution
                factor = BAND_RESOLUTION[band] // 10
                orig_win = subtile.intersecting_windows
                read_window = windows.Window(orig_win.col_off // factor,
                                             orig_win.row_off // factor,
                                             orig_win.width // factor,
                                             orig_win.height // factor)

                # read subtile and directly upsample to 10m resolution using
                # nearest-neighbor (default)
                # TODO lazy reading with rioxarray?
                subtile_array.loc[dict(time=item.datetime,
                                       band=band)] = dr.read(
                                           indexes=1,
                                           window=read_window,
                                           out_shape=(subtile_size,
                                                      subtile_size),
                                       )

                # save and validate epsg
                assert (crs is None) or (
                    crs == dr.crs), "CRS mismatch within one sentinel tile"
                crs = dr.crs

                # save transform for 10m tile
                if band == "B02":
                    transform = dr.transform

    # this is required for atenea
    subtile_array.attrs["epsg"] = crs.to_epsg()
    # this is required for rioxarray to figure out the crs
    subtile_array.attrs["crs"] = crs

    # determine bounds based on subtile window and tile transform
    subtile_bounds_utm = windows.bounds(subtile.intersecting_windows,
                                        transform)
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
    subtile_array = subtile_array.rio.set_spatial_dims(x_dim="x", y_dim="y")

    # 3 reproject to target_crs for each band
    # determine transform --> round to target resolution
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

    subtile_bounds_tcrs = bounds_from_transform_height_width_res(
        transform=subtile_repr_transform,
        height=subtile_repr_height,
        width=subtile_repr_width,
        resolution=target_resolution)

    write_win = windows.from_bounds(
        *subtile_bounds_tcrs,
        transform=overall_transform).round_offsets().round_lengths()

    write_win, local_win = recrop_write_window(write_win, overall_height,
                                               overall_width)

    pbar = tqdm(total=len(BANDS) * subtile_array.sizes["time"])

    for band_index, band in enumerate(BANDS):
        # using billinear resampling for spectral bands and nearest neighbor
        # resampling for everything else
        temp_arr = subtile_array.sel(band=band).rio.reproject(
            dst_crs=target_crs,
            resampling=Resampling.bilinear
            if band in BANDS else Resampling.nearest,
            transform=subtile_repr_transform,
            shape=(subtile_repr_height, subtile_repr_width),
            nodata=0)

        # TODO masked writing --> solution: write on seperate timestamps and then
        # do mean aggregate afterwards

        # change center to coordinates to top-left coords (rioxarray caveat)
        temp_arr = temp_arr.assign_coords(
            dict(x=temp_arr.x.data - (target_resolution / 2),
                 y=temp_arr.y.data + (target_resolution / 2)))

        temp_arr = temp_arr.compute()

        for ts_index, ts in enumerate(subtile_array.time.data):
            out_array[ts_index, band_index,
                      write_win.row_off:write_win.row_off + write_win.height,
                      write_win.col_off:write_win.col_off +
                      write_win.width] = temp_arr.loc[dict(
                          time=ts)][local_win.row_off:local_win.height +
                                    local_win.row_off,
                                    local_win.col_off:local_win.col_off +
                                    local_win.width]

            pbar.update()


def process(
        zarr_path: str,
        target_crs: CRS,
        target_resolution: float,
        bound_left: float,
        bound_bottom: float,
        bound_right: float,
        bound_top: float,
        datetime: DatetimeLike,
        subtile_size: int,
        num_cores: int = 1,
        kwargs_atenea: dict = dict(),
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
    kwargs_atenea: dict, default = None
        Arguments passed to atenea specifying processing steps that are applied to each tile.
    zarr_path: str
        Path where zarr storage is supposed to be created.
    """

    if zarr_path is None:
        # then return as xarray
        # otherwise store as zarr
        pass

    # TODO think about what to do with clouds and if to optionally filter them like atenea
    # --> mask_Clouds, drop_cloudy, clear_sky_threshold, mask_snow)

    # 1 obtain sub-sentinel tiles based on supplied bounds and CRS
    subtiles = obtain_subtiles(target_crs,
                               bound_left,
                               bound_bottom,
                               bound_right,
                               bound_top,
                               subtile_size=subtile_size)

    # 2 setup zarr storage
    # determine width and height based on bounds and resolution
    width, w_rem = divmod(abs(bound_right - bound_left), target_resolution)
    height, h_rem = divmod(abs(bound_top - bound_bottom), target_resolution)
    if h_rem > 0:
        warning.warn(
            "Specified top/bottom bounds are not perfectly divisable by specified target_resolution. The resulting coverage will be slightly cropped"
        )
    if w_rem > 0:
        warning.warn(
            "Specified left/right bounds are not perfectly divisable by specified target_resolution. The resulting coverage will be slightly cropped"
        )

    # sign into planetary computer
    stac_endpoint = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystac_client.Client.open(
        stac_endpoint,
        modifier=planetary_computer.sign_inplace,
    )

    # get all items within date range
    search = catalog.search(collections=["sentinel-2-l2a"],
                            datetime=datetime,
                            bbox=rasterio.warp.transform_bounds(
                                src_crs=target_crs,
                                dst_crs="EPSG:4326",
                                left=bound_left,
                                bottom=bound_bottom,
                                right=bound_right,
                                top=bound_top))

    items = pd.DataFrame()
    items["item"] = list(search.item_collection())
    items["tile"] = items["item"].apply(lambda x: x.properties["s2:mgrs_tile"])
    items["datetime"] = items["item"].apply(lambda x: x.datetime)
    items["id"] = items["item"].apply(lambda x: x.id)

    # determine overall transform from bounds
    overall_transform = transform.from_bounds(west=bound_left,
                                              south=bound_bottom,
                                              east=bound_right,
                                              north=bound_top,
                                              width=width,
                                              height=height)

    timesteps = items["datetime"].drop_duplicates().tolist()
    out_array = xr.DataArray(
        data=da.zeros((len(timesteps), len(BANDS), height, width),
                      chunks=(1, 12, 100, 100),
                      dtype=np.uint16),
        dims=["time", "band", "y", "x"],
        coords=dict(
            time=timesteps,
            band=BANDS,
            x=np.arange(bound_left, bound_right,
                        target_resolution).astype(np.float32),
            # we do y-axis in reverse for easier access with indexing
            y=np.arange(bound_top, bound_bottom,
                        -target_resolution).astype(np.float32)))

    # NOTE kind of magic values for now
    # one chunk per timestep!
    # chunks = (100, 100, 1)

    # TODO NEXT TODO understand the strucutre of xarray saved zarr arrays better and replicate here
    # create zarr storage for each band and mask
    #for band in BANDS:
    #    # TODO somehow specifying fill value fails here
    #    zarr.creation.empty(
    #        shape=(width, height, items["datetime"].nunique()),
    #        chunks=chunks,
    #        #fill_value=0,
    #        path=band,
    #        write_empty_chunks=False,
    #        store=store)

    # NOTE TEMP
    # subtiles = subtiles[:6]

    ns = subtiles.shape[0]
    subtiles = list(subtiles.itertuples(index=False, name="subtile"))
    for st in subtiles:
        process_subtile(subtile=st,
                        out_array=out_array,
                        atenea_args=kwargs_atenea,
                        subtile_size=subtile_size,
                        target_crs=target_crs,
                        target_resolution=target_resolution,
                        df=items[items["tile"] == st.name],
                        stac_endpoint=stac_endpoint,
                        overall_transform=overall_transform,
                        overall_width=width,
                        overall_height=height)

    # convert Timestamp object to UTC timestamp float so that it can be stored in zarr
    out_array = out_array.assign_coords(
        dict(time=[int(t.timestamp()) for t in out_array.time.data]))

    # out_array.assign_coords(dict(time=out_array.time.data))
    store = zarr.storage.DirectoryStore(zarr_path, dimension_separator=".")
    print(out_array)
    out_array.rename("S2").to_zarr(
        store=store,
        mode="w-",
        compute=True,
        encoding={"S2": {
            "write_empty_chunks": False
        }})

    # subtile_arrays = paral(process_subtile, [
    #     subtiles,
    #     [out_array] * ns,
    #     [kwargs_atenea] * ns,
    #     [subtile_size] * ns,
    #     [target_crs] * ns,
    #     [target_resolution] * ns,
    #     [items[items["tile"] == s.name] for s in subtiles],
    #     [stac_endpoint] * ns,
    #     [overall_transform] * ns,
    # ],
    #                        num_cores=num_cores)

    # 3 merge sub-sentinel tiles into one big sparse xarray and return
    # possible ISSUE: what happens when not everything fits into memory --> we
    # may want to have each subtile already being written to the harddrive once
    # it's done --> is that supported with ZARR?


x = process(target_crs=CRS.from_string("EPSG:8857"),
            bound_left=767300,
            bound_bottom=7290000,
            bound_right=776000,
            bound_top=7315000,
            datetime="2023-11-11/2023-12-01",
            subtile_size=732,
            target_resolution=10,
            zarr_path="bigout.zarr")
