import rioxarray as rxr
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


def process_subtile(subtile, datetime: DatetimeLike, zarrpath: str,
                    atenea_args: dict, subtile_size: int, target_crs: CRS,
                    target_resolution: float):

    stac_endpoint = "https://planetarycomputer.microsoft.com/api/stac/v1"

    # 1 download subtile with specified date range
    catalog = pystac_client.Client.open(
        stac_endpoint,
        modifier=planetary_computer.sign_inplace,
    )

    # get sentinel tiles within date range
    search = catalog.search(collections=["sentinel-2-l2a"],
                            datetime=datetime,
                            filter={
                                "filter": {
                                    "op":
                                    "=",
                                    "args": [{
                                        "property": "s2:mgrs_tile"
                                    }, subtile.name]
                                }
                            })

    items = list(search.item_collection())
    subtile_array = xr.DataArray(
        data=np.empty((len(items), len(BANDS), subtile_size, subtile_size)),
        dims=["time", "band", "y", "x"],
        coords=dict(time=[item.datetime for item in items],
                    band=BANDS,
                    id=("time", [item.id for item in items])),
        attrs=dict(stac=stac_endpoint, collection="sentinel-2-l2a"))

    # iterate through timestamps
    crs = None
    transform = None
    for item in items:
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
                subtile_array.loc[dict(time=item.datetime,
                                       band=band)] = dr.read(
                                           indexes=1,
                                           window=read_window,
                                           out_shape=(subtile_size,
                                                      subtile_size))

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
    subtile_bounds = windows.bounds(subtile.intersecting_windows, transform)
    assert (
        subtile_bounds[2] - subtile_bounds[0]
    ) // 10 == subtile_size, "mismatch between subtile size and bounds on x-axis"
    assert (
        subtile_bounds[3] - subtile_bounds[1]
    ) // 10 == subtile_size, "mismatch between subtile size and bounds on y-axis"

    # set array coordindates based on bounds and standard resolution of 10m
    subtile_array = subtile_array.assign_coords(
        dict(x=np.arange(start=subtile_bounds[0],
                         stop=subtile_bounds[2],
                         step=10),
             y=np.arange(start=subtile_bounds[1],
                         stop=subtile_bounds[3],
                         step=10))).compute()

    # 2 push that tile through atenea
    # TODO add atenea kwargs
    subtile_array = atenea.process(
        subtile_array,
        source="cubo",
        # TODO need to add padding and then reactivate cloud filtering
        mask_clouds=False,
        return_cloud_classification_layer=True,
        # chunksize=(len(items), subtile_size),
        stac=stac_endpoint).compute()

    # make sure that x and y are the correct spatial resolutions
    subtile_array = subtile_array.rio.set_spatial_dims(x_dim="x",
                                                       y_dim="y").compute()

    # 3 reproject to target_crs for each band
    # using billinear resampling for spectral bands and nearest neighbor
    # resampling for everything else
    subtile_array = xr.concat([
        subtile_array.sel(band=band).rio.reproject(
            dst_crs=target_crs,
            resolution=target_resolution,
            resampling=Resampling.bilinear
            if band in BANDS else Resampling.nearest)
        for band in subtile_array.band.data
    ],
                              dim="band").compute()

    return subtile_array

    # 4 remove data that is outside the bounds of the specified bounds

    # NOTE 1: think about the storage type that we want to use, definetly not Directory --> too many files
    # NOTE 2: timestamps: either we need to round to the number of days and then the
    # zarr array has one timestep per day --> possibly completely empty
    # timesteps
    # determine in advance using the stac api which timestamps we have across
    # the entire area and then this will be the timesteps
    # NOTE 3: also "stupid mode" simply overwrite timestamps where there
    # already is data -> should be the same anyway
    # NOTE 4: somehow prevent that places where this array has no data (nans)
    # are not used to overwrite eixisting data -> solution masked indexing
    exit()


def process(target_crs: CRS,
            target_resolution: float,
            bound_left: float,
            bound_bottom: float,
            bound_right: float,
            bound_top: float,
            datetime: DatetimeLike,
            subtile_size: int,
            num_cores: int = 1,
            kwargs_atenea: dict = dict(),
            zarr_path: str = None):
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
    zarr_path: str, default = None
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

    # create zarr storage
    # for each band and mask
    # TODO which dtype?
    # store = zarr.storage.SQLiteStore(zarr_path, dimension_separator=".")
    # for band in BANDS:
    #     zarr.creation.empty(shape=(width, height, timeseries_length),
    #                         chunks=(chunk_width, chunk_height,
    #                                 chunk_timeseries),
    #                         path=band,
    #                         fill_value=0,
    #                         write_empty_chunks=False,
    #                         store=store)

    # 2 in parallel for each sub-sentinel tile
    # 2a download each sub-sentinel tile from planetary computer
    # 2b run each sub-sentinel tile through atenea

    # NOTE TEMP
    subtiles = subtiles[:1]

    ns = subtiles.shape[0]
    subtile_arrays = paral(process_subtile, [
        list(subtiles.itertuples(index=False, name="subtile")),
        [datetime] * ns, [zarr_path] * ns, [kwargs_atenea] * ns,
        [subtile_size] * ns, [target_crs] * ns, [target_resolution] * ns
    ],
                           num_cores=num_cores)

    return subtile_arrays

    # need to close
    # store.close()

    # 3 merge sub-sentinel tiles into one big sparse xarray and return
    # possible ISSUE: what happens when not everything fits into memory --> we
    # may want to have each subtile already being written to the harddrive once
    # it's done --> is that supported with ZARR?


# process(
#     CRS.from_string("EPSG:8857"),
#     bound_left=767300,
#     bound_bottom=7290000,
#     bound_right=776000,
#     bound_top=7315000,
#     datetime="2023-11-22/2023-12-01",
#     subtile_size=732,
# )
