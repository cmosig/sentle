import warnings
from math import ceil

import geopandas as gpd
import numpy as np
import pandas as pd
import pkg_resources
import zarr
from joblib import Parallel, delayed
from numcodecs import Blosc
from pystac_client.item_search import DatetimeLike
from rasterio import transform, warp, windows
from rasterio.crs import CRS
from tqdm.auto import tqdm

from .cloud_mask import S2_cloud_mask_band, S2_cloud_prob_bands
from .const import *
from .reproject_util import *
from .sentinel1 import process_ptile_S1
from .sentinel2 import obtain_subtiles, process_ptile_S2_dispatcher
from .stac import *
from .utils import tqdm_joblib


def catalog_search_ptile(collection: str, ts, time_composite_freq, bound_left,
                         bound_bottom, bound_right, bound_top,
                         target_crs) -> list:
    # timestamp
    if time_composite_freq is None:
        datetime_range = ts
    else:
        timestamp_center = ts
        datetime_range = [
            timestamp_center - (pd.Timedelta(time_composite_freq) / 2),
            timestamp_center + (pd.Timedelta(time_composite_freq) / 2)
        ]

    # open stac catalog
    catalog = open_catalog()

    # retrieve items (possible across multiple sentinel tile) for specified
    # timestamp
    item_list = list(
        catalog.search(collections=[collection],
                       datetime=datetime_range,
                       bbox=warp.transform_bounds(
                           src_crs=target_crs,
                           dst_crs="EPSG:4326",
                           left=bound_left,
                           bottom=bound_bottom,
                           right=bound_right,
                           top=bound_top)).item_collection())

    return item_list


def process_ptile(
    zarr_path,
    ts,
    bound_left,
    bound_bottom,
    bound_right,
    bound_top,
    collection,
    S2_bands_to_save,
    S1_assets,
    target_crs: CRS,
    target_resolution: float,
    S2_cloud_classification_device: str,
    time_composite_freq: str,
    S2_apply_snow_mask: bool,
    S2_apply_cloud_mask: bool,
    S2_mask_snow: bool,
    S2_cloud_classification: bool,
    S2_return_cloud_probabilities: bool,
    zarr_save_slice: dict,
    S2_subtiles,
):
    """Passing chunk to either sentinel-1 or sentinel-2 processor"""

    # determine ptile dimensions and transform from bounds
    ptile_transform, ptile_height, ptile_width = transform_height_width_from_bounds_res(
        bound_left, bound_bottom, bound_right, bound_top, target_resolution)

    # TODO too many unessary stac requests are created here
    # when not using aggregation across large spatial scales
    # -> this function is called for each timestamp that there was a sentinel 2
    # tile anywhere in the entire bounds
    # one could share the initial item list with all processes and the
    # processes filter these instead based on extent -> we have the dataframe
    # ready anyway
    item_list = catalog_search_ptile(
        collection=collection,
        ts=ts,
        time_composite_freq=time_composite_freq,
        bound_left=bound_left,
        bound_right=bound_right,
        bound_bottom=bound_bottom,
        bound_top=bound_top,
        target_crs=target_crs,
    )

    assert len(
        item_list
    ) > 0, "Number of retrieved stac items is zero even though we found stac items previously."

    if collection == "sentinel-1-rtc":
        ptile_array = process_ptile_S1(bound_left=bound_left,
                                       bound_bottom=bound_bottom,
                                       bound_right=bound_right,
                                       bound_top=bound_top,
                                       target_crs=target_crs,
                                       ts=ts,
                                       ptile_height=ptile_height,
                                       ptile_width=ptile_width,
                                       ptile_transform=ptile_transform,
                                       item_list=item_list,
                                       target_resolution=target_resolution,
                                       time_composite_freq=time_composite_freq,
                                       S1_assets=S1_assets)
    elif collection == "sentinel-2-l2a":
        ptile_array = process_ptile_S2_dispatcher(
            bound_left=bound_left,
            bound_bottom=bound_bottom,
            bound_right=bound_right,
            bound_top=bound_top,
            ts=ts,
            target_crs=target_crs,
            ptile_height=ptile_height,
            ptile_width=ptile_width,
            ptile_transform=ptile_transform,
            item_list=item_list,
            target_resolution=target_resolution,
            S2_cloud_classification=S2_cloud_classification,
            S2_cloud_classification_device=S2_cloud_classification_device,
            S2_mask_snow=S2_mask_snow,
            S2_return_cloud_probabilities=S2_return_cloud_probabilities,
            time_composite_freq=time_composite_freq,
            S2_apply_snow_mask=S2_apply_snow_mask,
            S2_apply_cloud_mask=S2_apply_cloud_mask,
            S2_bands_to_save=S2_bands_to_save,
            S2_subtiles=S2_subtiles,
        )

    else:
        assert False

    # if we want the data instead of saving it we can do that here and then
    # create the xarray object in the process function
    if ptile_array is not None:
        # save to zarr
        dst = zarr.open(zarr_path)["sentle"]
        dst[zarr_save_slice["time"], zarr_save_slice["band"],
            zarr_save_slice["y"], zarr_save_slice["x"]] = ptile_array


def process(
    target_crs: CRS,
    target_resolution: float,
    bound_left: float,
    bound_bottom: float,
    bound_right: float,
    bound_top: float,
    datetime: DatetimeLike,
    zarr_path: str,
    processing_spatial_chunk_size: int = 4000,
    S1_assets: list[str] = ["vh", "vv"],
    S2_mask_snow: bool = False,
    S2_cloud_classification: bool = False,
    S2_cloud_classification_device="cpu",
    S2_return_cloud_probabilities: bool = False,
    num_workers: int = 1,
    time_composite_freq: str = None,
    S2_apply_snow_mask: bool = False,
    S2_apply_cloud_mask: bool = False,
):
    """
    Parameters
    ----------

    target_crs : CRS
        Specifies the target CRS that all data will be reprojected to.
    target_resolution : float
        Determines the resolution that all data is reprojected to in the `target_crs`.
    bound_left : float
        Left bound of area that is supposed to be covered. Unit is in `target_crs`.
    bound_bottom : float
        Bottom bound of area that is supposed to be covered. Unit is in `target_crs`.
    bound_right : float
        Right bound of area that is supposed to be covered. Unit is in `target_crs`.
    bound_top : float
        Top bound of area that is supposed to be covered. Unit is in `target_crs`.
    datetime : DatetimeLike
        Specifies time range of data to be downloaded. This is forwarded to the respective stac interface.
    S2_mask_snow : bool, default=False
        Whether to create a snow mask. Based on https://doi.org/10.1016/j.rse.2011.10.028.
    S2_cloud_classification : bool, default=False
        Whether to create cloud classification layer, where `0=clear sky`, `2=thick cloud`, `3=thin cloud`, `4=shadow`.
    S2_cloud_classification_device : str, default="cpu"
        On which device to run cloud classification. Either `cpu` or `cuda`.
    S2_return_cloud_probabilities : bool, default=False
        Whether to return raw cloud probabilities which were used to determine the cloud classes.
    num_workers : int, default=1
        Number of cores to scale computation across. Plan 4GiB of RAM per worker.
    time_composite_freq: str, default=None
        Rounding interval across which data is averaged.
    S2_apply_snow_mask: bool, default=False
        Whether to replace snow with NaN.  
    S2_apply_cloud_mask: bool, default=False
        Whether to replace anything that is not clear sky with NaN.  
    zarr_path: str,
       Path of where to create the zarr storage.
    """

    if time_composite_freq is not None and (not S2_apply_snow_mask
                                            and not S2_apply_cloud_mask):
        warnings.warn(
            "Temporal aggregation is specified, but neither cloud or snow mask is set to be applied. This may yield useless aggregations for Sentinel-2 data."
        )

    # TODO support to only download subset of bands (mutually exclusive with
    # cloud classification and partially snow_mask) -> or no sentinel 2 at all
    # derive bands to save from arguments
    S2_bands_to_save = S2_RAW_BANDS.copy()
    if S2_mask_snow and time_composite_freq is None:
        S2_bands_to_save.append(S2_snow_mask_band)
    if S2_cloud_classification and time_composite_freq is None:
        S2_bands_to_save.append(S2_cloud_mask_band)
    if S2_return_cloud_probabilities:
        S2_bands_to_save += S2_cloud_prob_bands
    total_bands_to_save = S2_bands_to_save.copy()

    # if S1_assets are supplied as empty list, convert to None
    if isinstance(S1_assets, list) and len(S1_assets) == 0:
        S1_assets = None

    # sanity check for S1 bands
    if S1_assets is not None:
        assert len(set(S1_assets) -
                   set(["vv", "vh"])) == 0, "Unsupported S1 bands."
        total_bands_to_save += S1_assets.copy()

    # sign into planetary computer
    catalog = open_catalog()

    # get all items within date range and area
    collections = ["sentinel-2-l2a"]
    if S1_assets is not None:
        collections.append("sentinel-1-rtc")
    search = catalog.search(collections=collections,
                            datetime=datetime,
                            bbox=warp.transform_bounds(src_crs=target_crs,
                                                       dst_crs="EPSG:4326",
                                                       left=bound_left,
                                                       bottom=bound_bottom,
                                                       right=bound_right,
                                                       top=bound_top))

    # sort timesteps and filter duplicates -> multiple items can have the
    # exact same timestamp
    df = pd.DataFrame()
    items = list(search.item_collection())
    if len(items) == 0:
        print("No items found for specified time range and area.")
        return

    df["ts_raw"] = [i.datetime for i in items]
    df["collection"] = [i.collection_id for i in items]

    if time_composite_freq is not None:
        df["ts"] = df["ts_raw"].dt.round(freq=time_composite_freq)
    else:
        df["ts"] = df["ts_raw"]

    # remove duplicates for timeaxis
    df = df.drop_duplicates(["ts", "collection"])

    # get only one row for each final timestamp
    df = df.groupby("ts")[["collection"]].apply(
        lambda x: x["collection"].tolist()).rename("collection").reset_index()

    # make sure ts is sorted in the correct order
    df = df.sort_values("ts", ascending=False)

    # compute bounds, with and height  for the entire dataset
    bound_left, bound_bottom, bound_right, bound_top = check_and_round_bounds(
        bound_left, bound_bottom, bound_right, bound_top, target_resolution)
    height, width = height_width_from_bounds_res(bound_left, bound_bottom,
                                                 bound_right, bound_top,
                                                 target_resolution)

    # setup zarr storage
    store = zarr.storage.DirectoryStore(zarr_path, dimension_separator=".")

    # create array for where to store the processed sentinel data
    # chunk size is the number of S2 bands, because we parallelize S1/S2
    data = zarr.create(shape=(df["ts"].count(), len(total_bands_to_save),
                              height, width),
                       chunks=(1, len(S2_bands_to_save),
                               processing_spatial_chunk_size,
                               processing_spatial_chunk_size),
                       dtype=np.float32,
                       fill_value=float("nan"),
                       store=store,
                       path="/sentle",
                       write_empty_chunks=False,
                       compressor=Blosc(cname="lz4"))
    data.attrs.update(ZARR_DATA_ATTRS)

    # ------
    # arrays for storage of dimension information

    # band dimension
    band = zarr.create(shape=(len(total_bands_to_save)),
                       dtype="<U3",
                       store=store,
                       path="/band")
    band[:] = total_bands_to_save
    band.attrs.update(ZARR_BAND_ATTRS)

    # x dimension
    x = zarr.create(shape=(width), dtype="float32", store=store, path="/x")
    x[:] = np.arange(bound_left, bound_right,
                     target_resolution).astype(np.float32)
    x.attrs.update(ZARR_X_ATTRS)

    # y dimension
    y = zarr.create(shape=(height), dtype="float32", store=store, path="/y")
    y[:] = np.arange(bound_top, bound_bottom,
                     -target_resolution).astype(np.float32)
    y.attrs.update(ZARR_Y_ATTRS)

    # time dimension
    time = zarr.create(shape=(df["ts"].count()),
                       dtype="int64",
                       store=store,
                       path="/time",
                       fill_value=None)

    time[:] = (df["ts"].drop_duplicates().dt.tz_localize(tz=None) -
               pd.Timestamp(0, tz=None)).dt.days.tolist()
    time.attrs.update(ZARR_TIME_ATTRS)

    # consolidating metadata
    zarr.consolidate_metadata(store)

    # close up store as we are done with init
    store.close()

    # figure out jobs for multiprocessing -> one per chunk
    config = {
        "target_crs": target_crs,
        "target_resolution": target_resolution,
        "S2_cloud_classification_device": S2_cloud_classification_device,
        "time_composite_freq": time_composite_freq,
        "S2_apply_snow_mask": S2_apply_snow_mask,
        "S2_apply_cloud_mask": S2_apply_cloud_mask,
        "S2_cloud_classification": S2_cloud_classification,
        "S2_mask_snow": S2_mask_snow,
        "S2_return_cloud_probabilities": S2_return_cloud_probabilities,
        "zarr_path": zarr_path,
        "S2_bands_to_save": S2_bands_to_save,
        "S1_assets": S1_assets,
    }

    processing_spatial_chunk_size_in_CRS_unit = processing_spatial_chunk_size * target_resolution
    s2grid = gpd.read_file(
        pkg_resources.resource_filename(
            __name__, "data/sentinel2_grid_stripped_with_epsg.gpkg"))

    def job_generator():
        for xi, x_min in enumerate(
                range(bound_left, bound_right,
                      processing_spatial_chunk_size_in_CRS_unit)):
            for yi, y_min in enumerate(
                    range(bound_bottom, bound_top,
                          processing_spatial_chunk_size_in_CRS_unit)):
                for tsi, (_, ser) in enumerate(df[["ts",
                                                   "collection"]].iterrows()):
                    for collection in ser["collection"]:
                        ret_config = dict(config)
                        ret_config["bound_left"] = x_min
                        ret_config["bound_bottom"] = y_min
                        # cap at the top
                        ret_config["bound_right"] = min(
                            x_min + processing_spatial_chunk_size_in_CRS_unit,
                            bound_right)
                        ret_config["bound_top"] = min(
                            y_min + processing_spatial_chunk_size_in_CRS_unit,
                            bound_top)
                        ret_config["ts"] = ser["ts"]
                        ret_config["collection"] = collection
                        ret_config["zarr_save_slice"] = dict(
                            x=slice(
                                xi * processing_spatial_chunk_size,
                                min((xi + 1) * processing_spatial_chunk_size,
                                    width)),
                            y=slice(
                                yi * processing_spatial_chunk_size,
                                min((yi + 1) * processing_spatial_chunk_size,
                                    height)),
                            band=slice(0, len(S2_bands_to_save))
                            if collection == "sentinel-2-l2a" else slice(
                                len(S2_bands_to_save),
                                len(total_bands_to_save)),
                            time=tsi)
                        ret_config["S2_subtiles"] = obtain_subtiles(
                            target_crs=target_crs,
                            left=ret_config["bound_left"],
                            bottom=ret_config["bound_bottom"],
                            right=ret_config["bound_right"],
                            top=ret_config["bound_top"],
                            s2grid=s2grid,
                        ) if collection == "sentinel-2-l2a" else None
                        yield ret_config

    num_chunks = df["collection"].explode().count() * (ceil(
        width / processing_spatial_chunk_size)) * (ceil(
            height / processing_spatial_chunk_size))
    with tqdm_joblib(
            tqdm(desc="processing",
                 unit="ptiles",
                 dynamic_ncols=True,
                 total=num_chunks)) as progress_bar:
        # backend can be loky or threading (or maybe something else)
        return Parallel(n_jobs=num_workers,
                        batch_size=1,
                        backend="multiprocessing")(delayed(process_ptile)(**p)
                                                   for p in job_generator())
