import gc
import multiprocessing as mp
import shutil
import tempfile
from time import time as currenttime
import warnings
from math import ceil
from os import path

import geopandas as gpd
import numcodecs
import numpy as np
import pandas as pd
import pkg_resources
import zarr
from joblib import Parallel, delayed, parallel_backend
from numcodecs import Blosc
from pystac_client.item_search import DatetimeLike
from rasterio import transform, warp, windows
from rasterio.crs import CRS
from tqdm.auto import tqdm
from zarr.sync import ProcessSynchronizer

from .cloud_mask import (S2_cloud_mask_band, S2_cloud_prob_bands,
                         init_cloud_prediction_service)
from .const import *
from .reproject_util import *
from .sentinel1 import process_ptile_S1
from .sentinel2 import obtain_subtiles, process_ptile_S2_dispatcher
from .stac import *
from .utils import *


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
    zarr_store,
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
    cloud_request_queue: mp.Queue,
    cloud_response_queue: mp.Queue,
    job_id: int,
    sync_file_path: str,
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
            cloud_request_queue=cloud_request_queue,
            cloud_response_queue=cloud_response_queue,
        )

    else:
        assert False

    # NOTE if we want the data instead of saving it we can do that here and then
    # create the xarray object in the process function

    # only save to zarr if we have data
    if ptile_array is not None and not np.isnan(ptile_array).all():

        # save to zarr
        dst = zarr.open(
            zarr_store,
            synchronizer=ProcessSynchronizer(sync_file_path))["sentle"]
        dst[zarr_save_slice["time"], zarr_save_slice["band"],
            zarr_save_slice["y"], zarr_save_slice["x"]] = ptile_array

    return job_id


def validate_user_input(target_crs: CRS,
                        target_resolution: float,
                        zarr_store: str | zarr.storage.Store,
                        bound_left: float,
                        bound_bottom: float,
                        bound_right: float,
                        bound_top: float,
                        zarr_store_chunk_size: dict,
                        datetime: DatetimeLike,
                        processing_spatial_chunk_size: int = 4000,
                        S1_assets: list[str] = S1_ASSETS,
                        S2_mask_snow: bool = False,
                        S2_cloud_classification: bool = False,
                        S2_cloud_classification_device="cpu",
                        S2_return_cloud_probabilities: bool = False,
                        num_workers: int = 1,
                        time_composite_freq: str = None,
                        S2_apply_snow_mask: bool = False,
                        S2_apply_cloud_mask: bool = False):

    # validate type zarr store
    if not isinstance(zarr_store, (str, zarr.storage.Store)):
        raise ValueError("zarr_store must be a string or zarr.storage.Store")

    # check if the target CRS is valid
    if not isinstance(target_crs, CRS):
        raise ValueError("target_crs must be an instance of rasterio.crs.CRS")

    # check if the target resolution is valid
    if not isinstance(target_resolution, (int, float)):
        raise ValueError("target_resolution must be an integer or float")

    # check if resolution greater than 0
    if target_resolution <= 0:
        raise ValueError("target_resolution must be greater than 0")

    # check if the bounds are valid
    if not all(
            isinstance(bound, (int, float))
            for bound in [bound_left, bound_bottom, bound_right, bound_top]):
        raise ValueError(
            "bound_left, bound_bottom, bound_right, and bound_top must be integers or floats"
        )

    # check if the bounds make sense
    if bound_left >= bound_right:
        raise ValueError("bound_left must be less than bound_right")
    if bound_bottom >= bound_top:
        raise ValueError("bound_bottom must be less than bound_top")

    # check processsing_spatial_chunk_size
    if not isinstance(processing_spatial_chunk_size, int):
        raise ValueError("processing_spatial_chunk_size must be an integer")
    # check if chunk size is greater than 1000
    if processing_spatial_chunk_size < 1000:
        raise ValueError(
            "processing_spatial_chunk_size must be greater than 1000")

    if time_composite_freq is not None and (not S2_apply_snow_mask
                                            and not S2_apply_cloud_mask):
        warnings.warn(
            "Temporal aggregation is specified, but neither cloud or snow mask is set to be applied. This may yield useless aggregations for Sentinel-2 data."
        )

    # validate that S1_assets is a list and contains only vv and vh
    if S1_assets is not None:
        if not isinstance(S1_assets, list):
            raise ValueError("S1_assets must be a list")

        if not all(isinstance(asset, str) for asset in S1_assets):
            raise ValueError("S1_assets must contain only strings")

        if not all(asset in S1_ASSETS for asset in S1_assets):
            raise ValueError(
                "S1_assets must contain only 'vh_asc', 'vv_asc', 'vh_desc', 'vv_desc'"
            )

    # check if S2_mask_snow is a boolean
    if not isinstance(S2_mask_snow, bool):
        raise ValueError("S2_mask_snow must be a boolean")

    # check if S2_cloud_classification is a boolean
    if not isinstance(S2_cloud_classification, bool):
        raise ValueError("S2_cloud_classification must be a boolean")

    # check if S2_cloud_classification_device is a string
    if not isinstance(S2_cloud_classification_device, str):
        raise ValueError("S2_cloud_classification_device must be a string")
    # check if S2_cloud_classification_device is either cpu or cuda
    if S2_cloud_classification_device not in ["cpu", "cuda"]:
        raise ValueError(
            "S2_cloud_classification_device must be either 'cpu' or 'cuda'")

    # check if S2_return_cloud_probabilities is a boolean
    if not isinstance(S2_return_cloud_probabilities, bool):
        raise ValueError("S2_return_cloud_probabilities must be a boolean")

    # check if num_workers is an integer
    if not isinstance(num_workers, int):
        raise ValueError("num_workers must be an integer")

    # check if time_composite_freq is a string
    if time_composite_freq is not None and not isinstance(
            time_composite_freq, str):
        raise ValueError("time_composite_freq must be a string")

    # check if S2_apply_snow_mask is a boolean
    if not isinstance(S2_apply_snow_mask, bool):
        raise ValueError("S2_apply_snow_mask must be a boolean")

    # check if S2_apply_cloud_mask is a boolean
    if not isinstance(S2_apply_cloud_mask, bool):
        raise ValueError("S2_apply_cloud_mask must be a boolean")

    # check if combinations of booleans make sense
    if S2_apply_snow_mask and not S2_mask_snow:
        raise ValueError(
            "S2_apply_snow_mask is set to True, but S2_mask_snow is set to False"
        )

    if S2_apply_cloud_mask and not S2_cloud_classification:
        raise ValueError(
            "S2_apply_cloud_mask is set to True, but S2_cloud_classification is set to False"
        )

    # make sure that zarr_store_chunk_size is a dictionary and has time, y, and x keys
    if not isinstance(zarr_store_chunk_size, dict):
        raise ValueError("zarr_store_chunk_size must be a dictionary")
    if not all(key in zarr_store_chunk_size for key in ["time", "y", "x"]):
        raise ValueError(
            "zarr_store_chunk_size must contain the keys 'time', 'y', and 'x'")


def setup_zarr_storage(zarr_store: str | zarr.storage.Store,
                       df: pd.DataFrame,
                       height: int,
                       width: int,
                       bound_left: float,
                       bound_right: float,
                       bound_top: float,
                       bound_bottom: float,
                       target_resolution: float,
                       processing_spatial_chunk_size: int,
                       zarr_store_chunk_size: dict,
                       S2_bands_to_save: list[str],
                       total_bands_to_save: list[str],
                       overwrite: bool = False) -> None:

    if isinstance(zarr_store, str):
        # setup zarr storage
        store = zarr.storage.DirectoryStore(zarr_store,
                                            dimension_separator=".")
    else:
        store = zarr_store

    sync_file_path = None
    if not (zarr_store_chunk_size["time"] == 1
            and zarr_store_chunk_size["y"] == processing_spatial_chunk_size
            and zarr_store_chunk_size["x"] == processing_spatial_chunk_size):
        numcodecs.blosc.use_threads = False

        # get a uuid for sync file
        sync_file_path = path.join(tempfile.gettempdir(),
                                   f"sentle_{currenttime()}.sync")

    # create array for where to store the processed sentinel data
    # chunk size is the number of S2 bands, because we parallelize S1/S2
    data = zarr.create(
        shape=(df["ts"].count(), len(total_bands_to_save), height, width),
        chunks=(zarr_store_chunk_size["time"], len(S2_bands_to_save),
                zarr_store_chunk_size["y"], zarr_store_chunk_size["x"]),
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
                       dtype="<U32",
                       store=store,
                       path="/band",
                       overwrite=overwrite)
    band[:] = total_bands_to_save
    band.attrs.update(ZARR_BAND_ATTRS)

    # x dimension
    x = zarr.create(shape=(width),
                    dtype="float32",
                    store=store,
                    path="/x",
                    overwrite=overwrite)
    x[:] = np.arange(bound_left, bound_right,
                     target_resolution).astype(np.float32)
    x.attrs.update(ZARR_X_ATTRS)

    # y dimension
    y = zarr.create(shape=(height),
                    dtype="float32",
                    store=store,
                    path="/y",
                    overwrite=overwrite)
    y[:] = np.arange(bound_top, bound_bottom,
                     -target_resolution).astype(np.float32)
    y.attrs.update(ZARR_Y_ATTRS)

    # time dimension
    time = zarr.create(shape=(df["ts"].count()),
                       dtype="int64",
                       store=store,
                       path="/time",
                       fill_value=None,
                       overwrite=overwrite)

    time[:] = (df["ts"].drop_duplicates().dt.tz_localize(tz=None) -
               pd.Timestamp(0, tz=None)).dt.days.tolist()
    time.attrs.update(ZARR_TIME_ATTRS)

    # consolidating metadata
    zarr.consolidate_metadata(store)

    # close up store as we are done with init
    store.close()

    return sync_file_path


def process(
    target_crs: CRS,
    target_resolution: float,
    bound_left: float,
    bound_bottom: float,
    bound_right: float,
    bound_top: float,
    datetime: DatetimeLike,
    zarr_store: str | zarr.storage.Store,
    processing_spatial_chunk_size: int = 4000,
    S1_assets: list[str] = S1_ASSETS,
    S2_mask_snow: bool = False,
    S2_cloud_classification: bool = False,
    S2_cloud_classification_device="cpu",
    S2_return_cloud_probabilities: bool = False,
    num_workers: int = 1,
    time_composite_freq: str = None,
    S2_apply_snow_mask: bool = False,
    S2_apply_cloud_mask: bool = False,
    overwrite: bool = False,
    zarr_store_chunk_size: dict = {
        "time": 50,
        "x": 100,
        "y": 100,
    },
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
        Number of cores to scale computation across. Plan 2GiB of RAM per worker. -1 uses all available cores.
    time_composite_freq: str, default=None
        Rounding interval across which data is averaged.
    S2_apply_snow_mask: bool, default=False
        Whether to replace snow with NaN.  
    S2_apply_cloud_mask: bool, default=False
        Whether to replace anything that is not clear sky with NaN.  
    zarr_store: str | zarr.storage.Store
       Path of where to create the zarr storage.
    processing_spatial_chunk_size: int, default=4000
       Size of spatial chunks across which we perform parallization.
    S1_assets: list[str], default=["vh_asc", "vh_desc", "vv_asc", "vv_desc"]
       Specify which bands to download for Sentinel-1. Only "vh" and "vv" are supported.
    overwrite: bool, default=False
       Whether to overwrite existing zarr storage. 
    """

    validate_user_input(
        target_crs=target_crs,
        target_resolution=target_resolution,
        bound_left=bound_left,
        bound_bottom=bound_bottom,
        bound_right=bound_right,
        bound_top=bound_top,
        datetime=datetime,
        processing_spatial_chunk_size=processing_spatial_chunk_size,
        S1_assets=S1_assets,
        S2_mask_snow=S2_mask_snow,
        S2_cloud_classification=S2_cloud_classification,
        S2_cloud_classification_device=S2_cloud_classification_device,
        S2_return_cloud_probabilities=S2_return_cloud_probabilities,
        num_workers=num_workers,
        zarr_store=zarr_store,
        time_composite_freq=time_composite_freq,
        S2_apply_snow_mask=S2_apply_snow_mask,
        S2_apply_cloud_mask=S2_apply_cloud_mask,
        zarr_store_chunk_size=zarr_store_chunk_size,
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

    if S1_assets is not None:
        total_bands_to_save += S1_assets

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
    sync_file_path = setup_zarr_storage(
        zarr_store=zarr_store,
        df=df,
        height=height,
        width=width,
        bound_left=bound_left,
        bound_bottom=bound_bottom,
        bound_right=bound_right,
        bound_top=bound_top,
        target_resolution=target_resolution,
        zarr_store_chunk_size=zarr_store_chunk_size,
        S2_bands_to_save=S2_bands_to_save,
        total_bands_to_save=total_bands_to_save,
        processing_spatial_chunk_size=processing_spatial_chunk_size)

    cloud_request_queue = None
    if S2_cloud_classification:
        global GLOBAL_QUEUE_MANAGER

        GLOBAL_QUEUE_MANAGER, cloud_request_queue = init_cloud_prediction_service(
            device=S2_cloud_classification_device)

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
        "zarr_store": zarr_store,
        "S2_bands_to_save": S2_bands_to_save,
        "S1_assets": S1_assets,
        "cloud_request_queue": cloud_request_queue,
        "sync_file_path": sync_file_path,
    }

    processing_spatial_chunk_size_in_CRS_unit = processing_spatial_chunk_size * target_resolution
    s2grid = gpd.read_file(
        pkg_resources.resource_filename(
            __name__, "data/sentinel2_grid_stripped_with_epsg.gpkg"))

    def job_generator():
        global GLOBAL_QUEUE_MANAGER
        global GLOBAL_QUEUES
        job_id = 0

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
                        if S2_cloud_classification and collection == "sentinel-2-l2a":
                            GLOBAL_QUEUES[job_id] = GLOBAL_QUEUE_MANAGER.Queue(
                                maxsize=1)
                            ret_config["cloud_response_queue"] = GLOBAL_QUEUES[
                                job_id]
                            ret_config["job_id"] = job_id
                            job_id += 1
                        else:
                            ret_config["cloud_response_queue"] = None
                            ret_config["job_id"] = None
                        yield ret_config

    num_chunks = df["collection"].explode().count() * (ceil(
        width / processing_spatial_chunk_size)) * (ceil(
            height / processing_spatial_chunk_size))
    with tqdm_joblib(
            tqdm(desc="processing",
                 unit="ptiles",
                 dynamic_ncols=True,
                 total=num_chunks)) as progress_bar:

        with parallel_backend("cleanupqueue"):
            # backend can be loky or threading (or maybe something else)
            Parallel(n_jobs=num_workers,
                     batch_size=1)(delayed(process_ptile)(**p)
                                   for p in job_generator())

    # close cloud queue
    if S2_cloud_classification:
        # end cloud prediction service by sending None to queue
        cloud_request_queue.put(None)

        # close response queues manager
        GLOBAL_QUEUE_MANAGER.shutdown()

        GLOBAL_QUEUE_MANAGER = None
        GLOBAL_QUEUES = dict()

    # try to remove sync file
    if sync_file_path is not None:
        try:
            shutil.rmtree(sync_file_path)
        except Exception:
            pass

    # clean up
    gc.collect()
