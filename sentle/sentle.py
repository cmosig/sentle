import gc
import multiprocessing as mp
import shutil
import tempfile
import warnings
from os import path
from time import time as currenttime

import geopandas as gpd
import numpy as np
import pandas as pd
import pkg_resources
import zarr
import zarr.storage
from joblib import Parallel, delayed, parallel_backend
from pystac_client.item_search import DatetimeLike
from rasterio import warp
from rasterio.crs import CRS
from rasterio.enums import Resampling
from tqdm.auto import tqdm
from filelock import FileLock

from .cloud_mask import (
    S2_cloud_mask_band,
    S2_cloud_prob_bands,
    init_cloud_prediction_service,
)
from .snow_mask import S2_snow_mask_band
from .const import S1_ASSETS, S2_NBAR_BANDS, S2_RAW_BANDS, ZARR_TIME_ATTRS
from .reproject_util import (
    check_and_round_bounds,
    height_width_from_bounds_res,
    spatial_chunk_grid,
    transform_height_width_from_bounds_res,
)
from .sentinel1 import process_ptile_S1
from .sentinel2 import obtain_subtiles, process_ptile_S2_dispatcher
from .stac import get_provider
from .utils import GLOBAL_QUEUE_MANAGER, GLOBAL_QUEUES, tqdm_joblib


def catalog_search_ptile(
    collection: str,
    ts,
    time_composite_freq,
    bound_left,
    bound_bottom,
    bound_right,
    bound_top,
    target_crs,
    provider,
) -> list:
    # timestamp
    if time_composite_freq is None:
        datetime_range = ts
    else:
        timestamp_center = ts
        datetime_range = [
            timestamp_center - (pd.Timedelta(time_composite_freq) / 2),
            timestamp_center + (pd.Timedelta(time_composite_freq) / 2),
        ]

    # open stac catalog
    catalog = provider.open_catalog()

    # retrieve items (possible across multiple sentinel tile) for specified
    # timestamp
    item_list = list(
        catalog.search(
            collections=[collection],
            datetime=datetime_range,
            bbox=warp.transform_bounds(
                src_crs=target_crs,
                dst_crs="EPSG:4326",
                left=bound_left,
                bottom=bound_bottom,
                right=bound_right,
                top=bound_top,
            ),
        ).item_collection())

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
    S2_bands,
    S1_assets,
    target_crs: CRS,
    target_resolution: float,
    S2_cloud_classification_device: str,
    time_composite_freq: str,
    time_composite_method: str,
    S2_apply_snow_mask: bool,
    S2_apply_cloud_mask: bool,
    S2_mask_snow: bool,
    S2_cloud_classification: bool,
    S2_nbar: bool,
    S2_return_cloud_probabilities: bool,
    zarr_save_slice: dict,
    S2_subtiles,
    cloud_request_queue: mp.Queue,
    cloud_response_queue: mp.Queue,
    job_id: int,
    sync_file_path: str,
    resampling_method: Resampling,
    save_as_uint16: bool,
    provider,
    reuse_open_datasets: bool,
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
        provider=provider,
    )

    if len(item_list) == 0:
        # no items found for this ptile, this happens sometimes. planetary problem. dunno why.
        return job_id

    if collection == "sentinel-1-rtc":
        ptile_array = process_ptile_S1(
            bound_left=bound_left,
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
            time_composite_method=time_composite_method,
            S1_assets=S1_assets,
            resampling_method=resampling_method,
        )
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
            S2_nbar=S2_nbar,
            time_composite_freq=time_composite_freq,
            time_composite_method=time_composite_method,
            S2_apply_snow_mask=S2_apply_snow_mask,
            S2_apply_cloud_mask=S2_apply_cloud_mask,
            S2_bands_to_save=S2_bands_to_save,
            S2_bands=S2_bands,
            S2_subtiles=S2_subtiles,
            cloud_request_queue=cloud_request_queue,
            cloud_response_queue=cloud_response_queue,
            resampling_method=resampling_method,
            provider=provider,
            reuse_open_datasets=reuse_open_datasets,
        )

    else:
        assert False

    # NOTE if we want the data instead of saving it we can do that here and then
    # create the xarray object in the process function

    # only save to zarr if we have data
    if ptile_array is not None and not np.isnan(ptile_array).all():
        if save_as_uint16:
            if collection != "sentinel-2-l2a":
                raise ValueError(
                    "save_as_uint16 can only be used when Sentinel-1 processing is disabled."
                )
            ptile_array = np.nan_to_num(ptile_array, nan=0)
            # round to nearest integer before casting to uint16 to avoid truncation artifacts
            ptile_array = np.rint(ptile_array, out=ptile_array)
            np.clip(ptile_array, 0, np.iinfo(np.uint16).max, out=ptile_array)
            if ptile_array.dtype != np.uint16:
                ptile_array = ptile_array.astype(np.uint16)

        # save to zarr
        lock = FileLock(sync_file_path)
        with lock:
            dst = zarr.open(zarr_store)["sentle"]
            dst[
                zarr_save_slice["time"],
                zarr_save_slice["band"],
                zarr_save_slice["y"],
                zarr_save_slice["x"],
            ] = ptile_array

    return job_id


def validate_user_input(
    target_crs: CRS | str,
    target_resolution: float,
    zarr_store: str | zarr.storage.StoreLike,
    bound_left: float,
    bound_bottom: float,
    bound_right: float,
    bound_top: float,
    zarr_store_chunk_size: dict,
    datetime: DatetimeLike,
    S2_nbar: bool,
    resampling_method: Resampling,
    processing_spatial_chunk_size: int = 4000,
    S1_assets: list[str] = S1_ASSETS,
    S2_mask_snow: bool = False,
    S2_cloud_classification: bool = False,
    S2_cloud_classification_device="cpu",
    S2_return_cloud_probabilities: bool = False,
    num_workers: int = 1,
    time_composite_freq: str = None,
    time_composite_method: str = "mean",
    S2_apply_snow_mask: bool = False,
    S2_apply_cloud_mask: bool = False,
    save_as_uint16: bool = False,
    S2_bands: list[str] = S2_RAW_BANDS,
    provider: str = "planetary_computer",
):
    # validate the data provider and its constraints
    if provider not in ("planetary_computer", "cdse"):
        raise ValueError(
            f"provider must be 'planetary_computer' or 'cdse', got "
            f"{provider!r}")
    if provider == "cdse":
        sentinel1_enabled = isinstance(S1_assets, list) and len(S1_assets) > 0
        if sentinel1_enabled:
            raise ValueError(
                "provider='cdse' does not offer Sentinel-1 RTC; set "
                "S1_assets=None (CDSE support is Sentinel-2 only)")
        if S2_nbar:
            raise ValueError(
                "provider='cdse' does not support S2_nbar yet (sen2nbar reads "
                "the granule metadata over HTTP, which CDSE serves from S3); "
                "set S2_nbar=False")

    # validate type zarr store
    # ``zarr.storage.StoreLike`` is a typing union that includes a
    # parameterized generic (``dict[str, Buffer]``), which cannot be used
    # directly in ``isinstance`` (it raises ``TypeError``). Restrict the check
    # to the union members that are concrete classes so an invalid store type
    # yields a clean ``ValueError`` instead.
    zarr_store_types = tuple(
        t for t in zarr.storage.StoreLike.__args__ if isinstance(t, type))
    if not isinstance(zarr_store, zarr_store_types):
        raise ValueError(
            "zarr_store must be a string or zarr.storage.StoreLike")

    # check if the target CRS is valid, allow string and convert if needed
    try:
        target_crs = CRS.from_user_input(target_crs)
    except Exception as e:
        raise ValueError(
            f"target_crs string must by interpretable with rasterio.crs.CRS.from_user_input(), for example 'EPSG:3031': {e}"
        )

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

    # check the temporal aggregation method
    _valid_composite_methods = ("mean", "median", "min", "max")
    if time_composite_method not in _valid_composite_methods:
        raise ValueError(
            f"time_composite_method must be one of {_valid_composite_methods}, "
            f"got {time_composite_method!r}")

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

    # validate that S2_nbar is a boolean
    if not isinstance(S2_nbar, bool):
        raise ValueError("S2_nbar must be a boolean")

    if not isinstance(resampling_method, Resampling):
        raise ValueError(
            "resampling_method must be a rasterio.enums.Resampling")

    if not isinstance(save_as_uint16, bool):
        raise ValueError("save_as_uint16 must be a boolean")

    if save_as_uint16:
        sentinel1_disabled = (S1_assets
                              is None) or (isinstance(S1_assets, list)
                                           and len(S1_assets) == 0)
        if not sentinel1_disabled:
            raise ValueError(
                "save_as_uint16 can only be used when Sentinel-1 is disabled (set S1_assets to None or an empty list)."
            )

    # validate the Sentinel-2 band selection. ``S2_bands`` mirrors the way
    # ``S1_assets`` works:
    #   <all bands> (default) -> all bands
    #   [...]                 -> a subset
    #   None or []            -> Sentinel-2 disabled (Sentinel-1 only)
    s2_disabled = S2_bands is None or (isinstance(S2_bands, list)
                                       and len(S2_bands) == 0)
    if s2_disabled:
        # Sentinel-2 disabled -> require Sentinel-1 and no S2-only options
        sentinel1_enabled = isinstance(S1_assets, list) and len(S1_assets) > 0
        if not sentinel1_enabled:
            raise ValueError(
                "S2_bands=None or [] disables Sentinel-2, which requires "
                "Sentinel-1 to be enabled (pass S1_assets); otherwise there is "
                "nothing to download")

        s2_only_flags = {
            "S2_mask_snow": S2_mask_snow,
            "S2_cloud_classification": S2_cloud_classification,
            "S2_return_cloud_probabilities": S2_return_cloud_probabilities,
            "S2_apply_snow_mask": S2_apply_snow_mask,
            "S2_apply_cloud_mask": S2_apply_cloud_mask,
            "S2_nbar": S2_nbar,
            "save_as_uint16": save_as_uint16,
        }
        enabled = [name for name, val in s2_only_flags.items() if val]
        if enabled:
            raise ValueError(
                f"S2_bands=None or [] disables Sentinel-2 and is incompatible "
                f"with Sentinel-2 options {enabled}; disable them for a "
                f"Sentinel-1-only run")
    else:
        if not isinstance(S2_bands, list) or not all(
                isinstance(b, str) for b in S2_bands):
            raise ValueError("S2_bands must be a list of band-name strings")

        unknown = [b for b in S2_bands if b not in S2_RAW_BANDS]
        if unknown:
            raise ValueError(
                f"S2_bands contains unknown bands {unknown}; valid bands "
                f"are {S2_RAW_BANDS}")

        # cloud classification feeds all 12 raw bands into the model, so a
        # subset is not allowed together with it
        if S2_cloud_classification and set(S2_bands) != set(S2_RAW_BANDS):
            raise ValueError(
                "S2_bands cannot be a subset when S2_cloud_classification "
                "is True (the cloud model requires all bands)")

        # snow mask is computed from B03/B08/B11
        if S2_mask_snow:
            missing = [b for b in ("B03", "B08", "B11") if b not in S2_bands]
            if missing:
                raise ValueError(
                    f"S2_bands must include {missing} when S2_mask_snow is "
                    f"True")

        # NBAR corrects a fixed set of bands
        if S2_nbar:
            missing = [b for b in S2_NBAR_BANDS if b not in S2_bands]
            if missing:
                raise ValueError(
                    f"S2_bands must include {missing} when S2_nbar is True")


def setup_zarr_storage(
    zarr_store: str | zarr.storage.StoreLike,
    timestamp_list: list[str],
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
    target_crs: CRS,
    consolidate_metadata: bool,
    overwrite: bool = False,
    coord_save_mode: str = "top-left",
    save_as_uint16: bool = False,
) -> None | str:
    """
    Parameters
    ----------
    coord_save_mode : str, default="top-left"
        Specifies how coordinates are saved in zarr.
        - "top-left": coordinates represent the top-left corner of each pixel (default, current behavior).
        - "center": coordinates represent the center of each pixel (shifted by half a pixel).
    """

    if isinstance(zarr_store, str):
        # setup zarr storage
        store = zarr.storage.LocalStore(zarr_store)
    else:
        store = zarr_store

    # Save CRS as WKT attribute to the Zarr root group
    root = zarr.group(store=store)
    root.attrs["crs_wkt"] = target_crs.to_wkt()

    sync_file_path = None
    if not (zarr_store_chunk_size["time"] == 1
            and zarr_store_chunk_size["y"] == processing_spatial_chunk_size
            and zarr_store_chunk_size["x"] == processing_spatial_chunk_size):
        # get a uuid for sync file
        sync_file_path = path.join(tempfile.gettempdir(),
                                   f"sentle_{currenttime()}.lock")

    # create array for where to store the processed sentinel data
    # chunk size is the number of S2 bands, because we parallelize S1/S2
    unique_timestamps = sorted(set(item["ts"] for item in timestamp_list),
                               reverse=True)
    data_dtype = np.uint16 if save_as_uint16 else np.float32
    data_fill_value = 0 if save_as_uint16 else float("nan")

    # the band axis is chunked at the number of S2 bands so S1 and S2 write
    # into disjoint band chunks. When S2 is disabled there are no S2 bands, so
    # fall back to a single chunk spanning all (S1) bands.
    band_chunk = len(S2_bands_to_save) if len(S2_bands_to_save) > 0 else len(
        total_bands_to_save)
    data = zarr.create(
        shape=(len(unique_timestamps), len(total_bands_to_save), height,
               width),
        chunks=(
            zarr_store_chunk_size["time"],
            band_chunk,
            zarr_store_chunk_size["y"],
            zarr_store_chunk_size["x"],
        ),
        dtype=data_dtype,
        fill_value=data_fill_value,
        store=store,
        path="/sentle",
        config=dict(write_empty_chunks=False),
        dimension_names=["time", "band", "y", "x"],
    )
    # CF grid-mapping so downstream tools (notably rioxarray's ``ds.rio.crs``)
    # pick up the CRS automatically. ``grid_mapping`` points at the scalar
    # ``spatial_ref`` variable created below; ``coordinates`` makes a plain
    # ``xr.open_zarr(...)`` (i.e. without ``decode_coords="all"``) promote
    # ``spatial_ref`` to a coordinate so the CRS is found. See issue #58.
    data.attrs["grid_mapping"] = "spatial_ref"
    data.attrs["coordinates"] = "spatial_ref"

    # scalar CRS-holder variable (the CF grid_mapping variable)
    spatial_ref = zarr.create(
        shape=(),
        dtype="int64",
        store=store,
        path="/spatial_ref",
        overwrite=overwrite,
        dimension_names=[],
    )
    spatial_ref[...] = 0
    spatial_ref.attrs["crs_wkt"] = target_crs.to_wkt()
    spatial_ref.attrs["spatial_ref"] = target_crs.to_wkt()

    # ------
    # arrays for storage of dimension information

    # band dimension
    band = zarr.create(
        shape=(len(total_bands_to_save)),
        dtype="string",
        store=store,
        path="/band",
        overwrite=overwrite,
        dimension_names=["band"],
    )
    band[:] = total_bands_to_save

    # x dimension
    x = zarr.create(
        shape=(width),
        dtype="float32",
        store=store,
        path="/x",
        overwrite=overwrite,
        dimension_names=["x"],
    )
    # build coordinates from integer pixel indices (bound + i * resolution)
    # rather than np.arange over the bounds, so a fractional resolution can
    # never produce an off-by-one number of coordinates vs. width/height.
    x_coords = bound_left + np.arange(width) * target_resolution
    if coord_save_mode == "center":
        x_coords = x_coords + target_resolution / 2
    x[:] = x_coords.astype(np.float32)
    x.attrs["coord_save_mode"] = coord_save_mode

    # y dimension
    y = zarr.create(
        shape=(height),
        dtype="float32",
        store=store,
        path="/y",
        overwrite=overwrite,
        dimension_names=["y"],
    )
    y_coords = bound_top - np.arange(height) * target_resolution
    if coord_save_mode == "center":
        y_coords = y_coords - target_resolution / 2
    y[:] = y_coords.astype(np.float32)
    y.attrs["coord_save_mode"] = coord_save_mode

    # time dimension
    time = zarr.create(
        shape=(len(unique_timestamps)),
        dtype="int64",
        store=store,
        path="/time",
        fill_value=None,
        overwrite=overwrite,
        dimension_names=["time"],
    )

    for i, ts in enumerate(unique_timestamps):
        time[i] = (ts.tz_localize(tz=None) -
                   pd.Timestamp(0, tz=None)).total_seconds()
    time.attrs.update(ZARR_TIME_ATTRS)

    # see https://zarr.readthedocs.io/en/main/user-guide/consolidated_metadata/
    if consolidate_metadata:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zarr.consolidate_metadata(store)

    # close up store as we are done with init
    store.close()

    return sync_file_path


def retrieve_timestamps(
    time_composite_freq: str,
    datetime: DatetimeLike,
    bound_left: float,
    bound_bottom: float,
    bound_right: float,
    bound_top: float,
    target_crs: CRS,
    collections: list[str],
    provider=None,
) -> list[dict]:
    if time_composite_freq is None:
        # get all items within date range and area

        # open the provider's stac catalog
        if provider is None:
            provider = get_provider("planetary_computer")
        catalog = provider.open_catalog()

        search = catalog.search(
            collections=collections,
            datetime=datetime,
            bbox=warp.transform_bounds(
                src_crs=target_crs,
                dst_crs="EPSG:4326",
                left=bound_left,
                bottom=bound_bottom,
                right=bound_right,
                top=bound_top,
            ),
        )

        # sort timesteps and filter duplicates -> multiple items can have the
        # exact same timestamp
        df = pd.DataFrame()
        items = list(search.item_collection())
        if len(items) == 0:
            print("No items found for specified time range and area.")
            exit()

        df["ts_raw"] = [i.datetime for i in items]
        df["collection"] = [i.collection_id for i in items]

        # remove duplicates for timeaxis
        df = df.drop_duplicates(["ts_raw", "collection"])

        # make sure ts is sorted in the correct order
        df = df.sort_values("ts_raw", ascending=False)

        return [
            dict(collection=row["collection"], ts=row["ts_raw"])
            for _, row in df.iterrows()
        ]

    else:
        # Use pystac_client.ItemSearch to parse/expand the datetime
        from pystac_client import ItemSearch

        # Create a minimal ItemSearch instance to use its datetime parsing logic
        bs = ItemSearch(
            url="http://dummy",
            datetime=datetime,
        )
        parsed_datetime = bs.get_parameters()["datetime"]

        # Now parsed_datetime is a string like "YYYY-MM-DDTHH:MM:SSZ/YYYY-MM-DDTHH:MM:SSZ"
        # Parse the start and end from this string
        if "/" in parsed_datetime:
            start_str, end_str = parsed_datetime.split("/")
        else:
            start_str = end_str = parsed_datetime

        # rounding start for backwards compatibility and so that every
        # timeseries is aligned regardless of the requested start/end
        start = pd.to_datetime(start_str).round(time_composite_freq)
        end = max(pd.to_datetime(end_str), start)

        # get all timestamps with freq
        timestamps = pd.date_range(start, end, freq=time_composite_freq)[::-1]

        return [{
            "collection": collection,
            "ts": ts
        } for ts in timestamps for collection in collections]


def process(
    target_crs: CRS | str,
    target_resolution: float,
    bound_left: float,
    bound_bottom: float,
    bound_right: float,
    bound_top: float,
    datetime: DatetimeLike,
    zarr_store: str | zarr.storage.StoreLike,
    provider: str = "planetary_computer",
    reuse_open_datasets: bool = True,
    processing_spatial_chunk_size: int = 4000,
    S1_assets: list[str] = S1_ASSETS,
    S2_bands: list[str] = S2_RAW_BANDS,
    S2_mask_snow: bool = False,
    S2_cloud_classification: bool = False,
    S2_cloud_classification_device="cpu",
    S2_return_cloud_probabilities: bool = False,
    num_workers: int = 1,
    time_composite_freq: str = None,
    time_composite_method: str = "mean",
    S2_apply_snow_mask: bool = False,
    S2_apply_cloud_mask: bool = False,
    S2_nbar: bool = False,
    overwrite: bool = False,
    zarr_store_chunk_size: dict = {
        "time": 10,
        "x": 250,
        "y": 250,
    },
    coord_save_mode: str = "top-left",
    resampling_method: Resampling = Resampling.nearest,
    consolidate_metadata: bool = True,
    save_as_uint16: bool = False,
):
    """
    Parameters
    ----------

    target_crs : CRS or str
        Specifies the target CRS that all data will be reprojected to. If a string is provided, it will be interpreted by rasterio.crs.CRS.from_user_input().
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
        Rounding interval across which data is aggregated.
    time_composite_method: str, default="mean"
        How to aggregate the acquisitions within each ``time_composite_freq``
        window. One of ``"mean"`` (default), ``"median"``, ``"min"`` or
        ``"max"``. NoData/masked pixels are ignored per band and pixel (a
        pixel that is NoData in every acquisition of the window stays NoData).
        Only used when ``time_composite_freq`` is set.
    S2_apply_snow_mask: bool, default=False
        Whether to replace snow with NaN.
    S2_apply_cloud_mask: bool, default=False
        Whether to replace anything that is not clear sky with NaN.
    S2_nbar: bool, default=False
        Whether to apply Nadir BRFD correction with sen2nbar package.
    zarr_store: str | zarr.storage.StoreLike
       Path of where to create the zarr storage.
    provider: str, default="planetary_computer"
       Which data catalog to download from: ``"planetary_computer"`` (default)
       or ``"cdse"`` (Copernicus Data Space Ecosystem). CDSE is Sentinel-2 only
       (no Sentinel-1 RTC and no NBAR yet) and reads JP2s from CDSE S3, so it
       needs CDSE S3 credentials via the standard AWS chain (e.g.
       ``AWS_PROFILE`` or ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``). Both
       providers serve the same ESA L2A product, so the reflectances are
       identical; CDSE is currently slower per subtile (JP2-over-S3 access).
    reuse_open_datasets: bool, default=True
       Keep each Sentinel-2 band raster open and reuse it across all subtiles
       of the same tile within a spatial chunk, instead of re-opening it per
       subtile (this also keeps GDAL's decoded-tile block cache warm). This
       mostly matters for CDSE on the older archive (processing baseline
       < 05.12): those JP2s carry no TLM markers, so the first windowed read
       pays a one-time tile-structure discovery (~7 s cold), and reusing the
       open dataset amortizes it across subtiles. Newer CDSE products
       (baseline >= 05.12, from 2026) carry native TLM markers that GDAL uses
       automatically, so they are already fast per crop without this. Set to
       ``False`` to open/close per subtile (lower memory).
    processing_spatial_chunk_size: int, default=4000
       Size of spatial chunks across which we perform parallization.
    S1_assets: list[str], default=["vh_asc", "vh_desc", "vv_asc", "vv_desc"]
       Specify which bands to download for Sentinel-1. Only "vh" and "vv" are supported.
    S2_bands: list[str], default=all 12 raw bands
       Controls the Sentinel-2 band selection. Behaves like ``S1_assets``:
         - the full band list (default): download all 12 raw reflectance
           bands (``sentle.const.S2_RAW_BANDS``).
         - a subset list, e.g. ``["B04", "B03", "B02"]``: download only those
           bands, e.g. for an RGB cube. A subset is not allowed together with
           ``S2_cloud_classification`` (the cloud model needs all bands); when
           ``S2_mask_snow`` or ``S2_nbar`` are enabled, the bands they depend
           on must be included.
         - ``None`` or ``[]``: disable Sentinel-2 entirely for a
           Sentinel-1-only cube (the two are equivalent, mirroring how
           ``S1_assets=None``/``[]`` disables Sentinel-1). Requires
           ``S1_assets`` to be set, and all Sentinel-2-only options (masks,
           NBAR, cloud probabilities, ``save_as_uint16``) must be disabled.
       Output band order always follows ``S2_RAW_BANDS`` regardless of the
       order given.
    overwrite: bool, default=False
       Whether to overwrite existing zarr storage.
    coord_save_mode: str, default="top-left"
       Specifies how coordinates are saved in zarr. Options:
         - "top-left": (default) coordinates represent the top-left corner of each pixel (current behavior)
         - "center": coordinates represent the center of each pixel (shifted by half a pixel)
       Note: This does not affect the processing scheme, only how coordinates are saved in the zarr store.
    resampling_method: rasterio.enums.Resampling, default=rasterio.enums.Resampling.nearest
        Specifies the resampling method that is used to reproject the raw data
        into the target CRS. It is recommended to use nearest neighbor to
        prevent potential issues near cloud edges and dynamic range changes.
    save_as_uint16 : bool, default=False
        When True and Sentinel-1 is disabled, persist Sentinel-2 data using unsigned 16-bit integers with zero fill.
    """

    # Accept either a CRS or a string for target_crs
    if isinstance(target_crs, str):
        target_crs = CRS.from_user_input(target_crs)

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
        time_composite_method=time_composite_method,
        S2_apply_snow_mask=S2_apply_snow_mask,
        S2_apply_cloud_mask=S2_apply_cloud_mask,
        zarr_store_chunk_size=zarr_store_chunk_size,
        S2_nbar=S2_nbar,
        resampling_method=resampling_method,
        save_as_uint16=save_as_uint16,
        S2_bands=S2_bands,
        provider=provider,
    )

    # instantiate the data provider (Planetary Computer or CDSE)
    data_provider = get_provider(provider)

    # resolve which raw Sentinel-2 reflectance bands to download/save. This
    # mirrors how S1_assets works:
    #   S2_bands=<all bands>  -> all bands (the default)
    #   S2_bands=[...]        -> just those bands (a subset)
    #   S2_bands=None or []   -> Sentinel-2 disabled entirely (Sentinel-1 only)
    # Order follows S2_RAW_BANDS regardless of the order the user passes.
    s2_enabled = S2_bands is not None and len(S2_bands) > 0
    if s2_enabled:
        S2_bands = [b for b in S2_RAW_BANDS if b in S2_bands]

        # derive bands to save from arguments
        S2_bands_to_save = S2_bands.copy()
        if S2_mask_snow and time_composite_freq is None:
            S2_bands_to_save.append(S2_snow_mask_band)
        if S2_cloud_classification and time_composite_freq is None:
            S2_bands_to_save.append(S2_cloud_mask_band)
        if S2_return_cloud_probabilities:
            S2_bands_to_save += S2_cloud_prob_bands
    else:
        # Sentinel-2 disabled -> no S2 bands at all
        S2_bands = []
        S2_bands_to_save = []
    total_bands_to_save = S2_bands_to_save.copy()

    # if S1_assets are supplied as empty list, convert to None
    if isinstance(S1_assets, list) and len(S1_assets) == 0:
        S1_assets = None

    if save_as_uint16 and S1_assets is not None:
        raise ValueError(
            "save_as_uint16 requires Sentinel-1 assets to be disabled.")

    if not s2_enabled and S1_assets is None:
        raise ValueError(
            "Nothing to download: Sentinel-2 is disabled (S2_bands=None or "
            "[]) and Sentinel-1 is disabled (S1_assets is None/empty).")

    if S1_assets is not None:
        total_bands_to_save += S1_assets

    # which collections to query -- either or both satellites (collection ids
    # come from the provider)
    collections = []
    if s2_enabled:
        collections.append(data_provider.s2_collection)
    if S1_assets is not None:
        collections.append(data_provider.s1_collection)

    timestamp_list = retrieve_timestamps(
        time_composite_freq=time_composite_freq,
        bound_left=bound_left,
        bound_bottom=bound_bottom,
        bound_right=bound_right,
        bound_top=bound_top,
        target_crs=target_crs,
        datetime=datetime,
        collections=collections,
        provider=data_provider,
    )

    # compute bounds, with and height  for the entire dataset
    bound_left, bound_bottom, bound_right, bound_top = check_and_round_bounds(
        bound_left, bound_bottom, bound_right, bound_top, target_resolution)

    height, width = height_width_from_bounds_res(bound_left, bound_bottom,
                                                 bound_right, bound_top,
                                                 target_resolution)

    # setup zarr storage
    sync_file_path = setup_zarr_storage(
        zarr_store=zarr_store,
        timestamp_list=timestamp_list,
        height=height,
        width=width,
        bound_left=bound_left,
        bound_bottom=bound_bottom,
        bound_right=bound_right,
        bound_top=bound_top,
        target_resolution=target_resolution,
        processing_spatial_chunk_size=processing_spatial_chunk_size,
        zarr_store_chunk_size=zarr_store_chunk_size,
        S2_bands_to_save=S2_bands_to_save,
        total_bands_to_save=total_bands_to_save,
        target_crs=target_crs,
        coord_save_mode=coord_save_mode,
        consolidate_metadata=consolidate_metadata,
        save_as_uint16=save_as_uint16,
    )

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
        "time_composite_method": time_composite_method,
        "S2_apply_snow_mask": S2_apply_snow_mask,
        "S2_apply_cloud_mask": S2_apply_cloud_mask,
        "S2_cloud_classification": S2_cloud_classification,
        "S2_mask_snow": S2_mask_snow,
        "S2_return_cloud_probabilities": S2_return_cloud_probabilities,
        "zarr_store": zarr_store,
        "S2_bands_to_save": S2_bands_to_save,
        "S2_bands": S2_bands,
        "S1_assets": S1_assets,
        "cloud_request_queue": cloud_request_queue,
        "sync_file_path": sync_file_path,
        "S2_nbar": S2_nbar,
        "resampling_method": resampling_method,
        "save_as_uint16": save_as_uint16,
        "provider": data_provider,
        "reuse_open_datasets": reuse_open_datasets,
    }

    s2grid = gpd.read_file(
        pkg_resources.resource_filename(
            __name__, "data/sentinel2_grid_stripped_with_epsg.gpkg"))

    def job_generator():
        global GLOBAL_QUEUE_MANAGER
        global GLOBAL_QUEUES
        job_id = 0

        # the S2 subtiles depend only on the spatial chunk, not on the
        # timestamp, so cache them per spatial chunk to avoid recomputing the
        # (relatively expensive) geometry intersection for every timestamp
        subtile_cache = {}

        # iterate spatial chunks in integer *pixel* space (see
        # ``spatial_chunk_grid``); the CRS bounds are derived from the pixel
        # offsets so fractional resolutions / geographic CRSs work.
        for (xi, yi, x_off, x_end, y_off, y_end,
             (x_min, y_min, x_max, y_max)) in spatial_chunk_grid(
                 bound_left, bound_top, width, height, target_resolution,
                 processing_spatial_chunk_size):
            last_ts = timestamp_list[0]["ts"]
            ts_save_index = 0
            for item in timestamp_list:
                ret_config = dict(config)
                ret_config["bound_left"] = x_min
                ret_config["bound_bottom"] = y_min
                ret_config["bound_right"] = x_max
                ret_config["bound_top"] = y_max
                ret_config["ts"] = item["ts"]
                ret_config["collection"] = item["collection"]

                if item["ts"] != last_ts:
                    last_ts = item["ts"]
                    ts_save_index += 1

                ret_config["zarr_save_slice"] = dict(
                    x=slice(x_off, x_end),
                    y=slice(y_off, y_end),
                    band=slice(0, len(S2_bands_to_save))
                    if item["collection"] == "sentinel-2-l2a" else slice(
                        len(S2_bands_to_save), len(total_bands_to_save)),
                    time=ts_save_index,
                )
                if item["collection"] == "sentinel-2-l2a":
                    if (xi, yi) not in subtile_cache:
                        subtile_cache[(xi, yi)] = obtain_subtiles(
                            target_crs=target_crs,
                            left=ret_config["bound_left"],
                            bottom=ret_config["bound_bottom"],
                            right=ret_config["bound_right"],
                            top=ret_config["bound_top"],
                            s2grid=s2grid,
                        )
                    ret_config["S2_subtiles"] = subtile_cache[(xi, yi)]
                else:
                    ret_config["S2_subtiles"] = None
                if (S2_cloud_classification
                        and item["collection"] == "sentinel-2-l2a"):
                    GLOBAL_QUEUES[job_id] = GLOBAL_QUEUE_MANAGER.Queue(
                        maxsize=1)
                    ret_config["cloud_response_queue"] = GLOBAL_QUEUES[job_id]
                    ret_config["job_id"] = job_id
                    job_id += 1
                else:
                    ret_config["cloud_response_queue"] = None
                    ret_config["job_id"] = None
                ret_config["save_as_uint16"] = save_as_uint16
                yield ret_config

    with tqdm_joblib(
            tqdm(
                desc="processing",
                unit="ptiles",
                dynamic_ncols=True,
                total=len(timestamp_list),
            )) as progress_bar:
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
