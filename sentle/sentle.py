import atenea
from typing import Optional, Union
from utils import paral
from pystac_client.item_search import DatetimeLike


class SentinelSubtile:

    def __init__(self, sentinel_tile_string: str, left: float, bottom: float,
                 right: float, top: float):
        self.sentinel_tile_string = sentinel_tile_string
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top

    def __str__(self):
        return str((sentinel_tile_string, self.left, self.bottom, self.right,
                    self.top))


def obtain_subtiles(target_crs: str, left: float, bottom: float, right: float,
                    top: float, subtile_width: int) -> list[SentinelSubtile]:
    """Retrieves the sentinel subtiles that intersect the with the specified
    bounds. The bounds are interpreted based on the given target_crs.
    """
    # 0 check if supplied sub_tile_width makes sense

    # 1 load sentinel grid

    # 2 convert box to sentinel grid crs

    # 3 extract overlapping sentinel tiles

    # 4 for each sentinel tile determin sub-sentinel tiles
    pass


def process_subtile(subtile: SentinelSubtile, datetime: DatetimeLike,
                    zarrpath: str, atenea_args: dict):
    # 1 download subtile with specified date range

    # 2 push that tile through atenea

    # 3 reproject to target_crs

    # 4 remove data that is outside the bounds of the specified bounds

    # 5 save that tile to a specified zarr

    pass


def process(target_crs: str,
            bound_left: float,
            bound_bottom: float,
            bound_right: float,
            bound_top: float,
            datetime: DatetimeLike,
            subtile_width: int,
            num_cores: int,
            nbar: bool = True,
            reduce_time: bool = True,
            mask_clouds: bool = True,
            cloud_mask_model: str = "rembrandt",
            return_cloud_classification_layer: bool = False,
            return_cloud_probabilities: bool = False,
            apply_cloud_mask: bool = False,
            drop_cloudy: bool = False,
            clear_sky_threshold: Union[int, float] = 0.5,
            mask_snow: bool = True,
            quiet: bool = False) -> xr.DataArray:

    # TODO think about what to do with clouds and if to optionally filter them like atenea
    # --> mask_Clouds, drop_cloudy, clear_sky_threshold, mask_snow)

    # 1 obtain sub-sentinel tiles based on supplied bounds and CRS
    subtiles = obtain_subtiles(target_crs,
                               bound_left,
                               bound_bottom,
                               bound_right,
                               bound_top,
                               subtile_width=subtile_width)

    # 2 in parallel for each sub-sentinel tile
    # 2a download each sub-sentinel tile from planetary computer
    # 2b run each sub-sentinel tile through atenea
    atenea_args = dict(
        nbar=nbar,
        reduce_time=reduce_time,
        mask_clouds=mask_clouds,
        cloud_mask_model=cloud_mask_model,
        return_cloud_probabilities=return_cloud_probabilities,
        return_cloud_classification_layer=return_cloud_classification_layer)
    ns = len(subtiles)
    paral(process_subtile,
          [subtiles, [datetime] * ns, [zarrpath] * ns, [atenea_args] * ns],
          num_cores=num_cores,
          progress_bar=not quiet)

    # 3 merge sub-sentinel tiles into one big sparse xarray and return
    # possible ISSUE: what happens when not everything fits into memory --> we
    # may want to have each subtile already being written to the harddrive once
    # it's done --> is that supported with ZARR?
