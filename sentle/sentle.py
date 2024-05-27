import itertools
import warnings

import dask.array
import geopandas as gpd
import numpy as np
import pandas as pd
import pkg_resources
import planetary_computer
import pystac_client
import rasterio
import scipy.ndimage as sc
import xarray as xr
import zarr
from affine import Affine
from dask.distributed import Client, LocalCluster, Variable
from numcodecs import Blosc
from pystac_client.item_search import DatetimeLike
from pystac_client.stac_api_io import StacApiIO
from rasterio import transform, warp, windows
from rasterio.crs import CRS
from rasterio.enums import Resampling
from shapely.geometry import Polygon, box
from urllib3 import Retry

from .cloud_mask import compute_cloud_mask, load_cloudsen_model, S2_cloud_prob_bands
from .snow_mask import compute_potential_snow_layer
from .utils import bounds_from_transform_height_width_res, transform_height_width_from_bounds_res
from .const import S2_RAW_BANDS, S2_RAW_BAND_RESOLUTION


class Sentle():

    def __init__(self,
                 num_workers: int = 1,
                 threads_per_worker: int = 1,
                 memory_limit_per_worker: str = "6GB",
                 dashboard_address: str = "127.0.0.1:9988"):

        if threads_per_worker > 1:
            warnings.warn(
                "More then one thread per worker may overflow memory. Not tested yet"
            )

        # setup local processor
        cluster = LocalCluster(dashboard_address=dashboard_address,
                               n_workers=num_workers,
                               threads_per_worker=threads_per_worker,
                               memory_limit=memory_limit_per_worker)
        client = Client(cluster)
        print("Dask client dashboard link:", client.dashboard_link)

        # load Sentinel 2 grid
        Variable("s2gridfile").set(
            gpd.read_file(
                pkg_resources.resource_filename(
                    __name__, "data/sentinel2_grid_stripped_with_epsg.gpkg")))

        # define main daskarray
        self.da = None

    @staticmethod
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
        assert lheight == gheight
        assert all(x % 1 == 0 for x in
                   [grow, gcol, gwidth, gheight, lrow, lcol, lwidth, lheight])

        return windows.Window(
            row_off=grow, col_off=gcol, height=gheight,
            width=gwidth).round_offsets().round_lengths(), windows.Window(
                row_off=lrow, col_off=lcol, height=lheight,
                width=lwidth).round_offsets().round_lengths()

    @staticmethod
    def obtain_subtiles(target_crs: CRS, left: float, bottom: float,
                        right: float, top: float, S2_subtile_size: int):
        """Retrieves the sentinel subtiles that intersect the with the specified
        bounds. The bounds are interpreted based on the given target_crs.
        """

        # TODO make it possible to not only use naive bounds but also MultiPolygons

        # check if supplied sub_tile_width makes sense
        assert (S2_subtile_size >= 16) and (
            S2_subtile_size
            <= 10980), "S2_subtile_size needs to within 16 and 10980"
        assert (10980 % S2_subtile_size
                ) == 0, "S2_subtile_size needs to be a divisor of 10980"

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
            windows.Window(col_off, row_off, S2_subtile_size, S2_subtile_size)
            for col_off, row_off in itertools.product(
                np.arange(0, 10980, S2_subtile_size),
                np.arange(0, 10980, S2_subtile_size))
        ]

        # reproject s2 footprint to local utm footprint
        s2grid["s2_footprint_utm"] = s2grid[["geometry", "crs"]].apply(
            lambda ser: Polygon(*warp.transform_geom(src_crs=s2grid.crs,
                                                     dst_crs=ser["crs"],
                                                     geom=ser["geometry"].
                                                     geoms[0])["coordinates"]),
            axis=1)

        # obtain transform of each sentinel 2 tile in local utm crs
        s2grid["tile_transform"] = s2grid["s2_footprint_utm"].apply(
            lambda x: transform.from_bounds(
                *x.bounds, width=10980, height=10980))

        # convert read window to polygon in S2 local CRS, then transform to
        # s2grid.crs and check overlap with transformed bounds
        s2grid["intersecting_windows"] = s2grid[[
            "tile_transform", "crs"
        ]].apply(lambda ser: [
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

    @staticmethod
    def get_stac_api_io():
        retry = Retry(total=5,
                      backoff_factor=1,
                      status_forcelist=[502, 503, 504],
                      allowed_methods=None)
        return StacApiIO(max_retries=retry)

    @staticmethod
    def calculate_aligned_transform(src_crs, dst_crs, height, width, left,
                                    bottom, right, top, tres):
        tf, repr_width, repr_height = warp.calculate_default_transform(
            src_crs=src_crs,
            dst_crs=dst_crs,
            width=width,
            height=height,
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            resolution=tres)

        tf = Affine(
            tf.a,
            tf.b,
            tf.c - (tf.c % tres),
            tf.d,
            tf.e,
            # + target_resolution because upper left corner
            tf.f - (tf.f % tres) + tres,
            tf.g,
            tf.h,
            tf.i,
        )

        # include one more pixel because rounding down
        repr_width += 1
        repr_height += 1

        return tf, repr_height, repr_width

    def process_S2_subtile(
            self, intersecting_windows, stac_item, timestamp,
            S2_subtile_size: int, target_crs: CRS, target_resolution: float,
            ptile_transform, ptile_width: int, ptile_height: int,
            S2_mask_snow: bool, S2_cloud_classification: bool,
            S2_return_cloud_probabilities: bool, S2_compute_nbar: bool,
            S2_cloud_classification_device: str, cloud_mask_model):

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

        # determine bounds based on subtile window and tile transform
        subtile_bounds_utm = windows.bounds(intersecting_windows,
                                            s2_tile_transform)
        assert (
            subtile_bounds_utm[2] - subtile_bounds_utm[0]
        ) // 10 == S2_subtile_size, "mismatch between subtile size and bounds on x-axis"
        assert (
            subtile_bounds_utm[3] - subtile_bounds_utm[1]
        ) // 10 == S2_subtile_size, "mismatch between subtile size and bounds on y-axis"

        if S2_cloud_classification or S2_return_cloud_probabilities:
            result_probs = compute_cloud_mask(
                subtile_array,
                cloud_mask_model,
                S2_cloud_classification_device=S2_cloud_classification_device)
            band_names += S2_cloud_prob_bands
            subtile_array = np.concatenate([subtile_array, result_probs])

        # 3 reproject to target_crs for each band
        # determine transform --> round to target resolution so that reprojected
        # subtiles align across subtiles
        subtile_repr_transform, subtile_repr_height, subtile_repr_width = self.calculate_aligned_transform(
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
                       src_transform=transform.from_bounds(
                           *subtile_bounds_utm,
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

        write_win, local_win = self.recrop_write_window(
            write_win, ptile_height, ptile_width)

        # crop subtile_array based on computed local win because it could overlap
        # with the overall bounds
        subtile_array_repr = subtile_array_repr[:, local_win.
                                                row_off:local_win.height +
                                                local_win.row_off, local_win.
                                                col_off:local_win.col_off +
                                                local_win.width]

        return subtile_array_repr, write_win, band_names

    @staticmethod
    def height_width_from_bounds_res(left, bottom, right, top, res):
        # determine width and height based on bounds and resolution
        width, w_rem = divmod(abs(right - left), res)
        height, h_rem = divmod(abs(top - bottom), res)

        if h_rem > 0:
            warnings.warn(
                "Specified top/bottom bounds are not perfectly divisable by specified target_resolution. The resulting coverage will be slightly cropped"
            )
        if w_rem > 0:
            warnings.warn(
                "Specified left/right bounds are not perfectly divisable by specified target_resolution. The resulting coverage will be slightly cropped"
            )

        return height, width

    def open_catalog(self):
        return pystac_client.Client.open(
            Variable("stac_endpoint").get(),
            modifier=planetary_computer.sign_inplace,
            stac_io=self.get_stac_api_io())

    def process_ptile(
        self,
        da: xr.DataArray,
        target_crs: CRS,
        target_resolution: float,
        S2_cloud_classification_device: str,
        S2_subtile_size: int = 732,
        S2_mask_snow: bool = False,
        S2_cloud_classification: bool = False,
        S2_return_cloud_probabilities: bool = False,
        S2_compute_nbar: bool = False,
    ):
        """Passing chunk to either sentinel-1 or sentinel-2 processor"""

        # check collection matches bands otherwise return
        if da.collection.data[0] == "sentinel-1-rtc":
            if ("vv" not in da.band.data or "vh" not in da.band.data):
                return da
            else:
                return self.process_ptile_S1(da, target_crs, target_resolution)

        if da.collection.data[0] == "sentinel-2-l2a":
            if ("vv" in da.band.data and "vh" in da.band.data):
                return da
            else:
                return self.process_ptile_S2(
                    da=da,
                    target_crs=target_crs,
                    target_resolution=target_resolution,
                    S2_cloud_classification=S2_cloud_classification,
                    S2_cloud_classification_device=
                    S2_cloud_classification_device,
                    S2_subtile_size=S2_subtile_size,
                    S2_mask_snow=S2_mask_snow,
                    S2_return_cloud_probabilities=S2_return_cloud_probabilities,
                    S2_compute_nbar=S2_compute_nbar)

    def process_ptile_S1(self, da: xr.DataArray, target_crs: CRS,
                         target_resolution: float):

        # extract the timestamp we are processing. there should only be one
        timestamp = da.time.data
        assert timestamp.shape == (1, )
        timestamp = timestamp[0]

        # compute bounds of ptile
        ptile_bounds = self.bounds_from_dataarray(da, target_resolution)

        # open stac catalog
        catalog = self.open_catalog()

        # retrieve items (possible across multiple sentinel tile) for specified
        # timestamp
        item_list = list(
            catalog.search(collections=["sentinel-1-rtc"],
                           datetime=timestamp,
                           bbox=warp.transform_bounds(
                               src_crs=target_crs,
                               dst_crs="EPSG:4326",
                               left=ptile_bounds[0],
                               bottom=ptile_bounds[1],
                               right=ptile_bounds[2],
                               top=ptile_bounds[3])).item_collection())

        if len(item_list) == 0:
            # if there is nothing within the bounds and for that timestamp return.
            # possible and normal
            return da

        # intiate one array representing the entire subtile for that timestamp
        tile_array = np.full(shape=(da.shape[1], da.shape[2], da.shape[3]),
                             fill_value=0,
                             dtype=np.float32)

        # count how many values we add per pixel to compute mean later
        tile_array_count = np.full(shape=(da.shape[1], da.shape[2],
                                          da.shape[3]),
                                   fill_value=0,
                                   dtype=np.uint8)

        # determine ptile dimensions and transform from bounds
        ptile_transform, ptile_height, ptile_width = transform_height_width_from_bounds_res(
            *ptile_bounds, target_resolution)

        for item in item_list:
            # iterate through S1 assets
            for i, s1_asset in enumerate(da.band.data):
                with rasterio.open(item.assets[s1_asset].href) as dr:
                    # reproject ptile bounds to S1 tile CRS
                    ptile_bounds_local_crs = warp.transform_bounds(
                        target_crs, dr.crs, *ptile_bounds)
                    # figure out which area of the image is interesting for us
                    read_win = dr.window(*ptile_bounds_local_crs)
                    # read windowed
                    data = dr.read(indexes=1,
                                   window=read_win,
                                   out_dtype=np.float32)

                    # compute aligned reprojection
                    tile_repr_transform, tile_repr_height, tile_repr_width = self.calculate_aligned_transform(
                        dr.crs, target_crs, data.shape[0], data.shape[1],
                        *ptile_bounds_local_crs, target_resolution)

                    data_repr = np.empty((tile_repr_height, tile_repr_width),
                                         dtype=np.float32)

                    # billinear reprojection for everything
                    warp.reproject(source=data,
                                   destination=data_repr,
                                   src_transform=transform.from_bounds(
                                       *ptile_bounds_local_crs,
                                       height=read_win.height,
                                       width=read_win.width),
                                   src_crs=dr.crs,
                                   dst_crs=target_crs,
                                   src_nodata=dr.nodata,
                                   dst_nodata=0,
                                   dst_transform=tile_repr_transform,
                                   resampling=Resampling.bilinear)

                    # explicit clear
                    del data

                    # compute bounds of reprojected tile in target crs
                    # this will have nans and so on
                    tile_bounds_trcs = bounds_from_transform_height_width_res(
                        tf=tile_repr_transform,
                        height=tile_repr_height,
                        width=tile_repr_width,
                        resolution=target_resolution)

                    # figure out where to write the subtile within the overall bounds
                    write_win = windows.from_bounds(
                        *tile_bounds_trcs, transform=ptile_transform
                    ).round_offsets().round_lengths()

                    # determine crop to avoid out of bounds
                    write_win, local_win = self.recrop_write_window(
                        write_win, ptile_height, ptile_width)

                    # crop reprojected downlaoded data
                    data_repr = data_repr[local_win.row_off:local_win.height +
                                          local_win.row_off,
                                          local_win.col_off:local_win.col_off +
                                          local_win.width]

                    # save it
                    tile_array[i, write_win.row_off:write_win.row_off +
                               write_win.height,
                               write_win.col_off:write_win.col_off +
                               write_win.width] += data_repr

                    # save where we have NaNs
                    tile_array_count[i, write_win.row_off:write_win.row_off +
                                     write_win.height,
                                     write_win.col_off:write_win.col_off +
                                     write_win.width] += ~(data_repr == 0)

        with warnings.catch_warnings():
            # filter out divide by zero warning, this is expected here
            warnings.simplefilter("ignore")
            tile_array /= tile_array_count

        # replace zeros with nans
        tile_array[tile_array == 0] = np.nan

        return xr.DataArray(data=np.expand_dims(tile_array, axis=0),
                            dims=["time", "band", "y", "x"],
                            coords=dict(time=[timestamp],
                                        band=da.band,
                                        x=da.x,
                                        y=da.y,
                                        collection=("time",
                                                    da.collection.data)))

    @staticmethod
    def bounds_from_dataarray(da, target_resolution):
        # (add target resolution to miny and maxx because we are using top-left
        # coordinates)
        bound_left = da.x.min().item()
        bound_bottom = da.y.min().item() - target_resolution
        bound_right = da.x.max().item() + target_resolution
        bound_top = da.y.max().item()

        return (bound_left, bound_bottom, bound_right, bound_top)

    def process_ptile_S2(
        self,
        da: xr.DataArray,
        target_crs: CRS,
        target_resolution: float,
        S2_cloud_classification_device: str,
        S2_subtile_size: int = 732,
        S2_mask_snow: bool = False,
        S2_cloud_classification: bool = False,
        S2_return_cloud_probabilities: bool = False,
        S2_compute_nbar: bool = False,
    ):

        # compute bounds of ptile
        bound_left, bound_bottom, bound_right, bound_top = self.bounds_from_dataarray(
            da, target_resolution)

        # extract the timestamp we are processing. there should only be one
        timestamp = da.time.data
        assert timestamp.shape == (1, )
        timestamp = timestamp[0]

        # open stac catalog
        catalog = self.open_catalog()

        # determine ptile dimensions and transform from bounds
        ptile_transform, ptile_height, ptile_width = transform_height_width_from_bounds_res(
            bound_left, bound_bottom, bound_right, bound_top,
            target_resolution)

        # retrieve items (possible across multiple sentinel tile) for specified
        # timestamp
        item_list = list(
            catalog.search(collections=["sentinel-2-l2a"],
                           datetime=timestamp,
                           bbox=warp.transform_bounds(
                               src_crs=target_crs,
                               dst_crs="EPSG:4326",
                               left=bound_left,
                               bottom=bound_bottom,
                               right=bound_right,
                               top=bound_top)).item_collection())

        if len(item_list) == 0:
            # if there is nothing within the bounds and for that timestamp return.
            # possible and normal
            return da

        items = pd.DataFrame()
        items["item"] = item_list
        items["tile"] = items["item"].apply(
            lambda x: x.properties["s2:mgrs_tile"])

        # cloud classification layer is added later
        num_bands = da.shape[1]
        if S2_cloud_classification:
            num_bands -= 1
        if S2_mask_snow:
            num_bands -= 1
        if not S2_return_cloud_probabilities and S2_cloud_classification:
            # we need the probs here, will remove when returning
            num_bands += 4

        # intiate one array representing the entire subtile for that timestamp
        subtile_array = np.full(shape=(num_bands, da.shape[2], da.shape[3]),
                                fill_value=0,
                                dtype=np.float32)

        # count how many values we add per pixel to compute mean later
        subtile_array_count = np.full(shape=(num_bands, da.shape[2],
                                             da.shape[3]),
                                      fill_value=0,
                                      dtype=np.uint8)

        # load cloudsen model
        cloudsen_model = load_cloudsen_model(
            S2_cloud_classification_device
        ) if S2_cloud_classification else None

        # obtain sub-sentinel2 tiles based on supplied bounds and CRS
        subtiles = self.obtain_subtiles(target_crs,
                                        bound_left,
                                        bound_bottom,
                                        bound_right,
                                        bound_top,
                                        S2_subtile_size=S2_subtile_size)

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

            subtile_array_ret, write_win, subtile_array_bands = self.process_S2_subtile(
                intersecting_windows=st.intersecting_windows,
                stac_item=stac_item,
                timestamp=timestamp,
                S2_subtile_size=S2_subtile_size,
                target_crs=target_crs,
                target_resolution=target_resolution,
                ptile_transform=ptile_transform,
                ptile_width=ptile_width,
                ptile_height=ptile_height,
                S2_mask_snow=S2_mask_snow,
                S2_cloud_classification=S2_cloud_classification,
                S2_return_cloud_probabilities=S2_return_cloud_probabilities,
                S2_compute_nbar=S2_compute_nbar,
                S2_cloud_classification_device=S2_cloud_classification_device,
                cloud_mask_model=cloudsen_model)

            # also replace nan with 0 so that the mean computation works
            # (this is reverted later)
            subtile_array[:, write_win.row_off:write_win.row_off +
                          write_win.height,
                          write_win.col_off:write_win.col_off +
                          write_win.width] += subtile_array_ret

            subtile_array_count[:, write_win.row_off:write_win.row_off +
                                write_win.height,
                                write_win.col_off:write_win.col_off +
                                write_win.width] += ~(subtile_array_ret == 0)

        with warnings.catch_warnings():
            # filter out divide by zero warning, this is expected here
            warnings.simplefilter("ignore")
            subtile_array /= subtile_array_count

        # compute cloud classification layer
        if S2_cloud_classification:

            # select cloud class based on maximum probability
            cloud_class = np.argmax(subtile_array[[
                subtile_array_bands.index(band) for band in S2_cloud_prob_bands
            ]],
                                    axis=0,
                                    keepdims=True)

            # apply max filter on cloud clases to dilate invalid pixels
            cloud_class = sc.maximum_filter(cloud_class,
                                            size=(1, 7, 7),
                                            mode="nearest").astype(np.float32)

            # save cloud classes layer
            subtile_array = np.concatenate([subtile_array, cloud_class],
                                           axis=0)
            subtile_array_bands.append("S2_cloud_classification")

        if S2_mask_snow:
            subtile_array = np.concatenate([
                subtile_array,
                np.expand_dims(compute_potential_snow_layer(
                    B03=subtile_array[subtile_array_bands.index("B03")],
                    B11=subtile_array[subtile_array_bands.index("B11")],
                    B08=subtile_array[subtile_array_bands.index("B08")]),
                               axis=0)
            ])
            subtile_array_bands.append("S2_snow_mask")

        # ... and set all such pixels to nan (of which some are already nan because
        # of divide by zero)
        # determine nodata mask based on where values are zero -> mean nodata for S2...
        # (need to do this here, because after computing mean there will be nans
        # from divide by zero)
        subtile_array[:,
                      np.any(subtile_array[[
                          subtile_array_bands.index(band)
                          for band in S2_RAW_BANDS
                      ]] == 0,
                             axis=0)] = np.nan

        # expand dimensions -> one timestep
        subtile_array = np.expand_dims(subtile_array, axis=0)

        # wrap numpy array into xarray again
        out_array = xr.DataArray(data=subtile_array,
                                 dims=["time", "band", "y", "x"],
                                 coords=dict(time=[timestamp],
                                             band=subtile_array_bands,
                                             x=da.x,
                                             y=da.y,
                                             collection=("time",
                                                         da.collection.data)))

        # only return bands that have been requested
        return out_array.sel(band=da.band)

    def process(
        self,
        target_crs: CRS,
        target_resolution: float,
        bound_left: float,
        bound_bottom: float,
        bound_right: float,
        bound_top: float,
        datetime: DatetimeLike,
        processing_spatial_chunk_size: int = 4000,
        S1_assets: list[str] = ["vh", "vv"],
        S2_subtile_size: int = 732,
        S2_mask_snow: bool = False,
        S2_cloud_classification: bool = False,
        S2_cloud_classification_device="cpu",
        S2_return_cloud_probabilities: bool = False,
        S2_compute_nbar: bool = False,
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
        S2_subtile_size : int, default=732
            Specifies the size of each subtile. The maximum is the size of a sentinel tile (10980). If cloud filtering is enabled the minimum tilesize is 256, otherwise 16. It also needs to be a divisor of 10980, so that each sentinel tile can be segmented without overlaps. At the moment this package only supports the default S2_subtile_size of 732.
        S2_mask_snow : bool, default=False
            Whether to create a snow mask. Based on https://doi.org/10.1016/j.rse.2011.10.028.
        S2_cloud_classification : bool, default=False
            Whether to create cloud classification layer, where `0=clear sky`, `2=thick cloud`, `3=thin cloud`, `4=shadow`.
        S2_cloud_classification_device : str, default="cpu"
            On which device to run cloud classification. Either `cpu` or `cuda`.
        S2_return_cloud_probabilities : bool, default=False
            Whether to return raw cloud probabilities which were used to determine the cloud classes.
        S2_compute_nbar : bool, default=False
            Whether to compute NBAR using the sen2nbar package. Coming soon.
        """

        assert S2_subtile_size == 732, "Unsupported subtile size."

        # TODO support to only download subset of bands (mutually exclusive with cloud classification and partially snow_mask) -> or no sentinel 2 at all

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
        if S2_mask_snow:
            bands_to_save.append("S2_snow_mask")
        if S2_compute_nbar:
            warnings.warn(
                "NBAR computation currently not supported. Coming Soon. Ignoring..."
            )
            S2_compute_nbar = False
            # bands_to_save += [
            #     "NBAR_B02",
            #     "NBAR_B03",
            #     "NBAR_B04",
            #     "NBAR_B05",
            #     "NBAR_B06",
            #     "NBAR_B07",
            #     "NBAR_B08",
            #     "NBAR_B11",
            #     "NBAR_B12",
            # ]
        if S2_cloud_classification:
            bands_to_save.append("S2_cloud_classification")
        if S2_return_cloud_probabilities:
            bands_to_save += S2_cloud_prob_bands
        if S1_assets is not None:
            assert len(set(S1_assets) -
                       set(["vv", "vh"])) == 0, "Unsupported S1 bands."
            bands_to_save += S1_assets

        # sign into planetary computer
        stac_endpoint = "https://planetarycomputer.microsoft.com/api/stac/v1"
        Variable("stac_endpoint").set(stac_endpoint)
        catalog = self.open_catalog()

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
        # remove timezone, otherwise crash -> zarr caveat
        # ... and use numpy.datetime64 with second precision
        df["ts"] = [i.datetime for i in items]
        df["collection"] = [i.collection_id for i in items]
        df = df.drop_duplicates("ts")
        assert df.shape[0] == df.drop_duplicates().shape[
            0], "Never expected that images of S1 are exactly at the same time, please open issue."

        height, width = self.height_width_from_bounds_res(
            bound_left, bound_bottom, bound_right, bound_top,
            target_resolution)

        # figure out band chunk shape
        if S1_assets is not None:
            band_chunks = (len(bands_to_save) - len(S1_assets), len(S1_assets))
        else:
            band_chunks = len(bands_to_save)

        # chunks with one per timestep -> many empty timesteps for specific areas,
        # because we have all the timesteps for Germany
        self.da = xr.DataArray(
            data=dask.array.full(
                shape=(df.shape[0], len(bands_to_save), height, width),
                chunks=(1, band_chunks, processing_spatial_chunk_size,
                        processing_spatial_chunk_size),
                # needs to be float in order to store NaNs
                dtype=np.float32,
                fill_value=np.nan),
            dims=["time", "band", "y", "x"],
            coords=dict(
                time=df["ts"].tolist(),
                band=bands_to_save,
                x=np.arange(bound_left, bound_right,
                            target_resolution).astype(np.float32),
                # we do y-axis in reverse: top-left coordinate
                y=np.arange(bound_top, bound_bottom,
                            -target_resolution).astype(np.float32),
                collection=("time", df["collection"].tolist())))

        self.da = self.da.map_blocks(
            self.process_ptile,
            kwargs=dict(
                target_crs=target_crs,
                target_resolution=target_resolution,
                S2_subtile_size=S2_subtile_size,
                S2_mask_snow=S2_mask_snow,
                S2_cloud_classification=S2_cloud_classification,
                S2_return_cloud_probabilities=S2_return_cloud_probabilities,
                S2_compute_nbar=S2_compute_nbar,
                S2_cloud_classification_device=S2_cloud_classification_device,
            ),
            template=self.da)

        # remove timezone, otherwise crash -> zarr caveat
        # ... and use numpy.datetime64 with second precision
        self.da = self.da.assign_coords(
            dict(time=[
                pd.Timestamp(i.replace(tzinfo=None)).to_datetime64()
                for i in self.da.time.data
            ]))

    def save_as_zarr(self, path: str):
        """
        Triggers dask compute and saves chunks whenever they have been
        processed. Empty chunks are not written. Chunks are compressed with
        lz4. 

        Parameters
        ----------
        path : str
            Specifies where save path of the zarr file.    
        """

        if self.da is None:
            print("No data proccessed, nothing to save.")
            return

        # NOTE the compression may not be optimal, need to benchmark
        store = zarr.storage.DirectoryStore(path, dimension_separator=".")
        self.da.rename("sentle").to_zarr(store=store,
                                     mode="w-",
                                     compute=True,
                                     encoding={
                                         "sentle": {
                                             "write_empty_chunks": False,
                                             "compressor": Blosc(cname="lz4"),
                                         }
                                     })

    def mask_array(self,
                   use_cloud_class_mask: bool = True,
                   use_snow_mask: bool = True,
                   cloud_mask_max_class: int = 0,
                   bands_to_mask: list[str] = [
                       'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
                       'B8A', 'B09', 'B11', 'B12'
                   ]):
        """
        Replaces pixels with clouds or snow with `np.nan`. Extends dask graph after calling `process()`.

        Parameters
        ----------
        use_cloud_class_mask : bool, default=True
            Whether to use the generated cloud mask. Requires `S2_cloud_classification=True` in `process()`.
        use_snow_mask : bool, default=True  
            Whether to use the generated snow mask. Requires `S2_mask_snow=True` in `process()`.
        cloud_mask_max_class : int, default=0
            Specifies the maximum acceptable class. `0` only uses clear sky. See notes in `process()` for other classes.
        """

        if use_snow_mask and "S2_snow_mask" not in self.da.band:
            warnings.warn(
                "use_snow_mask set to True for time composite, but no snow_mask in bands."
            )

        if use_cloud_class_mask and "S2_cloud_classification" not in self.da.band:
            warnings.warn(
                "use_cloud_class_mask set to True for time composite, but no S2_cloud_classification in bands."
            )

        def _mask_chunk(dc):
            band_intersection = set(dc.band.data).intersection(
                set(bands_to_mask))
            if len(band_intersection) == 0:
                return dc

            if use_cloud_class_mask:
                assert "S2_cloud_classification" in dc.band.data, "A chunk where cloud masking is supposed to be applied, but no cloud class layer exists. Please open GitHub issue."

                cloud_mask = (dc.sel(band="S2_cloud_classification")
                              <= cloud_mask_max_class)

            if use_snow_mask:
                assert "S2_snow_mask" in dc.band.data, "A chunk where snow masking is supposed to be applied, but no snow layer exists. Bug. Please open GitHub issue."

                snow_mask = (dc.sel(band="S2_snow_mask") == 1)

                if use_cloud_class_mask:
                    mask = snow_mask & cloud_mask
                else:
                    mask = snow_mask

            else:
                mask = cloud_mask

            return dc.where(mask, other=np.nan)

        # setting all values to nan that are snow/clouds
        self.da = self.da.map_blocks(_mask_chunk, template=self.da)

    def create_time_composite(self, freq: str = "7d", method: str = "median"):
        """
        Creates a (nan)mean across each time interval for each band.

        Parameters
        ----------
        ndays : int, default=7
            Number of days to perform mean on.
        method : str, default="median"
            Method to aggregate data across time. Can be either "mean" or "median".
        """

        # drop cloud / snow
        sub_bands = self.da.band[~(
            (self.da.band == "S2_snow_mask") |
            (self.da.band == "S2_cloud_classification"))]
        self.da = self.da.sel(band=sub_bands)

        # save chunk layout to apply it after aggregation again
        chunks_before_groupby = self.da.chunks

        # create on chunk per band
        self.da = self.da.chunk(
            (1, 1, chunks_before_groupby[2], chunks_before_groupby[3]))

        # create groupby index where we place
        seconds_in_day = 86400
        index = xr.IndexVariable(
            dims="time",
            data=np.array(
                list(
                    map(lambda x: pd.Timestamp(x).round(freq).to_datetime64(),
                        self.da.time.data))))

        # group by rounded timestamps
        self.da = self.da.groupby(index)

        # do nan mean/median for each group
        with warnings.catch_warnings():
            # this is expected, no warning needed
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

            if method == "median":
                self.da = self.da.median(dim="time", skipna=True)
            elif method == "mean":
                self.da = self.da.mean(dim="time", skipna=True)
            else:
                raise NotImplementedError(
                    "Aggregation methods other than mean or median are not implemented."
                )

        # rechunk to original format
        self.da = self.da.chunk(
            (1, sum(chunks_before_groupby[1]), chunks_before_groupby[2],
             chunks_before_groupby[3]))
