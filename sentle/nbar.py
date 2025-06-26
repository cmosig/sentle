from collections import OrderedDict

import numpy as np
import rasterio.crs
import sen2nbar.c_factor
from pystac import Item

# cheap least recently used cache
c_factor_cache = OrderedDict()


def get_c_factor_value(
    stac_item: Item,
    s2_crs: rasterio.crs.CRS,
    subtile_bounds_utm: tuple[float, float, float, float],
) -> np.ndarray:
    """
    Compute the c-factor (nadir BRDF correction factor) for a Sentinel-2 STAC item over a given subtile.

    Parameters
    ----------
    stac_item : pystac.Item
        The STAC item representing a Sentinel-2 scene. Must have an 'id' attribute and be compatible with sen2nbar.
    s2_crs : rasterio.crs.CRS
        The CRS of the Sentinel-2 data as a rasterio CRS object.
    subtile_bounds_utm : tuple[float, float, float, float]
        The bounds of the subtile in UTM coordinates (left, bottom, right, top).

    Returns
    -------
    np.ndarray
        The c-factor array interpolated to the subtile grid, to be multiplied with reflectance bands for NBAR correction.
    """

    # get item id from stac item
    item_id = stac_item.id

    # this part takes 99% of the time of this function which its why its cached
    if item_id in c_factor_cache:
        c = c_factor_cache[item_id]

        # move to end of cache so that wont be removed
        c_factor_cache.move_to_end(item_id)
    else:
        assert s2_crs is not None, "s2_crs is None"

        c = sen2nbar.c_factor.c_factor_from_item(stac_item,
                                                 f"EPSG:{s2_crs.to_epsg()}")
        c_factor_cache[item_id] = c

        # if dict size > 1000 remove oldest item
        if len(c_factor_cache) > 1000:
            c_factor_cache.popitem(last=False)

    # get top-left coordinates of subtile
    x_coords = np.arange(subtile_bounds_utm[0], subtile_bounds_utm[2], 10)
    y_coords = np.arange(subtile_bounds_utm[3], subtile_bounds_utm[1], -10)

    # get c-factor array for exact subtile bounds
    c = c.interp(
        y=y_coords,
        x=x_coords,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    # convert output to numpy array
    c = c.values

    return c
