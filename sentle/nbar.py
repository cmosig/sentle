from collections import OrderedDict

import numpy as np
import rasterio.crs
import sen2nbar.c_factor
from pystac import Item

from .stac import refresh_sas_token

# cheap least recently used cache
c_factor_cache = OrderedDict()


def _prepare_item_for_sen2nbar(stac_item: Item,
                               s2_crs: rasterio.crs.CRS) -> Item:
    """Make a STAC item digestible by ``sen2nbar.c_factor.c_factor_from_item``.

    sen2nbar (2024.6.0) reads the source EPSG from ``item.properties["proj:epsg"]``
    and fetches the ``granule-metadata`` asset over plain HTTP. Both break on the
    current Planetary Computer catalog:

    * PC dropped the deprecated ``proj:epsg`` (an int) in favour of the STAC
      projection-extension ``proj:code`` (e.g. ``"EPSG:32632"``), so the lookup
      raises ``KeyError: 'proj:epsg'`` (see issues #53, #59).
    * the ``granule-metadata`` XML lives in a private blob container and needs a
      SAS token, otherwise the request is rejected.

    sentle already knows the tile CRS (``s2_crs``), so we inject ``proj:epsg``
    from it and sign the metadata href. The mutation is idempotent (re-signing a
    signed href just refreshes the token). Returns the same item for convenience.
    """
    # provide the source EPSG sen2nbar expects, if the catalog didn't
    if "proj:epsg" not in stac_item.properties:
        epsg = s2_crs.to_epsg()
        if epsg is None:
            # fall back to the newer proj:code if present
            code = stac_item.properties.get("proj:code")
            if code is not None:
                epsg = int(str(code).split(":")[-1])
        stac_item.properties["proj:epsg"] = epsg

    # sign the granule-metadata asset so sen2nbar can download the XML
    gm = stac_item.assets.get("granule-metadata")
    if gm is not None:
        gm.href = refresh_sas_token(gm.href)

    return stac_item


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

        stac_item = _prepare_item_for_sen2nbar(stac_item, s2_crs)
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
