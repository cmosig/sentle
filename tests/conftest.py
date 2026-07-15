"""Shared fixtures for the sentle test suite.

The whole suite is designed to run **offline** -- no Planetary Computer access,
no model download, no network. Anything that would normally hit the network is
either a pure-geometry computation on the bundled MGRS grid or is driven with
monkeypatched stand-ins for the downloaded rasters / STAC items.
"""

from importlib.resources import files

import geopandas as gpd
import pytest


@pytest.fixture(scope="session")
def s2grid():
    """The bundled Sentinel-2 MGRS tiling grid (packaged with sentle)."""
    return gpd.read_file(
        str(files("sentle") / "data" /
            "sentinel2_grid_stripped_with_epsg.gpkg"))
