import contextlib
import re

import planetary_computer as pc
import pystac_client
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry
from urllib.parse import urlparse, urlunparse

from .const import (
    CDSE_S3_ENDPOINT,
    CDSE_STAC_ENDPOINT,
    S2_RAW_BAND_RESOLUTION,
    STAC_ENDPOINT,
)


def get_stac_api_io():
    """
    Returns a StacApiIO object with a retry policy that retries on 502, 503, 504
    with exponential backoff to handle server overload
    """
    retry = Retry(total=15,
                  backoff_factor=1.0,
                  backoff_jitter=0.2,
                  backoff_max=120,
                  status_forcelist=[502, 503, 504],
                  allowed_methods=None)
    return StacApiIO(max_retries=retry)


def open_catalog():
    return pystac_client.Client.open(STAC_ENDPOINT,
                                     stac_io=get_stac_api_io())


def refresh_sas_token(url):
    parsed = urlparse(url)
    unsigned = urlunparse(parsed._replace(query=""))
    new_signed = pc.sign(unsigned)
    return new_signed


# --------------------------------------------------------------------------- #
# Data-provider abstraction
#
# sentle can pull Sentinel data from more than one STAC catalog. The providers
# differ in the STAC endpoint, how an asset href is turned into something GDAL
# can read, how the (12) Sentinel-2 bands are named as assets, and which item
# properties are available. Everything provider-specific is isolated here so
# the rest of the pipeline only talks to a ``Provider``.
# --------------------------------------------------------------------------- #

# MGRS tile (e.g. 32TPS) and processing baseline (e.g. N0510) are encoded in the
# Sentinel-2 product id: S2A_MSIL2A_<sensing>_N<baseline>_R<orbit>_T<tile>_<proc>
_TILE_RE = re.compile(r"_T(\d{2}[A-Z]{3})_")
_BASELINE_RE = re.compile(r"_N(\d{4})_")


class PlanetaryComputerProvider:
    """Microsoft Planetary Computer (default). Signed HTTPS COGs."""

    name = "planetary_computer"
    supports_sentinel1 = True
    s2_collection = "sentinel-2-l2a"
    s1_collection = "sentinel-1-rtc"

    def open_catalog(self):
        return pystac_client.Client.open(STAC_ENDPOINT,
                                         stac_io=get_stac_api_io())

    def prepare_href(self, href):
        return refresh_sas_token(href)

    def rasterio_env(self):
        # PC hrefs are plain (signed) HTTPS -> no special GDAL config needed
        return contextlib.nullcontext()

    def s2_asset_key(self, band):
        return band

    def s2_mgrs_tile(self, item):
        return item.properties["s2:mgrs_tile"]

    def s2_processing_baseline(self, item):
        return float(item.properties["s2:processing_baseline"])

    def granule_metadata_href(self, item):
        return item.assets["granule-metadata"].href


class CDSEProvider:
    """Copernicus Data Space Ecosystem. JP2 assets read from CDSE S3.

    Reads require CDSE S3 credentials via the standard AWS chain (environment
    variables, or a profile selected with ``AWS_PROFILE``). Only Sentinel-2 is
    supported (CDSE has no Sentinel-1 RTC product).
    """

    name = "cdse"
    supports_sentinel1 = False
    s2_collection = "sentinel-2-l2a"
    s1_collection = None

    def open_catalog(self):
        return pystac_client.Client.open(CDSE_STAC_ENDPOINT)

    def prepare_href(self, href):
        # s3://eodata/...  ->  /vsis3/eodata/...
        if href.startswith("s3://"):
            return "/vsis3/" + href[len("s3://"):]
        return href

    def rasterio_env(self):
        # configure GDAL /vsis3/ for CDSE's (path-style) S3 endpoint using
        # whatever AWS credentials the standard chain provides.
        #
        # Small-AOI read cost depends on the processing baseline. From baseline
        # 05.12 (PSD 15.1, rolled out Q1 2026) the JP2s carry native TLM
        # (tile-part length) markers, so GDAL/openjpeg seeks straight to the
        # needed 1024x1024 tiles -- a cold crop is ~0.7s. Older products (< 05.12)
        # have NO TLM, so the first read must discover the tile structure by
        # scanning SOT markers via many small range requests (~7s cold). For that
        # older archive we mitigate by (a) ingesting ~1 MB at open + merging
        # consecutive ranges, and (b) keeping the dataset open across subtiles
        # (see ``reuse_open_datasets``), which amortizes the discovery and lets
        # GDAL reuse its decoded-tile block cache. See issue #75.
        import boto3
        import rasterio
        from rasterio.session import AWSSession
        return rasterio.Env(
            AWSSession(boto3.Session(), endpoint_url=CDSE_S3_ENDPOINT),
            AWS_VIRTUAL_HOSTING="FALSE",
            AWS_HTTPS="YES",
            GDAL_INGESTED_BYTES_AT_OPEN="1000000",
            GDAL_HTTP_MULTIRANGE="YES",
            GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
            VSI_CACHE="TRUE",
        )

    def s2_asset_key(self, band):
        # CDSE exposes each band at its native resolution as e.g. "B02_10m"
        return f"{band}_{S2_RAW_BAND_RESOLUTION[band]}m"

    def s2_mgrs_tile(self, item):
        # not exposed as a property on CDSE -> parse from the product id
        m = _TILE_RE.search(item.id)
        if m is None:
            raise ValueError(f"could not parse MGRS tile from id {item.id!r}")
        return m.group(1)

    def s2_processing_baseline(self, item):
        # not exposed as a property on CDSE -> parse N0510 -> 5.10 from the id
        m = _BASELINE_RE.search(item.id)
        if m is None:
            raise ValueError(f"could not parse baseline from id {item.id!r}")
        return int(m.group(1)) / 100.0

    def granule_metadata_href(self, item):
        # CDSE uses an underscore in the asset key
        return item.assets["granule_metadata"].href


_PROVIDERS = {
    PlanetaryComputerProvider.name: PlanetaryComputerProvider,
    CDSEProvider.name: CDSEProvider,
}


def get_provider(name):
    """Return a provider instance for ``name`` (validated by the caller)."""
    try:
        return _PROVIDERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown provider {name!r}; choose from {sorted(_PROVIDERS)}")
