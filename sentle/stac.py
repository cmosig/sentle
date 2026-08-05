import os
import queue
import re
import threading

import planetary_computer as pc
import pystac_client
import rasterio
from pystac_client.stac_api_io import StacApiIO
from urllib3 import Retry
from urllib.parse import urlparse, urlunparse

from .const import (
    CDSE_S3_ENDPOINT,
    CDSE_STAC_ENDPOINT,
    S2_RAW_BAND_RESOLUTION,
    SAS_SIGN_TIMEOUT,
    STAC_ENDPOINT,
    STAC_RETRY_AFTER_MAX,
    STAC_TIMEOUT,
)


class CappedRetry(Retry):
    """``Retry`` that never sleeps longer than ``STAC_RETRY_AFTER_MAX``.

    urllib3 honours a server-sent ``Retry-After`` verbatim, and only capped it
    from 2.6.3 on (via a ``retry_after_max`` argument that does not exist on
    older releases). sentle declares no urllib3 constraint -- it arrives
    transitively via requests -- so the cap is applied here instead, which works
    on every version. Without it a ``Retry-After: 3600`` answered 15 times is a
    15-hour wait inside a single search (issue #87).
    """

    def get_retry_after(self, response):
        seconds = super().get_retry_after(response)
        if seconds is None:
            return None
        return min(seconds, STAC_RETRY_AFTER_MAX)


def get_stac_api_io():
    """
    Returns a StacApiIO object with a retry policy that retries on 502, 503, 504
    with exponential backoff to handle server overload, and a timeout so a
    request can never wait forever (issue #87).

    ``read=3`` caps how often a *read timeout* is retried while leaving all 15
    attempts available for 502/503/504 responses: without it a stalled endpoint
    costs 16 x 60 s plus backoff (~34 min) per STAC request.
    """
    retry = CappedRetry(total=15,
                        read=3,
                        backoff_factor=1.0,
                        backoff_jitter=0.2,
                        backoff_max=120,
                        status_forcelist=[502, 503, 504],
                        allowed_methods=None)
    api_io = StacApiIO(max_retries=retry, timeout=STAC_TIMEOUT)
    # pystac-client 0.7.7 (the floor declared in setup.py) stores the timeout
    # and then calls update() without forwarding it, resetting it to None
    api_io.timeout = STAC_TIMEOUT
    return api_io


def open_catalog():
    return _open_stac_client(STAC_ENDPOINT)


def _open_stac_client(endpoint):
    """Open a STAC catalog with sentle's retry policy and request timeout.

    ``timeout`` has to be passed to ``Client.open`` as well as to the
    ``StacApiIO``: ``Client.from_file`` calls ``stac_io.update(...,
    timeout=timeout)`` unconditionally, so an unset ``timeout`` here would reset
    the one the ``StacApiIO`` was constructed with back to ``None``.
    """
    return pystac_client.Client.open(endpoint,
                                     stac_io=get_stac_api_io(),
                                     timeout=STAC_TIMEOUT)


# --------------------------------------------------------------------------- #
# GDAL/libcurl HTTP timeouts
#
# GDAL ships without any read timeout: every GDAL_HTTP_* knob below reads back
# as None on a stock install, so libcurl uses its own defaults --
# CURLOPT_TIMEOUT 0 (infinite) and CURLOPT_LOW_SPEED_LIMIT 0 (disabled). A peer
# that accepts the connection and then goes silent -- before or in the middle of
# a range response -- blocks rasterio.open()/read() forever, which wedges a
# worker and, since nothing else in the pipeline has a timeout either, the whole
# run. See issue #87.
#
# The bound is deliberately expressed with the *low-speed* knobs rather than
# GDAL_HTTP_TIMEOUT. GDAL_HTTP_TIMEOUT caps the total transfer time, so it
# aborts reads that are slow but healthy (a big cold JP2 range read over a thin
# link fails under GDAL_HTTP_TIMEOUT=5 even though it is progressing the whole
# time). LOW_SPEED_LIMIT/LOW_SPEED_TIME fire only when throughput actually
# collapses, which covers both stall shapes. 1000 B/s sustained for 30 s is
# orders of magnitude below any healthy Planetary Computer / CDSE read. GDAL
# retries a failed range download internally, so the wall-clock ceiling per dead
# asset is roughly 6 x GDAL_HTTP_LOW_SPEED_TIME.
_GDAL_HTTP_TIMEOUT_OPTIONS = {
    "GDAL_HTTP_CONNECTTIMEOUT": "30",
    "GDAL_HTTP_LOW_SPEED_LIMIT": "1000",  # bytes per second
    "GDAL_HTTP_LOW_SPEED_TIME": "30",  # seconds below the limit -> abort
    "GDAL_HTTP_TCP_KEEPALIVE": "YES",
}


def gdal_http_timeout_options():
    """HTTP timeout options for GDAL, minus any the user set themselves.

    ``rasterio.Env(**options)`` overrides GDAL's fallback to the process
    environment, so passing a key unconditionally would silently defeat a user
    who tuned it through the standard ``GDAL_HTTP_*`` environment variables.
    Keys the user actually supplied a value for are therefore dropped and the
    user's value wins. An empty value does not count: GDAL parses ``""`` as 0,
    which *disables* the abort, and an empty variable is easy to produce by
    accident (an ``ENV GDAL_HTTP_LOW_SPEED_LIMIT=`` in a Dockerfile, say), so
    sentle keeps its own default in that case rather than silently going back
    to an unbounded read.
    """
    return {
        key: value
        for key, value in _GDAL_HTTP_TIMEOUT_OPTIONS.items()
        if not os.environ.get(key, "").strip()
    }


class SasSigningTimeout(RuntimeError):
    """Signing an asset href with a Planetary Computer SAS token timed out."""


class _SignWorker:
    """One reusable daemon thread that runs ``planetary_computer.sign`` calls.

    ``pc.sign`` fetches the SAS token with a ``requests.Session`` it builds
    itself and passes no ``timeout``, so a token endpoint that accepts the
    connection and then goes silent blocks the caller forever -- the same
    unbounded wait as everything else in issue #87, on the hottest path in the
    pipeline (once per asset read). There is no way to inject a timeout into
    that session, so the call is handed to this thread and abandoned if it does
    not come back.

    The thread is reused (a fresh thread per call would cost ~6x more) and is a
    *daemon* so an abandoned, permanently blocked one cannot hold up interpreter
    shutdown. A worker that has been abandoned is never reused: it may still be
    stuck in the dead request, which would make every later call wait behind it.
    """

    def __init__(self):
        self._requests = queue.Queue()
        self.usable = True
        threading.Thread(target=self._loop,
                         daemon=True,
                         name="sentle-sas-sign").start()

    def _loop(self):
        while True:
            unsigned, done, outcome = self._requests.get()
            try:
                outcome["href"] = pc.sign(unsigned)
            except BaseException as exc:  # reported to the caller verbatim
                outcome["error"] = exc
            done.set()

    def sign(self, unsigned, timeout):
        done, outcome = threading.Event(), {}
        self._requests.put((unsigned, done, outcome))
        if not done.wait(timeout):
            self.usable = False
            raise SasSigningTimeout(
                f"signing {unsigned} with a Planetary Computer SAS token took "
                f"longer than {timeout:g}s -- the token endpoint is not "
                f"responding")
        if "error" in outcome:
            raise outcome["error"]
        return outcome["href"]


_sign_worker = None
_sign_worker_pid = None
_sign_worker_lock = threading.Lock()


def _get_sign_worker():
    global _sign_worker, _sign_worker_pid
    with _sign_worker_lock:
        # threads do not survive fork, and sentle forks its workers
        if (_sign_worker is None or not _sign_worker.usable
                or _sign_worker_pid != os.getpid()):
            _sign_worker = _SignWorker()
            _sign_worker_pid = os.getpid()
        return _sign_worker


def refresh_sas_token(url, timeout=SAS_SIGN_TIMEOUT):
    """Strip any (possibly expired) SAS token off ``url`` and re-sign it.

    ``timeout`` bounds the token request; pass ``None`` to sign on this thread
    and wait indefinitely.
    """
    parsed = urlparse(url)
    unsigned = urlunparse(parsed._replace(query=""))
    if timeout is None:
        return pc.sign(unsigned)
    return _get_sign_worker().sign(unsigned, timeout)


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
        return _open_stac_client(STAC_ENDPOINT)

    def prepare_href(self, href):
        return refresh_sas_token(href)

    def rasterio_env(self):
        # PC hrefs are plain (signed) HTTPS -> no S3 config needed, but the
        # reads still need a timeout (see gdal_http_timeout_options)
        return rasterio.Env(**gdal_http_timeout_options())

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
        # pystac-client's default StacApiIO has no timeout and retries GET only,
        # so give CDSE the same policy as Planetary Computer
        return _open_stac_client(CDSE_STAC_ENDPOINT)

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
        from rasterio.session import AWSSession
        return rasterio.Env(
            AWSSession(boto3.Session(), endpoint_url=CDSE_S3_ENDPOINT),
            AWS_VIRTUAL_HOSTING="FALSE",
            AWS_HTTPS="YES",
            GDAL_INGESTED_BYTES_AT_OPEN="1000000",
            GDAL_HTTP_MULTIRANGE="YES",
            GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
            VSI_CACHE="TRUE",
            **gdal_http_timeout_options(),
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
