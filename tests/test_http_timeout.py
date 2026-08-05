"""Tests for the GDAL/libcurl HTTP timeouts on the raster read path (issue #87).

GDAL ships with no read timeout at all: ``GDAL_HTTP_TIMEOUT``,
``GDAL_HTTP_LOW_SPEED_LIMIT`` and friends are unset on a stock install, so
libcurl uses ``CURLOPT_TIMEOUT=0`` (infinite). A peer that accepts the
connection and then goes silent blocked ``rasterio.open``/``read`` forever,
wedging a worker and -- since nothing else in the pipeline had a timeout either
-- the whole ``process()`` run.

sentle now enters a ``rasterio.Env`` with the *low-speed* knobs for every read.
They are the right ones: ``GDAL_HTTP_TIMEOUT`` caps total transfer time and
would false-abort a legitimately slow-but-progressing cold range read, while
``GDAL_HTTP_LOW_SPEED_LIMIT``/``_TIME`` fire only when throughput collapses.

Offline: the one socket test binds 127.0.0.1 and never leaves the machine.
"""

import socket
import threading
import time

import pytest
import rasterio
from rasterio import transform
from rasterio.crs import CRS
from rasterio.enums import Resampling

from sentle import sentinel1, stac
from sentle.stac import (
    CDSEProvider,
    PlanetaryComputerProvider,
    gdal_http_timeout_options,
)


class TestTimeoutOptions:

    def test_low_speed_abort_is_configured(self):
        options = gdal_http_timeout_options()
        assert options["GDAL_HTTP_LOW_SPEED_LIMIT"] == "1000"
        assert options["GDAL_HTTP_LOW_SPEED_TIME"] == "30"
        assert options["GDAL_HTTP_CONNECTTIMEOUT"] == "30"

    def test_no_hard_cap_on_total_transfer_time(self):
        # GDAL_HTTP_TIMEOUT aborts a read that is slow but healthy (a big cold
        # range read over a thin link), so it must not be set by default
        assert "GDAL_HTTP_TIMEOUT" not in gdal_http_timeout_options()

    def test_user_environment_variable_wins(self, monkeypatch):
        # rasterio.Env overrides GDAL's fallback to the process environment, so
        # keys the user set themselves have to be dropped or the documented
        # GDAL_HTTP_* escape hatch would silently stop working
        monkeypatch.setenv("GDAL_HTTP_LOW_SPEED_TIME", "5")
        options = gdal_http_timeout_options()
        assert "GDAL_HTTP_LOW_SPEED_TIME" not in options
        assert options["GDAL_HTTP_LOW_SPEED_LIMIT"] == "1000"

    def test_planetary_computer_env_applies_the_options(self):
        with PlanetaryComputerProvider().rasterio_env():
            # get_gdal_config normalizes the strings above to ints
            assert rasterio.env.get_gdal_config(
                "GDAL_HTTP_LOW_SPEED_LIMIT") == 1000
            assert rasterio.env.get_gdal_config("GDAL_HTTP_LOW_SPEED_TIME") == 30

    def test_cdse_env_keeps_s3_options_and_adds_timeouts(self, monkeypatch):
        # keep boto3 credential resolution local (no EC2 metadata endpoint)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "dummy")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy")
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")

        options = CDSEProvider().rasterio_env().options
        assert options["AWS_VIRTUAL_HOSTING"] == "FALSE"
        assert options["GDAL_INGESTED_BYTES_AT_OPEN"] == "1000000"
        assert options["GDAL_HTTP_LOW_SPEED_LIMIT"] == "1000"


def _stall_server():
    """A server that accepts connections and then never sends a single byte."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    held = []

    def accept_forever():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            # keep the connection open without ever answering
            held.append(conn)

    threading.Thread(target=accept_forever, daemon=True).start()
    return server, server.getsockname()[1]


def test_read_from_a_stalled_server_fails_in_bounded_time(monkeypatch):
    # shorten the abort window so the test is quick; without the Env this call
    # never returns at all
    monkeypatch.setitem(stac._GDAL_HTTP_TIMEOUT_OPTIONS,
                        "GDAL_HTTP_LOW_SPEED_TIME", "1")

    server, port = _stall_server()
    try:
        started = time.monotonic()
        with pytest.raises(rasterio.errors.RasterioIOError):
            with PlanetaryComputerProvider().rasterio_env():
                rasterio.open(f"http://127.0.0.1:{port}/stalled.tif")
        assert time.monotonic() - started < 30
    finally:
        server.close()


class _Asset:

    def __init__(self, href):
        self.href = href


class _Item:

    def __init__(self):
        self.id = "S1A_IW_GRDH_1SDV_20230615T054321"
        self.properties = {"sat:orbit_state": "ascending"}
        self.assets = {"vv": _Asset("https://host/vv.tif")}


def test_sentinel1_reads_are_inside_the_timeout_env(monkeypatch):
    # Sentinel-1 never went through provider.rasterio_env(), so a stac.py-only
    # fix would have left every S1 read unbounded
    seen = {}

    def fake_open(href, *args, **kwargs):
        seen["limit"] = rasterio.env.get_gdal_config("GDAL_HTTP_LOW_SPEED_LIMIT")
        seen["time"] = rasterio.env.get_gdal_config("GDAL_HTTP_LOW_SPEED_TIME")
        raise rasterio.errors.RasterioIOError("nope")

    monkeypatch.setattr(sentinel1.rasterio, "open", fake_open)
    monkeypatch.setattr(sentinel1, "refresh_sas_token", lambda href: href)

    target_crs = CRS.from_epsg(32632)
    with pytest.warns(UserWarning, match="stac_read_failure"):
        sentinel1.process_ptile_S1(
            target_crs=target_crs,
            target_resolution=10,
            time_composite_freq=None,
            bound_left=600000,
            bound_right=600100,
            bound_bottom=5099900,
            bound_top=5100000,
            ts=None,
            S1_assets=["vv_asc"],
            ptile_height=10,
            ptile_width=10,
            ptile_transform=transform.from_origin(600000, 5100000, 10, 10),
            item_list=[_Item()],
            resampling_method=Resampling.nearest,
        )

    assert seen["limit"] == 1000
    assert seen["time"] == 30
