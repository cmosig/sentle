"""Tests for the STAC HTTP request timeout and retry policy (issue #87).

``StacApiIO`` defaults to ``timeout=None``, which reaches ``session.send`` and
ends up as ``sock.settimeout(None)``: a peer that accepts the connection and
never answers blocks the worker forever. The urllib3 ``Retry`` policy cannot
rescue that -- it retries on *exceptions*, and a stalled ``recv()`` never
raises one.

Setting the timeout on the ``StacApiIO`` alone is not enough: ``Client.open``
forwards its own ``timeout`` to ``stac_io.update()`` unconditionally, resetting
it back to ``None``. These tests therefore drive the real ``Client.open`` path,
with ``StacApiIO.request`` stubbed out so nothing touches the network.
"""

import json

import pytest
from pystac_client.stac_api_io import StacApiIO

from sentle import stac

_FAKE_CATALOG = {
    "type": "Catalog",
    "id": "fake",
    "stac_version": "1.0.0",
    "description": "offline stand-in",
    "conformsTo": [
        "https://api.stacspec.org/v1.0.0/core",
        "https://api.stacspec.org/v1.0.0/item-search",
    ],
    "links": [
        {
            "rel": "self",
            "href": "https://example.invalid/stac"
        },
        {
            "rel": "search",
            "href": "https://example.invalid/stac/search",
            "method": "POST"
        },
    ],
}


@pytest.fixture
def offline_catalog(monkeypatch):
    requested = []

    def fake_request(self, href, *args, **kwargs):
        requested.append(href)
        return json.dumps(_FAKE_CATALOG)

    monkeypatch.setattr(StacApiIO, "request", fake_request)
    return requested


def test_stac_timeout_is_a_connect_read_pair():
    connect, read = stac.STAC_TIMEOUT
    assert connect > 0 and read > 0


def test_get_stac_api_io_sets_the_timeout():
    assert stac.get_stac_api_io().timeout == stac.STAC_TIMEOUT


def test_timeout_reaches_session_send(monkeypatch):
    captured = {}
    io = stac.get_stac_api_io()

    class _Response:
        status_code = 200
        content = b"{}"
        headers = {}

        def raise_for_status(self):
            pass

    def fake_send(self, request, **kwargs):
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr(type(io.session), "send", fake_send)
    io.request("https://example.invalid/stac")

    # stored on the StacApiIO is not enough -- it has to be handed to requests
    assert captured["timeout"] == stac.STAC_TIMEOUT


@pytest.mark.parametrize("open_catalog", [
    lambda: stac.open_catalog(),
    lambda: stac.PlanetaryComputerProvider().open_catalog(),
    lambda: stac.CDSEProvider().open_catalog(),
])
def test_opened_catalog_keeps_the_timeout(offline_catalog, open_catalog):
    catalog = open_catalog()
    assert offline_catalog, "the fake landing page was never requested"
    # guards the Client.from_file clobber: setting timeout only on the
    # StacApiIO constructor leaves this None
    assert catalog._stac_io.timeout == stac.STAC_TIMEOUT


@pytest.mark.filterwarnings("ignore")
def test_search_inherits_the_timeout(offline_catalog):
    catalog = stac.open_catalog()
    search = catalog.search(collections=["sentinel-2-l2a"], limit=1)
    assert search._stac_io is catalog._stac_io
    assert search._stac_io.timeout == stac.STAC_TIMEOUT


def test_cdse_catalog_gets_the_retry_policy(offline_catalog):
    # CDSE used to be opened without a stac_io at all, i.e. with pystac-client's
    # default adapter: no timeout, no 5xx retries and no POST retries
    catalog = stac.CDSEProvider().open_catalog()
    retries = catalog._stac_io.session.get_adapter(
        "https://example.invalid/").max_retries
    assert retries.total == 15
    assert set(retries.status_forcelist) >= {502, 503, 504}
    assert retries.allowed_methods is None


def test_read_timeouts_are_retried_only_a_few_times():
    # every read timeout counts against total=15 too, so without a read cap a
    # stalled endpoint costs 16 x 60 s plus backoff (~34 min) per request
    retries = stac.get_stac_api_io().session.get_adapter(
        "https://example.invalid/").max_retries
    assert retries.read is not None and retries.read <= 5
    assert (retries.read + 1) * stac.STAC_TIMEOUT[1] <= 600


class _RetryAfterResponse:

    def __init__(self, seconds):
        self.headers = {"Retry-After": str(seconds)}

    def getheader(self, name, default=None):
        return self.headers.get(name, default)


def test_retry_after_is_capped_on_every_urllib3():
    # urllib3 honours Retry-After verbatim and only grew a retry_after_max
    # argument in 2.6.3; sentle declares no urllib3 floor, so the cap is applied
    # by CappedRetry instead and must hold regardless of the installed version
    retries = stac.get_stac_api_io().session.get_adapter(
        "https://example.invalid/").max_retries

    assert isinstance(retries, stac.CappedRetry)
    assert retries.get_retry_after(_RetryAfterResponse(86400)) == (
        stac.STAC_RETRY_AFTER_MAX)
    # a short, legitimate backoff is still honoured as sent
    assert retries.get_retry_after(_RetryAfterResponse(5)) == 5


def test_capped_retry_survives_being_cloned():
    # urllib3 rebuilds the policy on every attempt via Retry.new(), which uses
    # type(self) -- a subclass that broke that contract would silently revert to
    # the uncapped behaviour after the first retry
    retries = stac.get_stac_api_io().session.get_adapter(
        "https://example.invalid/").max_retries.increment(
            method="POST", url="https://example.invalid/search")

    assert isinstance(retries, stac.CappedRetry)
    assert retries.get_retry_after(_RetryAfterResponse(86400)) == (
        stac.STAC_RETRY_AFTER_MAX)
