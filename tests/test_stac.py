"""Unit tests for ``stac.refresh_sas_token``.

Planetary Computer asset URLs carry a short-lived SAS token in the query
string. Before every read sentle strips whatever (possibly expired) token is on
the URL and re-signs the bare URL, so a long-running job never reads with a
stale token. These tests pin that contract without touching the network by
stubbing ``planetary_computer.sign``.
"""

from urllib.parse import parse_qs, urlparse

from sentle import stac


def test_strips_existing_query_and_resigns(monkeypatch):
    seen = {}

    def fake_sign(url):
        seen["unsigned"] = url
        return url + "?st=fresh&sig=NEWTOKEN"

    monkeypatch.setattr(stac.pc, "sign", fake_sign)

    stale = ("https://sentinel2.blob.core.windows.net/"
             "sentinel-2-l2a/tile/B02.tif?st=old&se=old&sig=EXPIRED")
    out = stac.refresh_sas_token(stale)

    # the bare URL handed to pc.sign carries no query string
    assert seen["unsigned"] == ("https://sentinel2.blob.core.windows.net/"
                                "sentinel-2-l2a/tile/B02.tif")
    # the returned URL carries the fresh token and none of the old one
    q = parse_qs(urlparse(out).query)
    assert q["sig"] == ["NEWTOKEN"]
    assert "EXPIRED" not in out


def test_preserves_path_and_host(monkeypatch):
    monkeypatch.setattr(stac.pc, "sign", lambda url: url + "?sig=X")
    out = stac.refresh_sas_token("https://host.example/a/b/c.tif?sig=Y")
    parsed = urlparse(out)
    assert parsed.netloc == "host.example"
    assert parsed.path == "/a/b/c.tif"


def test_unsigned_url_without_query_is_signed_as_is(monkeypatch):
    monkeypatch.setattr(stac.pc, "sign", lambda url: url + "?sig=X")
    out = stac.refresh_sas_token("https://host.example/a.tif")
    assert out == "https://host.example/a.tif?sig=X"
