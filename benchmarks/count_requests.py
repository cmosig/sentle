"""Count the HTTP requests and bytes one ``sentle.process`` run sends to
Planetary Computer's blob storage, plus the main process' STAC and SAS-token
requests.

Every signed blob URL is rewritten to a local proxy that forwards each request
(with its Range header) to the real blob and counts it, so the numbers are
exact. Run one measurement per fresh process: GDAL caches file headers per URL
for the life of a process.

    python benchmarks/count_requests.py '{"target_crs": "EPSG:32632",
        "target_resolution": 10, "bound_left": 380000, "bound_bottom": 5280000,
        "bound_right": 410000, "bound_top": 5310000, "datetime": "2023-06-03",
        "S1_assets": null, "num_workers": 1}'
"""

import json
import multiprocessing as mp
import os
import re
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

import sentle.sentinel1 as sentinel1
import sentle.stac as stac
from sentle.sentle import process

_local = threading.local()


class _Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    counts = None  # shared across the forked workers, see main()

    def log_message(self, *args):
        pass

    def _forward(self, method):
        account, _, rest = self.path.lstrip("/").partition("/")
        if not hasattr(_local, "session"):
            _local.session = requests.Session()
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() == "range"}
        r = _local.session.request(
            method, f"https://{account}.blob.core.windows.net/{rest}",
            headers=headers, timeout=120)
        body = r.content if method == "GET" else b""
        self.send_response(r.status_code)
        for key in ("Content-Range", "Content-Type", "Accept-Ranges", "ETag",
                    "Last-Modified"):
            if key in r.headers:
                self.send_header(key, r.headers[key])
        self.send_header("Content-Length", str(
            len(body) if method == "GET" else
            r.headers.get("Content-Length", 0)))
        self.end_headers()
        self.wfile.write(body)
        with self.counts.get_lock():
            self.counts[0] += 1
            self.counts[1] += method == "HEAD"
            self.counts[2] += r.status_code >= 400
            self.counts[3] += len(body)

    def do_GET(self):
        self._forward("GET")

    def do_HEAD(self):
        self._forward("HEAD")


def main():
    kwargs = json.loads(sys.argv[1])
    _Proxy.counts = mp.Array("d", 4)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    def via_proxy(sign):

        def signed(*args, **kw):
            return re.sub(r"https://([a-z0-9]+)\.blob\.core\.windows\.net/",
                          rf"http://127.0.0.1:{port}/\1/", sign(*args, **kw))

        return signed

    sign = stac.PlanetaryComputerProvider.prepare_href
    stac.PlanetaryComputerProvider.prepare_href = (
        lambda self, href: via_proxy(sign)(self, href))
    sentinel1.refresh_sas_token = via_proxy(sentinel1.refresh_sas_token)

    api = {"stac_search": 0, "stac_other": 0, "sas": 0}
    send = requests.Session.send

    def counting_send(self, request, **kw):
        if "/api/sas/" in request.url:
            api["sas"] += 1
        elif "/api/stac/" in request.url:
            api["stac_search" if "/search" in request.url else
                "stac_other"] += 1
        return send(self, request, **kw)

    requests.Session.send = counting_send

    with tempfile.TemporaryDirectory() as tmp:
        kwargs.setdefault("zarr_store", os.path.join(tmp, "cube.zarr"))
        start = time.time()
        process(**kwargs)
        elapsed = time.time() - start

    total, head, errors, nbytes = _Proxy.counts[:]
    print(json.dumps(dict(blob_requests=int(total), blob_head=int(head),
                          blob_errors=int(errors),
                          blob_megabytes=round(nbytes / 1e6, 1),
                          seconds=round(elapsed, 1), **api)))


if __name__ == "__main__":
    main()
