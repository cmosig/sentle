"""Count the HTTP requests sentle's Sentinel-2 read path makes for one ptile.

Planetary Computer rate-limits by request count, so this measures what a
change does to it without touching the real service: a local HTTP server
that understands range requests stands in for the blob storage and counts
what arrives; the read path under test (sentle's subtile/band reads, GDAL) is
the real one. The data is synthetic but shaped like a Sentinel-2 L2A tile
(10980 px at 10 m, 5490 px at 20 m, 1830 px at 60 m, 1024 px COG blocks), so
the block layout -- what decides the request count -- is realistic. Byte and
time figures are not: the data compresses far worse than real reflectances and
there is no network latency.

    python benchmarks/request_count.py [--size 3000] [--data-dir DIR]

Each configuration runs in a fresh process because GDAL caches file sizes and
blocks per URL for the life of the process, which would flatter every run
after the first.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RES = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20,
    "B07": 20, "B08": 10, "B8A": 20, "B09": 60, "B11": 20, "B12": 20
}
TILE = "32TPT"
# a chunk inside tile 32TPT (UTM 32N)
LEFT, TOP = 620000, 5280000


def make_data(directory):
    import numpy as np
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(0)
    for res, n in ((10, 10980), (20, 5490), (60, 1830)):
        path = f"{directory}/res{res}.tif"
        if os.path.exists(path):
            continue
        profile = dict(driver="GTiff", height=n, width=n, count=1,
                       dtype="uint16", crs=CRS.from_epsg(32632),
                       transform=from_origin(600000, 5300040, res, res),
                       tiled=True, blockxsize=1024, blockysize=1024,
                       compress="deflate")
        base = np.add.outer(np.arange(1024), np.arange(1024)) + 1200
        with rasterio.open(path, "w", **profile) as dst:
            for _, w in dst.block_windows(1):
                noise = rng.integers(0, 50, base.shape)
                dst.write((base + noise).astype("uint16")[:w.height, :w.width],
                          1, window=w)


def make_handler(directory, stats, lock):

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _serve(self, head):
            band = re.search(r"/(B\w+)\.tif", self.path).group(1)
            path = f"{directory}/res{RES[band]}.tif"
            size = os.path.getsize(path)
            rng = self.headers.get("Range")
            start, end, code = 0, size - 1, 200
            if rng:
                m = re.match(r"bytes=(\d+)-(\d*)", rng)
                start = int(m.group(1))
                end = min(int(m.group(2)) if m.group(2) else size - 1,
                          size - 1)
                code = 206
            n = end - start + 1
            with lock:
                stats["requests"] += 1
                stats["head" if head else "range"] += 1
                if not head:
                    stats["MB"] += n / 1e6
            self.send_response(code)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size if head else n))
            if code == 206:
                self.send_header("Content-Range",
                                 f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not head:
                with open(path, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(n))

        def do_HEAD(self):
            self._serve(True)

        def do_GET(self):
            self._serve(False)

    return Handler


def run_one(directory, size_px, options):
    """Child process: read one ptile with ``options``, print the counts."""
    import collections
    import time
    import warnings

    warnings.simplefilter("ignore")
    # the sandbox/CI proxy must not sit between GDAL and 127.0.0.1
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1"

    import geopandas as gpd
    import pandas as pd
    import pystac
    import rasterio
    from importlib.resources import files
    from rasterio.crs import CRS
    from rasterio.enums import Resampling

    from sentle import sentinel2
    from sentle.reproject_util import transform_height_width_from_bounds_res
    from sentle.stac import PlanetaryComputerProvider, gdal_read_options

    stats, lock = collections.Counter(), threading.Lock()
    server = ThreadingHTTPServer(("127.0.0.1", 0),
                                 make_handler(directory, stats, lock))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    class Provider(PlanetaryComputerProvider):

        def prepare_href(self, href):
            return href

        def rasterio_env(self):
            return rasterio.Env(**gdal_read_options())

    right, bottom = LEFT + size_px * 10, TOP - size_px * 10
    transform, height, width = transform_height_width_from_bounds_res(
        LEFT, bottom, right, TOP, 10)
    grid = gpd.read_file(str(files("sentle") / "data" /
                             "sentinel2_grid_stripped_with_epsg.gpkg"))
    subtiles = sentinel2.obtain_subtiles(CRS.from_epsg(32632), LEFT, bottom,
                                         right, TOP, grid)

    item_id = f"S2B_MSIL2A_20230601T101559_N0509_R065_T{TILE}_20230601T140000"
    item = pystac.Item(id=item_id, geometry=None, bbox=None,
                       datetime=pd.Timestamp("2023-06-01T10:15:59Z"
                                             ).to_pydatetime(),
                       properties={"s2:mgrs_tile": TILE,
                                   "s2:processing_baseline": "05.09"},
                       collection="sentinel-2-l2a")
    for band in RES:
        item.add_asset(band, pystac.Asset(
            href=f"http://127.0.0.1:{port}/{item_id}/{band}.tif"))

    started = time.time()
    out = sentinel2.process_ptile_S2_dispatcher(
        target_crs=CRS.from_epsg(32632), target_resolution=10,
        S2_cloud_classification_device="cpu", time_composite_freq=None,
        S2_apply_snow_mask=False, S2_apply_cloud_mask=False,
        S2_bands_to_save=list(RES), ptile_height=height, ptile_width=width,
        ptile_transform=transform, item_list=[item], ts=item.datetime,
        bound_left=LEFT, bound_right=right, bound_bottom=bottom,
        bound_top=TOP, S2_mask_snow=False, S2_cloud_classification=False,
        S2_return_cloud_probabilities=False, S2_nbar=False,
        S2_subtiles=subtiles, cloud_request_queue=None,
        cloud_response_queue=None, resampling_method=Resampling.nearest,
        S2_bands=list(RES), provider=Provider(), **options)
    print(json.dumps(dict(stats, seconds=round(time.time() - started, 1),
                          subtiles=len(subtiles),
                          checksum=float(out.sum(dtype="float64")))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--size", type=int, default=3000,
                        help="chunk edge in 10 m pixels (3000 = 30 km)")
    parser.add_argument("--data-dir",
                        default=os.path.join(tempfile.gettempdir(),
                                             "sentle_bench_data"))
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        run_one(args.data_dir, args.size, json.loads(args.child))
        return

    make_data(args.data_dir)
    print(f"{args.size} px chunk ({args.size / 100:g} km), one acquisition\n")
    checksums = set()
    for name, options in (("per-subtile reads (default)", {}),
                          ("read_ptile_windows=True",
                           {"read_ptile_windows": True})):
        result = subprocess.run(
            [sys.executable, __file__, "--size", str(args.size),
             "--data-dir", args.data_dir, "--child", json.dumps(options)],
            capture_output=True, text=True)
        if result.returncode:
            sys.exit(result.stderr)
        r = json.loads(result.stdout.strip().splitlines()[-1])
        checksums.add(r["checksum"])
        print(f"{name:30s} {r['requests']:4d} requests "
              f"({r.get('head', 0)} HEAD, {r['range']} range)  "
              f"{r['MB']:.0f} MB  {r['seconds']}s")
    print("\npixels identical:", len(checksums) == 1)


if __name__ == "__main__":
    main()
