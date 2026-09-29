"""Count the HTTP requests sentle's Sentinel-2 read path makes for one ptile.

Planetary Computer rate-limits by request count, so this measures what a
change does to it without touching the real service: a local HTTP server that
understands range requests stands in for the blob storage and counts what
arrives; the read path under test (sentle's subtile/band reads, GDAL) is the
real one.

For the chosen area every MGRS tile that covers it gets its own synthetic
COGs, on that tile's real CRS and 10 m / 20 m / 60 m pixel grid with 1024 px
blocks, so tile overlaps and UTM zone boundaries behave as they do on Planetary
Computer. Request counts are therefore realistic; byte and time figures are
not (the data compresses differently from real reflectances and there is no
network latency). The synthetic pixel values encode the tile they come from,
so where tiles overlap the output shows which one was read.

    python benchmarks/request_count.py                 # 2 tiles, one UTM zone
    python benchmarks/request_count.py --at 18.88,50.454  # 4 tiles, 2 zones
    python benchmarks/request_count.py --at 9.937,60.762 --size 3000

Each mode runs in a fresh process because GDAL caches file sizes and blocks
per URL for the life of the process, which would flatter every run after the
first.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RES = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20,
    "B07": 20, "B08": 10, "B8A": 20, "B09": 60, "B11": 20, "B12": 20
}
MODES = {
    "per-subtile reads (default)": {},
    "read_ptile_windows": {"read_ptile_windows": True},
    "skip_redundant_reads": {"skip_redundant_reads": True},
}


def chunk(lon, lat, size_px):
    """The ptile: ``size_px`` 10 m pixels square, centred on lon/lat, in the
    local UTM zone."""
    from rasterio import warp
    from rasterio.crs import CRS
    crs = CRS.from_epsg(32600 + int((lon + 180) // 6) + 1)
    x, y = warp.transform("EPSG:4326", crs, [lon], [lat])
    half = size_px * 10 / 2
    left, top = round(x[0] - half, -1), round(y[0] + half, -1)
    return crs, (left, top - size_px * 10, left + size_px * 10, top)


def subtiles_for(crs, bounds, drop_redundant=True):
    import geopandas as gpd
    from importlib.resources import files
    from sentle.sentinel2 import obtain_subtiles
    grid = gpd.read_file(str(files("sentle") / "data" /
                             "sentinel2_grid_stripped_with_epsg.gpkg"))
    return obtain_subtiles(crs, *bounds, grid, drop_redundant=drop_redundant)


def make_data(directory, subtiles):
    """One COG per tile and resolution. Blocks carry noise so they compress
    to roughly the size of real Sentinel-2 blocks; the base value is unique
    per tile."""
    import numpy as np
    import rasterio
    from rasterio.crs import CRS

    os.makedirs(directory, exist_ok=True)
    for tile in subtiles.drop_duplicates("name").itertuples():
        seed = zlib.crc32(tile.name.encode())
        rng = np.random.default_rng(seed)
        level = 1500 + seed % 3000
        for res, n in ((10, 10980), (20, 5490), (60, 1830)):
            path = f"{directory}/{tile.name}_{res}.tif"
            if os.path.exists(path):
                continue
            tf = tile.tile_transform
            profile = dict(driver="GTiff", height=n, width=n, count=1,
                           dtype="uint16",
                           crs=CRS.from_user_input(tile.tile_crs),
                           transform=rasterio.Affine(res, 0, tf.c, 0, -res,
                                                     tf.f),
                           tiled=True, blockxsize=1024, blockysize=1024,
                           compress="deflate")
            with rasterio.open(path + ".part", "w", **profile) as dst:
                for _, w in dst.block_windows(1):
                    noise = rng.integers(0, 50, (w.height, w.width))
                    dst.write((level + noise).astype("uint16"), 1, window=w)
            os.rename(path + ".part", path)


def make_handler(directory, stats, lock):

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _serve(self, head):
            tile, band = re.search(r"/(\w+)/(B\w+)\.tif", self.path).groups()
            path = f"{directory}/{tile}_{RES[band]}.tif"
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
                stats["tiles_opened"] = stats.get("tiles_opened", 0)
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


def run_one(args, options, out_path):
    """Child process: read one ptile with ``options``, print the counts."""
    import collections
    import time
    import warnings

    warnings.simplefilter("ignore")
    # a proxy must not sit between GDAL and 127.0.0.1
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1"

    import numpy as np
    import pandas as pd
    import pystac
    import rasterio
    from rasterio.enums import Resampling

    from sentle import sentinel2
    from sentle.reproject_util import transform_height_width_from_bounds_res
    from sentle.stac import PlanetaryComputerProvider, gdal_read_options

    stats, lock = collections.Counter(), threading.Lock()
    server = ThreadingHTTPServer(("127.0.0.1", 0),
                                 make_handler(args.data_dir, stats, lock))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    class Provider(PlanetaryComputerProvider):

        def prepare_href(self, href):
            return href

        def rasterio_env(self):
            return rasterio.Env(**gdal_read_options())

    lon, lat = map(float, args.at.split(","))
    crs, (left, bottom, right, top) = chunk(lon, lat, args.size)
    ptile_transform, height, width = transform_height_width_from_bounds_res(
        left, bottom, right, top, 10)
    subtiles = subtiles_for(crs, (left, bottom, right, top),
                            drop_redundant="skip_redundant_reads" not in options)

    when = pd.Timestamp("2023-06-01T10:15:59Z").to_pydatetime()
    items = []
    for tile in subtiles["name"].unique():
        item = pystac.Item(
            id=f"S2B_MSIL2A_20230601T101559_N0509_R065_T{tile}_2023",
            geometry=None, bbox=None, datetime=when,
            properties={"s2:mgrs_tile": tile,
                        "s2:processing_baseline": "05.09"},
            collection="sentinel-2-l2a")
        for band in RES:
            item.add_asset(band, pystac.Asset(
                href=f"http://127.0.0.1:{port}/{tile}/{band}.tif"))
        items.append(item)

    started = time.time()
    out = sentinel2.process_ptile_S2_dispatcher(
        target_crs=crs, target_resolution=10,
        S2_cloud_classification_device="cpu", time_composite_freq=None,
        S2_apply_snow_mask=False, S2_apply_cloud_mask=False,
        S2_bands_to_save=list(RES), ptile_height=height, ptile_width=width,
        ptile_transform=ptile_transform, item_list=items, ts=when,
        bound_left=left, bound_right=right, bound_bottom=bottom,
        bound_top=top, S2_mask_snow=False, S2_cloud_classification=False,
        S2_return_cloud_probabilities=False, S2_nbar=False,
        S2_subtiles=subtiles, cloud_request_queue=None,
        cloud_response_queue=None, resampling_method=Resampling.nearest,
        S2_bands=list(RES), provider=Provider(), **options)
    np.save(out_path, out)
    print(json.dumps(dict(stats, seconds=round(time.time() - started, 1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--at", default="11.335,37.689",
                        help="lon,lat of the chunk centre")
    parser.add_argument("--size", type=int, default=3000,
                        help="chunk edge in 10 m pixels (3000 = 30 km)")
    parser.add_argument("--data-dir",
                        default=os.path.join(tempfile.gettempdir(),
                                             "sentle_bench_tiles"))
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        run_one(args, json.loads(args.child), args.out)
        return

    import numpy as np

    lon, lat = map(float, args.at.split(","))
    subtiles = subtiles_for(*chunk(lon, lat, args.size), drop_redundant=False)
    tiles = sorted(subtiles["name"].unique())
    make_data(args.data_dir, subtiles)
    print(f"{args.size / 100:g} km chunk at {args.at}: {len(tiles)} MGRS "
          f"tile(s) {', '.join(tiles)}; one acquisition\n")

    outputs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, options in MODES.items():
            out_path = os.path.join(tmp, f"{len(outputs)}.npy")
            result = subprocess.run(
                [sys.executable, __file__, "--at", args.at, "--size",
                 str(args.size), "--data-dir", args.data_dir,
                 "--child", json.dumps(options), "--out", out_path],
                capture_output=True, text=True)
            if result.returncode:
                sys.exit(result.stderr)
            r = json.loads(result.stdout.strip().splitlines()[-1])
            outputs[name] = np.load(out_path)
            print(f"{name:28s} {r['requests']:4d} requests "
                  f"({r.get('head', 0):2d} HEAD, {r['range']:3d} range)  "
                  f"{r['MB']:5.0f} MB  {r['seconds']:5.1f}s")

    base = outputs.pop("per-subtile reads (default)")
    print()
    for name, out in outputs.items():
        same = (out == base) | (np.isnan(out) & np.isnan(base))
        print(f"{name}: {same.mean():.2%} of output values identical to the "
              f"default")
        if os.environ.get("SENTLE_BENCH_KEEP"):
            np.save(os.path.join(os.environ["SENTLE_BENCH_KEEP"],
                                 name + ".npy"), out)
    if os.environ.get("SENTLE_BENCH_KEEP"):
        np.save(os.path.join(os.environ["SENTLE_BENCH_KEEP"], "default.npy"),
                base)


if __name__ == "__main__":
    main()
