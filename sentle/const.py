S2_subtile_size = 732

S2_RAW_BANDS = [
    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09',
    'B11', 'B12'
]
S2_RAW_BAND_RESOLUTION = {
    'B01': 60,
    'B02': 10,
    'B03': 10,
    'B04': 10,
    'B05': 20,
    'B06': 20,
    'B07': 20,
    'B08': 10,
    'B8A': 20,
    'B09': 60,
    'B11': 20,
    'B12': 20
}
S2_NBAR_BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12']
S2_NBAR_INDICES_RAW_BANDS = [
    S2_RAW_BANDS.index(band) for band in S2_NBAR_BANDS
]

# zarr attrs that are necessary for xarray to be able to read the data in the end
ZARR_TIME_ATTRS = {
    'calendar': 'proleptic_gregorian',
    'units': 'seconds since 1970-01-01 00:00:00'
}

STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Copernicus Data Space Ecosystem (alternative provider, Sentinel-2 only)
CDSE_STAC_ENDPOINT = "https://stac.dataspace.copernicus.eu/v1"
CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"

# (connect, read) timeout in seconds for every STAC HTTP request. Without it
# requests passes timeout=None down to the socket and a peer that accepts the
# connection but never answers blocks the worker forever -- urllib3's Retry only
# fires on an exception, and a stalled recv() never raises one. See issue #87.
STAC_TIMEOUT = (10, 60)

# Items requested per STAC search page. Left unset, Planetary Computer serves
# 10 items per page, so a search over a year of a few tiles took dozens of
# round trips. 1000 is Planetary Computer's maximum and well within the 10000
# every stac-fastapi deployment (including CDSE) accepts.
STAC_SEARCH_PAGE_SIZE = 1000

# Degrees added on every side of the area's bbox for the single up-front item
# search. The per-ptile filter compares against each spatial chunk's own
# lon/lat bbox, which ``transform_bounds`` densifies independently of the full
# area's, so a chunk on the edge can poke out of the area bbox by a hair. The
# pad keeps the search a superset of every chunk; the extra items it pulls in
# are dropped again by the local filter.
STAC_SEARCH_BBOX_PAD = 0.01

# Upper bound in seconds on a server-sent Retry-After. urllib3 honours the
# header verbatim, so without a cap a single search can legally sleep for hours.
STAC_RETRY_AFTER_MAX = 120

# Extra attempts made after a raster read fails, before the run is aborted.
DEFAULT_READ_RETRIES = 2

# Seconds to wait before the first retry of a failed raster read; doubled for
# each further attempt.
READ_RETRY_BACKOFF = 1.0

# How long to wait for a Planetary Computer SAS token before giving up.
# ``planetary_computer.sign`` passes no timeout to requests, so a silent token
# endpoint would block a worker forever -- see ``stac.refresh_sas_token``.
SAS_SIGN_TIMEOUT = 120

# ``S2_skip_redundant_reads``: pixels (10 m) read beyond the part of a tile an
# acquisition needs, so resampling at the edge of that part still sees real
# neighbours. A multiple of 6 keeps the 60 m band grid aligned.
S2_NEEDED_WINDOW_MARGIN = 36

# ``S2_skip_redundant_reads``: degrees an item's footprint is grown by before
# reads outside it are skipped. STAC footprints are simplified outlines of the
# valid data, so this keeps real pixels just outside the outline.
S2_FOOTPRINT_BUFFER_DEG = 0.01

S1_ASSETS = ["vh_asc", "vh_desc", "vv_asc", "vv_desc"]
S1_TRUE_ASSETS = ["vv", "vh"]

ORBIT_STATE_ABBREVIATION = {"ascending": "asc", "descending": "desc"}
