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

S1_ASSETS = ["vh_asc", "vh_desc", "vv_asc", "vv_desc"]
S1_TRUE_ASSETS = ["vv", "vh"]

ORBIT_STATE_ABBREVIATION = {"ascending": "asc", "descending": "desc"}
