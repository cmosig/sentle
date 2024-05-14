import utm

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


def bounds_from_transform_height_width_res(transform, height, width,
                                           resolution):
    # minx, miny, maxx, maxy
    return (transform.c, transform.f - (height * resolution),
            transform.c + (width * resolution), transform.f)


def utm_epsg_from_latlon(lat: float, lon: float):
    utm_code = 32600 + utm.from_latlon(lat, lon)[2]
    utm_code = (utm_code - 100) if (lat < 0) else utm_code
    return f"EPSG:{utm_code}"
