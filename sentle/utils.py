from rasterio import transform


def bounds_from_transform_height_width_res(tf, height, width, resolution):
    # minx, miny, maxx, maxy
    return (tf.c, tf.f - (height * resolution), tf.c + (width * resolution),
            tf.f)


def transform_height_width_from_bounds_res(left, bottom, right, top, res):
    width, rem = divmod(right - left, res)
    assert rem == 0
    width = int(width)
    height, rem = divmod(top - bottom, res)
    assert rem == 0
    height = int(height)
    tf = transform.from_bounds(west=left,
                               south=bottom,
                               east=right,
                               north=top,
                               width=width,
                               height=height)

    return tf, height, width
