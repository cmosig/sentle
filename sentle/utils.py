def bounds_from_transform_height_width_res(transform, height, width,
                                           resolution):
    # minx, miny, maxx, maxy
    return (transform.c, transform.f - (height * resolution),
            transform.c + (width * resolution), transform.f)


def transform_height_width_from_bounds_res(left, bottom, right, top, res):
    width = (right - left) / res
    height = (top - bottom) / res
    transform = transform.from_bounds(west=left,
                                      south=bottom,
                                      east=right,
                                      north=top,
                                      width=width,
                                      height=height)

    return transform, height, width
