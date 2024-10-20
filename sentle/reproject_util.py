import warnings

from affine import Affine
from rasterio import transform, warp, windows


def check_and_round_bounds(left, bottom, right, top, res):
    h_rem = abs(top - bottom) % res
    if h_rem != 0:
        warnings.warn(
            "Specified top/bottom bounds are not perfectly divisable by specified target_resolution. The resulting coverage will be rounded up to the next pixel value."
        )
        top -= h_rem

    w_rem = abs(right - left) % res
    if w_rem != 0:
        warnings.warn(
            "Specified left/right bounds are not perfectly divisable by specified target_resolution. The resulting coverage will be rounded up to the next pixel value."
        )
        right -= w_rem

    return left, bottom, right, top


def calculate_aligned_transform(src_crs, dst_crs, height, width, left, bottom,
                                right, top, tres):
    tf, repr_width, repr_height = warp.calculate_default_transform(
        src_crs=src_crs,
        dst_crs=dst_crs,
        width=width,
        height=height,
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        resolution=tres)

    tf = Affine(
        tf.a,
        tf.b,
        tf.c - (tf.c % tres),
        tf.d,
        tf.e,
        # + target_resolution because upper left corner
        tf.f - (tf.f % tres) + tres,
        tf.g,
        tf.h,
        tf.i,
    )

    # include one more pixel because rounding down
    repr_width += 1
    repr_height += 1

    return tf, repr_height, repr_width


def height_width_from_bounds_res(left, bottom, right, top, res):
    # determine width and height based on bounds and resolution
    width, w_rem = divmod(abs(right - left), res)
    assert w_rem == 0
    height, h_rem = divmod(abs(top - bottom), res)
    assert h_rem == 0
    return height, width


def recrop_write_window(win, overall_height, overall_width):
    """ Determine write window based on overlap with actual bounds and also
    return how the array that will be written needs to be cropped. """

    # global
    grow = win.row_off
    gcol = win.col_off
    gwidth = win.width
    gheight = win.height

    # local
    lrow = 0
    lcol = 0
    lwidth = win.width
    lheight = win.height

    # if overlapping to the left
    if gcol < 0:
        lcol = abs(gcol)
        gwidth -= abs(gcol)
        lwidth -= abs(gcol)
        gcol = 0

    # if overlapping on the bottom
    if grow < 0:
        lrow = abs(grow)
        gheight -= abs(grow)
        lheight -= abs(grow)
        grow = 0

    # if overlapping to the right
    if overall_width < (gcol + gwidth):
        difwidth = (gcol + gwidth) - overall_width
        gwidth -= difwidth
        lwidth -= difwidth

    # if overlapping to the top
    if overall_height < (grow + gheight):
        difheight = (grow + gheight) - overall_height
        gheight -= difheight
        lheight -= difheight

    assert gcol >= 0
    assert grow >= 0
    assert gwidth <= win.width
    assert gheight <= win.height
    assert overall_height >= (grow + gheight)
    assert overall_width >= (gcol + gwidth)
    assert lcol >= 0
    assert lrow >= 0
    assert lwidth <= win.width
    assert lheight <= win.height
    assert lwidth == gwidth
    assert lheight == gheight
    assert all(
        x % 1 == 0
        for x in [grow, gcol, gwidth, gheight, lrow, lcol, lwidth, lheight])

    return windows.Window(
        row_off=grow, col_off=gcol, height=gheight,
        width=gwidth).round_offsets().round_lengths(), windows.Window(
            row_off=lrow, col_off=lcol, height=lheight,
            width=lwidth).round_offsets().round_lengths()


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
