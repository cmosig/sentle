import warnings

import numpy as np
from affine import Affine
from rasterio import transform, warp, windows


def reproject_nodata_zero(*, source, destination, src_transform, src_crs,
                          dst_transform, dst_crs, resampling):
    """Reproject ``source`` into ``destination`` treating 0 as NoData.

    sentle uses an exact ``== 0`` comparison as its NoData sentinel everywhere
    downstream of reprojection (NoData masks, cloud masks, valid-pixel counts).
    The natural call is therefore ``warp.reproject(..., src_nodata=0,
    dst_nodata=0)``.

    However, GDAL >= 3.11 refuses to write a *valid* (non-NoData) pixel whose
    resampled value happens to equal ``dst_nodata``: with ``dst_nodata=0`` it
    silently bumps such pixels to ~1.4013e-45 (FLT_TRUE_MIN) and logs a
    ``CPLE_AppDefined`` warning ("Value 0 ... changed to 1.4013e-45 ... to
    avoid being treated as NoData"). In a multi-band warp this fires for every
    pixel that is zero in some-but-not-all bands. Those bumped values are no
    longer exactly 0, so they leak past sentle's ``== 0`` masks -- under-masking
    NoData and dropping clear pixels as cloudy. See GDAL issue #13677.

    To avoid the collision entirely we warp with ``dst_nodata=NaN`` (valid
    reflectance can never equal NaN, so GDAL never bumps and never warns) and
    then normalize the NaN fill back to 0, preserving sentle's 0-based NoData
    convention. The result is byte-identical to ``dst_nodata=0`` on GDAL < 3.11.
    """
    warp.reproject(source=source,
                   destination=destination,
                   src_transform=src_transform,
                   src_crs=src_crs,
                   dst_transform=dst_transform,
                   dst_crs=dst_crs,
                   src_nodata=0,
                   dst_nodata=np.nan,
                   resampling=resampling)

    # restore sentle's 0-based NoData sentinel (NaN fill -> 0)
    np.nan_to_num(destination, copy=False, nan=0.0)

    return destination


# relative tolerance used when checking whether a span is a whole number of
# pixels. Needed because fractional resolutions (e.g. 0.1 degrees for EPSG:4326)
# cannot be represented exactly in floating point, so ``span % res`` is almost
# never exactly 0 even when the user intended a whole number of pixels.
_PIXEL_TOL = 1e-6


def pixel_count(span, res):
    """Number of whole pixels spanning ``span`` at resolution ``res``.

    Rounds to the nearest integer when ``span`` is within ``_PIXEL_TOL`` (in
    pixel units) of a whole number of pixels; otherwise raises so callers that
    require exact divisibility still fail loudly.
    """
    n = abs(span) / res
    n_round = round(n)
    if abs(n - n_round) > _PIXEL_TOL:
        raise AssertionError(
            f"span {span} is not an integer multiple of resolution {res} "
            f"({n} pixels)")
    return int(n_round)


def _is_whole_pixels(span, res):
    n = abs(span) / res
    return abs(n - round(n)) <= _PIXEL_TOL


def check_and_round_bounds(left, bottom, right, top, res):
    if not _is_whole_pixels(top - bottom, res):
        warnings.warn(
            "Specified top/bottom bounds are not perfectly divisable by specified target_resolution. The resulting coverage will be rounded up to the next pixel value."
        )
        # trim the top down to a whole number of pixels above the bottom
        top = bottom + (abs(top - bottom) // res) * res

    if not _is_whole_pixels(right - left, res):
        warnings.warn(
            "Specified left/right bounds are not perfectly divisable by specified target_resolution. The resulting coverage will be rounded up to the next pixel value."
        )
        right = left + (abs(right - left) // res) * res

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

    if repr_width is None or repr_height is None:
        raise ValueError(
            "calculate_default_transform returned None for width or height")

    # include one more pixel because rounding down
    repr_width += 1
    repr_height += 1

    return tf, repr_height, repr_width


def height_width_from_bounds_res(left, bottom, right, top, res):
    # determine width and height based on bounds and resolution. Uses
    # tolerant rounding so fractional resolutions (e.g. 0.1 degrees) work
    # despite floating-point representation error, and always returns ints.
    width = pixel_count(right - left, res)
    height = pixel_count(top - bottom, res)
    return height, width


def spatial_chunk_grid(bound_left, bound_top, width, height, resolution,
                       chunk_size):
    """Yield the spatial processing chunks that tile a ``width`` x ``height``
    pixel grid in ``chunk_size`` steps.

    Iteration is in integer *pixel* space, deriving each chunk's CRS bounds
    from the pixel offsets (``bound + offset * resolution``). This avoids
    stepping ``range`` with a float stride -- which is invalid for fractional
    resolutions / geographic CRSs -- while producing exactly the same chunks as
    a coordinate-space loop for the integer case.

    Yields tuples ``(xi, yi, x_off, x_end, y_off, y_end, (left, bottom, right,
    top))`` where the ``*_off``/``*_end`` are pixel indices (the zarr write
    window) and the bounds are in CRS units. ``y_off`` counts pixels **down
    from the top**, so ``top`` is ``bound_top - y_off * resolution``.
    """
    for xi, x_off in enumerate(range(0, width, chunk_size)):
        x_end = min(x_off + chunk_size, width)
        left = bound_left + x_off * resolution
        right = bound_left + x_end * resolution
        for yi, y_off in enumerate(range(0, height, chunk_size)):
            y_end = min(y_off + chunk_size, height)
            top = bound_top - y_off * resolution
            bottom = bound_top - y_end * resolution
            yield xi, yi, x_off, x_end, y_off, y_end, (left, bottom, right, top)


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


def window_overlaps_bounds(win, overall_height, overall_width):
    """Return True if the window intersects the bounding box."""
    overlap_width = min(overall_width, win.col_off + win.width) - max(
        0, win.col_off)
    overlap_height = min(overall_height, win.row_off + win.height) - max(
        0, win.row_off)
    return overlap_width > 0 and overlap_height > 0


def bounds_from_transform_height_width_res(tf, height, width, resolution):
    # minx, miny, maxx, maxy
    return (tf.c, tf.f - (height * resolution), tf.c + (width * resolution),
            tf.f)


def transform_height_width_from_bounds_res(left, bottom, right, top, res):
    # tolerant rounding so fractional resolutions work despite float error
    width = pixel_count(right - left, res)
    height = pixel_count(top - bottom, res)
    tf = transform.from_bounds(west=left,
                               south=bottom,
                               east=right,
                               north=top,
                               width=width,
                               height=height)

    return tf, height, width
