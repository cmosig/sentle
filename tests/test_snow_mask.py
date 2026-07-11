"""Unit tests for the Potential Snow Layer (Zhu & Woodcock 2012, Eq. 20).

``compute_potential_snow_layer`` returns a boolean layer where **True means
clear** (keep) and **False means snow** -- it is used as a multiplicative mask
downstream, so the polarity matters. A pixel is flagged as snow only when all
three conditions hold: NDSI > 0.15, NIR (B08) > 0.11 and green (B03) > 0.1
(reflectance, i.e. DN / 10000).
"""

import numpy as np

from sentle.snow_mask import compute_potential_snow_layer


def test_snow_pixel_flagged_false():
    # high green, low SWIR -> high NDSI; NIR and green above thresholds -> snow
    B03 = np.array([[5000.0]])  # G = 0.5
    B11 = np.array([[1000.0]])  # S1 = 0.1 -> NDSI = 0.4/0.6 = 0.67 > 0.15
    B08 = np.array([[2000.0]])  # N = 0.2 > 0.11
    psl = compute_potential_snow_layer(B03=B03, B08=B08, B11=B11)
    assert psl.item() == False  # snow


def test_non_snow_pixel_flagged_true():
    # low NDSI (SWIR brighter than green) -> not snow -> clear
    B03 = np.array([[2000.0]])
    B11 = np.array([[3000.0]])  # NDSI negative
    B08 = np.array([[2000.0]])
    psl = compute_potential_snow_layer(B03=B03, B08=B08, B11=B11)
    assert psl.item() == True  # clear


def test_dark_pixel_below_green_threshold_is_clear():
    # green reflectance <= 0.1 -> snow test fails even with high NDSI
    B03 = np.array([[500.0]])   # G = 0.05 (< 0.1)
    B11 = np.array([[100.0]])   # very high NDSI ...
    B08 = np.array([[2000.0]])  # ... and NIR ok, but green too dark -> clear
    psl = compute_potential_snow_layer(B03=B03, B08=B08, B11=B11)
    assert psl.item() == True


def test_all_zero_pixel_does_not_crash_and_is_clear():
    # G + S1 == 0 -> NDSI is 0/0 = NaN; NaN > 0.15 is False -> clear, no error
    z = np.zeros((3, 3))
    psl = compute_potential_snow_layer(B03=z, B08=z, B11=z)
    assert psl.dtype == bool
    assert psl.all()  # every pixel clear (True)


def test_output_shape_and_dtype_preserved():
    rng = np.random.default_rng(0)
    shape = (16, 16)
    B03 = rng.uniform(0, 10000, shape)
    B08 = rng.uniform(0, 10000, shape)
    B11 = rng.uniform(0, 10000, shape)
    psl = compute_potential_snow_layer(B03=B03, B08=B08, B11=B11)
    assert psl.shape == shape
    assert psl.dtype == bool
