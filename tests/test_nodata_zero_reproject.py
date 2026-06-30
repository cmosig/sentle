"""Regression tests for the GDAL >= 3.11 NoData "value bump" issue.

Background
----------
sentle reprojects Sentinel-1/2 tiles with ``warp.reproject`` and uses an exact
``== 0`` comparison as its NoData sentinel everywhere downstream (NoData masks,
cloud masks, valid-pixel counts). The natural call is therefore
``src_nodata=0, dst_nodata=0``.

GDAL >= 3.11 (see GDAL issue #13677) refuses to write a *valid* (non-NoData)
pixel whose resampled value equals ``dst_nodata``. With ``dst_nodata=0`` it
silently bumps such pixels to ~1.4013e-45 (FLT_TRUE_MIN) and logs a
``CPLE_AppDefined`` warning. In a multi-band warp this fires for every pixel
that is zero in some-but-not-all bands. Those bumped values are no longer
exactly 0, so they leak past sentle's ``== 0`` masks -- under-masking NoData
and dropping clear pixels as cloudy.

The fix (``reproject_util.reproject_nodata_zero``) warps with
``dst_nodata=NaN`` (which valid reflectance can never collide with, so GDAL
never bumps and never warns) and normalizes the NaN fill back to 0. The result
is byte-identical to ``dst_nodata=0`` on GDAL < 3.11.

The tests come in two groups:

* ``TestReprojectHelper`` exercises the actual fix -- the helper must preserve
  the legacy ``dst_nodata=0`` output and must never emit a denormal "bump"
  artifact. On GDAL >= 3.11 the no-bump test fails against the old inline warp.

* ``TestDownstreamNoDataMasks`` drives the real ``process_ptile_S2_dispatcher``
  masking code and shows, deterministically on any GDAL version, that those
  ``== 0`` masks behave correctly when fed exact-0 NoData (the helper's
  contract) and misbehave when fed a bumped ~1.4013e-45 value -- i.e. why the
  helper must not emit one.
"""

import numpy as np
import pandas as pd
import pytest
from rasterio import transform, warp
from rasterio.crs import CRS
from rasterio.enums import Resampling

from sentle import sentinel2
from sentle.const import S2_RAW_BANDS
from sentle.cloud_mask import S2_cloud_mask_band
from sentle.reproject_util import reproject_nodata_zero

# smallest positive float32 denormal -- the value GDAL >= 3.11 bumps a valid
# zero-valued pixel to, to keep it from being read as NoData.
FLT_TRUE_MIN = np.nextafter(np.float32(0), np.float32(1))

UTM = CRS.from_epsg(32633)
WEBM = CRS.from_epsg(3857)


def _legacy_warp(source, destination, *, src_transform, src_crs, dst_transform,
                 dst_crs, resampling):
    """The pre-fix call: ``src_nodata=0, dst_nodata=0``."""
    warp.reproject(source=source,
                   destination=destination,
                   src_transform=src_transform,
                   src_crs=src_crs,
                   dst_transform=dst_transform,
                   dst_crs=dst_crs,
                   src_nodata=0,
                   dst_nodata=0,
                   resampling=resampling)
    return destination


def _make_warp_inputs(nbands, h=80, w=80, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.uniform(1, 10000, size=(nbands, h, w)).astype(np.float32)
    # a fully-NoData block (all bands 0) -> "unified" NoData
    src[:, :20, :20] = 0.0
    if nbands > 1:
        # pixels that are zero in some-but-not-all bands: the exact
        # UNIFIED_SRC_NODATA collision GDAL >= 3.11 bumps.
        src[0, 40:, 40:] = 0.0
        src[3, 25:30, 50:55] = 0.0
    src_tf = transform.from_bounds(300000, 5000000, 300000 + w * 10,
                                   5000000 + h * 10, w, h)
    dst_tf, dw, dh = warp.calculate_default_transform(
        UTM, WEBM, w, h, 300000, 5000000, 300000 + w * 10, 5000000 + h * 10,
        resolution=10)
    src = src if nbands > 1 else src[0]
    dst_shape = (nbands, dh, dw) if nbands > 1 else (dh, dw)
    return src, src_tf, dst_tf, dst_shape


class TestReprojectHelper:
    """Tests for the fix itself: ``reproject_nodata_zero``."""

    @pytest.mark.parametrize("nbands", [1, 12])
    @pytest.mark.parametrize(
        "resampling",
        [Resampling.nearest, Resampling.bilinear, Resampling.cubic])
    def test_matches_legacy_dst_nodata_zero(self, nbands, resampling):
        """On GDAL < 3.11 the helper is byte-identical to the legacy warp.

        (On GDAL >= 3.11 it differs only by *not* bumping zeros, which is the
        whole point -- see ``test_emits_no_denormal_bump``.)
        """
        src, src_tf, dst_tf, dst_shape = _make_warp_inputs(nbands)

        legacy = _legacy_warp(src, np.empty(dst_shape, np.float32),
                              src_transform=src_tf, src_crs=UTM,
                              dst_transform=dst_tf, dst_crs=WEBM,
                              resampling=resampling)
        fixed = reproject_nodata_zero(source=src,
                                      destination=np.empty(dst_shape, np.float32),
                                      src_transform=src_tf, src_crs=UTM,
                                      dst_transform=dst_tf, dst_crs=WEBM,
                                      resampling=resampling)

        # Both encode NoData as exactly 0; valid data must agree exactly.
        bump = (np.abs(legacy) > 0) & (np.abs(legacy) < 1e-30)
        if not bump.any():
            # GDAL < 3.11: no bump happened, outputs must be identical.
            assert np.array_equal(legacy, fixed)
        else:
            # GDAL >= 3.11: outputs agree everywhere the legacy warp did not
            # bump; where it did, the fix keeps an exact 0 instead.
            assert np.array_equal(fixed[~bump], legacy[~bump])
            assert np.all(fixed[bump] == 0)

    def test_emits_no_denormal_bump(self):
        """The helper never produces a denormal ~1.4e-45 bump artifact.

        This is the contract every downstream ``== 0`` mask relies on. It
        fails against the legacy ``dst_nodata=0`` warp on GDAL >= 3.11.
        """
        src, src_tf, dst_tf, dst_shape = _make_warp_inputs(12)
        out = reproject_nodata_zero(source=src,
                                    destination=np.empty(dst_shape, np.float32),
                                    src_transform=src_tf, src_crs=UTM,
                                    dst_transform=dst_tf, dst_crs=WEBM,
                                    resampling=Resampling.bilinear)
        assert not np.any((np.abs(out) > 0) & (np.abs(out) < 1e-30)), (
            "reproject output contains a denormal NoData-bump artifact")

    def test_nodata_region_is_zero_and_valid_preserved(self):
        """NoData fills as exactly 0; valid data is left untouched."""
        nbands, h, w = 4, 40, 40
        src = np.full((nbands, h, w), 500.0, dtype=np.float32)
        src[:, :10, :10] = 0.0  # NoData block
        tf = transform.from_bounds(0, 0, w, h, w, h)  # identity grid

        out = reproject_nodata_zero(source=src,
                                    destination=np.empty((nbands, h, w), np.float32),
                                    src_transform=tf, src_crs=UTM,
                                    dst_transform=tf, dst_crs=UTM,
                                    resampling=Resampling.nearest)

        assert np.all(out[:, :10, :10] == 0)        # NoData -> 0
        assert np.all(out[:, 20:, 20:] == 500.0)    # valid preserved exactly
        assert not np.isnan(out).any()              # NaN normalized away

    def test_single_band_s1_path_has_no_nan_or_bump(self):
        """Sentinel-1 reprojects single-band through the same helper."""
        src, src_tf, dst_tf, dst_shape = _make_warp_inputs(1)
        out = reproject_nodata_zero(source=src,
                                    destination=np.empty(dst_shape, np.float32),
                                    src_transform=src_tf, src_crs=UTM,
                                    dst_transform=dst_tf, dst_crs=WEBM,
                                    resampling=Resampling.bilinear)
        assert not np.isnan(out).any()
        assert not np.any((np.abs(out) > 0) & (np.abs(out) < 1e-30))


# --------------------------------------------------------------------------- #
# Downstream `== 0` masks: deterministic on any GDAL version.
# --------------------------------------------------------------------------- #

PTILE_H = 4
PTILE_W = 4


def _run_dispatcher(monkeypatch, *, returned_array, bands, apply_cloud_mask,
                    time_composite_freq=None, second_array=None):
    """Drive the real ``process_ptile_S2_dispatcher`` with faked per-timestamp
    outputs so the genuine ``== 0`` masking code runs."""

    class FakeItem:
        def __init__(self, tile, ts):
            self.properties = {"s2:mgrs_tile": tile}
            self.datetime = ts

    arrays = [returned_array] if second_array is None else [returned_array,
                                                            second_array]
    item_list = [FakeItem(f"T{i}", i) for i in range(len(arrays))]
    by_ts = {i: arrays[i] for i in range(len(arrays))}

    def fake_process_ptile_S2(*, timestamp, **kwargs):
        return by_ts[timestamp].copy(), list(bands)

    monkeypatch.setattr(sentinel2, "process_ptile_S2", fake_process_ptile_S2)

    return sentinel2.process_ptile_S2_dispatcher(
        target_crs=None,
        target_resolution=10.0,
        S2_cloud_classification_device="cpu",
        time_composite_freq=time_composite_freq,
        S2_apply_snow_mask=False,
        S2_apply_cloud_mask=apply_cloud_mask,
        S2_bands_to_save=list(bands),
        ptile_height=PTILE_H,
        ptile_width=PTILE_W,
        ptile_transform=None,
        item_list=item_list,
        ts=None,
        bound_left=0, bound_right=0, bound_bottom=0, bound_top=0,
        S2_mask_snow=False,
        S2_cloud_classification=apply_cloud_mask,
        S2_return_cloud_probabilities=False,
        S2_nbar=False,
        S2_subtiles=None,
        cloud_request_queue=None,
        cloud_response_queue=None,
        resampling_method=None,
    )


class TestDownstreamNoDataMasks:
    """The genuine ``== 0`` masks behave correctly on exact-0 NoData (the
    helper's contract) and misbehave on a bumped ~1.4013e-45 value."""

    def _raw_only_array(self, fill=500.0):
        arr = np.full((len(S2_RAW_BANDS), PTILE_H, PTILE_W), fill,
                      dtype=np.float32)
        return arr

    def test_nodata_mask_masks_zero_valued_band_pixel(self, monkeypatch):
        """A pixel that is exactly 0 in one raw band -> whole pixel NaN."""
        arr = self._raw_only_array()
        arr[0, 1, 1] = 0.0  # band B01 is NoData at (1, 1)

        out = _run_dispatcher(monkeypatch, returned_array=arr,
                              bands=S2_RAW_BANDS, apply_cloud_mask=False)

        assert np.all(np.isnan(out[:, 1, 1])), "zero-band pixel not masked"
        # every other pixel stays valid
        assert not np.isnan(np.delete(out.reshape(len(S2_RAW_BANDS), -1),
                                      1 * PTILE_W + 1, axis=1)).any()

    def test_nodata_mask_leaks_when_value_is_bumped(self, monkeypatch):
        """Demonstrates the bug the fix prevents: a bumped ~1.4e-45 value is
        no longer ``== 0`` so the NoData mask misses it and the pixel leaks
        through as valid."""
        arr = self._raw_only_array()
        arr[0, 1, 1] = FLT_TRUE_MIN  # what unpatched GDAL >= 3.11 would emit

        out = _run_dispatcher(monkeypatch, returned_array=arr,
                              bands=S2_RAW_BANDS, apply_cloud_mask=False)

        assert not np.isnan(out[:, 1, 1]).any(), (
            "bumped value unexpectedly masked -- update the test if GDAL "
            "behaviour changed")
        # the leaked value is ~0 but not exactly 0
        assert 0 < out[0, 1, 1] < 1e-30

    def test_cloud_mask_retains_clear_pixel(self, monkeypatch):
        """A clear pixel (cloud band exactly 0) is kept by the cloud mask."""
        bands = list(S2_RAW_BANDS) + [S2_cloud_mask_band]
        arr = np.full((len(bands), PTILE_H, PTILE_W), 500.0, dtype=np.float32)
        arr[-1, :, :] = 1.0          # everything cloudy ...
        arr[-1, 2, 2] = 0.0          # ... except one clear pixel

        out = _run_dispatcher(monkeypatch, returned_array=arr, bands=bands,
                              apply_cloud_mask=True)

        # clear pixel keeps its reflectance; cloudy pixels are zeroed -> NaN
        assert np.all(out[:len(S2_RAW_BANDS), 2, 2] == 500.0)
        assert np.isnan(out[0, 0, 0])

    def test_cloud_mask_drops_clear_pixel_when_bumped(self, monkeypatch):
        """Demonstrates the bug: a clear pixel whose cloud value got bumped to
        ~1.4e-45 is no longer ``== 0`` and is wrongly dropped as cloudy."""
        bands = list(S2_RAW_BANDS) + [S2_cloud_mask_band]
        arr = np.full((len(bands), PTILE_H, PTILE_W), 500.0, dtype=np.float32)
        arr[-1, :, :] = 1.0
        arr[-1, 2, 2] = FLT_TRUE_MIN  # clear pixel, but bumped

        out = _run_dispatcher(monkeypatch, returned_array=arr, bands=bands,
                              apply_cloud_mask=True)

        # the would-be-clear pixel was treated as cloudy -> zeroed -> NaN
        assert np.all(np.isnan(out[:len(S2_RAW_BANDS), 2, 2])), (
            "bumped clear pixel unexpectedly retained")

    def test_valid_pixel_count_excludes_exact_zero(self, monkeypatch):
        """In a temporal composite, an exact-0 (NoData) contribution must not
        be counted, so it does not dilute the mean of the valid acquisition."""
        # acquisition 1: valid 600 everywhere; acquisition 2: NoData (0) at the
        # probe pixel, valid 200 elsewhere.
        a1 = np.full((len(S2_RAW_BANDS), PTILE_H, PTILE_W), 600.0, np.float32)
        a2 = np.full((len(S2_RAW_BANDS), PTILE_H, PTILE_W), 200.0, np.float32)
        a2[:, 0, 0] = 0.0  # NoData in second acquisition at (0, 0)

        out = _run_dispatcher(monkeypatch, returned_array=a1, second_array=a2,
                              bands=S2_RAW_BANDS, apply_cloud_mask=False,
                              time_composite_freq="1W")

        # (0,0): only acquisition 1 counted -> mean is 600, not (600+0)/2=300
        assert np.allclose(out[:, 0, 0], 600.0)
        # elsewhere both counted -> mean of 600 and 200 = 400
        assert np.allclose(out[:, 1, 1], 400.0)
