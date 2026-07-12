"""Tests for ``validate_user_input`` -- sentle's public-API guardrail.

``process()`` funnels every user argument through ``validate_user_input`` before
any (expensive, networked) work starts. Each check below corresponds to a way a
caller can misconfigure a run; the point of the guardrail is to fail loudly and
early rather than halfway through a multi-hour download.
"""

import warnings

import pytest
from rasterio.enums import Resampling

from sentle.sentle import validate_user_input


def _valid_kwargs(**overrides):
    """A fully-valid argument set; override single keys to probe one check."""
    kwargs = dict(
        target_crs="EPSG:32633",
        target_resolution=10.0,
        zarr_store="out.zarr",
        bound_left=0.0,
        bound_bottom=0.0,
        bound_right=1000.0,
        bound_top=1000.0,
        zarr_store_chunk_size={"time": 1, "y": 100, "x": 100},
        datetime="2020-01-01/2020-01-31",
        S2_nbar=False,
        resampling_method=Resampling.nearest,
        processing_spatial_chunk_size=4000,
        S1_assets=["vh_asc", "vv_asc"],
        S2_mask_snow=False,
        S2_cloud_classification=False,
        S2_cloud_classification_device="cpu",
        S2_return_cloud_probabilities=False,
        num_workers=1,
        time_composite_freq=None,
        S2_apply_snow_mask=False,
        S2_apply_cloud_mask=False,
        save_as_uint16=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_input_passes():
    # a well-formed configuration must not raise
    validate_user_input(**_valid_kwargs())


class TestCrsAndResolution:
    def test_invalid_crs_raises(self):
        with pytest.raises(ValueError, match="target_crs"):
            validate_user_input(**_valid_kwargs(target_crs="not-a-crs"))

    def test_non_numeric_resolution_raises(self):
        with pytest.raises(ValueError, match="target_resolution"):
            validate_user_input(**_valid_kwargs(target_resolution="10"))

    def test_zero_resolution_raises(self):
        with pytest.raises(ValueError, match="greater than 0"):
            validate_user_input(**_valid_kwargs(target_resolution=0))

    def test_negative_resolution_raises(self):
        with pytest.raises(ValueError, match="greater than 0"):
            validate_user_input(**_valid_kwargs(target_resolution=-5))


class TestBounds:
    def test_non_numeric_bound_raises(self):
        with pytest.raises(ValueError, match="bound"):
            validate_user_input(**_valid_kwargs(bound_left="0"))

    def test_left_not_less_than_right_raises(self):
        with pytest.raises(ValueError, match="bound_left must be less"):
            validate_user_input(**_valid_kwargs(bound_left=1000, bound_right=0))

    def test_bottom_not_less_than_top_raises(self):
        with pytest.raises(ValueError, match="bound_bottom must be less"):
            validate_user_input(**_valid_kwargs(bound_bottom=1000,
                                                bound_top=0))


class TestChunkSizes:
    def test_non_int_processing_chunk_raises(self):
        with pytest.raises(ValueError, match="processing_spatial_chunk_size"):
            validate_user_input(
                **_valid_kwargs(processing_spatial_chunk_size=4000.0))

    def test_too_small_processing_chunk_raises(self):
        with pytest.raises(ValueError, match="greater than 1000"):
            validate_user_input(
                **_valid_kwargs(processing_spatial_chunk_size=500))

    def test_zarr_chunk_size_must_be_dict(self):
        with pytest.raises(ValueError, match="must be a dictionary"):
            validate_user_input(**_valid_kwargs(zarr_store_chunk_size=[1, 2, 3]))

    def test_zarr_chunk_size_missing_keys(self):
        with pytest.raises(ValueError, match="'time', 'y', and 'x'"):
            validate_user_input(
                **_valid_kwargs(zarr_store_chunk_size={"time": 1, "y": 100}))


class TestS1Assets:
    def test_non_list_raises(self):
        with pytest.raises(ValueError, match="S1_assets must be a list"):
            validate_user_input(**_valid_kwargs(S1_assets="vh_asc"))

    def test_unknown_asset_raises(self):
        with pytest.raises(ValueError, match="vh_asc"):
            validate_user_input(**_valid_kwargs(S1_assets=["vh_asc", "bogus"]))

    def test_none_is_allowed(self):
        validate_user_input(**_valid_kwargs(S1_assets=None))


class TestBooleanFlags:
    @pytest.mark.parametrize("flag", [
        "S2_mask_snow", "S2_cloud_classification",
        "S2_return_cloud_probabilities", "S2_apply_snow_mask",
        "S2_apply_cloud_mask", "S2_nbar", "save_as_uint16",
    ])
    def test_non_boolean_flag_raises(self, flag):
        with pytest.raises(ValueError, match=flag):
            validate_user_input(**_valid_kwargs(**{flag: "yes"}))

    def test_device_must_be_cpu_or_cuda(self):
        with pytest.raises(ValueError, match="cpu.*cuda"):
            validate_user_input(
                **_valid_kwargs(S2_cloud_classification_device="gpu"))

    def test_num_workers_must_be_int(self):
        with pytest.raises(ValueError, match="num_workers"):
            validate_user_input(**_valid_kwargs(num_workers=1.5))


class TestDependentFlagCombinations:
    def test_apply_snow_mask_requires_mask_snow(self):
        with pytest.raises(ValueError, match="S2_apply_snow_mask"):
            validate_user_input(
                **_valid_kwargs(S2_apply_snow_mask=True, S2_mask_snow=False))

    def test_apply_cloud_mask_requires_cloud_classification(self):
        with pytest.raises(ValueError, match="S2_apply_cloud_mask"):
            validate_user_input(**_valid_kwargs(
                S2_apply_cloud_mask=True, S2_cloud_classification=False))

    def test_temporal_composite_without_masks_warns(self):
        with pytest.warns(UserWarning, match="Temporal aggregation"):
            validate_user_input(**_valid_kwargs(time_composite_freq="1W"))

    def test_temporal_composite_with_mask_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_user_input(**_valid_kwargs(
                time_composite_freq="1W",
                S2_cloud_classification=True,
                S2_apply_cloud_mask=True))


class TestUint16:
    def test_uint16_with_s1_enabled_raises(self):
        with pytest.raises(ValueError, match="save_as_uint16"):
            validate_user_input(**_valid_kwargs(
                save_as_uint16=True, S1_assets=["vh_asc"]))

    def test_uint16_with_s1_disabled_none_ok(self):
        validate_user_input(
            **_valid_kwargs(save_as_uint16=True, S1_assets=None))

    def test_uint16_with_s1_empty_list_ok(self):
        validate_user_input(
            **_valid_kwargs(save_as_uint16=True, S1_assets=[]))


class TestResamplingAndStore:
    def test_resampling_method_must_be_enum(self):
        with pytest.raises(ValueError, match="resampling_method"):
            validate_user_input(**_valid_kwargs(resampling_method="nearest"))

    def test_zarr_store_wrong_type_raises(self):
        with pytest.raises(ValueError, match="zarr_store"):
            validate_user_input(**_valid_kwargs(zarr_store=12345))


class TestS2SubtileSize:
    # divisors of 10980 that are multiples of 6 and >= 366
    @pytest.mark.parametrize("size", [366, 732, 1098, 1830, 2196, 10980])
    def test_valid_sizes_are_allowed(self, size):
        validate_user_input(**_valid_kwargs(S2_subtile_size=size))

    def test_default_is_valid(self):
        # the default 732 must always pass
        validate_user_input(**_valid_kwargs())

    def test_non_integer_raises(self):
        with pytest.raises(ValueError, match="S2_subtile_size must be an integer"):
            validate_user_input(**_valid_kwargs(S2_subtile_size=732.0))

    def test_non_divisor_raises(self):
        with pytest.raises(ValueError, match="divisor of 10980"):
            validate_user_input(**_valid_kwargs(S2_subtile_size=500))

    @pytest.mark.parametrize("size", [244, 305, 610])
    def test_non_multiple_of_six_raises(self, size):
        # 244 (=4*61) divides 10980 but is not a multiple of 6, so the 60 m
        # bands would read on a fractional grid -> rejected
        with pytest.raises(ValueError, match="multiple of 6"):
            validate_user_input(**_valid_kwargs(S2_subtile_size=size))

    def test_too_small_raises(self):
        # 180 divides 10980 and is a multiple of 6 but is below the 366 floor
        with pytest.raises(ValueError, match="between 366 and 10980"):
            validate_user_input(**_valid_kwargs(S2_subtile_size=180))

    def test_too_large_raises(self):
        with pytest.raises(ValueError, match="divisor of 10980"):
            validate_user_input(**_valid_kwargs(S2_subtile_size=20000))
