"""The saved band labels must match the order the pipeline produces (issue #87).

``process_ptile`` writes the assembled array into the zarr band axis
positionally, while the ``/band`` coordinate is written from
``S2_bands_to_save``. Those two orders disagreed: the labels were built as
raw, snow, cloud, probabilities, but the pipeline produces raw, probabilities,
cloud, snow. So a cube built with both ``S2_mask_snow`` and
``S2_cloud_classification`` stored the two layers under each other's names, and
``S2_return_cloud_probabilities`` rotated the whole tail.

Only single-flag runs happened to line up, which is why nothing caught it.

Offline: pure list bookkeeping, no download and no model.
"""

import pytest

from sentle.cloud_mask import S2_cloud_mask_band, S2_cloud_prob_bands
from sentle.const import S2_RAW_BANDS
from sentle.sentle import _s2_bands_to_save
from sentle.snow_mask import S2_snow_mask_band


def _produced(snow, cloud, probs, freq):
    """The order process_ptile_S2 actually returns the bands in.

    process_S2_subtile appends the probability bands, process_ptile_S2 then
    appends the cloud classification and finally the snow mask; the mask layers
    are dropped again for a temporal composite.
    """
    bands = list(S2_RAW_BANDS)
    if cloud and probs:
        bands += S2_cloud_prob_bands
    if cloud and freq is None:
        bands.append(S2_cloud_mask_band)
    if snow and freq is None:
        bands.append(S2_snow_mask_band)
    return bands


@pytest.mark.parametrize("snow", [False, True])
@pytest.mark.parametrize("cloud", [False, True])
@pytest.mark.parametrize("probs", [False, True])
@pytest.mark.parametrize("freq", [None, "7d"])
def test_labels_match_production_order(snow, cloud, probs, freq):
    if probs and not cloud:
        pytest.skip("probabilities are only produced with cloud classification")

    labelled = _s2_bands_to_save(S2_bands=list(S2_RAW_BANDS),
                                 S2_mask_snow=snow,
                                 S2_cloud_classification=cloud,
                                 S2_return_cloud_probabilities=probs,
                                 time_composite_freq=freq)

    assert labelled == _produced(snow, cloud, probs, freq)


def test_snow_and_cloud_are_not_swapped():
    # the specific regression: both flags on, labels used to read
    # [..., S2_snow_mask, S2_cloud_classification] while the array held
    # [..., cloud, snow]
    labelled = _s2_bands_to_save(S2_bands=list(S2_RAW_BANDS),
                                 S2_mask_snow=True,
                                 S2_cloud_classification=True,
                                 S2_return_cloud_probabilities=False,
                                 time_composite_freq=None)

    assert labelled[-2:] == [S2_cloud_mask_band, S2_snow_mask_band]
