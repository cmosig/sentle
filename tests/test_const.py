"""Consistency checks for the band/asset constants.

These are the lookup tables the whole pipeline indexes into. A mismatch (a band
without a resolution, an NBAR index pointing at the wrong band) would silently
mis-align data rather than raise, so it is worth pinning them down.
"""

from sentle.const import (
    ORBIT_STATE_ABBREVIATION,
    S1_ASSETS,
    S2_NBAR_BANDS,
    S2_NBAR_INDICES_RAW_BANDS,
    S2_RAW_BAND_RESOLUTION,
    S2_RAW_BANDS,
    S2_subtile_size,
)


def test_every_raw_band_has_a_resolution():
    assert set(S2_RAW_BAND_RESOLUTION) == set(S2_RAW_BANDS)


def test_resolutions_are_valid_sentinel2_ground_sample_distances():
    assert set(S2_RAW_BAND_RESOLUTION.values()) <= {10, 20, 60}


def test_subtile_size_divides_the_full_tile():
    # obtain_subtiles / process_S2_subtile rely on this exact divisibility
    assert 10980 % S2_subtile_size == 0
    assert 16 <= S2_subtile_size <= 10980


def test_nbar_indices_point_at_the_correct_raw_bands():
    assert len(S2_NBAR_INDICES_RAW_BANDS) == len(S2_NBAR_BANDS)
    for band, idx in zip(S2_NBAR_BANDS, S2_NBAR_INDICES_RAW_BANDS):
        assert S2_RAW_BANDS[idx] == band


def test_nbar_bands_are_a_subset_of_raw_bands():
    assert set(S2_NBAR_BANDS) <= set(S2_RAW_BANDS)


def test_s1_assets_orbit_encoding():
    # every S1 asset is <polarization>_<orbit-abbreviation>
    valid_orbits = set(ORBIT_STATE_ABBREVIATION.values())
    for asset in S1_ASSETS:
        pol, orbit = asset.split("_")
        assert pol in {"vv", "vh"}
        assert orbit in valid_orbits
