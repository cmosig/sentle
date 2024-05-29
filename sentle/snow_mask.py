import numpy as np
import xarray as xr

S2_snow_mask_band = "S2_snow_mask"


def compute_potential_snow_layer(B03, B08, B11):
    """Creates the Potential Snow Layer (PSL) as described by Equation 20 of Zhu and
    Woodcock, 2012 [1]_.

    References
    ----------
    .. [1] https://doi.org/10.1016/j.rse.2011.10.028
    """

    # Select the required bands and scale
    G = B03 / 10000
    N = B08 / 10000
    S1 = B11 / 10000

    # Compute the Normalized Difference Snow Index
    NDSI = (G - S1) / (G + S1)

    # Eq. 20. (Zhu and Woodcock, 2012) and invert (True is clear, False is snow)
    PSL = ~((NDSI > 0.15) & (N > 0.11) & (G > 0.1))

    return PSL
