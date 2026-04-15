"""Functions for component comparison quality control tests of irradiance measurements."""

import numpy as np


def diffuse_fraction_flag(
    ghi, dhi, solar_zenith, *, check="both", outside_domain_flag=False, nan_flag=False
):
    """Flag measurements where the diffuse fraction exceeds physically plausible limits.

    The diffuse fraction K = DHI / GHI is tested against solar-zenith-dependent
    upper limits when GHI exceeds 50 W/m². The limits are:

    - K must be < 1.05 for solar zenith < 75°
    - K must be < 1.10 for 75° ≤ solar zenith < 93°
    - not tested for GHI ≤ 50 W/m² or solar zenith ≥ 93°

    The comparison test is part of the BSRN QC tests [1]_, [2]_.

    Parameters
    ----------
    ghi : array-like of float
        Global horizontal irradiance [W/m²].
    dhi : array-like of float
        Diffuse horizontal irradiance [W/m²].
    solar_zenith : array-like of float
        Solar zenith angle [degrees].
    check : {'high-zenith', 'low-zenith', 'both'}, optional
        Which solar zenith angle domain to check. Default is ``'both'``.
    outside_domain_flag : bool, optional
        Value to assign to the flag when conditions are outside the
        valid test boundary. Can be either ``True`` or ``False``.
        Default is ``False``, which does not flag untested values as
        suspicious.
    nan_flag : bool, optional
        If ``True``, flag values where *ghi* or *dhi* is NaN. Default
        is ``False``, which does not flag NaN values as suspicious.

    Returns
    -------
    flag : same type as *ghi*
        Boolean array. ``True`` indicates the value failed the test,
        ``False`` indicates it passed or was outside the test domain.

    See Also
    --------
    bsrn_limits_flag

    References
    ----------
    .. [1] C. N. Long and Y. Shi, "An Automated Quality Assessment and Control
       Algorithm for Surface Radiation Measurements," *The Open Atmospheric
       Science Journal*, vol. 2, no. 1, pp. 23–37, Apr. 2008.
       :doi:`10.2174/1874282300802010023`
    .. [2] `C. N. Long and E. G. Dutton, "BSRN Global Network recommended QC
       tests, V2.0," BSRN, 2002.
       <https://bsrn.awi.de/fileadmin/user_upload/bsrn.awi.de/Publications/BSRN_recommended_QC_tests_V2.pdf>`_
    """
    # Suppress divide-by-zero warning
    with np.errstate(divide="ignore", invalid="ignore"):
        K = dhi / ghi
    # TODO: Consideer adding option to also test for (dhi > 50) | (ghi > 50)
    is_ghi_50 = ghi > 50

    is_low_zenith = solar_zenith < 75
    is_high_zenith = (solar_zenith >= 75) & (solar_zenith < 93)

    if check == "high-zenith":
        flag = is_ghi_50 & is_high_zenith & (K >= 1.10)
        outside_domain = np.logical_not(is_ghi_50 & is_high_zenith)
    elif check == "low-zenith":
        flag = is_ghi_50 & is_low_zenith & (K >= 1.05)
        outside_domain = np.logical_not(is_ghi_50 & is_low_zenith)
    elif check == "both":
        flag = is_ghi_50 & (is_low_zenith & (K >= 1.05)) | (
            is_high_zenith & (K >= 1.10)
        )
        outside_domain = np.logical_not(is_ghi_50 & (is_low_zenith | is_high_zenith))
    else:
        raise ValueError(
            f"check must be 'both', 'low-zenith', or 'high-zenith', got '{check}'."
        )

    if outside_domain_flag:
        flag = flag | outside_domain

    if nan_flag:
        flag = flag | np.isnan(dhi) | np.isnan(ghi)

    return flag
