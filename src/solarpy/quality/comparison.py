"""Functions for component comparison quality tests of irradiance measurements."""

import numpy as np


def diffuse_fraction_flag(
    ghi,
    dhi,
    solar_zenith,
    *,
    zenith_domain="both",
    outside_domain_flag=False,
    ghi_threshold_on="measured",
    nan_flag=False,
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
    zenith_domain : {'both', 'low', 'high'}, optional
        Which solar zenith angle domain to check. Default is ``'both'``.
    outside_domain_flag : bool, optional
        Value to assign to the flag when conditions are outside the
        valid test boundary. Can be either ``True`` or ``False``.
        Default is ``False``, which does not flag untested values as
        suspicious.
    ghi_threshold_on : {'measured', 'both'}, optional
        Which parameter to apply the GHI threshold to, either measured GHI
        or both measured GHI and DHI. Default is ``'measured'``.
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
    if ghi_threshold_on == "measured":
        is_ghi_50 = ghi > 50
    elif ghi_threshold_on == "both":
        is_ghi_50 = (ghi > 50) | (dhi > 50)
    else:
        raise ValueError(
            f"ghi_threshold_on must be 'measured' or 'both', got '{ghi_threshold_on}'."
        )

    is_low_zenith = solar_zenith < 75
    is_high_zenith = (solar_zenith >= 75) & (solar_zenith < 93)

    if zenith_domain == "high":
        flag = is_ghi_50 & is_high_zenith & (K >= 1.10)
        outside_domain = np.logical_not(is_ghi_50 & is_high_zenith)
    elif zenith_domain == "low":
        flag = is_ghi_50 & is_low_zenith & (K >= 1.05)
        outside_domain = np.logical_not(is_ghi_50 & is_low_zenith)
    elif zenith_domain == "both":
        flag = is_ghi_50 & (is_low_zenith & (K >= 1.05)) | (
            is_high_zenith & (K >= 1.10)
        )
        outside_domain = np.logical_not(is_ghi_50 & (is_low_zenith | is_high_zenith))
    else:
        raise ValueError(
            f"zenith_domain must be 'both', 'low', or 'high', got '{zenith_domain}'."
        )

    if outside_domain_flag:
        flag = flag | outside_domain

    if nan_flag:
        flag = flag | np.isnan(dhi) | np.isnan(ghi)

    return flag


def closure_flag(
    ghi,
    dni,
    dhi,
    solar_zenith,
    *,
    zenith_domain="both",
    outside_domain_flag=False,
    ghi_threshold_on="calculated",
    nan_flag=False,
):
    """Flag measurements where the three-component closure ratio exceeds plausible limits.

    The closure ratio R = GHI / (DHI + DNI · cos(SZA)) is compared against
    solar-zenith-dependent limits when the component sum exceeds 50 W/m².

    The limits are:

    - R must be within ±8% of 1.0 for solar zenith < 75°
    - R must be within ±15% of 1.0 for 75° ≤ solar zenith < 93°
    - not tested for component sum ≤ 50 W/m² or solar zenith ≥ 93°

    The comparison test is part of the BSRN QC tests [1]_, [2]_.

    Parameters
    ----------
    ghi : array-like of float
        Global horizontal irradiance [W/m²].
    dni : array-like of float
        Direct normal irradiance [W/m²].
    dhi : array-like of float
        Diffuse horizontal irradiance [W/m²].
    solar_zenith : array-like of float
        Solar zenith angle [degrees].
    zenith_domain : {'both', 'low', high'}, optional
        Which solar zenith angle domain to zenith_domain. Default is ``'both'``.
    outside_domain_flag : bool, optional
        Value to assign to the flag when conditions are outside the
        valid test boundary. Can be either ``True`` or ``False``.
        Default is ``False``, which does not flag untested values as
        suspicious.
    ghi_threshold_on : {'measured', 'calculated', 'both'}, optional
        Which parameter to apply the GHI threshold to, either measured GHI,
        calculated GHI, or both. Default is ``'calculated'``.
    nan_flag : bool, optional
        If ``True``, flag values where *ghi*, *dhi*, or *dni* is NaN.
        Default is ``False``, which does not flag NaN values as suspicious.

    Returns
    -------
    flag : same type as *ghi*
        Boolean array. ``True`` indicates the value failed the test,
        ``False`` indicates it passed or was outside the test domain.

    See Also
    --------
    diffuse_fraction_flag

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
    mu0 = np.cos(np.radians(solar_zenith))
    sum_sw = dhi + dni * mu0

    with np.errstate(divide="ignore", invalid="ignore"):
        R = ghi / sum_sw

    if ghi_threshold_on == "measured":
        is_ghi_50 = ghi > 50
    elif ghi_threshold_on == "calculated":
        is_ghi_50 = sum_sw > 50
    elif ghi_threshold_on == "both":
        is_ghi_50 = (ghi > 50) | (sum_sw > 50)
    else:
        raise ValueError(
            f"ghi_threshold_on must be 'measured', 'calculated', or 'both', got '{ghi_threshold_on}'."
        )

    is_low_zenith = solar_zenith < 75
    is_high_zenith = (solar_zenith >= 75) & (solar_zenith < 93)

    if zenith_domain == "high":
        flag = is_ghi_50 & is_high_zenith & (np.abs(R - 1.0) >= 0.15)
        outside_domain = np.logical_not(is_ghi_50 & is_high_zenith)
    elif zenith_domain == "low":
        flag = is_ghi_50 & is_low_zenith & (np.abs(R - 1.0) >= 0.08)
        outside_domain = np.logical_not(is_ghi_50 & is_low_zenith)
    elif zenith_domain == "both":
        flag = is_ghi_50 & (
            (is_low_zenith & (np.abs(R - 1.0) >= 0.08))
            | (is_high_zenith & (np.abs(R - 1.0) >= 0.15))
        )
        outside_domain = np.logical_not(is_ghi_50 & (is_low_zenith | is_high_zenith))
    else:
        raise ValueError(
            f"zenith_domain must be 'both', 'low', or 'high', got '{zenith_domain}'."
        )

    if outside_domain_flag:
        flag = flag | outside_domain

    if nan_flag:
        flag = flag | np.isnan(ghi) | np.isnan(dhi) | np.isnan(dni)

    return flag
