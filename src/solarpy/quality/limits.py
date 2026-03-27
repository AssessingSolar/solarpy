"""Functions for BSRN quality control irradiance limit tests."""

import numpy as np


_BSRN_LIMITS = {
    "ppl-ghi": (1.50, 1.2,  100.,   -4.),
    "erl-ghi": (1.20, 1.2,   50.,   -2.),
    "ppl-dni": (1.00, 1.0,    0.,   -4.),
    "erl-dni": (0.95, 0.2,   10.,   -2.),
    "ppl-dhi": (0.95, 1.2,   50.,   -4.),
    "erl-dhi": (0.75, 1.2,   30.,   -2.),
}


def bsrn_limits(solar_zenith, dni_extra, limits):
    """Calculate the BSRN upper and/or lower irradiance limit values.

    The BSRN upper and lower bound limit checks were developed by Long & Shi
    (2008) [1]_, [2]_. The upper limit follows the form::

        upper = scale * DNI_extra * cos(solar_zenith) ^ exponent + offset

    where *scale*, *exponent*, and *offset* are coefficients that depend on
    the variable and test level. A value is flagged if it lies outside
    [lower, upper].

    Parameters
    ----------
    solar_zenith : array-like of float
        Solar zenith angle [degrees].
    dni_extra : array-like of float
        Extraterrestrial normal irradiance [W/m²].
    limits : str or tuple of float
        Either a named limit string or a tuple ``(scale, exponent, offset, lower)``.

        Named limit (Long & Shi, 2008):

        - ``"ppl-ghi"`` — Physically Possible Limit for GHI
        - ``"erl-ghi"`` — Extremely Rare Limit for GHI
        - ``"ppl-dni"`` — Physically Possible Limit for DNI
        - ``"erl-dni"`` — Extremely Rare Limit for DNI
        - ``"ppl-dhi"`` — Physically Possible Limit for DHI
        - ``"erl-dhi"`` — Extremely Rare Limit for DHI

        When passing a tuple, provide ``(scale, exponent, offset, lower)``
        where the upper bound is
        ``scale * dni_extra * cos(solar_zenith) ** exponent + offset`` and
        *lower* is the minimum allowed value.

    Returns
    -------
    lower : float
        Lower limit value [W/m²].
    upper : same type as input
        Upper limit values [W/m²].

    See Also
    --------
    bsrn_limits_flag : Test irradiance values against these limits.

    References
    ----------
    .. [1] C. N. Long and Y. Shi, "An Automated Quality Assessment and Control
       Algorithm for Surface Radiation Measurements," *The Open Atmospheric
       Science Journal*, vol. 2, no. 1, pp. 23–37, Apr. 2008.
       :doi:`10.2174/1874282300802010023`
    .. [2] C. N. Long and Y. Shi, "An Automated Quality Assessment and Control
       Algorithm for Surface Radiation Measurements," BSRN, 2002. [Online].
       Available: `BSRN recommended QC tests v2
       <https://bsrn.awi.de/fileadmin/user_upload/bsrn.awi.de/Publications/BSRN_recommended_QC_tests_V2.pdf>`_
    """
    if isinstance(limits, str):
        if limits not in _BSRN_LIMITS:
            raise ValueError(
                f"Unknown limit '{limits}'. "
                f"Valid options are: {list(_BSRN_LIMITS.keys())}."
            )
        scale, exponent, offset, lower = _BSRN_LIMITS[limits]
    elif isinstance(limits, tuple):
        if len(limits) != 4:
            raise ValueError(
                f"limit tuple must have 4 elements (scale, exponent, offset, lower), "
                f"got {len(limits)}."
            )
        scale, exponent, offset, lower = limits
    else:
        raise ValueError("limit must be a string or a tuple of 4 floats.")

    cos_sza = np.cos(np.deg2rad(solar_zenith))
    upper = scale * dni_extra * cos_sza ** exponent + offset

    return lower, upper


def bsrn_limits_flag(irradiance, solar_zenith, dni_extra, limits, check='both', nan_flag=False):
    """Flag irradiance values that fall outside the BSRN quality control limits.

    Parameters
    ----------
    irradiance : array-like of float
        Irradiance values to check [W/m²].
    solar_zenith : array-like of float
        Solar zenith angle [degrees]. Must be the same length as *irradiance*.
    dni_extra : array-like of float
        Extraterrestrial normal irradiance [W/m²]. Must be the same length
        as *irradiance*.
    limits : str or tuple of float
        Either a named limit string or a tuple of coefficients
        ``(scale, exponent, offset, lower)``.

        Named limit (Long & Shi, 2008) [1]_, [2]_:

        - ``"ppl-ghi"`` — Physically Possible Limit for GHI
        - ``"erl-ghi"`` — Extremely Rare Limit for GHI
        - ``"ppl-dni"`` — Physically Possible Limit for DNI
        - ``"erl-dni"`` — Extremely Rare Limit for DNI
        - ``"ppl-dhi"`` — Physically Possible Limit for DHI
        - ``"erl-dhi"`` — Extremely Rare Limit for DHI

        When passing a tuple, provide ``(scale, exponent, offset, lower)`` where the upper
        bound is
        ``scale * dni_extra * cos(solar_zenith) ** exponent + offset`` and
        *lower* is the minimum allowed value.
    check : {'both', 'upper', 'lower'}, optional
        Which bounds to check. Default is ``'both'``.
    nan_flag : bool, optional
        Flag value to assign when *irradiance* is NaN. Default is ``False``,
        which does not flag NaN values as suspicious.

    Returns
    -------
    flag : same type as *irradiance*
        Boolean array of the same length as *irradiance*. ``True`` indicates
        the value failed the test (outside bounds), ``False`` indicates
        it passed.

    See Also
    --------
    bsrn_limits : Calculate the limit values without testing.

    Examples
    --------
    Test GHI measurements against the BSRN limits:

    >>> import pandas as pd
    >>> import numpy as np
    >>> import pvlib
    >>>
    >>> # One year of hourly timestamps for Copenhagen
    >>> times = pd.date_range("2023-01-01", periods=8760, freq="h", tz="UTC")
    >>> latitude, longitude = 55.68, 12.57
    >>>
    >>> # Calculate solar position and extraterrestrial irradiance using pvlib
    >>> solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    >>> solar_zenith = solpos["apparent_zenith"]
    >>> dni_extra = pvlib.irradiance.get_extra_radiation(times)
    >>>
    >>> # Create synthetic GHI: sine wave clipped to daytime
    >>> rng = np.random.default_rng(seed=0)
    >>> cos_sza = np.cos(np.deg2rad(solar_zenith))
    >>> ghi = np.clip(900 * cos_sza + rng.standard_normal(8760) * 20, 0, None)
    >>>
    >>> # Run PPL and ERL tests
    >>> ppl_flag = bsrn_limits_flag(ghi, solar_zenith, dni_extra, limits="ppl-ghi")
    >>> erl_flag = bsrn_limits_flag(ghi, solar_zenith, dni_extra, limits="erl-ghi")

    Use custom coefficients:

    >>> flag = bsrn_limits_flag(ghi, solar_zenith, dni_extra, limits=(1.2, 1.2, 50, -4))

    References
    ----------
    .. [1] C. N. Long and Y. Shi, "An Automated Quality Assessment and Control
       Algorithm for Surface Radiation Measurements," *The Open Atmospheric
       Science Journal*, vol. 2, no. 1, pp. 23–37, Apr. 2008.
       :doi:`10.2174/1874282300802010023`
    .. [2] C. N. Long and Y. Shi, "An Automated Quality Assessment and Control
       Algorithm for Surface Radiation Measurements," BSRN, 2002. [Online].
       Available: `BSRN recommended QC tests v2
       <https://bsrn.awi.de/fileadmin/user_upload/bsrn.awi.de/Publications/BSRN_recommended_QC_tests_V2.pdf>`_
    """
    lower, upper = bsrn_limits(solar_zenith, dni_extra, limits)
    if check == 'upper':
        flag = irradiance > upper
    elif check == 'lower':
        flag = irradiance < lower
    elif check == 'both':
        flag = (irradiance < lower) | (irradiance > upper)
    else:
        raise ValueError(f"check must be 'both', 'upper', or 'lower', got '{check}'.")
    if nan_flag:
        flag = flag | np.isnan(irradiance)
    return flag
