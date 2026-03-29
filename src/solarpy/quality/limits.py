"""Functions for BSRN quality control irradiance limit tests."""

import numpy as np


_BSRN_LIMITS = {
    "ghi-ppl": {"scale": 1.50, "exponent": 1.2, "offset": 100., "lower": -4.},
    "ghi-erl": {"scale": 1.20, "exponent": 1.2, "offset":  50., "lower": -2.},
    "dni-ppl": {"scale": 1.00, "exponent": 0.0, "offset":   0., "lower": -4.},
    "dni-erl": {"scale": 0.95, "exponent": 0.2, "offset":  10., "lower": -2.},
    "dhi-ppl": {"scale": 0.95, "exponent": 1.2, "offset":  50., "lower": -4.},
    "dhi-erl": {"scale": 0.75, "exponent": 1.2, "offset":  30., "lower": -2.},
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
    limits : str or dict
        Either a named limit string or a dict with keys ``scale``, ``exponent``,
        ``offset``, and ``lower``.

        Named limits (Long & Shi, 2008):

        - ``"ghi-ppl"`` — Physically possible imits for GHI
        - ``"ghi-erl"`` — Extremely rare limits for GHI
        - ``"dni-ppl"`` — Physically possible limits for DNI
        - ``"dni-erl"`` — Extremely rare limits for DNI
        - ``"dhi-ppl"`` — Physically possible limits for DHI
        - ``"dhi-erl"`` — Extremely rare limits for DHI

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
        scale = _BSRN_LIMITS[limits]["scale"]
        exponent = _BSRN_LIMITS[limits]["exponent"]
        offset = _BSRN_LIMITS[limits]["offset"]
        lower = _BSRN_LIMITS[limits]["lower"]
    elif isinstance(limits, dict):
        missing = {"scale", "exponent", "offset", "lower"} - limits.keys()
        if missing:
            raise ValueError(
                f"limit dict is missing keys: {sorted(missing)}."
            )
        scale = limits["scale"]
        exponent = limits["exponent"]
        offset = limits["offset"]
        lower = limits["lower"]
    else:
        raise ValueError("limits must be a string or a dict with keys scale, exponent, offset, lower.")

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
    limits : str or dict
        Either a named limit string or a dict with keys ``scale``, ``exponent``,
        ``offset``, and ``lower``.

        Named limit (Long & Shi, 2008) [1]_, [2]_:

        - ``"ghi-ppl"`` — Physically Possible Limit for GHI
        - ``"ghi-erl"`` — Extremely Rare Limit for GHI
        - ``"dni-ppl"`` — Physically Possible Limit for DNI
        - ``"dni-erl"`` — Extremely Rare Limit for DNI
        - ``"dhi-ppl"`` — Physically Possible Limit for DHI
        - ``"dhi-erl"`` — Extremely Rare Limit for DHI

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
    >>> ppl_flag = bsrn_limits_flag(ghi, solar_zenith, dni_extra, limits="ghi-ppl")
    >>> erl_flag = bsrn_limits_flag(ghi, solar_zenith, dni_extra, limits="ghi-erl")

    Use custom coefficients:

    >>> flag = bsrn_limits_flag(ghi, solar_zenith, dni_extra,
    ...                         limits={"scale": 1.2, "exponent": 1.2, "offset": 50, "lower": -4})

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
