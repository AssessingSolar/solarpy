"""
shadowband_correction.py
------------------------
Shadowband diffuse-irradiance correction factor for the Kipp & Zonen CM 121
shadow-ring pyranometer accessory.

Reference
---------
Kipp & Zonen *CM 121 B/C Shadow Ring Instruction Manual*, Appendix §6.1.
https://www.kippzonen.com/Download/46/CM121-B-C-Shadow-Ring-Manual
"""

import numpy as np
import pvlib as pv
import pandas as pd
import datetime


def shadowband_correction_factor(
    input_date,
    latitude: float,
    V: float = 0.185,
):
    """Calculate the daily shadowband correction factor for a Kipp & Zonen CM 121.

    A shadow ring blocks a strip of the sky dome to prevent direct beam
    radiation from reaching the pyranometer, so that only diffuse radiation
    is measured.  Because the ring also blocks some of the diffuse sky, a
    correction factor C > 1 must be applied to recover the true diffuse
    irradiance::

        G_d_true = C * G_d_measured

    The factor depends on the ring geometry (angular half-width *V*), the
    site latitude, and the solar declination on the measurement day.  It is
    derived analytically by integrating the fraction of the isotropic sky
    hemisphere obscured by the ring over the daily arc of the sun.

    Parameters
    ----------
    input_date : scalar or array-like
        Date(s) for which to compute the factor.  Accepts anything that
        ``pandas.to_datetime`` understands: a ``pd.Timestamp``, a
        ``datetime.date`` / ``datetime.datetime``, an ISO-8601 string, a
        list/array of any of the above, or a ``pd.DatetimeIndex``.
        Sub-daily time information is ignored; only the calendar date matters.
    latitude : float
        Observer latitude in decimal degrees.  Positive = North, negative = South.
        Valid range: −90 to +90.
    V : float, optional
        Angular width of the shadow band in radians.  This is arc subtended by
        the ring as seen from the sensor.  Default is 0.185 rad (~10.6°),
        which matches the standard Kipp & Zonen CM 121 ring geometry.

    Returns
    -------
    float or array-like
        Dimensionless correction factor C ≥ 1.

    Notes
    -----
    The correction factor is computed as:

    .. math::

        S = \\frac{2V \\cos\\delta}{\\pi}
            \\bigl(U_0 \\sin\\phi\\sin\\delta
                  + \\sin U_0 \\cos\\phi\\cos\\delta\\bigr)

        C = \\frac{1}{1 - S}

    where

    * :math:`\\delta` – solar declination (Spencer 1971 approximation)
    * :math:`\\phi`   – site latitude
    * :math:`U_0`    – sunrise/sunset hour angle,
      :math:`\\arccos(-\\tan\\phi\\,\\tan\\delta)`, clamped to [−1, 1] to
      handle polar day/night conditions
    """
    # ------------------------------------------------------------------
    # 1. Normalise input to pandas datetime so scalar and array paths
    #    share a single code route below.
    # ------------------------------------------------------------------
    dates = pd.to_datetime(input_date)

    # ------------------------------------------------------------------
    # 2. Solar declination δ for each day of year.
    #    Spencer (1971) approximation, returned in radians by pvlib.
    # ------------------------------------------------------------------
    day_of_year = dates.dayofyear
    delta = pv.solarposition.declination_spencer71(day_of_year)  # radians

    # ------------------------------------------------------------------
    # 3. Site latitude φ in radians.
    # ------------------------------------------------------------------
    phi = np.deg2rad(latitude)

    # ------------------------------------------------------------------
    # 4. Sunrise/sunset hour angle U₀.
    #    cos U₀ = −tan φ · tan δ.
    #    Clamped to [−1, 1] so arccos is valid at the poles (midnight sun
    #    / polar night), where the tangent product can exceed ±1.
    # ------------------------------------------------------------------
    cos_U0 = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
    U0 = np.arccos(cos_U0)

    # ------------------------------------------------------------------
    # 5. Fraction S of the isotropic sky dome obscured by the ring.
    #    Derived by integrating the ring's shadow across the daily solar arc
    #    (Kipp & Zonen manual, Appendix §6.1).
    # ------------------------------------------------------------------
    S = (2 * V * np.cos(delta) / np.pi) * (
        U0 * np.sin(phi) * np.sin(delta)
        + np.sin(U0) * np.cos(phi) * np.cos(delta)
    )

    # ------------------------------------------------------------------
    # 6. Correction factor C = 1 / (1 − S).
    #    S is always < 1 for physically reasonable inputs, so C > 1.
    # ------------------------------------------------------------------
    C = 1.0 / (1.0 - S)

    # ------------------------------------------------------------------
    # 7. Return a plain float for scalar input, Series for array input.
    # ------------------------------------------------------------------
    if np.isscalar(input_date) or isinstance(input_date, (pd.Timestamp, datetime.date, datetime.datetime)):
        return float(C)
    return pd.Series(C, index=dates)
