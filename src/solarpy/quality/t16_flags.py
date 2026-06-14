"""Combined IEA PVPS Task 16 quality control flags for irradiance data."""

import solarpy
import numpy as np
import pandas as pd


def t16_quality_flags(data, altitude=0, horizon=None):
    """Calculate the IEA PVPS Task 16 quality control flags for irradiance data.

    Apply a series of quality control tests on a time series of GHI, DHI,
    and DNI measurements, combining the BSRN limit tests
    (:func:`bsrn_limits_flag`), the BSRN comparison tests
    (:func:`closure_flag` and :func:`diffuse_fraction_flag`), several
    clearness-index based tests, and an optional horizon-shading test.

    Parameters
    ----------
    data : pandas.DataFrame
        Time series of irradiance and auxiliary data. Must contain the
        following columns:

        - ``"ghi"``, ``"dhi"``, ``"dni"`` : measured irradiance components
          [W/m²]
        - ``"dni_extra"`` : extraterrestrial irradiance normal to the sun [W/m²]
        - ``"solar_zenith"``, ``"solar_azimuth"``: solar position angles [degrees]

    altitude : float, optional
        Site altitude above sea level [m], used in the ``"flagKn"`` test.
        Default is ``0``.
    horizon : pandas.Series, optional
        Horizon elevation profile. The index must contain azimuth angles in
        degrees and the values the corresponding horizon elevation angles in
        degrees, matching the output of
        :func:`solarpy.horizon.get_horizon_mines`. If ``None`` (default), the
        ``"flagHorizon"`` column is set to ``NaN`` for all rows.

    Returns
    -------
    flags : pandas.DataFrame
        Boolean flags with the same index as *data*. ``True`` indicates the
        observation failed the corresponding test. The columns are:

        - ``"flagPPLGHI"``, ``"flagPPLDIF"``, ``"flagPPLDNI"`` :
          BSRN physically possible limit (PPL) test
        - ``"flagERLGHI"``, ``"flagERLDIF"``, ``"flagERLDNI"`` :
          BSRN extremely rare limit (ERL) test
        - ``"flag3lowSZA"``, ``"flag3highSZA"`` : BSRN three-component closure
          test, for solar zenith angle below/above 75°
        - ``"flagKlowSZA"``, ``"flagKhighSZA"`` : diffuse fraction test, for
          solar zenith angle below/above 75°
        - ``"flagKnKt"`` : flagged when the beam clearness index ``Kn``
          exceeds the global clearness index ``Kt``
        - ``"flagKn"`` : flagged when ``Kn`` exceeds an altitude-dependent
          upper bound
        - ``"flagKt"`` : flagged when the global clearness index ``Kt`` is
          greater than or equal to ``1.35``
        - ``"flagKKt"`` : flagged when the diffuse clearness index ``K`` is
          greater than or equal to ``0.96`` under high-sun, clear-sky
          conditions
        - ``"flagTracker"`` : flagged when GHI and DNI jointly indicate that
          a sun tracker is not tracking correctly
        - ``"flagHorizon"`` : flagged when the sun is below the horizon
          profile given by *horizon*, i.e. far-field shading.
          ``NaN`` for all rows if *horizon* is ``None``.

    See Also
    --------
    bsrn_limits_flag
    closure_flag
    diffuse_fraction_flag
    solarpy.horizon.get_horizon_mines
    """
    cos_sza = np.cos(np.deg2rad(data["solar_zenith"])).clip(lower=0)
    ghi_extra = data["dni_extra"] * cos_sza

    with np.errstate(divide="ignore", invalid="ignore"):
        Kt = data["ghi"] / ghi_extra
        Kn = data["dni"] / data["dni_extra"]
        K = data["dhi"] / data["ghi"]

    component_dict = {"ghi": "GHI", "dhi": "DIF", "dni": "DNI"}

    flags = pd.DataFrame(index=data.index)
    # BSRN LIMITS CHECK
    for component in ["ghi", "dhi", "dni"]:
        for limit_type in ["ppl", "erl"]:
            limit = f"{component}-{limit_type}"
            flag_name = f"flag{limit_type.upper()}{component_dict[component]}"
            flags[flag_name] = solarpy.quality.bsrn_limits_flag(
                data[component],
                solar_zenith=data["solar_zenith"],
                dni_extra=data["dni_extra"],
                limits=limit,
                check="both",
                nan_flag=False,
            )
    # BSRN CLOSURE CHECK
    for check in ["low", "high"]:
        flags[f"flag3{check}SZA"] = solarpy.quality.closure_flag(
            ghi=data["ghi"],
            dni=data["dni"],
            dhi=data["dhi"],
            solar_zenith=data["solar_zenith"],
            check=f"{check}-zenith",
            outside_domain_flag=False,
            nan_flag=False,
        )

        flags[f"flagK{check}SZA"] = solarpy.quality.diffuse_fraction_flag(
            ghi=data["ghi"],
            dhi=data["dhi"],
            solar_zenith=data["solar_zenith"],
            check=f"{check}-zenith",
            outside_domain_flag=False,
            nan_flag=False,
        )

    # K/Kn/Kt CHECK
    flags["flagKnKt"] = False
    KnKt_domain = (data["ghi"] > 50) & (Kn >= 0) & (Kt > 0)
    KnKt_condition = Kn > Kt
    flags.loc[KnKt_domain & KnKt_condition, "flagKnKt"] = True

    # Kn CHECK
    flags["flagKn"] = False
    Kn_domain = (data["ghi"] > 50) & (Kn > 0)
    Kn_condition = Kn >= ((1100 + 0.03 * altitude) / data["dni_extra"])
    flags.loc[Kn_domain & Kn_condition, "flagKn"] = True

    # Kt CHECK
    flags["flagKt"] = False
    Kt_domain = (data["ghi"] > 50) & (Kt > 0)
    Kt_condition = Kt >= 1.35
    flags.loc[Kt_domain & Kt_condition, "flagKt"] = True

    # flagKKt CHECK
    flags["flagKKt"] = False
    KKt_domain = (
        (data["ghi"] > 150) & (Kt > 0.6) & (data["solar_zenith"] < 85) & (K > 0)
    )
    KKt_condition = K >= 0.96
    flags.loc[KKt_domain & KKt_condition, "flagKKt"] = True

    # flagTracker CHECK
    flags["flagTracker"] = False
    GHI_clear = 0.8 * ghi_extra
    DNI_clear = 0.688 * data["dni_extra"]
    tracker_domain = data["solar_zenith"] < 85
    tracker_condition = (
        ((GHI_clear - data["ghi"]) / (GHI_clear + data["ghi"])) <= 0.2
    ) & (((DNI_clear - data["dni"]) / (DNI_clear + data["dni"])) >= 0.95)
    flags.loc[tracker_domain & tracker_condition, "flagTracker"] = True

    # flagHorizon CHECK
    if horizon is not None:
        horizon = horizon.sort_index()  # necessary for np.interp
        horizon_ts = pd.Series(
            np.interp(data["solar_azimuth"], horizon.index, horizon),
            index=data.index,
        )
        horizon_ts = horizon_ts.clip(lower=0)

        flags["flagHorizon"] = False
        horizon_condition = (90 - data["solar_zenith"]) < horizon_ts
        horizon_domain = (90 - data["solar_zenith"]) > 0

        flags.loc[horizon_domain & horizon_condition, "flagHorizon"] = True
    else:
        flags["flagHorizon"] = np.nan

    return flags
