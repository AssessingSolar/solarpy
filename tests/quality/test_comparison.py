import numpy as np
import pandas as pd
import pytest
import warnings
from solarpy.quality.comparison import diffuse_fraction_flag


GHI = 100.0
SZA_LOW = 45.0  # < 75°
SZA_HIGH = 80.0  # 75° ≤ SZA < 93°
SZA_NIGHT = 95.0  # ≥ 93°


# --- low-zenith domain (SZA < 75) ---


def test_low_zenith_within_limit_not_flagged():
    dhi = GHI * 1.04  # K = 1.04 < 1.05
    assert diffuse_fraction_flag(GHI, dhi, SZA_LOW) == False  # noqa: E712


def test_low_zenith_at_limit_flagged():
    dhi = GHI * 1.05  # K = 1.05, >= threshold
    assert diffuse_fraction_flag(GHI, dhi, SZA_LOW) == True  # noqa: E712


def test_low_zenith_above_limit_flagged():
    dhi = GHI * 1.06  # K = 1.06 > 1.05
    assert diffuse_fraction_flag(GHI, dhi, SZA_LOW) == True  # noqa: E712


# --- high-zenith domain (75 ≤ SZA < 93) ---


def test_high_zenith_within_limit_not_flagged():
    dhi = GHI * 1.09  # K = 1.09 < 1.10
    assert diffuse_fraction_flag(GHI, dhi, SZA_HIGH) == False  # noqa: E712


def test_high_zenith_at_limit_flagged():
    dhi = GHI * 1.10  # K = 1.10, >= threshold
    assert diffuse_fraction_flag(GHI, dhi, SZA_HIGH) == True  # noqa: E712


def test_high_zenith_above_limit_flagged():
    dhi = GHI * 1.11  # K = 1.11 > 1.10
    assert diffuse_fraction_flag(GHI, dhi, SZA_HIGH) == True  # noqa: E712


# --- outside domain ---


def test_ghi_below_threshold_not_flagged():
    assert diffuse_fraction_flag(40.0, 200.0, SZA_LOW) == False  # noqa: E712


def test_nighttime_not_flagged():
    dhi = GHI * 1.5  # K far above limit, but SZA ≥ 93
    assert diffuse_fraction_flag(GHI, dhi, SZA_NIGHT) == False  # noqa: E712


def test_ghi_zero_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flag = diffuse_fraction_flag(np.array(0.0), np.array(50.0), SZA_LOW)
    assert flag == False  # noqa: E712


# --- check parameter ---


def test_check_low_zenith_ignores_high_zenith_violation():
    dhi = GHI * 1.11  # K > 1.10, violation in high-zenith
    flag = diffuse_fraction_flag(GHI, dhi, SZA_HIGH, check="low-zenith")
    assert flag == False  # noqa: E712


def test_check_high_zenith_ignores_low_zenith_violation():
    dhi = GHI * 1.06  # K >= 1.05, violation in low-zenith
    flag = diffuse_fraction_flag(GHI, dhi, SZA_LOW, check="high-zenith")
    assert flag == False  # noqa: E712


def test_check_invalid_raises():
    with pytest.raises(ValueError, match="check must be"):
        diffuse_fraction_flag(GHI, GHI, SZA_LOW, check="invalid")


# --- outside_domain_flag ---


def test_outside_domain_flag_true_flags_nighttime():
    flag = diffuse_fraction_flag(GHI, GHI, SZA_NIGHT, outside_domain_flag=True)
    assert flag == True  # noqa: E712


def test_outside_domain_flag_true_flags_low_ghi():
    flag = diffuse_fraction_flag(40, 40, SZA_LOW, outside_domain_flag=True)
    assert flag == True  # noqa: E712


def test_outside_domain_flag_false_does_not_flag_nighttime():
    flag = diffuse_fraction_flag(GHI, GHI, SZA_NIGHT, outside_domain_flag=False)
    assert flag == False  # noqa: E712


# --- nan_flag ---


def test_nan_ghi_not_flagged_by_default():
    assert diffuse_fraction_flag(float("nan"), GHI, SZA_LOW) == False  # noqa: E712


def test_nan_dhi_not_flagged_by_default():
    assert diffuse_fraction_flag(GHI, float("nan"), SZA_LOW) == False  # noqa: E712


def test_nan_ghi_flagged_when_nan_flag_true():
    flag = diffuse_fraction_flag(np.nan, GHI, SZA_LOW, nan_flag=True)
    assert flag == True  # noqa: E712


def test_nan_dhi_flagged_when_nan_flag_true():
    flag = diffuse_fraction_flag(GHI, float("nan"), SZA_LOW, nan_flag=True)
    assert flag == True  # noqa: E712


# --- array inputs ---


def test_numpy_array():
    ghi = np.array([100.0, 100.0, 40.0, 55.0])
    dhi = np.array([104.0, 106.0, 100.0, 100.0])
    sza = np.array([SZA_LOW, SZA_LOW, SZA_LOW, SZA_LOW])
    flag = diffuse_fraction_flag(ghi, dhi, sza)
    assert flag[0] == False  # noqa: E712
    assert flag[1] == True  # noqa: E712
    assert flag[2] == False  # noqa: E712
    assert flag[3] == True  # noqa: E712


def test_pandas_series():
    ghi = pd.Series([100.0, 100.0])
    dhi = pd.Series([104.0, 106.0])
    sza = pd.Series([SZA_LOW, SZA_LOW])
    flag = diffuse_fraction_flag(ghi, dhi, sza)
    assert isinstance(flag, pd.Series)
