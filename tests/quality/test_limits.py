import numpy as np
import pandas as pd
import pytest

from solarpy.quality.limits import bsrn_limit, bsrn_limit_flag


SZA = 45.0          # solar zenith angle [degrees]
DNI_EXTRA = 1361.0  # extraterrestrial normal irradiance [W/m²]
COS_SZA = np.cos(np.deg2rad(SZA))


# bsrn_limit — float inputs

def test_bsrn_limit_returns_tuple_of_two():
    lower, upper = bsrn_limit(SZA, DNI_EXTRA, "ppl-ghi")
    assert isinstance(lower, float)
    assert isinstance(upper, float)


def test_bsrn_limit_ppl_ghi():
    lower, upper = bsrn_limit(SZA, DNI_EXTRA, "ppl-ghi")
    expected = 1.50 * DNI_EXTRA * COS_SZA ** 1.2 + 100
    assert upper == pytest.approx(expected)
    assert lower == -4


def test_bsrn_limit_erl_ghi():
    _, upper = bsrn_limit(SZA, DNI_EXTRA, "erl-ghi")
    expected = 1.20 * DNI_EXTRA * COS_SZA ** 1.2 + 50
    assert upper == pytest.approx(expected)


def test_bsrn_limit_ppl_dni_upper():
    _, upper = bsrn_limit(SZA, DNI_EXTRA, "ppl-dni")
    expected = 1.00 * DNI_EXTRA * COS_SZA ** 1.0 + 0
    assert upper == pytest.approx(expected)


def test_bsrn_limit_ppl_dhi_upper():
    _, upper = bsrn_limit(SZA, DNI_EXTRA, "ppl-dhi")
    expected = 0.95 * DNI_EXTRA * COS_SZA ** 1.2 + 50
    assert upper == expected


def test_bsrn_limit_zenith_90_upper_is_c():
    _, upper = bsrn_limit(90.0, DNI_EXTRA, "ppl-ghi")
    assert upper == 100.0


def test_bsrn_limit_custom_tuple():
    lower, upper = bsrn_limit(SZA, DNI_EXTRA, (1.0, 1.0, 99, -99))
    assert lower == -99
    assert upper == pytest.approx(DNI_EXTRA * COS_SZA + 99)


def test_bsrn_limit_invalid_string_raises():
    with pytest.raises(ValueError, match="Unknown limit"):
        bsrn_limit(SZA, DNI_EXTRA, "invalid")


def test_bsrn_limit_invalid_tuple_length_raises():
    with pytest.raises(ValueError, match="4 elements"):
        bsrn_limit(SZA, DNI_EXTRA, (1.0, 1.0, 50))


def test_bsrn_limit_invalid_type_raises():
    with pytest.raises(ValueError):
        bsrn_limit(SZA, DNI_EXTRA, 42)


# bsrn_limit — array inputs

def test_bsrn_limit_numpy_array():
    sza = np.array([0.0, 45.0, 90.0])
    _, upper = bsrn_limit(sza, DNI_EXTRA, "ppl-ghi")
    assert upper.shape == (3,)


def test_bsrn_limit_pandas_series():
    sza = pd.Series([0.0, 45.0, 90.0])
    _, upper = bsrn_limit(sza, DNI_EXTRA, "ppl-ghi")
    assert isinstance(upper, pd.Series)
    assert len(upper) == 3


# bsrn_limit_flag — float inputs

def test_bsrn_limit_flag_value_within_bounds_not_flagged():
    lower, upper = bsrn_limit(SZA, DNI_EXTRA, "ppl-ghi")
    mid = (lower + upper) / 2
    assert bsrn_limit_flag(mid, SZA, DNI_EXTRA, "ppl-ghi") == False  # noqa: E712


def test_bsrn_limit_flag_value_above_upper_flagged():
    _, upper = bsrn_limit(SZA, DNI_EXTRA, "ppl-ghi")
    assert bsrn_limit_flag(upper + 1, SZA, DNI_EXTRA, "ppl-ghi") == True  # noqa: E712


def test_bsrn_limit_flag_value_below_lower_flagged():
    lower, _ = bsrn_limit(SZA, DNI_EXTRA, "ppl-ghi")
    assert bsrn_limit_flag(lower - 1, SZA, DNI_EXTRA, "ppl-ghi") == True  # noqa: E712


def test_bsrn_limit_flag_check_upper_only():
    lower, _ = bsrn_limit(SZA, DNI_EXTRA, "ppl-ghi")
    # value below lower but only checking upper — should not be flagged
    result = bsrn_limit_flag(lower - 1, SZA, DNI_EXTRA, "ppl-ghi", check='upper')
    assert result == False  # noqa: E712


def test_bsrn_limit_flag_check_lower_only():
    _, upper = bsrn_limit(SZA, DNI_EXTRA, "ppl-ghi")
    # value above upper but only checking lower — should not be flagged
    result = bsrn_limit_flag(upper + 1, SZA, DNI_EXTRA, "ppl-ghi", check='lower')
    assert result == False  # noqa: E712


def test_bsrn_limit_flag_nan_flagged_by_default():
    assert bsrn_limit_flag(np.nan, SZA, DNI_EXTRA, "ppl-ghi") == True  # noqa: E712


def test_bsrn_limit_flag_nan_not_flagged_when_nan_flag_false():
    result = bsrn_limit_flag(np.nan, SZA, DNI_EXTRA, "ppl-ghi", nan_flag=False)
    assert result == False  # noqa: E712


def test_bsrn_limit_flag_invalid_check_raises():
    with pytest.raises(ValueError, match="check must be"):
        bsrn_limit_flag(500.0, SZA, DNI_EXTRA, "ppl-ghi", check='invalid')


# bsrn_limit_flag — array inputs

def test_bsrn_limit_flag_numpy_array():
    ghi = np.array([-10.0, 500.0, 9999.0])
    flag = bsrn_limit_flag(ghi, SZA, DNI_EXTRA, "ppl-ghi")
    assert flag[0] == True   # below lower  # noqa: E712
    assert flag[1] == False  # within bounds  # noqa: E712
    assert flag[2] == True   # above upper  # noqa: E712


def test_bsrn_limit_flag_pandas_series():
    ghi = pd.Series([-10.0, 500.0, 9999.0])
    flag = bsrn_limit_flag(ghi, SZA, DNI_EXTRA, "ppl-ghi")
    assert isinstance(flag, pd.Series)


def test_bsrn_limit_flag_nan_in_array_flagged():
    ghi = np.array([500.0, np.nan])
    flag = bsrn_limit_flag(ghi, SZA, DNI_EXTRA, "ppl-ghi")
    assert flag[0] == False  # noqa: E712
    assert flag[1] == True   # noqa: E712
