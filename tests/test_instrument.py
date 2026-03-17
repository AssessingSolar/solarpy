import solarpy
import pandas as pd


def test_shadowband_correction():
    # Norhtern hemisphere, winter
    assert solarpy.instrument.shadowband_correction_factor("2026-03-17", latitude=55.29007) == 1.0668908588370676
    # Norhtern hemisphere, winter - pd.Timestamp
    assert solarpy.instrument.shadowband_correction_factor(pd.Timestamp("2026-03-17"), latitude=55.29007) == 1.0668908588370676
    # Southern hemisphere

    # Norhtern hemisphere, summer


def test_shadowband_correction_series():
    index = pd.to_datetime(["2026-03-17"])
    pd.testing.assert_series_equal(
        solarpy.instrument.shadowband_correction_factor(index, latitude=55.29007),
        pd.Series(1.0668908588370676, index=index))
