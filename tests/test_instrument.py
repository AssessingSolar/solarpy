import solarpy
import pandas as pd
import datetime
import pytest


def test_shadowband_correction():
    # Northern hemisphere, spring (near-zero declination)
    assert solarpy.instrument.shadowband_correction_factor("2026-03-17", latitude=55.29007) == 1.0668908588370676
    # Northern hemisphere, spring - pd.Timestamp
    assert solarpy.instrument.shadowband_correction_factor(pd.Timestamp("2026-03-17"), latitude=55.29007) == 1.0668908588370676
    # pd.Timestamp with time component - sub-daily info must be ignored
    assert solarpy.instrument.shadowband_correction_factor(pd.Timestamp("2026-03-17 14:30:01"), latitude=55.29007) == pytest.approx(1.0668908588370676)
    # datetime.date scalar
    assert solarpy.instrument.shadowband_correction_factor(datetime.date(2026, 3, 17), latitude=55.29007) == pytest.approx(1.0668908588370676)
    # datetime.datetime scalar
    assert solarpy.instrument.shadowband_correction_factor(datetime.datetime(2026, 3, 17, 14, 30), latitude=55.29007) == pytest.approx(1.0668908588370676)

    # Northern hemisphere, summer
    assert solarpy.instrument.shadowband_correction_factor("2026-06-21", latitude=55.29007) == pytest.approx(1.1408347415382885)
    # Northern hemisphere, autumn
    assert solarpy.instrument.shadowband_correction_factor("2026-10-21", latitude=55.29007) == pytest.approx(1.0418060380205798)
    # Northern hemisphere, winter
    assert solarpy.instrument.shadowband_correction_factor("2026-12-21", latitude=55.29007) == pytest.approx(1.0126112472695903)

    # Southern hemisphere
    assert solarpy.instrument.shadowband_correction_factor("2026-10-21", latitude=-33.87) == pytest.approx(1.1282175055807726)
    
def test_shadowband_correction_series():
    # Single value series
    index = pd.to_datetime(["2026-03-17"])
    pd.testing.assert_series_equal(
        solarpy.instrument.shadowband_correction_factor(index, latitude=55.29007),
        pd.Series(1.0668908588370676, index=index))
    
    # Multiple value series
    index = pd.DatetimeIndex(["2026-03-17", "2026-10-21"])
    pd.testing.assert_series_equal(
    solarpy.instrument.shadowband_correction_factor(index, latitude=55.29007),
    pd.Series([1.0668908588370676, 1.0418060380205798], index=index))
    
def test_shadowband_correction_always_above_one():
    # Test if shadowband correction factor C > 1 
    dates = pd.DatetimeIndex(["2026-03-20", "2026-06-21", "2026-09-22", "2026-12-21"])
    result = solarpy.instrument.shadowband_correction_factor(dates, latitude=55.29007)
    assert (result > 1.0).all()
    
    
def test_shadowband_correction_v_zero():
    # With no ring there is nothing to correct for
    assert solarpy.instrument.shadowband_correction_factor("2026-06-21", latitude=45.0, V=0.0) == pytest.approx(1.0)

def test_shadowband_correction_polar():
    # Test that the clip works and it follows the table provided by Kipp & Zonen
    # At the pole in winter the sun never rises — ring blocks nothing, C = 1
    assert solarpy.instrument.shadowband_correction_factor("2026-01-01", latitude=90.0) == pytest.approx(1.0)
    # At the pole in summer the sun skims the horizon — C rises above 1
    assert solarpy.instrument.shadowband_correction_factor("2026-06-21", latitude=90.0) > 1.0
    