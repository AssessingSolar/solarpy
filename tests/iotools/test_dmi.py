"""Integration test for get_dmi_climate_data against the live DMI API."""

from __future__ import annotations

import pandas as pd
import pytest

import solarpy


@pytest.fixture(scope="module")
def EXPECTED_GHI():
    ghi = pd.Series(
        [
            0.0,
            0.0,
            2.0,
            17.0,
            69.0,
            168.0,
            300.0,
            347.0,
            586.0,
            804.0,
            856.0,
            858.0,
            822.0,
            741.0,
            620.0,
            496.0,
            349.0,
            213.0,
            72.0,
            6.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        index=pd.date_range("2023-06-01", "2023-06-02", freq="h", tz="UTC"),
    )
    ghi.index.freq = None
    return ghi


@pytest.fixture(scope="module")
def result():
    data, meta = solarpy.iotools.get_dmi_climate_data(
        station="06188",
        start=pd.Timestamp("2023-06-01"),
        end=pd.Timestamp("2023-06-02"),
        parameters=["ghi"],
        time_resolution="1h",
        timeout=30,
    )
    return data, meta


@pytest.fixture(scope="module")
def data(result):
    return result[0]


@pytest.fixture(scope="module")
def meta(result):
    return result[1]


def test_ghi_values(data, EXPECTED_GHI):
    pd.testing.assert_series_equal(data["ghi"], EXPECTED_GHI, check_names=False)


def test_meta_keys(meta):
    assert set(meta.keys()) == {
        "station_id",
        "name",
        "latitude",
        "longitude",
        "altitude",
        "country",
    }


def test_meta_station_id(meta):
    assert meta["station_id"] == "06188"
