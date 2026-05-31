"""Integration test for get_dmi_climate_station_data against the live DMI API."""

from __future__ import annotations

import pandas as pd
import pytest
import requests
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
    data, meta = solarpy.iotools.get_dmi_climate_station_data(
        station="06188",
        start=pd.Timestamp("2023-06-01"),
        end=pd.Timestamp("2023-06-02"),
        parameters=["ghi"],
        time_resolution="1h",
        timeout=30,
    )
    return data, meta


def test_pagination(monkeypatch, EXPECTED_GHI):
    # use monkeypatch to set LIMIT=10, forcing the 25 records to be fetched
    # across 3 pages
    monkeypatch.setattr(solarpy.iotools.dmi, "LIMIT", 10)
    data, _ = solarpy.iotools.get_dmi_climate_station_data(
        station="06188",
        start=pd.Timestamp("2023-06-01"),
        end=pd.Timestamp("2023-06-02"),
        parameters="ghi",
        time_resolution="1h",
        timeout=30,
    )
    pd.testing.assert_series_equal(data["ghi"], EXPECTED_GHI, check_names=False)


@pytest.fixture(scope="module")
def data(result):
    return result[0]


@pytest.fixture(scope="module")
def data2(result):
    return result[0]


@pytest.fixture(scope="module")
def meta(result):
    return result[1]


def test_ghi_values(data, EXPECTED_GHI):
    pd.testing.assert_series_equal(data["ghi"], EXPECTED_GHI, check_names=False)


def test_meta_station_id(meta):
    assert meta["stationId"] == "06188"


def test_identical_requests(data, data2):
    pd.testing.assert_frame_equal(data, data2)


def test_dmi_nonexisting_station():
    with pytest.raises(ValueError, match="not_a_station"):
        solarpy.iotools.get_dmi_climate_station_data(
            station="not_a_station",
            start=pd.Timestamp("2023-06-01"),
            end=pd.Timestamp("2023-06-02"),
            parameters="ghi",
            time_resolution="1h",
            timeout=30,
        )


def test_dmi_incorrect_time_resolution():
    with pytest.raises(requests.HTTPError, match="Invalid time resolution"):
        solarpy.iotools.get_dmi_climate_station_data(
            station="06188",
            start=pd.Timestamp("2023-06-01"),
            end=pd.Timestamp("2023-06-02"),
            parameters="ghi",
            time_resolution="not_a_time_resolution",
        )


# --- get_dmi_station_meta tests ---


@pytest.fixture(scope="module")
def station_meta_recent_entry():
    return solarpy.iotools.get_dmi_station_meta("06188", timeout=30)


@pytest.fixture(scope="module")
def station_meta_first_entry():
    return solarpy.iotools.get_dmi_station_meta("06188", entry_no=0, timeout=30)


def test_station_meta_content(station_meta_recent_entry):
    meta = station_meta_recent_entry
    assert meta["latitude"] == 55.8764
    assert meta["longitude"] == 12.4121
    assert meta["country"] == "Denmark"
    assert meta["status"] == "Active"
    assert meta["validFrom"] == "2019-02-01T18:43:18Z"
    assert meta["validTo"] is None


def test_station_meta_first_entry_date(station_meta_first_entry):
    meta = station_meta_first_entry
    assert meta["latitude"] == 55.8764
    assert meta["longitude"] == 12.4121
    assert meta["stationId"] == "06188"
    assert meta["country"] == "Denmark"
    assert meta["status"] == "Active"
    assert meta["validFrom"] == "2003-08-08T00:00:00Z"
    assert meta["validTo"] == "2019-01-15T13:34:47Z"
