"""Integration test for get_dmi_climate_data_station against the live DMI API."""

from __future__ import annotations

import pandas as pd
import pytest
import requests
import solarpy
from solarpy.iotools.dmi import URL_CLIMATE_DATA, URL_METOBS


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
def EXPECTED_METOBS_GHI():
    ghi = pd.Series(
        [836.0, 846.0, 850.0, 856.0, 860.0, 861.0, 862.0],
        index=pd.DatetimeIndex(
            [
                "2023-06-01T10:00:00+00:00",
                "2023-06-01T10:10:00+00:00",
                "2023-06-01T10:20:00+00:00",
                "2023-06-01T10:30:00+00:00",
                "2023-06-01T10:40:00+00:00",
                "2023-06-01T10:50:00+00:00",
                "2023-06-01T11:00:00+00:00",
            ],
            name="timestamp",
        ),
        name="ghi",
    )
    return ghi

# ---------------------------------------------------------------------------
# get_dmi_climate_data_station tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result():
    data, meta = solarpy.iotools.get_dmi_climate_data_station(
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
    data, _ = solarpy.iotools.get_dmi_climate_data_station(
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
        solarpy.iotools.get_dmi_climate_data_station(
            station="not_a_station",
            start=pd.Timestamp("2023-06-01"),
            end=pd.Timestamp("2023-06-02"),
            parameters="ghi",
            time_resolution="1h",
            timeout=30,
        )


def test_dmi_incorrect_time_resolution():
    with pytest.raises(requests.HTTPError, match="Invalid time resolution"):
        solarpy.iotools.get_dmi_climate_data_station(
            station="06188",
            start=pd.Timestamp("2023-06-01"),
            end=pd.Timestamp("2023-06-02"),
            parameters="ghi",
            time_resolution="not_a_time_resolution",
        )

# ---------------------------------------------------------------------------
# get_dmi_station_meta tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def station_meta_recent_entry():
    return solarpy.iotools.get_dmi_station_meta("06188", url=URL_CLIMATE_DATA, timeout=30)


@pytest.fixture(scope="module")
def station_meta_recent_entry_metobs():
    return solarpy.iotools.get_dmi_station_meta("06188", url=URL_METOBS, timeout=30)


@pytest.fixture(scope="module")
def station_meta_first_entry():
    return solarpy.iotools.get_dmi_station_meta("06188", entry_no=0,
                                                url=URL_CLIMATE_DATA, timeout=30)


@pytest.fixture(scope="module")
def station_meta_first_entry_metobs():
    return solarpy.iotools.get_dmi_station_meta("06188", entry_no=0, url=URL_METOBS, timeout=30)


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


# ---------------------------------------------------------------------------
# get_dmi_metobs tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def metobs_result():
    data, meta = solarpy.iotools.get_dmi_metobs(
        station="06188",
        start=pd.Timestamp("2023-06-01T10:00:00"),
        end=pd.Timestamp("2023-06-01T11:00:00"),
        parameters=["radia_glob"],
        timeout=30,
    )
    return data, meta


@pytest.fixture(scope="module")
def metobs_data(metobs_result):
    return metobs_result[0]


@pytest.fixture(scope="module")
def metobs_meta(metobs_result):
    return metobs_result[1]


def test_metobs_columns_mapped(metobs_data):
    # with map_variables=True (default), DMI names should be renamed to pvlib names
    assert "ghi" in metobs_data.columns
    assert "radia_glob" not in metobs_data.columns


def test_metobs_columns_unmapped():
    data, _ = solarpy.iotools.get_dmi_metobs(
        station="06188",
        start=pd.Timestamp("2023-06-01T10:00:00"),
        end=pd.Timestamp("2023-06-01T11:00:00"),
        parameters=["radia_glob"],
        map_variables=False,
        timeout=30,
    )
    assert "radia_glob" in data.columns
    assert "ghi" not in data.columns


def test_metobs_pagination(monkeypatch):
    monkeypatch.setattr(solarpy.iotools.dmi, "LIMIT", 10)
    data, _ = solarpy.iotools.get_dmi_metobs(
        station="06188",
        start=pd.Timestamp("2023-06-01T10:00:00"),
        end=pd.Timestamp("2023-06-01T11:00:00"),
        parameters="radia_glob",
        timeout=30,
    )
    assert not data.empty


def test_metobs_pvlib_name_input():
    # passing pvlib name 'ghi' should be accepted and resolve to 'radia_glob'
    data, _ = solarpy.iotools.get_dmi_metobs(
        station="06188",
        start=pd.Timestamp("2023-06-01T10:00:00"),
        end=pd.Timestamp("2023-06-01T11:00:00"),
        parameters="ghi",
        timeout=30,
    )
    assert "ghi" in data.columns


def test_metobs_nonexisting_station():
    with pytest.raises(ValueError, match="not_a_station"):
        solarpy.iotools.get_dmi_metobs(
            station="not_a_station",
            start=pd.Timestamp("2023-06-01T10:00:00"),
            end=pd.Timestamp("2023-06-01T11:00:00"),
            parameters="radia_glob",
            timeout=30,
        )
