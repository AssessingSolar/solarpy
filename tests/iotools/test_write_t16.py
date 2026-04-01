import pathlib

import numpy as np
import pandas as pd
import pytest

import solarpy
from solarpy.iotools.write_t16 import write_t16

DATA_FILE = pathlib.Path(__file__).parents[2] / "data" / "LYN_2023.csv"

LINES = [
    "GHI is the global horizontal irradiance in W/m2",
    "DNI is the direct normal irradiance in W/m2",
    "DIF is the diffuse horizontal irradiance in W/m2",
]

META = {
    "stationcode": "TST",
    "latitude": 55.0,
    "longitude": 12.0,
    "altitude": 40.0,
}

LINES_DATA_FILE = [
    # "stationcode LYN",
    # "latitude deg N 55.79065",
    # "longitude deg E 12.52509",
    # "altitude in m amsl 40",
    # "",
    # "time stamp (Year Month Day Hour Minute) is in UTC and refers to the end of the period",
    # "missing data points in GHI DNI and DIF are noted as empty cells",
    "GHI is the global horizontal irradiance in W/m2",
    "DNI is the direct normal irradiance in W/m2",
    "DIF is the diffuse horizontal irradiance in W/m2",
    "GHIcalc is the calculated GHI from DNI and DIF",
    "Elev is the solar elevation in deg",
    "Azim is the solar azimuth angle in deg N",
    "Kc is the clearsky index calculated with CAMS mcclear with GHIcalc/GHI McClear",
    "usable is the validity of a data point with 1 being valid and 0 being not usable",
    "flag values from various tests: 1 means the data failed the test. 0 means the test was passed. -999 means the test domain was not met",
]


@pytest.fixture
def example_data():
    idx = pd.date_range("2023-01-01", periods=5, freq="1min", tz="UTC")
    return pd.DataFrame({"GHI": [0.0, 10.0, np.nan, 30.0, 40.0]}, index=idx)


# identical output to DATA_FILE


def test_output_identical_to_data_file(tmp_path):
    data, meta = solarpy.iotools.read_t16(DATA_FILE)
    out_path = tmp_path / "out.csv"
    write_t16(out_path, data, meta, LINES_DATA_FILE)
    assert out_path.read_text() == DATA_FILE.read_text(encoding="utf-8")


# round-trip on real data file


def roundtrip(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("test_roundtrip")
    data, meta = solarpy.iotools.read_t16(DATA_FILE)
    out_path = tmp_path / "roundtrip.csv"
    write_t16(out_path, data, meta, LINES_DATA_FILE)
    data_rt, meta_rt = solarpy.iotools.read_t16(out_path)

    assert data_rt.index.equals(data.index)

    pd.testing.assert_frame_equal(
        data_rt[["GHI", "DNI", "DIF"]],
        data[["GHI", "DNI", "DIF"]],
    )

    assert meta_rt["latitude deg N"] == meta["latitude deg N"]
    assert meta_rt["longitude deg E"] == meta["longitude deg E"]
    assert meta_rt["altitude in m amsl"] == meta["altitude in m amsl"]
    assert meta_rt["stationcode"] == meta["stationcode"]

    out_path


# data file header lines


def test_data_file_header_lines():
    text = DATA_FILE.read_text(encoding="utf-8")
    header_lines = [
        line.lstrip("# ") for line in text.splitlines() if line.startswith("#")
    ]
    for line in LINES_DATA_FILE:
        assert line in header_lines


# header content


def test_header_lines_written(example_data, tmp_path):
    out_path = tmp_path / "out.csv"
    write_t16(out_path, example_data, META, LINES)
    text = out_path.read_text()
    for line in LINES:
        assert "# " + line in text


def test_header_meta_written(example_data, tmp_path):
    out_path = tmp_path / "out.csv"
    write_t16(out_path, example_data, META, LINES)
    text = out_path.read_text()
    assert "# latitude deg N 55.0" in text
    assert "# longitude deg E 12.0" in text
    assert "# altitude in m amsl 40.0" in text
    assert "# stationcode TST" in text


# missing values


def test_missing_default_empty_cell(example_data, tmp_path):
    out_path = tmp_path / "out.csv"
    write_t16(out_path, example_data, META, LINES)
    text = out_path.read_text()
    assert ",\n" in text  # NaN row ends with trailing empty field


def test_missing_custom_value(example_data, tmp_path):
    out_path = tmp_path / "out.csv"
    write_t16(out_path, example_data, META, LINES, missing="-999")
    text = out_path.read_text()
    assert "-999" in text
    assert ",," not in text


# datetime columns


def test_no_duplicate_datetime_columns(tmp_path):
    # data already contains Year/Month/Day/Hour/Minute (as from read_t16)
    data, meta = solarpy.iotools.read_t16(DATA_FILE)
    out_path = tmp_path / "out.csv"
    write_t16(out_path, data, meta, LINES)
    data_rt, _ = solarpy.iotools.read_t16(out_path)
    datetime_cols = ["Year", "Month", "Day", "Hour", "Minute"]
    assert data_rt.columns.tolist().count("Year") == 1
    for col in datetime_cols:
        assert data_rt.columns.tolist().count(col) == 1


def test_datetime_columns_are_utc(example_data, tmp_path):
    # write data with a non-UTC timezone, check output timestamps are UTC
    data_cet = example_data.copy()
    data_cet.index = data_cet.index.tz_convert("Europe/Paris")
    out_path = tmp_path / "out.csv"
    write_t16(out_path, data_cet, META, LINES)
    data_rt, _ = solarpy.iotools.read_t16(out_path)
    assert data_rt.index.equals(example_data.index)


# meta key forms


def test_meta_long_key_form(example_data, tmp_path):
    meta_long = {
        "stationcode": "TST",
        "latitude deg N": 55.0,
        "longitude deg E": 12.0,
        "altitude in m amsl": 40.0,
    }
    out_path = tmp_path / "out.csv"
    write_t16(out_path, example_data, meta_long, LINES)
    _, meta_rt = solarpy.iotools.read_t16(out_path)
    assert meta_rt["latitude deg N"] == pytest.approx(55.0)
