import pathlib

import pandas as pd
import pytest

import solarpy

DATA_FILE = pathlib.Path(__file__).parents[2] / 'data' / 'LYN_2023.csv'


@pytest.fixture(scope='module')
def result():
    return solarpy.iotools.read_t16(DATA_FILE)


@pytest.fixture(scope='module')
def data(result):
    return result[0]


@pytest.fixture(scope='module')
def meta(result):
    return result[1]


def test_returns_tuple(result):
    assert isinstance(result, tuple) and len(result) == 2


def test_data_is_dataframe(data):
    assert isinstance(data, pd.DataFrame)


def test_meta_is_dict(meta):
    assert isinstance(meta, dict)


def test_index_is_datetimeindex(data):
    assert isinstance(data.index, pd.DatetimeIndex)


def test_index_is_utc(data):
    assert str(data.index.tz) == 'UTC'


def test_first_timestamp(data):
    assert data.index[0] == pd.Timestamp('2023-01-01 00:00', tz='UTC')


def test_last_timestamp(data):
    assert data.index[-1] == pd.Timestamp('2023-12-31 23:59', tz='UTC')


def test_row_count(data):
    assert len(data) == 365 * 24 * 60


def test_meta(meta):
    assert meta['stationcode'] == 'LYN'
    assert meta['latitude deg N'] == pytest.approx(55.79065)
    assert meta['longitude deg E'] == pytest.approx(12.52509)
    assert meta['altitude in m amsl'] == pytest.approx(40.0)
    assert meta['timezone offset from UTC in hours'] == pytest.approx(1.0)
    assert isinstance(meta['latitude deg N'], float)


def test_datetime_columns_present_by_default(data):
    for col in ['Year', 'Month', 'Day', 'Hour', 'Minute']:
        assert col in data.columns


def test_irradiance_columns_present(data):
    for col in ['GHI', 'DNI', 'DIF']:
        assert col in data.columns


def test_drop_dates():
    data, _ = solarpy.iotools.read_t16(DATA_FILE, drop_dates=True)
    for col in ['Year', 'Month', 'Day', 'Hour', 'Minute']:
        assert col not in data.columns
    assert data.index[0] == pd.Timestamp('2023-01-01 00:00', tz='UTC')


def test_map_variables():
    data, meta = solarpy.iotools.read_t16(DATA_FILE, map_variables=True)
    assert 'ghi' in data.columns
    assert 'dni' in data.columns
    assert 'dhi' in data.columns
    # test original columns are not present
    assert 'GHI' not in data.columns
    assert 'DNI' not in data.columns
    assert 'DIF' not in data.columns
    # test metadata renaming
    assert 'latitude' in meta
    assert 'longitude' in meta
    assert 'altitude' in meta
    # test original metadata entries are not present
    assert 'latitude deg N' not in meta
    assert 'longitude deg E' not in meta
    assert 'altitude in m amsl' not in meta


def test_empty_stationcode_returns_none(tmp_path):
    content = (
        "# stationcode ,\n"
        "# latitude deg N 55.0,\n"
        "# longitude deg E 12.0,\n"
        "# altitude in m amsl 40,\n"
        "# timezone offset from UTC in hours 0,\n"
        "Year,Month,Day,Hour,Minute,GHI\n"
        "2023,1,1,0,0,100.0\n"
    )
    f = tmp_path / "test.csv"
    f.write_text(content)
    _, meta = solarpy.iotools.read_t16(f)
    assert meta['stationcode'] is None
