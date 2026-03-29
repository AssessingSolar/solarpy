import numpy as np
import pandas as pd
import pytest

from solarpy.processing.resample import resample_to_freq

TIMETSTAMPS_1MIN = pd.date_range("2023-01-01 06:00", "2023-01-01 08:00", freq="1min")


@pytest.fixture
def example_series():
    idx = pd.DatetimeIndex(TIMETSTAMPS_1MIN)
    return pd.Series(np.ones(len(idx)), index=idx)


@pytest.fixture
def example_dataframe():
    idx = pd.DatetimeIndex(TIMETSTAMPS_1MIN)
    return pd.DataFrame({"a": np.ones(len(idx)), "b": np.zeros(len(idx))}, index=idx)


def test_output_index_is_regular(example_series):
    result = resample_to_freq(example_series, "1min", full_days=False, verbose=False)
    assert result.index.equals(TIMETSTAMPS_1MIN)


def test_full_days_true_start_end(example_series):
    result = resample_to_freq(example_series, "1min", full_days=True, verbose=False)
    assert result.index.min() == pd.Timestamp("2023-01-01 00:00:00")
    assert result.index.max() == pd.Timestamp("2023-01-01 23:59:00")


def test_full_days_false_preserves_original_bounds(example_series):
    result = resample_to_freq(example_series, "1min", full_days=False, verbose=False)
    assert result.index.min() == TIMETSTAMPS_1MIN.min()
    assert result.index.max() == TIMETSTAMPS_1MIN.max()


# missing and discarded values


def test_missing_timesteps_filled_with_nan(example_series):
    # drop one timestamp from the middle
    data = example_series.drop(example_series.index[5])
    result = resample_to_freq(data, "1min", full_days=False, verbose=False)
    assert result.isna().any()


def test_original_values_preserved_at_aligned_timestamps():
    data = pd.Series(
        np.arange(len(TIMETSTAMPS_1MIN), dtype=float), index=TIMETSTAMPS_1MIN
    )
    result = resample_to_freq(data, "1min", full_days=False, verbose=False)
    for ts, val in zip(TIMETSTAMPS_1MIN, data.values):
        assert result[ts] == val


def test_misaligned_timesteps_are_discarded():
    # mix aligned and off-grid timestamps; the off-grid one should be dropped
    off_grid = pd.Timestamp("2023-01-01 06:00:30")  # not on any 1-min mark
    idx = pd.DatetimeIndex(sorted(list(TIMETSTAMPS_1MIN) + [off_grid]))
    data = pd.Series(np.ones(len(idx)), index=idx)
    result = resample_to_freq(data, "1min", full_days=False, verbose=False)
    assert off_grid not in result.index


# input types


def test_accepts_dataframe(example_dataframe):
    result = resample_to_freq(example_dataframe, "1min", full_days=False, verbose=False)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["a", "b"]


def test_accepts_series(example_series):
    result = resample_to_freq(example_series, "1min", full_days=False, verbose=False)
    assert isinstance(result, pd.Series)


# verbose output


def test_verbose_prints_output(example_series, capsys):
    resample_to_freq(example_series, "1min", full_days=True, verbose=True)
    captured = capsys.readouterr()
    assert "Resampled:" in captured.out


def test_verbose_false_prints_nothing(example_series, capsys):
    resample_to_freq(example_series, "1min", full_days=True, verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ""


# frequency aliases


def test_hourly_frequency():
    idx = pd.date_range("2023-06-01 00:00", "2023-06-01 23:00", freq="1h")
    data = pd.Series(np.ones(len(idx)), index=idx)
    result = resample_to_freq(data, "1h", full_days=False, verbose=False)
    assert len(result) == 24


def test_one_minute_frequency():
    idx = pd.date_range("2023-01-01 10:00", "2023-01-01 10:05", freq="1min")
    data = pd.Series(np.ones(len(idx)), index=idx)
    result = resample_to_freq(data, "1min", full_days=False, verbose=False)
    assert len(result) == 6
