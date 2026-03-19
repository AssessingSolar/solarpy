"""Tests for get_horizon_mines."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

import solarpy

def _make_response(elevations: list[float]) -> MagicMock:
    """Build a mock requests.Response with a CSV body matching the API format."""
    # 27 header rows (content doesn't matter, just need the right count)
    header = "\n".join(f"# header line {i}" for i in range(27))
    rows = "\n".join(f"{az};{el}" for az, el in enumerate(elevations))
    mock_res = MagicMock()
    mock_res.text = header + "\n" + rows
    mock_res.raise_for_status = MagicMock()
    return mock_res


_N = 360
_ELEVATIONS = [float(i % 10) for i in range(_N)]  # deterministic test data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_get():
    with patch("solarpy.horizon.horizon_mines.requests.get") as mock:
        mock.return_value = _make_response(_ELEVATIONS)
        yield mock


def test_returns_series_and_dict(mock_get):
    horizon, meta = solarpy.horizon.get_horizon_mines(48.8566, 2.3522)
    assert isinstance(horizon, pd.Series)
    assert isinstance(meta, dict)


def test_horizon_length(mock_get):
    horizon, _ = solarpy.horizon.get_horizon_mines(48.8566, 2.3522)
    assert len(horizon) == _N


def test_horizon_values(mock_get):
    horizon, _ = solarpy.horizon.get_horizon_mines(48.8566, 2.3522)
    assert list(horizon) == _ELEVATIONS


def test_meta_keys(mock_get):
    _, meta = solarpy.horizon.get_horizon_mines(48.8566, 2.3522)
    assert set(meta.keys()) == {
        "data_provider", "database",
        "latitude", "longitude", "altitude", "ground_offset",
    }


def test_meta_coordinates(mock_get):
    _, meta = solarpy.horizon.get_horizon_mines(48.8566, 2.3522)
    assert meta["latitude"] == 48.8566
    assert meta["longitude"] == 2.3522


def test_altitude_none_uses_sentinel(mock_get):
    _, meta = solarpy.horizon.get_horizon_mines(48.8566, 2.3522, altitude=None)
    call_url = mock_get.call_args[0][0]
    assert "altitude=-999" in call_url


def test_altitude_explicit(mock_get):
    _, meta = solarpy.horizon.get_horizon_mines(48.8566, 2.3522, altitude=100)
    call_url = mock_get.call_args[0][0]
    assert "altitude=100" in call_url
    assert meta["altitude"] == 100


def test_ground_offset_in_url(mock_get):
    solarpy.horizon.get_horizon_mines(48.8566, 2.3522, ground_offset=2.5)
    call_url = mock_get.call_args[0][0]
    assert "ground_offset=2.5" in call_url


def test_raise_for_status_called(mock_get):
    solarpy.horizon.get_horizon_mines(48.8566, 2.3522)
    mock_get.return_value.raise_for_status.assert_called_once()


def test_http_error_propagates():
    mock_res = MagicMock()
    mock_res.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    with patch("solarpy.horizon.horizon_mines.requests.get", return_value=mock_res):
        with pytest.raises(requests.HTTPError):
            solarpy.horizon.get_horizon_mines(48.8566, 2.3522)


def test_kwargs_forwarded(mock_get):
    solarpy.horizon.get_horizon_mines(48.8566, 2.3522, timeout=10)
    _, kwargs = mock_get.call_args
    assert kwargs.get("timeout") == 10
