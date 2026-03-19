import matplotlib.pyplot as plt
import numpy as np
import pytest

import solarpy


@pytest.fixture
def az_el_val():
    rng = np.random.default_rng(0)
    n = 5000
    az = rng.uniform(90, 270, n)
    el = rng.uniform(0, 60, n)
    val = np.sin(np.radians(el)) * 1000 + rng.standard_normal(n) * 50
    return az, el, val


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_returns_fig_ax(az_el_val):
    az, el, val = az_el_val
    fig, ax = solarpy.plotting.plot_shading_heatmap(val, az, el)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_axis_labels(az_el_val):
    az, el, val = az_el_val
    _, ax = solarpy.plotting.plot_shading_heatmap(val, az, el)
    assert ax.get_xlabel() == "Solar azimuth [°]"
    assert ax.get_ylabel() == "Solar elevation [°]"


def test_nh_tick_labels(az_el_val):
    az, el, val = az_el_val
    _, ax = solarpy.plotting.plot_shading_heatmap(val, az, el, northern_hemisphere=True)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["N", "E", "S", "W", "N"]


def test_nh_xlim(az_el_val):
    az, el, val = az_el_val
    _, ax = solarpy.plotting.plot_shading_heatmap(val, az, el, northern_hemisphere=True)
    assert ax.get_xlim() == (0, 360)


def test_sh_tick_labels(az_el_val):
    az, el, val = az_el_val
    _, ax = solarpy.plotting.plot_shading_heatmap(val, az, el, northern_hemisphere=False)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["S", "W", "N", "E", "S"]


def test_sh_lim():
    rng = np.random.default_rng(2)
    n = 5000
    az = np.concatenate([rng.uniform(310, 360, n // 2), rng.uniform(0, 50, n // 2)])
    el = rng.uniform(0, 60, n)
    val = np.sin(np.radians(el)) * 1000
    _, ax = solarpy.plotting.plot_shading_heatmap(val, az, el, northern_hemisphere=False)
    assert ax.get_xlim() == (-180, 180)
    assert ax.get_ylim() == (0, 60)


def test_colorbar_present(az_el_val):
    az, el, val = az_el_val
    fig, _ = solarpy.plotting.plot_shading_heatmap(val, az, el, colorbar=True)
    assert len(fig.axes) == 2


def test_colorbar_absent(az_el_val):
    az, el, val = az_el_val
    fig, _ = solarpy.plotting.plot_shading_heatmap(val, az, el, colorbar=False)
    assert len(fig.axes) == 1


def test_existing_ax_reused(az_el_val):
    az, el, val = az_el_val
    fig_in, ax_in = plt.subplots()
    fig_out, ax_out = solarpy.plotting.plot_shading_heatmap(val, az, el, ax=ax_in)
    assert fig_out is fig_in
    assert ax_out is ax_in
