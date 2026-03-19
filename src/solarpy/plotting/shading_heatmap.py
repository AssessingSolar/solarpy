"""Visualising solar irradiance as a shading heatmap in azimuth–elevation space."""

from __future__ import annotations

from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import binned_statistic_2d


def plot_shading_heatmap(
    value: Any,
    solar_azimuth: Any,
    solar_elevation: Any,
    azimuth_bin_size: float = 1.0,
    elevation_bin_size: float = 1.0,
    encoding: Callable[[np.ndarray], float] | str = "max",
    cmap: str = "viridis",
    norm=None,
    northern_hemisphere: bool = True,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    ax: plt.Axes | None = None,
    pcolormesh_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a heatmap of solar irradiance in azimuth–elevation space.

    Each cell of the heatmap represents a bin of solar azimuth (x-axis)
    and solar elevation (y-axis). Cell colour encodes the value returned
    by *encoding* for all observations in that bin.

    It is recommended to first filter out obvious outliers using
    automatic QC checks.

    Parameters
    ----------
    value : array-like of float
        Irradiance time series. Value can be normalized with respect to
        extraterrestrial irradiance.
    solar_azimuth : array-like of float
        Solar azimuth angle in degrees (0–360, measured clockwise from
        North). Must be the same length as *value*.
    solar_elevation : array-like of float
        Solar elevation angle in degrees (0–90 above the horizon). Must
        be the same length as *value*.
    azimuth_bin_size : float, optional
        Width of each azimuth bin in degrees. Default is ``1.0``.
    elevation_bin_size : float, optional
        Height of each elevation bin in degrees. Default is ``1.0``.
    northern_hemisphere : bool, optional
        Set to ``False`` for southern hemisphere sites. The sun transits
        north there, so the solar azimuth path crosses 0°/360°. When
        ``False``, azimuths are shifted to centre the plot around north,
        keeping the sun path continuous. Default is ``True``.
    encoding : callable or str, optional
        Reduction function applied to the values in each bin. Accepts any
        string supported by ``scipy.stats.binned_statistic_2d`` (``'max'``,
        ``'min'``, ``'mean'``, ``'median'``, ``'sum'``, ``'count'``), or a
        callable that takes a 1-D ``np.ndarray`` and returns a scalar, e.g.
        ``lambda x: np.quantile(x, 0.95)``. Default is ``'max'``.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"viridis"``.
    norm : matplotlib.colors.Normalize, optional
        Normalization instance to map data values to the colormap range.
        If ``None`` (default), linear normalization over the data range
        is used.
    colorbar : bool, optional
        Whether to plot a colorbar. Default is ``True``.
    colorbar_label : str, optional
        Label displayed alongside the colorbar.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure and axes are created.
    pcolormesh_kwargs : dict, optional
        Extra keyword arguments forwarded directly to ``ax.pcolormesh``.
        Note that ``cmap``, ``norm``, and ``shading`` are set by the
        function and will raise a ``TypeError`` if passed here. Default
        is ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the heatmap.
    ax : matplotlib.axes.Axes
        The axes containing the heatmap.

    Notes
    -----
    For the heatmap to be useful for detecting shading in solar irradiance
    measurement data, the data frequency needs to be less than 10 minutes.

    Only data points with ``solar_elevation >= 0`` are included; negative
    elevations (sun below the horizon) are discarded.

    Examples
    --------
    >>> import solarpy
    >>> import numpy as np
    >>> n = 50000
    >>> azimuth = np.random.uniform(90, 270, n)
    >>> elevation = np.random.uniform(0, 60, n)
    >>> irradiance = np.sin(np.radians(elevation)) * 1000 + np.random.randn(n) * 50
    >>> fig, ax = solarpy.plotting.plot_shading_heatmap(
    ...     irradiance, azimuth, elevation)
    """
    value = np.asarray(value, dtype=float)
    solar_azimuth = np.asarray(solar_azimuth, dtype=float)
    solar_elevation = np.asarray(solar_elevation, dtype=float)

    # Discard sub-horizon data and non-finite values (nan and inf)
    above_and_finite = (solar_elevation >= 0) & np.isfinite(value)
    value = value[above_and_finite]
    solar_azimuth = solar_azimuth[above_and_finite]
    solar_elevation = solar_elevation[above_and_finite]

    # Southern hemisphere: sun transits north, azimuth wraps around 0°/360°
    if northern_hemisphere:
        az_min = 0
        az_max = 360 - elevation_bin_size
    else:
        az_min = -180
        az_max = 180 - elevation_bin_size
        solar_azimuth = np.where(solar_azimuth > 180, solar_azimuth - 360, solar_azimuth)

    # Build bin edges
    el_min = 0
    el_max = np.ceil(solar_elevation.max() / elevation_bin_size) * elevation_bin_size

    az_edges = np.arange(az_min, az_max + azimuth_bin_size, azimuth_bin_size)
    el_edges = np.arange(el_min, el_max + elevation_bin_size, elevation_bin_size)

    # Accumulate per-bin encoding (n_el rows × n_az cols)
    matrix, _x_edges, _y_edges, _binnumber = binned_statistic_2d(
        solar_azimuth, solar_elevation, value,
        statistic=encoding,
        bins=[az_edges, el_edges],
    )
    matrix = matrix.T  # rows=elevation, cols=azimuth

    # Figure / axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # pcolormesh
    mesh = ax.pcolormesh(
        az_edges,
        el_edges,
        matrix,
        cmap=cmap,
        norm=norm,
        shading="flat",
        **(pcolormesh_kwargs or {}),
    )

    # Colorbar
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(colorbar_label)

    # Axes labels and ticks
    _az_compass = {0: "N", 90: "E", 180: "S", 270: "W", 360: "N"}
    if northern_hemisphere:
        az_ticks = np.arange(0, 360 + 1, 90)
        ax.set_xticks(az_ticks)
        ax.set_xticklabels([_az_compass[x] for x in az_ticks])
        ax.set_xlim(0, 360)
    else:
        az_ticks = np.arange(-180, 180 +1, 90)
        ax.set_xticks(az_ticks)
        ax.set_xticklabels(_az_compass.get(int(x % 360)) for x in az_ticks)
        ax.set_xlim(-180, 180)

    ax.set_xlabel("Solar azimuth [°]")

    el_tick_step = 5 if (el_max - el_min) <= 45 else 10
    el_ticks = np.arange(0, el_max + 1, el_tick_step)
    ax.set_yticks(el_ticks)
    ax.set_ylim(0, el_ticks[-1])
    ax.set_ylabel("Solar elevation [°]")

    return fig, ax
