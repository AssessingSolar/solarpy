"""Visualising solar irradiance as a shading heatmap in azimuth–elevation space."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_shading_heatmap(
    value: Any,
    solar_azimuth: Any,
    solar_elevation: Any,
    azimuth_bin_size: float = 1.0,
    elevation_bin_size: float = 1.0,
    cmap: str = "viridis",
    norm=None,
    colorbar: bool = False,
    colorbar_label=None,
    ax: plt.Axes = None,
    figsize: tuple = (5, 7),
    pcolormesh_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a heatmap of solar irradiance in azimuth–elevation space.

    Each cell of the heatmap represents a bin of solar azimuth (x-axis)
    and solar elevation (y-axis). Cell colour encodes the *maximum*
    irradiance value observed in that bin. Bins with no data are shown
    as white cells.

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
    encoding : function, optional
        Method for encoding the values in each bin. Default is ``np.max``.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"viridis"``.
    norm : matplotlib.colors.Normalize, optional
        Normalization instance to map data values to the colormap range.
        If ``None`` (default), linear normalization over the data range
        is used.
    colorbar : bool, optional
        Whether to plot a colorbar. Default is ``False``.
    colorbar_label : str, optional
        Label displayed alongside the colorbar.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure and axes are created.
    figsize : tuple
        The size of the figure if *ax* is not specified. Default is
        ``(5, 7)``.
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
    For the heatmap to be usuful for detecting shading in solar irradiance
    measurement data, the data frquency needs to be less than 10 minutes.

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


    # Discard sub-horizon data
    above = solar_elevation >= 0
    value = value[above]
    solar_azimuth = solar_azimuth[above]
    solar_elevation = solar_elevation[above]

    # ------------------------------------------------------------------ #
    # Build bin edges                                                    #
    # ------------------------------------------------------------------ #
    az_min = np.floor(solar_azimuth.min() / azimuth_bin_size) * azimuth_bin_size
    az_max = np.ceil(solar_azimuth.max() / azimuth_bin_size) * azimuth_bin_size
    el_min = np.floor(solar_elevation.min() / elevation_bin_size) * elevation_bin_size
    el_max = np.ceil(solar_elevation.max() / elevation_bin_size) * elevation_bin_size

    az_edges = np.arange(az_min, az_max + azimuth_bin_size, azimuth_bin_size)
    el_edges = np.arange(el_min, el_max + elevation_bin_size, elevation_bin_size)

    n_az = len(az_edges) - 1
    n_el = len(el_edges) - 1

    # ------------------------------------------------------------------ #
    # Bin indices for each observation                                   #
    # ------------------------------------------------------------------ #
    az_idx = np.clip(
        np.searchsorted(az_edges, solar_azimuth, side="right") - 1, 0, n_az - 1
    )
    el_idx = np.clip(
        np.searchsorted(el_edges, solar_elevation, side="right") - 1, 0, n_el - 1
    )

    # ------------------------------------------------------------------ #
    # Accumulate per-bin maximum (n_el rows × n_az cols)                 #
    # ------------------------------------------------------------------ #
    matrix = np.full((n_el, n_az), np.nan)

    # Sort by irradiance so the last write into each cell is the maximum
    order = np.argsort(value)
    for i in order:
        r, c = el_idx[i], az_idx[i]
        v = value[i]
        if np.isnan(matrix[r, c]) or v > matrix[r, c]:
            matrix[r, c] = v

    # ------------------------------------------------------------------ #
    # Figure / axes                                                      #
    # ------------------------------------------------------------------ #
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    fig = ax.figure

    # ------------------------------------------------------------------ #
    # pcolormesh                                                         #
    # ------------------------------------------------------------------ #
    mesh = ax.pcolormesh(
        az_edges,
        el_edges,
        matrix,
        cmap=cmap,
        norm=norm,
        shading="flat",
        **(pcolormesh_kwargs or {}),
    )

    # ------------------------------------------------------------------ #
    # Colorbar                                                           #
    # ------------------------------------------------------------------ #
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(colorbar_label)

    # ------------------------------------------------------------------ #
    # Axes labels and ticks                                              #
    # ------------------------------------------------------------------ #
    ax.set_xticks(np.arange(0, 360+1, 90))
    ax.set_xlim(0, 360)
    ax.set_xlabel("Solar azimuth [°]")

    el_tick_step = 5 if (el_max - el_min) <= 45 else 10
    el_ticks = np.arange(0, el_max + 1, el_tick_step)
    ax.set_yticks(el_ticks)
    ax.set_ylim(0, el_ticks[-1])
    ax.set_ylabel("Solar elevation [°]")

    return fig, ax
