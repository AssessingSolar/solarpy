"""Visualising solar irradiance data against the BSRN comparison checks."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from solarpy.plotting.colors import two_part_colormap
from solarpy.plotting.plot_scatter import plot_scatter_heatmap


def plot_bsrn_closure(
    ghi: Any,
    dhi: Any,
    dni: Any,
    solar_zenith: Any,
    relative: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    scatter_vmax: float | None = None,
    cmap: Any = None,
    s: float = 1.5,
    bins: tuple[int, int] = (200, 200),
    norm: Any = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a scatter heatmap of the BSRN closure test.

    Compares measured GHI against the GHI computed from its components,
    *GHI_calc* = DHI + DNI·cos(Z), using
    :func:`solarpy.plotting.plot_scatter_heatmap`. Closure limits
    of ±8% (dashed) and ±15% (dash-dot) are overlaid.

    Parameters
    ----------
    ghi : array-like of float
        Measured GHI [W/m²].
    dhi : array-like of float
        Measured DHI [W/m²]. Must be the same length as *ghi*.
    dni : array-like of float
        Measured DNI [W/m²]. Must be the same length as *ghi*.
    solar_zenith : array-like of float
        Solar zenith angle [°]. Must be the same length as *ghi*.
    relative : bool, optional
        If ``False`` (default), GHI is plotted on the x-axis and the y-axis
        shows the absolute difference ``ghi_calc - ghi`` in W/m²; the
        closure limits are sloped lines at ±8%/±15% of GHI. If ``True``,
        the solar zenith angle is plotted on the x-axis and the y-axis
        shows the ratio ``ghi / ghi_calc`` [-]; the closure limits widen
        from ±8% to ±15% above a zenith angle of 75°, reflecting the
        larger measurement uncertainty at low sun elevations.
    xlim : tuple of float, optional
        Limits of the x-axis. If ``None`` (default), ``(0, 1400)`` is used
        when *relative* is ``False`` (GHI [W/m²]), and ``(0, 95)`` when
        *relative* is ``True`` (solar zenith [°]).
    ylim : tuple of float, optional
        Limits of the y-axis. If ``None`` (default), ``(-200, 200)`` is
        used when *relative* is ``False``, and ``(0.5, 1.5)`` when
        *relative* is ``True``.
    scatter_vmax : float, optional
        Upper bound of the scatter heatmap color scale. If ``None``
        (default), 175 is used when *relative* is ``False``, and 250 when
        *relative* is ``True``.
    cmap : matplotlib.colors.Colormap, optional
        Colormap used for the scatter heatmap. If ``None`` (default),
        :func:`solarpy.plotting.two_part_colormap` is used.
    s : float, optional
        Marker size for the scatter heatmap. Default is 1.5.
    bins : tuple of int, optional
        Number of bins for the scatter heatmap in the x and y directions.
        Default is ``(200, 200)``.
    norm : matplotlib.colors.Normalize, optional
        Normalization for the scatter heatmap. If ``None`` (default), a
        :class:`matplotlib.colors.TwoSlopeNorm` with ``vmin=1``,
        ``vcenter=20`` (or ``40`` if *relative*), and ``vmax=scatter_vmax``
        is used.
    ax : matplotlib.axes.Axes, optional
        Axes to draw the plot on. If ``None``, a new figure with a single
        axis is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the heatmap.
    ax : matplotlib.axes.Axes
        The axes containing the heatmap.

    See Also
    --------
    solarpy.plotting.plot_bsrn_limits
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if cmap is None:
        cmap = two_part_colormap()

    cos_sza = np.clip(np.cos(np.deg2rad(solar_zenith)), 0, None)
    ghi_calc = np.clip(dhi, 0, None) + np.clip(dni, 0, None) * cos_sza

    if relative:
        x, y = solar_zenith, ghi / ghi_calc
        xlim = (0, 95) if xlim is None else xlim
        ylim = (0.5, 1.5) if ylim is None else ylim
        scatter_vmax = 250 if scatter_vmax is None else scatter_vmax
        vcenter = 40
        xlabel, ylabel = "Solar zenith [°]", "GHI / (DHI + DNI·cos(Z)) [-]"
    else:
        x, y = ghi, ghi_calc - ghi
        xlim = (0, 1400) if xlim is None else xlim
        ylim = (-200, 200) if ylim is None else ylim
        scatter_vmax = 175 if scatter_vmax is None else scatter_vmax
        vcenter = 20
        xlabel, ylabel = "GHI [W/m²]", "DHI + DNI·cos(Z) - GHI [W/m²]"

    if norm is None:
        norm = TwoSlopeNorm(vmin=1, vcenter=vcenter, vmax=scatter_vmax)

    plot_scatter_heatmap(
        x=x,
        y=y,
        ax=ax,
        xlim=xlim,
        ylim=ylim,
        s=s,
        xbins=bins[0],
        ybins=bins[1],
        sort_points=True,
        cmap=cmap,
        norm=norm,
    )

    limit_line_params = {"lw": 1.5, "alpha": 0.8, "c": "r", "linestyle": "--"}
    if relative:
        ax.plot([0, 75, 75, 93, 93], [1.08, 1.08, 1.15, 1.15, 1], **limit_line_params)
        ax.plot([0, 75, 75, 93, 93], [0.92, 0.92, 0.85, 0.85, 1], **limit_line_params)
    else:
        x_limits = np.array([max(xlim[0], 50), xlim[1]])
        for frac, linestyle in zip([0.08, 0.15], ["--", "-."]):
            ax.plot(
                x_limits,
                frac * x_limits,
                **{**limit_line_params, "linestyle": linestyle},
            )
            ax.plot(
                x_limits,
                -frac * x_limits,
                **{**limit_line_params, "linestyle": linestyle},
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax
