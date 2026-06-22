"""Visualising solar irradiance data against the BSRN upper and lower limits."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from solarpy.plotting.colors import two_part_colormap
from solarpy.plotting.plot_scatter import plot_scatter_heatmap
from solarpy.quality.limits import bsrn_limits


def plot_bsrn_limits(
    irradiance: Any,
    component: str,
    ghi_extra: Any,
    xlim: tuple[float, float] = (0, 1400),
    ylim: tuple[float, float] = (0, 1600),
    min_limit_diff: float = 15,
    scatter_vmax: float | None = None,
    cmap: Any = None,
    s: float = 1.5,
    bins: tuple[int, int] = (200, 200),
    norm: Any = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot scatter heatmaps of irradiance vs. extraterrestrial irradiance
    with BSRN limits visualized.

    The plot can either plot GHI, DHI, or DNI, as a function of the extraterrestrial
    irradiance on a horizontal plane (*ghi_extra*) using
    :func:`solarpy.plotting.plot_scatter_heatmap`. The
    extremely rare limits (ERL) and physically possible limits (PPL) from
    :func:`solarpy.quality.bsrn_limits` are overlaid as shaded bands.

    Parameters
    ----------
    irradiance : array-like of float
        Irradiance measurements [W/m²].
    component : {"ghi", "dhi", "dni"}
        The type of irradiance measurements provided in *irradiance*.
    ghi_extra : array-like of float
        Extraterrestrial irradiance on a horizontal plane [W/m²]. Must be
        the same length as *irradiance*.
    xlim : tuple of float, optional
        Limits of the x-axis (extraterrestrial irradiance).
        Default is ``(0, 1400)``.
    ylim : tuple of float, optional
        Limits of the y-axis (measured irradiance). Default is ``(0, 1600)``.
    min_limit_diff : float, optional
        Minimum difference between the ERL and PPL limits for visibility purposes.
        Default is 15 W/m².
    scatter_vmax : float, optional
        Upper bound of the scatter heatmap color scale. If ``None`` (default),
        ``{"ghi": 175, "dhi": 250, "dni": 50}[component]`` is used.
    cmap : matplotlib.colors.Colormap, optional
        Colormap used for the scatter heatmaps. If ``None`` (default),
        :func:`solarpy.plotting.two_part_colormap` is used.
    norm : matplotlib.colors.Normalize, optional
        Normalization for the scatter heatmaps. If ``None`` (default), a
        :class:`matplotlib.colors.TwoSlopeNorm` with ``vmin=1``, ``vcenter=20``,
        and ``vmax=scatter_vmax`` is used.
    s : float, optional
        Marker size for the scatter heatmap. Default is 1.5.
    bins : tuple of int, optional
        Number of bins for the scatter heatmap in the x and y directions.
        Default is ``(200, 200)``.
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
    solarpy.quality.bsrn_limits
    solarpy.quality.bsrn_limits_flag
    solarpy.plotting.plot_bsrn_closure
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if cmap is None:
        cmap = two_part_colormap()

    if scatter_vmax is None:
        scatter_vmax = {"ghi": 175, "dhi": 250, "dni": 50}[component]

    if norm is None:
        norm = TwoSlopeNorm(vmin=1, vcenter=20, vmax=scatter_vmax)

    low_extra, high_extra = 1320, 1414
    discrete_toa = np.linspace(1, low_extra)

    plot_scatter_heatmap(
        x=ghi_extra,
        y=irradiance,
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

    for limit_type in ["erl", "ppl"]:
        limit = f"{component}-{limit_type}"
        # Generate limits for the lowest and highest extraterrestrial irradiance
        low_extra_lim = bsrn_limits(
            np.rad2deg(np.arccos(discrete_toa / low_extra)), low_extra, limit
        )[1]
        high_extra_lim = bsrn_limits(
            np.rad2deg(np.arccos(discrete_toa / high_extra)), high_extra, limit
        )[1]
        # Determine the lower and upper boundary (switches between components)
        lower_lim = np.min([low_extra_lim, high_extra_lim], axis=0)
        upper_lim = np.max(
            [
                low_extra_lim,
                high_extra_lim,
                # Add a minimum difference for visibility purposes
                lower_lim + min_limit_diff,
            ],
            axis=0,
        )
        ax.fill_between(discrete_toa, lower_lim, upper_lim, facecolor="r", alpha=0.5)

    ax.set_aspect("equal")
    ax.set_ylabel(f"{component.upper()} [W/m²]")
    ax.set_xlabel("Top of atmosphere (TOA) irradiance\non horizontal [W/m²]")

    return fig, ax
