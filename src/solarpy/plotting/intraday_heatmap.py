"""Visualising intraday time series data as a time vs. date heatmap."""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_intraday_heatmap(
    time: Any,
    values: Any,
    resolution: int = 1,
    cmap: str = "viridis",
    norm=None,
    plot_colorbar=True,
    colorbar_label: str = "",
    ax: plt.Axes = None,
    pcolormesh_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a heatmap of intraday time series data.

    Each column of the heatmap represents one calendar date; each row
    represents one time bin of *resolution* minutes. Cell colour encodes
    the mean of *values* falling in that date and bin. Dates with no data
    are included as all-NaN columns so the time axis is always contiguous.

    Parameters
    ----------
    time : array-like of datetime-like
        Timestamps corresponding to each value. Must be convertible to
        ``numpy.datetime64``.
    values : array-like of float
        Observed values, one per timestamp. Must be the same length as
        *time*.
    resolution : int, optional
        Bin size in minutes. Must evenly divide 1440. Default is ``1``
        (one row per minute). Use ``10`` for 10-minute bins, ``60`` for
        hourly bins, etc.
    cmap : str, optional
        Matplotlib colormap name. Default is ``"viridis"``.
    norm : matplotlib.colors.Normalize, optional
        Normalization instance to map data values to the colormap range.
        Accepts any ``matplotlib.colors`` norm, e.g. ``Normalize``,
        ``LogNorm``, ``TwoSlopeNorm``, ``BoundaryNorm``. If ``None``
        (default), linear normalization over the data range is used.
    plot_colorbar : bool, optional
        Whether to plot a colorbar. Default is ``True``.
    colorbar_label : str, optional
        Label displayed alongside the colorbar. Default is ``""``.
    pcolormesh_kwargs : dict, optional
        Extra keyword arguments forwarded directly to ``ax.pcolormesh``.
        Useful for parameters not exposed explicitly, such as ``vmin``,
        ``vmax``, ``alpha``, or ``rasterized``. Note that ``cmap``,
        ``norm``, and ``shading`` are set by the function and will raise
        a ``TypeError`` if passed here. Default is ``None``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the heatmap.
    ax : matplotlib.axes.Axes
        The axes containing the heatmap.

    Raises
    ------
    ValueError
        If *time* and *values* have different lengths, if either is empty,
        or if *resolution* does not evenly divide 1440.

    Notes
    -----
    When multiple values fall in the same bin their mean is displayed.
    Missing bin/date combinations are shown as white cells.

    The y-axis runs from midnight (00:00) at the bottom to 23:59 at the
    top (for 1-minute resolution). X-axis tick density adapts to the date
    range: daily labels for short ranges, weekly or monthly for longer ones.

    Examples
    --------
    Minute-resolution data over two weeks:

    >>> import solarpy
    >>> import numpy as np
    >>> import pandas as pd
    >>> mins = np.arange(14 * 1440)
    >>> time = pd.Timestamp("2024-01-01") + pd.to_timedelta(mins, unit='min')
    >>> values = np.sin(mins / 1440 * np.pi) + 0.1 * np.random.randn(len(mins))
    >>> fig, ax = solarpy.plotting.plot_intraday_heatmap(
    ...     time, values, cmap="viridis")

    Ten-minute bins over one year:

    >>> mins = np.arange(365 * 144) * 10
    >>> time = pd.Timestamp("2024-01-01") + pd.to_timedelta(mins, unit='min')
    >>> values = np.random.randn(len(mins))
    >>> fig, ax = solarpy.plotting.plot_intraday_heatmap(
    ...     time, values, resolution=10)
    """
    time = np.asarray(time, dtype="datetime64[ns]")
    values = np.asarray(values, dtype=float)

    if len(time) != len(values):
        raise ValueError(
            f"time and values must have the same length, "
            f"got {len(time)} and {len(values)}."
        )
    # The smallest resolution currently supported is 1min
    if 1440 % resolution != 0:
        raise ValueError(f"resolution must evenly divide 1440, got {resolution}.")

    n_bins = 1440 // resolution

    # ------------------------------------------------------------------ #
    # Extract date and bin index                                         #
    # ------------------------------------------------------------------ #
    dates = time.astype("datetime64[D]")
    minutes = (time - dates).astype("timedelta64[m]").astype(int)
    bin_idx = minutes // resolution

    # Contiguous date range — missing dates become all-NaN columns
    all_dates = np.arange(
        dates.min(), dates.max() + np.timedelta64(1, "D"), np.timedelta64(1, "D")
    )
    n_dates = len(all_dates)

    # ------------------------------------------------------------------ #
    # Figure / axes                                                      #
    # ------------------------------------------------------------------ #
    if ax is None:
        fig, ax = plt.subplots(figsize=(min(max(4, n_dates * 0.5), 8), 2))
    fig = ax.figure

    # ------------------------------------------------------------------ #
    # Build n_bins × n_dates matrix, averaging duplicate timestamps      #
    # ------------------------------------------------------------------ #
    date_idx = np.searchsorted(all_dates, dates)

    total = np.zeros((n_bins, n_dates), dtype=float)
    count = np.zeros((n_bins, n_dates), dtype=int)
    np.add.at(total, (bin_idx, date_idx), values)
    np.add.at(count, (bin_idx, date_idx), 1)

    matrix = np.where(count > 0, total / count, np.nan)

    # ------------------------------------------------------------------ #
    # pcolormesh expects cell edges: (n+1,) arrays                       #
    # ------------------------------------------------------------------ #
    x_edges = mdates.date2num(all_dates.astype("datetime64[ms]").astype(object))
    x_edges = np.append(x_edges, x_edges[-1] + 1)

    mesh = ax.pcolormesh(
        x_edges,
        np.arange(n_bins + 1),
        matrix,
        cmap=cmap,
        norm=norm,
        shading="flat",
        **(pcolormesh_kwargs or {}),
    )

    # ------------------------------------------------------------------ #
    # Colorbar                                                           #
    # ------------------------------------------------------------------ #
    if plot_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(colorbar_label)

    # ------------------------------------------------------------------ #
    # X-axis — dynamic tick density based on date range                  #
    # ------------------------------------------------------------------ #
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y %b"))
    ax.tick_params(axis="x", rotation=30)

    # ------------------------------------------------------------------ #
    # Y-axis — time of day (HH), ticks every 3 hours, midnight at bottom #
    # ------------------------------------------------------------------ #
    bins_per_hour = 3 * 60 // resolution
    tick_bins = np.arange(0, n_bins, bins_per_hour)
    ax.set_yticks(tick_bins + 0.5)
    ax.set_yticklabels(
        [f"{(b * resolution) // 60:02d}" for b in tick_bins],
    )
    ax.set_ylabel("Time of day")

    return fig, ax
