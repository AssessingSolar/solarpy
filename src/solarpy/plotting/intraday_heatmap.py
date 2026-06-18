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
    time_resolution: int | str = "infer",
    cmap: str = "viridis",
    norm=None,
    plot_colorbar=True,
    colorbar_label: str = "",
    ax: plt.Axes = None,
    pcolormesh_kwargs: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a heatmap of intraday time series data.

    Each column of the heatmap represents one calendar date; each row
    represents one time bin of *time_resolution* minutes. Cell colour encodes
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
    time_resolution : int or ``"infer"``, optional
        Bin size in minutes. Must evenly divide 1440. When ``"infer"``
        (default), the resolution is estimated from the median difference
        between consecutive timestamps. Use ``10`` for 10-minute bins,
        ``60`` for hourly bins, etc.
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
        or if *time_resolution* does not evenly divide 1440.

    Notes
    -----
    When multiple values fall in the same bin their mean is displayed.
    Missing bin/date combinations are shown as white cells.

    The y-axis runs from midnight (00:00) at the bottom to the following midnight at the
    top, labelled in hours, with ticks every 3 hours. X-axis tick density adapts to the date
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
    ...     time, values, time_resolution=10)
    """
    time = np.asarray(time, dtype="datetime64[ns]")
    values = np.asarray(values, dtype=float)

    if len(time) != len(values):
        raise ValueError(
            f"time and values must have the same length, "
            f"got {len(time)} and {len(values)}."
        )
    if time_resolution == "infer":
        time_resolution = int(
            np.median(np.diff(np.sort(time)).astype("timedelta64[m]").astype(int))
        )

    # The smallest time_resolution currently supported is 1min
    if 1440 % time_resolution != 0:
        raise ValueError(
            f"time_resolution must evenly divide 1440, got {time_resolution}."
        )

    n_bins = 1440 // time_resolution

    # ------------------------------------------------------------------ #
    # Extract date and bin index                                         #
    # ------------------------------------------------------------------ #
    dates = time.astype("datetime64[D]")
    minutes = (time - dates).astype("timedelta64[m]").astype(int)
    bin_idx = minutes // time_resolution

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

    # Should be in hours
    y_edges = np.arange(n_bins + 1) * time_resolution / 60

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
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
    tick_hours = np.arange(0, 24, 3)
    ax.set_yticks([f"t:{t:02d}" for t in tick_hours])
    ax.set_ylabel("Time of day [h]")
    ax.set_ylim(0, 24)

    return fig, ax
