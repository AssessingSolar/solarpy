"""Visualising intraday time series data as a date × minute heatmap."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_intraday_heatmap(
    time: Any,
    values: Any,
    resolution: int = 1,
    cmap: str = "viridis",
    colorbar_label: str = "",
    ax: plt.Axes = None,
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
    colorbar_label : str, optional
        Label displayed alongside the colorbar. Default is ``""``.
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
    top. X-axis tick density adapts to the date range: daily labels for
    short ranges, weekly or monthly for longer ones.

    Examples
    --------
    Minute-resolution data over two weeks:

    >>> import numpy as np
    >>> mins = np.arange(14 * 1440)
    >>> time = np.datetime64("2024-01-01") + mins * np.timedelta64(1, "m")
    >>> values = np.sin(mins / 1440 * np.pi) + np.random.randn(len(mins)) * 0.1
    >>> fig, ax = plot_time_heatmap(time, values, cmap="RdYlGn")
    >>> fig.savefig("heatmap.png", dpi=150, bbox_inches="tight")

    Ten-minute bins over one year:

    >>> mins = np.arange(365 * 144) * 10
    >>> time = np.datetime64("2024-01-01") + mins * np.timedelta64(1, "m")
    >>> values = np.random.randn(len(mins))
    >>> fig, ax = plot_time_heatmap(time, values, resolution=10)
    """
    time = np.asarray(time, dtype="datetime64[ns]")
    values = np.asarray(values, dtype=float)

    if time.size == 0 or values.size == 0:
        raise ValueError("time and values must not be empty.")
    if len(time) != len(values):
        raise ValueError(
            f"time and values must have the same length, "
            f"got {len(time)} and {len(values)}."
        )
    if 1440 % resolution != 0:
        raise ValueError(f"resolution must evenly divide 1440, got {resolution}.")

    n_bins = 1440 // resolution

    # ------------------------------------------------------------------ #
    # Extract date and bin index                                           #
    # ------------------------------------------------------------------ #
    dates = time.astype("datetime64[D]")
    minutes = (time - dates).astype("timedelta64[m]").astype(int)
    bin_idx = minutes // resolution

    # Contiguous date range — missing dates become all-NaN columns
    all_dates = np.arange(dates.min(), dates.max() + np.timedelta64(1, "D"),
                          np.timedelta64(1, "D"))
    n_dates = len(all_dates)

    # ------------------------------------------------------------------ #
    # Build n_bins × n_dates matrix, averaging duplicate timestamps       #
    # ------------------------------------------------------------------ #
    date_idx = np.searchsorted(all_dates, dates)

    total = np.zeros((n_bins, n_dates), dtype=float)
    count = np.zeros((n_bins, n_dates), dtype=int)
    np.add.at(total, (bin_idx, date_idx), values)
    np.add.at(count, (bin_idx, date_idx), 1)

    matrix = np.where(count > 0, total / count, np.nan)

    # ------------------------------------------------------------------ #
    # Figure / axes                                                        #
    # ------------------------------------------------------------------ #
    if ax is None:
        fig, ax = plt.subplots(figsize=(min(max(6, n_dates * 0.5), 12), 8))
    fig = ax.figure

    # ------------------------------------------------------------------ #
    # pcolormesh expects cell edges: (n+1,) arrays                        #
    # ------------------------------------------------------------------ #
    mesh = ax.pcolormesh(
        np.arange(n_dates + 1),
        np.arange(n_bins + 1),
        matrix,
        cmap=cmap,
        shading="flat",
    )

    # ------------------------------------------------------------------ #
    # Colorbar                                                             #
    # ------------------------------------------------------------------ #
    # cbar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(colorbar_label)

    # ------------------------------------------------------------------ #
    # X-axis — dynamic tick density based on date range                   #
    # ------------------------------------------------------------------ #
    if n_dates <= 30:           # daily
        tick_step = 1
        date_fmt = "%Y-%m-%d"
    elif n_dates <= 180:        # weekly
        tick_step = 7
        date_fmt = "%Y-%m-%d"
    else:                       # monthly (approx)
        tick_step = 30
        date_fmt = "%b %Y"

    tick_positions = np.arange(0, n_dates, tick_step)
    ax.set_xticks(tick_positions + 0.5)
    ax.set_xticklabels(
        [all_dates[i].astype("datetime64[D]").astype(object).strftime(date_fmt)
         for i in tick_positions],
        rotation=45,
        ha="right",
    )

    # ------------------------------------------------------------------ #
    # Y-axis — time of day (HH:MM), ticks every hour, midnight at bottom  #
    # ------------------------------------------------------------------ #
    bins_per_hour = 60 // resolution
    tick_bins = np.arange(0, n_bins, bins_per_hour)
    ax.set_yticks(tick_bins + 0.5)
    ax.set_yticklabels(
        [f"{(b * resolution) // 60:02d}:{(b * resolution) % 60:02d}"
         for b in tick_bins],
    )
    ax.set_ylabel("Time of day")

    fig.tight_layout()
    return fig, ax
