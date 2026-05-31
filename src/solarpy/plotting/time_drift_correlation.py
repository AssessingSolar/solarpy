from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_time_drift_correlation(
    times: pd.DatetimeIndex,
    ghi: pd.Series,
    ghi_clear: pd.Series,
    is_clearsky: pd.Series[bool],
    window: str = "5D",
    min_periods: int = 240,
    plot_colorbar: bool = True,
    colorbar_label: str = "Correlation [-]",
    cmap: str = "viridis_r",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot time-drift correlation between measured and clear-sky GHI.

    For each time lag between −30 and +30 minutes, the Pearson correlation
    between measured GHI and the time-shifted clear-sky GHI is computed using
    a rolling window over clear-sky periods only. The result is resampled to
    daily means and displayed as a heatmap with date on the x-axis and lag on
    the y-axis. A trend in the correlation peak at a non-zero lag indicates
    a systematic timing offset in the measured data.

    Parameters
    ----------
    times : array-like of datetime-like
        Timestamps corresponding to each observation.
    ghi : array-like of float
        Measured global horizontal irradiance [W/m²].
    ghi_clear : array-like of float
        Modelled clear-sky global horizontal irradiance [W/m²].
    is_clearsky : array-like of bool
        Boolean mask that is ``True`` for clear-sky periods. Only flagged
        periods are used to calculate correlation.
    window : str, optional
        Rolling window size passed to :meth:`pandas.Series.rolling`.
        Default is ``'5D'`` (five days).
    min_periods : int, optional
        Minimum number of observations required within the rolling window,
        passed  to :meth:`pandas.Series.rolling`. Windows with fewer
        observations produce ``NaN``. Default is ``240`` (4 hours of 1-min data).
    plot_colorbar : bool, optional
        Whether to add an inset colorbar. Default is ``True``.
    colorbar_label : str, optional
        Label displayed alongside the colorbar. Default is
        ``"Correlation (-)"``.
    cmap : str, optional
        Matplotlib colormap name used for the correlation heatmap.
        Default is ``'viridis_r'``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes containing the plot.

    Examples
    --------
    >>> import pvlib
    >>> import solarpy
    >>> data, meta = solarpy.iotools.read_t16("data/LYN_2023.csv", map_variables=True)
    >>> location = pvlib.location.Location(meta["latitude"], meta["longitude"])
    >>> cs = location.get_clearsky(data.index)
    >>> is_clearsky = pvlib.clearsky.detect_clearsky(data["ghi"], cs["ghi"], data.index)
    >>> fig, ax = solarpy.plotting.plot_time_drift_correlation(
    ...     times=data.index,
    ...     ghi=data["ghi"],
    ...     ghi_clear=cs["ghi"],
    ...     is_clearsky=is_clearsky,
    ... )
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # select measured and clearsky GHI data and set other values to NaN
    times = pd.DatetimeIndex(times)
    days = pd.date_range(min(times), max(times), freq="1d")

    ghi_clear_csd = pd.Series(ghi_clear, index=times)[is_clearsky]
    ghi_csd = pd.Series(ghi, index=times)[is_clearsky]

    # calculate correlation for time lags varying between -30 and +30 minutes
    time_lags = np.arange(-30, 31)
    correlations = np.zeros((len(days), len(time_lags)))

    for ii, time_lag in enumerate(time_lags):
        ghi_clear_shifted = pd.Series(ghi_clear, index=times).shift(
            time_lag, freq="min"
        )

        corr = ghi_csd.rolling(window=window, min_periods=min_periods).corr(
            ghi_clear_shifted
        )
        corr_daily = corr.resample("1D").mean()

        correlations[:, ii] = corr_daily.values

    x_lims = mdates.date2num((min(days), max(days)))
    y_lims = min(time_lags), max(time_lags)

    im = ax.imshow(
        correlations.T,
        aspect="auto",
        extent=[x_lims[0], x_lims[1], y_lims[0], y_lims[1]],
        interpolation="nearest",
        vmin=0.9925,
        vmax=1,
        cmap=cmap,
        alpha=0.65,
    )

    ax.axhline(0, c="r", linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y %b"))
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Time lag [min.]")
    ax.set_xlim(x_lims)

    if plot_colorbar:
        cax = inset_axes(
            ax,
            width="30%",
            height="3%",
            loc="upper right",
            # bbox_to_anchor=(0, 0.01, 1, 1),
            bbox_transform=ax.transAxes,
        )
        cbar = fig.colorbar(
            im,
            cax=cax,
            orientation="horizontal",
            label=colorbar_label,
        )

    return fig, ax
