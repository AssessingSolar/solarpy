import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.gridspec as gridspec
from solarpy.plotting import (
    plot_intraday_heatmap,
    irradiance_colormap_and_norm,
    plot_shading_heatmap,
    plot_google_maps,
    plot_scatter_heatmap,
    two_part_colormap,
)
from solarpy.quality import bsrn_limits
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import pvlib


def _multiplot_layout(figsize=(24, 16)):
    """
    Create the visual plausibility control figure layout.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict with keys:
        'line' : list of 3 axes - time series line charts
        'heatmap' : list of 3 axes - intraday heat maps
        'ts_scatter' : list of 3 axes - time series scatter
        'mid_l' : list of 4 axes  - (middle-left)
        'mid_r' : list of 4 axes  - (middle-right)
        'maps' : list of 3 axes - maps
        'meta' : metadata text
        'hist' : list of 3 axes  - histograms
        'corr' : cross-correlation
        'sun1' : sun-path GHI/TOA/sin(Z)
        'sun2' : sun-path DNI/TOA
    """
    fig = plt.figure(figsize=figsize)

    outer = gridspec.GridSpec(
        1,
        4,
        figure=fig,
        left=0.04,
        right=0.98,
        top=0.97,
        bottom=0.04,
        wspace=0.15,
        width_ratios=[1, 0.6, 0.6, 1],
    )

    # Column 0 - 9 time-series rows
    gs_left = gridspec.GridSpecFromSubplotSpec(
        9,
        1,
        subplot_spec=outer[0],
        hspace=0.1,
        height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1.4],
        # bottom subplot has extra height to accommodate x-ticks
    )
    ax_line = [fig.add_subplot(gs_left[i]) for i in range(0, 3)]
    ax_heatmap = [fig.add_subplot(gs_left[i]) for i in range(3, 6)]
    ax_ts_scatter = [fig.add_subplot(gs_left[i]) for i in range(6, 9)]

    # Column 1 - 4 left scatter plots
    gs_mid_l = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[1], hspace=0.2)
    ax_mid_l = [fig.add_subplot(gs_mid_l[i]) for i in range(4)]

    # Column 2 - 4 right scatter plots
    gs_mid_r = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[2], hspace=0.2)
    ax_mid_r = [fig.add_subplot(gs_mid_r[i]) for i in range(4)]

    # Column 3 - nested layout
    gs_right = gridspec.GridSpecFromSubplotSpec(
        6,
        1,
        subplot_spec=outer[3],
        hspace=0.1,
        height_ratios=[1.5, 0.7, 1.2, 1.2, 1.4, 1.4],
    )
    gs_maps = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_right[0], wspace=0.15
    )
    ax_maps = [fig.add_subplot(gs_maps[i]) for i in range(3)]
    ax_meta = fig.add_subplot(gs_right[1])
    gs_hist = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_right[2], wspace=0.15
    )
    ax_hist = [fig.add_subplot(gs_hist[i]) for i in range(3)]
    ax_corr = fig.add_subplot(gs_right[3])
    ax_sun1 = fig.add_subplot(gs_right[4])
    ax_sun2 = fig.add_subplot(gs_right[5])

    axes = dict(
        line=ax_line,
        heatmap=ax_heatmap,
        ts_scatter=ax_ts_scatter,
        mid_l=ax_mid_l,
        mid_r=ax_mid_r,
        maps=ax_maps,
        meta=ax_meta,
        hist=ax_hist,
        corr=ax_corr,
        sun1=ax_sun1,
        sun2=ax_sun2,
    )
    return fig, axes


def multiplot(times, data, meta, horizon=None, google_api_key=None, figsize=(24, 16)):
    """Create a multiplot for visual checking plausibility of irradiance data.

    Produces a multi-panel figure combining time series, intraday heatmaps,
    scatter plots, sun-path shading heatmaps, histograms, maps, and station
    metadata for a single solar irradiance measurement site.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Timestamps of the measurements. Must have a consistent frequency.
        Consider using :py:func:`solarpy.processing.resample_to_freq`
    data : pandas.DataFrame
        Measurement data. Required columns:

        - ``"ghi"`` — Global Horizontal Irradiance [W/m²]
        - ``"dni"`` — Direct Normal Irradiance [W/m²]
        - ``"dhi"`` — Diffuse Horizontal Irradiance [W/m²]
        - ``"solar_zenith"`` — Solar zenith angle [°]
        - ``"solar_azimuth"`` — Solar azimuth angle [°]

        ``"ghi_extra"``, ``"dni_extra"``, and ``"ghi_calc"`` are computed
        internally from ``times`` and the irradiance columns.

        Optional columns:

        - ``"ghi_clear"`` — Clearsky GHI [W/m²]; if present together with
          ``"is_clearsky"``, a clearsky-index time series panel is shown.
        - ``"is_clearsky"`` — Boolean mask for clearsky conditions; see
          ``"ghi_clear"`` above.
        - ``"flag"`` — Boolean quality flag; if present, flagged and
          unflagged data are shown separately in the clearness-index
          histograms.

    meta : dict
        Station metadata. Required keys: ``"latitude"``, ``"longitude"``.
        Optional keys: ``"altitude"``, ``"name"``, ``"country"``,
        ``"climate"``.
    horizon : pandas.Series, optional
        Horizon elevation profile indexed by azimuth angle [°]. Overlaid
        on the sun-path shading plots. See
        :py:func:`solarpy.horizon.get_horizon_mines`.
    google_api_key : str, optional
        Google Maps Static API key. If not specified, the map panels are
        replaced with a placeholder message.
    figsize : tuple of (float, float), default (24, 16)
        Figure size in inches ``(width, height)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The multiplot figure.
    axes : dict
        Dictionary of axes, with keys ``"line"``, ``"heatmap"``,
        ``"ts_scatter"``, ``"mid_l"``, ``"mid_r"``, ``"maps"``,
        ``"meta"``, ``"hist"``, ``"corr"``, ``"sun1"``, ``"sun2"``.
    """
    # Derive variables
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    cos_sza = np.cos(np.deg2rad(data["solar_zenith"])).clip(lower=0)
    ghi_extra = dni_extra * cos_sza
    ghi_calc = data["dhi"].clip(lower=0) + data["dni"].clip(lower=0) * cos_sza

    components = ["ghi", "dni", "dhi"]
    is_daytime = data["solar_zenith"] < 90
    is_ghi_above_50 = is_daytime & ((data["ghi"] > 50) | (data["dhi"] > 50))
    is_overcast = is_ghi_above_50 & (data["dni"] < 5)
    has_flag = "flag" in data.columns

    Kt = data["ghi"] / ghi_extra
    Kn = data["dni"] / dni_extra
    Kd = data["dhi"] / ghi_extra
    K = data["dhi"] / data["ghi"]

    fig, axes = _multiplot_layout(figsize=figsize)

    ts_xlim = (times.date.min(), times.date.max() + dt.timedelta(days=1))
    days = int(np.ceil((ts_xlim[1] - ts_xlim[0]).days)) + 1
    limit_line_params = {"lw": 1.5, "alpha": 0.8, "c": "r"}

    # Time series plots
    # TODO: remove pandas dependency
    # resampling to speed up the process
    for ax, c in zip(axes["line"], components):
        ax.plot(data[c].resample("5min").max(), lw=0.5)
        ax.set_ylabel(f"{c.upper()} [W/m²]")
        ax.set_xlim(ts_xlim)
        ax.set_ylim(0, None)

    # Determine sunrise and sunset
    sun_rise_set = pvlib.solarposition.sun_rise_set_transit_spa(
        pd.date_range(times.min(), periods=days, freq="1d"),
        meta["latitude"],
        meta["longitude"],
    )
    sun_rise_minutes = (
        sun_rise_set["sunrise"].dt.hour.values * 60
        + sun_rise_set["sunrise"].dt.minute.values
    )
    sun_set_minutes = (
        sun_rise_set["sunset"].dt.hour.values * 60
        + sun_rise_set["sunset"].dt.minute.values
    )

    # Intraday heat map plots
    cmap, norm = irradiance_colormap_and_norm(vmax=1000)
    for ax, c in zip(axes["heatmap"], components):
        plot_intraday_heatmap(
            time=times, values=data[c], ax=ax, plot_colorbar=False, cmap=cmap, norm=norm
        )
        ax.text(0.02, 0.95, c.upper(), va="top", ha="left", transform=ax.transAxes)
        ax.plot(sun_rise_minutes, **limit_line_params)
        ax.plot(sun_set_minutes, **limit_line_params)

    ts_scatter_params = dict(
        xlim=mdates.date2num(ts_xlim),
        xbins=days,
        plot_type="scatter",
        sort_points=True,
        sigma=(1, 0.5),
        s=1,
        cmap=two_part_colormap(),
        norm=TwoSlopeNorm(vmin=1, vcenter=20, vmax=120),
    )
    # Diffuse fraction (K) time series scatter plot (overcast conditions)
    plot_scatter_heatmap(
        x=mdates.date2num(times[is_overcast]),
        y=K[is_overcast],
        ylim=(0.75, 1.25),
        ax=axes["ts_scatter"][0],
        ybins=200,
        mincnt=1,
        **ts_scatter_params,
    )
    axes["ts_scatter"][0].set_ylabel("K = DHI / GHI [-]")
    axes["ts_scatter"][0].text(
        0.02,
        0.98,
        "DNI < 5 W/m²",
        ha="left",
        va="top",
        alpha=0.5,
        transform=axes["ts_scatter"][0].transAxes,
    )

    # Closure equation ratio time series scatter plot (all conditions)
    plot_scatter_heatmap(
        x=mdates.date2num(times[is_ghi_above_50]),
        y=data.loc[is_ghi_above_50, "ghi"] / ghi_calc[is_ghi_above_50],
        ylim=(0.75, 1.25),
        ax=axes["ts_scatter"][1],
        ybins=200,
        mincnt=3,
        **ts_scatter_params,
    )
    axes["ts_scatter"][1].text(
        0.02,
        0.98,
        "GHI > 50 W/m²",
        ha="left",
        va="top",
        alpha=0.5,
        transform=axes["ts_scatter"][1].transAxes,
    )
    axes["ts_scatter"][1].set_ylabel("GHI / (DHI + DNI·cos(Z)) [-]")

    # Clearsky index time series scatter plot (clearsky conditions)
    if ("ghi_clear" in data.columns) & ("is_clearsky" in data.columns):
        plot_scatter_heatmap(
            x=mdates.date2num(times[data["is_clearsky"]]),
            y=data.loc[data["is_clearsky"], "ghi"]
            / data.loc[data["is_clearsky"], "ghi_clear"],
            ylim=(0.75, 1.25),
            ax=axes["ts_scatter"][2],
            ybins=200,
            mincnt=1,
            **ts_scatter_params,
        )
        axes["ts_scatter"][2].set_ylabel("Kc = GHI / GHIclear [-]")
        axes["ts_scatter"][2].text(
            0.02,
            0.98,
            "Clearsky index calculated using McClear for clearsky conditions",
            ha="left",
            va="top",
            alpha=0.75,
            transform=axes["ts_scatter"][2].transAxes,
        )

    for ax in axes["ts_scatter"]:
        ax.axhline(1, linestyle="--", **limit_line_params)

    fig.align_ylabels(axes["line"] + axes["heatmap"] + axes["ts_scatter"])
    # remove xticks
    [ax.set_xticks([]) for ax in axes["line"] + axes["heatmap"] + axes["ts_scatter"]]
    ts_xticks = pd.date_range(ts_xlim[0], ts_xlim[1], freq="MS")
    axes["ts_scatter"][2].set_xticks(ts_xticks, ts_xticks.strftime("%b %Y"))

    # Scatter plot settings
    scatter_params = {
        "cmap": two_part_colormap(),
        "s": 1.5,
        "xbins": 200,
        "ybins": 200,
        "sort_points": True,
    }

    # Irradiance vs. TOA
    discrete_toa = np.linspace(1, 1320)

    scatter_vmax = {"ghi": 175, "dni": 50, "dhi": 250}  # less points for dni
    for ax, c in zip(axes["mid_l"][:3], components):
        plot_scatter_heatmap(
            x=ghi_extra[is_daytime],
            y=data.loc[is_daytime, c],
            ax=ax,
            xlim=(0, 1400),
            ylim=(0, 1600),
            norm=TwoSlopeNorm(vmin=1, vcenter=20, vmax=scatter_vmax[c]),
            **scatter_params,
        )
        # Plot BSRN upper limits for irradiance
        for limit_type in ["erl", "ppl"]:
            limit = f"{c}-{limit_type}"
            # Generate limits for the lowest and highest extraterrestrial irradiance
            low_extra_lim = bsrn_limits(
                np.rad2deg(np.arccos(discrete_toa / 1320)), 1320, limit
            )[1]
            high_extra_lim = bsrn_limits(
                np.rad2deg(np.arccos(discrete_toa / 1414)), 1414, limit
            )[1]
            # Determine which is the lower and upper boundary (switches between components)
            lower_lim = np.min([low_extra_lim, high_extra_lim], axis=0)
            upper_lim = np.max(
                [
                    low_extra_lim,
                    high_extra_lim,
                    # Add a mimium difference (15 W/m^2) for visibility purposes
                    np.min([low_extra_lim, high_extra_lim], axis=0) + 15,
                ],
                axis=0,
            )
            ax.fill_between(
                discrete_toa, lower_lim, upper_lim, facecolor="r", alpha=0.5
            )
        ax.set_xlabel("Top of atmosphere (TOA) horizontal [W/m²]")
        ax.set_ylabel(f"{c.upper()} [W/m²]")

    # Closure test
    plot_scatter_heatmap(
        x=data.loc[is_daytime, "ghi"],
        y=ghi_calc[is_daytime] - data.loc[is_daytime, "ghi"],
        ax=axes["mid_l"][3],
        xlim=(0, 1400),
        ylim=(-200, 200),
        norm=TwoSlopeNorm(vmin=1, vcenter=20, vmax=175),
        **scatter_params,
    )
    axes["mid_l"][3].set_xlabel("GHI [W/m²]")
    axes["mid_l"][3].set_ylabel("DHI + DNI·cos(Z) - GHI [W/m²]")
    x_limits = np.array([50, 1400])
    axes["mid_l"][3].plot(x_limits, +0.08 * x_limits, ls="--", **limit_line_params)
    axes["mid_l"][3].plot(x_limits, -0.08 * x_limits, ls="--", **limit_line_params)
    axes["mid_l"][3].plot(x_limits, +0.15 * x_limits, ls="-.", **limit_line_params)
    axes["mid_l"][3].plot(x_limits, -0.15 * x_limits, ls="-.", **limit_line_params)

    # K vs. zenith
    ax = axes["mid_r"][0]
    plot_scatter_heatmap(
        x=data["solar_zenith"][is_ghi_above_50],
        y=K[is_ghi_above_50],
        ax=ax,
        xlim=(0, 95),
        ylim=(0, 1.4),
        norm=TwoSlopeNorm(vmin=1, vcenter=40, vmax=250),
        **scatter_params,
    )
    ax.plot([0, 75, 75, 93], [1.05, 1.05, 1.10, 1.10], ls="--", **limit_line_params)
    ax.set_xlabel("Solar zenith [°]")
    ax.set_ylabel("K = DHI / GHI [-]")
    ax.text(
        0.02,
        0.98,
        "GHI > 50 W/m²",
        transform=ax.transAxes,
        ha="left",
        va="top",
        alpha=0.5,
    )

    # Kn vs. Kt
    ax = axes["mid_r"][1]
    plot_scatter_heatmap(
        x=Kt[is_daytime],
        y=Kn[is_daytime],
        ax=ax,
        xlim=(0, 1.5),
        ylim=(0, 1.0),
        norm=TwoSlopeNorm(vmin=1, vcenter=40, vmax=250),
        **scatter_params,
    )
    # Kn limit from (Gueymard and Ruiz-Arias 2016): 10.1016/j.solener.2015.10.010
    # Kn < (1100 + 0.03*altitude) / TOA
    # Kd limit from (Nollas et al. 2023): 10.1016/j.renene.2022.11.056
    Kn_limit = (1100 + 0.03 * meta.get("altitude", 0)) / 1320
    ax.plot(
        [0, Kn_limit, 1.4, 1.4],
        [0, Kn_limit, Kn_limit, 0],
        ls="--",
        **limit_line_params,
    )
    ax.set_xlabel("Kt = GHI / TOA / cos(Z) [-]")
    ax.set_ylabel("Kn = DNI / TOA [-]")

    # K vs. Kt
    ax = axes["mid_r"][2]
    plot_scatter_heatmap(
        x=Kt[is_ghi_above_50],
        y=K[is_ghi_above_50],
        ax=ax,
        xlim=(0, 1.5),
        ylim=(0, 1.5),
        norm=TwoSlopeNorm(vmin=1, vcenter=40, vmax=250),
        **scatter_params,
    )
    ax.text(
        0.02,
        0.98,
        "GHI > 50 W/m²",
        transform=ax.transAxes,
        ha="left",
        va="top",
        alpha=0.5,
    )
    # K <= 0.96 for Kt > 0.6 from (Geuder et al. 2015): 10.1016/j.egypro.2015.03.205
    ax.plot(
        [0, 0.6, 0.6, 1.4, 1.4], [1.1, 1.1, 0.96, 0.96, 0], ls="--", **limit_line_params
    )
    ax.set_xlabel("Kt = GHI / TOA / cos(Z) [-]")
    ax.set_ylabel("K = DHI / GHI [-]")

    # Closure test - ratio
    ax = axes["mid_r"][3]
    plot_scatter_heatmap(
        x=data.loc[is_ghi_above_50, "solar_zenith"],
        y=data.loc[is_ghi_above_50, "ghi"] / ghi_calc[is_ghi_above_50],
        ax=ax,
        xlim=(0, 95),
        ylim=(0.5, 1.5),
        norm=TwoSlopeNorm(vmin=1, vcenter=40, vmax=250),
        **scatter_params,
    )
    ax.plot(
        [0, 75, 75, 93, 93], [1.08, 1.08, 1.15, 1.15, 1], ls="--", **limit_line_params
    )
    ax.plot(
        [0, 75, 75, 93, 93], [0.92, 0.92, 0.85, 0.85, 1], ls="--", **limit_line_params
    )
    ax.set_xlabel("Solar zenith [°]")
    ax.set_ylabel("GHI / (DHI + DNI·cos(Z)) [-]")
    ax.text(
        0.02,
        0.98,
        "GHI > 50 W/m²",
        transform=ax.transAxes,
        ha="left",
        va="top",
        alpha=0.5,
    )

    # Maps
    if google_api_key is not None:
        for ax, zoom in zip(axes["maps"], [3, 16, 20]):
            map_type = "satellite" if zoom > 5 else "hybrid"
            plot_google_maps(
                meta["latitude"],
                meta["longitude"],
                api_key=google_api_key,
                zoom=zoom,
                map_type=map_type,
                ax=ax,
                size=(400, 400),
            )
    else:
        [ax.set_axis_off() for ax in axes["maps"]]
        axes["maps"][1].text(
            0.5,
            0.5,
            "Maps not shown.\nA Google API key was not provided.",
            ha="center",
            va="center",
            color="grey",
            transform=axes["maps"][1].transAxes,
        )
    # Make the maps wider
    maps_delta = 0.02
    for ax in axes["maps"]:
        pos = ax.get_position()
        ax.set_position(
            [
                pos.x0 - maps_delta / 3,
                pos.y0 - maps_delta / 3 + 0.018,
                pos.width + maps_delta * 2 / 3,
                pos.height + maps_delta * 2 / 3,
            ]
        )

    # Metadata text
    meta_text = {
        "Name": meta.get("name", "N/A"),
        "Country": meta.get("country", "N/A"),
        "Latitude": f"{meta['latitude']:.4f} °N",
        "Longitude": f"{meta['longitude']:.4f} °E",
        "Altitude": (
            f"{meta['altitude']:.0f} m" if meta.get("altitude") is not None else "N/A"
        ),
        "Climate (KG)": meta.get("climate", "N/A"),
    }
    for ii, (k, v) in enumerate(meta_text.items()):
        axes["meta"].text(
            0.02,
            0.98 - ii * 0.18,
            f"{k}: {v}",
            va="top",
            ha="left",
            transform=axes["meta"].transAxes,
        )
    axes["meta"].axis("off")

    # Statistics text
    min_date, max_date = min(times).date(), max(times).date()
    dt_hours = np.median(np.diff(times.astype("int64"))) / 3.6e12
    period_text = (
        f"Period: {min_date.strftime('%Y-%m-%d')} to "
        f"{max_date.strftime('%Y-%m-%d')}  (days: {days})"
    )
    axes["meta"].text(
        0.4, 0.98, period_text, va="top", ha="left", transform=axes["meta"].transAxes
    )
    axes["meta"].text(
        0.4,
        0.62,
        "Annual equivalent sums ↓",
        va="top",
        ha="left",
        transform=axes["meta"].transAxes,
    )
    for ii, c in enumerate(components):
        s = (
            f"{np.nansum(data[c]) * dt_hours / 1000:>4.0f} kWh/m² "
            f"({np.mean(np.isnan(data[c]))*100:1.1f}% missing)"
        )
        axes["meta"].text(
            0.4,
            0.44 - ii * 0.18,
            f"{c.upper()}: {s}",
            family="monospace",
            va="top",
            ha="left",
            transform=axes["meta"].transAxes,
        )

    # Histograms
    threshold = 5
    ghi_threshold = data["ghi"] > threshold
    dni_threshold = data["dni"] > threshold
    dhi_threshold = data["dhi"] > threshold
    hist_params = dict(range=(0, 1.2), bins=60)
    if has_flag:
        axes["hist"][0].hist(Kt[ghi_threshold].dropna(), color="C1", **hist_params)
        axes["hist"][1].hist(Kn[dni_threshold].dropna(), color="C1", **hist_params)
        axes["hist"][2].hist(
            Kd[dhi_threshold].dropna(), color="C1", **hist_params, label="True"
        )
        flag_mask = ~data["flag"]
    else:
        flag_mask = True
    axes["hist"][0].hist(
        Kt[ghi_threshold & flag_mask].dropna(), color="C0", **hist_params
    )
    axes["hist"][1].hist(
        Kn[dni_threshold & flag_mask].dropna(), color="C0", **hist_params
    )
    axes["hist"][2].hist(
        Kd[dhi_threshold & flag_mask].dropna(), color="C0", **hist_params, label="False"
    )
    if has_flag:
        axes["hist"][2].legend(title="Flagged data")
    axes["hist"][0].set_xlabel("Kt = GHI / TOA / cos(Z) [-]")
    axes["hist"][1].set_xlabel("Kn = DNI / TOA [-]")
    axes["hist"][2].set_xlabel("Kd = DHI / TOA / cos(Z) [-]")
    axes["hist"][0].set_ylabel("count (> 5 W/m²)")
    hist_max_ylim = max([axes["hist"][i].get_ylim()[1] for i in range(3)])
    for ax in axes["hist"]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(hist_params["range"])
        ax.set_ylim(0, hist_max_ylim)
        ax.set_xticks([0, 0.5, 1.0], minor=False)
        ax.set_xticks(np.arange(0, 1.2 + 0.01, 0.1), minor=True)
        ax.set_yticks([])
        # Shrink height to make room for xlabels
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + 0.02, pos.width, pos.height - 0.02])

    # TODO: Add correlation plot
    axes["corr"].set_axis_off()

    # Plot shading plots
    shading_clabels = ["Kt = GHI / TOA / cos(Z) [-]", "Kn = DNI / TOA [-]"]
    for value, clabel, ax in zip(
        [Kt, Kn], shading_clabels, (axes["sun1"], axes["sun2"])
    ):
        mask = value < 1
        # TODO: Make a better filtering
        plot_shading_heatmap(
            value=value[mask],
            solar_azimuth=data["solar_azimuth"][mask],
            solar_elevation=90 - data["solar_zenith"][mask],
            ax=ax,
            cmap=two_part_colormap(),
            norm=TwoSlopeNorm(vmin=0, vcenter=0.05, vmax=0.7),
            colorbar_label=clabel,
            northern_hemisphere=meta["latitude"] > 0,
        )
        if horizon is not None:
            ax.plot(horizon.index, horizon, c="r", label="Horizon line")
    axes["sun1"].set_xticks([])
    axes["sun1"].set_xlabel(None)
    if horizon is not None:
        axes["sun1"].legend(loc="upper right", frameon=False)

    return fig, axes
