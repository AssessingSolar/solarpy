@ -0,0 +1,351 @@
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.gridspec as gridspec
from solarpy.plotting import plot_intraday_heatmap, irradiance_colormap_and_norm, plot_shading_heatmap, plot_google_maps
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm


# %%

def create_multiplot_layout(figsize=(24, 16)):
    """
    Create the visual plausibility control figure layout.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict with keys:
        'ts'    : list of 9 axes  — time series (left column)
        'mid_l'   : list of 4 axes  — (middle-left)
        'mid_r'   : list of 4 axes  — (middle-right)
        'maps'  : list of 3 axes — maps
        'meta'  : metadata text
        'hist'  : list of 3 axes  — histograms
        'corr' : cross-correlation
        'sun1'  : sun-path GHI/TOA
        'sun2'  : sun-path DNI/TOANI
    """
    fig = plt.figure(figsize=figsize)

    outer = gridspec.GridSpec(
        1, 4, figure=fig,
        left=0.04, right=0.98, top=0.97, bottom=0.04,
        wspace=0.15, width_ratios=[1, 0.6, 0.6, 1],
    )

    # Column 0 — 9 time-series rows
    gs_left = gridspec.GridSpecFromSubplotSpec(
        9, 1, subplot_spec=outer[0], hspace=0.1,
        height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1.4],
        # bottom subplot has extra height to accommodate x-ticks
    )
    ax_ts = [fig.add_subplot(gs_left[i]) for i in range(9)]

    # Column 1 — 4 left scatter plots
    gs_mid_l = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[1], hspace=0.2)
    ax_mid_l = [fig.add_subplot(gs_mid_l[i]) for i in range(4)]

    # Column 2 — 4 right scatter plots
    gs_mid_r = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[2], hspace=0.2)
    ax_mid_r = [fig.add_subplot(gs_mid_r[i]) for i in range(4)]

    # Column 3 — nested layout
    gs_right = gridspec.GridSpecFromSubplotSpec(
        6, 1, subplot_spec=outer[3], hspace=0.1,
        height_ratios=[1.5, 1, 1.2, 1.2, 1.4, 1.4],
    )
    gs_maps = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[0], wspace=0.05)
    ax_maps = [fig.add_subplot(gs_maps[i]) for i in range(3)]
    ax_meta  = fig.add_subplot(gs_right[1])
    gs_hist  = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[2], wspace=0.15)
    # make ax_hist share y-axis limit
    ax_hist  = [fig.add_subplot(gs_hist[0])]
    ax_hist += [fig.add_subplot(gs_hist[i], sharey=ax_hist[0]) for i in range(1, 3)]
    ax_corr = fig.add_subplot(gs_right[3])
    ax_sun1  = fig.add_subplot(gs_right[4])
    ax_sun2  = fig.add_subplot(gs_right[5])

    axes = dict(
        ts=ax_ts, mid_l=ax_mid_l, mid_r=ax_mid_r,
        maps=ax_maps, meta=ax_meta,
        hist=ax_hist, corr=ax_corr, sun1=ax_sun1, sun2=ax_sun2,
    )
    return fig, axes


def create_multiplot(data, meta, horizon, figsize=(24, 16)):
    
    # Derive variables
    components = ['ghi', 'dni', 'dhi']
    is_daytime = data['solar_zenith'] < 90
    ghi_above_50 = is_daytime & ((data['ghi'] > 50) | (data['dhi'] > 50))
    Kt = data['ghi'] / data['ghi_extra']
    Kn = data['dni'] / data['dni_extra']
    Kd = data['dhi'] / data['ghi_extra']
    K = data['dhi'] / data['ghi']

    fig, axes = create_multiplot_layout(figsize=figsize)


    ts_xlim = (data.index.date.min(), data.index.date.max() + dt.timedelta(days=1))
    days = (ts_xlim[1] - ts_xlim[0]).days

    # Time series plots
    # xxx: pandas dependency
    # resampling to speed up the process
    for ax, c in zip(axes['ts'][0:3], components)
        ax.plot(data[c].resample('5min').max(), lw=0.5)
        ax.set_ylabel(f"{c.upper} [W/m²]")

    # Intraday heat map plots
    cmap, norm = irradiance_colormap_and_norm(vmax=1000)
    for ax, c in zip(axes['ts'][3:6], components):
        plot_intraday_heatmap(time=data.index, values=data[c], ax=ax,
                              plot_colorbar: False, cmap: cmap, norm: norm)
        ax.text(0.02, 0.95, c.upper(), va='top', ha='left', transform=ax.transAxes)

    # TODO: Plot sunrise/sunset lines
    _ = [ax.set_xticks([]) for ax in axes['ts'][:-1]]
    
    for ax, qty in zip(axes['ts'][6:], ['K', 'GHIratio', 'kc']):
        Plot_TimeseriesRatio(
            data.rename(columns={'ghi': 'GHI', 'dhi': 'DIF', 'dni': 'DNI',
                                 'ghi_extra': 'TOA', 'ghi_calc': 'GHI_est',
                                 'ghi_clear': 'CLEAR_SKY_GHI'}),
            qty=qty, showXlabel=False, ax=ax)


    # Scatter plot settings
    scatter_kwargs = {
        'cmap': solarpy.plotting.two_part_colormap(),
        's': 1,
        'xbins': 200,
        'ybins': 200,
    }

    # Irradiance vs. TOA
    vmax = {'ghi': 175, 'dni': 50, 'dhi': 250}  # less points for dni
    for ax, c in zip(axes['mid_l'][:3], components):
        plot_scatter_heatmap(
            x=data.loc[is_daytime, 'ghi_extra'],
            y=data.loc[is_daytime, c],
            ax=ax,
            xlim=(0, 1400), ylim=(0, 1600),
            norm=TwoSlopeNorm(vmin=1, vcenter=20, vmax=vmax[c]),
            **scatter_kwargs)
        ax.set_xlabel("Top of atmosphere [W/m²]")
        ax.set_ylabel(f"{c.upper()} [W/m²]")

    # Closure test
    plot_scatter_heatmap(
        x=data.loc[is_daytime, 'ghi'],
        y=data.loc[is_daytime, 'ghi_calc'],
        ax=axes['mid_l'][3],
        xlim=(0, 1400), ylim=(0, 1400),
        norm=TwoSlopeNorm(vmin=1, vcenter=20, vmax=175),
        **scatter_kwargs)
    axes['mid_l'][3].set_xlabel("GHI [W/m²]")
    axes['mid_l'][3].set_ylabel("DHI + DNI·cos(Z) [W/m²]")
    limit_line_kwargs = {'lw': 1, 'alpha': 0.75, 'c': 'r'}
    axes['mid_l'][3].plot([0, 1400], [0, 1400*1.08], ls='--', **limit_line_kwargs)
    axes['mid_l'][3].plot([0, 1400], [0, 1400*0.92], ls='--', **limit_line_kwargs)
    axes['mid_l'][3].plot([0, 1400], [0, 1400*1.15], ls='-.', **limit_line_kwargs)
    axes['mid_l'][3].plot([0, 1400], [0, 1400*0.85], ls='-.', **limit_line_kwargs)

    # Diffuse fraction (K) vs. zenith
    ax = axes['mid_r'][0]
    plot_scatter_heatmap(
        x=data['solar_zenith'][ghi_above_50],
        y=K[ghi_above_50],
        ax=ax,
        xlim=(0, 95), ylim=(0, 1.4),
        norm=TwoSlopeNorm(vmin=1, vcenter=40, vmax=250),
        **scatter_kwargs)
    ax.plot([0, 75, 75, 93], [1.05, 1.05, 1.08, 1.08], ls='--', **limit_line_kwargs)
    ax.set_xlabel('Solar zenith [°]')
    ax.set_ylabel('K = DHI / GHI [-]')
    ax.text(0.02, 0.98, "GHI > 50 W/m²", transform=ax.transAxes, ha='left', va='top', alpha=0.5)

    # Kn vs. Kt
    ax = axes['mid_r'][1]
    plot_scatter_heatmap(
        x=Kt[is_daytime],
        y=Kn[is_daytime],
        ax=ax,
        xlim=(0, 1.5), ylim=(0, 1.0),
        norm=TwoSlopeNorm(vmin=1, vcenter=40, vmax=250),
        **scatter_kwargs)
    # TODO: Check these values
    ax.plot([0, 0.8, 1.3, 1.3], [0, 0.8, 0.8, 0], ls='--', **limit_line_kwargs)
    ax.set_xlabel('Kt = GHI/TOA [-]')
    ax.set_ylabel('Kn = DNI/TOANI [-]')

    # K vs. Kt
    ax = axes['mid_r'][2]
    plot_scatter_heatmap(
        x=Kt[is_daytime],
        y=K[is_daytime],
        ax=ax,
        xlim=(0, 1.5), ylim=(0, 1.5),
        norm=TwoSlopeNorm(vmin=1, vcenter=40, vmax=250),
        **scatter_kwargs)
    # TODO: Check these values
    ax.plot([0, 0.6, 0.6, 1.3, 1.3], [1.1, 1.1, 1, 1, 0], ls='--', **limit_line_kwargs)
    ax.set_xlabel('Kt = GHI/TOA [-]')
    ax.set_ylabel('K = DHI/GHI [-]')

    # Closure test - ratio
    ax = axes['mid_r'][3]
    plot_scatter_heatmap(
        x=data.loc[ghi_above_50, 'solar_zenith'],
        y=data.loc[ghi_above_50, 'ghi'] / data.loc[ghi_above_50, 'ghi_calc'],
        ax=ax,
        xlim=(0, 95), ylim=(0.5, 1.5),
        norm=TwoSlopeNorm(vmin=1, vcenter=40, vmax=250),
        **scatter_kwargs)
    ax.plot([0, 75, 75, 93, 93], [1.08, 1.08, 1.15, 1.15, 1], ls='--', **limit_line_kwargs)
    ax.plot([0, 75, 75, 93, 93], [0.92, 0.92, 0.85, 0.85, 1], ls='--', **limit_line_kwargs)
    ax.set_xlabel('Solar zenith [°]')
    ax.set_ylabel('GHI / (DHI + DNI·cos(Z)) [-]')
    ax.text(0.02, 0.98, "GHI > 50 W/m²", transform=ax.transAxes, ha='left', va='top', alpha=0.5)

    # Maps
    # TODO: Make the maps wider
    for ax, zoom in zip(axes['maps'], [3, 16, 20]):
        map_type = 'satellite' if zoom > 5 else 'hybrid'
        plot_google_maps(
            meta['latitude'], meta['longitude'], api_key=google_api_key,
            zoom=zoom, map_type=map_type, ax=ax, size=(400, 400))

    # Text

    meta_text = {
        "Name": meta.get("name", ""),
        "Country": meta.get("country", ""),
        "Latitude": f"{meta['latitude']:.4f} °N",
        "Longitude": f"{meta['longitude']:.4f} °E",
        "Altitude": f"{meta['altitude']:.0f} m" if meta.get('altitude') is not None else "",
        "Climate": meta.get("climate", ""),
    }

    axes['meta'].axis('off')
    x_coll_1 = 0.02
    for ii, (k, v) in enumerate(meta_text.items()):
        axes['meta'].text(x_coll_1, 0.98 - ii * 0.18, f"{k}: {v}", va='top', ha='left', transform=axes['meta'].transAxes)

    # Histograms
    hist_xlim = (0, 1.2)
    hist_bins = 60
    axes['hist'][0].set_ylabel('count (> 5 W/m²)')
    threshold = 5

    axes['hist'][0].hist(Kt[data['ghi'] > threshold].dropna(), range=hist_xlim, bins=hist_bins)
    axes['hist'][1].hist(Kn[data['dni'] > threshold].dropna(), range=hist_xlim, bins=hist_bins)
    axes['hist'][2].hist(Kd[data['dhi'] > threshold].dropna(), range=hist_xlim, bins=hist_bins)
    axes['hist'][0].set_xlabel('Kt = GHI / TOA·cos(Z) [-]')
    axes['hist'][0].set_xlabel('Kn = DNI / TOA [-]')
    axes['hist'][0].set_xlabel('Kd = DHI / TOA·cos(Z) [-]')

    for ii, ax in enumerate(axes['hist']):
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(hist_xlim)
        if ii > 2:
            ax.set_yticks([])


    plot_clearsky_cross_correlation(
        times=data.index,
        ghi=data['ghi'],
        ghi_clear=data['ghi_clear'],
        is_clearsky=data['is_clearsky'],
        ax=axes['corr'],
    )

    # Plot shading plots
    for c, ax in zip(['ghi', 'dni'], (axes['sun1'], axes['sun2'])):
        value = data[c] / data[f"{c}_extra"]
        mask = value < 1
        # TODO: Make a better filtering
        plot_shading_heatmap(
            value=value[mask], solar_azimuth=data['solar_azimuth'][mask],
            solar_elevation=90 - data['solar_zenith'][mask],
            ax=ax, colorbar_label=f"{c.upper()} [W/m²]",
            northern_hemisphere=meta['latitude'] > 0)
        ax.plot(horizon.index, horizon, c='r', label='Horizon line')
    axes['sun1'].set_xticks([])
    axes['sun1'].legend(loc='upper right', frameon=False)
    axes['sun1'].set_xlabel(None)

    return fig, axes
