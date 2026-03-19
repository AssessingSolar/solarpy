import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.gridspec as gridspec
from solarpy.plotting import plot_intraday_heatmap, irradiance_colormap_and_norm, plot_scatter_heatmap, plot_shading_heatmap
import matplotlib.dates as mdates


# %%

# ── Layout ────────────────────────────────────────────────────────────────────

def create_multiplot_layout(figsize=(24, 16)):
    """
    Create the visual plausibility control figure layout.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict with keys:
        'ts'    : list of 9 axes  — time series (left column)
        'toa'   : list of 4 axes  — TOA scatter (middle-left)
        'zen'   : list of 4 axes  — zenith/KT scatter (middle-right)
        'map1'  : regional map
        'map2'  : station aerial
        'meta'  : metadata text
        'hist'  : list of 3 axes  — histograms
        'xcorr' : cross-correlation
        'sun1'  : sun-path GHI/TOA
        'sun2'  : sun-path DNI/TOANI
    """
    fig = plt.figure(figsize=figsize)

    outer = gridspec.GridSpec(
        1, 4, figure=fig,
        left=0.04, right=0.98, top=0.97, bottom=0.04,
        wspace=0.08, width_ratios=[1.4, 1, 1, 1.5],
    )

    # Column 0 — 9 time-series rows
    gs_left = gridspec.GridSpecFromSubplotSpec(
        9, 1, subplot_spec=outer[0], hspace=0.08,
        height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1.4],
        # bottom subplot has extra height to accomodate x-ticks
    )
    ax_ts = [fig.add_subplot(gs_left[i]) for i in range(9)]

    # Column 1 — 4 TOA scatter rows
    gs_mid_l = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[1], hspace=0.10)
    ax_toa = [fig.add_subplot(gs_mid_l[i]) for i in range(4)]

    # Column 2 — 4 zenith/KT scatter rows
    gs_mid_r = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[2], hspace=0.10)
    ax_zen = [fig.add_subplot(gs_mid_r[i]) for i in range(4)]

    # Column 3 — nested layout
    gs_right = gridspec.GridSpecFromSubplotSpec(
        6, 1, subplot_spec=outer[3], hspace=0.35,
        height_ratios=[1.5, 1.5, 1.2, 1.0, 1.4, 1.4],
    )
    gs_maps = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_right[0], wspace=0.05)
    ax_map1  = fig.add_subplot(gs_maps[0])
    ax_map2  = fig.add_subplot(gs_maps[1])
    ax_meta  = fig.add_subplot(gs_right[1])
    gs_hist  = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_right[2], wspace=0.15)
    ax_hist  = [fig.add_subplot(gs_hist[i]) for i in range(3)]
    ax_xcorr = fig.add_subplot(gs_right[3])
    ax_sun1  = fig.add_subplot(gs_right[4])
    ax_sun2  = fig.add_subplot(gs_right[5])

    axes = dict(
        ts=ax_ts, toa=ax_toa, zen=ax_zen,
        map1=ax_map1, map2=ax_map2, meta=ax_meta,
        hist=ax_hist, xcorr=ax_xcorr, sun1=ax_sun1, sun2=ax_sun2,
    )
    return fig, axes


def create_multiplot(data, figsize=(24, 16)):
    fig, axes = create_multiplot_layout(figsize=figsize)


    ts_xlim = (data.index.date.min(), data.index.date.max() + dt.timedelta(days=1))
    days = (ts_xlim[1] - ts_xlim[0]).days

    # Time series plots
    # xxx: pandas dependency
    # resampling to speed up the process
    axes['ts'][0].plot(data['ghi'].resample('5min').max(), lw=0.5)
    axes['ts'][1].plot(data['dni'].resample('5min').max(), lw=0.5)
    axes['ts'][2].plot(data['dhi'].resample('5min').max(), lw=0.5)
    axes['ts'][0].set_ylabel('GHI [W/m²]')
    axes['ts'][1].set_ylabel('DNI [W/m²]')
    axes['ts'][2].set_ylabel('DHI [W/m²]')

    # Intraday heat map plots
    cmap, norm = irradiance_colormap_and_norm(vmax=1000)
    heatmap_kwargs = {'plot_colorbar': False, 'cmap': cmap, 'norm': norm}
    plot_intraday_heatmap(time=data.index, values=data['ghi'], ax=axes['ts'][3], **heatmap_kwargs)
    plot_intraday_heatmap(time=data.index, values=data['dni'], ax=axes['ts'][4], **heatmap_kwargs)
    plot_intraday_heatmap(time=data.index, values=data['dhi'], ax=axes['ts'][5], **heatmap_kwargs)
    axes['ts'][3].text(0.02, 0.95, 'GHI', va='top', ha='left', transform=axes['ts'][3].transAxes)
    axes['ts'][4].text(0.02, 0.95, 'DNI', va='top', ha='left', transform=axes['ts'][4].transAxes)
    axes['ts'][5].text(0.02, 0.95, 'DHI', va='top', ha='left', transform=axes['ts'][5].transAxes)
    # TODO: Plot sunrise/sunset lines
    _ = [ax.set_xticks([]) for ax in axes['ts'][:-1]]
    
    # Diffuse fraction time-series plot
    data['K'] = data['dhi'] / data['ghi']
    # todo: should threshold be one should be greater than 50 W/m2?
    data.loc[(data[['ghi', 'dhi']].min(1) <= 0) | (data['solar_zenith'] > 85), 'K'] = np.nan
    data['K_cloudy'] = data['K']
    data.loc[data['dni']>5, 'K_cloudy'] = np.nan
    plot_scatter_heatmap(
        #x=np.asarray(data.index, dtype="datetime64[ns]"), y=data['K_cloudy'],
        x=mdates.date2num(data.index), y=data['K_cloudy'],
        #x=np.arange(data.shape[0]), y=data['K_cloudy'],
        plot_type='scatter', cmap='viridis', norm=None,
        xlim=mdates.date2num(ts_xlim),
        ylim=(0.8, 1.2),
        xbins=days, ybins=100, ax=axes['ts'][6])

    

    # Plot shading plots
    for c, ax in zip(['ghi', 'dni'], (axes['sun1'], axes['sun2'])):
        value = data[c] / data[f"{c}_extra"]
        value[value>1] = np.nan
        plot_shading_heatmap(
            value=value, solar_azimuth=data['solar_azimuth'],
            solar_elevation=90 - data['solar_zenith'],
            ax=ax, colorbar_label=f"{c.capitalize()} [W/m²]")
    axes['sun1'].set_xticks([])
    axes['sun1'].set_xlabel(None)

    return fig, axes

fig, axes = create_multiplot(data)

# TODO: align all TS plots. Simply set xlim for all of them?
# %%



fig, ax = plt.subplots(figsize=(10, 4))
plot_scatter_heatmap(
    #x=np.asarray(data.index, dtype="datetime64[ns]"), y=data['K_cloudy'],
    x=mdates.date2num(data.index), y=data['K_cloudy'],
    #x=np.arange(data.shape[0]), y=data['K_cloudy'],
    plot_type='hist2d', cmap='viridis', norm=None,
    xlim=mdates.date2num(ts_xlim),
    ylim=(0.8, 1.2),
    xbins=days, ybins=50,
    mincnt=15, ax=ax)#, **{'s': 22, 'alpha': 0.5})
ax.axhline(1, c='r', linestyle='dashed', lw=1)