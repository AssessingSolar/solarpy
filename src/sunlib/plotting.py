'''Heatmap as a function of solar elevation vs. azimuth.'''

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pvlib


def plot_heat_map_elevation_vs_azimuth(
        data, solar_azimuth, solar_elevation, ax=None,
        northern_hemisphere=True, agg_method='max',
        azimuth_resolution=1, elevation_resolution=0.2,
        azimuth_lim=(-180, 180), elevation_lim=None,
        figsize=(7, 3), cmap='jet', vmin=0, vmax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)


    # if not northern_hemisphere:
    #     solar_azimuth = (solar_azimuth-180 ) % 360

    solar_azimuth = (solar_azimuth/azimuth_resolution).round(0)*azimuth_resolution
    solar_elevation = (solar_elevation/elevation_resolution).round(0)*elevation_resolution

    df_2d = data.groupby([solar_azimuth, solar_elevation]).agg(agg_method).unstack(level=0).sort_index()

    extent = [df_2d.columns.min(), df_2d.columns.max(), df_2d.index.min(), df_2d.index.max()]

    if vmax is None:
        vmax = data.quantile(0.99)

    im = ax.imshow(df_2d, aspect='auto', origin='lower', cmap=cmap,
                   extent=extent, vmin=vmin, vmax=vmax)

    if elevation_lim is None:
        elevation_lim = (0, np.max(np.ceil(solar_elevation/10))*10)
    ax.set_ylim(elevation_lim)

    ax.set_xlabel('Solar azimuth [°N]')
    ax.set_ylabel('Solar elevation [°]')
    ax.set_xticks((np.array([0, 90, 180, 270, 360]) + northern_hemisphere*180 ) % 360)
    ax.set_yticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1]+0.001, 10))

    return ax


def plot_heatmap_hour_vs_date(
        data, times, latitude, longitude, ax=None, na_threshold=1,
        figsize=(7, 3), cmap='jet', vmin=0, vmax=None):
    hourofday = times.hour + times.minute/60
    dates = times.date

    # Create dataframe with rows corresponding to days and columns to hours
    df_2d = pd.DataFrame(data).set_index([dates, hourofday]).unstack(level=0)
    df_2d = df_2d[data.name]
    
    complete_date_index = pd.date_range(times.min(), times.max(), freq='1d')
    df_2d = df_2d.reindex(complete_date_index.date, axis='columns')
    if na_threshold is not None:
        df_2d[df_2d < na_threshold] = np.nan
    
    # Calculate the extents of the 2D plot [x_start, x_end, y_start, y_end]
    xlims = mdates.date2num([dates.min(), dates.max()])
    extent = list(xlims) + [0, 24]
    
    # Plot heat map
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if vmax is None:
        vmax = data.quantile(0.99)


    im = ax.imshow(df_2d,  aspect='auto', origin='lower', cmap='jet',
                   extent=extent, vmin=vmin, vmax=vmax)
    
    # Plot sunrise and sunset
    sunrise_sunset = pvlib.solarposition.sun_rise_set_transit_spa(
        complete_date_index, latitude=latitude, longitude=longitude)
    
    # Convert sunrise/sunset from datetime to decimal hours
    sunrise_sunset['sunrise'] = sunrise_sunset['sunrise'].dt.hour + \
        sunrise_sunset['sunrise'].dt.minute/60
    sunrise_sunset['sunset'] = sunrise_sunset['sunset'].dt.hour + \
        sunrise_sunset['sunset'].dt.minute/60

    ax.plot(mdates.date2num(sunrise_sunset.index),
            sunrise_sunset[['sunrise', 'sunset']].to_numpy(),
            c='r', linestyle='--', lw=2)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.01, label='[W/m$^2$]')

    # Format plot
    ax.set_xlim()
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_ylabel('Time of day [h]')
    ax.set_facecolor('lightgrey')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    return ax
