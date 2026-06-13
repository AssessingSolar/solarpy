"""
Shading heatmap
================

This example shows how to visualise solar irradiance in
azimuth-elevation (sun-path) space using
:py:func:`solarpy.plotting.plot_shading_heatmap`.
"""

# %%
import pandas as pd
import pvlib
from matplotlib.colors import TwoSlopeNorm

import solarpy

times = pd.date_range("2023-01-01", periods=24 * 8760, freq="min", tz="UTC")
location = pvlib.location.Location(latitude=55.68, longitude=12.57)
solar_position = location.get_solarposition(times)
clearsky = location.get_clearsky(times)
dni_extra = pvlib.irradiance.get_extra_radiation(times)

# %%
# Simulate a fictional obstacle that blocks direct irradiance whenever the
# sun is within a fixed range of azimuth and elevation angles.
shaded = (
    (solar_position["azimuth"] > 90)
    & (solar_position["azimuth"] < 110)
    & (solar_position["elevation"] < 8)
)
dni_clear = clearsky["dni"].copy()
dni_clear[shaded] = 0

fig, ax = solarpy.plotting.plot_shading_heatmap(
    value=dni_clear / dni_extra,
    solar_azimuth=solar_position["azimuth"],
    solar_elevation=solar_position["elevation"],
    cmap=solarpy.plotting.two_part_colormap(),
    norm=TwoSlopeNorm(vmin=0, vcenter=0.05, vmax=0.7),
    colorbar_label="Normalized DNI [-]",
)
