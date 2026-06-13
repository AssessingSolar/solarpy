"""
Shading heatmap
================

This example shows how to visualise solar irradiance in
solar azimuth-elevation space using
:py:func:`solarpy.plotting.plot_shading_heatmap`.
"""

# %%
import pvlib
from matplotlib.colors import TwoSlopeNorm

import solarpy

# Read a year of 1-minute GHI measurements
data, meta = solarpy.iotools.read_t16(
    "https://raw.githubusercontent.com/AssessingSolar/solarpy/refs/heads/main/data/LYN_2023.csv",  # noqa: E501
    map_variables=True,
)

location = pvlib.location.Location(meta["latitude"], meta["longitude"])
solar_position = location.get_solarposition(data.index)

# %%
# Normalize DNI by the extraterrestrial irradiance to obtain the
# direct clearness index, which shows shaded periods in the
# solar elevation/azimuth heatmap.
dni_extra = pvlib.irradiance.get_extra_radiation(data.index)

fig, ax = solarpy.plotting.plot_shading_heatmap(
    value=data["dni"] / dni_extra,
    solar_azimuth=solar_position["azimuth"],
    solar_elevation=solar_position["elevation"],
    cmap=solarpy.plotting.two_part_colormap(),
    norm=TwoSlopeNorm(vmin=0, vcenter=0.3, vmax=0.8),
    colorbar_label="Kn = DNI / TOA [-]",
)

# %%
# The grey areas of the heatmap indicate times orientations that never receive direct sunlight,
# which typically corresponds to shading, such as nearby building vegeations or mountains.
