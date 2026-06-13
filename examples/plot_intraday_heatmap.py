"""
Intraday heatmap
=================

This example shows how to visualise solar irradiance as a time-of-day vs.
date heatmap using :py:func:`solarpy.plotting.plot_intraday_heatmap`.
"""

# %%
import solarpy

# Read a year of 1-minute GHI measurements
data, meta = solarpy.iotools.read_t16(
    "https://raw.githubusercontent.com/AssessingSolar/solarpy/refs/heads/main/data/LYN_2023.csv",  # noqa: E501
    map_variables=True,
)

fig, ax = solarpy.plotting.plot_intraday_heatmap(
    data.index, data["ghi"], colorbar_label="GHI [W/m$^2$]"
)
