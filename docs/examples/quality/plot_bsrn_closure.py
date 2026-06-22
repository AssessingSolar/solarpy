"""
BSRN closure test
==================

The BSRN closure test compares measured GHI against the GHI computed from its
components, DHI + DNI·cos(Z), to check the consistency of the three
irradiance measurements.
"""

# %%
# In the example below, GHI is plotted against DHI + DNI·cos(Z) using
# :py:func:`solarpy.plotting.plot_bsrn_closure`. The BSRN closure threshold
# # limits of for low zenith (±8%, dashed) and high zenith (±15%, dash-dot)
# are overlaid. Measurements falling outside these bands indicate a possible
# inconsistency in at least one of the components.

# %%
# Load example data
# -----------------
#
# The example data is from DTU's station in Lyngby, Denmark north of Copenhagen.
# The data is from 2023 and includes measurements of GHI, DHI, and DNI at a 1-minute resolution.

import matplotlib.pyplot as plt
import pvlib
import solarpy

data, meta = solarpy.iotools.read_t16(
    "https://raw.githubusercontent.com/AssessingSolar/solarpy/refs/heads/main/data/LYN_2023.csv",  # noqa: E501
    map_variables=True,
)

# Calculate solar position
solar_position = pvlib.solarposition.get_solarposition(
    data.index, latitude=meta["latitude"], longitude=meta["longitude"]
)
data["solar_zenith"] = solar_position["apparent_zenith"]

# Only daytime measurements are meaningful for the closure test
is_daytime = data["solar_zenith"] < 90
data = data[is_daytime]

# %%
# Plot the closure test (absolute difference)
# --------------------------------------------
#
# By default, the y-axis shows the absolute difference between the GHI
# components, DHI + DNI·cos(Z) - GHI, in W/m². The closure limits are sloped
# lines, since they are defined as a percentage of GHI.

fig, ax = solarpy.plotting.plot_bsrn_closure(
    ghi=data["ghi"],
    dhi=data["dhi"],
    dni=data["dni"],
    solar_zenith=data["solar_zenith"],
)

fig.tight_layout()

# %%
# Plot the closure test (relative difference)
# --------------------------------------------
#
# Setting ``relative=True`` shows the relative difference instead, i.e.,
# (DHI + DNI·cos(Z) - GHI) / GHI. The closure limits are now horizontal
# lines, since they represent the same fixed percentage thresholds (±8%
# and ±15%) regardless of the GHI level.

fig, ax = solarpy.plotting.plot_bsrn_closure(
    ghi=data["ghi"],
    dhi=data["dhi"],
    dni=data["dni"],
    solar_zenith=data["solar_zenith"],
    relative=True,
)

fig.tight_layout()
