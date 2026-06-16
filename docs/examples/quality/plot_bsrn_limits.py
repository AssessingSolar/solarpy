"""
BSRN limits
===========

The BSRN limit checks specify upper and lower limits for solar irradiance measurements,
defining boundaries for GHI, DHI, and DNI as a function of the extraterrestrial irradiance.
"""

# %%
# The BSRN limits can be calculated using
# :py:func:`solarpy.quality.bsrn_limits`, and the corresponding quality flags
# can be calculated using :py:func:`solarpy.quality.bsrn_limits_flag`.
#
# There are two sets of limits for each component: the extremely rare limits (ERL),
# which are expected to be exceeded only in very rare cases, and the physically
# possible limits (PPL), which are not expected to be exceeded.
#
# In the example below, GHI, DHI, and DNI measurements are plotted against
# the extraterrestrial irradiance on a horizontal plane using
# :py:func:`solarpy.plotting.plot_bsrn_limits`. The red shaded areas correspond to
# the ERL and PPL limits; the width of each band accounts for the variation of
# the Sun-Earth distance over the course of a year.

# %%
# Load example data
# -----------------
#
# The example data is from DTU's station in Lyngby, Denmark north of Copenhagen.
# The data is from 2023 and includes measurements of GHI, DHI, and DNI at a 1-minute resolution.

import matplotlib.pyplot as plt
import numpy as np
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

# Calculate extraterrestrial irradiance on a horizontal plane
dni_extra = pvlib.irradiance.get_extra_radiation(data.index)
cos_sza = np.cos(np.deg2rad(solar_position["apparent_zenith"])).clip(lower=0)
ghi_extra = dni_extra * cos_sza

# %%
# Plot BSRN limits for GHI
# ------------------------
#
# The irradiance data is plotted against the extraterrestrial irradiance
# using a scatter heatmap to differentiate between high and low density areas.
#
# The BSRN limits are plotted as red shaded areas. # Measurements falling outside
# the red bands are flagged as suspicious by
# :py:func:`solarpy.quality.bsrn_limits_flag` and should be inspected further.

fig, axes = solarpy.plotting.plot_bsrn_limits(
    irradiance=data["ghi"],
    component="ghi",
    ghi_extra=ghi_extra,
)

fig.tight_layout()

# %%
# Plot BSRN limits for all three components
# -----------------------------------------

fig, axes = plt.subplots(ncols=3, figsize=(10, 5), sharey=True)

for ax, component in zip(axes, ["ghi", "dhi", "dni"]):
    solarpy.plotting.plot_bsrn_limits(
        irradiance=data[component],
        component=component,
        ghi_extra=ghi_extra,
        ax=ax,
    )

fig.tight_layout()
