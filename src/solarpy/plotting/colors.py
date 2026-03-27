"""Functions for styling plots."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors


def two_part_colormap(start_color='lightgrey', colormap='viridis', n_gradient=64,
                      n_colormap=64, colormap_start=0.1):
    """Create a two-part colormap blending a flat color into a standard colormap.

    Parameters
    ----------
    start_color : str or color-like, optional
        The colour at the bottom of the gradient segment. Any Matplotlib
        color specification is accepted. Default is ``"lightgrey"``.
    colormap : str, optional
        Name of the Matplotlib colormap to use for the upper segment.
        Default is ``"viridis"``.
    n_gradient : int, optional
        Number of color steps in the gradient segment. Default is ``64``.
    n_colormap : int, optional
        Number of color steps sampled from *colormap*. Default is ``64``.
    colormap_start : float, optional
        Starting point within *colormap*, in [0, 1]. Values above 0 skip
        the first portion of the colormap, which can improve the visual
        transition from *start_color*. Default is ``0.1``.

    Returns
    -------
    cm_custom : matplotlib.colors.ListedColormap
        A colormap of ``n_gradient + n_colormap`` discrete colors.
        Values below the data range are rendered transparent.

    Notes
    -----
    The resulting colormap has two segments: a smooth gradient from
    *start_color* to the first color of *colormap*, followed by the
    *colormap* itself. Values below the colormap range (e.g. zero or
    negative) are rendered as transparent.

    This is particularly useful for density plots where low-count bins should
    appear as a neutral colour before transitioning into the main colormap.

    Examples
    --------
    Use as a drop-in colormap for a density hexbin plot:

    >>> import solarpy
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import TwoSlopeNorm
    >>> x = np.random.randn(100000)
    >>> y = 2 * x + np.random.randn(100000)
    >>> cm = solarpy.plotting.two_part_colormap(
    ...     start_color='lightgrey', colormap='viridis', colormap_start=0.05)
    >>> norm = TwoSlopeNorm(vmin=1, vcenter=30, vmax=175)
    >>> _ = plt.hexbin(x, y, gridsize=100, cmap=cm, norm=norm, mincnt=1)
    """
    colors_viridis = plt.colormaps[colormap](np.linspace(colormap_start, 1, n_colormap))

    # Create gradient from start color to first colormap color
    t = np.linspace(0, 1, n_gradient)[:, np.newaxis]
    gradient = to_rgb(start_color) + t * (colors_viridis[0, :3] - to_rgb(start_color))
    gradient = np.hstack([gradient, np.ones((n_gradient, 1))])

    # Stack: start color to colormap transition | colormap
    cm_custom = ListedColormap(np.vstack([gradient, colors_viridis]),
                               name=f"{start_color}_to_{colormap}")
    # Set zero and negative values as transparent
    cm_custom.set_under((0, 0, 0, 0))
    return cm_custom


IRRADIANCE_COLOR_BANDS = [
    (-100, -10, "deeppink"),  # unfeasible high negative offsets
    (-10, -2, "orange"),      # high negative offsets
    (-2, 0, "darkgrey"),      # acceptable negative offsets
    (0, 2, "lightgrey"),      # near-zero positive values
]


def irradiance_colormap_and_norm(
    colormap: str | mcolors.Colormap = 'viridis',
    colormap_start: float = 0.1,
    colormap_end: float = 1,
    n_colormap: int = 256,
    vmax: float = 1000,
    solid_bands: list[tuple[float, float, str]] | None = None,
) -> tuple[ListedColormap, BoundaryNorm]:
    """Create a colormap and norm for visualizing solar irradiance data (W/m²).

    The colormap is split into two regions:

    - **Solid color bands** for negative and near-zero values, where irradiance
      measurements are dominated by thermal offsets. The default bands
      (``IRRADIANCE_COLOR_BANDS``) encode data quality: unfeasible offsets
      [-100, -10), high negative offsets [-10, -2), acceptable negative offsets
      [-2, 0), and near-zero positive values [0, 2).
    - **Continuous colormap** for the physically meaningful range from the upper
      edge of the solid bands to ``vmax``.

    Parameters
    ----------
    colormap : str or matplotlib.colors.Colormap, optional
        Colormap used for the continuous irradiance range. Any matplotlib
        colormap name or ``Colormap`` object is accepted. Default is
        ``'viridis'``.
    colormap_start : float, optional
        Start point of the colormap in the range [0, 1]. Values above 0 skip
        the initial portion of the colormap, which can improve visibility on
        some backgrounds. Default is ``0.1``.
    colormap_end : float, optional
        End point of the colormap in the range [0, 1]. Default is ``1``.
    n_colormap : int, optional
        Number of discrete steps in the continuous colormap range. Higher
        values produce smoother color transitions. Default is ``256``.
    vmax : float, optional
        Upper bound of the continuous colormap range [W/m²]. Values above
        ``vmax`` are clipped to the top color. Default is ``1000``.
    solid_bands : list of tuple(float, float, str), optional
        List of ``(lower, upper, color)`` tuples defining the solid color
        bands. Each tuple specifies the lower bound (inclusive), upper bound
        (exclusive), and a matplotlib color string. Bands must be contiguous
        and sorted in ascending order. If ``None``, defaults to
        ``IRRADIANCE_COLOR_BANDS``.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        Combined colormap with solid bands followed by the continuous range.
    norm : matplotlib.colors.BoundaryNorm
        Norm that maps data values to the colormap boundaries.

    Examples
    --------
    Use as a drop-in colormap for an intraday heatmap:

    >>> import solarpy
    >>> import numpy as np
    >>> import pandas as pd
    >>> time = pd.date_range("2024-01-01", periods=365 * 1440, freq="1min")
    >>> daily_cycle = np.sin(np.linspace(0, np.pi, 1440))
    >>> values = np.tile(daily_cycle, 365) * 900 + np.random.randn(365 * 1440) * 5
    >>> values -= 3  # introduce a small thermal offset
    >>> cmap, norm = solarpy.plotting.irradiance_colormap_and_norm(vmax=1000)
    >>> fig, ax = solarpy.plotting.plot_intraday_heatmap(
    ...     time=time,
    ...     values=values,
    ...     cmap=cmap,
    ...     norm=norm,
    ... )
    """
    if solid_bands is None:
        solid_bands = IRRADIANCE_COLOR_BANDS

    cmap_obj = plt.colormaps[colormap] if isinstance(colormap, str) else colormap

    cmap_colors = [cmap_obj(t) for t in np.linspace(colormap_start, colormap_end, n_colormap)]

    boundaries = [sb[0] for sb in solid_bands] + [solid_bands[-1][1]]
    boundaries += list(np.linspace(solid_bands[-1][1], vmax, n_colormap + 1)[1:])
    colors = [mcolors.to_rgba(c) for *_, c in solid_bands] + cmap_colors

    irradiance_cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=len(colors))

    return irradiance_cmap, norm
