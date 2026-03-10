"""Functions for styling plots."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, ListedColormap


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

    >>> from matplotlib.colors import TwoSlopeNorm
    >>> x = np.random.randn(100_000)
    >>> y = 2 * x + np.random.randn(100_000)
    >>> cm = two_part_colormap(start_color='whitesmoke', colormap='plasma',
    ...                        colormap_start=0.05)
    >>> norm = TwoSlopeNorm(vmin=1, vcenter=30, vmax=175)
    >>> fig, ax = plt.subplots()
    >>> ax.hexbin(x, y, gridsize=100, cmap=cm, norm=norm, mincnt=1)
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
