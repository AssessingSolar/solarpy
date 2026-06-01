import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def plot_scatter_heatmap(
    x,
    y,
    xlim,
    ylim,
    plot_type="scatter",
    cmap="viridis",
    norm=None,
    sigma=None,
    sort_points=False,
    xbins=100,
    ybins=100,
    mincnt=1,
    ax=None,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()
    fig = ax.figure

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Allow for plotting when all values are nan
    if len(x) == 0:
        return fig, ax

    x, y = np.asarray(x), np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)

    H, xedges, yedges = np.histogram2d(
        x[finite], y[finite], bins=(xbins, ybins), range=(xlim, ylim)
    )

    if sigma is not None:
        H = gaussian_filter(H, sigma=sigma)

    H[H < mincnt] = np.nan

    if plot_type == "hist2d":
        ax.pcolormesh(xedges, yedges, H.T, cmap=cmap, norm=norm, **kwargs)

    elif plot_type == "scatter":
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(xcenters, ycenters)
        counts = H.T.flatten()
        if sort_points:
            order = np.argsort(counts)
        else:
            order = np.arange(len(counts))

        ax.scatter(
            X.flatten()[order],
            Y.flatten()[order],
            c=counts[order],
            cmap=cmap,
            norm=norm,
            **kwargs,
        )

    else:
        raise ValueError(f"plot_type must be 'hist2d' or 'scatter', got '{plot_type}'")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig, ax
