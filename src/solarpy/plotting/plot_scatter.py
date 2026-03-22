import matplotlib.pyplot as plt
import numpy as np


def conv2(v1, v2, m, mode='same'):
    
    tmp = np.apply_along_axis(np.convolve, 0, m, v1, mode)
    return np.apply_along_axis(np.convolve, 1, tmp, v2, mode)


def plot_scatter_heatmap(x, y, plot_type='scatter', cmap='viridis', norm=None,
                         xlim=(0, 95), ylim=(0, 1.4),
                         xbins=100, ybins=100, mincnt=1, ax=None, **kwargs):
    x, y = np.asarray(x), np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)

    if ax is None:
        fig, ax = plt.subplots()
    fig = ax.figure

    H, xedges, yedges = np.histogram2d(
        x[finite], y[finite], bins=(xbins, ybins), range=(xlim, ylim))
    #hist=conv2(x[finite], y[finite], H)

    if plot_type == 'hist2d':
        H[H < mincnt] = np.nan
        ax.pcolormesh(xedges, yedges, H.T, cmap=cmap, norm=norm, **kwargs)

    elif plot_type == 'scatter':
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(xcenters, ycenters)
        counts = H.T.flatten()
        mask = counts > mincnt
        ax.scatter(X.flatten()[mask], Y.flatten()[mask], c=counts[mask],
                   cmap=cmap, norm=norm, **kwargs)

    else:
        raise ValueError(f"plot_type must be 'hist2d' or 'scatter', got '{plot_type}'")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig, ax
