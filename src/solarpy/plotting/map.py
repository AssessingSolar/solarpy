"""Utilities for fetching and displaying Google Maps Static API imagery."""

from io import BytesIO

import matplotlib.pyplot as plt
import requests

GOOGLE_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"


def plot_google_maps(
    latitude: float,
    longitude: float,
    api_key: str,
    zoom: int = 20,
    map_type: str = "satellite",
    size: tuple = (400, 400),
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Fetch a Google Maps Static image and render it on a Matplotlib Axes.

    Queries the Google Maps Static API to retrieve a map image centered on the
    given coordinates and renders it using ``ax.imshow``. A crosshair marker is
    overlaid when zoomed out (``zoom < 20``) to indicate the center point.

    Parameters
    ----------
    latitude : float
        Latitude of the map center, in decimal degrees. Must be in [-90, 90].
    longitude : float
        Longitude of the map center, in decimal degrees. Must be in [-180, 180].
    api_key : str
        Google Maps Static API key.
    zoom : int, optional
        Zoom level between 0 (world) and 21 (building). Default is ``20``.
    map_type : str, optional
        Map rendering style. One of ``"roadmap"``, ``"satellite"``,
        ``"terrain"``, or ``"hybrid"``. Default is ``"satellite"``.
    size : tuple, optional
        Width and height of the requested image in pixels. Default is
        ``(400, 400)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None``, a new figure and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the map.
    ax : matplotlib.axes.Axes
        The axes containing the rendered map image.

    Raises
    ------
    requests.HTTPError
        If the API returns a non-2xx status code (e.g. 403 for an invalid
        API key, 400 for bad parameters).
    requests.Timeout
        If the request exceeds the 10-second timeout.

    Notes
    -----
    A white crosshair (``'w+'``) is plotted at the image center when
    ``zoom < 20``, where individual features are too small to locate the
    center point visually.

    Examples
    --------
    Plot a satellite view of Copenhagen:

    >>> import solarpy
    >>> solarpy.plotting.plot_google_maps(
    ...     55.6761, 12.5683, api_key="YOUR_KEY", zoom=12)  # doctest: +SKIP
    """
    if ax is None:
        fig, ax = plt.subplots()
    fig = ax.figure

    params = {
        "center": f"{latitude},{longitude}",
        "zoom": zoom,
        "size": f"{size[0]}x{size[0]}",
        "markers": f"color:red|{latitude},{longitude}",
        "maptype": map_type,
        "key": api_key,
    }

    response = requests.get(GOOGLE_MAPS_URL, params=params)
    response.raise_for_status()

    image = plt.imread(BytesIO(response.content), format="png")
    ax.imshow(image)

    if zoom < 20:
        ax.plot(size[0] // 2, size[1] // 2, "w+", ms=10)

    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig, ax
