from __future__ import annotations

import io

import pandas as pd
import requests


def get_horizon_mines(
    latitude: float,
    longitude: float,
    altitude: float | None = None,
    ground_offset: float = 0,
    url: str = 'http://toolbox.1.webservice-energy.org/service/wps',
    **kwargs,
) -> tuple[pd.Series, dict]:
    """
    Retrieve a horizon elevation profile from the MINES ParisTech SRTM web service.

    Parameters
    ----------
    latitude : float
        in decimal degrees, between -90 and 90, north is positive (ISO 19115)
    longitude : float
        in decimal degrees, between -180 and 180, east is positive (ISO 19115)
    altitude : float, optional
        Altitude in meters. If None, then the altitude is determined from the
        NASA SRTM database.
    ground_offset : float, optional
        Vertical offset in meters for the point of view for which to calculate
        horizon profile. Default is ``0``.
    url : str, default: 'http://toolbox.1.webservice-energy.org/service/wps'
        Base URL for MINES ParisTech horizon profile API
    kwargs:
        Additional keyword arguments passed to ``requests.get``.

    Returns
    -------
    horizon : pd.Series
        Pandas Series of the retrived horizon elevation angles. Index is the
        corresponding horizon azimuth angles.
    metadata : dict
        Dictionary with keys ``'data_provider'``, ``'database'``,
        ``'latitude'``, ``'longitude'``, ``'altitude'``, ``'ground_offset'``.

    Notes
    -----
    The azimuthal resolution is one degree. Also, the returned horizon
    elevations can also be negative.

    Examples
    --------
    Retrieve the horizon profile for Paris, France:

    >>> import solarpy
    >>> horizon, meta = solarpy.horizon.get_horizon_mines(
    ...     latitude=48.8566, longitude=2.3522, timeout=10)
    """
    if altitude is None:  # API will then infer altitude
        altitude = -999

    # Manual formatting of the input parameters separating each by a semicolon
    data_inputs = f"latitude={latitude};longitude={longitude};altitude={altitude};ground_offset={ground_offset}"  # noqa: E501

    params = {
        'service': 'WPS',
        'request': 'Execute',
        'identifier': 'compute_horizon_srtm',
        'version': '1.0.0',
    }

    # The DataInputs parameter of the URL has to be manually formatted and
    # added to the base URL as it contains sub-parameters seperated by
    # semi-colons, which gets incorrectly formatted by the requests function
    # if passed using the params argument.
    res = requests.get(url + '?DataInputs=' + data_inputs, params=params,
                       **kwargs)
    res.raise_for_status()

    # The response text is first converted to a StringIO object as otherwise
    # pd.read_csv raises a ValueError stating "Protocol not known:
    # <!-- PyWPS 4.0.0 --> <wps:ExecuteResponse xmlns:gml="http"
    # Alternatively it is possible to pass the url straight to pd.read_csv
    horizon = pd.read_csv(io.StringIO(res.text), skiprows=27, nrows=360,
                          delimiter=';', index_col=0,
                          names=['horizon_azimuth', 'horizon_elevation'])
    horizon = horizon['horizon_elevation']  # convert to series
    # Note, there is no way to detect if the request is correct. In all cases,
    # the API always returns a status code of OK/200 and no useful error
    # message.

    meta = {'data_provider': 'MINES ParisTech - Armines (France)',
            'database': 'Shuttle Radar Topography Mission (SRTM)',
            'latitude': latitude, 'longitude': longitude, 'altitude': altitude,
            'ground_offset': ground_offset}

    return horizon, meta
