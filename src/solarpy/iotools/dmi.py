from __future__ import annotations

import pandas as pd
import requests

URL = "https://opendataapi.dmi.dk/v2/climateData/"
LIMIT = 300_000

# Maps DMI parameter IDs to pvlib/solarpy standard variable names.
VARIABLE_MAP = {
    "mean_radiation": "ghi",
    "mean_temp": "temp_air",
    "mean_wind_speed": "wind_speed",
    "mean_wind_dir": "wind_direction",
    "mean_relative_hum": "relative_humidity",
    "mean_pressure": "pressure",
}

# Maps pandas frequency aliases to DMI timeResolution values.
# Covers current aliases (pandas >= 2.2)
TIME_RESOLUTION_MAP = {
    # Hourly
    "h": "hour",
    "1h": "hour",
    # Daily
    "D": "day",
    "d": "day",
    "1D": "day",
    "1d": "day",
    # Monthly
    "MS": "month",  # month start
    "1MS": "month",
    # Yearly
    "YS": "year",  # year start
    "y": "year",
    "1YS": "year",
    "1y": "year",
}


def _to_utc_timestamp(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def _format_datetime_interval(start, end) -> str:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (
        _to_utc_timestamp(start).strftime(fmt)
        + "/"
        + _to_utc_timestamp(end).strftime(fmt)
    )


def _raise_for_status(res):
    # Custom raise for status function which correctly returns error message
    try:
        res.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f"{e} | Response body: {res.text}") from e


def get_dmi_station_meta(
    station: str,
    entry_no: int = -1,
    url: str = URL,
    **kwargs,
) -> dict:
    """
    Retrieve metadata for a DMI climate station.

    Parameters
    ----------
    station : str
        DMI station identifier, e.g. ``'06180'`` for Copenhagen Airport.
    entry_no : int, default -1
        Index into the list of station entries returned by the API. The
        default of ``-1`` selects the most recent entry, which is appropriate
        for stations that have been relocated over time.
    url : str, optional
        Base URL for the DMI Climate Data API.
    **kwargs
        Additional keyword arguments forwarded to :func:`requests.get`,
        e.g. ``timeout=30``.

    Returns
    -------
    meta : dict
        Station metadata with keys ``'station_id'``, ``'name'``,
        ``'latitude'``, ``'longitude'``, ``'altitude'``, and ``'country'``.

    Notes
    -----
    The DMI Climate Data API is documented at
    https://www.dmi.dk/friedata/dokumentation/apis/climate-data-api-1.
    A list of stations can be found at
    https://www.dmi.dk/friedata/dokumentation/data/climate-data-stations.

    Examples
    --------
    >>> import solarpy
    >>> meta = solarpy.iotools.get_dmi_station_meta('06188', timeout=30)
    """
    params: dict = {"stationId": station}
    res = requests.get(url + "collections/station/items", params=params, **kwargs)
    _raise_for_status(res)
    body = res.json()

    features = body.get("features", [])
    if features == []:
        raise ValueError(f"No metadata was found for station '{station}'.")
    feat = features[entry_no]
    meta = feat["properties"]

    props = feat.get("properties", {})
    coords = feat.get("geometry", {}).get("coordinates", [None, None])
    meta["longitude"] = coords[0]
    meta["latitude"] = coords[1]
    meta["altitude"] = props.get("stationHeight")
    meta["country"] = {"GRL": "Greenland", "DNK": "Denmark"}.get(
        meta["country"], meta["country"]
    )

    return meta


def _fetch_dmi_data(
    station: str,
    datetime_interval: str,
    parameter_id: str | None,
    time_resolution: str,
    url: str,
    **kwargs,
) -> list[dict]:
    """Fetch all pages for a single parameter (or all parameters if None)."""
    params: dict = {
        "stationId": station,
        "datetime": datetime_interval,
        "timeResolution": time_resolution,
        "limit": LIMIT,
    }
    if parameter_id != [None]:
        params["parameterId"] = parameter_id

    endpoint = url + "collections/stationValue/items"
    records: list[dict] = []
    offset = 0

    while True:
        params["offset"] = offset
        res = requests.get(endpoint, params=params, **kwargs)
        _raise_for_status(res)
        body = res.json()
        for feat in body.get("features", []):
            props = feat["properties"]
            records.append(
                {
                    "timestamp": props["from"],
                    "parameterId": props["parameterId"],
                    "value": props["value"],
                }
            )
        if body.get("numberReturned", 0) < LIMIT:
            break
        offset += LIMIT

    return records


def get_dmi_climate_station_data(
    station: str,
    start,
    end,
    parameters: str | list[str] | None = None,
    time_resolution: str = "hour",
    map_variables: bool = True,
    url: str = URL,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """
    Retrieve data from DMI's Climate Data API.

    The Danish Meteorological Institute (DMI) operates automatic
    weather stations in Denmark and Greenland.

    Parameters
    ----------
    station : str
        DMI station identifier, e.g. ``'06180'`` for Copenhagen Airport.
    start : datetime-like
        First timestamp of the requested period (inclusive). Timezone-naive
        values are assumed to be UTC.
    end : datetime-like
        Last timestamp of the requested period (inclusive). Timezone-naive
        values are assumed to be UTC.
    parameters : str or list of str, optional
        DMI parameter identifiers to retrieve, e.g. ``'mean_temp'`` or
        ``['mean_temp', 'mean_wind_speed']``. If no value is passed, all
        available parameters for the station are returned. Note, that the
        parameter naming convention differs from DMI's observation data API.
    time_resolution : str, default ``'hour'``
        Temporal resolution of the data. DMI climate data supports ``'hour'``,
        ``'day'``, ``'month'``, and ``'year'``. Most standard pandas frequency
        aliases (e.g. ``'h'``, ``'D'``, ``'MS'``, ``'YS'``) are
        also accepted and mapped via :data:`TIME_RESOLUTION_MAP`.
    map_variables : bool, default True
        Whether to rename column names from DMI parameter IDs to
        standard pvlib variable names. Parameters without a mapping are
        not renamed.
    url : str, optional
        Base URL for the DMI Climate Data API.
    **kwargs
        Additional keyword arguments forwarded to :func:`requests.get`,
        e.g. ``timeout=30``.

    Returns
    -------
    data : pd.DataFrame
        Time series with a :class:`~pandas.DatetimeIndex`. For hourly data
        the timezone is set to UTC.
    meta : dict
        Station metadata with keys ``'station_id'``, ``'name'``,
        ``'latitude'``, ``'longitude'``, ``'altitude'``, and ``'country'``.

    Notes
    -----
    The DMI Climate Data API is documented at
    https://www.dmi.dk/friedata/dokumentation/apis/climate-data-api-1.
    Data availability and available parameters vary by station.

    A list of stations can be found here:
    https://www.dmi.dk/friedata/dokumentation/data/climate-data-stations.

    Examples
    --------
    Retrieve hourly measured mean temperature and irradiance for
    the Sjælsmark station north of Copenhagen:

    >>> import solarpy
    >>> data, meta = solarpy.iotools.get_dmi_climate_station_data(
    ...     station='06188',  # Sjælsmark station id
    ...     start='2023-06-01',
    ...     end='2023-06-30',
    ...     parameters=['mean_temp', 'mean_radiation'],
    ...     timeout=30,
    ... )
    """
    datetime_interval = _format_datetime_interval(start, end)
    time_resolution = TIME_RESOLUTION_MAP.get(time_resolution, time_resolution)

    if parameters is None or isinstance(parameters, str):
        parameters = [parameters]
    # allow for passing in standard pvlib/solarpy names
    reverse_variable_map = {v: k for k, v in VARIABLE_MAP.items()}
    parameters = [reverse_variable_map.get(p, p) for p in parameters]

    records: list[dict] = []
    for pid in parameters:
        records.extend(
            _fetch_dmi_data(
                station,
                datetime_interval,
                pid,
                time_resolution,
                url,
                **kwargs,
            )
        )

    if records:

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        data = df.pivot_table(
            index="timestamp", columns="parameterId", values="value", aggfunc="first"
        )

        if map_variables:
            data = data.rename(columns=VARIABLE_MAP)

    else:
        data = pd.DataFrame()

    meta = get_dmi_station_meta(station, url=url, **kwargs)

    return data, meta
