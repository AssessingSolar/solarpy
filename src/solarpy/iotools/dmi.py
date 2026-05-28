from __future__ import annotations

import pandas as pd
import requests

URL = "https://opendataapi.dmi.dk/v2/climateData/"
LIMIT = 300_000

# Maps DMI parameter IDs to pvlib/solarpy standard variable names.
VARIABLE_MAP = {
    "radia_glob": "ghi",
    "mean_temp": "temp_air",
    "mean_wind_speed": "wind_speed",
    "mean_wind_dir": "wind_direction",
    "mean_relative_hum": "relative_humidity",
    "mean_pressure": "pressure",
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


def _fetch_station_meta(
    station_id: str, url: str, api_key: str | None, **kwargs
) -> dict:
    params: dict = {"stationId": station_id}
    if api_key is not None:
        params["api-key"] = api_key
    res = requests.get(url + "collections/station/items", params=params, **kwargs)
    res.raise_for_status()
    body = res.json()

    meta: dict = {
        "station_id": station_id,
        "name": None,
        "latitude": None,
        "longitude": None,
        "altitude": None,
        "country": None,
    }
    features = body.get("features", [])
    if features:
        feat = features[0]
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [None, None])
        meta["longitude"] = coords[0]
        meta["latitude"] = coords[1]
        meta["name"] = props.get("name")
        meta["country"] = props.get("country")
        meta["altitude"] = props.get("stationHeight")
    return meta


def _fetch_parameter(
    station_id: str,
    datetime_interval: str,
    parameter_id: str | None,
    time_resolution: str,
    url: str,
    api_key: str | None,
    **kwargs,
) -> list[dict]:
    """Fetch all pages for a single parameter (or all parameters if None)."""
    params: dict = {
        "stationId": station_id,
        "datetime": datetime_interval,
        "timeResolution": time_resolution,
        "limit": LIMIT,
    }
    if parameter_id is not None:
        params["parameterId"] = parameter_id
    if api_key is not None:
        params["api-key"] = api_key

    endpoint = url + "collections/stationValue/items"
    records: list[dict] = []
    offset = 0

    while True:
        params["offset"] = offset
        res = requests.get(endpoint, params=params, **kwargs)
        res.raise_for_status()
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


def get_dmi_climate_data(
    station: str,
    start,
    end,
    parameters: str | list[str] | None = None,
    time_resolution: str = "hour",
    api_key: str | None = None,
    map_variables: bool = True,
    url: str = URL,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """
    Retrieve data from the DMI's Climate Data API.

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
    parameters : str or list of str or None, default None
        DMI parameter identifiers to retrieve, e.g. ``'mean_temp'`` or
        ``['mean_temp', 'mean_wind_speed']``. If ``None``, all available
        parameters for the station are returned.
    time_resolution : str, default ``'hour'``
        Temporal resolution of the data. Valid values include ``'hour'``,
        ``'day'``, ``'month'``, and ``'year'``.
    map_variables : bool, default True
        If ``True``, rename DataFrame columns from DMI parameter IDs to
        standard pvlib variable names. Parameters without a mapping are
        kept under their original names.
    url : str, optional
        Base URL for the DMI Climate Data API.
    **kwargs
        Additional keyword arguments forwarded to :func:`requests.get`,
        e.g. ``timeout=30``.

    Returns
    -------
    data : pd.DataFrame
        Time series with a UTC-aware :class:`~pandas.DatetimeIndex`. Each
        column corresponds to one climate parameter.
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
    Retrieve hourly mesaured mean temperature and wind speed for
    Copenhagen Airport:

    >>> import solarpy
    >>> import pandas as pd
    >>> data, meta = solarpy.iotools.get_dmi_climate_data(
    ...     station='06180',
    ...     start=pd.Timestamp('2023-06-01'),
    ...     end=pd.Timestamp('2023-06-30'),
    ...     parameters=['mean_temp', 'mean_wind_speed'],
    ...     timeout=30,
    ... )
    """
    datetime_interval = _format_datetime_interval(start, end)

    records: list[dict] = []
    for pid in parameters:
        records.extend(
            _fetch_parameter(
                station,
                datetime_interval,
                pid,
                time_resolution,
                url,
                api_key,
                **kwargs,
            )
        )

    meta = _fetch_station_meta(station, url, api_key, **kwargs)

    if not records:
        return pd.DataFrame(), meta

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    data = df.pivot_table(
        index="timestamp", columns="parameterId", values="value", aggfunc="first"
    )
    data.columns.name = None
    data.index.name = None

    if map_variables:
        data = data.rename(columns=VARIABLE_MAP)

    return data, meta
