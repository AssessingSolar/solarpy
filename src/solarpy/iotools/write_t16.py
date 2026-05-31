import pandas as pd


# %%
def write_t16(filename, data, meta, lines=None, missing=None, encoding="utf-8"):
    """Write a time series DataFrame to a file in the IEA PVPS T16 format.

    Parameters
    ----------
    filename : str or path-like
        Path to the output file.
    data : pandas.DataFrame
        Time series data with a timezone-aware ``DatetimeIndex``; the index
        is converted to UTC before extracting Year/Month/Day/Hour/Minute.
        Column names are written as-is to the header row.
    meta : dict
        Station metadata. Recognised keys (either short or long form):

        - ``"latitude"`` or ``"latitude deg N"``
        - ``"longitude"`` or ``"longitude deg E"``
        - ``"altitude"`` or ``"altitude in m amsl"``
        - ``"stationcode"`` *(optional)*
    lines : list of str or None, default ``None``
        Additional lines written to the header after the location metadata,
        each prefixed with ``"# "``. If ``None``, no extra lines are written.
    missing : str or None, default ``None``
        String written in place of missing (NaN) values. ``None`` leaves the
        cell empty.
    encoding : str, default ``"utf-8"``
        File encoding passed to ``open()``.
    """
    latitude = meta.get("latitude", meta.get("latitude deg N"))
    longitude = meta.get("longitude", meta.get("longitude deg E"))
    altitude = meta.get("altitude", meta.get("altitude in m amsl"))
    stationcode = meta.get("stationcode", "")

    utc_index = data.index.tz_convert("UTC")
    datetime_cols = pd.DataFrame(
        {
            "Year": utc_index.year,
            "Month": utc_index.month,
            "Day": utc_index.day,
            "Hour": utc_index.hour,
            "Minute": utc_index.minute,
        },
        index=data.index,
    )

    # Add time columns and drop existing time columns if present
    _datetime_col_names = ["Year", "Month", "Day", "Hour", "Minute"]
    out = pd.concat(
        [datetime_cols, data.drop(columns=_datetime_col_names, errors="ignore")], axis=1
    )

    missing_ref = "empty cells" if missing is None else missing

    with open(filename, "w", encoding=encoding) as f:
        f.write(f"# stationcode {stationcode}\n")
        f.write(f"# latitude deg N {latitude}\n")
        f.write(f"# longitude deg E {longitude}\n")
        f.write(f"# altitude in m amsl {altitude}\n")
        f.write("# \n")
        f.write(
            "# time stamp (Year Month Day Hour Minute) is in UTC and refers to the end of the period\n"  # noqa: E501
        )
        f.write(
            f"# missing data points in GHI DNI and DIF are noted as {missing_ref}\n"
        )
        for line in lines or []:
            f.write(f"# {line}\n")
        na_rep = "" if missing is None else missing
        out.to_csv(f, index=False, na_rep=na_rep, lineterminator="\n")
