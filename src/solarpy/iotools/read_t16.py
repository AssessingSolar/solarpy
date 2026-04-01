import numpy as np
import pandas as pd

VARIABLE_MAP = {
    "GHI": "ghi",
    "DIF": "dhi",
    "DNI": "dni",
    "GHI_clear": "ghi_clear",
    "DIF_clear": "dhi_clear",
    "DNI_clear": "dni_clear",
    "BHI_clear": "bhi_clear",
    "GHI_extra": "ghi_extra",
    "DNI_extra": "dni_extra",
}


def read_t16(filename, drop_dates=False, map_variables=False, encoding="utf-8"):
    """
    Read a time series data file following the IEA PVPS T16 file format.

    Parameters
    ----------
    filename : str or path-like
        Filename to the data file
    drop_dates : bool, default False
        If True, drop the Year, Month, Day, Hour, and Minute columns from the
        returned DataFrame. The datetime index is always set regardless.
    map_variables : bool, default False
        If True, renames columns of the DataFrame to pvlib variable names
        (GHI→ghi, DIF→dhi, DNI→dni) and shortens metadata keys to
        ``latitude``, ``longitude``, and ``altitude``.
    encoding : str, default 'utf-8'
        Encoding of the file passed to ``open()``.

    Returns
    -------
    data : pd.DataFrame
        Time series data
    meta : dict
        Metadata

    """
    meta = {
        "stationcode": None,
        "latitude deg N": np.nan,
        "longitude deg E": np.nan,
        "altitude in m amsl": np.nan,
    }

    with open(filename, "r", encoding=encoding) as fbuf:
        # Parse through initial metadata section (lines starting with #)
        while True:
            line = fbuf.readline().strip()

            # lines with metadata
            if line.startswith("#"):
                line = line.lstrip("#")
            # line with column names
            else:
                names = line.split(",")
                break

            for mk in meta.keys():
                if mk in line:
                    meta[mk] = line.replace(mk, "").replace(",", "").strip()
                    # Fix issue in older files
                    meta[mk] = meta[mk].replace("longitude deg N", "longitude deg E")

        for k in [
            "latitude deg N",
            "longitude deg E",
            "altitude in m amsl",
        ]:
            # convert to float or int
            meta[k] = float(meta[k]) if "." in meta[k] else int(meta[k])
        if meta["stationcode"] == "":
            meta["stationcode"] = None

        data = pd.read_csv(
            fbuf,
            sep=",",
            names=names,
            na_values=[-999, -9999, "NAN"],
            # Both naming conventions of comment occur
            dtype={"comments": str, "Comments": str, "Remarks": str},
        )

    datetime_columns = ["Year", "Month", "Day", "Hour", "Minute"]

    data.index = pd.to_datetime(
        data[datetime_columns].rename(columns=str.lower)
    ).dt.tz_localize("UTC")

    if drop_dates:
        data = data.drop(columns=datetime_columns)

    if map_variables:
        meta["latitude"] = meta.pop("latitude deg N")
        meta["longitude"] = meta.pop("longitude deg E")
        meta["altitude"] = meta.pop("altitude in m amsl")
        data = data.rename(columns=VARIABLE_MAP)

    return data, meta
