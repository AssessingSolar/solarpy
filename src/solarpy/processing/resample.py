import pandas as pd


def resample_to_freq(data, freq, full_days=True, verbose=True):
    """Resample a time series to a regular frequency by reindexing.

    Creates a complete, evenly spaced DatetimeIndex at the given frequency and
    reindexes the input data to it. Timesteps not present in the original data
    are filled with NaN; original timesteps that do not align with the new
    index are discarded.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        Time series with a DatetimeIndex.
    freq : str or pandas offset alias
        Target frequency (e.g. ``"1min"``, ``"10min"``, ``"1h"``).
    full_days : bool, default True
        If True, extend the resampled index to cover the full first and last
        calendar days (00:00:00 to 23:59:59) rather than stopping at the first
        and last timestamps in ``data``.
    verbose : bool, default True
        If True, print the number and percentage of timesteps added and
        discarded during resampling.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        Data reindexed to the new regular DatetimeIndex. Missing values are
        NaN.
    """
    start, end = data.index.min(), data.index.max()
    if full_days:  # ensure all timesteps in the start and end day are present
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end.replace(hour=23, minute=59, second=59, microsecond=999999)

    full_index = pd.date_range(start, end, freq=freq)

    if verbose:
        n_total = len(full_index)
        n_added = len(full_index.difference(data.index))
        n_discarded = len(data.index.difference(full_index))
        print(
            f"Resampled: {n_added} timesteps added ({n_added/n_total*100:2.1f}%), "
            f"{n_discarded} discarded ({n_discarded/n_added*100:2.1f}%)"
        )

    data = data.reindex(full_index)

    return data
