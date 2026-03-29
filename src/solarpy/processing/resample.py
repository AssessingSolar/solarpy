import pandas as pd


def resample_to_freq(data, freq, full_days=True, verbose=True):
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
