import marimo

__generated_with = "0.23.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Interactive selection of data!

    An example of interactive selection of data in Python using Marimo notebooks in the browser.
    """)
    return


@app.cell
def _(np, pvlib, solarpy):
    data, meta = solarpy.iotools.read_t16(
        "https://raw.githubusercontent.com/AssessingSolar/solarpy/refs/heads/main/data/LYN_2023.csv",  # noqa: E501
        map_variables=True,
    )

    # Calculate solar position
    solar_position = pvlib.solarposition.get_solarposition(
        data.index, latitude=meta["latitude"], longitude=meta["longitude"]
    )

    # Calculate extraterrestrial irradiance on a horizontal plane
    dni_extra = pvlib.irradiance.get_extra_radiation(data.index)
    cos_sza = np.cos(np.deg2rad(solar_position["apparent_zenith"])).clip(lower=0)
    ghi_extra = dni_extra * cos_sza
    return data, ghi_extra


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the below plot, you can select a cluster of points to see which dates
    """)
    return


@app.cell
def _(data, ghi_extra, mo, plt, solarpy):
    x, y = ghi_extra, data["ghi"]

    fig, ax = solarpy.plotting.plot_bsrn_limits(
        irradiance=data["ghi"],
        component="ghi",
        ghi_extra=ghi_extra,
    )

    fig.tight_layout()

    ax = mo.ui.matplotlib(plt.gca())
    ax
    return ax, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Selected points can be seen below
    """)
    return


@app.cell
def _(ax, data, x, y):
    mask = ax.value.get_mask(x, y)
    value_counts = data.index[mask].to_series().dt.date.value_counts()
    value_counts
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pvlib
    import solarpy

    return np, plt, pvlib, solarpy


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
