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
    # Horizon line example
    """)
    return


@app.cell
def _(mo):
    latitude = mo.ui.number(start=-90, stop=90, value=55.79, step=0.01, label="Latitude [°N]")
    longitude = mo.ui.number(start=-180, stop=180, value=12.53, step=0.01, label="Longitude [°E]")
    altitude = mo.ui.number(start=-10, stop=8000, value=0, step=0.1, label="Altitude [m]")
    ground_offset = mo.ui.number(start=-100, stop=200, value=0, step=0.1, label="Ground offset [m]")

    mo.vstack([
        mo.md("Select location of interest"),
        latitude,
        longitude,
        altitude,
        ground_offset,
    ])
    return altitude, ground_offset, latitude, longitude


@app.cell
def _():
    import matplotlib.pyplot as plt
    import solarpy

    return plt, solarpy


@app.cell
def _(altitude, ground_offset, latitude, longitude, solarpy):
    horizon, meta = solarpy.horizon.get_horizon_mines(
        latitude=latitude.value,
        longitude=longitude.value,
        altitude=altitude.value,
        ground_offset=ground_offset.value,
    )
    return (horizon,)


@app.cell
def _(horizon, plt):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(horizon.index, horizon.values)
    ax.set_xlabel("Azimuth")
    ax.set_ylabel("Elevation [°]")
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 90))
    # ax.set_yticklabels([f"{t:2.1f}°" for t in ax.get_yticks()])
    ax.set_xticklabels(
        ["N\n0°", "E\n90°", "S\n180°", "W\n270°", "N\n360°"],
        ha='center')
    ax.axhline(0, c='k', lw=3, zorder=-2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
