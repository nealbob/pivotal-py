# Pivotal

Pivotal is a data analysis language for Python.  It offers a concise syntax for common data operations that compiles to Pandas, Polars or DuckDB code.  With comprehensive JupyterLab and VS Code support (syntax highlighting, autocomplete, interactive viewer and GUI controls) Pivotal provides a friendly entry point to the Python data ecosystem.

=== "Pivotal"

    ```pivotal
    load "daily_climate.csv" as climate
    load "crop_data.csv" as crops
    
    python grow_min = 8; grow_max = 32; crop_season = [4, 10]

    with climate as climate_features
        year = year(date)
        month = month(date)
        filter month between :crop_season
        grow_degrees = (max_temp + min_temp) / 2 - :grow_min
            where max_temp < :grow_max and min_temp >= :grow_min
            else 0
        group by year, region
            agg sum grow_degrees as gdd, sum rain as grow_rain

    with crops as training_data
        filter crop_type == "wheat" and area > 0
        yield = production / area
        inner merge climate_features on year, region
        select year, area, yield, gdd, grow_rain, region

    python
        from sklearn.linear_model import LinearRegression
        X = training_data[["year", "area", "gdd", "grow_rain"]]
        training_data["yield_hat"] = LinearRegression().fit(X, training_data["yield"]).predict(X)
    end

    with training_data
        pivot plot line nat_pred_vs_actual
            x year "Year"
            y wmean yield area, wmean yield_hat area "Wheat yield (t/ha)"
    ```

=== "Python"

    ```python
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    grow_min = 8; grow_max = 32; crop_season = [4, 10]

    climate = pd.read_csv("daily_climate.csv")
    crops = pd.read_csv("crop_data.csv")

    climate_features = climate.copy()
    climate_features["year"] = pd.to_datetime(climate_features["date"]).dt.year
    climate_features["month"] = pd.to_datetime(climate_features["date"]).dt.month
    climate_features = climate_features[
        climate_features["month"].between(crop_season[0], crop_season[1])
    ].copy()

    climate_features["grow_degrees"] = 0.0
    mask = (
        (climate_features["max_temp"] < grow_max)
        & (climate_features["min_temp"] >= grow_min)
    )
    climate_features.loc[mask, "grow_degrees"] = (
        (climate_features.loc[mask, "max_temp"] + climate_features.loc[mask, "min_temp"]) / 2
        - grow_min
    )

    climate_features = (
        climate_features
        .groupby(["year", "region"], as_index=False)
        .agg(
            gdd=("grow_degrees", "sum"),
            grow_rain=("rain", "sum"),
        )
    )

    training_data = crops.copy()
    training_data = training_data[
        (training_data["crop_type"] == "wheat") & (training_data["area"] > 0)
    ].copy()
    training_data["yield"] = training_data["production"] / training_data["area"]

    training_data = (
        training_data
        .merge(climate_features, on=["year", "region"], how="inner")
        [["year", "area", "yield", "gdd", "grow_rain", "region"]]
        .copy()
    )

    X = training_data[["year", "area", "gdd", "grow_rain"]]
    training_data["yield_hat"] = LinearRegression().fit(X, training_data["yield"]).predict(X)

    plot_data = (
        training_data
        .groupby("year", as_index=False)
        .apply(
            lambda g: pd.Series({
                "yield": (g["yield"] * g["area"]).sum() / g["area"].sum(),
                "yield_hat": (g["yield_hat"] * g["area"]).sum() / g["area"].sum(),
            })
        )
        .reset_index(drop=True)
    )

    plot_data.plot(x="year", y=["yield", "yield_hat"], kind="line", xlabel="Year", ylabel="Wheat yield (t/ha)")
    ```
---

A live-demo of [Pivotal in Jupyter Lab](https://mybinder.org/v2/gh/nealbob/pivotal-demo/HEAD?urlpath=%2Fdoc%2Ftree%2Ffootball_demo.ipynb) is available via Binder:

[![JupyterLab demo](assets/piv2.png)](https://mybinder.org/v2/gh/nealbob/pivotal-demo/HEAD?urlpath=%2Fdoc%2Ftree%2Ffootball_demo.ipynb)

## Key features

- **Readable, Writable syntax** — write data transformations in a simple declarative syntax
- **Multiple backends** — run the same code as Pandas, Polars or DuckDB/SQL

**JupyterLab and VS Code integration** — syntax highlighting, autocomplete, `%%pivotal` cell magic, interactive object viewer and explorer,  GUI controls

- **Export to code** — compile any notebook or `.pivotal` file to `.py` or `.sql`
- **Plotting & tables** — built-in chart and publication-ready table support
- **Data packages** — export all output (DataFrames, charts, tables) to a single [Frictionless](https://specs.frictionlessdata.io/) data package
  
## Install

```bash
pip install pivotal
```

This installs the full feature set — Pandas, Polars, DuckDB, Great Tables.

For a minimal Pandas-only install:

```bash
pip install --no-deps pivotal
pip install lark pandas matplotlib

## Quick example

=== "Jupyter"

    ```python
    %%pivotal
    load "orders.csv" as orders

    with orders as monthly
        filter status == "complete"
        assign month = left(date, 7)
        group by month
            agg sum amount as revenue, count id as n_orders
        sort month
    ```

=== "Python API"

    ```python
    from pivotal import DSLParser

    parser = DSLParser()
    parser.execute("""
    load "orders.csv" as orders

    with orders as monthly
        filter status == "complete"
        assign month = left(date, 7)
        group by month
            agg sum amount as revenue, count id as n_orders
        sort month
    """)
    ```

=== ".pivotal file"

    ```
    # monthly_report.pivotal
    load "orders.csv" as orders

    with orders as monthly
        filter status == "complete"
        assign month = left(date, 7)
        group by month
            agg sum amount as revenue, count id as n_orders
        sort month
    ```

    ```bash
    pivotal monthly_report.pivotal
    ```

## Next steps

- [Getting Started](getting-started.md) — installation and first steps
- [Syntax Reference](syntax/index.md) — complete DSL documentation
- [Backends](backends.md) — pandas, Polars, DuckDB, and SQL
- [JupyterLab](jupyter.md) — cell magic, viewer, and export
- [VS Code](vscode.md) — editor integration
