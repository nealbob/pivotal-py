# Saving

## Save a table to a file

```pivotal
save sales as "outputs/sales.csv"
save summary as "outputs/summary.parquet"
```

The file format is inferred from the destination suffix. Pandas supports CSV,
Parquet, JSON Lines, Feather, Excel, and pickle. Polars and DuckDB support the
formats available through their backends.

Spark DataFrames can also write Delta destinations:

```pivotal
save sales as "dbfs:/mnt/results/sales.delta"
```

## Save a managed table

```pivotal
save sales as table "main.analytics.sales"
```

Managed table saves use the active backend. Spark DataFrames use
`saveAsTable`; DuckDB and SQL compilation use `CREATE OR REPLACE TABLE`.

## Save a data package

```pivotal
save package as "outputs/my_analysis"
```

This saves all tables, charts, rendered tables, and Pivotal parameters in the
current session.

```pivotal
save package as "~/projects/output/my_analysis"
    format parquet
    chart_format svg
    include sales, summary, chart1
    exclude raw, debug_plot
```

| Option | Description |
|--------|-------------|
| `format <fmt>` | Table format: `csv` (default) or `parquet` |
| `chart_format <fmt>` | Chart image format: `png` (default) or `svg` |
| `include <names>` | Comma-separated list of objects to include |
| `exclude <names>` | Comma-separated list of objects to exclude |

The earlier `save "<package_name>"` form remains supported for compatibility,
but new code should use `save package as "<path>"`.

## Loading a saved package

```pivotal
python import pivotal; _pivotal_pkg = pivotal.Package.open("my_analysis")
```

Then load package tables from Python:

```python
tables = _pivotal_pkg.load_all()
sales = _pivotal_pkg.load_table("sales")
```

---

## `delete` - remove a table

Remove a table from the current session to free memory:

```pivotal
delete raw_import
delete scratch
```

In Jupyter, this is equivalent to `del raw_import` in Python.
