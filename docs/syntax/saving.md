# Saving

## `save` — export a data package

Export tables and charts to a self-contained data package (a directory of files).

### Basic save

```pivotal
save "my_analysis"
```

Saves all tables and charts in the current session to a directory named `my_analysis`.

### Options

```pivotal
save "my_analysis"
    path "~/projects/output"        # output directory (default: current dir)
    format parquet                   # file format: csv (default) or parquet
    chart_format svg                 # chart format: png (default) or svg
    include sales, summary, chart1  # only save these objects
    exclude raw, debug_plot          # save everything except these
```

| Option | Description |
|--------|-------------|
| `path "<dir>"` | Output directory path |
| `format <fmt>` | Data format: `csv` (default) or `parquet` |
| `chart_format <fmt>` | Chart image format: `png` (default) or `svg` |
| `include <names>` | Comma-separated list of objects to include |
| `exclude <names>` | Comma-separated list of objects to exclude |

### Loading a saved package

Load all tables from a previously saved package:

```pivotal
python import pivotal; _pivotal_pkg = pivotal.Package.open("my_analysis")
```

Then load tables from Python:

```python
tables = _pivotal_pkg.load_all()
sales = _pivotal_pkg.load_table("sales")
```

---

## `delete` — remove a table

Remove a table from the current session to free memory:

```pivotal
delete raw_import
delete scratch
```

In Jupyter, equivalent to `del raw_import` in Python.
