# Data Sources

## `load` — load a file

Load a CSV, Excel, or Parquet file into a named table.

```pivotal
load "<file_path>" as <table_name>
```

```pivotal
load "data/sales.csv" as sales
load "catalog.xlsx" as products
load "archive.parquet" as transactions
```

Use a Python variable for the path:

```pivotal
load :my_file_path as data
```

### Options

Indent options beneath the `load` statement:

```pivotal
load "data/sales.csv" as sales
    header 0           # row index of header (default 0)
    names ["product", "quantity", "price"]  # override column names
```

| Option | Description |
|--------|-------------|
| `header <n>` | Row index to use as column headers (default `0`) |
| `names [...]` | List of column names to use instead of the file header |

---

## `with` — set or create a table

### Set active table

Make an existing table the active table for subsequent operations:

```pivotal
with sales
    filter price > 100
    sort price desc
```

### Create a derived table

Create a new table by applying operations to an existing one. The original is unchanged:

```pivotal
with sales as top_sales
    filter revenue > 1000
    sort revenue desc
```

```pivotal
with orders as summary
    group by region
        agg sum amount as total
```

Without `as`, operations apply to the named table in-place:

```pivotal
with sales              # operates on 'sales', modifying it
    filter active == True
```

With `as`, a new table is created:

```pivotal
with sales as active_sales    # creates 'active_sales', 'sales' unchanged
    filter active == True
```

### Chaining

Multiple `with` blocks can be chained — the output of one becomes the input of the next:

```pivotal
load "data.csv" as raw

with raw as cleaned
    dropna price, quantity
    fillna 0

with cleaned as summary
    group by category
        agg sum price as total
    sort total desc
```

---

## `delete` — remove a table

Remove a table from memory:

```pivotal
delete sales
delete temp_table
```

In Jupyter, this is equivalent to `del sales` in Python.


Tables must have compatible columns.
