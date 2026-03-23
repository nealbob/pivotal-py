# Data Sources

## `load` — load a file

Load a CSV, Excel, or Parquet file into a named table.

```
load <table_name> "<file_path>"
```

```
load sales "data/sales.csv"
load products "catalog.xlsx"
load transactions "archive.parquet"
```

Use a Python variable for the path:

```
load data :my_file_path
```

### Options

Indent options beneath the `load` statement:

```
load sales "data/sales.csv"
    header 0           # row index of header (default 0)
    names ["product", "quantity", "price"]  # override column names
```

| Option | Description |
|--------|-------------|
| `header <n>` | Row index to use as column headers (default `0`) |
| `names [...]` | List of column names to use instead of the file header |

---

## `df` — set or create a table

### Set active table

Make an existing table the active table for subsequent operations:

```
df sales
    filter price > 100
    sort price desc
```

### Create a derived table

Create a new table by applying operations to an existing one. The original is unchanged:

```
df top_sales from sales
    filter revenue > 1000
    sort revenue desc
```

```
df summary from orders
    group by region
        agg sum amount as total
```

The `from <table>` clause is optional. Without it, operations apply to the named table in-place:

```
df sales              # operates on 'sales', modifying it
    filter active == True
```

With `from`, a new table is created:

```
df active_sales from sales    # creates 'active_sales', 'sales' unchanged
    filter active == True
```

### Chaining

Multiple `df` blocks can be chained — the output of one becomes the input of the next:

```
load raw "data.csv"

df cleaned from raw
    dropna price, quantity
    fillna 0

df summary from cleaned
    group by category
        agg sum price as total
    sort total desc
```

---

## `delete` — remove a table

Remove a table from memory:

```
delete sales
delete temp_table
```

In Jupyter, this is equivalent to `del sales` in Python.

---

## `concat` — stack tables vertically

Append rows from one table onto another:

```
df all_sales from jan
    concat feb
    concat mar
```

Append multiple tables at once:

```
df all_sales from q1
    concat q2, q3, q4
```

Tables must have compatible columns.
