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

## `bulk load` - load generated file lists

Load many CSV or Parquet files from a Python list. With one alias, Pivotal
concatenates the files into one table and adds a `source` column containing the
input filename.

```python
files = ["data/jan.csv", "data/feb.csv", "data/mar.csv"]
```

```pivotal
bulk load :files as sales
```

The concat mode unions columns by name, so files with extra or missing columns
are combined with missing values where needed.

Customize the provenance column:

```pivotal
bulk load :files as sales
    source column file
    source value stem
```

`source value` can be `filename`, `path`, or `stem`.

Use multiple aliases to load each file separately:

```pivotal
bulk load :files as jan_sales, feb_sales, mar_sales
```

Or provide aliases from a Python list:

```python
tables = ["jan_sales", "feb_sales", "mar_sales"]
```

```pivotal
bulk load :files as :tables
```

The file list and alias list must have the same length. Aliases must be valid
Pivotal table identifiers.

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

## `from` — load from a database

Connect to a database and load one or more tables or query results into named DataFrames.

```pivotal
from "<database_path_or_uri>"
    load <sql_table> as <df_name>, ...
    query "<SQL string>" as <df_name>
```

### Load tables by name

```pivotal
from "data/mydb.sqlite"
    load orders as orders, customers as customers
```

Each `load` item reads the named SQL table with `SELECT * FROM <table>`. Multiple tables can be listed on a single line, comma-separated.

### Run arbitrary SQL

```pivotal
from "data/mydb.sqlite"
    query "SELECT id, name FROM customers WHERE active = 1" as active_customers
```

### Mix `load` and `query` in one block

```pivotal
from "data/mydb.sqlite"
    load orders as orders_raw
    query "SELECT category, SUM(amount) as total FROM orders GROUP BY category" as summary
```

The connection is opened once and all reads share it.

### Supported sources

| Source | Example path | Notes |
|--------|-------------|-------|
| SQLite | `"data/mydb.sqlite"` | Also `.db`, `.sqlite3` |
| DuckDB | `"data/warehouse.duckdb"` | Also `.ddb` |
| Python variable | `:conn_str` | Variable must hold a path or URI |
| PostgreSQL URI | `"postgresql://user:pass@host/db"` | Requires SQLAlchemy — see below |
| MySQL URI | `"mysql+pymysql://user:pass@host/db"` | Requires SQLAlchemy + driver |
| Other SQLAlchemy | any `dialect+driver://...` URI | Requires SQLAlchemy + driver |

### SQLAlchemy support (PostgreSQL, MySQL, and others)

For databases other than SQLite and DuckDB, Pivotal uses [SQLAlchemy](https://www.sqlalchemy.org/) as the connection layer. SQLAlchemy is a Python library that provides a unified interface to many databases via connection URI strings.

**Install SQLAlchemy and a driver:**

```bash
# PostgreSQL
pip install sqlalchemy psycopg2-binary

# MySQL
pip install sqlalchemy pymysql

# Microsoft SQL Server
pip install sqlalchemy pyodbc
```

**Usage:**

```pivotal
from "postgresql://user:password@localhost:5432/mydb"
    load products as products
    query "SELECT * FROM orders WHERE status = 'open'" as open_orders
```

If SQLAlchemy is not installed, Pivotal will raise a clear error with the install command rather than a cryptic import failure.

> **Polars backend**: Uses `connectorx` for URI sources if available (faster), falling back to SQLAlchemy + `pl.from_pandas()`. Install with `pip install connectorx`.
>
> **DuckDB backend**: Uses DuckDB's native extensions (`postgres_scanner`, etc.) for URI sources rather than SQLAlchemy.

---

## `delete` — remove a table

Remove a table from memory:

```pivotal
delete sales
delete temp_table
```

In Jupyter, this is equivalent to `del sales` in Python.


Tables must have compatible columns.
