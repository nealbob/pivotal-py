# Pivotal

<img src="pivotal_logo.svg" width="120">

**Pivotal** is a Python-based Domain-Specific Language (DSL) for data processing. It provides a clean, readable SQL-like syntax for common data operations which compiles to Python (pandas) code.

## Features

**Readable, Writable Syntax** - Write data transformations in a simple SQL-like language

**Pandas-Powered** - Compiles to pandas, integrates with python code

**Pipeline-Oriented** - Chain operations naturally with indentation blocks

**VS Code Integration** - Syntax highlighting, compile to Python or execute in interactive window

**JupyterLab Integration** - `%%pivotal` cell magic with syntax highlighting 

## At a Glance

The syntax of Pivotal has some similarites with "piped-SQL" varients including [PRQL](https://prql-lang.org), while replicating some aspects of Python/Pandas (i.e., indentation rather than brackets):

**Pivotal**

```
load invoices "invoices.csv"
load customers "customers.csv"

df invoices
    filter invoice_date >= "1970-01-16"
    assign transaction_fees = 0.8
    assign income = total - transaction_fees
    filter income > 1

df summary from invoices
    group by customer_id
        agg mean total, sum income as sum_income, count total as ct
    sort sum_income desc
    left merge customers on customer_id
    python summary["name"] = summary["last_name"] + ", " + summary["first_name"]
    select customer_id, name, sum_income
```

Note the `python` line above is Pivotal's escape hatch for expressions that fall outside the grammar — string formatting, custom functions, anything pandas can do.

---

## Table of Contents

- [Installation](#installation)
- [Editor Integrations](#editor-integrations)
- [Quick Start](#quick-start)
- [Language Syntax](#language-syntax)
  - [Loading Data](#loading-data)
  - [Table Operations](#table-operations)
  - [Filtering](#filtering)
  - [Selecting Columns](#selecting-columns)
  - [Creating/Modifying Columns](#creatingmodifying-columns)
  - [Sorting](#sorting)
  - [Grouping and Aggregation](#grouping-and-aggregation)
  - [Merging Tables](#merging-tables)
  - [Pivot Tables](#pivot-tables)
  - [Data Cleaning](#data-cleaning)
  - [Applying Python Functions](#applying-python-functions)
  - [Package Management](#package-management)
- [API Reference](#api-reference)
- [Examples](#examples)

---

---

## Installation

### Prerequisites
- Python 3.7+
- pandas
- lark-parser

### Install Dependencies

```bash
pip install pandas lark-parser
```

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/pivotal/pivotal-py
   cd pivotal-py
   ```
2. Install the core package:
   ```bash
   pip install .
   ```

### VS Code Extension

1. Install the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extensions for VS Code
2. Install the Pivotal extension from the VS Code Marketplace, or build it locally:
   ```bash
   cd editors/vscode
   npm install
   npm run build
   ```
   Then install the generated `.vsix` file via **Extensions → Install from VSIX**.

### JupyterLab Extension

```bash
cd editors/jupyterlab
jlpm install
jlpm run build
pip install -e .
```

Restart JupyterLab — the extension activates automatically.

---

## Editor Integrations

### VS Code

Open any `.pivotal` file in VS Code to get:

- **Syntax highlighting** for `.pivotal` files and `%%pivotal` blocks embedded in `.py` files
- **Execute File** (`Ctrl+F5` / `Cmd+F5`) — runs the file via `python -m pivotal` in the integrated terminal
- **Execute in Interactive Notebook** — sends the file to a VS Code Interactive Window as `%%pivotal` cells, with live DataFrame previews. Sections separated by `#%%` markers run as individual cells. The window opens to the right and is reused on subsequent runs.
- **Execute Selection** (`Ctrl+Shift+F5` / `Cmd+Shift+F5`) — sends the selected block to the Interactive Window
- **Compile to Python** — generates a `.py` file from the current `.pivotal` source and saves it alongside it

All commands are also available via the Command Palette (`Ctrl+Shift+P`).

### JupyterLab

Use the `%%pivotal` cell magic in any notebook:

```
%%pivotal
load sales "data/sales.csv"

df sales
filter amount > 1000
group by region
    agg sum amount as total
sort total desc
```

- Syntax highlighting activates automatically on any cell whose first line is `%%pivotal`
- Run the cell normally — results display as interactive DataFrames

---

## Quick Start

### In VS Code

1. Create a file named `analysis.pivotal`
2. Write your Pivotal code:
   ```
   load sales "data/sales.csv"

   df sales
   filter amount > 1000
   select customer_id, product, amount
   sort amount desc
   ```
3. Press `Ctrl+F5` to run in the terminal, or click **Execute in Interactive Notebook** in the title bar to see live DataFrame output

### In JupyterLab

Create a new notebook cell, set its first line to `%%pivotal`, and write your code:

```
%%pivotal
load sales "data/sales.csv"

df sales
filter amount > 1000
group by region
    agg sum amount as total
sort total desc
```

Run the cell — the resulting DataFrame displays inline.

### From the command line

```bash
python -m pivotal analysis.pivotal
```

### Programmatic API

```python
import pivotal

parser = pivotal.DSLParser()

dsl_code = """
load sales "sales_data.csv"

df sales
filter amount > 1000
select customer_id, product, amount
sort amount desc
"""

ns = {}
parser.execute(dsl_code, ns, verbose=False)
print(ns['sales'])
```

---

## Language Syntax

### Loading Data

Load data files into named tables. The file format is detected automatically from the extension:

```pivotal
# CSV
load sales "data/sales.csv"

# Excel
load budget "report.xlsx"

# Parquet
load events "events.parquet"

# With pandas reader options
load inventory "data/inventory_2024.csv"
    names ["product", "quantity", "price"]
    header 0

# From a runtime variable (path stored in a Python variable)
load df :my_path_variable
```

**Supported formats:** `.csv`, `.xlsx`, `.xls`, `.parquet`

**Parameters:** any keyword argument accepted by the underlying pandas reader (`read_csv`, `read_excel`, `read_parquet`).

**Package-based loading** (requires `_pivotal_pkg` to be set in the session — see [Package Management](#package-management)):

```pivotal
# Load a single named table from the active package's data/ folder
load clean

# Load all tables saved in the package at once
load all
```

---

### Table Operations

`df <name>` sets the active table for all following operations until the next `df` statement.

#### Set Active Table

```pivotal
# Work with an existing table
df sales
filter price > 100
select product, price
sort price desc
```

#### Create a Derived Table

```pivotal
# Copy from an existing table and work on the copy
df filtered_data from sales
filter price > 100
```


---

### Filtering

Filter rows based on conditions. Conditions go on the same line as `filter`:

```pivotal
# Comparison
df active_users from users
filter status == "active"

# Logical operators
df premium_sales from sales
filter amount > 1000 and category == "premium"

# Membership
df regional_data from sales
filter region in ["North", "South", "East"]

# Range (inclusive)
df mid_range from sales
filter price between [100, 500]

# String matching
df laptop_sales from sales
filter product contains "Laptop"

df recent from logs
filter event startswith "login"

df errors from logs
filter message not contains "warning"
```

**Supported Operators:**
- Comparison: `==`, `!=`, `>`, `<`, `>=`, `<=`
- Membership: `in`, `not in`
- Range: `between [lo, hi]`
- String: `contains`, `not contains`, `startswith`, `endswith`
- Logical: `and`, `or`

---

### Selecting Columns

Choose specific columns to keep:

```pivotal
df customer_summary from customers
select customer_id, name, email

df sales_metrics from sales
select product, quantity, revenue, profit_margin
```

---

### Creating/Modifying Columns

Create new columns or modify existing ones using the `assign` statement:

```pivotal
# Simple calculation
df sales
assign total = price * quantity

# Conditional assignment (only sets the column where the condition is true)
df products from catalog
assign discount_price = price * 0.9
    where category == "clearance"

# Multiple operations chained
df analysis from sales
assign revenue = price * quantity
assign profit = revenue - cost
assign margin = profit / revenue
    where revenue > 0
```

**Expression Syntax:**
- Reference columns directly by name
- Standard arithmetic operators: `+`, `-`, `*`, `/`, `**`
- Expressions that don't involve user-defined functions are evaluated via pandas `.eval()`

**Calling a Python function** (function must be in the session namespace):

```pivotal
python
    def clean_price(s):
        return s.str.replace("$", "").astype(float)

df sales
assign price = clean_price(price)
```

---

### Sorting

Sort data by one or more columns:

```pivotal
# Single column, ascending (default)
df sorted_sales from sales
sort amount

# Single column, descending
df top_performers from sales
sort revenue desc

# Multiple columns
df ranked_products from sales
sort category asc, sales desc, price asc
```

**Sort Orders:**
- `asc` - Ascending (default)
- `desc` - Descending

---

### Grouping and Aggregation

Group rows and compute aggregate statistics with an indented `agg` block:

```pivotal
# Sum one column
df revenue_by_region from sales
group by region
    agg sum amount

# Multiple aggregations
df summary from sales
group by category
    agg sum amount as total, mean amount as avg_amount, count amount as n

# Group by multiple columns
df detailed from sales
group by region, category
    agg sum amount as total, max amount as peak
```

**Aggregation Functions:**
- `sum`, `mean` / `avg`, `count`, `min`, `max`, `median`, `std`

---

### Merging Tables

Merge two tables together:

```pivotal
# Inner merge (default)
df combined from sales
merge other_table on customer_id

# Left merge
df sales_with_customers from sales
left merge customers on customer_id

# Outer merge
df full_data from table1
outer merge secondary on id

# Merge on multiple keys
df matched from table1
merge other on key1, key2
```

**Merge Types:**
- `merge` or `inner merge` — inner (intersection)
- `left merge` — left (all left, matching right)
- `right merge` — right (all right, matching left)
- `outer merge` — outer (union)

**Advanced Parameters:**
```pivotal
df complex_merge from table1
left merge table2
    left_on id
    right_on customer_id
    suffixes ["_left", "_right"]
```

Accepts all keyword arguments of `pandas.merge()`.

---

### Pivot Tables

Create pivot tables with aggregations:

```pivotal
# Basic pivot
df sales_pivot from sales
pivot
    agg sum amount
    rows product
    cols region

# Multiple aggregations on multiple columns
df multi_metric_pivot from sales
pivot
    agg sum revenue, mean quantity
    rows category
    cols quarter

# Complex pivot with multiple functions per column
df detailed_summary from sales
pivot
    agg sum sales, mean profit, sum units
    rows product, category
    cols region, quarter
```

**Aggregation Functions:**
- `sum` - Sum of values
- `mean` / `avg` - Average value
- `count` - Count of values
- `min` - Minimum value
- `max` - Maximum value
- `median` - Median value
- `std` - Standard deviation

---

### Data Cleaning

#### Drop Columns

Remove one or more columns:

```pivotal
df clean from sales
drop id, internal_ref
```

#### Rename Columns

Rename columns with `as`:

```pivotal
df renamed from sales
rename product as item, quantity as qty, unit_price as price
```

#### Handle Missing Values

Fill or drop rows with null values:

```pivotal
# Fill all nulls with a scalar value
df filled from raw
fillna 0

df filled_str from raw
fillna "unknown"

# Drop rows that contain any null
df complete from raw
dropna

# Drop rows where specific columns are null
df complete from raw
dropna price, quantity
```

#### De-duplicate

Remove duplicate rows:

```pivotal
# Remove fully duplicate rows
df unique from sales
distinct

# Remove duplicates based on specific columns
df unique from sales
distinct product, category
```

#### Concatenate Tables

Stack tables vertically:

```pivotal
# Append one table to another
df combined from jan_sales
concat feb_sales

# Append multiple tables at once
df all_sales from q1
concat q2, q3, q4
```

---

### Applying Python Functions

Define functions in a `python` block and call them from `assign` or `apply`.

#### `assign` with a user function

When the expression is a user-defined function call `func(col)`, Pivotal generates
`df['target'] = func(df['col'])` instead of routing through `df.eval()`:

```pivotal
python
    def clean_price(s):
        return s.str.replace("$", "").astype(float)

    def initials(s):
        return s.str[0].str.upper()

df sales
assign price = clean_price(price)
assign abbr  = initials(name)
```

#### `apply` — DataFrame-level transforms

`apply func_name` passes the entire active DataFrame through `func_name` and assigns
the result back:

```pivotal
python
    def remove_outliers(df):
        lo = df["price"].quantile(0.05)
        hi = df["price"].quantile(0.95)
        return df[df["price"].between(lo, hi)]

df sales
apply remove_outliers
group by category
    agg mean price as avg_price
```

---

### Package Management

A **package** is a self-contained folder of exported data tables and charts:

```
my_analysis/
  datapackage.json    ← resource manifest
  data/
    sales.csv
    summary.parquet
  charts/
    summary_bar.png
```

Code lives wherever it lives — the package is output only.

#### `save` — export a package snapshot

```pivotal
# Save all tables and charts in the session to a package
save "my_analysis"
    path "~/projects/output"

# Parquet format
save "my_analysis"
    path "~/projects/output"
    format parquet

# Include only specific tables
save "my_analysis"
    path "~/projects/output"
    tables sales, summary

# Include only specific charts
save "my_analysis"
    path "~/projects/output"
    charts summary_bar

# Exclude specific tables or charts
save "my_analysis"
    path "~/projects/output"
    exclude tables raw_import, temp
    exclude charts debug_plot
```

`save` is a **snapshot export** — each call wipes and recreates the package folder,
equivalent to Save-As.  Calling it twice with the same name and path overwrites the
first.  Use different names or paths to keep intermediate snapshots:

```pivotal
-- snapshot after cleaning
save "sales_v1_clean"
    path "~/output"

-- snapshot after enrichment
save "sales_v1_enriched"
    path "~/output"
```

Charts created by `plot` are tracked automatically and included in the export:

```pivotal
df summary
plot bar
    x category
    y total_revenue

save "my_analysis"
    path "~/output"
    -- saves the summary table and the summary_bar chart
```

#### `load` from a package

To load from a previously saved package, open it via a `python` block and then use
`load all` or `load <table>`:

```pivotal
python
    from pivotal import Package
    _pivotal_pkg = Package.open("my_analysis", path="~/projects/output")

load all

df summary
filter total_revenue > 1000
sort total_revenue desc
```

#### Example: two-file pipeline

```pivotal
-- pipeline.pivotal — process and export
load raw "raw/sales_2024.csv"

df clean from raw
dropna amount, customer_id
distinct

df summary from clean
group by category
    agg sum amount as total, count amount as n
sort total desc

save "sales_pipeline"
    path "~/projects/output"
    format parquet
```

```pivotal
-- analysis.pivotal — reload and continue
python
    from pivotal import Package
    _pivotal_pkg = Package.open("sales_pipeline", path="~/projects/output")

load all

df top from summary
filter total > 10000
sort total desc
```

---


## API Reference

### DSLParser

```python
parser = pivotal.DSLParser(backend="pandas")
```

**`backend`** — code generation backend: `"pandas"` (default).

#### Methods

##### `parse(code: str) -> list`
Parse DSL code and return the AST (a list of statement dicts).

```python
ast = parser.parse(dsl_code)
```

Raises `ValueError` for keyword-collision errors; returns `{'error': ...}` for
all other parse errors.

##### `generate_code(ast: list, backend: str = "pandas") -> list[str]`
Convert an AST to a list of Python code-block strings.

```python
blocks = parser.generate_code(ast)
```

##### `execute(code: str, globals_dict: dict, backend: str = "pandas", verbose: bool = True) -> None`
Parse and execute DSL code in one step, modifying `globals_dict` in place.

```python
ns = {}
parser.execute(dsl_code, ns, verbose=False)
# ns now contains all loaded/computed DataFrames
```

**Parameters:**
- `code` — Pivotal DSL code string
- `globals_dict` — namespace dict to execute in; DataFrames are added here
- `backend` — `"pandas"` (default)
- `verbose` — print execution summary (default: `True`)

##### `export(code: str) -> str | None`
Parse DSL code and return the generated Python as a single clean string, ready to
save as a `.py` file.  Internal Pivotal bookkeeping markers are stripped and
`import pandas as pd` is prepended automatically.

```python
python_script = parser.export(open("analysis.pivotal").read())

with open("analysis.py", "w") as f:
    f.write(python_script)
```

Returns `None` (and prints the error) if the DSL fails to parse.

##### `parse_file(path: str) -> list`
Convenience wrapper: read a `.pivotal` file and return its AST.

### Package

#### `Package.export()` — create a package from the session

```python
pkg = pivotal.Package.export(
    name="my_analysis",
    namespace=globals(),
    path="~/projects/output",   # optional, defaults to CWD
    fmt="csv",                  # "csv" (default) or "parquet"
    tables=["sales", "summary"],  # optional include list
    charts=["summary_bar"],       # optional include list
    exclude_tables=["raw"],       # optional exclude list
    exclude_charts=[],            # optional exclude list
)
```

Each call wipes and recreates the package folder (Save-As semantics).

#### `Package.open()` — open an existing package for loading

```python
pkg = pivotal.Package.open("my_analysis", path="~/projects/output")
```

| Method | Description |
|---|---|
| `export(name, namespace, path, fmt, tables, charts, ...)` | Export a fresh package snapshot |
| `open(name, path)` | Open an existing package for loading |
| `load_table(name)` | Load one table from `data/` (parquet preferred over CSV) |
| `load_all()` | Return a `{name: DataFrame}` dict of all tables in `data/` |

---

## Examples

### Example 1: Sales Analysis

```pivotal
# Load sales data
load sales "sales_data.csv"
    header 0

load products "product_catalog.csv"

# Filter high-value sales
df high_value from sales
filter amount > 500
select customer_id, product_id, amount, date

# Merge with product info
df enriched_sales from high_value
left merge products on product_id
select customer_id, product_name, category, amount

# Calculate metrics
df analysis from enriched_sales
assign revenue = amount
assign is_premium = amount > 1000

# Create summary pivot
df category_summary from analysis
pivot
    agg sum revenue, mean revenue, count revenue
    rows category
    cols is_premium
```

### Example 2: Customer Segmentation

```pivotal
# Load customer data
load customers "customer_data.csv"
load transactions "transaction_log.csv"

# Aggregate by customer
df customer_summary from transactions
group by customer_id
    agg sum amount as total_spent

# Segment customers
df segments from customer_summary
assign segment = "low"

df high_value from segments
assign segment = "high"
    where total_spent > 1000

df medium_value from segments
assign segment = "medium"
    where total_spent > 500 and total_spent <= 1000
```

### Example 3: Time Series Analysis

```pivotal
# Load time series data
load timeseries "sensor_data.csv"
    header 0

# Filter by date range
df recent_data from timeseries
filter date >= "2024-01-01"
select sensor_id, date, temperature, humidity

# Calculate derived columns
df with_metrics from recent_data
assign temp_fahrenheit = temperature * 9 / 5 + 32
assign comfort_index = temperature * 0.7 + humidity * 0.3

# Sort chronologically
df chronological from with_metrics
sort date asc, sensor_id asc

# Create pivot by sensor
df sensor_pivot from chronological
pivot
    agg mean temperature, min temperature, max temperature
    rows date
    cols sensor_id
```

### Example 4: Data Cleaning Pipeline

```pivotal
# Load data
load raw_data "input.csv"

# Drop columns we don't need
df trimmed from raw_data
drop internal_id, last_modified

# Rename for clarity
df renamed from trimmed
rename cust_nm as customer_name, val as value, cat as category

# Remove rows with missing critical fields
df no_nulls from renamed
dropna customer_name, value

# Fill remaining nulls in non-critical fields
df filled from no_nulls
fillna "uncategorised"

# Remove duplicates on key columns
df deduped from filled
distinct customer_name, value, category

# Filter to valid range and known categories
df final_data from deduped
filter value between [0, 10000]
filter category not contains "test"
sort value desc
```

---

## Comments

Pivotal supports single-line and multi-line comments:

```pivotal
# This is a single-line comment

-- This is also a single-line comment

/*
This is a
multi-line comment
*/

load data file.csv
   # Comments can appear anywhere
   header 0  -- Even after code
```

---

## Tips & Best Practices

### 1. **Use Descriptive Table Names**
```pivotal
# Good
df high_value_customers from customers
filter total_spent > 1000

# Less clear
df t1 from customers
filter total_spent > 1000
```

### 2. **Chain Operations on the Active Table**
```pivotal
df analysis from raw_data
filter status == "active"      # filter first
select id, name, value          # then narrow columns
assign normalized = value / 100 # then compute
sort normalized desc             # finally sort
```

### 3. **Use Indentation Consistently**
Pivotal uses indentation to define sub-blocks (`agg`, `where`, `pivot` params, `save` params). Use 4 spaces; do not mix tabs and spaces.

### 4. **Break Complex Pipelines into Named Steps**
```pivotal
df step1 from raw_data
filter condition1

df step2 from step1
merge other_data on key

df final from step2
select needed_columns
```

### 5. **Test Incrementally**
Execute code step-by-step in an interactive session to verify each operation.

---

## Troubleshooting

### Common Issues

**Import Error: `No module named 'pivotal'`**
- Ensure you have installed the package: `pip install .`
- Check that your virtual environment is active

**Parse Error: Unexpected indentation**
- Check that indentation is consistent (use spaces, not tabs)
- Ensure nested operations are properly indented

**Table Not Found Error**
- Make sure to `load` or create a table before referencing it
- Check table name spelling

**Column Not Found Error**
- Verify column exists in the table
- Check for typos in column names
- Use `print(table_name.columns)` to see available columns

---

## Advanced Usage

### Compile to Python Script

From the command line:

```bash
python -m pivotal --compile analysis.pivotal
# writes analysis.py alongside analysis.pivotal
```

Programmatically using `export()`, which strips internal markers and adds the pandas
import automatically:

```python
import pivotal

parser = pivotal.DSLParser()
script = parser.export(open("analysis.pivotal").read())

with open("analysis.py", "w") as f:
    f.write(script)
```

### Programmatic Execution

```python
import pivotal

parser = pivotal.DSLParser()
ns = {}
parser.execute(open("analysis.pivotal").read(), ns, verbose=False)

# DataFrames are in ns
print(ns["analysis"].describe())
```

### Package API

```python
import pivotal
import pandas as pd

# Export a package from the current session
sales = pd.read_csv("sales.csv")
summary = sales.groupby("category")["amount"].sum().reset_index()

pkg = pivotal.Package.export(
    "my_analysis",
    namespace={"sales": sales, "summary": summary},
    path="~/projects/output",
    fmt="parquet",
)

# Open a previously saved package and load tables
pkg = pivotal.Package.open("my_analysis", path="~/projects/output")
tables = pkg.load_all()   # returns {"sales": DataFrame, "summary": DataFrame}
sales = pkg.load_table("sales")
```

---

## Contributing

Contributions are welcome! 

---

## License

[Specify your license here]

---

## Authors

Neal Hughes

---

## Version History

- **v0.1.0** - Initial release
 

---

## Contact & Support

For questions, issues, or feature requests, please contact hughes.neal@gmail.com.
