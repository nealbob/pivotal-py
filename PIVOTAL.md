# Pivotal DSL — Language Reference

Pivotal is a DSL for data transformation that compiles to Python or SQL code. It runs in two contexts:
- **`%%pivotal` cells** in JupyterLab notebooks (using the Pivotal cell magic)
- **`.pivotal` files** executed via `python -m pivotal <file.pivotal>` or compiled to `.py` with `python -m pivotal --compile <file.pivotal>`

The same DSL source runs unchanged on any of four backends:

| Backend | Description |
|---------|-------------|
| `pandas` | Default. Uses pandas DataFrames; best for interactive analysis. |
| `polars` | Polars DataFrames; faster for larger datasets. |
| `duckdb` | In-process analytical SQL engine. |
| `sql` | Generates plain SQL (export/compile only). |

### Setting the backend

In a notebook, set the backend for the whole session with `%pivotal_set`:

```python
%pivotal_set backend=duckdb
```

Or override it for a single cell:

```python
%%pivotal backend=polars
with sales
    filter amount > 0
```

Run `%pivotal_set` with no arguments to display current settings.


## Core conventions

All operations are **in-place on the active DataFrame** — `filter`, `select`, `sort`, column assignments etc. all modify the current table directly rather than returning a new one.

- `with <name>` sets the **active table** for all following statements until the next `with`
- `with <source> as <name>` creates a derived copy and makes it active
- Statements that operate on a table are indented under the `with` line
- Sub-clauses (window options, pivot args, assign conditions) are indented a further level
- `load`, `save`, and `delete` are standalone — not indented under `with`
- Column names and table names are bare identifiers (no quotes)
- Strings use double or single quotes: `"North"`, `'2024-01-01'`
- **Python runtime variables** are referenced with a `:` prefix: `:my_var` — the value is substituted at execution time


## Loading data

```pivotal
load "data/sales.csv" as sales
load "report.xlsx" as budget
load "events.parquet" as events

# With pandas reader options
load "data/sales.csv" as sales
    header 0
    sep ";"

# Path from a Python runtime variable
load :my_path_variable as sales
```

## Setting the active table

```pivotal
# Work with an existing table — all following statements modify 'sales' in place
with sales
    filter amount > 0

# Create a derived copy — 'sales' is left unchanged; all following statements modify 'clean'
with sales as clean
    filter amount > 0
```

`with <source> as <name>` creates a new table that is an independent copy of `<source>`. The original table continues to exist unchanged. Use this whenever you want to preserve the original.

## Filtering

```pivotal
with sales
    filter amount > 100
    filter region == "North" and status == "active"
    filter category in ["A", "B", "C"]
    filter price between [10, 500]
    filter name contains "Ltd"
    filter code startswith "UK"
    filter note not contains "test"
    filter amount > :min_amount        # runtime variable as filter value
```

Operators: `==  !=  >  <  >=  <=  and  or  in  not in  between  contains  not contains  startswith  endswith`

## Selecting columns

```pivotal
with sales
    select region, amount, date
    select customer_id, revenue as rev
```

## Column assignment

Simple expression:
```pivotal
with sales
    revenue = price * quantity
    margin = (revenue - cost) / revenue
```

Conditional (sets column only where condition is true, NaN elsewhere):
```pivotal
with sales
    discount = price * 0.9
        where category == "clearance"
```

With explicit default (else):
```pivotal
with sales
    discount = price * 0.9
        where category == "clearance"
        else price
```

> **Note:** `where` here is a sub-clause of the assignment — it is indented under the assignment line. This is different from `filter`, which is its own statement. Do not write `filter category == "clearance"` when you mean a conditional assignment.

Group-level aggregate in expression:
```pivotal
with sales
    pct = amount / sum(amount)            # percent of total
    pct = amount / sum(amount)            # percent of group total
        by region
    z = (amount - mean(amount)) / std(amount)
```

Supported agg functions in expressions: `sum  mean  min  max  count  std  median  var  nunique  wmean(col, weight)`

Multi-case (CASE WHEN equivalent) — use explicit `else` for the default branch:
```pivotal
with sales
    tier =
        where amount > 1000; "high"
        where amount > 500;  "medium"
        else "low"
```

String functions:
```pivotal
with sales
    code = upper(category)
    abbr = left(name, 3)
    full = first_name + " " + last_name
```

String functions: `upper  lower  trim  ltrim  rtrim  left(col,n)  right(col,n)  substr(col,start,n)  len  replace(col,from,to)`

Date functions:
```pivotal
with sales
    yr       = year(order_date)
    mo       = month(order_date)
    label    = date_format(order_date, "%b %Y")
    days_open = date_diff(close_date, open_date)
    due_date  = date_add(order_date, 30)
    parsed   = to_date(date_string_col)
```

Date functions: `year  month  day  quarter  dayofweek  hour  minute  date_format(col,fmt)  to_date(col)  date_diff(end,start)  date_add(col,n)`

## Sorting

```pivotal
with sales
    sort amount desc
    sort region asc, amount desc
```

## Grouping and aggregation

```pivotal
with sales as summary
    group by region
        agg sum amount as total, mean amount as avg, count amount as n

with sales as detail
    group by region, category
        agg sum amount as total
        agg nunique customer_id as customers
```

Agg functions: `sum  mean/avg  min  max  count  median  std  nunique  wmean weight col`

Both space and bracket syntax are accepted: `agg sum revenue as total` or `agg sum(revenue) as total`.

To apply one aggregation function to multiple columns, list the extra columns after commas:
`agg mean price, cost, margin` is equivalent to `agg mean price, mean cost, mean margin`.

To aggregate over all rows without grouping, use `agg` without `group by`:

```pivotal
with sales as totals
    agg sum amount as total, mean amount as avg
```

## Window functions

All share optional `by` (partition) and `order` sub-clauses:

```pivotal
with sales
    rank amount desc as sales_rank
        by region
        order date

    lag amount 1 as prev_amount
        by region
        order date

    lead amount 1 as next_amount
        by region
        order date

    cumsum amount as running_total
        by region
        order date

    rolling mean amount 7 as rolling_avg
        by region
        order date
```

`rank` with `pct` gives percentile ranks (0–1):
```pivotal
with sales
    rank amount pct as r
        by region
```

Rolling functions: `mean  sum  min  max  std`
Cumulative: `cumsum  cummean  cummin  cummax`

## Merging

```pivotal
with sales as result
    merge customers on customer_id
    left merge products on product_id
    merge other
        left_on id
        right_on customer_id
```

Merge types: `merge` (inner)  `left merge`  `right merge`  `outer merge`

## Pivot and unpivot

```pivotal
with sales as pivot_result
    pivot
        rows category
        cols region
        agg sum amount

with wide as long
    unpivot
        id region
        cols jan, feb, mar
        variable "month"
        value "amount"
```

## Type casting

```pivotal
with sales
    cast price as float            # coerce (bad values → NaN/null)
    cast qty as int
    cast name as string
    cast created_at as datetime
    cast price, cost as float      # multiple columns at once
    cast price as float strict     # strict mode — error on bad values
```

Types: `int` / `integer`  `float`  `string` / `str`  `bool` / `boolean`  `datetime`

Inline cast in expressions:
```pivotal
with sales
    price = float(price)
    label = str(code)
    ts = datetime(ts_col)
```

## Data cleaning

```pivotal
with raw as clean
    drop internal_id, temp_col
    rename cust_nm as customer, val as amount
    fillna 0
    dropna customer, amount
    distinct customer, date
```

## Concatenate / Set operations

```pivotal
with q1 as all_sales
    concat q2, q3, q4      # union all (stack rows)

with a as common
    intersect b             # rows present in both tables

with all_leads as new
    exclude converted       # rows in all_leads but not in converted
```

`fillna` can fill per-column with an indented block:

```pivotal
with raw as clean
    fillna
        price = 0
        name = "unknown"
```

Or use comma-separated syntax:

```pivotal
with raw as clean
    fillna price 0, name "unknown", region "N/A"
```

## Plotting

```pivotal
with summary
    plot bar revenue_chart
        x category
        y total
        title "Revenue by Category"
        by region
        cols 2
```

`pivot plot` produces a plot directly from a grouped aggregation without creating a separate summary table. Each `y` entry is a `func col` pair; multiple pairs are comma-separated. An optional `filter` before the statement pre-filters the data:
```pivotal
with sales
    pivot plot bar revenue_chart
        x region
        y sum amount, mean price
```

```pivotal
with sales
    filter year > 2020
    pivot plot line trend_chart
        x month
        y mean amount "Avg Sale"
        by category
```

Chart types: `line  bar  scatter  hist  box  area`

## Tables (Great Tables)

```pivotal
with results
    table my_table
        title "Summary"
        stub product
        format number 2
        format revenue as currency GBP
        stripe
```

## Runtime variables

Python variables in the kernel/runtime namespace are referenced with `:` prefix anywhere a value is expected:

```pivotal
# Use a Python variable as a filter value
with sales
    filter region == :target_region
    filter amount > :min_amount
    filter category in :allowed_categories

# Use a Python variable as a file path
load :data_path as sales
```

`:var` supports plain variable names only — subscript indexing (`:mylist[0]`, `:mydict['key']`) is not supported. Extract the value first:

```pivotal
python val = mylist[0]

with sales
    newvar = "prefix" + :val
```

## Python blocks

Python code can be embedded directly. This is the primary way to define helper functions or perform operations that Pivotal doesn't cover. The `python`/`end` block is available in both `%%pivotal` cells and `.pivotal` files.

Multi-line block — **must be closed with `end` on its own line**:
```pivotal
python
    def clean(s):
        return s.str.strip().str.upper()

    def flag_outlier(df):
        return df["amount"] > df["amount"].quantile(0.99)
end

with sales
    name = clean(name)
    python sales["outlier"] = flag_outlier(sales)
```

Single-line (inline Python, prefixed with `python` on the same line — no `end` needed):
```pivotal
with sales
    python sales["flag"] = sales["amount"] > 1000
```

## Show

Display the active table (or a preview of it) without saving:

```pivotal
with sales
    show                # display the full table
    show head           # display the first rows
    show summary        # display descriptive statistics
```

## Apply

Call a Python function on the active DataFrame and replace it with the result:

```pivotal
with sales
    apply my_cleaning_function
```

The function must accept a DataFrame and return a DataFrame. Define it in a `python` block first.

## Save

```pivotal
save "my_analysis"
    path "~/output"
    format parquet
    include sales, summary, revenue_chart
```

## Delete

```pivotal
delete temp_table
```

## Comments

```pivotal
# hash comment
-- SQL-style comment
/* multi-line
   comment */
```

---

## Common mistakes to avoid

These are patterns that look plausible but are wrong:

| Wrong | Right | Why |
|---|---|---|
| `sales.filter(amount > 0)` | `with sales` / `filter amount > 0` | No method chaining — Pivotal is line-by-line |
| `with sales = load("data.csv")` | `load "data.csv" as sales` | `load` is a standalone statement, not an assignment |
| `select *` | omit `select` entirely, or `show` | There is no wildcard select |
| `filter amount > 0` (at top level) | `with sales` / `    filter amount > 0` | All row/column ops must be indented under a `with` block |
| `with clean from sales` | `with sales as clean` | Copy syntax uses `with <source> as <name>` |
| `where amount > 0` (as a statement) | `filter amount > 0` | `where` is only valid as a sub-clause inside an assignment |
| `python` block without `end` | close every multi-line `python` block with `end` | Missing `end` is a syntax error |
| `:mylist[0]` or `:mydict['key']` | `python val = mylist[0]` then `:val` | Subscript indexing on `:var` is not supported |

---

## Key differences from SQL

| SQL | Pivotal |
|---|---|
| `SELECT * FROM sales WHERE amount > 0` | `with sales` / `filter amount > 0` |
| `SELECT a, b FROM sales` | `select a, b` |
| `SELECT price * qty AS revenue FROM sales` | `revenue = price * qty` |
| `CASE WHEN x > 1 THEN ... END` | `col =` / `where x > 1; ...` |
| `GROUP BY region` | `group by region` / `agg sum amount as total` |
| `JOIN` | `merge other on key` |
| `OVER (PARTITION BY region ORDER BY date)` | `by region` / `order date` sub-clauses |

## Key differences from pandas

- No index management — Pivotal always works on reset-index DataFrames
- No method chaining — statements are line-by-line under a `with` block
- `group by` produces a new table (not a GroupBy object to chain from)
- Window functions add a column in-place; they don't require `.groupby().transform()`
