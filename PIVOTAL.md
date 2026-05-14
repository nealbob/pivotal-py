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
- **Python runtime variables and callables** are referenced with a `:` prefix: `:my_var`, `:my_func(col)` — the value is resolved at execution time


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

Bulk load CSV or Parquet files from a Python list or folder. A single alias
concatenates all files into one table and adds a `source` column with the input
filename:

```pivotal
bulk load :monthly_files as sales
bulk load "data/monthly" as sales
```

When a folder is provided, Pivotal loads the files inside it in sorted filename
order. The folder must be non-empty and contain only CSV files or only Parquet
files.

Use multiple aliases, or a Python list of aliases, to load each file into a
separate table instead:

```pivotal
bulk load :monthly_files as jan_sales, feb_sales, mar_sales
bulk load :monthly_files as :table_names
```

Customize the concat provenance column:

```pivotal
bulk load :monthly_files as sales
    source column file
    source value stem
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
    filter email matches ".+@.+\\..+"
    filter code startswith "UK"
    filter note not contains "test"
    filter amount > :min_amount        # runtime variable as filter value
```

Operators: `==  !=  >  <  >=  <=  and  or  in  not in  between  contains  not contains  matches  not matches  startswith  endswith`

## Data quality

Use `assert` when bad data should stop the pipeline. Use `check` when bad data
should produce a warning and let later statements continue.

```pivotal
with orders
    assert order_id unique
    assert customer_id not null
    assert status in ["open", "closed", "cancelled"]
    check amount >= 0
```

`assert` and `check` accept the same condition syntax as `filter`. They also
support `unique` and `not null` shorthand rules for one or more columns:

```pivotal
with orders
    assert order_id, line_id unique
    check shipped_at not null
```

The pandas, Polars, and DuckDB backends evaluate these rules at runtime. The
plain SQL backend leaves data-quality commands as skipped comments because SQL
export cannot emit Python warnings or exceptions.

## Selecting columns

```pivotal
with sales
    select region, amount, date
    select customer_id, revenue as rev
    select matches "^revenue_|_rate$"      # keep columns whose names match regex
    drop matches "^temp_"                  # remove columns whose names match regex
```

## Column assignment

Simple expression:
```pivotal
with sales
    revenue = price * quantity
    margin = (revenue - cost) / revenue
```

Apply the same assignment across several columns with a column loop:
```pivotal
with sales
    for col in price, cost, revenue
        col = col / cpi
```

The loop variable is a placeholder for each listed column. Only exact bare identifier matches are replaced, so a loop variable named `x` does not alter a column named `colx`.

Column loops can also use a Python list variable at runtime:
```pivotal
with sales
    for col in :money_cols
        col = col / cpi
```

Python-list loops are not supported by the plain SQL backend because SQL export needs concrete column names.

## Reusable values: lists, scalars, and dicts

Pivotal lists define reusable values that also persist into the Python namespace in notebooks and `%%pivotal` cells:
```pivotal
list money_cols = price, cost, revenue
list regions = "AU", "NZ", "US"
list limits = -5, 5

with sales
    for col in money_cols
        col = col / cpi

    filter region in regions
    filter zscore > limits[0] and zscore < limits[1]
```

Pivotal also supports native scalar and dictionary values. These persist in the Python namespace and are resolved before backend code generation when their values are known:
```pivotal
scalar gst = 0.1

dict config
    thresholds
        low = -5
        high = 5
    columns
        money = price, cost, revenue
    labels
        AU = "Australia"
        NZ = "New Zealand"
    class_names
        1 = "1st"
        2 = "2nd"
        3 = "3rd"

with sales
    filter zscore > config.thresholds.low and zscore < config.thresholds.high
    select product, config.columns.money
    tax = price * gst
```

Dictionary/config values can also be loaded from JSON or YAML based on file extension, or bound from an existing Python dictionary:
```pivotal
dict config from "config.json"
dict labels from "labels.yml"
dict pivotal_dict = :python_dict

with sales
    filter amount > config.thresholds.high
    region_name = labels.regions.AU
    filter amount > pivotal_dict.thresholds.high
```

Pivotal `list`, `scalar`, and `dict` definitions can be used either through native Pivotal syntax such as `config.thresholds.high` and `limits[0]`, or through `:` runtime references once they exist in the Python namespace. Prefer native lookup inside Pivotal code when the value belongs to the pipeline; use `:pythonvar` references when the value is owned by surrounding Python code. Python integration syntax is covered in more detail below.

Inline dictionary keys may be identifiers, quoted strings, or numbers. Numeric keys are stored as string keys in the Python dictionary, matching the rest of Pivotal's inline dict syntax.

## Column loops

Loop assignment targets can build new names with string suffixes or prefixes:
```pivotal
with sales
    for col in price, cost, revenue
        col + "_real" = col / cpi
```

Column loops also support simple column operations such as `cast`, `fillna`, `drop`, `dropna`, and window functions:
```pivotal
with sales
    for col in price, cost, revenue
        fillna col 0
        cast col as float
```

## Pipeline functions

Use `function` to define a reusable non-recursive pipeline. Functions are expanded before validation and backend code generation, so they behave like compile-time pipeline macros in Pivotal:

```pivotal
list money_cols = price, cost, revenue

function clean_sales(input, output, cols, min_amount=0)
    with input as output
        dropna cols
        for col in cols
            cast col as float
        filter price >= min_amount
    return output

clean_sales(sales_raw, sales_clean, money_cols, min_amount=10)
```

Function calls use parentheses. Inline list arguments also use round brackets:

```pivotal
clean_sales(sales_raw, sales_clean, (price, cost, revenue))
```

For longer lists, prefer a named `list` and pass the list name. Keyword arguments and Python runtime values are supported:

```pivotal
clean_sales(sales_raw, sales_clean, money_cols, min_amount=:threshold)
```

`return` is optional. It has no effect during normal Pivotal execution, but records which output table should be returned when Pivotal functions are exposed through a Python-callable API:

```python
import pivotal

funcs = pivotal.load_functions("transforms.pivotal")
sales_clean = funcs.clean_sales(
    sales_raw,
    cols=["price", "cost", "revenue"],
    min_amount=10,
)
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
    p90 = quantile(amount, 0.9)
        by region
```

Supported agg functions in expressions: `sum  mean  min  max  count  std  median  var  nunique  quantile(col, q)  percentile(col, p)  wmean(col, weight)`

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
    postcode = regex_extract(address, "\\b\\d{4}\\b")
    clean_phone = regex_replace(phone, "[^0-9]", "")
    full = first_name + " " + last_name
```

String functions: `upper  lower  trim  ltrim  rtrim  left(col,n)  right(col,n)  substr(col,start,n)  len  replace(col,from,to)  regex_extract(col,pattern)  regex_extract(col,pattern,group)  regex_replace(col,pattern,replacement)`

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

Agg functions: `sum  mean/avg  min  max  count  median  std  nunique  quantile col q  percentile col p  wmean weight col`

Both space and bracket syntax are accepted: `agg sum revenue as total` or `agg sum(revenue) as total`.
For quantiles, use column-first order: `agg quantile revenue 0.9 as p90` or `agg percentile(revenue, 90) as p90`.

To apply one aggregation function to multiple columns, list the extra columns after commas:
`agg mean price, cost, margin` is equivalent to `agg mean price, mean cost, mean margin`.

For pandas pipelines, custom Python functions can be used as aggregations by prefixing the function name with `:`. Each following column is passed as a Series argument to the Python function:

```pivotal
python from sklearn.metrics import r2_score

with predictions as model_scores
    group by year
        agg :r2_score actual predicted as r2
```

Bracket syntax supports keyword arguments:

```pivotal
with predictions as model_scores
    group by year
        agg :my_metric(actual, predicted, squared=False, threshold=:cutoff) as score
```

Polars supports custom Python aggregations by using a pandas fallback for that aggregation step, then converting the result back to Polars. DuckDB and SQL backends raise or emit an unsupported-backend message for custom aggregation functions.

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
        min_periods 1
```

`rank` with `pct` gives percentile ranks (0–1):
```pivotal
with sales
    rank amount pct as r
        by region
```

Rolling functions: `mean  sum  min  max  std`
Use `min_periods <n>` when early rows should produce values before a full window is available. `min_periods=<n>` is also accepted.
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

## Pivot tables and unpivot (melt)

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

Here `variable` and `value` are optional parameters that allow the user to customtise the names of the variable and value columns.

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

## Rounding

```pivotal
with sales
    round revenue 2 as revenue_rounded
    round price, cost, margin 3
```

`round <col> <digits> as <new_col>` creates a rounded copy of one column.
`round <colA>, <colB> <digits>` rounds multiple columns in place.

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

Unquoted fill values in per-column `fillna` statements are treated as column references:

```pivotal
with sales
    med_rev = median(revenue)
        by product
    fillna revenue med_rev
```

## Plotting

Standard `plot` creates a chart from the active table:

```pivotal
with summary
    plot revenue_chart
        kind bar
        x category
        y total
        title "Revenue by Category"
```

Or use the shorthand form with the chart type after `plot`:

```pivotal
with summary
    plot bar revenue_chart
        x category
        y total
        title "Revenue by Category"
        by region
        cols 2
```

Labels can be supplied after plotted values:

```pivotal
with summary
    plot line sales_chart
        x month "Month"
        y revenue "Revenue"
```

`pivot plot` produces a plot directly from a grouped aggregation without creating a separate summary table. Each `y` entry is a `func col` pair; multiple pairs are comma-separated. Like `plot`, extra keyword arguments such as `title`, `xlabel`, `ylabel`, `legend`, and `figsize` are passed through to pandas plotting. An optional `filter` before the statement pre-filters the data:
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

Use `table` to turn the active table into a formatted report table:

```pivotal
with results
    table my_table
        title "Summary"
        stub product
        format number 2
        format revenue as currency GBP
        stripe
```

Useful options include `subtitle`, `font size`, `font`, `label`, `summary`,
`spanner`, `canvas`, `style`, and `show`.

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

Subscript indexing is supported for Python lists and dictionaries after `:`:

```pivotal
with sales
    filter amount < :limits[1]
    filter amount < :config["thresholds"]["high"]
```

Python runtime functions also use `:` in column expressions. The function should accept a Series-like column and return a Series-like result:

```pivotal
python
    def clean_name(s):
        return s.str.strip().str.upper()
end

with sales
    name = :clean_name(name)
```

Bare function calls in column expressions are reserved for Pivotal built-ins such as `upper(name)`, `year(date)`, or backend-native functions like `log(col)`. Use `:my_func(col)` for Python functions.

## Python blocks

Python code can be embedded directly. This is the primary way to define helper functions or perform operations that Pivotal doesn't cover. The `python`/`end` block is available in both `%%pivotal` cells and `.pivotal` files.

Multi-line `python`/`end` block — **must be closed with `end` on its own line**:
```pivotal
python
    def clean(s):
        return s.str.strip().str.upper()

    def flag_outlier(df):
        return df["amount"] > df["amount"].quantile(0.99)
end

with sales
    name = :clean(name)
    python sales["outlier"] = flag_outlier(sales)
```

Single-line (inline Python, prefixed with `python` on the same line — no `end` needed):
```pivotal
with sales
    python sales["flag"] = sales["amount"] > 1000
```

Please DO NOT use Python blocks if it is possible to produce the same operation / outcome with native Pivotal code.

## Show

Display the active table (or a preview of it) without saving:

```pivotal
with sales
    show                # display the full table
    show head           # display the first rows
    show summary        # display descriptive statistics
    show shape          # display row/column counts
    show columns        # display column names
```

## Apply

Call a Python function on the active DataFrame and replace it with the result:

```pivotal
with sales
    apply :my_cleaning_function
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
| `clean_name(name)` for a Python helper | `:clean_name(name)` | Python runtime functions in column expressions need `:` |
| `:mylist[0]` or `:mydict['key']` | `:mylist[0]` or `:mydict['key']` | Indexed Python runtime references are supported |

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
