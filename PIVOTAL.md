# Pivotal DSL — Language Reference

Pivotal is a DSL for data transformation that compiles to pandas Python code. It runs in two contexts:
- **`%%pivotal` cells** in JupyterLab notebooks (using the Pivotal cell magic)
- **`.pivotal` files** executed via `python -m pivotal <file.pivotal>` or compiled to `.py` with `python -m pivotal --compile <file.pivotal>`

In both cases the DSL is compiled to pandas code and executed in the current Python runtime. 


## Core conventions

All operations are **in-place on the active DataFrame** — `filter`, `select`, `sort`, column assignments etc. all modify the current table directly rather than returning a new one.

- `df <name>` sets the **active table** for all following statements until the next `df`
- Statements that operate on a table are **indented 4 spaces** under the `df` line
- Sub-clauses (window options, pivot args, assign conditions) are indented a further 4 spaces
- `load` and `save` are standalone — not indented under `df`
- Column names and table names are bare identifiers (no quotes)
- Strings use double or single quotes: `"North"`, `'2024-01-01'`
- **Python runtime variables** are referenced with a `:` prefix: `:my_var` — the value is substituted at execution time


## Loading data

```pivotal
load sales "data/sales.csv"
load budget "report.xlsx"
load events "events.parquet"

# With pandas reader options
load sales "data/sales.csv"
    header 0
    sep ";"

# Path from a Python runtime variable
load sales :my_path_variable
```

## Setting the active table

```pivotal
# Work with an existing table
df sales
    filter amount > 0

# Create a derived copy
df clean from sales
    filter amount > 0
```

## Filtering

```pivotal
df sales
    filter amount > 100
    filter region == "North" and status == "active"
    filter category in ["A", "B", "C"]
    filter price between [10, 500]
    filter name contains "Ltd"
    filter code startswith "UK"
    filter note not contains "test"
```

Operators: `==  !=  >  <  >=  <=  and  or  in  not in  between  contains  not contains  startswith  endswith`

## Selecting columns

```pivotal
df sales
    select region, amount, date
    select customer_id, revenue as rev
```

## Column assignment

Simple expression:
```pivotal
df sales
    revenue = price * quantity
    margin = (revenue - cost) / revenue
```

Conditional (sets column only where condition is true, NaN elsewhere):
```pivotal
df sales
    discount = price * 0.9
        where category == "clearance"
```

Group-level aggregate in expression:
```pivotal
df sales
    pct = amount / sum(amount)            # percent of total
    pct = amount / sum(amount)            # percent of group total
        by region
    z = (amount - mean(amount)) / std(amount)
```

Supported agg functions in expressions: `sum  mean  min  max  count  std  median  var  nunique  wavg(col, weight)`

Multi-case (CASE WHEN equivalent):
```pivotal
df sales
    tier =
        where amount > 1000: "high"
        where amount > 500: "medium"
        "low"
```

String functions:
```pivotal
df sales
    code = upper(category)
    abbr = left(name, 3)
    full = first_name + " " + last_name
```

String functions: `upper  lower  trim  ltrim  rtrim  left(col,n)  right(col,n)  substr(col,start,n)  len  replace(col,from,to)`

## Sorting

```pivotal
df sales
    sort amount desc
    sort region asc, amount desc
```

## Grouping and aggregation

```pivotal
df summary from sales
    group by region
        sum amount as total, mean amount as avg, count amount as n

df detail from sales
    group by region, category
        sum amount as total, nunique customer_id as customers
```

Agg functions: `sum  mean/avg  min  max  count  median  std  nunique  wavg col weight`

## Window functions

All share optional `by` (partition) and `order` sub-clauses:

```pivotal
df sales
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
df sales
    rank amount pct as r
        by region
```

Rolling functions: `mean  sum  min  max  std`
Cumulative: `cumsum  cummean  cummin  cummax`

## Merging

```pivotal
df result from sales
    merge customers on customer_id
    left merge products on product_id
    merge other
        left_on id
        right_on customer_id
```

Merge types: `merge` (inner)  `left merge`  `right merge`  `outer merge`

## Pivot and unpivot

```pivotal
df pivot_result from sales
    pivot
        sum amount, mean amount
        rows category
        cols region

df long from wide
    unpivot
        id region
        cols jan, feb, mar
        variable "month"
        value "amount"
```

## Data cleaning

```pivotal
df clean from raw
    drop internal_id, temp_col
    rename cust_nm as customer, val as amount
    fillna 0
    dropna customer, amount
    distinct customer, date
```

## Concatenate

```pivotal
df all_sales from q1
    concat q2, q3, q4
```

## Plotting

```pivotal
df summary
    plot bar revenue_chart
        x category
        y total
        title "Revenue by Category"
        by region
        cols 2
```

Chart types: `line  bar  scatter  hist  box  area`

## Tables (Great Tables)

```pivotal
df results
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
df sales
    filter region == :target_region
    filter amount > :min_amount
    filter category in :allowed_categories

# Use a Python variable as a file path
load sales :data_path
```

## Python blocks

Python code can be embedded directly. This is the primary way to define helper functions or perform operations that Pivotal doesn't cover. The `python`/`end` block is available in both `%%pivotal` cells and `.pivotal` files.

Multi-line block:
```pivotal
python
    def clean(s):
        return s.str.strip().str.upper()

    def flag_outlier(df):
        return df["amount"] > df["amount"].quantile(0.99)
end

df sales
    name = clean(name)
    python sales["outlier"] = flag_outlier(sales)
```

Single-line (inline Python, the active DataFrame is available by its table name):
```pivotal
df sales
    python sales["flag"] = sales["amount"] > 1000
```

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

## Key differences from SQL

| SQL | Pivotal |
|---|---|
| `SELECT * FROM sales WHERE amount > 0` | `df sales` / `filter amount > 0` |
| `SELECT a, b FROM sales` | `select a, b` |
| `SELECT price * qty AS revenue FROM sales` | `revenue = price * qty` |
| `CASE WHEN x > 1 THEN ... END` | `col =` / `where x > 1: ...` |
| `GROUP BY region` | `group by region` / `sum amount as total` |
| `JOIN` | `merge other on key` |
| `OVER (PARTITION BY region ORDER BY date)` | `by region` / `order date` sub-clauses |

## Key differences from pandas

- No index management — Pivotal always works on reset-index DataFrames
- No method chaining — statements are line-by-line under a `df` block
- `group by` produces a new table (not a GroupBy object to chain from)
- Window functions add a column in-place; they don't require `.groupby().transform()`
