# Pipeline Control: Checks, Loops, and Functions

These commands make pipelines safer and more reusable. `assert` and `check` validate assumptions, `for` repeats column-oriented work, and `function` packages a pipeline pattern for reuse.

## Assertions and Checks

`assert` and `check` validate the active table without changing its rows or columns.

Use `assert` when bad data should stop the pipeline:

```pivotal
with orders
    assert order_id unique
    assert customer_id not null
    assert status in ["open", "closed", "cancelled"]
```

Use `check` when bad data should warn but allow the pipeline to continue:

```pivotal
with orders
    check amount >= 0
    check shipped_at not null
```

`assert` and `check` accept the same condition syntax as `filter`:

```pivotal
with sales
    assert amount >= 0
    check region not in ["test", "sandbox"]
```

They also support shorthand rules for uniqueness and missing values:

```pivotal
with orders
    assert order_id, line_id unique
    check customer_id not null
```

The pandas, Polars, and DuckDB backends evaluate data-quality commands at runtime. The SQL CTE backend emits skipped comments for them because plain SQL export cannot raise Python exceptions or emit Python warnings.

## Column Loops

Use a `for` block to apply the same column-oriented operation across several columns:

```pivotal
with sales
    for col in price, cost, revenue
        col = col / cpi
```

The loop variable is a placeholder for each listed column. Only exact bare identifier matches are replaced, so a loop variable named `x` does not alter a column named `colx`.

Named Pivotal lists can provide the loop columns:

```pivotal
list money_cols = price, cost, revenue

with sales
    for col in money_cols
        cast col as float
```

Python list variables can also provide loop columns at runtime:

```pivotal
with sales
    for col in :money_cols
        fillna col 0
```

Python-list loops are not supported by the plain SQL backend because SQL export needs concrete column names.

Loop assignment targets can build new names with string suffixes or prefixes:

```pivotal
with sales
    for col in price, cost, revenue
        col + "_real" = col / cpi
```

Column loops support assignment, `cast`, `fillna`, `drop`, `dropna`, `round`, and window functions such as `rank`, `lag`, `lead`, `cumsum`, `cummean`, `cummin`, `cummax`, and `rolling`.

## Pipeline Functions

Use `function` to define a reusable non-recursive pipeline. Functions are expanded before validation and backend code generation, so they behave like compile-time pipeline macros:

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

Function calls use parentheses. Inline list arguments use round brackets:

```pivotal
clean_sales(sales_raw, sales_clean, (price, cost, revenue))
```

Keyword arguments and Python runtime values are supported:

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

