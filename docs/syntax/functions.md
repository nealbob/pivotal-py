# Lists and Pipeline Functions

Use `list` for reusable compile-time lists:

```pivotal
list money_cols = price, cost, revenue
list regions = "AU", "NZ", "US"
```

Lists can be used anywhere Pivotal expects a concrete list of columns, tables, or literal values:

```pivotal
with sales
    select region, money_cols
    filter region in regions
```

Use `:name` for Python runtime values such as variables or callables. Pivotal lists are resolved before backend generation, which makes them suitable for SQL export.

## Pipeline Functions

`function` defines a reusable non-recursive pipeline. During Pivotal execution, functions are expanded before validation and code generation:

```pivotal
list money_cols = price, cost, revenue

function clean_sales(input, output, cols, min_amount=0)
    with input as output
        dropna cols
        for col in cols
            cast col as float
        filter price >= min_amount
    return output

clean_sales(raw_sales, sales_clean, money_cols, min_amount=10)
```

Function arguments can be identifiers, literals, Python variables, named lists, or inline round-bracket lists:

```pivotal
clean_sales(raw_sales, sales_clean, (price, cost), min_amount=:threshold)
```

`return` is optional in Pivotal scripts. It records output metadata for Python-callable function wrappers.

```python
import pivotal

funcs = pivotal.load_functions("transforms.pivotal")
sales_clean = funcs.clean_sales(
    raw_sales,
    cols=["price", "cost"],
    min_amount=10,
)
```
