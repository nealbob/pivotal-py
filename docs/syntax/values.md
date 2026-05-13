# Values: Scalars, Dicts, and Lists

Use native Pivotal values when you want reusable constants, column groups, filter lists, or configuration that should be understood before backend code is generated.

Native values persist in the Python namespace in notebooks and `%%pivotal` cells, so you can also refer to them later with `:` runtime references. The native form is usually better inside Pivotal code because Pivotal can resolve it during validation and SQL export when the value is known.

## Lists

Use `list` for reusable lists of columns or literal values:

```pivotal
list money_cols = price, cost, revenue
list regions = "AU", "NZ", "US"
list limits = -5, 5
```

Lists can be used anywhere Pivotal expects a concrete list of columns, tables, or literal values:

```pivotal
with sales
    select product, money_cols
    filter region in regions
    filter zscore > limits[0] and zscore < limits[1]
```

Lists can also drive column loops:

```pivotal
with sales
    for col in money_cols
        col = col / cpi
```

## Scalars

Use `scalar` for a reusable single value:

```pivotal
scalar gst = 0.1
scalar min_amount = 10

with sales
    tax = price * gst
    filter amount >= min_amount
```

Scalar values may be numbers, booleans, strings, identifiers, paths, or Python runtime references:

```pivotal
scalar threshold = :notebook_threshold
```

## Dicts

Use `dict` for nested configuration:

```pivotal
dict config
    thresholds
        low = -5
        high = 5
    columns
        money = price, cost, revenue
    labels
        AU = "Australia"
        NZ = "New Zealand"

with sales
    filter zscore > config.thresholds.low and zscore < config.thresholds.high
    select product, config.columns.money
    region_name = config.labels.AU
```

Dictionary entries can use either `=` or `:` for values:

```pivotal
dict config
    thresholds
        low: -5
        high: 5
```

Values separated by commas become lists:

```pivotal
dict config
    columns
        money = price, cost, revenue
```

## Loading Dicts

Load configuration from JSON or YAML by file extension:

```pivotal
dict config from "config.json"
dict labels from "labels.yml"
```

Bind an existing Python dictionary when it is already available in the runtime namespace:

```pivotal
dict config = :python_config

with sales
    filter amount < config.thresholds.high
```

YAML loading requires `PyYAML` to be installed.

## Native Values and `:pythonvar`

Native values and `:pythonvar` references overlap, but they are useful in different places.

Use native `list`, `scalar`, and `dict` definitions when the value is part of the Pivotal pipeline:

```pivotal
list money_cols = price, cost
scalar cutoff = 100

with sales
    select region, money_cols
    filter amount > cutoff
```

Use `:` references when the value is owned by surrounding Python code:

```python
money_cols = ["price", "cost"]
cutoff = 100
```

```pivotal
with sales
    select :money_cols
    filter amount > :cutoff
```

Once a native value has been defined, it also exists in the Python namespace, so `:money_cols` or `:config["thresholds"]["high"]` can work in later cells. For Pivotal expressions, prefer native lookup such as `money_cols`, `config.thresholds.high`, and `limits[0]` when possible.

