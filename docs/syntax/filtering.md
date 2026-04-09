# Filtering

The `filter` statement keeps rows that match a condition. Rows that do not match are dropped.

```pivotal
with sales as active
    filter status == "active"
```

## Comparison operators

| Operator | Meaning |
|----------|---------|
| `==` | Equal |
| `!=` | Not equal |
| `>` | Greater than |
| `<` | Less than |
| `>=` | Greater than or equal |
| `<=` | Less than or equal |

```pivotal
filter price > 100
filter quantity != 0
filter status == "active"
filter score >= 0.9
filter discount <= 0.5
```

## Logical operators

Combine conditions with `and` / `or`:

```pivotal
filter amount > 1000 and category == "premium"
filter status == "active" or price > 500
filter region == "North" and status == "active" and amount > 100
```

!!! note
    `and` binds more tightly than `or`. Use parentheses if you need explicit grouping — though in practice most filters are simple enough that this doesn't arise.

## Membership — `in` / `not in`

Test whether a value is in a list:

```pivotal
filter region in ["North", "South", "East"]
filter category not in ["test", "draft"]
```

Use a Python variable for the list:

```python
valid_regions = ["North", "South"]
```

```pivotal
filter region in :valid_regions
```

## Range — `between`

Test whether a value falls within an inclusive range:

```pivotal
filter price between [100, 500]
filter score between [0.8, 1.0]
```

## String matching

| Operator | Meaning |
|----------|---------|
| `contains "x"` | String contains substring |
| `not contains "x"` | String does not contain substring |
| `startswith "x"` | String starts with prefix |
| `not startswith "x"` | String does not start with prefix |
| `endswith "x"` | String ends with suffix |
| `not endswith "x"` | String does not end with suffix |

```pivotal
filter product contains "Laptop"
filter name not contains "test"
filter event startswith "login"
filter filename endswith ".csv"
```

## Python variable references

Any filter value can be a Python variable, prefixed with `:`:

```python
min_price = 100
target_region = "North"
```

```pivotal
filter price > :min_price
filter region == :target_region
```

## Multiple filter lines

Multiple `filter` lines under the same `df` are applied in sequence (AND logic):

```pivotal
with sales as result
    filter status == "active"
    filter amount > 500
    filter region in ["North", "South"]
```
