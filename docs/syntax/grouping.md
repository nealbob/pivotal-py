# Grouping & Aggregation

## `group by` — aggregate by groups

Group rows by one or more columns and compute aggregate statistics.
Each aggregation sub-clause is prefixed with `agg`:

```pivotal
with sales as summary
    group by region
        agg sum revenue as total
```

## Aggregation functions

| Function | Description |
|----------|-------------|
| `sum` | Sum |
| `mean` | Arithmetic mean |
| `count` | Count of non-null values |
| `min` | Minimum |
| `max` | Maximum |
| `median` | Median |
| `std` | Standard deviation |
| `nunique` | Count of unique values |
| `wmean` | Weighted average |

Both space and bracket syntax are accepted:

```pivotal
agg sum revenue as total              # space syntax
agg sum(revenue) as total             # bracket syntax — both are equivalent
agg wmean quantity price as avg       # space syntax: wmean weight_col value_col
agg wmean(price, quantity) as avg     # bracket syntax: wmean(value_col, weight_col)
```

## Basic usage

```pivotal
with sales as by_region
    group by region
        agg sum revenue as total
```

```pivotal
with sales as by_category
    group by category
        agg mean price as avg_price
```

```pivotal
with events as counts
    group by event_type
        agg count id as n
```

## Multiple aggregations

List multiple aggregation functions separated by commas on a single `agg` line, or on multiple `agg` lines:

```pivotal
with sales as summary
    group by region
        agg sum revenue as total, mean revenue as avg, count id as deals
```

When the same aggregation function applies to several columns, you can write the function once:

```pivotal
with sales as averages
    group by region
        agg mean price, cost, margin
```

This is equivalent to `agg mean price, mean cost, mean margin`.

```pivotal
with sales as detailed
    group by region
        agg sum revenue as total
        agg mean revenue as avg_deal
        agg max revenue as top_deal
        agg count id as n_deals
```

## Multiple group-by columns

```pivotal
with sales as by_region_category
    group by region, category
        agg sum revenue as total, count id as deals
```

## Named results

The `as <name>` clause gives the aggregated column a name. Without it, Pivotal generates a name automatically:

```pivotal
agg sum revenue as total_revenue
agg count id as deal_count
agg mean price as avg_price
```

## Weighted average

`wmean <weight_col> <value_col>` computes a weighted mean. Note the argument order differs between the two syntaxes: space form takes weight first, bracket form takes value first.

```pivotal
with products
    group by category
        agg wmean quantity price as avg_price   # space: weight first
        agg wmean(price, quantity) as avg_price  # bracket: value first
```

`wavg` is accepted as a backward-compatible alias for `wmean`.

## `nunique` — count distinct values

```pivotal
with orders as summary
    group by region
        agg nunique customer_id as unique_customers
```

## `agg` — aggregate over all rows

To aggregate without grouping, use `agg` on its own:

```pivotal
with sales as totals
    agg sum revenue as total, mean price as avg_price, count id as n
```

This produces a single-row result. The same aggregation functions are supported as with `group by`.

The multi-column shorthand also works here:

```pivotal
with sales as averages
    agg mean price, cost, margin
```

## Example: full summary table

```pivotal
with orders as sales_summary
    filter status == "complete"
    group by region, category
        agg sum revenue as total, mean revenue as avg, count id as n, nunique customer_id as customers
    sort total desc
```
