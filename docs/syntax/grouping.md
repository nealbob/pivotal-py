# Grouping & Aggregation

## `group by` — aggregate by groups

Group rows by one or more columns and compute aggregate statistics.

```
df summary from sales
    group by region
        agg sum revenue as total
```

The `agg` line(s) are indented under `group by`.

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
| `wavg` | Weighted average |

## Basic usage

```
df by_region from sales
    group by region
        agg sum revenue as total
```

```
df by_category from sales
    group by category
        agg mean price as avg_price
```

```
df counts from events
    group by event_type
        agg count id as n
```

## Multiple aggregations

List multiple `agg` functions separated by commas on a single line, or use multiple `agg` lines:

```
df summary from sales
    group by region
        agg sum revenue as total, mean revenue as avg, count id as deals
```

```
df detailed from sales
    group by region
        agg sum revenue as total
        agg mean revenue as avg_deal
        agg max revenue as top_deal
        agg count id as n_deals
```

## Multiple group-by columns

```
df by_region_category from sales
    group by region, category
        agg sum revenue as total, count id as deals
```

## Named results

The `as <name>` clause gives the aggregated column a name. Without it, Pivotal generates a name automatically:

```
agg sum revenue as total_revenue
agg count id as deal_count
agg mean price as avg_price
```

## Weighted average

`wavg <value_col> <weight_col>` computes a weighted mean:

```
df products
    group by category
        agg wavg price quantity as avg_price
```

## `nunique` — count distinct values

```
df summary from orders
    group by region
        agg nunique customer_id as unique_customers
```

## Example: full summary table

```
df sales_summary from orders
    filter status == "complete"
    group by region, category
        agg sum revenue as total, mean revenue as avg, count id as n, nunique customer_id as customers
    sort total desc
```
