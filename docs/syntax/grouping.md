# Grouping & Aggregation

## `group by` — aggregate by groups

Group rows by one or more columns and compute aggregate statistics.

```pivotal
df summary from sales
    group by region
        sum revenue as total
```

Aggregation functions are indented under `group by`.

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

```pivotal
df by_region from sales
    group by region
        sum revenue as total
```

```pivotal
df by_category from sales
    group by category
        mean price as avg_price
```

```pivotal
df counts from events
    group by event_type
        count id as n
```

## Multiple aggregations

List multiple aggregation functions separated by commas on a single line, or on multiple lines:

```pivotal
df summary from sales
    group by region
        sum revenue as total, mean revenue as avg, count id as deals
```

```pivotal
df detailed from sales
    group by region
        sum revenue as total
        mean revenue as avg_deal
        max revenue as top_deal
        count id as n_deals
```

## Multiple group-by columns

```pivotal
df by_region_category from sales
    group by region, category
        sum revenue as total, count id as deals
```

## Named results

The `as <name>` clause gives the aggregated column a name. Without it, Pivotal generates a name automatically:

```pivotal
sum revenue as total_revenue
count id as deal_count
mean price as avg_price
```

## Weighted average

`wavg <value_col> <weight_col>` computes a weighted mean:

```pivotal
df products
    group by category
        wavg price quantity as avg_price
```

## `nunique` — count distinct values

```pivotal
df summary from orders
    group by region
        nunique customer_id as unique_customers
```

## Example: full summary table

```pivotal
df sales_summary from orders
    filter status == "complete"
    group by region, category
        sum revenue as total, mean revenue as avg, count id as n, nunique customer_id as customers
    sort total desc
```
