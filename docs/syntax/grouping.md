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
| `quantile` | Quantile, with `q` from 0 to 1 |
| `percentile` | Percentile, with `p` from 0 to 100 |
| `wmean` | Weighted average |

Both space and bracket syntax are accepted:

```pivotal
agg sum revenue as total              # space syntax
agg sum(revenue) as total             # bracket syntax — both are equivalent
agg quantile revenue 0.9 as p90       # quantile: value column first
agg percentile(revenue, 90) as p90    # percentile: value column first
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

## Quantiles and percentiles

`quantile <value_col> <q>` uses a probability from 0 to 1. `percentile <value_col> <p>` is equivalent but uses 0 to 100.

```pivotal
with sales as thresholds
    group by product
        agg quantile revenue 0.9 as p90
        agg percentile(revenue, 95) as p95
```

## Custom Python aggregations

In pandas pipelines, prefix a Python function with `:` to use it in an `agg` statement. Pivotal passes each listed column as a pandas Series to the function.

```pivotal
python from sklearn.metrics import r2_score

with predictions as model_scores
    group by year
        agg :r2_score actual predicted as r2
```

Custom aggregations also work without `group by`:

```pivotal
with predictions as overall_score
    agg :r2_score actual predicted as r2
```

Bracket syntax supports keyword arguments:

```pivotal
with predictions as model_scores
    group by year
        agg :my_metric(actual, predicted, squared=False, threshold=:cutoff) as score
```

Use spaces between input columns in the space form. Commas continue to separate aggregation items outside brackets, so custom functions do not use the multi-column shorthand form.

Polars supports custom Python aggregations by using a pandas fallback for that aggregation step, then converting the result back to Polars. DuckDB and SQL backends raise or emit an unsupported-backend message for custom aggregation functions.

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
