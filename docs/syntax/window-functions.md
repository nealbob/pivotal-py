# Window Functions

Window functions compute values across a sliding window or partition of rows, without collapsing the table. All window functions are written under the `with` statement.

## `rank` — ranking

Rank rows by a column value.

### Basic rank

```pivotal
with sales
    rank amount desc as sales_rank
```

### Ascending rank

```pivotal
with sales
    rank amount asc as rank_asc
```

### Percentile rank

Returns a value between 0 and 1:

```pivotal
with sales
    rank amount pct as pct_rank
```

### Partitioned rank (rank within groups)

Use `by` to rank within groups:

```pivotal
with sales
    rank amount desc as regional_rank
        by region
```

```pivotal
with sales
    rank amount desc as cat_rank
        by region, category
```

---

## `lag` / `lead` — shifted values

Access values from a previous (`lag`) or future (`lead`) row.

### Basic lag

```pivotal
with sales
    lag amount 1 as prev_amount
        order date
```

The `order` clause specifies which column defines row order. It is **required**.

### Lead

```pivotal
with sales
    lead amount 1 as next_amount
        order date
```

### With partition

```pivotal
with sales
    lag amount 1 as prev_regional_amount
        by region
        order date
```

### Larger offsets

```pivotal
with sales
    lag amount 3 as amount_3_periods_ago
        order date
```

---

## Cumulative statistics

Compute running totals, averages, min, and max.

| Statement | Description |
|-----------|-------------|
| `cumsum <col> as <name>` | Running sum |
| `cummean <col> as <name>` | Running average |
| `cummin <col> as <name>` | Running minimum |
| `cummax <col> as <name>` | Running maximum |

The `order` clause specifies row order and is **required**. Use `by` for partitioned cumulation.

```pivotal
with sales
    cumsum amount as running_total
        order date

with sales
    cummean amount as running_avg
        order date

with sales
    cummin amount as running_min
        by region
        order date

with sales
    cummax amount as running_max
        by region
        order date
```

### Partitioned cumulation

```pivotal
with sales
    cumsum amount as regional_running_total
        by region
        order date
```

---

## `rolling` — rolling window statistics

Compute statistics over a sliding window of N rows.

```pivotal
with sales
    rolling mean amount 7 as rolling_7d_avg
        order date
```

### Rolling functions

| Function | Description |
|----------|-------------|
| `rolling mean <col> <n>` | Rolling average |
| `rolling sum <col> <n>` | Rolling sum |
| `rolling std <col> <n>` | Rolling standard deviation |
| `rolling min <col> <n>` | Rolling minimum |
| `rolling max <col> <n>` | Rolling maximum |

### Options

```pivotal
with sales
    rolling mean amount 4 as rolling_avg
        by region     # compute within groups
        order date    # required: defines row order
```

### Examples

```pivotal
with daily_sales
    rolling sum revenue 7 as weekly_revenue
        order date

with daily_sales
    rolling mean revenue 30 as monthly_avg
        order date

with regional_sales
    rolling sum revenue 4 as quarterly_rolling
        by region
        order date
```
