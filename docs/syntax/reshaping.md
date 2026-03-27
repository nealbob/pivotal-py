# Pivot Tables

## `pivot` — wide format

Pivot a long table into wide format by spreading a column's values across multiple columns.

### Minimal pivot

```pivotal
df pivot_table from sales
    pivot
        sum amount
        rows product
        cols region
```

This creates a table where each unique value of `region` becomes a column, and each row is a `product`, with `sum(amount)` as the values.

### Multiple aggregations

```pivotal
df summary from sales
    pivot
        sum revenue, mean quantity
        rows category
        cols quarter
```

### Multiple row and column dimensions

```pivotal
df detailed from sales
    pivot
        sum sales, mean profit
        rows product, category
        cols region, quarter
```

### Options

| Option | Description |
|--------|-------------|
| `<fn> <col>` | Aggregation function and value column |
| `rows <cols>` | Columns to use as row index |
| `cols <cols>` | Column whose values become column headers |

---

## `unpivot` — long format

Convert wide-format data (multiple value columns) to long format (single value column with a variable identifier).

### Basic unpivot

Specify the ID column(s) to keep fixed:

```pivotal
df long from monthly_sales
    unpivot
        id region
```

All other columns become rows, with a generated `variable` and `value` column.

### Specify value columns

```pivotal
df long from monthly_sales
    unpivot
        id region
        cols jan, feb, mar, apr
```

Columns not listed in `id` or `cols` are dropped.

### Custom column names

```pivotal
df long from monthly_sales
    unpivot
        id region
        cols jan, feb, mar
        variable "month"
        value "amount"
```

### Multiple ID columns

```pivotal
df long from sales
    unpivot
        id region, year
        cols q1, q2, q3, q4
        variable "quarter"
        value "revenue"
```

| Option | Description |
|--------|-------------|
| `id <cols>` | Columns to keep as identifiers (required) |
| `cols <cols>` | Columns to unpivot (optional; default = all non-id columns) |
| `variable "name"` | Name for the variable column (default `"variable"`) |
| `value "name"` | Name for the value column (default `"value"`) |
