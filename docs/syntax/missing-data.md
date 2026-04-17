# Missing Data

## `fillna` — fill missing values

Replace `null` / `NaN` values with a constant.

### Fill all columns

```pivotal
with raw as clean
    fillna 0
```

```pivotal
with raw as clean
    fillna "unknown"
```

### Fill specific columns

Use an indented block to fill different columns with different values:

```pivotal
with raw as clean
    fillna
        price = 0
        name = "unknown"
        region = "N/A"
```

Or use comma-separated syntax:

```pivotal
with raw as clean
    fillna price 0, name "unknown", region "N/A"
```

Only the listed columns are filled. All other columns are unchanged.

---

## `dropna` — drop rows with missing values

### Drop rows with any missing value

```pivotal
with raw as complete
    dropna
```

### Drop rows where specific columns are missing

Only drop a row if any of the listed columns are null:

```pivotal
with raw as complete
    dropna price, quantity
```

```pivotal
with raw as complete
    dropna customer_id, product_id, date
```

