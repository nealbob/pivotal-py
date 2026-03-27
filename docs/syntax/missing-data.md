# Missing Data

## `fillna` — fill missing values

Replace `null` / `NaN` values with a constant.

### Fill all columns

```pivotal
df clean from raw
    fillna 0
```

```pivotal
df clean from raw
    fillna "unknown"
```

### Fill specific columns

Use an indented block to fill different columns with different values:

```pivotal
df clean from raw
    fillna
        price = 0
        name = "unknown"
        region = "N/A"
```

Only the listed columns are filled. All other columns are unchanged.

---

## `dropna` — drop rows with missing values

### Drop rows with any missing value

```pivotal
df complete from raw
    dropna
```

### Drop rows where specific columns are missing

Only drop a row if any of the listed columns are null:

```pivotal
df complete from raw
    dropna price, quantity
```

```pivotal
df complete from raw
    dropna customer_id, product_id, date
```

