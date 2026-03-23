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

```pivotal
df clean from raw
    fillna price=0, name="unknown", region="N/A"
```

!!! note
    When specific columns are named, only those columns are filled. All other columns are unchanged.

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

---

## `concat` — stack tables vertically

Append rows from another table onto the active table:

```pivotal
df all_sales from jan_sales
    concat feb_sales

df all_sales from q1
    concat q2, q3, q4
```

Both tables must have compatible columns. Extra columns in either table will be filled with `null` for the rows where they are absent.
