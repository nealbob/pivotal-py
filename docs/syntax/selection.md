# Selection

## `select` — keep columns

Keep only the specified columns, in the given order:

```pivotal
df lean from customers
    select customer_id, name, email
```

```pivotal
df report from sales
    select product, region, revenue, date
```

## `drop` — remove columns

Remove specific columns, keep everything else:

```pivotal
df clean from raw
    drop internal_id, debug_flag, temp_col
```

## `distinct` — remove duplicate rows

### Deduplicate on all columns

Remove rows where every column is identical to another row:

```pivotal
df unique from sales
    distinct
```

### Deduplicate on specific columns

Keep the first occurrence for each unique combination of the specified columns:

```pivotal
df unique_products from sales
    distinct product, category
```

!!! note
    When specific columns are provided, only those columns are kept in the result. If you want to deduplicate based on certain columns but keep all columns, combine with a merge or use `apply` with a Python function.

## `rename` — rename columns

```pivotal
df renamed from sales
    rename product as item, quantity as qty, unit_price as price
```

Multiple renames in one statement, separated by commas.
