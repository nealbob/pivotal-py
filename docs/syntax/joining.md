# Merge and Concatenate

The `merge` statement joins two tables. The active table is the left table; the named table is the right table.

## Basic merge (inner join)

```pivotal
with sales as enriched
    merge customers on customer_id
```

Keeps only rows where `customer_id` exists in both tables.

## Join types

Prefix the merge with the join type:

```pivotal
with sales as enriched
    left merge customers on customer_id

with sales as enriched
    right merge customers on customer_id

with sales as enriched
    inner merge customers on customer_id

with sales as enriched
    outer merge customers on customer_id
```

| Type | Behaviour |
|------|-----------|
| (none) | Inner join — only matching rows |
| `left` | All rows from left, matched rows from right |
| `right` | All rows from right, matched rows from left |
| `inner` | Same as default — only matching rows |
| `outer` | All rows from both tables |

## Join keys

### Single key

```pivotal
with sales as result
    left merge customers on customer_id
```

### Multiple keys

```pivotal
with orders as result
    left merge inventory on product_id, warehouse_id
```

### Different key names

When the join columns have different names in each table:

```pivotal
with sales as result
    left merge customers
        left_on id
        right_on customer_id
```

## Handling duplicate column names

When both tables have columns with the same name (other than the join key), suffixes are added automatically. Customise them:

```pivotal
with sales as result
    left merge targets
        on region, category
        suffixes ["_actual", "_target"]
```

## Example: enriching a fact table

```pivotal
load "orders.csv" as orders
load "customers.csv" as customers
load "products.csv" as products

with orders as enriched
    left merge customers on customer_id
    left merge products on product_id
    filter status == "complete"
    select order_id, date, customer_name, product_name, amount
```

---

## `concat` — stack tables vertically

Append rows from another table onto the active table:

```pivotal
with jan_sales as all_sales
    concat feb_sales

with q1 as all_sales
    concat q2, q3, q4
```

Both tables must have compatible columns. Extra columns in either table will be filled with `null` for the rows where they are absent.

---

## `intersect` — keep only common rows

Keep rows that appear in both the active table and the named table (set intersection):

```pivotal
with all_customers as common
    intersect active_customers
```

Duplicate rows are removed from the result.

---

## `exclude` — remove rows present in another table

Remove rows from the active table that appear in the named table (set difference):

```pivotal
with all_customers as new_customers
    exclude existing_customers
```

```pivotal
with leads as unmatched
    exclude converted, disqualified
```

Duplicate rows are removed before the comparison.
