# Merge and Concatenate

The `merge` statement joins two tables. The active table is the left table; the named table is the right table.

## Basic merge (inner join)

```pivotal
df enriched from sales
    merge customers on customer_id
```

Keeps only rows where `customer_id` exists in both tables.

## Join types

Prefix the merge with the join type:

```pivotal
df enriched from sales
    left merge customers on customer_id

df enriched from sales
    right merge customers on customer_id

df enriched from sales
    inner merge customers on customer_id

df enriched from sales
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
df result from sales
    left merge customers on customer_id
```

### Multiple keys

```pivotal
df result from orders
    left merge inventory on product_id, warehouse_id
```

### Different key names

When the join columns have different names in each table:

```pivotal
df result from sales
    left merge customers
        left_on id
        right_on customer_id
```

## Handling duplicate column names

When both tables have columns with the same name (other than the join key), suffixes are added automatically. Customise them:

```pivotal
df result from sales
    left merge targets
        on region, category
        suffixes ["_actual", "_target"]
```

## Example: enriching a fact table

```pivotal
load orders "orders.csv"
load customers "customers.csv"
load products "products.csv"

df enriched from orders
    left merge customers on customer_id
    left merge products on product_id
    filter status == "complete"
    select order_id, date, customer_name, product_name, amount
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

---

## `intersect` — keep only common rows

Keep rows that appear in both the active table and the named table (set intersection):

```pivotal
df common from all_customers
    intersect active_customers
```

Duplicate rows are removed from the result.

---

## `exclude` — remove rows present in another table

Remove rows from the active table that appear in the named table (set difference):

```pivotal
df new_customers from all_customers
    exclude existing_customers
```

```pivotal
df unmatched from leads
    exclude converted, disqualified
```

Duplicate rows are removed before the comparison.
