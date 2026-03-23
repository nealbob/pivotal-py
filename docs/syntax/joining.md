# Joining

The `merge` statement joins two tables. The active table is the left table; the named table is the right table.

## Basic merge (inner join)

```
df enriched from sales
    merge customers on customer_id
```

Keeps only rows where `customer_id` exists in both tables.

## Join types

Prefix the merge with the join type:

```
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

```
df result from sales
    left merge customers on customer_id
```

### Multiple keys

```
df result from orders
    left merge inventory on product_id, warehouse_id
```

### Different key names

When the join columns have different names in each table:

```
df result from sales
    left merge customers
        left_on id
        right_on customer_id
```

## Handling duplicate column names

When both tables have columns with the same name (other than the join key), suffixes are added automatically. Customise them:

```
df result from sales
    left merge targets
        on region, category
        suffixes ["_actual", "_target"]
```

## Example: enriching a fact table

```
load orders "orders.csv"
load customers "customers.csv"
load products "products.csv"

df enriched from orders
    left merge customers on customer_id
    left merge products on product_id
    filter status == "complete"
    select order_id, date, customer_name, product_name, amount
```
