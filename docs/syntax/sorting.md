# Sorting & Limiting

## `sort` — sort rows

Sort the table by one or more columns.

### Single column

```
df sales
    sort revenue
```

Default order is ascending. Specify explicitly:

```
df sales
    sort revenue asc
df sales
    sort revenue desc
```

### Multiple columns

Separate columns with commas:

```
df sales
    sort category asc, revenue desc
    sort region asc, date asc, amount desc
```

Each column can have its own direction.

## `head` — keep first N rows

Keep only the first `n` rows after any sorting:

```
df top10 from sales
    sort revenue desc
    head 10
```

```
df sample from customers
    head 100
```

`head` is commonly used after `sort` to get top/bottom N records.
