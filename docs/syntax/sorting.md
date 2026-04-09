# Sorting

## `sort` — sort rows

Sort the table by one or more columns.

### Single column

```pivotal
with sales
    sort revenue
```

Default order is ascending. Specify explicitly:

```pivotal
with sales
    sort revenue asc
with sales
    sort revenue desc
```

### Multiple columns

Separate columns with commas:

```pivotal
with sales
    sort category asc, revenue desc
    sort region asc, date asc, amount desc
```

Each column can have its own direction.
