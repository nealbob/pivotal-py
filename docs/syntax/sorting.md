# Sorting

## `sort` — sort rows

Sort the table by one or more columns.

### Single column

```pivotal
df sales
    sort revenue
```

Default order is ascending. Specify explicitly:

```pivotal
df sales
    sort revenue asc
df sales
    sort revenue desc
```

### Multiple columns

Separate columns with commas:

```pivotal
df sales
    sort category asc, revenue desc
    sort region asc, date asc, amount desc
```

Each column can have its own direction.
