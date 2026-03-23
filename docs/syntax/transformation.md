# Column Expressions

Create new columns or modify existing ones by writing `column_name = expression` indented under the `df` statement.

## Simple expressions

```pivotal
df sales
    revenue = price * quantity
    margin = (revenue - cost) / revenue
    tax = revenue * 0.1
```

Any arithmetic expression is valid: `+`, `-`, `*`, `/`, `**` (power).

## String functions

| Function | Description |
|----------|-------------|
| `upper(col)` | Uppercase |
| `lower(col)` | Lowercase |
| `trim(col)` | Strip leading/trailing whitespace |
| `ltrim(col)` | Strip leading whitespace |
| `rtrim(col)` | Strip trailing whitespace |
| `left(col, n)` | First `n` characters |
| `right(col, n)` | Last `n` characters |
| `substr(col, start, length)` | Substring |
| `len(col)` | String length |
| `replace(col, old, new)` | Replace substring |

```pivotal
df products
    name = upper(name)
    code = left(sku, 4)
    slug = lower(trim(name))
    abbr = substr(code, 1, 3)
    n = len(description)
    clean = replace(notes, "N/A", "")
```

**String concatenation** with `+`:

```pivotal
df contacts
    full_name = last_name + ", " + first_name
    label = code + "-" + region
```

**Nesting** is supported:

```pivotal
df products
    abbr = upper(left(name, 3))
```

## Aggregate functions in expressions

Use aggregate functions to compute values relative to the whole table:

| Function | Description |
|----------|-------------|
| `sum(col)` | Total |
| `mean(col)` | Average |
| `std(col)` | Standard deviation |
| `min(col)` | Minimum |
| `max(col)` | Maximum |
| `count(col)` | Count |

```pivotal
df sales
    pct_of_total = amount / sum(amount)
    z_score = (amount - mean(amount)) / std(amount)
```

**Windowed aggregates** — compute the aggregate within groups using `by`:

```pivotal
df sales
    pct_of_region = amount / sum(amount)
        by region

    regional_z = (amount - mean(amount)) / std(amount)
        by region, category
```

## Conditional assignment — `where`

Create a column with different values depending on a condition:

```pivotal
df sales
    discounted_price = price * 0.9
        where category == "clearance"
```

Rows where the condition is false receive `null` / `NaN`. To provide a fallback, use multi-case syntax.

## Multi-case assignment

Test multiple conditions in order; the last line is the default (else):

```pivotal
df sales
    tier =
        where amount > 500: "Gold"
        where amount > 100: "Silver"
        "Bronze"
```

```pivotal
df products
    price_band =
        where price > 1000: "premium"
        where price > 200:  "mid"
        "budget"
```

```pivotal
df sales
    adjusted =
        where amount > 500: amount * 2
        where amount > 100: amount * 1.5
        amount
```

The conditions are evaluated in order; the first match wins. The final line (no `where`) is the default value.
