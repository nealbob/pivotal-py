# Column Expressions

Create new columns or modify existing ones by writing `column_name = expression` indented under the `with` statement.

## Simple expressions

```pivotal
with sales
    revenue = price * quantity
    margin = (revenue - cost) / revenue
    tax = revenue * 0.1
```

Any arithmetic expression is valid: `+`, `-`, `*`, `/`, `**` (power).

## Column loops

Use a `for` block to apply assignment statements to several columns:

```pivotal
with sales
    for col in price, cost, revenue
        col = col / cpi
```

The loop variable is a placeholder for each listed column. Only exact bare identifier matches are replaced, so a loop variable named `x` does not alter a column named `colx`.

Use a Python list variable when the column list is prepared in Python:

```pivotal
with sales
    for col in :money_cols
        col = col / cpi
```

Python-list loops work when Pivotal has a runtime namespace, such as notebook execution. They are not supported by the plain SQL backend because SQL export needs concrete column names.

Build new column names by combining the loop variable with string literals:

```pivotal
with sales
    for col in price, cost, revenue
        col + "_real" = col / cpi
```

Column loops also support column-oriented operations such as `cast`, `fillna`, `drop`, `dropna`, and window functions:

```pivotal
with sales
    for col in price, cost, revenue
        fillna col 0
        cast col as float
```

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
| `regex_extract(col, pattern)` | Extract first regex match |
| `regex_extract(col, pattern, group)` | Extract regex capture group |
| `regex_replace(col, pattern, replacement)` | Replace regex matches |

```pivotal
with products
    name = upper(name)
    code = left(sku, 4)
    slug = lower(trim(name))
    abbr = substr(code, 1, 3)
    n = len(description)
    clean = replace(notes, "N/A", "")
    postcode = regex_extract(address, "\\b\\d{4}\\b")
    clean_phone = regex_replace(phone, "[^0-9]", "")
```

Regex patterns use normal Pivotal strings. Escape backslashes inside quoted patterns, for example `"\\b\\d{4}\\b"` for a four-digit postcode on a word boundary.

**String concatenation** with `+`:

```pivotal
with contacts
    full_name = last_name + ", " + first_name
    label = code + "-" + region
```

**Nesting** is supported:

```pivotal
with products
    abbr = upper(left(name, 3))
```

## Date functions

Extract components from date/datetime columns or perform date arithmetic:

| Function | Description |
|----------|-------------|
| `year(col)` | Year as integer |
| `month(col)` | Month (1–12) |
| `day(col)` | Day of month |
| `quarter(col)` | Quarter (1–4) |
| `dayofweek(col)` | Day of week (0=Monday in pandas, 1=Monday in Polars) |
| `hour(col)` | Hour (0–23) |
| `minute(col)` | Minute (0–59) |
| `date_format(col, fmt)` | Format as string, e.g. `"%b %Y"` → `"Mar 2024"` |
| `to_date(col)` | Parse a string column to date/datetime |
| `date_diff(end_col, start_col)` | Difference in days (integer) |
| `date_add(col, n)` | Add `n` days to a date column |

```pivotal
with sales
    yr       = year(order_date)
    mo       = month(order_date)
    quarter  = quarter(order_date)
    label    = date_format(order_date, "%b %Y")
    days_open = date_diff(close_date, open_date)
    due_date  = date_add(order_date, 30)
```

**Tip:** to filter or group by a date part, create the column first, then use it:

```pivotal
with sales as monthly
    yr = year(order_date)
    mo = month(order_date)
    filter yr == 2024
    group by yr, mo
        agg sum amount as total
```

## Type casting

Convert column data types using `cast ... as <type>`:

```pivotal
with sales
    cast price as float            # coerce — bad values become NaN/null
    cast qty as int
    cast created_at as datetime
    cast price, cost as float      # multiple columns in one statement
    cast price as float strict     # strict — raises error on bad values
```

Types: `int` / `integer`  `float`  `string` / `str`  `bool` / `boolean`  `datetime`

Inline cast inside an expression:

```pivotal
with sales
    price = float(price)
    label = str(code)
    ts    = datetime(ts_col)
```

## Rounding

Round numeric columns with `round`:

```pivotal
with sales
    round revenue 2 as revenue_rounded
    round price, cost, margin 3
```

Use `as` with one source column to create a new rounded column. Omit `as` to
round one or more columns in place.

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
with sales
    pct_of_total = amount / sum(amount)
    z_score = (amount - mean(amount)) / std(amount)
```

**Windowed aggregates** — compute the aggregate within groups using `by`:

```pivotal
with sales
    pct_of_region = amount / sum(amount)
        by region

    regional_z = (amount - mean(amount)) / std(amount)
        by region, category
```

## Conditional assignment — `where`

Create a column with different values depending on a condition:

```pivotal
with sales
    discounted_price = price * 0.9
        where category == "clearance"
```

`else` is optional. Without it, rows where the condition is false receive `null` / `NaN`. Use `else` to provide a default:

```pivotal
with sales
    discounted_price = price * 0.9
        where category == "clearance"
        else price
```

## Multi-case assignment

Test multiple conditions in order; use explicit `else` for the default:

```pivotal
with sales
    tier =
        where amount > 500; "Gold"
        where amount > 100; "Silver"
        else "Bronze"
```

```pivotal
with products
    price_band =
        where price > 1000; "premium"
        where price > 200;  "mid"
        else "budget"
```

```pivotal
with sales
    adjusted =
        where amount > 500; amount * 2
        where amount > 100; amount * 1.5
        else amount
```

The conditions are evaluated in order; the first match wins. The `else` line is the default value.
