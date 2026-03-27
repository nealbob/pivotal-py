# Python Integration

Pivotal is designed to be embedded in Python. When its built-in operations are not enough, you can drop into Python directly.

## Single-line Python statements

For a single Python statement, write it on the same line as `python` — no `end` required:

```pivotal
df sales
    python sales["price"] = sales["price"].str.replace("$", "").astype(float)
    python sales = sales.dropna(subset=["amount"])
```

This is convenient for quick one-liner transformations without the overhead of a full block.

---

## `python...end` blocks

Embed arbitrary Python code in a Pivotal script. The block's contents are executed in the same namespace as the rest of the script, so tables defined in Pivotal are available as Python variables and vice versa.

```pivotal
python
    import numpy as np

    def clean_price(s):
        return s.str.replace("$", "").astype(float)

    def winsorize(df, col, lo=0.05, hi=0.95):
        low = df[col].quantile(lo)
        high = df[col].quantile(hi)
        df[col] = df[col].clip(low, high)
        return df
end

df sales
    price = clean_price(price)
```

### In Jupyter notebooks

In a `%%pivotal` cell, `python...end` blocks can reference any variable from the notebook kernel:

```python
# Python cell
threshold = 1000
```

```pivotal
%%pivotal
python
    filtered = sales[sales["amount"] > threshold]
end
```

### In .pivotal files

`python...end` blocks run in the file's execution namespace. Tables created before the block are available:

```pivotal
load data "file.csv"

python
    # data is available as a pandas DataFrame
    print(data.dtypes)
    data["clean"] = data["raw"].str.strip()
end
```

---

## User-defined functions

Define Python functions in a `python...end` block and call them in column expressions:

```pivotal
python
    def clean_price(s):
        return s.str.replace("$", "").astype(float)

    def initials(s):
        return s.str[0].str.upper()
end

df sales
    price = clean_price(price)
    abbr = initials(name)
```

---

## `apply` — apply a Python function to a table

Apply a Python function that takes a DataFrame and returns a DataFrame:

```pivotal
python
    def remove_outliers(df):
        lo = df["price"].quantile(0.05)
        hi = df["price"].quantile(0.95)
        return df[df["price"].between(lo, hi)]

    def normalise(df):
        df["amount"] = (df["amount"] - df["amount"].mean()) / df["amount"].std()
        return df
end

df sales
    apply remove_outliers

df sales
    apply normalise
    group by category
        mean amount as avg_z_score
```

The function receives the active DataFrame and must return a DataFrame.

---

## Python variable references

Prefix any Python variable name with `:` to use its value inline in Pivotal expressions:

```python
min_date = "2024-01-01"
regions = ["North", "South"]
path = "data/sales.csv"
```

```pivotal
load sales :path

df filtered from sales
    filter date >= :min_date
    filter region in :regions
```

This works with:

- Filter conditions
- Load paths
- Column values in expressions (constants)
- Lists in `in` / `not in` filters

---

## Accessing results in Python

After running Pivotal code, tables are available as Python variables in the same namespace:

```python
from pivotal import DSLParser

parser = DSLParser()
parser.execute("""
df summary from sales
    group by region
        sum revenue as total
""")

# Access the result
summary_df = parser.namespace['summary']
print(summary_df)
```

In Jupyter, tables are available directly in the next cell:

```python
%%pivotal
df summary from sales
    group by region
        sum revenue as total
```

```python
# Next Python cell
print(summary)  # available as a regular pandas DataFrame
```
