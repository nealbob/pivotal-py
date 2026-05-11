# Python Integration

Pivotal is designed to be embedded in Python. When its built-in operations are not enough, you can drop into Python directly.

## Single-line Python statements

For a single Python statement, write it on the same line as `python` — no `end` required:

```pivotal
with sales
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

with sales
    price = :clean_price(price)
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
load "file.csv" as data

python
    # data is available as a pandas DataFrame
    print(data.dtypes)
    data["clean"] = data["raw"].str.strip()
end
```

---

## User-defined functions

Define Python functions in a `python...end` block and call them in column expressions with `:`:

```pivotal
python
    def clean_price(s):
        return s.str.replace("$", "").astype(float)

    def initials(s):
        return s.str[0].str.upper()
end

with sales
    price = :clean_price(price)
    abbr = :initials(name)
```

The `:` prefix marks the function as a Python runtime callable. Bare calls such as `upper(name)` are reserved for Pivotal built-ins or backend-native functions.

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

with sales
    apply :remove_outliers

with sales
    apply :normalise
    group by category
        agg mean amount as avg_z_score
```

The `:` prefix marks the function as a Python runtime callable. The function receives the active DataFrame and must return a DataFrame.

---

## Python variable references

Prefix any Python variable name with `:` to use its value inline in Pivotal expressions:

```python
min_date = "2024-01-01"
regions = ["North", "South"]
path = "data/sales.csv"
```

```pivotal
load :path as sales

with sales as filtered
    filter date >= :min_date
    filter region in :regions
```

This works with:

- Filter conditions
- Load paths
- Column values in expressions (constants)
- Lists in `in` / `not in` filters

---

## Working with `.py` and `.pivotal` files

For larger projects — or when working in an editor like VS Code — it's natural to split Python logic into a separate `.py` file and import it into your `.pivotal` script. Because `python` executes any valid Python statement, standard imports work as-is.

### Importing a Python module

```pivotal
python from myscript import clean_price

with sales
    price = :clean_price(price)
```

Or use `from ... import *` to bring everything into the shared namespace:

```pivotal
python from myscript import *

with sales
    price = :clean_price(price)
```

### VS Code workflow

Keep a `.py` file and a `.pivotal` file side by side. Import the `.py` file at the top of your script to connect them.

```
project/
├── transforms.py     # Python functions, full IDE support
└── pipeline.pivotal  # Data pipeline that imports transforms.py
```

```python
# transforms.py
import pandas as pd

def clean_price(s):
    return s.str.replace("$", "").astype(float)

def flag_outliers(df, col, threshold=3.0):
    z = (df[col] - df[col].mean()) / df[col].std()
    df["is_outlier"] = z.abs() > threshold
    return df
```

```pivotal
# pipeline.pivotal
python from transforms import *

load "sales.csv" as sales

with sales
    price = :clean_price(price)
    apply :flag_outliers
```

### Data science pipeline pattern

A common pattern is to alternate between Pivotal and Python: use Pivotal for data loading and wrangling, Python for analysis or modelling, then Pivotal again for result processing.
```
raw_data.csv
    └── pipeline.pivotal   (load, clean, filter)
            └── analysis.py        (model, stats)
                    └── pipeline.pivotal   (aggregate, format results)
```

```python
# analysis.py
from sklearn.linear_model import LinearRegression
import numpy as np

def fit_model(df):
    X = df[["price", "quantity"]].values
    y = df["revenue"].values
    model = LinearRegression().fit(X, y)
    df["predicted"] = model.predict(X)
    df["residual"] = y - df["predicted"]
    return df
```

```pivotal
python from analysis import *

load "sales.csv" as sales

with sales
    filter region == "North"
    apply :fit_model

with sales as summary
    group by category
        agg mean residual as avg_residual
        agg mean predicted as avg_predicted
```

---

## Accessing results in Python

After running Pivotal code, tables are available as Python variables in the same namespace:

```python
from pivotal import DSLParser

parser = DSLParser()
parser.execute("""
with sales as summary
    group by region
        agg sum revenue as total
""")

# Access the result
summary_df = parser.namespace['summary']
print(summary_df)
```

In Jupyter, tables are available directly in the next cell:

```python
%%pivotal
with sales as summary
    group by region
        agg sum revenue as total
```

```python
# Next Python cell
print(summary)  # available as a regular pandas DataFrame
```
