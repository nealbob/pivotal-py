# Getting Started

## Installation

```bash
pip install pivotal-lang
```

This installs the complete feature set: Pandas, Polars, DuckDB, Great Tables, Jupyter widgets, and SQLAlchemy.

**Minimal install (pandas only):**

```bash
pip install --no-deps pivotal-lang
pip install lark pandas matplotlib
```

**Single-backend extras** — if you only need one backend in a constrained environment:

| Extra | Installs |
|-------|----------|
| `pivotal-lang[polars]` | polars |
| `pivotal-lang[duckdb]` | duckdb |
| `pivotal-lang[jupyter]` | ipywidgets, ipyfilechooser |
| `pivotal-lang[tables]` | great-tables, css_inline |
| `pivotal-lang[sql]` | sqlalchemy |

## JupyterLab extension

Install the JupyterLab extension separately for the viewer panel, syntax highlighting, and export tools:

```bash
pip install pivotal-lab
```

Restart JupyterLab after installing.

## VS Code extension

Install from the `editors/vscode` directory or search for **Pivotal** in the VS Code extension marketplace. See [VS Code](vscode.md) for details.

---

## First steps in Jupyter

After installing the JupyterLab extension, open a notebook and create a `%%pivotal` cell:

```python
%%pivotal
load sales "my_data.csv"

df summary from sales
    group by category
        sum revenue as total
    sort total desc
```

Run the cell — results appear in the **Pivotal Viewer** panel on the right.

To set options that persist for the notebook session:

```python
%pivotal_set backend=duckdb output_code=true
```

See [JupyterLab](jupyter.md) for the full list of options.

---

## First steps with the Python API

```python
from pivotal import DSLParser

parser = DSLParser()

# Execute directly
parser.execute("""
load sales "data/sales.csv"

df top from sales
    filter revenue > 1000
    sort revenue desc
""")

# Access the result
import pandas as pd
# tables are available in the parser's namespace
print(parser.namespace['top'])
```

**Generate code instead of running it:**

```python
results = parser.parse("""
df top from sales
    filter revenue > 1000
""")

code = parser.generate_code(results, backend='pandas')
print(code[0])
```

See [Python API](api.md) for full reference.

---

## First steps with .pivotal files

Create a file called `analysis.pivotal`:

```pivotal
# analysis.pivotal

load sales "data/sales.csv"

df summary from sales
    filter status == "active"
    group by region
        sum revenue as total
    sort total desc
```

Run it from the terminal:

```bash
pivotal analysis.pivotal
```

Compile it to Python:

```bash
pivotal --compile analysis.pivotal
# creates analysis.py
```

See [CLI](cli.md) for all commands.

---

## Core concepts

### The active table

Most operations act on the **active table**, set with `df`:

```pivotal
df sales           # set sales as the active table
    filter ...     # operates on sales
    sort ...       # still operating on sales
```

Create a new table derived from an existing one:

```pivotal
df top_sales from sales   # new table 'top_sales', reads from 'sales'
    filter revenue > 1000
    sort revenue desc
```

### Indentation

Sub-options and clauses are indented under their parent statement. The exact number of spaces doesn't matter — just be consistent:

```pivotal
df summary from sales
    group by region         # indented under df
        sum revenue     # indented under group by
    sort revenue desc       # back to df level
```

### Comments

```pivotal
# Hash comments

-- Double-dash comments

/* Multi-line
   comments */
```

### Python variable references

Prefix a Python variable name with `:` to use it inline:

```python
# In a Python cell or script
my_threshold = 1000
regions = ["North", "South"]
```

```pivotal
df filtered from sales
    filter revenue > :my_threshold
    filter region in :regions
```
