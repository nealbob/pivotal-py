# Python API

## `DSLParser`

The main entry point for using Pivotal programmatically.

```python
from pivotal import DSLParser

parser = DSLParser()
```

### `parse(source)`

Parse a Pivotal source string into an abstract syntax tree (list of AST nodes).

```python
results = parser.parse("""
with sales as summary
    group by region
        agg sum revenue as total
""")
```

**Returns:** A list of AST node dicts, or `{'error': '...'}` on parse failure.

```python
if isinstance(results, dict) and 'error' in results:
    print(f"Parse error: {results['error']}")
```

---

### `generate_code(results, backend='pandas')`

Generate code from parsed AST nodes.

```python
results = parser.parse(source)
code_blocks = parser.generate_code(results, backend='pandas')
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `results` | list | AST from `parse()` |
| `backend` | str | `'pandas'` (default), `'duckdb'`, or `'sql'` |

**Returns:** A list of code strings (one per logical block). Join them to produce a complete script:

```python
full_code = '\n\n'.join(code_blocks)
print(full_code)
```

---

### `execute(source, backend='pandas', namespace=None)`

Parse and execute Pivotal source in one call.

```python
parser.execute("""
load "data/sales.csv" as sales

with sales as summary
    group by region
        agg sum revenue as total
    sort total desc
""")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | str | Pivotal DSL source |
| `backend` | str | `'pandas'` (default), `'duckdb'`, or `'sql'` |
| `namespace` | dict | Execution namespace (default: creates a new one) |

Tables are accessible after execution via `parser.namespace`:

```python
parser.execute(source)
summary_df = parser.namespace['summary']
```

Or pass in an existing namespace to share variables:

```python
ns = {'threshold': 1000, 'sales': existing_df}
parser.execute("""
with sales as filtered
    filter amount > :threshold
""", namespace=ns)

filtered = ns['filtered']
```

---

### `export(source, backend='pandas')`

Generate a clean, standalone Python script from Pivotal source.

```python
script = parser.export(source, backend='duckdb')
with open('analysis.py', 'w') as f:
    f.write(script)
```

**Returns:** A string containing the complete Python script including imports.

---

## `Package`

Manage data packages — collections of tables and charts saved to disk.

```python
from pivotal import Package
```

### `Package.export(name, globals, path=None, fmt='csv', chart_fmt='png', include=None, exclude=None)`

Save all tables and charts from the current session to a data package.

```python
Package.export('my_analysis', globals(), path='~/output')
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Package name (becomes directory name) |
| `globals` | dict | The calling scope's globals (pass `globals()`) |
| `path` | str | Output directory (default: current directory) |
| `fmt` | str | Data format: `'csv'` (default) or `'parquet'` |
| `chart_fmt` | str | Chart format: `'png'` (default) or `'svg'` |
| `include` | list | Object names to include (default: all) |
| `exclude` | list | Object names to exclude (default: none) |

---

### `Package.open(name, path=None)`

Load a previously saved package.

```python
pkg = Package.open('my_analysis', path='~/output')
```

**Returns:** A `Package` object.

---

### `pkg.load_all()`

Load all tables from the package into a dict:

```python
pkg = Package.open('my_analysis')
tables = pkg.load_all()
sales = tables['sales']
```

---

### `pkg.load_table(name)`

Load a single table by name:

```python
pkg = Package.open('my_analysis')
sales = pkg.load_table('sales')
```

---

## Notebook export functions

### `notebook_to_python(path, backend='pandas')`

Export a Jupyter notebook to a Python or SQL file.

```python
from pivotal.__main__ import notebook_to_python

notebook_to_python('analysis.ipynb', backend='duckdb')
# creates analysis.py
```

```python
notebook_to_python('analysis.ipynb', backend='sql')
# creates analysis.sql
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str | Absolute path to `.ipynb` file |
| `backend` | str | `'pandas'`, `'duckdb'`, or `'sql'` |

- `%%pivotal` cells are parsed and compiled to the target backend
- Regular Python cells are included as-is (except for `backend='sql'`)
- GUI cells (`pivotal.*_gui()`) are skipped

---

### `notebook_to_pivotal(path)`

Export a Jupyter notebook to a `.pivotal` file.

```python
from pivotal.__main__ import notebook_to_pivotal

notebook_to_pivotal('analysis.ipynb')
# creates analysis.pivotal
```

- `%%pivotal` cells are written as-is (DSL source only, magic line stripped)
- Regular Python cells are wrapped in `python...end` blocks
- GUI cells are skipped

The resulting `.pivotal` file is fully executable with the Pivotal CLI.
