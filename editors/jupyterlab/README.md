# pivotal-lab

JupyterLab extension for [Pivotal](https://github.com/nealbob/pivotal-py) — a data analysis language for Python that compiles to Pandas, Polars, or DuckDB code.

<img src="https://raw.githubusercontent.com/nealbob/pivotal-py/master/images/piv2.png" width="600">

## Features

- `%%pivotal` cell magic to write and execute Pivotal DSL in JupyterLab notebooks
- Syntax highlighting for Pivotal DSL cells
- Autocomplete for keywords, table names, and column names
- Interactive data viewer and object explorer for DataFrames, charts, and tables
- GUI controls for filtering and exploring data without writing code

## Requirements

- JupyterLab 4.x
- `pivotal-lang` (the Pivotal compiler and runtime)

## Installation

```bash
pip install pivotal-lang
pip install pivotal-lab
```

## Quick start

In a JupyterLab notebook, use the `%%pivotal` cell magic:

```python
%%pivotal
load "data/sales.csv" as sales

with sales
    filter amount > 100
    group by region
        agg sum amount as total
```

Change the execution backend for the session:

```python
%pivotal_set backend=duckdb
```

## Documentation

Full documentation including syntax reference, backend guide, and examples:

**[nealbob.github.io/pivotal-py](https://nealbob.github.io/pivotal-py)**

## License

MIT
