# Pivotal

<img src="pivotal_logo.svg" width="120">

**Pivotal** is a simple data transformation language for Python. Process and analyze your data within Python without getting bogged down in Pandas syntax or SQL.

<img src="ataglance.png" width="600">

Check out the live demo in JupyterLab:

[Pivotal in Jupyter Lab](https://mybinder.org/v2/gh/nealbob/pivotal-demo/HEAD?urlpath=%2Fdoc%2Ftree%2Ffootball_demo.ipynb)

## Features

**Readable, Writable syntax** — write data transformations in a simple declarative syntax

**Multiple backends** — compile to Pandas (default), in-process DuckDB (SQL)

**Pipeline-oriented** — chain operations naturally with indentation blocks

**JupyterLab integration** — `%%pivotal` cell magic, interactive object viewer, syntax highlighting, autocomplete, export to Python code

**VS Code integration** — syntax highlighting, autocomplete, interactive execution, and Python code export

**Plotting and tables** — simple syntax for charts and publication-ready tables via matplotlib and Great Tables

**Data packages** — export all notebook output (DataFrames, charts, tables) to a single [Frictionless](https://specs.frictionlessdata.io/) data package

---

## Installation

```bash
pip install pivotal
```

With optional extras:

```bash
pip install pivotal[duckdb]    # DuckDB backend
pip install pivotal[jupyter]   # Jupyter GUI widgets
pip install pivotal[tables]    # Great Tables support
pip install pivotal[all]       # Everything
```

### JupyterLab extension

```bash
pip install git+https://github.com/nealbob/pivotal-py.git#subdirectory=editors/jupyterlab
```

### VS Code extension

Install from the VS Code Marketplace, or build locally from `editors/vscode`.

---

## Quick start

```python
%%pivotal
load sales "data/sales.csv"

df summary from sales
    filter status == "active"
    group by region
        agg sum revenue as total, count id as deals
    sort total desc
```

---

## Documentation

Full documentation including the complete syntax reference, backend guide, and API reference:

**[nealbob.github.io/pivotal-py](https://nealbob.github.io/pivotal-py)**

---

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

---

## License

MIT

---

## Authors

Neal Hughes

---

## Version History

- **v0.1.0** — Initial release

---

## Contact & Support

For questions, issues, or feature requests please open an issue on GitHub or contact [hughes.neal@gmail.com](mailto:hughes.neal@gmail.com).
