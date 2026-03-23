# Pivotal

<img src="pivotal_logo.svg" width="120">

**Pivotal** is a pipeline-oriented data transformation language for Python. Write concise, readable data operations in Jupyter notebooks or `.pivotal` files — Pivotal compiles them to pandas, DuckDB, or SQL.

<img src="ataglance.png" width="600">

Check out the live demo in JupyterLab:

[Pivotal in Jupyter Lab](https://mybinder.org/v2/gh/nealbob/pivotal-demo/HEAD?urlpath=%2Fdoc%2Ftree%2Ffootball_demo.ipynb)

## Features

**Readable syntax** — write data transformations in a clean, line-oriented DSL that reads top-to-bottom

**Multiple backends** — compile to pandas (default), in-process DuckDB, or pure SQL CTEs

**Pipeline-oriented** — chain operations naturally with indentation blocks

**JupyterLab integration** — `%%pivotal` cell magic with an interactive viewer panel, syntax highlighting, autocomplete, and no-code pivot tables and charts

**VS Code integration** — syntax highlighting, autocomplete, interactive execution, and code export

**Plotting and tables** — simple syntax for charts and publication-ready tables via matplotlib and Great Tables

**Export to code** — convert any notebook to `.py`, `.sql`, or `.pivotal` files

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

**[pivotal.readthedocs.io](https://pivotal.readthedocs.io)**

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
