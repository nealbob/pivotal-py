# Pivotal

<img src="https://raw.githubusercontent.com/nealbob/pivotal-py/master/images/pivotal_logo.svg" width="120">

Pivotal is a data analysis language for Python.  It offers a concise syntax for common data operations that compiles to Pandas, Polars or DuckDB code.  With comprehensive JupyterLab and VS Code support (syntax highlighting, autocomplete, interactive viewer and GUI controls) Pivotal provides a friendly entry point to the Python data ecosystem.

<br>

<img src="https://raw.githubusercontent.com/nealbob/pivotal-py/master/images/ataglance.png" width="600">

A live-demo of [Pivotal in Jupyter Lab](https://mybinder.org/v2/gh/nealbob/pivotal-demo/HEAD?urlpath=%2Fdoc%2Ftree%2Ffootball_demo.ipynb) is available via Binder:

[![JupyterLab demo](https://raw.githubusercontent.com/nealbob/pivotal-py/master/images/piv2.png)](https://mybinder.org/v2/gh/nealbob/pivotal-demo/HEAD?urlpath=%2Fdoc%2Ftree%2Ffootball_demo.ipynb)

## Features

**Readable, Writable syntax** — write data transformations in a simple declarative syntax

**Multiple backends** — compile to Pandas (default), Polars or in-process DuckDB (SQL)

**JupyterLab and VS Code integration** — syntax highlighting, autocomplete, `%%pivotal` cell magic, interactive object viewer and explorer,  GUI controls

**Plotting and tables** — simple syntax for charts and publication-ready tables via matplotlib and Great Tables

**Data packages** — export all output (DataFrames, charts, tables) to a single [Frictionless](https://specs.frictionlessdata.io/) data package

---

## Installation

```bash
pip install pivotal-lang
```

This installs the full feature set — Pandas, Polars, DuckDB, Great Tables.

For a minimal Pandas-only install:

```bash
pip install --no-deps pivotal-lang
pip install lark pandas matplotlib
```

### JupyterLab extension

```bash
pip install pivotal-lab
```

### VS Code extension

Install from the VS Code Marketplace, or build locally from `editors/vscode`.

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
