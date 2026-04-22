# CLI

The `pivotal` command is installed automatically with `pip install pivotal-lang`.

## Run a `.pivotal` file

Execute a Pivotal source file directly:

```bash
pivotal analysis.pivotal
```

Tables are computed and any `show`, `plot`, or `table` statements produce output. The script runs in a fresh Python namespace with `pandas` available.

Equivalent to:

```bash
python -m pivotal analysis.pivotal
```

---

## Compile to Python

Compile a `.pivotal` file to a `.py` file:

```bash
pivotal --compile analysis.pivotal
# creates analysis.py in the same directory
```

The output file contains the generated pandas code with `import pandas as pd` at the top. Edit and run it independently of Pivotal.

---

## Export a notebook to Python or SQL

Convert a Jupyter notebook's `%%pivotal` cells to code:

```bash
pivotal --export-py analysis.ipynb
# creates analysis.py (pandas backend)
```

Specify a different backend:

```bash
pivotal --export-py analysis.ipynb --backend duckdb
# creates analysis.py with DuckDB code

pivotal --export-py analysis.ipynb --backend sql
# creates analysis.sql
```

- `%%pivotal` cells are compiled to the target backend
- Regular Python cells are included as-is (except for `sql` backend)
- GUI cells are skipped

---

## Export a notebook to `.pivotal`

Convert a Jupyter notebook to a standalone `.pivotal` file:

```bash
pivotal --export-pivotal analysis.ipynb
# creates analysis.pivotal
```

- `%%pivotal` cells → written as DSL source
- Python cells → wrapped in `python...end` blocks
- GUI cells → skipped

The output file is a valid `.pivotal` file that can be run with `pivotal analysis.pivotal` or opened in VS Code.

---

## Summary

| Command | Output | Description |
|---------|--------|-------------|
| `pivotal file.pivotal` | (runs) | Execute a `.pivotal` file |
| `pivotal --compile file.pivotal` | `file.py` | Compile to pandas Python |
| `pivotal --export-py nb.ipynb` | `nb.py` | Export notebook to Python |
| `pivotal --export-py nb.ipynb --backend duckdb` | `nb.py` | Export notebook with DuckDB |
| `pivotal --export-py nb.ipynb --backend sql` | `nb.sql` | Export notebook to SQL |
| `pivotal --export-pivotal nb.ipynb` | `nb.pivotal` | Export notebook to Pivotal DSL |
