# JupyterLab

The Pivotal JupyterLab extension provides a cell magic, a live viewer panel, an object explorer sidebar, keyboard shortcuts, and notebook export tools.

## Installation

```bash
pip install pivotal-lab
```

Restart JupyterLab after installing.

---

## Cell magic — `%%pivotal`

Write Pivotal DSL in a notebook cell by starting it with `%%pivotal`:

```python
%%pivotal
load sales "data/sales.csv"

df summary from sales
    group by region
        agg sum revenue as total
    sort total desc
```

Run the cell normally (Shift+Enter). Results appear in the Pivotal Viewer panel.

### Per-cell options

Pass options on the magic line to override session settings for that cell only:

```python
%%pivotal backend=duckdb output_code=true
df summary from sales
    group by region
        agg sum revenue as total
```

---

## `%pivotal_set` — session settings

Set options that apply to all subsequent `%%pivotal` cells:

```python
%pivotal_set backend=duckdb
%pivotal_set output_code=true viewer=false
%pivotal_set canvas=a4 margins=20
```

Run `%pivotal_set` with no arguments to display current settings.

### All settings

| Setting | Values | Default | Description |
|---------|--------|---------|-------------|
| `backend` | `pandas`, `duckdb`, `sql` | `pandas` | Execution backend |
| `viewer` | `true`, `false` | `true` | Send results to the Viewer panel |
| `output_code` | `true`, `false` | `false` | Print generated Python code below each cell |
| `canvas` | `none`, `a4`, `a4_landscape`, `a3`, `a3_landscape`, `letter`, `slide` | `none` | Page size for table rendering |
| `margins` | float (mm) | `25.4` | Page margins for table rendering |
| `chart_width` | `full`, `half` | `full` | Chart width as fraction of page |
| `viewer_font` | float (em) | `1.0` | Font size in the Viewer panel |
| `viewer_num_format` | integer | `5` | Significant digits for floats (0 = no formatting) |
| `use_visions` | `true`, `false` | `false` | Use the `visions` library for type inference |

---

## Viewer panel

The Viewer panel opens on the right side of JupyterLab and displays DataFrames, charts, and tables as they are computed.

- **Navigation:** use the back/forward arrows or `Alt+←` / `Alt+→`
- **Zoom:** `Alt+=` / `Alt+-`
- **Delete object:** `Alt+Delete`
- **Focus table:** `Alt+V`

The viewer formats large DataFrames with configurable float precision (`viewer_num_format`) and font size (`viewer_font`).

---

## Object Explorer

The Object Explorer sidebar lists all tables and charts currently in memory. Open it with `Alt+E` or from the left sidebar.

Click any object to load it in the Viewer.

---

## Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| `Alt+P` | Insert a new Pivotal cell below |
| `Alt+L` | Insert a load-data cell |
| `Alt+T` | Insert a pivot table cell |
| `Alt+C` | Insert a pivot chart cell |
| `Alt+S` | Insert a save cell |
| `Alt+V` | Focus Viewer panel |
| `Alt+E` | Open Object Explorer |
| `Alt+N` | Focus notebook |
| `Alt+←` | Viewer: back |
| `Alt+→` | Viewer: forward |
| `Alt+=` | Viewer: zoom in |
| `Alt+-` | Viewer: zoom out |
| `Alt+Delete` | Delete selected object from Viewer |

---

## Export to Code File

Convert the current notebook to a Python or SQL file via the **Pivotal** menu → **Export to Code File**.

A dialog lets you choose the format:

| Option | Output | Description |
|--------|--------|-------------|
| Pandas (.py) | `.py` | Each `%%pivotal` cell compiled to pandas Python |
| DuckDB (.py) | `.py` | Each `%%pivotal` cell compiled to DuckDB Python |
| SQL (.sql) | `.sql` | Each `%%pivotal` cell as a SQL CTE chain |
| Pivotal (.pivotal) | `.pivotal` | DSL cells as-is; Python cells wrapped in `python...end` |

The file is created in the same directory as the notebook.

---

## GUI tools

Pivotal provides interactive GUI helpers accessible from Python cells:

```python
import pivotal

pivotal.load_gui()      # file chooser to load data
pivotal.pivot_gui()     # visual pivot table builder
pivotal.plot_gui()      # visual chart builder
pivotal.save_gui()      # save data package
pivotal.settings_gui()  # Pivotal settings panel
```

These launch widget-based dialogs and generate the corresponding Pivotal code into a new cell.

---

## Accessing results in Python

Tables computed in `%%pivotal` cells are available as regular Python variables in subsequent cells:

```python
%%pivotal
df summary from sales
    group by region
        agg sum revenue as total
```

```python
# next cell — summary is a pandas DataFrame
print(summary.head())
print(summary.dtypes)
```
