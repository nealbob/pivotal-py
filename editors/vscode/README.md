# Pivotal for VS Code

Syntax highlighting and execution support for the [Pivotal](https://github.com/pivotal/pivotal-py) data transformation DSL — a readable, SQL-like language that compiles to pandas.

## Features

- **Syntax highlighting** for `.pivotal` files
- **Syntax highlighting** for `%%pivotal` cell magic blocks embedded in `.py` files
- **Execute** a `.pivotal` file directly from the editor
- **Execute in Interactive Notebook** — runs the file as `%%pivotal` cells in a VS Code Interactive Window, with DataFrame previews
- **Execute Selection** — send a selected block of Pivotal code to the Interactive Window
- **Compile to Python** — generate a `.py` file from a `.pivotal` source file

## Requirements

- [Python extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Jupyter extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- `pivotal` Python package installed in your active environment (`pip install -e .` from the repo root)

## Commands

| Command | Description | Default keybinding |
|---|---|---|
| **Pivotal: Execute File** | Run the current `.pivotal` file via `python -m pivotal` in the integrated terminal | `Ctrl+F5` / `Cmd+F5` |
| **Pivotal: Execute in Interactive Notebook** | Run the file as `%%pivotal` cells in a VS Code Interactive Window | — (title bar button) |
| **Pivotal: Execute Selection in Interactive Notebook** | Send the selected text to the Interactive Window as a Pivotal cell | `Ctrl+Shift+F5` / `Cmd+Shift+F5` (when text is selected) |
| **Pivotal: Compile to Python File** | Parse the current `.pivotal` file and save the generated Python as a `.py` file in the same folder | — (title bar button) |

Commands are also available via the Command Palette (`Ctrl+Shift+P`).

## Cell markers in `.pivotal` files

Use `#%%` to split a `.pivotal` file into sections. When running via **Execute in Interactive Notebook**, each section is sent as a separate cell:

```
load sales "data/sales.csv"
filter sales
    revenue > 1000

#%%

group by region
    agg sum revenue as total_revenue
sort total_revenue desc
```

## Embedding Pivotal in Python files

You can write Pivotal cells directly inside a `.py` file using the `%%pivotal` IPython cell magic. The extension provides syntax highlighting for these blocks:

```python
import pivotal

# %%
%%pivotal
load df "data.csv"
filter df
    value > 0
```

Run these cells in the Interactive Window using the standard Python extension cell run buttons, or select the Pivotal block and use **Execute Selection in Interactive Notebook**.
