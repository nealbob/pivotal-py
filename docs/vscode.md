# VS Code

The Pivotal VS Code extension provides syntax highlighting, autocomplete, and execution commands for `.pivotal` files.

## Installation

Install from the `editors/vscode` directory:

```bash
cd editors/vscode
npm install
npm run package       # produces pivotal-language-<version>.vsix
code --install-extension pivotal-language-<version>.vsix
```

Or install directly from the VS Code Extensions panel when the extension is published to the marketplace.

**Requires:** VS Code 1.75.0 or later.

---

## File extension

Pivotal source files use the `.pivotal` extension. The extension activates automatically when you open a `.pivotal` file.

---

## Syntax highlighting

The extension provides syntax highlighting for:

- **Keywords:** `with`, `load`, `filter`, `select`, `sort`, `order by`, `group`, `agg`, `merge`, `pivot`, `unpivot`, `plot`, `table`, `drop`, `fillna`, `dropna`, `distinct`, `concat`, `intersect`, `exclude`, `cast`, `rename`, `apply`, `save`, `delete`, `show`, `python`, `rank`, `lag`, `lead`, `cumsum`, `cummean`, `cummin`, `cummax`, `rolling`, `end`
- **Clause keywords:** `from`, `where`, `as`, `on`, `by`, `rows`, `cols`, `head`
- **Operators:** `==`, `!=`, `>`, `<`, `>=`, `<=`, `in`, `not`, `between`, `contains`, `startswith`, `endswith`, `and`, `or`
- **Sort modifiers:** `asc`, `desc`
- **Aggregation functions:** `mean`, `min`, `max`, `sum`, `count`, `avg`, `median`, `std`
- **Merge types:** `left`, `right`, `inner`, `outer`
- **Strings, numbers, comments**

Syntax highlighting also activates inside `%%pivotal` blocks in Python files and notebooks.

---

## Autocomplete

The extension provides context-aware completions:

| Context | Completions |
|---------|-------------|
| Start of line | All DSL commands |
| After `with ... as` | Available table names |
| After `group by`, `sort`, `filter`, etc. | Column names for the active table |
| After `group by` | Aggregation functions |
| After `plot` | Chart types (`bar`, `line`, `scatter`, `hist`, `box`, `area`) |

Table and column completions are powered by a `pivotal_autocomplete.json` file in the workspace root. This file is generated automatically when you run Pivotal code via the Jupyter Interactive Window integration. You can also create it manually:

```json
{
  "tables": {
    "sales": ["date", "product", "region", "amount", "quantity"],
    "customers": ["customer_id", "name", "email", "region"]
  }
}
```

---

## Commands and keyboard shortcuts

Access commands via the **Command Palette** (`Ctrl+Shift+P` / `Cmd+Shift+P`) or the editor toolbar.

| Command | Shortcut | Description |
|---------|----------|-------------|
| **Execute File** | `Ctrl+F5` / `Cmd+F5` | Run the entire `.pivotal` file in a terminal |
| **Execute in Interactive Notebook** | — | Send the file to a Jupyter Interactive Window |
| **Execute Selection in Interactive Notebook** | `Ctrl+Shift+F5` / `Cmd+Shift+F5` | Send selected text to a Jupyter Interactive Window |
| **Compile to Python File** | — | Compile to a `.py` file in the same directory |

Commands also appear in the editor title bar run menu and (for selection execution) the right-click context menu.

---

## Execute File

Runs the `.pivotal` file using the Pivotal CLI:

```bash
python -m pivotal "path/to/file.pivotal"
```

Output appears in an integrated terminal.

---

## Execute in Interactive Notebook

Splits the file by `#%%` section markers and sends each section as a `%%pivotal` cell to an open [Jupyter Interactive Window](https://code.visualstudio.com/docs/datascience/jupyter-notebooks). This gives you the Pivotal viewer experience inside VS Code.

Use `#%%` to divide a `.pivotal` file into sections:

```pivotal
#%% Load data
load "data/sales.csv" as sales
load "customers.csv" as customers

#%% Analysis
with sales as summary
    left merge customers on customer_id
    group by region
        agg sum amount as total
```

---

## Execute Selection

Select any portion of a `.pivotal` file and press `Ctrl+Shift+F5` to send just that selection to the Interactive Window. Useful for iterating on specific operations.

---

## Compile to Python

Compiles the current file to a `.py` file:

```bash
python -m pivotal --compile "path/to/file.pivotal"
```

The `.py` file is created in the same directory and opened automatically in VS Code.

---

## Language configuration

The extension configures:

- **Comments:** `#` and `--` for line comments; `/* */` for block comments
- **Auto-closing pairs:** `"..."`, `[...]`
- **Auto-indent:** increases after block-starting statements (`with`, `load`, `group by`, `pivot`, `unpivot`, `table`, `python`, etc.)
