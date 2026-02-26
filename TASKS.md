# Project: Pivotal

  ## Current focus
 
  ## Backlog

  - [ ] For pivotal statements that currently have multiple keywords pick and chose one only. Ensure that these changes are reflected in the tests, examples and readme. I think use merge rather than join and df rather than dataframe.

  - [ ] Keyword collision validation — detect when user-defined names clash with Pivotal reserved words and emit clear errors/warnings:
    - **Hard error** at parse time: DataFrame name in `df <name> from ...` is a reserved keyword (e.g. `df filter from sales`)
    - **Hard error** at parse time: column name in `assign <col> = expr` is a reserved keyword (e.g. `assign select = price * 0.9`)
    - **Warning only** at runtime: a column loaded from data (CSV, Excel etc.) has the same name as a reserved keyword — not the user's fault, but flag it so they know to use `python` if they need to reference it

  - [ ] Save session as a Frictionless Data Package - see implementation plan. (note this is closely realted to below 'start' command session management)
  
  - [ ] add a `start` command — declare package membership, create/open package, configure autosave and styles — see implementation plan below. (closely related to save session as data package point above)

  - [ ] Update the README.md language syntax and API sections. I'm not sure if they are complete. I am wondering if we need a standalone documentation page that details each command (either as well as or instead of the content currently in the README)

  ## Ideas (not ready to be implemented)

  - [ ] Context aware auto-complete in jupyter lab and vs code. For example, when editing pivotal text on a new line autocomplet the command keywords, but if you are within a statement then autocomplete the apprioate thing (dataframe name or column name) drawing on the existing .pivotal_autocomple.json file - see implementation plan below 
  
  - [ ] String functions in `assign` expressions — see implementation plan below.

  - [ ] Polars support — see implementation plan below.
  
  - [ ] Python function calls in Pivotal — define functions in Python, call them from `assign` and a new `apply` statement — see implementation plan below.

  - [ ] Enhanced plot syntax — style files, faceted subplots (`by`), `save` param — see implementation plan below.

  - [ ] VSCODE extension: Fix bug where pivotal code is embedded inside a *.py file. This currently works fine (it runs inside the interactive notebook, and has syntax highlighting in the editor as expected) but in the editor the pivotal code section has pylance errors (red underlines) as it is still expecting python code. Is there a way to fix this...
  
  - [ ] Table print/export support, json formating....   

  - [ ] Jupyter lab, is there a way to make cells pivotal by default, with some kind of toggle...

  - cast / type conversion — type coercion is fiddly and infrequent. Python is the right place for it.

  - load multiple files or a folder and merge or concat, apply type conversion on load (use json metadata or something to guide this). Perhaps simple load then add settings sub command or, modify metadata then reload using settings in metadata??

  - describe / sample — pure exploration helpers. One-liners in Python (df.describe(), df.sample(10)), not worth adding to the grammar.

  - melt / unpivot — complex, infrequent, and the syntax would be awkward. Python is clearly the right escape hatch.

  - Window / rolling functions — i.e., 

rolling cola as colamean, colb as cobmean
  by time
  window 3
  agg mean

  - head / tail — in a notebook context this is about quick exploration. limit 10 at the end of a pipeline to preview results is very natural and saves a Python cell.

  - connections to other data sources...


## Implementation plans

### Context-aware autocomplete

**Goal:** Provide intelligent completions in VS Code and JupyterLab that are aware of (1) where the cursor is in the Pivotal grammar and (2) what tables and columns exist in the session — drawing on the existing `.pivotal_autocomplete.json` file.

**The autocomplete file:**

`.pivotal_autocomplete.json` already exists and is written by the Pivotal engine after each run. Its current shape:
```json
{
  "tables": {
    "sales": {
      "columns": ["id", "product", "price", "quantity", "category"],
      "dtypes":  { "price": "float64", ... },
      "shape":   [100, 5]
    }
  },
  "current_table": "sales",
  "timestamp": "..."
}
```
No changes to the file format are needed — it already has everything required.

**Approach — editor-side context parsing (no Language Server):**

Context detection is implemented in TypeScript inside each editor extension, reading the autocomplete file from disk. A full Language Server Protocol (LSP) server would be more powerful but is significantly more complex and overkill for this use case. Both editors share the same context-detection logic (extracted into a shared module where possible).

**Context detection algorithm:**

Walk backwards from the cursor line to find the current table in scope (nearest `df <name>` or `load <name>`). Then inspect the current line's indentation and first keyword to decide what to offer:

| Position | Offer |
|---|---|
| Line start, no indent | Command keywords: `df`, `load`, `filter`, `select`, `sort`, `assign`, `group`, `merge`, `concat`, `pivot`, `plot`, `drop`, `rename`, `fillna`, `dropna`, `distinct`, `python` |
| After `df` | Existing table names from autocomplete file |
| After `df <name> from` | Existing table names |
| After `select`, `drop`, `rename`, `sort`, `distinct`, `filter` (column position) | Column names for current table |
| After `assign` (expression position, after `=`) | Column names for current table |
| After `group by` | Column names for current table |
| After `agg` | Agg keywords: `mean`, `sum`, `count`, `min`, `max`, `median`, `std` |
| After `agg <func>` | Column names for current table |
| After `merge`, `join`, `concat` | Existing table names |
| After `plot` | Chart types: `line`, `bar`, `scatter`, `hist`, `box`, `area` |
| After plot sub-params `x`, `y`, `by`, `c` | Column names for current table |
| After `load` | Nothing (free-form path) |

**VS Code implementation (`editors/vscode/src/extension.ts`):**

Register a `CompletionItemProvider` for the `pivotal` language:
```typescript
vscode.languages.registerCompletionItemProvider(
  { language: 'pivotal' },
  { provideCompletionItems(document, position) {
      const autocomplete = loadAutocompleteFile(document.uri);
      const ctx = detectContext(document, position, autocomplete);
      return buildCompletionItems(ctx);
  }},
  ' ', '\n'  // trigger characters
);
```

- `loadAutocompleteFile()` reads `.pivotal_autocomplete.json` from the same directory as the open file, with a simple in-memory cache invalidated by file `mtime`
- `detectContext()` walks the document text to determine table in scope and what kind of completion is needed
- `buildCompletionItems()` returns `vscode.CompletionItem[]` with appropriate `kind` (Keyword, Field, Value)

**JupyterLab implementation (`editors/jupyterlab/src/index.ts`):**

CodeMirror 6's `@codemirror/autocomplete` package is already in the project's yarn cache. Register a `CompletionSource`:
```typescript
import { autocompletion, CompletionContext } from '@codemirror/autocomplete';

function pivotalCompletions(context: CompletionContext) {
  const autocomplete = await fetchAutocompleteFile();  // JupyterLab Contents API
  const ctx = detectContext(context.state, context.pos, autocomplete);
  if (!ctx.options.length) return null;
  return { from: context.pos - ctx.wordSoFar.length, options: ctx.options };
}
```

Add `autocompletion({ override: [pivotalCompletions] })` to the CodeMirror extension list. For `%%pivotal` notebook cells, the same extension is added via the existing `magic-highlight` compartment mechanism.

Reading the autocomplete file in JupyterLab: use `fetch('/api/contents/.pivotal_autocomplete.json')` against the JupyterLab Contents API, with the path relative to the notebook's directory.

**Shared context detection module:**

Extract `detectContext()` into `editors/shared/context.ts` (or duplicate with identical logic if the build setup makes sharing awkward). Inputs: document text as a string, cursor offset, parsed autocomplete data. Output: `{ type: 'command' | 'column' | 'table' | 'agg' | 'charttype', options: string[], wordSoFar: string }`.

**Files to create / change:**
- `editors/vscode/src/extension.ts` — add `registerCompletionItemProvider` and helper functions
- `editors/jupyterlab/src/index.ts` — add `autocompletion()` extension and `pivotalCompletions` source
- `editors/jupyterlab/package.json` — add `@codemirror/autocomplete` to dependencies (already cached)
- `editors/shared/context.ts` *(new)* — shared context detection logic (optional, depends on build setup)

**What triggers a refresh of the autocomplete file:**

The file is already written by `pivotal/__main__.py` after each execution. No changes needed there. Completions are as fresh as the last run — acceptable, and consistent with how Python type stubs work. A future improvement could be a `--watch` mode that updates the file on save without a full run.

**Effort estimate:**
- ~3–4 hours for Claude Code: context detection logic is the main work; editor registration is boilerplate
- ~20–30 hours for a human developer: unfamiliar VS Code and CodeMirror 6 APIs add significant ramp-up time

**Biggest risk:** JupyterLab — fetching the autocomplete file path correctly relative to the notebook's working directory, especially when notebooks are opened from different locations. Start with VS Code (simpler file access) and validate the context detection logic there first. **Note:** If the `start` command is implemented first, this risk largely disappears — the extension scans for `start` in the file to find the package path, and the autocomplete file lives at a known location inside the package.

---

### Polars backend

**Goal:** Generate Polars code instead of (or alongside) pandas, giving users who prefer Polars the same high-level DSL experience.

**Current state:**
- The dispatch mechanism `generate_{type}_{backend}` is already in place in `dsl_parser.py`.
- `DSLParser(backend='polars')` already switches the backend — the architecture is ready.
- One generator already exists: `generate_load_table_polars` (reads CSV/Excel/Parquet with `pl.read_csv` etc.).
- All other statement types currently raise `NotImplementedError` or fall through to pandas.

**Core challenge — expression conversion:**
Pandas generators produce string expressions passed to `df.eval()` or `df.query()`. Polars requires `pl.col('name')` expression objects. A shared helper `_expr_to_polars(expr_str)` that walks a simple AST (or uses a regex substitution) to turn identifier tokens into `pl.col('identifier')` will be needed for `filter` and `assign`.

**Statement-by-statement implementation plan:**

| Statement | Current pandas output | Polars equivalent |
|---|---|---|
| `load` | `pd.read_csv(...)` | `pl.read_csv(...)` ✅ already done |
| `copy_table` (`df X from Y`) | `X = Y.copy()` | `X = Y.clone()` |
| `filter` | `df.query('expr')` | `df.filter(pl.Expr)` — convert condition via `_expr_to_polars()` |
| `select` | `df[['a','b']]` | `df.select(['a','b'])` |
| `sort` | `df.sort_values(...)` | `df.sort(['col'], descending=[True/False])` |
| `assign` | `df.eval('col = expr')` | `df.with_columns((expr).alias('col'))` — convert via `_expr_to_polars()` |
| `group + agg` | `df.groupby().agg(...)` | `df.group_by([...]).agg([pl.col(...).mean().alias(...)])` |
| `merge/join` | `pd.merge(df, other, how=..., on=...)` | `df.join(other, on=..., how=...)` |
| `pivot` | `df.pivot_table(...)` | `df.pivot(on=..., index=..., values=..., aggregate_function=...)` |
| `drop` | `df.drop(columns=[...])` | `df.drop([...])` |
| `rename` | `df.rename(columns={...})` | `df.rename({...})` |
| `fillna` | `df.fillna(value)` | `df.fill_null(value)` |
| `dropna` | `df.dropna(subset=[...])` | `df.drop_nulls(subset=[...])` |
| `distinct` | `df.drop_duplicates(subset=[...])` | `df.unique(subset=[...])` |
| `concat` | `pd.concat([df, other])` | `pl.concat([df, other])` |
| `python` | raw Python block unchanged | same — user is responsible for using `pl` API |

**`_expr_to_polars()` helper approach:**
- Tokenise the expression string (re-use the same Lark grammar or a lightweight regex pass).
- Replace bare identifiers (non-quoted, non-numeric) with `pl.col('identifier')`.
- Keep string literals, numbers, operators, and parentheses as-is.
- For `filter` conditions with `and`/`or`, wrap each side and combine with `&`/`|`.
- For `between [lo, hi]`: generate `pl.col('x').is_between(lo, hi)`.
- For `contains`/`startswith`/`endswith`: generate `pl.col('x').str.contains(...)` etc.

**New files / changes:**
- `pivotal/dsl_parser.py`: Add `generate_*_polars` methods for every statement type listed above; add `_expr_to_polars()` private helper.
- `tests/test_commands_polars.py`: Mirror tests from `test_commands.py` using `DSLParser(backend='polars')` and a Polars `pytest.importorskip('polars')` guard.
- `README.md`: Add a brief note that `DSLParser(backend='polars')` is supported.

**What stays as `python` escape:**
- Any operation not in the table above (window functions, complex string formatting, etc.).
- The `python` block passes code through verbatim regardless of backend.

**Effort estimate:**
- ~3–4 hours for Claude Code (one focused session): implement all generators, write the helper, add tests, update README.
- ~25–40 hours for a human developer: includes learning the Polars API, debugging edge cases in expression conversion, and writing comprehensive tests.

**Biggest risk:** Expression conversion for complex `filter`/`assign` conditions (nested parens, mixed arithmetic and boolean logic). Start with simple cases and add test coverage before handling edge cases.

---

### Enhanced plot syntax

**Goals:** Style files to separate formatting from data params; `by` keyword for faceted subplots; `save` param to write chart files; Polars plot backend via hvPlot.

**Proposed syntax:**
```
plot bar
    x category
    y quantity
    by region
    cols 2
    style reports
    save category_by_region
```

**Rule for what goes where:**
- Inline params = structural decisions: what to plot (`x`, `y`), how to facet (`by`, `cols`), chart type
- Style file = cosmetic decisions: sizes, labels, colours, fonts, grid, tight layout
- Inline params override style file values where both are specified

**Style files:**

Named styles are defined in the session's `datapackage.json` under a `styles` key, or referenced as an external JSON file path:

```json
{
  "styles": {
    "default": { "figsize": [12, 8], "grid": true },
    "reports": { "figsize": [15, 10], "tight_layout": true, "xlabel_fontsize": 12 }
  }
}
```

In Pivotal: `style reports` (named) or `style "path/to/style.json"` (file path).

**`by` / faceted subplots:**
- `by region` creates one subplot per unique value in the `region` column
- `cols 2` sets number of columns; rows are calculated automatically
- Empty subplot cells are hidden
- Each subplot title defaults to the category value

Generated code:
```python
import matplotlib.pyplot as plt
_vals = df['region'].unique()
_cols = 2
_rows = -(-len(_vals) // _cols)
fig, axes = plt.subplots(_rows, _cols, figsize=(15, 10))
axes = axes.flatten()
for i, _val in enumerate(_vals):
    df[df['region'] == _val].plot(kind='bar', x='category', y='quantity',
                                   ax=axes[i], title=str(_val))
for ax in axes[len(_vals):]:
    ax.set_visible(False)
plt.tight_layout()
plt.show()
```


**Polars plot backend:**
- Polars `.plot` accessor uses hvPlot (Bokeh-based), not matplotlib
- Chart type becomes a method rather than a `kind=` kwarg: `df.plot.bar(x='category', y='quantity')`
- Output is interactive HTML (self-contained) rather than a static matplotlib figure — a feature, not a limitation
- `save` for Polars uses hvPlot's export to write standalone HTML
- Requires `hvplot` as an additional dependency

**Grammar changes (`dsl_parser.py`):**
- `by`, `cols`, `style`, `save` added as recognised structural params in `plot_statement` — intercepted before forwarding remaining kwargs to `df.plot()`

**Code generator changes:**
- `generate_plot_pandas`: split params into structural (`by`, `cols`, `style`, `save`) vs. cosmetic (everything else); load style JSON if specified; generate loop code when `by` is present; append savefig if `save`
- `generate_plot_polars`: new method using `df.plot.<kind>(...)` via hvPlot; handle `save` with hvPlot HTML export

---

### `start` command and package session management

**Goal:** A `start` command at the top of a `.pivotal` file declares which package the file belongs to, creates the package on first run, and opens it on subsequent runs — giving the whole session a persistent identity and a known location on disk.

**Basic syntax:**
```
start "my_analysis"
```
Minimal form — uses a default path (sibling folder named after the package, relative to the `.pivotal` file).

**Full syntax:**
```
start "my_analysis"
    path "~/projects/data"
    title "My Sales Analysis"
    autosave True
    autosave_limit 500000
    format parquet
    style reports
        figsize 15 10
        grid True
        tight_layout True
    style corporate "~/.pivotal/styles/corporate.json"
```

**Create vs. open — idempotent:**
- Package doesn't exist → create folder structure and minimal `datapackage.json`, then continue
- Package exists → open it, then merge any params from `start` into `datapackage.json`
- Re-running is always safe — like `mkdir -p`

**Merge behaviour on existing packages:**

Params present in `start` are written to `datapackage.json`. Params not mentioned in `start` are left untouched. This means the `.pivotal` file is the source of truth for config — changing `title` in `start` and re-running updates the package.

| Param | Persisted to `datapackage.json` | Notes |
|---|---|---|
| `title` | Yes | Updated on every run if specified |
| `style` (inline definition) | Yes | Merged — new styles added, existing names overwritten |
| `style` (file path import) | Yes — values copied in | Package stays self-contained, no external dependency |
| `format` | Yes | `csv` (default) or `parquet` |
| `autosave` | No — session only | Controls this run only |
| `autosave_limit` | No — session only | Controls this run only |

**Multi-file packages:**

One file "owns" the package definition and specifies all config params. Other files that share the package just declare membership:
```
# cleaning.pivotal — owns the package
start "my_analysis"
    path "~/projects/data"
    title "My Sales Analysis"
    style reports
        figsize 15 10
```
```
# analysis.pivotal — joins the package, no config changes
start "my_analysis"

load all
```

If two files both define the same named style, the last one to run wins — a warning is emitted: *"style 'reports' already defined in datapackage.json — overwriting."*

**`load` from package:**

`load <name>` with no path looks up `<name>` in the active package's `data/` folder:
```
load clean              ← loads data/clean.csv (or .parquet) from the package
load all                ← loads every table in data/ into the session namespace
```
`load all` is the standard pattern at the top of a dependent file — restores the full state saved by a previous file without re-running the pipeline.

**`autosave`:**
- `autosave True` — saves any DataFrame assigned with `df <name> from ...` automatically after each statement
- `autosave False` (default) — nothing written unless `save` is called explicitly
- `autosave_limit 500000` — skips autosave for tables with more than N rows; emits a warning
- Format controlled by `format` param (`csv` or `parquet`); parquet recommended for large files

**`save` command simplified:**

With `start` having established the package path and name, `save` needs no arguments:
```
save
```
With `start` active this writes all tables, code, and charts to the package. The named form `save "path"` still works for one-off saves without a `start`.

**Style resolution order:**

When a `plot` statement (or any other statement) references `style reports`:
1. Package styles in `datapackage.json` (highest priority — set by `start`)
2. User global styles in `~/.pivotal/styles/reports.json`
3. Built-in Pivotal themes (`default`, `minimal`, `dark`, `print`)
4. Error if not found

**How `start` solves the autocomplete path problem:**

The autocomplete plan flags JupyterLab file path resolution as the biggest risk. With `start`, the editor extensions scan the `.pivotal` file for a `start` command, extract the package path, and look for the autocomplete file at `<package_path>/.pivotal_autocomplete.json`. No working-directory guessing needed. Files without `start` fall back to the current working directory as before.

**Grammar changes (`dsl_parser.py`):**
- New `start_statement: "start" STRING (_NL _INDENT start_params _DEDENT)?`
- `start_params` covers `path`, `title`, `autosave`, `autosave_limit`, `format`, and `style` blocks
- Must be the first statement in the file (enforced at parse or execution time with a clear error)

**New module (`pivotal/package.py`):**
- `Package` class: `open_or_create(name, path)`, `merge_config(params)`, `load_table(name)`, `load_all()`, `save_table(name, df, format)`, `save_all(namespace)`, `resolve_style(name)`
- Keeps the session's active package reference; `__main__.py` passes it through to the code generator

**Code generator changes:**
- `generate_start`: calls `Package.open_or_create()`, sets session package context
- `generate_load_table`: check for no-path form — if active package, delegate to `Package.load_table()`
- `generate_save`: if active package, delegate to `Package.save_all()`; otherwise fall back to existing behaviour

**Effort estimate:**
- ~4–6 hours for Claude Code: new grammar rule is simple; most work is in `package.py` and wiring it through the code generator
- ~30–45 hours for a human developer: file I/O, JSON merging, path resolution, and the style lookup chain all have edge cases

---

### Save session as Frictionless Data Package

**Goal:** Save the entire Pivotal session as a self-contained, reproducible package — data tables, source code, compiled Python, charts, and style files in one folder.

**Package structure:**
```
my_analysis/
  datapackage.json          ← Frictionless descriptor + named styles
  code/
    analysis.pivotal        ← Pivotal source (the reproducible spec)
    analysis.py             ← compiled Python
  data/
    sales.csv
    summary.csv
  charts/
    product_prices.png      ← matplotlib static charts
    category_bar.html       ← hvPlot/Plotly interactive charts
```

**Key design decision — no Frictionless `views`:**
The `.pivotal` source file is a better chart specification than a translated Frictionless views entry. It's more expressive, already exists, and anyone with Pivotal installed can re-run it to regenerate all outputs. No translation layer is needed.

**`datapackage.json` structure:**
```json
{
  "name": "my-analysis",
  "title": "My Analysis",
  "resources": [
    { "name": "sales",     "path": "data/sales.csv",                "mediatype": "text/csv" },
    { "name": "summary",   "path": "data/summary.csv",              "mediatype": "text/csv" },
    { "name": "source",    "path": "code/analysis.pivotal",         "mediatype": "text/x-pivotal" },
    { "name": "compiled",  "path": "code/analysis.py",              "mediatype": "text/x-python" },
    { "name": "chart-prices", "path": "charts/product_prices.png",  "mediatype": "image/png" },
    { "name": "chart-cat",    "path": "charts/category_bar.html",   "mediatype": "text/html" }
  ],
  "styles": {
    "default": { "figsize": [12, 8], "grid": true },
    "reports": { "figsize": [15, 10], "tight_layout": true }
  }
}
```

**Pivotal `save` command syntax:**
```
save "my_analysis"
```
or with options:
```
save "my_analysis"
    title "My Analysis"
```

**What the save command does:**
1. Creates the folder structure (`data/`, `code/`, `charts/`)
2. Writes all DataFrames in the session namespace to `data/` as CSVs
3. Copies the `.pivotal` source and compiled `.py` to `code/`
4. Copies any chart files (produced by `save` params on `plot` statements) to `charts/`
5. Writes `datapackage.json` with all resources and named styles from the session config

**Dependency on `start` command:**
The `save` plan depends on the `start` implementation. `start` establishes the package path, name, autosave config, and styles — `save` just writes to that already-known location. Implement `start` first.

**Remaining open questions:**
- Should tables loaded from external sources be re-saved into the package? Probably yes — the package should be self-contained and not depend on the original file paths.
- Folder or zip? Folder for development, optional zip export for sharing (`save zip`).

---

### Python function calls in Pivotal

**Goal:** Allow Python functions defined in the session namespace to be called from Pivotal syntax, without trying to define functions in Pivotal itself. Python defines the tools; Pivotal orchestrates them.

**Design principle:** Function *definition* stays in Python — it's the right tool for that. Pivotal only needs to support function *calling* in two specific contexts.

**Context 1 — Column/Series transforms in `assign`:**

```
python
    def clean_price(s):
        return s.str.replace('£', '').astype(float)

    def initials(s):
        return s.str[0].str.upper()

df sales
    assign price = clean_price(price)
    assign abbr = initials(name)
```

The code generator detects that `clean_price` is not a known built-in and generates:
```python
sales['price'] = clean_price(sales['price'])
```
rather than routing through `df.eval()`.

This is a natural extension of the string functions plan — same mechanism, user-defined rather than built-in.

**Context 2 — DataFrame-level transforms with `apply`:**

```
python
    def normalize(df):
        df = df.copy()
        df['price'] = (df['price'] - df['price'].mean()) / df['price'].std()
        return df

    def remove_outliers(df):
        return df[df['price'].between(df['price'].quantile(0.05), df['price'].quantile(0.95))]

df sales
    apply normalize
    apply remove_outliers
    group by category
        agg mean price
```

Generates `sales = normalize(sales)` then `sales = remove_outliers(sales)`. The function is expected to take a DataFrame and return a DataFrame. Simple, fits the pipeline model, no grammar complexity.

**What is explicitly out of scope:**
- Defining functions in Pivotal syntax — the function body would be Python anyway, adding a wrapper syntax buys nothing
- Custom aggregation functions in `group by` (`agg my_func price`) — requires distinguishing user functions from built-in agg names at parse time; the `python` escape is the right answer here for now

**Grammar changes (`dsl_parser.py`):**
- `assign` expression: when the expression is of the form `IDENTIFIER "(" IDENTIFIER ")"` and the outer name is not a known built-in string function, treat it as a user function call
- New `apply_statement: "apply" IDENTIFIER _NL` — single identifier, no sub-params needed

**Code generator changes:**
- `generate_assign_pandas`: detect user function call pattern; generate `df['col'] = func(df['other_col'])` instead of `df.eval(...)`
- `generate_apply_pandas`: new method; generates `df_name = func_name(df_name)`
- `generate_apply_polars`: same pattern — `df_name = func_name(df_name)`

**Tests to add (`tests/test_commands.py`):**
- `apply` with a function that adds a column
- `apply` with a function that filters rows
- `assign col = user_func(col)` where `user_func` is in the namespace
- Confirm existing built-in string functions (`upper`, `lower` etc.) still resolve correctly and are not mistaken for user functions

**Effort estimate:**
- ~1–2 hours for Claude Code: grammar is minimal, code generator changes are localised
- ~8–12 hours for a human developer: most time spent on the user-function detection logic in `assign` and edge case testing

---

### String functions in `assign` expressions

**Goal:** Allow SQL-style string operations inside `assign` statements without needing the `python` escape hatch.

**Proposed syntax:**
```
assign name = last_name + ", " + first_name   -- concatenation with + operator
assign code = upper(category)
assign abbr = left(name, 3)
assign end  = right(ref, 4)
assign slug = lower(trim(title))
assign note = substr(description, 0, 100)
assign n    = len(name)
assign fixed = replace(notes, "N/A", "")
```

**Design decisions:**
- Use `+` for string concatenation (Python-style). Disambiguate at code-gen time: if either operand is a quoted string literal, generate direct pandas string addition rather than `df.eval()`.
- Use SQL-style named functions: `upper`, `lower`, `trim`, `ltrim`, `rtrim`, `left`, `right`, `substr`, `len`, `replace`. Functions can be nested: `upper(left(name, 1))`.
- Keep existing arithmetic `assign` (e.g. `assign revenue = price * quantity`) unchanged — still routes through `df.eval()`.

**Grammar changes** (`dsl_parser.py`):
- Extend `expression` rule to add two new branches:
  - `string_concat_expr: string_arg ("+" string_arg)+` where `string_arg: IDENTIFIER | STRING`
  - `string_func_expr: STRING_FUNC "(" string_func_arg ("," string_func_arg)* ")"` (supports nesting by making `string_func_arg` recursive)
- Add terminal: `STRING_FUNC: "upper" | "lower" | "trim" | "ltrim" | "rtrim" | "left" | "right" | "substr" | "len" | "replace"`
- Add `STRING_FUNC` to the JupyterLab and VS Code syntax highlighter keyword lists as builtins.

**Code generator changes** (`dsl_parser.py` Transformer):
- Add `string_concat_expr` transformer: iterate operands, wrap `IDENTIFIER` tokens as `df['col']`, leave `STRING` tokens as Python string literals, join with ` + `.
- Add `string_func_expr` transformer: map each `STRING_FUNC` to its pandas equivalent:

  | Function | Generated pandas |
  |---|---|
  | `upper(col)` | `df['col'].str.upper()` |
  | `lower(col)` | `df['col'].str.lower()` |
  | `trim(col)` | `df['col'].str.strip()` |
  | `ltrim(col)` | `df['col'].str.lstrip()` |
  | `rtrim(col)` | `df['col'].str.rstrip()` |
  | `left(col, n)` | `df['col'].str[:n]` |
  | `right(col, n)` | `df['col'].str[-n:]` |
  | `substr(col, s, n)` | `df['col'].str[s:s+n]` |
  | `len(col)` | `df['col'].str.len()` |
  | `replace(col, a, b)` | `df['col'].str.replace('a', 'b', regex=False)` |

- For nesting (e.g. `upper(left(name, 1))`), each transformer call returns a pandas expression string that can be wrapped by the outer call.

**Tests to add** (`tests/test_commands.py`):
- `assign` with `+` concatenation (col + literal + col)
- Each named function individually
- Nested functions (`upper(left(...))`)
- Mixed: existing arithmetic `assign` still works unchanged
- `where` clause combined with a string `assign`

**Scope that stays as `python` escape:**
- f-strings / format strings
- Regex operations
- Multi-column conditional string logic

---

## Completed

  

  - [x] Drop columns e.g., drop colA, colB  -> dfA.drop(["colA", "colB"])

  - [x] fillna / dropna — missing value handling is arguably the single most common data cleaning step. Having to drop to Python for this every time would be a constant friction point. These belong in the language.

  - [x] dedupe — keyword is `distinct`, consistent with SQL and R/dplyr

  - [x] concat — combining two tables vertically (e.g. appending monthly CSVs)

  - [x] rename — i.e., rename colA as newcol

  - [x] between / contains in filters — `between [lo, hi]`, `contains`, `not contains`, `startswith`, `endswith`

  - [x] load excel and parquet format data sources — file format auto-detected from suffix (.xlsx, .xls, .parquet, .csv). Works for both literal paths and variable paths (runtime detection).

  - [x] VS Code: interactive notebook opens to the right and reuses existing window on re-run

  - [x] VS Code: compile .pivotal file to .py file command (`python -m pivotal --compile`)

  - [x] VS Code: extension README

  - [x] JupyterLab: file browser icon for .pivotal files

  - [x] JupyterLab: extension manager icon (pivotal_logo.svg)
