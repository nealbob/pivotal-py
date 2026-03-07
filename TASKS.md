# Project: Pivotal

  ## Current focus


  ## Backlog


  ## Ideas (not ready to be implemented)

  - [x] Enhanced plot syntax — style files, faceted subplots (`by`) and style files see implementation plan below.
  
  - [ ] String functions in `assign` expressions — see implementation plan below.

  - [ ] Polars support — see implementation plan below.

  - [ ] Object Viewer panel — see implementation plan below.

  - [ ] In addition to charts I want to support generattion of publication ready tables using the Great Tables package. Can you develop an implementation plan for this.

  - [ ] VSCODE extension: Fix bug where pivotal code is embedded inside a *.py file. This currently works fine (it runs inside the interactive notebook, and has syntax highlighting in the editor as expected) but in the editor the pivotal code section has pylance errors (red underlines) as it is still expecting python code. Is there a way to fix this...

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

**Polars plot backend:**
- Polars `.plot` accessor uses hvPlot (Bokeh-based), not matplotlib
- Chart type becomes a method rather than a `kind=` kwarg: `df.plot.bar(x='category', y='quantity')`
- Output is interactive HTML (self-contained) rather than a static matplotlib figure — a feature, not a limitation
- `save` for Polars uses hvPlot's export to write standalone HTML
- Requires `hvplot` as an additional dependency

**Grammar changes (`dsl_parser.py`):**
- `by`, `cols`, `style`, added as recognised structural params in `plot_statement` — intercepted before forwarding remaining kwargs to `df.plot()`

**Code generator changes:**
- `generate_plot_pandas`: split params into structural (`by`, `cols`, `style`, `save`) vs. cosmetic (everything else); load style JSON if specified; generate loop code when `by` is present; append savefig if `save`
- `generate_plot_polars`: new method using `df.plot.<kind>(...)` via hvPlot; handle `save` with hvPlot HTML export



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

**Goals:** Style files to separate formatting from data params; `by` keyword for faceted subplots; Polars plot backend via hvPlot.

**Proposed syntax:**
```
plot bar
    x category "Category"
    y quantity "Quantity"
    by region "Region"
    cols 2
    style reports
```

**Rule for what goes where:**
- Inline params = structural decisions: what to plot (`x`, `y`), how to facet (`by`, `cols`), chart type
- Style file = cosmetic decisions: sizes, labels, colours, fonts, grid, tight layout
- Inline params override style file values where both are specified

**Style files:**

What format to use for style files (matplotlib style files type for now i guess)


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

---

### Object Viewer Panel

**Goal:** A persistent right-side panel in JupyterLab that receives DataFrames and charts from every `%%pivotal` cell execution and displays them interactively — an alternative to inline cell output. Navigable history lets the user cycle through previous outputs without re-running cells.

---

#### User experience

- Panel opens automatically (or via command/shortcut) and docks to the right side of the JupyterLab workspace.
- After each `%%pivotal` cell runs, the panel updates once per object produced, in execution order. If a cell produces a DataFrame then a chart, the panel shows the DataFrame briefly then the chart — the chart is visible at cell completion because it was produced last. The DataFrame is one step back in history.
- If a cell modifies an existing object (same variable name, new content), the updated version is pushed as a new history entry — so the old version remains accessible via Back, but the panel always shows the freshest state at completion.
- Panel header shows: object name, type (`DataFrame` / `Chart`), shape or chart kind, and a position indicator (e.g. `4 / 7`).
- Back / Forward buttons (and keyboard shortcuts) cycle through the cached history.
- Only one object is shown at a time. Cache holds the last 50 outputs (configurable constant).

---

#### Data flow

```
%%pivotal cell executes
    → magic.py sends comm message to frontend
        → JupyterLab extension receives message
            → Panel widget renders DataFrame or chart
```

---

#### Grid library choice

**Recommended: `@lumino/datagrid`** — ships with JupyterLab 4, zero extra npm dependencies, virtual scrolling for large tables, column sorting on header click. This is the same grid used by JupyterLab's built-in CSV viewer.

Alternative: **AG Grid Community** (MIT) — richer filtering/grouping UI, adds ~500 KB to bundle. Worth revisiting if column filtering becomes a requirement.

#### Row rendering and transfer limit

`@lumino/datagrid` uses **virtual rendering** — it only paints the rows currently in the viewport, so there is no inherent rendering limit. A 500 000-row DataFrame renders just as smoothly as a 100-row one from the grid's perspective.

The practical constraint is the **comm payload** (JSON serialisation over the WebSocket). Sending a very wide or very long DataFrame as JSON can be slow and memory-heavy. The plan:

- Default transfer limit: **10 000 rows**. This is large enough to be useful for most exploratory work and fast enough to feel instant.
- A **row limit control** is surfaced in the panel footer (e.g. a small input: `Show: [10000] rows`). The user can raise or lower it; changing the value re-requests data from the kernel via a reply comm message.
- If the DataFrame is truncated, the footer shows a clear notice: `Showing 10 000 of 284 391 rows`.
- The `MAX_ROWS` constant in `_PivotalViewer` becomes the default; the panel can override it per request.

---

#### Chart interactivity

Matplotlib figures are static; the panel will render them as high-res base64 PNG with:
- Zoom in / out buttons (CSS `transform: scale()`)
- Click-and-drag pan when zoomed (pointer events on a `<canvas>` or `<div>`)

Optional V2 upgrade: use **mpld3** (`pip install mpld3`) to convert a matplotlib figure to an interactive D3.js chart. Richer but requires an extra Python dependency and larger payloads.

---

#### Python side — `magic.py`

Add a `_PivotalViewer` helper class (instantiated once per `PivotalMagics` instance):

```python
class _PivotalViewer:
    MAX_ROWS = 2000   # rows sent to frontend

    def __init__(self, shell):
        self._shell = shell
        self._comm = None

    def _ensure_comm(self):
        if self._comm is not None:
            return
        try:
            from ipykernel.comm import Comm
            self._comm = Comm(target_name='pivotal_viewer')
            self._comm.open()
        except Exception:
            pass   # viewer not installed or not in a kernel context

    def send_dataframe(self, name: str, df):
        self._ensure_comm()
        if self._comm is None:
            return
        import pandas as pd
        truncated = len(df) > self.MAX_ROWS
        payload = df.head(self.MAX_ROWS)
        self._comm.send({
            'type': 'dataframe',
            'name': name,
            'records': payload.to_dict('records'),
            'columns': list(payload.columns),
            'dtypes': {c: str(t) for c, t in payload.dtypes.items()},
            'shape': list(df.shape),
            'truncated': truncated,
        })

    def send_chart(self, name: str, fig):
        self._ensure_comm()
        if self._comm is None:
            return
        import io, base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        self._comm.send({
            'type': 'chart',
            'name': name,
            'data': base64.b64encode(buf.getvalue()).decode(),
        })
```

Wire into the `pivotal()` cell magic after `run_cell`. Walk the AST result list in execution order and for each node:
- If the node produces a DataFrame (any statement that sets `table_name`): call `send_dataframe(name, df)`.
- If the node is a `plot` statement: call `send_chart(name, fig)`.

This strict AST-order traversal ensures the panel reflects execution order within the cell. The last item sent is always visible at cell completion.

**Modified objects:** always push a new history entry even if the name already exists in the cache. This means re-running a cell that modifies `df` appends a fresh snapshot rather than overwriting, so the previous state is still reachable via Back.

---

#### Frontend side — new file `editors/jupyterlab/src/viewer.ts`

**`PivotalViewerWidget`** extends `Widget`:

```
┌─────────────────────────────────────┐
│ ◀  goal_chart · Chart  3/7  ▶  ✕   │  ← header bar
├─────────────────────────────────────┤
│                                     │
│   [DataFrame grid / chart image]    │
│                                     │
│   shape: 1 248 × 7 (truncated)      │  ← footer (DF only)
└─────────────────────────────────────┘
```

Internal state:
```ts
interface ViewerItem {
  type: 'dataframe' | 'chart';
  name: string;
  payload: DataFramePayload | ChartPayload;
}

private _items: ViewerItem[] = [];
private _index = -1;            // currently displayed
private _grid: DataGrid | null; // Lumino DataGrid instance
private _img: HTMLImageElement; // chart image element
```

Key methods:
- `push(item)` — append to cache (cap at 50), advance `_index`, call `render()`
- `back()` / `forward()` — decrement/increment `_index`, call `render()`
- `render()` — swap between grid and img views depending on item type

**DataFrame rendering** using `@lumino/datagrid`:
- Implement a lightweight `BasicDataModel extends DataModel` that wraps the JSON records array.
- Supports column sorting on header click via `SortedModel` (Lumino built-in).
- Numeric columns right-aligned; string columns left-aligned (via `CellRenderer`).
- Frozen first column when more than 6 columns present (nice-to-have).

**Chart rendering:**
- `<img>` element with `src = 'data:image/png;base64,...'`
- Zoom toolbar: `+`, `-`, `1:1` buttons that adjust a CSS `transform: scale()`.
- Pan: `pointerdown` / `pointermove` on a wrapping `<div>` with `overflow: hidden`.

---

#### `index.ts` changes

1. **Comm registration** — on every kernel connection, register the `pivotal_viewer` comm target and wire incoming messages to the panel widget:

```ts
app.serviceManager.sessions.runningChanged.connect(() => {
  // re-register when kernel restarts
});
kernel.registerCommTarget('pivotal_viewer', (comm, _msg) => {
  comm.onMsg = msg => {
    const data = msg.content.data as ViewerMessage;
    viewerWidget.push(data);
  };
});
```

2. **Widget creation** — create `PivotalViewerWidget` once, add to `app.shell` at `'right'`.

3. **Commands**:

| Command ID | Default shortcut | Action |
|---|---|---|
| `pivotal:show-viewer` | `Alt+Shift+P` | Focus / open viewer panel |
| `pivotal:viewer-back` | `Alt+[` | Navigate back |
| `pivotal:viewer-forward` | `Alt+]` | Navigate forward |

4. **`package.json`** — add `@lumino/datagrid` and `@lumino/widgets` as dependencies (both ship with JupyterLab 4; peer-dep approach keeps bundle size neutral).

---

#### New / changed files

| File | Change |
|---|---|
| `pivotal/magic.py` | Add `_PivotalViewer` class; call from `pivotal()` magic |
| `editors/jupyterlab/src/viewer.ts` | New — `PivotalViewerWidget`, `BasicDataModel` |
| `editors/jupyterlab/src/index.ts` | Register viewer plugin, comm target, commands, shortcuts |
| `editors/jupyterlab/package.json` | Add `@lumino/datagrid`, `@lumino/widgets` deps |
| `editors/jupyterlab/style/base.css` | Add viewer panel styles |

---

#### Implementation sequence

1. `magic.py`: add `_PivotalViewer`, wire `send_dataframe` / `send_chart` into magic.
2. `viewer.ts`: skeleton widget, header bar, back/forward wired to stub.
3. `index.ts`: register comm target, connect to widget, register commands.
4. Build and verify comm messages arrive in browser console.
5. `viewer.ts`: implement `BasicDataModel` + `DataGrid` for DataFrame rendering.
6. `viewer.ts`: implement chart PNG display + zoom/pan.
7. Polish: header/footer info, keyboard shortcuts, cache limit, truncation notice.
8. CSS styling.

---

#### Open questions

- **Comm registration timing**: the comm target must be registered before the kernel sends — add a small retry queue on the Python side (store unsent payloads and flush on first successful open).
- **Multiple kernels**: track one comm per kernel; clean up on kernel death / restart.
- **Bidirectional comm for row limit changes**: when the user edits the row limit in the panel footer, the frontend sends a reply comm message (`{type: 'request', name, limit}`) back to the kernel, which re-sends the DataFrame at the new limit. This requires the Python `_PivotalViewer` to register an `on_msg` handler and cache the last-seen DataFrame per name.
- **mpld3**: Make chart interactivity opt-in via a `%%pivotal` option or a Jupyter setting rather than a hard dependency.

---

## Completed

- Enhanced plot syntax: `by <col>` for faceted subplots, `cols <n>` for column count, `style <name>` for matplotlib style files.

