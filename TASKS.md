# Project: Pivotal

  ## Current focus


  ## Backlog

  ## Ideas 

 - [ ] scale issue with viewer in table mode due to resizing of window

 - [ ] delete buttons and comands

 - [] pivotal.update() command - will check if any dfs have changed or been added and add them to the list. check if any charts or tables have been modified and update them in the viewer?

  - [ ] Add highlighting of the current table in the left pane. Also need sperate indication of which item in left pane is currently being shown in viewer. Perhpas a rectangle / background color highlighting the current dataframe in the left pane and a small eye icon for the currently visible item in the viewer. This will handle cases where the viewer is showing a chart or table  but we still want users to know which data frame is the current one (so it is obvious which dataframe new pivotal commands will apply to). Perhaps we could also have a status bar at the bottom of the left pane, which says "Current df: <name>"
  
  - [ ] Need to check syntax highlighting and auto correct...

  - [ ] Export to ms word / excel option to keep formating excel one chart or table per sheet, or dataframe include / exclude data and chart images tables include formating... 

  - [ ] Button on viewer to copy table or chart to clipboard

  - [ ] Style sheet support - GUI based menu to create plot or table styles. Menu items to apply them to specific plots/tables or set as default. 

  - [ ] Interactive plots (pandas hvplot?)

  - [ ] Polars support — see implementation plan below.

  - [ ] Pivot charts / tables - graphical low code option to make charts and tables 

  - [ ]  Bulk data load. Load multiuple csv files and combine (merge / concat). Load multiple sql tables (combine or don't)

  - [] Bugs in plot (y and x) technicaly need to swap for barh...??
  
  - [ ] VSCODE extension: Fix bug where pivotal code is embedded inside a *.py file. 

  - [ ] Jupyter lab, is there a way to make cells pivotal by default, with some kind of toggle...

  - cast / type conversion — type coercion is fiddly and infrequent. Python is the right place for it.

  - describe / sample — pure exploration helpers. One-liners in Python (df.describe(), df.sample(10)), not worth adding to the grammar.

  - melt / unpivot — complex, infrequent, and the syntax would be awkward. Python is clearly the right escape hatch.

  - Window / rolling functions 

  - head / tail — in a notebook context this is about quick exploration. limit 10 at the end of a pipeline to preview results is very natural and saves a Python cell.



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



## Completed

