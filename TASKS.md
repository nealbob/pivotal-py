# Project: Pivotal

  ## Current focus


  ## Backlog

  ## Ideas 

  - [ ] Generate publication ready tables using the Great Tables package. See implementation plan below.
  
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

### Great Tables (`table` command)

**Goal:** Add a `table` command to the Pivotal DSL that wraps the `great_tables` package to generate publication-ready HTML tables. Tables are previewed in the Pivotal Viewer (with optional page canvas), appear in the Explorer pane with a distinct icon, and are exported as self-contained `.html` files via the `save` command.

**Dependency:** Add `great-tables>=0.10` to `pyproject.toml`. PNG/MD export is out of scope — HTML only.

---

**Grammar:**

```
table <name>
    [title "string"]
    [subtitle "string"]
    [font size <number>]
    [font "font-family-name"]
    [stub <column>]
    [col <column> [as "label"] [number <decimals>|integer|currency <code>|percent <decimals>|date]]
    [stripe]
    [canvas a4|a3|letter]
```

- `stub <column>` — pulls the column into a styled left-margin row-label area (visually separated by a heavier border). Use for the natural row identifier (name, category, date).
- `stripe` — enables alternating row background colours (zebra striping) via `opt_row_striping()`.
- `col` — one line per column; all column options (label rename + format) grouped together. Either part can be omitted independently.
- `canvas` — per-table page size for the viewer preview. If omitted the table renders as a free-scrolling iframe. If the table is wider than the canvas it overflows naturally (no auto-scaling).
- Output format is always HTML. The `save` command exports `<name>.html` (fully self-contained, inline CSS).

**Examples:**

```
df results

table summary
    title "Season Results"
    subtitle "All matches, 2023-24"
    font size 11
    font "Georgia"
    stub team
    col goals as "Goals Scored" number 1
    col win_rate as "Win %" percent 1
    col revenue as "Revenue" currency GBP
    col matches integer
    stripe
    canvas a4

table quick
    title "Top Teams"
```

---

**Lark grammar additions (`dsl_parser.py`):**

```lark
statement: ... | table_statement

table_statement: "table" IDENTIFIER (_NL | _NL _INDENT table_params _DEDENT)?

table_params: table_param+

table_param: "title"    STRING                              _NL?  -> table_title
           | "subtitle" STRING                              _NL?  -> table_subtitle
           | "font" "size" NUMBER                           _NL?  -> table_font_size
           | "font" STRING                                  _NL?  -> table_font_family
           | "stub" IDENTIFIER                              _NL?  -> table_stub
           | "stripe"                                       _NL?  -> table_stripe
           | "canvas" IDENTIFIER                            _NL?  -> table_canvas
           | "col" IDENTIFIER "as" STRING table_fmt?        _NL?  -> table_col_labeled
           | "col" IDENTIFIER table_fmt                     _NL?  -> table_col_fmt_only
           | "col" IDENTIFIER                               _NL?  -> table_col_bare

table_fmt: "number"   NUMBER?     -> fmt_number
         | "integer"               -> fmt_integer
         | "currency" IDENTIFIER?  -> fmt_currency
         | "percent"  NUMBER?      -> fmt_percent
         | "date"                  -> fmt_date
```

Add `table`, `stub`, `stripe`, `col` to the keywords list in the Lark grammar and in `language.ts`.

---

**AST node:**

```python
{
    'type': 'gt_table',
    'name': 'summary',           # table variable name
    'table_name': 'results',     # active DataFrame
    'title': 'Season Results',
    'subtitle': 'All matches, 2023-24',
    'font_size': 11,
    'font_family': 'Georgia',
    'stub': 'team',
    'stripe': True,
    'canvas': 'a4',
    'cols': [
        {'col': 'goals',    'label': 'Goals Scored', 'fmt': 'number',   'decimals': 1},
        {'col': 'win_rate', 'label': 'Win %',        'fmt': 'percent',  'decimals': 1},
        {'col': 'revenue',  'label': 'Revenue',      'fmt': 'currency', 'code': 'GBP'},
        {'col': 'matches',  'label': None,            'fmt': 'integer'},
    ],
}
```

---

**Code generation (`generate_gt_table_pandas`):**

Generates Python that builds the GT object and stores the rendered HTML in `_pivotal_gt_tables`:

```python
import great_tables as _gt_mod
_gt = _gt_mod.GT(results, rowname_col='team')
_gt = _gt.tab_header(title='Season Results', subtitle='All matches, 2023-24')
_gt = _gt.opt_table_font(font='Georgia', size=11)
_gt = _gt.opt_row_striping()
_gt = _gt.cols_label(goals='Goals Scored', win_rate='Win %', revenue='Revenue')
_gt = _gt.fmt_number(columns='goals', decimals=1)
_gt = _gt.fmt_percent(columns='win_rate', decimals=1)
_gt = _gt.fmt_currency(columns='revenue', currency='GBP')
_gt = _gt.fmt_integer(columns='matches')
if '_pivotal_gt_tables' not in globals(): globals()['_pivotal_gt_tables'] = {}
globals()['_pivotal_gt_tables']['summary'] = {
    'html': _gt.as_raw_html(make_page=True, inline_css=True),
    'canvas': 'a4',
}
```

---

**`magic.py` changes:**

1. `_PivotalViewer.send_table(name, html, canvas)` — new method. Builds canvas metadata (page_width_mm, page_height_mm, margin_mm=20, label) from `_PAPER_SIZES_MM` if canvas is set; sends `{'type': 'gt_table', 'name': name, 'html': html, 'canvas': meta}` via comm.

2. `_send_results_to_viewer()` — extend to walk `gt_table` nodes and call `viewer.send_table()` by looking up entries in `ns['_pivotal_gt_tables']`.

3. `save` command — for each entry in `_pivotal_gt_tables`, write `<name>.html` alongside the other saved files.

---

**`viewer.ts` changes:**

New payload type:
```typescript
export interface GtTablePayload {
  type: 'gt_table';
  name: string;
  html: string;
  canvas?: CanvasMeta;   // no chart_width/height fields — table fills page width
}
export type ViewerMessage = DataFramePayload | ChartPayload | GtTablePayload;
```

New render methods:
- `_renderGtTable(p)` — dispatches to free or page-layout variant.
- `_renderGtTableFree(p)` — renders `<iframe srcdoc=...>` filling the body. Iframe isolates GT's CSS from JupyterLab.
- `_renderGtTableOnPage(p)` — same page-layout scaffold as `_renderChartOnPage` (ResizeObserver + RAF) but with an iframe sized to usable page width instead of an `<img>`.

Update `_render()` dispatch, title display, and `ExplorerItem` type union.

---

**`explorer.ts` changes:**

Add `GT_TABLE_ICON` — a formal ruled-lines icon distinct from the DataFrame grid icon:
```svg
<svg viewBox="0 0 14 14" width="14" height="14">
  <rect x="1" y="1.5" width="12" height="2.5" rx="0.4" fill="currentColor" opacity="0.85"/>
  <line x1="1" y1="6"   x2="13" y2="6"   stroke="currentColor" stroke-width="0.8" opacity="0.55"/>
  <line x1="1" y1="8.5" x2="13" y2="8.5" stroke="currentColor" stroke-width="0.8" opacity="0.45"/>
  <line x1="1" y1="11"  x2="13" y2="11"  stroke="currentColor" stroke-width="0.8" opacity="0.35"/>
  <line x1="1" y1="13"  x2="13" y2="13"  stroke="currentColor" stroke-width="1.2" opacity="0.6"/>
</svg>
```

Update `_renderItem()` to use this icon for `type === 'gt_table'` (no expand toggle — GT tables have no column tree).

---

**`index.ts` changes:**

- Add `'table'` to `COMMAND_KEYWORDS` for autocomplete.
- Add sub-keyword completions for the `table` context: after `col <identifier>`, offer `as`, `number`, `integer`, `currency`, `percent`, `date`; after `canvas`, offer `a4`, `a3`, `letter`.

---

**Implementation order:**

| Step | File | Notes |
|------|------|-------|
| 1. Grammar + transformer | `dsl_parser.py` | Add Lark rules, transformer methods, AST node |
| 2. Code generator | `dsl_parser.py` | `generate_gt_table_pandas` |
| 3. magic.py send + save | `magic.py` | `send_table()`, extend `_send_results_to_viewer`, extend `save` |
| 4. Viewer payload + render | `viewer.ts` | `GtTablePayload`, `_renderGtTable*` methods |
| 5. Explorer icon | `explorer.ts` | `GT_TABLE_ICON`, handle `gt_table` type |
| 6. Syntax + autocomplete | `language.ts`, `index.ts` | Add keywords |
| 7. Tests | `tests/test_gt_table.py` | Basic generation tests (no GT import required — mock or skip) |
| 8. README | `README.md` | Document `table` command grammar |

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

