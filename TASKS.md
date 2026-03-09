# Project: Pivotal

  ## Current focus


  ## Backlog

  ## Ideas 
  - [ ]  Bulk data load. Load multiuple csv files and combine (merge / concat). Load multiple sql tables (combine or don't)

  - [ ] Some bugs in the viewer. Zoom in charts resets should remember. Viewer is not cleared when notebook stops (what happens with multiple notebooks?) Table interactivity - into code outputs?

  - [ ] left pane list of objects

  - [ ] Pivot charts / tables - graphical low code option to make charts and tables 

  - [ ] String functions in `assign` expressions — see implementation plan below.

  - [ ] Polars support — see implementation plan below.

  - [ ] Generate publication ready tables using the Great Tables package. Need to develop an implementation plan for this.

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

