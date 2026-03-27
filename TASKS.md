# Project: Pivotal

## Current focus

## Backlog

## Ideas

  - [] grammer extensions, date operations, type casting, intersect / exclude, range merge

  - [] make gui pop ups not inline.
  
  - [] graphicwalker in the viewer pane AG grid and gwalker code generation...

    - scatter diagrams

  - describe / sample Visual summary via tabulator? plus showing df changes in left and right pane

  - [] Bugs in plot (y and x) technicaly need to swap for barh...ALSO sharex share y override in plot, rotate x or y text overide, y axis label position?

  - [ ] Export to ms word / excel option to keep formating excel one chart or table per sheet, or dataframe include / exclude data and chart images tables include formating... 

  - [ ] Interactive plots (pandas hvplot?)

  - [ ] Polars support — see implementation plan below.

  - [ ]  Bulk data load. Load multiuple csv files and combine (merge / concat). Load multiple sql tables (combine or don't)

  - [ ] VSCODE extension: Fix bug where pivotal code is embedded inside a *.py file. 

  - cast / type conversion

  - schema support (read and write) in frictionless / sql


## Implementation plans


### Polars backend

**Goal:** Full-parity Polars backend — every statement type that works in the pandas backend also works in Polars, including window functions, plotting (via matplotlib, consistent with the viewer), and Great Tables output. No fallback to pandas mid-pipeline.

---

**Current state:**
- Dispatch mechanism `generate_{type}_{backend}` is in `dsl_parser.py` — architecture is ready.
- `DSLParser(backend='polars')` already switches the backend.
- Existing polars generators: `load_table`, `apply`, `gt_table`, `save`, `load_all`, `load_package_table`.
- All other statement types raise `NotImplementedError`.

---

**Core challenge — expression conversion:**

The pandas generators pass expression strings to `df.eval()` / `df.query()`. Polars requires `pl.Expr` objects. A shared helper `_expr_to_polars(expr_str)` is needed for `filter`, `assign`, and window statements.

Approach:
- Tokenise using the same lightweight regex already used by `_try_string_func` / `_split_on_plus`.
- Replace bare identifiers (non-quoted, non-numeric, non-keyword) with `pl.col('name')`.
- Map string functions: `left(col, n)` → `pl.col(col).str.slice(0, n)`, `right(col, n)` → `pl.col(col).str.slice(-n)`, `upper/lower/trim` → `.str.to_uppercase()` etc.
- String concat with `+`: `pl.col('a') + '-' + pl.col('b')` → `pl.concat_str([pl.col('a'), pl.lit('-'), pl.col('b')])`.
- Boolean operators: `and`/`or` → `&`/`|`; `not` → `~`.
- Comparisons: `==`, `!=`, `<`, `>`, `<=`, `>=` map directly.
- `between [lo, hi]` → `pl.col('x').is_between(lo, hi)`.
- `in [list]` → `pl.col('x').is_in([...])`.
- `contains`/`startswith`/`endswith` → `.str.contains()`/`.str.starts_with()`/`.str.ends_with()`.
- Python variable injection (`:varname`) → runtime `pl.lit(varname)` via f-string in generated code.

---

**Statement-by-statement plan:**

| Statement | Pandas output | Polars equivalent | Notes |
|---|---|---|---|
| `load` | `pd.read_csv(...)` | `pl.read_csv(...)` | ✅ done |
| `copy_table` | `X = Y.copy()` | `X = Y.clone()` | trivial |
| `validate_table` | type check | `isinstance(X, pl.DataFrame)` | trivial |
| `filter` | `df.query('expr')` | `df.filter(_expr_to_polars(expr))` | needs helper |
| `select` | `df[['a','b']]` | `df.select(['a','b'])` or with rename: `df.select([pl.col('a').alias('b')])` | |
| `rename` | `df.rename(columns={...})` | `df.rename({...})` | trivial |
| `drop` | `df.drop(columns=[...])` | `df.drop([...])` | trivial |
| `distinct` | `df.drop_duplicates()` | `df.unique(subset=[...])` | |
| `sort` | `df.sort_values(...)` | `df.sort([cols], descending=[bools])` | |
| `assign` (simple) | `df.eval('col = expr')` | `df.with_columns(_expr_to_polars(expr).alias('col'))` | needs helper |
| `assign` (conditional) | multi-line with `np.where` | `df.with_columns(pl.when(cond).then(expr).otherwise(pl.col(col)).alias(col))` | |
| `assign` (by-agg) | `groupby().transform()` | `df.with_columns(pl.col(col).agg_func().over(by))` | window expr |
| `groupby + agg` | `df.groupby().agg(...)` | `df.group_by([...]).agg([pl.col(c).func().alias(a)])` | |
| `merge/join` | `pd.merge(...)` | `df.join(other, on=..., how=...)` or `df.join(other, left_on=..., right_on=...)` | |
| `concat` | `pd.concat([df, other])` | `pl.concat([df, other])` | |
| `pivot` | `df.pivot_table(...)` | `df.pivot(on=cols, index=rows, values=val, aggregate_function=func)` | |
| `unpivot` | `df.melt(...)` | `df.unpivot(on=[...], index=[...], variable_name=..., value_name=...)` | Polars 0.20+ |
| `fillna` | `df.fillna(value)` | `df.fill_null(value)` | |
| `dropna` | `df.dropna(subset=[...])` | `df.drop_nulls(subset=[...])` | |
| `rank` | `df[col].rank(...)` | `df.with_columns(pl.col(col).rank(method=...).over(partition).alias(result))` | |
| `shift` (lag/lead) | `df[col].shift(n)` | `df.with_columns(pl.col(col).shift(n).over(partition).alias(result))` after sort | |
| `cumulative` | `df[col].cumsum()` etc. | `df.with_columns(pl.col(col).cum_sum().over(partition).alias(result))` after sort | `cummean` → `cum_sum()/cum_count()` |
| `rolling` | `df[col].rolling(n).mean()` | `df.with_columns(pl.col(col).rolling_mean(n).over(partition).alias(result))` after sort | |
| `plot` | matplotlib via pandas `.plot()` | convert to pandas for matplotlib: `df.to_pandas().plot(...)` — same matplotlib output | keeps viewer compatibility |
| `agg_plot` | groupby + matplotlib | `df.group_by().agg()` in Polars, then `.to_pandas()` for plot | |
| `gt_table` | Great Tables (pandas) | convert: `df.to_pandas()` → GT | ✅ partially done |
| `show` | `display(df.head())` | `display(df.head())` | trivial |
| `apply` | `df[col].apply(func)` | `df.with_columns(pl.col(col).map_elements(func))` | ✅ done |
| `save` | frictionless package | convert to pandas for frictionless | ✅ done |
| `python` | raw block verbatim | same — user uses `pl` API directly | |

---

**Plotting strategy:**

Use matplotlib throughout (same as pandas backend) by calling `.to_pandas()` before the plot step. This keeps the Pivotal viewer working without changes and avoids adding `hvplot` as a dependency. The Polars computation pipeline stays lazy/Polars until the plot boundary.

Native Polars/hvPlot interactive plots can be added later as a separate `plot_interactive` statement or `interactive=True` kwarg.

---

**Preamble:**

The Polars preamble (emitted once per cell, like the DuckDB preamble) is:
```python
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
```

---

**Files to create/modify:**
- `pivotal/dsl_parser.py`: Add `generate_*_polars` methods for every statement above; add `_expr_to_polars()` helper and `_expr_to_polars_str_concat()` for string concat.
- `tests/test_commands_polars.py`: Mirror `test_commands.py` using `DSLParser(backend='polars')` with `pytest.importorskip('polars')`.
- `tests/test_window_polars.py`: Window function tests (rank, lag/lead, cumulative, rolling) mirroring `test_phase3_duckdb.py`.
- `docs/backends.md`: Add Polars section.
- `pyproject.toml`: Add `polars` to optional extras.

---

**Implementation order (phases):**

1. **Phase 1 — Core pipeline:** `copy_table`, `validate_table`, `filter`, `select`, `rename`, `drop`, `sort`, `distinct`, `concat`, `fillna`, `dropna` — these are all straightforward once `_expr_to_polars()` exists.
2. **Phase 2 — Assign + merge:** `assign` (simple, conditional, by-agg), `merge/join` with all key variants.
3. **Phase 3 — Aggregation and reshape:** `groupby`, `pivot`, `unpivot`.
4. **Phase 4 — Window functions:** `rank`, `shift`, `cumulative`, `rolling`.
5. **Phase 5 — Output:** `plot`, `agg_plot`, `gt_table`, `show`, `save`.

---

**Effort estimate:**
- ~6–8 hours for Claude Code across two sessions: implement all phases, write helper, add tests.
- The biggest lift vs the original plan is the six new statement types (window functions + `agg_plot` + `unpivot`) and the `_expr_to_polars()` helper covering string functions and by-agg expressions.

**Biggest risks:**
- `_expr_to_polars()` for complex nested expressions — start simple, add test coverage per case.
- `rolling` with `over()` partition: Polars `rolling_mean(...).over(...)` has known limitations in some versions — may need a `group_by().map_groups()` fallback.
- `pivot` aggregate function naming differs between Polars versions.

---


## Completed

