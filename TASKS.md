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




---

### Date operations

**Goal:** Add date/time extraction and arithmetic functions to Pivotal `assign` expressions. Filter and `group by` are out of scope — the idiomatic workaround is to create the derived column first, then filter/group on it, which is equally readable and has no performance cost in-memory.

```pivotal
df sales
    year = year(date)
    month = month(date)
    quarter = quarter(date)
    weekday = dayofweek(date)
    days_open = date_diff(closed_date, opened_date)
    due = date_add(order_date, 30)
    label = date_format(date, "%b %Y")

df monthly from sales
    filter year == 2024
    group by year, month
        sum amount as total
```

---

**Current state:**

Expressions in assignments are raw strings captured by the grammar. A two-level dispatch exists:
1. `_try_string_func` + `_try_string_func_polars` — matched function calls, e.g. `upper(col)`, dispatch to backend-specific string methods.
2. `_BUILTIN_FUNCS` frozenset controls which function names are intercepted (currently only string functions).
3. Anything not intercepted falls through to `df.eval()` (pandas) / `pl.Expr` (Polars) / SQL verbatim (DuckDB/SQL).

Date functions slot into this existing pattern with no grammar changes needed.

---

**Function syntax and backend mapping:**

| Pivotal syntax | Pandas | Polars | DuckDB/SQL |
|---|---|---|---|
| `year(col)` | `df['col'].dt.year` | `pl.col('col').dt.year()` | `YEAR(col)` |
| `month(col)` | `df['col'].dt.month` | `pl.col('col').dt.month()` | `MONTH(col)` |
| `day(col)` | `df['col'].dt.day` | `pl.col('col').dt.day()` | `DAY(col)` |
| `quarter(col)` | `df['col'].dt.quarter` | `pl.col('col').dt.quarter()` | `QUARTER(col)` |
| `dayofweek(col)` | `df['col'].dt.dayofweek` | `pl.col('col').dt.weekday()` | `DAYOFWEEK(col)` |
| `hour(col)` | `df['col'].dt.hour` | `pl.col('col').dt.hour()` | `HOUR(col)` |
| `minute(col)` | `df['col'].dt.minute` | `pl.col('col').dt.minute()` | `MINUTE(col)` |
| `date_diff(end, start)` | `(df['end'] - df['start']).dt.days` | `(pl.col('end') - pl.col('start')).dt.total_days()` | `DATE_DIFF('day', start, end)` |
| `date_add(col, n)` | `df['col'] + pd.Timedelta(days=n)` | `pl.col('col') + pl.duration(days=n)` | `col + INTERVAL n DAY` |
| `date_format(col, fmt)` | `df['col'].dt.strftime(fmt)` | `pl.col('col').dt.strftime(fmt)` | `STRFTIME(col, fmt)` |
| `to_date(col)` | `pd.to_datetime(df['col'])` | `pl.col('col').str.to_date()` | `CAST(col AS DATE)` |

---

**Implementation plan:**

**Step 1 — Pandas expression helper:**
- Add `_DATE_FUNCS = frozenset({'year', 'month', 'day', 'quarter', 'dayofweek', 'hour', 'minute', 'date_format', 'to_date'})` and two-arg set `_DATE_TWO_ARG = frozenset({'date_diff', 'date_add'})`.
- Add `_try_date_func(self, expr, table)` — matches `func(col)` or `func(col, arg)` where `func in _DATE_FUNCS | _DATE_TWO_ARG`, emits pandas `.dt.*` code.
- In `_parse_string_expr`, try `_try_date_func` before `_try_string_func`.
- Add all date function names to `_BUILTIN_FUNCS` so `_parse_user_func_call` doesn't intercept them.

**Step 2 — Polars expression helper:**
- Add `_try_date_func_polars(self, expr)` mapping the same calls to `pl.col('col').dt.*`.
- Call it in the Polars assign path before the existing string helpers.

**Step 3 — DuckDB/SQL assign:**
- SQL accepts `YEAR(col)`, `MONTH(col)` etc. natively. Add a name-mapping dict for the few that differ (`dayofweek` → `DAYOFWEEK`, `date_format` → `STRFTIME`, `date_diff` → `DATE_DIFF('day', ...)`, `date_add` → `col + INTERVAL n DAY`, `to_date` → `CAST(col AS DATE)`).
- Apply substitution in the assign expression string before emitting SQL.

**Step 4 — Tests:**
- `test_date_extract_pandas`: `year = year(date)` → `df['date'].dt.year`.
- `test_date_diff_pandas`: `days = date_diff(end_date, start_date)`.
- `test_date_add_pandas`: `due = date_add(order_date, 30)`.
- `test_date_format_pandas`: `label = date_format(date, "%b %Y")`.
- Mirror in `test_commands_polars.py` and `test_commands_duckdb.py`.

**Step 5 — Docs + autocomplete:**
- Add "Date functions" section to `docs/syntax/column-operations.md` (or new `docs/syntax/dates.md`). Include the workaround pattern (assign first, then filter/group).
- Update `PIVOTAL.md` with date function table.
- Add all date function names to VS Code extension and JupyterLab autocompletion keyword lists.

---

**Implementation order:** Steps 1 → 2 → 3 → 4 → 5.

**Effort estimate:** ~2 hours. No grammar changes, no filter rewriting — just expression helpers and codegen.

**Biggest risk:** `date_add` with a runtime variable (`:n_days`) needs `pd.Timedelta(days=n_days)` with the variable name substituted — handle the same way `:varname` → `@varname` substitution works in the existing assign path.

---

### Type casting

**Goal:** Add explicit type conversion to Pivotal — both as a column assignment expression and as a standalone `cast` statement. Default behaviour is **coerce** (bad values become NaN/null rather than raising an error), which is the most useful default when fixing up messy ingested data. Add `strict` keyword for explicit hard-error mode.

```pivotal
df sales
    # Standalone cast — coerce mode (default)
    cast amount as float
    cast price, cost as float
    cast event_date as date
    cast code as string

    # Strict mode — error on unparseable values
    cast amount as int strict

    # Inline cast in assignment expressions
    amount = float(amount)
    label = str(code)
```

---

**Current state:**

No type casting exists. Expressions use `df.eval()` which doesn't support type conversion functions. There is no `cast` grammar rule. Adding inline cast via `float(col)` in assign expressions requires a new branch in `_try_string_func` / `_try_string_func_polars`. Adding `cast` as a statement requires a grammar rule.

Both approaches are needed:
- **Inline cast** in expressions (e.g. `amount = float(amount)`, `label = str(code)`) — handled at the expression level, no grammar change. Inline cast always uses coerce mode.
- **Standalone `cast` statement** (e.g. `cast price, cost as float`) — cleaner for casting multiple columns at once, requires a grammar rule. Supports optional `strict` keyword.

---

**Default mode: coerce vs strict — backend differences:**

This is the main complexity. Each backend has a different mechanism:

| Backend | Coerce (default) | Strict |
|---|---|---|
| Pandas (numeric) | `pd.to_numeric(col, errors='coerce')` → NaN | `.astype(float)` → raises `ValueError` |
| Pandas (int) | `pd.to_numeric(col, errors='coerce')` → float with NaN (nullable int not assumed) | `.astype(int)` → raises |
| Pandas (datetime) | `pd.to_datetime(col, errors='coerce')` → NaT | `pd.to_datetime(col)` → raises |
| Pandas (string) | `.astype(str)` — always succeeds (everything has a string repr) | same |
| Pandas (bool) | `.map({'true': True, 'false': False, ...})` or `.astype(bool)` | `.astype(bool)` |
| Polars (numeric) | `.cast(pl.Float64, strict=False)` → null | `.cast(pl.Float64)` → raises |
| Polars (datetime) | `.cast(pl.Datetime, strict=False)` → null | `.cast(pl.Datetime)` → raises |
| DuckDB | `TRY_CAST(col AS DOUBLE)` → NULL | `CAST(col AS DOUBLE)` → raises |
| SQL (non-DuckDB) | `CAST(col AS DOUBLE)` — no safe fallback in standard SQL | same |

Note: for `int` in coerce mode, pandas `pd.to_numeric(errors='coerce')` returns float (since NaN is float). Use nullable integer `pd.Int64Dtype()` if the user needs true integer with nulls — but this adds complexity. Default coerce-to-int returns float; add a note in docs.

---

**Type mapping:**

`date` is dropped as a cast type — it is meaningless in pandas (which has no native date dtype, only `datetime64`) and the LALR conflict risk is high (`date` is a very common column name). Use the `to_date(col)` function instead for date-only columns in Polars/DuckDB.

| Pivotal type | Pandas (coerce) | Polars (coerce) | DuckDB (coerce) | SQL type |
|---|---|---|---|---|
| `int` / `integer` | `pd.to_numeric(..., errors='coerce').astype('Int64')` | `.cast(pl.Int64, strict=False)` | `TRY_CAST(col AS INTEGER)` | `INTEGER` |
| `float` | `pd.to_numeric(..., errors='coerce')` | `.cast(pl.Float64, strict=False)` | `TRY_CAST(col AS DOUBLE)` | `DOUBLE` |
| `string` / `str` | `.astype(str)` | `.cast(pl.Utf8)` | `CAST(col AS VARCHAR)` | `VARCHAR` |
| `bool` / `boolean` | `.astype(bool)` | `.cast(pl.Boolean, strict=False)` | `TRY_CAST(col AS BOOLEAN)` | `BOOLEAN` |
| `datetime` | `pd.to_datetime(..., errors='coerce')` | `.cast(pl.Datetime, strict=False)` | `TRY_CAST(col AS TIMESTAMP)` | `TIMESTAMP` |

For date-only columns, use the `to_date()` expression function (already implemented): `event_date = to_date(event_date_str)` — this maps to `pl.Date` in Polars and `CAST(col AS DATE)` in DuckDB.

---

**Implementation plan:**

**Step 1 — Grammar: `cast` statement:**
- Add a `cast_statement` rule:
  ```
  cast_statement: "cast" IDENTIFIER ("," IDENTIFIER)* "as" TYPE_NAME "strict"? _NL?
  TYPE_NAME: "int" | "integer" | "float" | "string" | "str" | "bool" | "boolean" | "datetime"
  ```
- `cast_statement` goes in `table_op` alongside `filter_statement`, `select_statement` etc.
- Transformer: emit `{'type': 'cast', 'table_name': ..., 'columns': [...], 'cast_type': '...', 'strict': bool}`.

**Step 2 — Inline cast in expressions:**
- Add `_CAST_FUNCS = frozenset({'int', 'integer', 'float', 'str', 'string', 'bool', 'boolean', 'datetime'})`.
- Add a `_try_cast_func(self, expr, table)` method: matches `func(col)` where `func in _CAST_FUNCS`, always emits coerce-mode code.
- In `_parse_string_expr`, try `_try_cast_func` first (before string functions, before `df.eval()`).
- `_BUILTIN_FUNCS` must include the cast function names so `_parse_user_func_call` does not intercept them.

**Step 3 — Pandas codegen (`generate_cast_pandas`):**
```python
def generate_cast_pandas(self, ast_node):
    table = ast_node['table_name']
    cols = ast_node['columns']
    cast_type = ast_node['cast_type']
    strict = ast_node.get('strict', False)
    lines = []
    for col in cols:
        c = f"{table}['{col}']"
        if cast_type in ('int', 'integer'):
            if strict:
                lines.append(f"{c} = {c}.astype(int)")
            else:
                lines.append(f"{c} = pd.to_numeric({c}, errors='coerce').astype('Int64')")
        elif cast_type == 'float':
            if strict:
                lines.append(f"{c} = {c}.astype(float)")
            else:
                lines.append(f"{c} = pd.to_numeric({c}, errors='coerce')")
        elif cast_type in ('str', 'string'):
            lines.append(f"{c} = {c}.astype(str)")
        elif cast_type in ('bool', 'boolean'):
            lines.append(f"{c} = {c}.astype(bool)")
        elif cast_type == 'datetime':
            err = "" if strict else ", errors='coerce'"
            lines.append(f"{c} = pd.to_datetime({c}{err})")
    return '\n'.join(lines)
```

**Step 4 — Pandas inline cast (in `_try_cast_func`, always coerce):**
```python
if func in ('int', 'integer'):
    return f"pd.to_numeric({base}, errors='coerce').astype('Int64')"
if func == 'float':
    return f"pd.to_numeric({base}, errors='coerce')"
if func in ('str', 'string'):
    return f"{base}.astype(str)"
if func in ('bool', 'boolean'):
    return f"{base}.astype(bool)"
if func == 'datetime':
    return f"pd.to_datetime({base}, errors='coerce')"
```

**Step 5 — Polars codegen (`generate_cast_polars`):**
```python
_POLARS_TYPES = {
    'int': 'pl.Int64', 'integer': 'pl.Int64',
    'float': 'pl.Float64',
    'str': 'pl.Utf8', 'string': 'pl.Utf8',
    'bool': 'pl.Boolean', 'boolean': 'pl.Boolean',
    'datetime': 'pl.Datetime',
}
strict_flag = "" if strict else ", strict=False"
for col in cols:
    lines.append(
        f"{table} = {table}.with_columns("
        f"pl.col('{col}').cast({_POLARS_TYPES[cast_type]}{strict_flag}))"
    )
```

**Step 6 — DuckDB/SQL codegen (`generate_cast_duckdb`, `generate_cast_sql`):**
```python
_SQL_TYPES = {
    'int': 'INTEGER', 'integer': 'INTEGER', 'float': 'DOUBLE',
    'str': 'VARCHAR', 'string': 'VARCHAR',
    'bool': 'BOOLEAN', 'boolean': 'BOOLEAN',
    'datetime': 'TIMESTAMP',
}
cast_fn = "CAST" if strict else "TRY_CAST"   # TRY_CAST is DuckDB only
# SQL backend always uses CAST (no TRY_CAST in standard SQL — document caveat)
# DuckDB: SELECT * REPLACE (TRY_CAST(col AS TYPE) AS col) FROM t
for col in cols:
    expr = f"{cast_fn}({col} AS {_SQL_TYPES[cast_type]}) AS {col}"
    lines.append(
        f'{table} = con.execute("SELECT * REPLACE ({expr}) FROM {table}").df()'
    )
```

For the SQL (non-DuckDB) backend, always emit `CAST` regardless of strict flag and add a doc note that coerce mode is not available in standard SQL.

**Step 7 — LALR conflict check:**
- `TYPE_NAME` keywords (`int`, `float`, `str`, etc.) are likely already tokenised as bare `IDENTIFIER`. They must be declared as a dedicated terminal reserved only in the `cast ... as TYPE_NAME` position. The contextual lexer will handle this correctly since `TYPE_NAME` only appears after `"as"` within a `cast_statement`.
- `strict` must similarly be a terminal that is only live after `TYPE_NAME` in `cast_statement`. Do not add it to the global keyword list.
- `cast` itself added to the reserved keyword list.
- `date` is intentionally excluded from `TYPE_NAME` — it is a very common column name and would create LALR conflicts. Use `to_date()` instead.

**Step 8 — Tests:**
- `test_cast_float_coerce_pandas`: `cast amount as float` → `pd.to_numeric(..., errors='coerce')`.
- `test_cast_float_strict_pandas`: `cast amount as float strict` → `.astype(float)`.
- `test_cast_multi_pandas`: `cast price, cost as float`.
- `test_cast_inline_pandas`: `amount = float(amount)`.
- `test_cast_datetime_coerce_pandas`: `cast event_ts as datetime` → `pd.to_datetime(..., errors='coerce')`.
- Mirror in `test_commands_polars.py` (check `strict=False` in output) and `test_commands_duckdb.py` (check `TRY_CAST` vs `CAST`).

**Step 9 — Docs:**
- Add "Type casting" section to `docs/syntax/column-operations.md` (or a new `docs/syntax/types.md`).
- Cover default coerce behaviour prominently — explain that bad values become NaN/null rather than errors.
- Note the `int` coerce caveat: result is nullable integer (`Int64`) not plain `int`, because NaN requires a nullable type.
- Note SQL backend limitation: `TRY_CAST` / coerce not available in standard SQL.
- Update `PIVOTAL.md` with `cast` statement syntax.
- Add `cast`, `strict` to VS Code extension keyword list and JupyterLab autocomplete.

---

**Implementation order:**
1. Grammar + transformer for `cast` statement (Step 1).
2. LALR conflict check (Step 7) — immediately after grammar change.
3. Pandas codegen (Steps 3, 4) — full pandas backend end-to-end, both modes.
4. Polars codegen (Steps 5, 6).
5. DuckDB/SQL codegen (Step 6).
6. Tests + docs (Steps 8, 9).

**Effort estimate:** ~3 hours. The coerce/strict split approximately doubles the codegen surface but each branch is simple.

**Biggest risks:**
- `datetime` as a `TYPE_NAME` may conflict if someone has a column called `datetime` — low risk but worth testing.
- Pandas nullable integer `Int64` vs plain `int`: downstream operations on a nullable-integer column behave slightly differently (e.g. `mean()` returns float, operations with NaN propagate). Document clearly.
- `SELECT * REPLACE (...)` is DuckDB-specific; SQL backend fallback needs explicit column listing or a separate codepath.

---

## Completed

