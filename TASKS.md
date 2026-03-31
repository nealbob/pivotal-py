# Project: Pivotal

## Current focus

## Backlog

- [ ] **Bug: string literals and Python variable references in `assign` expressions** — `newcol = "mystring"` and `newcol = :var + "string"` pass through to `pandas.eval()` incorrectly. String literals should be handled as plain Python assignment (not eval); Python variable references (`:var`) should inject the variable value into the expression before eval. Currently these silently produce wrong results or errors.

## Ideas


  - [] bug in load GUi parsing of file paths (fix wiht some pyhton processing)

  - [] graphicwalker in the viewer pane AG grid and gwalker code generation...

    - switch polars to pandas?>?

    - error handling

  - left menu upgrades

    - scatter diagrams

  - describe / sample Visual summary via tabulator? plus showing df changes in left and right pane

  - [] Bugs in plot (y and x) technicaly need to swap for barh...ALSO sharex share y override in plot, rotate x or y text overide, y axis label position?

  - [ ] Export to ms word / excel option to keep formating excel one chart or table per sheet, or dataframe include / exclude data and chart images tables include formating... 

  - [ ] Interactive plots (plotly?)

  - [ ]  Bulk data load. Load multiuple csv files and combine (merge / concat). Load multiple sql tables (combine or don't)

  - [ ] VSCODE extension improvements

  - schema support (read and write) in frictionless / sql

        │          Feature           │                                    Rationale                                     │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤    
  │ 1   │ Error handling             │ Affects every user on every run. A tool marketed as friendlier than Pandas that  │     
  │     │                            │ surfaces Pandas tracebacks is contradicting itself. Foundational trust.          │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤     
  │     │ Database / cloud           │ DuckDB already does the heavy lifting. Exposing it in load syntax turns Pivotal  │
  │ 2   │ connectors (DuckDB)        │ from a local-file tool into something usable on real production data. High ROI,  │   
  │     │                            │ relatively contained.                                                            │
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ 3   │ VS Code interactive output │ VS Code is where most data practitioners work. Without a viewer panel, Pivotal   │   
  │     │                            │ is effectively JupyterLab-only. Biggest audience expansion available.            │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │ 4   │ Data quality assertions    │ Turns scripts into self-validating pipelines. Natural fit for the grammar. Moves │   
  │     │                            │  Pivotal toward production use cases, not just exploratory analysis.             │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │ 5   │ AI / natural language →    │ Pivotal's small grammar makes it a much better LLM target than raw Pandas.       │   
  │     │ Pivotal                    │ Strong differentiation story. Lowers barrier to entry for non-technical users.   │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │ 6   │ Column profiling in        │ Inline dtype, null %, histogram per column. Low effort, high daily utility —     │   
  │     │ explorer                   │ replaces a df.describe() call users currently have to make manually.             │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │     │ More output formats        │ Plotly fills the interactive charts gap. Excel is practical for business users   │   
  │ 7   │ (Plotly first, then Excel) │ handing off results. Both are self-contained additions. Word/docx lowest         │   
  │     │                            │ priority of the three.                                                           │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │     │ Explorer pane enhancements │ Search bar is genuinely useful at scale. Click-to-cast/rename is nice but        │   
  │ 8   │  (search bar, click to     │ secondary — prioritise search first.                                             │   
  │     │ cast/rename)               │                                                                                  │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │     │                            │ Most exciting concept but highest risk. The Vega-Lite → Pivotal translation is   │   
  │ 9   │ Graphic Walker integration │ hard to get right. Better tackled once the core is more mature and after a spike │   
  │     │                            │  to assess feasibility.                                                          │   
  ├─────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │     │                            │ High long-term value (editor-agnostic, consolidates autocomplete/error logic)    │   
  │ 10  │ LSP                        │ but pure infrastructure with no visible user-facing payoff until the editors     │   
  │     │                            │ consuming it are built. Revisit after VS Code work is further along.   


## Implementation plans

---

### Error Handling

**Goal:** Catch errors before they reach Pandas/Polars backends. Show friendly, actionable messages instead of Python tracebacks or raw Lark errors.

**Architecture overview:**
```
User input
    │
    ▼
Phase 2: Lark parse errors    ──►  friendly syntax message with line/column
    │
    ▼
Phase 3: Semantic validator   ──►  table/column checks before code runs
    │
    ▼
Phase 4: Runtime error filter ──►  catch backend tracebacks, translate to Pivotal terms
    │
    ▼
Phase 5: Cleanup              ──►  remove embedded guard clauses from generated code
```

Each phase is independently deliverable and testable.

---

#### Phase 1 — Error infrastructure (`pivotal/errors.py`, new file)

**What:** Create the shared error type and display function used by all subsequent phases.

**Files changed:** `pivotal/errors.py` (new), `magic.py` (wire up display)

**Deliverables:**
- `PivotalError` dataclass with fields: `message`, `line`, `column`, `source_line`, `suggestion`, `error_type`
- `format_error(err, source_code)` producing output like:
  ```
  Pivotal Error (line 4): Unknown column 'reveue' in table 'sales'
    → Did you mean 'revenue'?

    4 | agg sum reveue as total
                ^^^^^^
  ```
- Plain text output by default; HTML with red highlight when running in JupyterLab
- Fuzzy "did you mean?" matching via `difflib.get_close_matches` (stdlib, no new dependency)
- Update `magic.py` to call `format_error()` instead of `print(f"Pivotal Parse Error: {results['error']}")` for existing parse errors

**Tests:** None specific to this phase — it's display infrastructure. Verified visually.

---

#### Phase 2 — Friendly Lark syntax errors

**What:** Replace raw Lark exception strings with readable messages pointing to the problem line.

**Files changed:** `dsl_parser.py` (`DSLParser.parse()`)

**Deliverables:**
- Catch specific Lark exception types instead of bare `except Exception`:

| Lark exception | User message |
|---------------|--------------|
| `UnexpectedCharacters` | "Unrecognised character 'x' at line N, column M" |
| `UnexpectedToken` (normal) | "Unexpected token '{token}' — expected one of: filter, select, ..." |
| `UnexpectedToken` where token is `$END` | "Unexpected end of input — is a statement incomplete?" |
| `UnexpectedToken` where token is `INDENT`/`DEDENT` | "Indentation error — check sub-statements are correctly indented" |
| `UnexpectedEOF` | "Unexpected end of input — is a statement incomplete?" |
| `VisitError` | Unwrap to inner exception, then re-classify |

- All cases return a `PivotalError` (not a raw string) with `.line` and `.column` populated
- Phase 1's `format_error()` handles display automatically

**Tests:** `tests/test_errors.py` (new file)
- Malformed syntax inputs → assert correct `PivotalError.message` and `.line`
- Bad indentation, unknown token, truncated statement
- Full existing test suite passes (no regressions)

---

#### Phase 3 — Semantic validator: table and column checking

**What:** Walk the parsed AST before code generation and validate table names and column names against the current session namespace. The most user-visible improvement.

**Files changed:** `dsl_parser.py` (new `validate()` method), `magic.py` (call validate between parse and generate)

**Deliverables:**

*3a. Table existence* — for `from`, `merge`, `concat`, `apply`: check table name exists in `user_ns` or was defined by an earlier statement in the same cell (the within-cell forward pass — see edge case note below).
```
Error: Table 'slaes' not found.
  → Available tables: sales, customers, orders
  → Did you mean 'sales'?
```

*3b. Column name validation* — for `filter`, `select`, `drop`, `sort`, `group by`, `agg`, `assign`, `rename`, `cast`, `merge on`: check all column references exist by tracking the current column set through the pipeline:

| Statement | Effect on column set |
|-----------|---------------------|
| `load` / `df from` | Seed from `user_ns` DataFrame columns |
| `filter`, `sort`, `fillna`, `dropna`, `distinct` | Unchanged |
| `select col1, col2` | Becomes `{col1, col2}` |
| `drop col1` | Remove col1 |
| `rename old as new` | Replace old → new |
| `group by x` + `agg sum y as z` | Becomes `{x, z}` |
| `assign new_col = ...` | Add new_col |
| `cast col as type` | Unchanged |
| `pivot` / `unpivot` | Reset to unknown (skip subsequent column checks) |
| `merge` | Union of both tables' columns |

```
Error (line 4): Unknown column 'reveue' in table 'sales'
  → Did you mean 'revenue'?

  4 | agg sum reveue as total
              ^^^^^^
```

*3c. Merge key validation* — check `on` / `left_on` / `right_on` columns exist in both left and right tables.

**Edge case — within-cell forward pass:** The validator runs before any code executes, so a table created by `load sales "data.csv"` in line 1 is not yet in `user_ns` when validating `df summary from sales` in line 3 of the same cell. The validator makes a first pass to collect all table names defined by `load` and `df` statements in the cell, then uses this set to suppress false "table not found" errors. Column checking for within-cell-defined tables is skipped (columns not yet known).

**Validation is skipped silently** when the source table cannot be resolved — no false positives.

**Blocking vs non-blocking:** Unknown column/table → block execution. Warnings (e.g. type mismatches) → print but allow execution.

**Tests:** `tests/test_errors.py`
- Wrong column name → correct error message and line number
- Wrong table name → correct error + "did you mean?" suggestion
- Same-cell load+use → no false positive
- Valid code with all checks → passes cleanly
- Full existing test suite passes (no regressions)

---

#### Phase 4 — Runtime error filter

**What:** Safety net for errors that slip past Phase 3 (e.g. columns that only exist after runtime transforms). Intercept `shell.run_cell()` errors and translate known Python exception patterns to Pivotal messages, suppressing the traceback.

**Files changed:** `magic.py`

**Deliverables:**
- After `shell.run_cell()`, inspect `result.error_in_exec` before IPython displays it:

| Python exception pattern | Pivotal message |
|--------------------------|----------------|
| `KeyError: 'col_name'` from Pandas/Polars | "Column 'col_name' not found" |
| `NameError: name 'table' is not defined` | "Table 'table' not found — was it loaded in a previous cell?" |
| `FileNotFoundError` | "File not found: [path]" |
| `TypeError: not supported between ... 'str' and 'int'` | "Type mismatch — check column types in your expression" |

- Matched errors → suppress traceback, show `PivotalError` via `format_error()`
- Unmatched errors → let IPython display the traceback normally (genuine errors from `python...end` blocks should still surface)

**Tests:** `tests/test_errors.py`
- Known exception patterns → translated message, no traceback
- Unknown exception patterns → traceback passes through unchanged

---

#### Phase 5 — Remove embedded guard clauses from generated code

**What:** Now that Phases 3 and 4 handle validation, remove the inline `if 'table' not in locals()... raise NameError(...)` and `if not isinstance(...)... raise TypeError(...)` guards that are currently baked into generated Pandas/Polars code. Generated code becomes clean and readable.

**Files changed:** `dsl_parser.py` — `generate_copy_table_pandas`, `generate_validate_table_pandas`, and their Polars equivalents

**Deliverables:**
- Remove all embedded `raise NameError` / `raise TypeError` guard clauses from code generators
- Generated code for `df summary from sales` becomes simply `summary = sales.copy()` with no guards
- Keyword collision validation moves from the Lark transformer into the Phase 3 semantic validator

**Tests:** Full existing test suite passes. Manually verify that a missing-table error still produces a clean Pivotal message (not a traceback) via Phase 3/4.