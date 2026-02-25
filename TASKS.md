# Project: Pivotal

  ## Current focus
 
  - [ ] Finalise vscode extension

  ## Backlog

  ## Ideas (not ready to be implemented)
  
  - [ ] String functions in `assign` expressions — see implementation plan below.

  - [ ] Fix bug where pivotal code is embedded inside a *.py file. This currently works fine (it runs inside the interactive notebook, and has syntax highlighting in the editor as expected) but in the editor the pivotal code section has pylance errors (red underlines) as it is still expecting python code. Is there a way to fix this...

  - save - I want to have a save option that involves a is more excel like in that the whole session can be saved as a package (like a workbook). So this would mean by default a save command would save all of the tables in the session (each to csvs) in a folder. My preference is to implement this using the Frictionless data standard, so that means saving the session as a frictionless data package, with csvs in folders and matching metadata in json. Meta data needs to be updated each time a command is executed. In future it would be useful to have a version of the metadata in memory so it could be used for autocomplete or AI prompting (should this use a different format in memoery like TOON?). Does it matter that metadata might be duplicative if there are multiple copies of the same columns in different dataframes... Should there be some form of autosave or only save on command...?

  - Audit keywords used in all statements. Where there are multiple keywords pick one and stick with it. Choices should be guided by common keywords in R/dplyr, pandas, SQL, and DAX / power query / excel picking the one that is most obvious to users of these tools (where it is less obvious we can lean towards pandas keywords, e.g., merge rather than join).

  - cast / type conversion — type coercion is fiddly and infrequent. Python is the right place for it.

  - load multiple files or a folder and merge or concat, apply type conversion on load (use json metadata or something to guide this). Perhaps simple load then add assigntings sub command or, modify metadata then reload using assigntings in metadata??

  - describe / sample — pure exploration helpers. One-liners in Python (df.describe(), df.sample(10)), not worth adding to the grammar.

  - melt / unpivot — complex, infrequent, and the syntax would be awkward. Python is clearly the right escape hatch.

  - Window / rolling functions — same. The pandas API for these is already fairly readable and they're an advanced use case.

  - head / tail — in a notebook context this is about quick exploration. limit 10 at the end of a pipeline to preview results is very natural and saves a Python cell.

  - connections to other data formats / sources...

## Implementation plans

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
