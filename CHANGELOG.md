# Changelog

All notable changes to `pivotal-lang` will be documented in this file.

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Add user-visible changes under `Unreleased` as they land, then move them into a versioned section when publishing a release.

## [Unreleased]

### Added

- Added `bulk load` for CSV/Parquet file lists, including concat-with-source-column mode and separate-table loading from static aliases or Python alias lists.
- Added column `for` loops for applying assignment statements and column-oriented operations across multiple columns, including Python list variables and generated assignment target names.
- Added aggregation shorthand support so a single aggregation function can be applied across multiple columns.
- Added pandas custom aggregation functions with `agg :function col1 col2 ... as name` syntax for whole-table and grouped aggregations, plus bracket syntax for keyword arguments.
- Added Polars support for custom aggregation functions via a pandas fallback.
- Added regex string operations with `regex_extract`, `regex_replace`, and `matches` / `not matches` filters across pandas, Polars, DuckDB, and SQL CTE backends.
- Added `select matches "pattern"` and `drop matches "pattern"` for regex-based column selection and removal.
- Added `assert` and `check` data-quality commands with `unique`, `not null`, and condition-based rules.
- Added `pivotal.run_pivotal()` as a structured execution API for AI tools and automation, returning JSON-friendly parse, codegen, runtime, warning, and table-preview results.
- Added `pivotal.run_pivotal_isolated()` to execute structured Pivotal verifier calls in a clean subprocess with timeout handling.
- Added `python -m pivotal --verify --json` for CLI-based structured verification, including CSV/Parquet input loading and table preview output.
- Added pandas-to-Pivotal comparison APIs and `python -m pivotal --compare --json` for AI conversion verification.
- Added an optional local MCP server (`python -m pivotal.mcp_server`) with syntax, run, compile, and pandas-to-Pivotal comparison tools.
- Added package extras for test dependencies.
- Added `round` column syntax and `min_periods=<n>` for rolling window commands.
- Added read-only Streamable HTTP MCP mode for hosted syntax/examples/compile tools.
- Added shared syntax-token metadata and a read-only `pivotal_highlight` MCP tool.
- Added a copy-to-clipboard button to the default `pivotal_highlight` HTML output.
- Added compile-time `list` definitions, non-recursive `function` pipeline expansion with positional and keyword arguments, and `pivotal.load_functions()` for Python-callable wrappers.
- Added MCP documentation tools for indexing, reading, and searching Pivotal docs, including command-reference and topic docs.
- Added and reorganized documentation, including a complete syntax reference, a pandas-to-Pivotal cheatsheet, clearer backend and Pivotal-vs-Python navigation, and a discoverable MCP Server page.
- Added `show shape` and `show columns` output variants.
- Added `quantile` and `percentile` as built-in aggregation functions for assignment expressions and `agg` statements.
- Added persistent native `scalar` and `dict` definitions, JSON/YAML config dict loading, dot-path config lookup, Pivotal list indexing, and indexed `:` runtime references for lists/dicts.
- Added clearer plotting and table documentation, including explicit `plot`, `pivot plot`, and `table` syntax coverage in the user docs and syntax reference.

### Fixed

- Changed the default Jupyter `%pivotal_set` canvas from `none` to `a4` for viewer output in JupyterLab and VS Code notebooks.
- Fixed and expanded the test suite around command parsing and backend behavior, including pandas, Polars, DuckDB, and SQL CTE coverage.
- Fixed MCP compile validation so incomplete inline Python such as `python def ...:` is rejected instead of reporting successful code generation.
- Fixed MCP syntax topic lookup so aliases such as `melt` and `reshape` return specific pivot/unpivot guidance instead of the start of `PIVOTAL.md`.
- Standardized Python runtime function calls to require `:` in column expressions and `apply` statements, matching custom aggregation syntax.
- Fixed JupyterLab autocomplete and highlighting for column `for` loops, including column suggestions in `for <name> in ...` headers and distinct loop-placeholder styling.
- Fixed the JupyterLab extension asset copy step so local builds and editable installs work on Windows.
- Added embedded Python syntax highlighting in VS Code for inline `python <code>` statements and multi-line `python`/`end` blocks in Pivotal files.
- Fixed per-column `fillna` so unquoted fill values can reference another column, while quoted strings remain literal fill values.
- Fixed canvas viewer chart rendering in VS Code and JupyterLab so charts that exceed the printable canvas are scaled down proportionally with a small fit notice.

## [0.3.0]

- Current published package version.
