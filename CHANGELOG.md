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
- Added package extras for test dependencies.

### Fixed

- Fixed and expanded the test suite around command parsing and backend behavior, including pandas, Polars, DuckDB, and SQL CTE coverage.
- Fixed JupyterLab autocomplete and highlighting for column `for` loops, including column suggestions in `for <name> in ...` headers and distinct loop-placeholder styling.
- Fixed the JupyterLab extension asset copy step so local builds and editable installs work on Windows.
- Added embedded Python syntax highlighting in VS Code for inline `python <code>` statements and multi-line `python`/`end` blocks in Pivotal files.

## [0.3.0]

- Current published package version.
