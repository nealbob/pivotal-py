## Pivotal 0.6.0

Pivotal 0.6.0 adds a native backend-independent expression system, explicit
package and table persistence syntax, and several editor and plotting fixes.

### Added

- Added explicit table and package save syntax: `save <table> as "<path>"`,
  `save <table> as table "<catalog>.<schema>.<table>"`, and
  `save package as "<path>"`, with backend-aware file and managed-table writes.
- Added explicit `\` line continuation so long Pivotal statements can span
  multiple physical lines.
- Added standalone expression parsing, semantic `expression_ir` and
  `condition_ir`, and IR-driven code generation for supported assignments and
  conditions across pandas, Polars, DuckDB, and SQL. Unsupported legacy
  expressions and conditions retain their compatibility fallback.

### Fixed

- Fixed up/down arrow-key navigation in Pivotal autocomplete menus inside
  JupyterLab notebooks.
- Fixed faceted `plot ... by ...` statements with a title so they generate and
  register populated charts with the overall title and axis labels applied
  correctly.
- Constrained Polars to versions below 1.38 pending resolution of an upstream
  Windows CPU-feature detection failure during import.
