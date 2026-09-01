## Pivotal 0.7.0

Pivotal 0.7.0 makes multi-column aggregation more consistent, adds a clearer
target-first syntax for weighted means, and fixes native Pivotal values used as
load paths.

### Added

- Added target-first weighted aggregation blocks with optional `=` syntax:

  ```pivotal
  agg wmean price, cost
      weight = quantity
  ```

- Added named Pivotal target lists for built-in and weighted aggregations, such
  as `agg mean measures` and `agg wmean measures` with an indented shared weight.
- Kept bracket syntax, legacy weight-first syntax, and the `wavg` alias backward
  compatible. A dataframe column named `weight` remains valid.

### Fixed

- Fixed `load` rejecting native Pivotal scalar paths unless they were prefixed
  with `:`, and added compile-time dictionary path reference support.
- Fixed multi-case assignments failing when the assignment header ended with
  trailing whitespace.
