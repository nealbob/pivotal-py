# Expression IR Stage 2 Conformance Report

## Scope

Stage 2 adds conformance coverage for the standalone expression parser without
changing statement parsing, semantic validation, or backend code generation.

The conformance test corpus is derived from current parseable Pivotal snippets
in:

- `tests/test_commands.py`
- `tests/test_commands_polars.py`
- `tests/test_commands_duckdb.py`
- `tests/test_phase5_sql_cte.py`
- `PIVOTAL.md`
- Markdown files under `docs/`
- Markdown files under `docs/syntax/`

The tests collect assignment expressions from those snippets, parse each
expression with `pivotal.expression_parser.parse_expression()`, and verify that
non-fallback ASTs are JSON-serializable. They also verify that assignment nodes
returned by `DSLParser.parse()` attach the same `expression_ast` produced by the
standalone parser.

## Current Result

At the time this report was added, the repository-derived corpus contained 131
distinct assignment expressions from parseable snippets.

- 129 expressions parse to expression ASTs.
- 2 expressions intentionally use the raw-string fallback.

Representative covered expressions include:

- `price * quantity`
- `(revenue - cost) / revenue`
- `amount / sum(amount)`
- `(amount - mean(amount)) / std(amount)`
- `quantile(amount, 0.9)`
- `regex_replace(phone, "[^0-9]", "")`
- `first_name + " " + last_name`
- `:clean_name(name)`

## Raw-String Fallbacks

The current fallback expressions are:

- `:class_names["1"]`
- `revenue >= p90`

These are expected Stage 2 fallbacks.

`:class_names["1"]` requires runtime subscript/index handling in the expression
grammar. Existing code generation already supports the raw string path.

`revenue >= p90` requires comparison operators. Comparisons are intentionally
deferred until expressions and conditions are unified in a later stage.

## Non-Goals

Stage 2 does not:

- migrate semantic `expression_ir` into backend generators
- validate unknown columns or function arity
- parse conditions as expression IR
