# Expression IR Design

## Status And Scope

This document defines the boundary between Pivotal assignment syntax and
backend-independent expression handling.

The migration is deliberately additive:

- The statement parser continues to capture assignment expressions as strings.
- `DSLParser.parse()` continues to return the existing public syntax AST.
- Assignment nodes retain their public `expression` string unchanged.
- Supported assignment expressions also receive additive `expression_ast` and
  `expression_ir` fields.
- Backend generators consume `expression_ir` for the migrated assignment
  subsets: basic arithmetic, scalar functions, casts, and aggregates.
  Unsupported nodes continue using the raw `expression` fallback.

This document defines expression and condition nodes only. It does not define
a complete program IR or change expression semantics.

## Compiler Boundary

The intended compiler pipeline is:

```text
Pivotal source
  -> statement parser
  -> public syntax AST
  -> compile-time expansion and column-loop lowering
  -> standalone expression parser
  -> additive expression AST
  -> additive semantic normalization
  -> IR-first backend generation with legacy fallback
```

Expression parsing happens after compile-time function/value expansion and
column-loop lowering. This ensures the expression AST describes the assignment
that backend generators actually receive.

## Initial Expression Language

Stage 1 recognizes:

- Column references: `price`
- Integer and float literals: `1`, `0.5`, `1e3`
- String literals using single or double quotes
- Boolean literals: `true`, `false` (case-insensitive)
- Null literals: `null`, `none` (case-insensitive)
- Parentheses
- Unary `+` and `-`
- Binary `+`, `-`, `*`, `/`, `%`, and `**`
- Generic function calls: `upper(name)`, `log(price)`
- Runtime references: `:multiplier`
- Runtime calls: `:clean_name(name)`

Operator precedence, from highest to lowest, is:

1. Parenthesized expressions, literals, references, and calls
2. Power `**` (right-associative)
3. Unary `+` and `-`
4. Multiply, divide, and modulo
5. Add and subtract

Parentheses affect tree structure but are not retained as separate nodes.

The initial expression parser does not recognize conditions, attribute access,
subscripts, lists, dictionaries, keyword arguments, or arbitrary Python
expressions.

## Node Vocabulary

All expression nodes are JSON-serializable dictionaries.

Column reference:

```json
{"kind": "column", "name": "price"}
```

Literal:

```json
{"kind": "literal", "literal_type": "float", "value": 0.1}
```

`literal_type` is one of `integer`, `float`, `string`, `boolean`, or `null`.
Null literals have a JSON `null` value.

Unary operation:

```json
{
  "kind": "unary",
  "operator": "negative",
  "operand": {"kind": "column", "name": "amount"}
}
```

Binary operation:

```json
{
  "kind": "binary",
  "operator": "multiply",
  "left": {"kind": "column", "name": "price"},
  "right": {"kind": "column", "name": "quantity"}
}
```

Binary operator names are `add`, `subtract`, `multiply`, `divide`, `modulo`,
and `power`. Unary operator names are `positive` and `negative`.

Generic function call:

```json
{
  "kind": "call",
  "name": "upper",
  "arguments": [{"kind": "column", "name": "name"}]
}
```

Runtime reference and runtime call:

```json
{"kind": "runtime_reference", "name": "multiplier"}
```

```json
{
  "kind": "runtime_call",
  "name": "clean_name",
  "arguments": [{"kind": "column", "name": "name"}]
}
```

## Function Semantics

Stage 1 records generic bare calls without deciding their meaning. Stage 3
semantic normalization distinguishes:

- Pivotal built-in scalar functions such as `upper(name)`
- Pivotal aggregate functions such as `mean(amount)`
- Arity-dependent functions such as aggregate `max(amount)` versus scalar
  `max(amount, 0)`
- Backend-native functions such as `log(price)`

Runtime calls have a separate node kind because `:clean_name(name)` explicitly
requests a host Python callable. Runtime references are also distinct from
columns.

Compile-time references do not have an expression node kind in Stage 1.
Compile-time values are expanded before expression parsing, so the expression
parser sees the resulting literal, column, or expression text.

## Fallback And Compatibility Policy

Expression parsing is best-effort during this migration.

For a supported expression, an assignment node is additive:

```json
{
  "type": "assign",
  "expression": "price * quantity",
  "expression_ast": {
    "kind": "binary",
    "operator": "multiply",
    "left": {"kind": "column", "name": "price"},
    "right": {"kind": "column", "name": "quantity"}
  },
  "expression_ir": {
    "kind": "binary",
    "operator": "multiply",
    "left": {"kind": "column", "name": "price"},
    "right": {"kind": "column", "name": "quantity"}
  }
}
```

For an unsupported legacy expression, `expression_ast` and `expression_ir` are
`null` and the raw `expression` string remains authoritative. Unsupported
expression syntax must not turn a previously valid Pivotal program into a parse
failure.

Backend generators may consume explicitly migrated `expression_ir` subsets.
Unsupported nodes must fall back to the raw `expression` string so current
pandas, Polars, DuckDB, and SQL behavior is preserved outside migrated slices.

## Condition IR Metadata

Condition-bearing syntax nodes receive best-effort condition metadata. The
existing public `conditions`, `operators`, and `query_str` fields are retained
for compatibility and as the fallback for unsupported condition shapes.

Condition-bearing nodes receive additive fields:

```json
{
  "condition_ast": {
    "kind": "predicate",
    "operator": "greater_than",
    "left": {"kind": "column", "name": "amount"},
    "right": {"kind": "literal", "literal_type": "integer", "value": 100}
  },
  "condition_ir": {
    "kind": "predicate",
    "operator": "greater_than",
    "left": {"kind": "column", "name": "amount"},
    "right": {"kind": "literal", "literal_type": "integer", "value": 100}
  }
}
```

Multiple conditions are represented as left-associative logical nodes:

```json
{
  "kind": "logical",
  "operator": "and",
  "left": {"kind": "predicate"},
  "right": {"kind": "predicate"}
}
```

Predicate operands reuse the expression node vocabulary where possible:
columns, literals, and runtime references. Condition-specific collection and
reference operands use `list`, `list_reference`, and `compile_reference` nodes.

Stage 7 attaches condition metadata after compile-time expansion and
column-loop lowering. Unsupported condition metadata must produce `null`
condition fields rather than turning a previously valid Pivotal program into a
parse failure.

Stage 8 lets backend condition emitters consume supported `condition_ir` nodes
for filters, condition-based data-quality rules, conditional assignments, and
case branches. If `condition_ir` is absent or contains an unsupported node, the
emitters fall back to the legacy `conditions` and `operators` fields.

Stage 8 does not change the public syntax AST, accepted syntax, or backend
surface behavior. It only moves the supported condition codegen subset onto the
structured predicate/logical metadata.

## Semantic IR Node Vocabulary

Stage 3 adds semantic nodes for known expression meaning.

Scalar function:

```json
{
  "kind": "scalar_function",
  "function": "upper",
  "arguments": [{"kind": "column", "name": "name"}]
}
```

Aggregate:

```json
{
  "kind": "aggregate",
  "function": "mean",
  "arguments": [{"kind": "column", "name": "amount"}]
}
```

Cast:

```json
{
  "kind": "cast",
  "target_type": "float",
  "expression": {"kind": "column", "name": "amount"}
}
```

Backend function candidate:

```json
{
  "kind": "backend_function",
  "name": "log",
  "arguments": [{"kind": "column", "name": "price"}]
}
```

The Stage 3 function registry normalizes aliases and arity-dependent functions:

- `avg(amount)` -> aggregate `mean`
- `wavg(amount, weight)` and `wmean(amount, weight)` -> aggregate
  `weighted_mean`
- `float(amount)`, `integer(code)`, `str(code)`, and `datetime(ts)` -> `cast`
- `max(amount)` and `min(amount)` -> aggregate `maximum` and `minimum`
- `max(amount, 0)` and `min(amount, 0)` -> scalar `greatest` and `least`
- unknown bare calls such as `log(price)` -> `backend_function`

Known function arity is validated by the normalizer. Public parser attachment
remains best-effort: if semantic normalization fails, `expression_ir` is `null`
and raw-string execution remains authoritative.

## Source Locations

Future expression nodes may receive an optional `source` object:

```json
{
  "source": {
    "start_offset": 12,
    "end_offset": 28,
    "line": 3,
    "column": 9,
    "end_line": 3,
    "end_column": 25
  }
}
```

Offsets are zero-based and end-exclusive. Lines and columns are one-based.
Locations should refer to the original Pivotal source when available.

Stage 1 does not attach locations because expression parsing currently occurs
after compile-time expansion and loop lowering, where generated expressions do
not always have a direct original-source span.

## Deferred Decisions

The following are intentionally deferred:

- Complete program IR and schema versioning
- Backend capability validation
- Backend generation from unsupported expression nodes
- Backend generation from unsupported condition nodes
- Substrait lowering
