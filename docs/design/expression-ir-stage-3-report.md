# Expression IR Stage 3 Semantic Normalization Report

## Scope

Stage 3 adds best-effort semantic normalization for assignment expressions. It
does not migrate backend generators.

The public assignment node is now additive:

```json
{
  "type": "assign",
  "expression": "avg(amount)",
  "expression_ast": {
    "kind": "call",
    "name": "avg",
    "arguments": [{"kind": "column", "name": "amount"}]
  },
  "expression_ir": {
    "kind": "aggregate",
    "function": "mean",
    "arguments": [{"kind": "column", "name": "amount"}]
  }
}
```

## Implemented Semantics

The normalizer currently handles:

- literals, columns, unary operations, binary operations, and runtime references
- runtime calls such as `:clean_name(name)`
- known scalar functions such as `upper`, `trim`, date functions, and regex
  string functions
- casts such as `float(amount)`, `integer(code)`, `str(code)`, and
  `datetime(ts)`
- aggregate aliases such as `avg(amount)` -> `mean`
- weighted mean aliases `wavg` and `wmean` -> `weighted_mean`
- arity-dependent `min` / `max`
- unknown bare calls as backend-function candidates, for example `log(price)`

Known function arity is validated by the normalizer. Public parser attachment
remains best-effort: invalid or unsupported semantic forms get
`expression_ir: null` rather than breaking existing raw-string execution.

## Non-Goals

Stage 3 does not:

- change pandas, Polars, DuckDB, or SQL generation
- validate column existence
- decide backend capability support
- parse comparison conditions into expression IR
- define a complete program IR

