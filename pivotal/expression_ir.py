"""Semantic normalization for parsed Pivotal assignment expressions.

The normalizer is additive metadata. Backend generators may consume supported
IR subsets, while unsupported nodes keep using the public raw ``expression``
fallback during the staged migration.
"""

import copy


class ExpressionIRValidationError(ValueError):
    """Raised when a known expression construct has invalid semantics."""


def _function(category, canonical_name=None, arity=None):
    return {
        "category": category,
        "canonical_name": canonical_name,
        "arity": arity,
    }


FUNCTIONS = {
    # Aggregates. ``min`` and ``max`` are handled specially because one
    # argument is aggregate and two or more arguments are row-wise scalar.
    "avg": _function("aggregate", "mean", 1),
    "mean": _function("aggregate", "mean", 1),
    "sum": _function("aggregate", "sum", 1),
    "count": _function("aggregate", "count", 1),
    "std": _function("aggregate", "std", 1),
    "median": _function("aggregate", "median", 1),
    "var": _function("aggregate", "var", 1),
    "nunique": _function("aggregate", "nunique", 1),
    "first": _function("aggregate", "first", 1),
    "last": _function("aggregate", "last", 1),
    "quantile": _function("aggregate", "quantile", 2),
    "percentile": _function("aggregate", "percentile", 2),
    "wavg": _function("aggregate", "weighted_mean", 2),
    "wmean": _function("aggregate", "weighted_mean", 2),
    # Casts.
    "bool": _function("cast", "boolean", 1),
    "boolean": _function("cast", "boolean", 1),
    "datetime": _function("cast", "datetime", 1),
    "float": _function("cast", "float", 1),
    "int": _function("cast", "integer", 1),
    "integer": _function("cast", "integer", 1),
    "str": _function("cast", "string", 1),
    "string": _function("cast", "string", 1),
    # Known scalar functions.
    "date_add": _function("scalar", "date_add", 2),
    "date_diff": _function("scalar", "date_diff", 2),
    "date_format": _function("scalar", "date_format", 2),
    "day": _function("scalar", "day", 1),
    "dayofweek": _function("scalar", "dayofweek", 1),
    "hour": _function("scalar", "hour", 1),
    "left": _function("scalar", "left", 2),
    "len": _function("scalar", "len", 1),
    "lower": _function("scalar", "lower", 1),
    "ltrim": _function("scalar", "ltrim", 1),
    "minute": _function("scalar", "minute", 1),
    "month": _function("scalar", "month", 1),
    "quarter": _function("scalar", "quarter", 1),
    "regex_extract": _function("scalar", "regex_extract", (2, 3)),
    "regex_replace": _function("scalar", "regex_replace", 3),
    "replace": _function("scalar", "replace", 3),
    "right": _function("scalar", "right", 2),
    "rtrim": _function("scalar", "rtrim", 1),
    "substr": _function("scalar", "substr", 3),
    "to_date": _function("scalar", "to_date", 1),
    "trim": _function("scalar", "trim", 1),
    "upper": _function("scalar", "upper", 1),
    "year": _function("scalar", "year", 1),
}


def _check_arity(name, arity, arguments):
    if arity is None:
        return

    actual = len(arguments)
    if isinstance(arity, tuple):
        low, high = arity
        if low <= actual <= high:
            return
        expected = f"between {low} and {high}"
    else:
        if actual == arity:
            return
        expected = str(arity)

    raise ExpressionIRValidationError(
        f"Function '{name}' expected {expected} argument(s), got {actual}."
    )


def _copy_base(node):
    return copy.deepcopy(node)


def normalize_expression_ast(expression_ast):
    """Return semantic expression IR for a parsed expression AST.

    ``None`` means the expression had no supported syntax AST and should keep
    using the raw-string fallback.
    """
    if expression_ast is None:
        return None
    return _normalize(expression_ast)


def normalize_expression_ast_safe(expression_ast):
    """Best-effort semantic normalization for additive parser metadata."""
    try:
        return normalize_expression_ast(expression_ast)
    except (ExpressionIRValidationError, TypeError, KeyError):
        return None


def _normalize(node):
    kind = node.get("kind")

    if kind in {"column", "literal", "runtime_reference"}:
        return _copy_base(node)

    if kind == "unary":
        return {
            "kind": "unary",
            "operator": node["operator"],
            "operand": _normalize(node["operand"]),
        }

    if kind == "binary":
        return {
            "kind": "binary",
            "operator": node["operator"],
            "left": _normalize(node["left"]),
            "right": _normalize(node["right"]),
        }

    if kind == "runtime_call":
        return {
            "kind": "runtime_call",
            "name": node["name"],
            "arguments": [_normalize(arg) for arg in node.get("arguments", [])],
        }

    if kind == "call":
        return _normalize_call(node)

    raise ExpressionIRValidationError(f"Unsupported expression AST node kind: {kind!r}.")


def _normalize_call(node):
    name = node["name"]
    lowered = name.lower()
    arguments = [_normalize(arg) for arg in node.get("arguments", [])]

    if lowered in {"min", "max"}:
        if not arguments:
            raise ExpressionIRValidationError(
                f"Function '{name}' expected at least 1 argument, got 0."
            )
        if len(arguments) == 1:
            return {
                "kind": "aggregate",
                "function": "minimum" if lowered == "min" else "maximum",
                "arguments": arguments,
            }
        return {
            "kind": "scalar_function",
            "function": "least" if lowered == "min" else "greatest",
            "arguments": arguments,
        }

    spec = FUNCTIONS.get(lowered)
    if spec is None:
        return {
            "kind": "backend_function",
            "name": name,
            "arguments": arguments,
        }

    _check_arity(name, spec["arity"], arguments)
    category = spec["category"]
    canonical_name = spec["canonical_name"] or lowered

    if category == "aggregate":
        return {
            "kind": "aggregate",
            "function": canonical_name,
            "arguments": arguments,
        }

    if category == "cast":
        return {
            "kind": "cast",
            "target_type": canonical_name,
            "expression": arguments[0],
        }

    if category == "scalar":
        return {
            "kind": "scalar_function",
            "function": canonical_name,
            "arguments": arguments,
        }

    raise ExpressionIRValidationError(
        f"Function '{name}' has unknown category {category!r}."
    )
