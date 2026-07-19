"""Additive condition AST/IR metadata for Pivotal condition-bearing nodes.

Backend generators still consume the public ``conditions`` and ``operators``
structures. This module only builds best-effort metadata for the staged
expression IR migration.
"""

import copy


COMPARATOR_OPERATORS = {
    "==": "equal",
    "!=": "not_equal",
    ">": "greater_than",
    "<": "less_than",
    ">=": "greater_than_or_equal",
    "<=": "less_than_or_equal",
    "between": "between",
    "contains": "contains",
    "not contains": "not_contains",
    "matches": "matches",
    "not matches": "not_matches",
    "startswith": "starts_with",
    "endswith": "ends_with",
    "in": "in",
    "not in": "not_in",
}


class ConditionIRValidationError(ValueError):
    """Raised when condition metadata cannot be normalized."""


def build_condition_ast(conditions, operators):
    """Return a JSON-serializable condition AST, or ``None`` when unsupported."""
    if not conditions:
        return None

    try:
        predicates = [_condition_predicate(condition) for condition in conditions]
        if any(predicate is None for predicate in predicates):
            return None
        if len(predicates) != len(operators or []) + 1:
            return None

        ast = predicates[0]
        for index, operator in enumerate(operators or []):
            logical_operator = str(operator).lower()
            if logical_operator not in {"and", "or"}:
                return None
            ast = {
                "kind": "logical",
                "operator": logical_operator,
                "left": ast,
                "right": predicates[index + 1],
            }
        return ast
    except (KeyError, TypeError, ValueError):
        return None


def normalize_condition_ast(condition_ast):
    """Return semantic condition IR for a parsed condition AST."""
    if condition_ast is None:
        return None
    return _normalize(condition_ast)


def normalize_condition_ast_safe(condition_ast):
    """Best-effort semantic normalization for additive condition metadata."""
    try:
        return normalize_condition_ast(condition_ast)
    except (ConditionIRValidationError, TypeError, KeyError):
        return None


def _condition_predicate(condition):
    comparator = str(condition["comparator"])
    operator = COMPARATOR_OPERATORS.get(comparator)
    if operator is None:
        return None

    left = {"kind": "column", "name": str(condition["column"])}
    right = _condition_value_ast(
        condition.get("value"),
        quoted=comparator in _text_comparators(),
    )
    if right is None:
        return None

    return {
        "kind": "predicate",
        "operator": operator,
        "left": left,
        "right": right,
    }


def _text_comparators():
    return {
        "contains",
        "not contains",
        "matches",
        "not matches",
        "startswith",
        "endswith",
    }


def _condition_value_ast(value, *, quoted=False):
    if isinstance(value, dict):
        value_type = value.get("type")
        if value_type == "var":
            return {"kind": "runtime_reference", "name": value["name"]}
        if value_type == "list_ref":
            return {"kind": "list_reference", "name": value["name"]}
        if value_type == "compile_ref":
            return {"kind": "compile_reference", "path": value["path"]}
        return None

    if isinstance(value, list):
        values = [_condition_value_ast(item, quoted=True) for item in value]
        if any(item is None for item in values):
            return None
        return {"kind": "list", "values": values}

    if value is None:
        return {"kind": "literal", "literal_type": "null", "value": None}
    if isinstance(value, bool):
        return {"kind": "literal", "literal_type": "boolean", "value": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"kind": "literal", "literal_type": "integer", "value": value}
    if isinstance(value, float):
        return {"kind": "literal", "literal_type": "float", "value": value}
    if _is_quoted_string(value) or quoted:
        return {"kind": "literal", "literal_type": "string", "value": str(value)}
    if isinstance(value, str):
        return {"kind": "column", "name": value}
    return None


def _is_quoted_string(value):
    return type(value).__name__ == "_LiteralStr"


def _normalize(node):
    kind = node.get("kind")

    if kind in {
        "column",
        "literal",
        "runtime_reference",
        "list_reference",
        "compile_reference",
    }:
        return copy.deepcopy(node)

    if kind == "list":
        return {
            "kind": "list",
            "values": [_normalize(value) for value in node["values"]],
        }

    if kind == "predicate":
        return {
            "kind": "predicate",
            "operator": node["operator"],
            "left": _normalize(node["left"]),
            "right": _normalize(node["right"]),
        }

    if kind == "logical":
        return {
            "kind": "logical",
            "operator": node["operator"],
            "left": _normalize(node["left"]),
            "right": _normalize(node["right"]),
        }

    raise ConditionIRValidationError(f"Unsupported condition AST node kind: {kind!r}.")
