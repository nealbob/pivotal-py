"""Tests for Stage 3 semantic expression normalization."""

import json

import pytest

from pivotal import DSLParser
from pivotal.expression_ir import (
    FUNCTIONS,
    ExpressionIRValidationError,
    normalize_expression_ast,
)
from pivotal.expression_parser import parse_expression


def column(name):
    return {"kind": "column", "name": name}


def literal(literal_type, value):
    return {"kind": "literal", "literal_type": literal_type, "value": value}


def normalize(source):
    return normalize_expression_ast(parse_expression(source))


def test_function_registry_covers_stage_3_categories():
    assert FUNCTIONS["avg"]["canonical_name"] == "mean"
    assert FUNCTIONS["float"]["category"] == "cast"
    assert FUNCTIONS["upper"]["category"] == "scalar"
    assert FUNCTIONS["wavg"]["canonical_name"] == "weighted_mean"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "price * quantity",
            {
                "kind": "binary",
                "operator": "multiply",
                "left": column("price"),
                "right": column("quantity"),
            },
        ),
        (
            "upper(trim(name))",
            {
                "kind": "scalar_function",
                "function": "upper",
                "arguments": [
                    {
                        "kind": "scalar_function",
                        "function": "trim",
                        "arguments": [column("name")],
                    }
                ],
            },
        ),
        (
            "float(amount)",
            {
                "kind": "cast",
                "target_type": "float",
                "expression": column("amount"),
            },
        ),
        (
            "avg(amount)",
            {
                "kind": "aggregate",
                "function": "mean",
                "arguments": [column("amount")],
            },
        ),
        (
            "wmean(amount, weight)",
            {
                "kind": "aggregate",
                "function": "weighted_mean",
                "arguments": [column("amount"), column("weight")],
            },
        ),
        (
            "max(amount)",
            {
                "kind": "aggregate",
                "function": "maximum",
                "arguments": [column("amount")],
            },
        ),
        (
            "max(amount, 0)",
            {
                "kind": "scalar_function",
                "function": "greatest",
                "arguments": [column("amount"), literal("integer", 0)],
            },
        ),
        (
            "log(price)",
            {
                "kind": "backend_function",
                "name": "log",
                "arguments": [column("price")],
            },
        ),
        (
            ":clean_name(name)",
            {
                "kind": "runtime_call",
                "name": "clean_name",
                "arguments": [column("name")],
            },
        ),
    ],
)
def test_normalize_expression_ast_golden(source, expected):
    expression_ir = normalize(source)
    assert expression_ir == expected
    json.dumps(expression_ir)


def test_normalize_nested_scalar_and_aggregate_functions():
    assert normalize("min(max(amount - 150, 0), max(amount) / 2)") == {
        "kind": "scalar_function",
        "function": "least",
        "arguments": [
            {
                "kind": "scalar_function",
                "function": "greatest",
                "arguments": [
                    {
                        "kind": "binary",
                        "operator": "subtract",
                        "left": column("amount"),
                        "right": literal("integer", 150),
                    },
                    literal("integer", 0),
                ],
            },
            {
                "kind": "binary",
                "operator": "divide",
                "left": {
                    "kind": "aggregate",
                    "function": "maximum",
                    "arguments": [column("amount")],
                },
                "right": literal("integer", 2),
            },
        ],
    }


def test_known_function_arity_is_validated_directly():
    with pytest.raises(ExpressionIRValidationError, match="expected 1"):
        normalize("upper(first, last)")


def test_assignment_expression_ir_drives_basic_pandas_arithmetic_codegen():
    parser = DSLParser()
    nodes = parser.parse("with sales\nrevenue = price * quantity\n")
    assignment = nodes[1]

    assert assignment["expression"] == "price * quantity"
    assert assignment["expression_ir"] == normalize("price * quantity")
    code = "\n".join(parser.generate_code(nodes))
    assert "sales['revenue'] = (sales['price'] * sales['quantity'])" in code
    assert "sales.eval('price * quantity')" not in code


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("pandas", "sales['revenue'] = (sales['price'] * sales['quantity'])"),
        ("polars", "((pl.col('price') * pl.col('quantity'))).alias('revenue')"),
        ("duckdb", "(price * quantity) AS revenue"),
        ("sql", "(price * quantity) AS revenue"),
    ],
)
def test_basic_arithmetic_assignment_uses_expression_ir_across_backends(backend, expected):
    parser = DSLParser()
    nodes = parser.parse("with sales\nrevenue = price * quantity\n")

    code = "\n".join(parser.generate_code(nodes, backend=backend))

    assert expected in code


def test_assignment_expression_ir_attached_after_expansion():
    parser = DSLParser()
    nodes = parser.parse(
        "scalar gst = 0.1\n"
        "with sales\n"
        "for col in price, cost\n"
        "    col = col * gst\n"
    )

    assert nodes[1]["expression"] == "price * 0.1"
    assert nodes[1]["expression_ir"] == normalize("price * 0.1")
    assert nodes[2]["expression"] == "cost * 0.1"
    assert nodes[2]["expression_ir"] == normalize("cost * 0.1")


def test_expression_ir_falls_back_when_expression_ast_is_unavailable():
    parser = DSLParser()
    nodes = parser.parse('with sales\nvalue = :config["value"]\n')

    assert nodes[1]["expression_ast"] is None
    assert nodes[1]["expression_ir"] is None
    code = "\n".join(parser.generate_code(nodes))
    assert ".eval(" in code
    assert "config" in code
