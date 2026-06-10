"""Focused tests for the additive standalone expression parser."""

import json

import pytest

from pivotal import DSLParser
from pivotal.expression_parser import parse_expression


def column(name):
    return {"kind": "column", "name": name}


def literal(literal_type, value):
    return {"kind": "literal", "literal_type": literal_type, "value": value}


def binary(operator, left, right):
    return {
        "kind": "binary",
        "operator": operator,
        "left": left,
        "right": right,
    }


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "price * quantity",
            binary("multiply", column("price"), column("quantity")),
        ),
        (
            "(revenue - cost) / revenue",
            binary(
                "divide",
                binary("subtract", column("revenue"), column("cost")),
                column("revenue"),
            ),
        ),
        (
            "max(price, 100)",
            {
                "kind": "call",
                "name": "max",
                "arguments": [column("price"), literal("integer", 100)],
            },
        ),
        (
            "upper(trim(name))",
            {
                "kind": "call",
                "name": "upper",
                "arguments": [
                    {
                        "kind": "call",
                        "name": "trim",
                        "arguments": [column("name")],
                    }
                ],
            },
        ),
        (
            "amount * :multiplier",
            binary(
                "multiply",
                column("amount"),
                {"kind": "runtime_reference", "name": "multiplier"},
            ),
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
def test_parse_expression_golden(source, expected):
    assert parse_expression(source) == expected


def test_expression_operator_precedence_and_associativity():
    assert parse_expression("-a ** 2 + b * 3") == binary(
        "add",
        {
            "kind": "unary",
            "operator": "negative",
            "operand": binary("power", column("a"), literal("integer", 2)),
        },
        binary("multiply", column("b"), literal("integer", 3)),
    )
    assert parse_expression("a ** b ** c") == binary(
        "power",
        column("a"),
        binary("power", column("b"), column("c")),
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1", literal("integer", 1)),
        ("0.5", literal("float", 0.5)),
        ("1e3", literal("float", 1000.0)),
        ('"hello"', literal("string", "hello")),
        ("'hello'", literal("string", "hello")),
        ("TRUE", literal("boolean", True)),
        ("false", literal("boolean", False)),
        ("null", literal("null", None)),
        ("None", literal("null", None)),
    ],
)
def test_parse_expression_literals(source, expected):
    assert parse_expression(source) == expected
    json.dumps(expected)


def test_unsupported_expression_returns_none():
    assert parse_expression("config['multiplier'] * amount") is None


def test_assignment_expression_ast_is_additive_and_codegen_is_unchanged():
    parser = DSLParser()
    nodes = parser.parse("with sales\nrevenue = price * quantity\n")
    assignment = nodes[1]

    assert assignment["expression"] == "price * quantity"
    assert assignment["expression_ast"] == binary(
        "multiply", column("price"), column("quantity")
    )
    assert "sales.eval('price * quantity')" in "\n".join(parser.generate_code(nodes))


def test_assignment_expression_ast_attached_after_expansion():
    parser = DSLParser()
    nodes = parser.parse(
        "scalar gst = 0.1\n"
        "with sales\n"
        "for col in price, cost\n"
        "    col = col * gst\n"
    )

    assert [(node["target"], node["expression"]) for node in nodes[1:]] == [
        ("price", "price * 0.1"),
        ("cost", "cost * 0.1"),
    ]
    assert nodes[1]["expression_ast"] == binary(
        "multiply", column("price"), literal("float", 0.1)
    )
    assert nodes[2]["expression_ast"] == binary(
        "multiply", column("cost"), literal("float", 0.1)
    )


def test_unsupported_assignment_keeps_raw_expression_fallback():
    parser = DSLParser()
    nodes = parser.parse("with sales\nvalue = :config['value']\n")

    assert nodes[1]["expression"] == ":config['value']"
    assert nodes[1]["expression_ast"] is None
