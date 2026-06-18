"""Focused tests for the additive standalone expression parser."""

import ast
import json
import re
from pathlib import Path

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


def test_assignment_expression_ast_is_additive_and_codegen_uses_expression_ir():
    parser = DSLParser()
    nodes = parser.parse("with sales\nrevenue = price * quantity\n")
    assignment = nodes[1]

    assert assignment["expression"] == "price * quantity"
    assert assignment["expression_ast"] == binary(
        "multiply", column("price"), column("quantity")
    )
    code = "\n".join(parser.generate_code(nodes))
    assert "sales['revenue'] = (sales['price'] * sales['quantity'])" in code
    assert "sales.eval('price * quantity')" not in code


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


CONFORMANCE_PY_FILES = [
    Path("tests/test_commands.py"),
    Path("tests/test_commands_polars.py"),
    Path("tests/test_commands_duckdb.py"),
    Path("tests/test_phase5_sql_cte.py"),
]

CONFORMANCE_MARKDOWN_FILES = [
    Path("PIVOTAL.md"),
    *Path("docs").glob("*.md"),
    *Path("docs/syntax").glob("*.md"),
]

RAW_FALLBACK_EXPRESSIONS = {
    ':class_names["1"]',
    "revenue >= p90",
}

REPRESENTATIVE_CORPUS_EXPRESSIONS = {
    "price * quantity",
    "(revenue - cost) / revenue",
    "amount / sum(amount)",
    "(amount - mean(amount)) / std(amount)",
    "quantile(amount, 0.9)",
    'regex_replace(phone, "[^0-9]", "")',
    'first_name + " " + last_name',
    ":clean_name(name)",
    ':class_names["1"]',
    "revenue >= p90",
}


def _literal_string_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string_node(node.left)
        right = _literal_string_node(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _pivotal_sources_from_python(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        source = _literal_string_node(node)
        if not source or "\n" not in source:
            continue
        stripped = source.lstrip()
        if "with " in source or stripped.startswith(
            ("scalar ", "list ", "dict ", "function ")
        ):
            yield source


def _pivotal_sources_from_markdown(path):
    text = path.read_text(encoding="utf-8")
    yield from re.findall(r"```pivotal\s*\n(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for inline in re.findall(r"`([^`\n]*\b[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^`\n]+)`", text):
        yield f"with _table\n{inline}\n"


def _assignment_nodes(value):
    if isinstance(value, dict):
        if value.get("type") == "assign":
            yield value
        for child in value.values():
            yield from _assignment_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _assignment_nodes(child)


def _assignment_expressions(value):
    for node in _assignment_nodes(value):
        if isinstance(node.get("expression"), str):
            yield node["expression"]
        if isinstance(node.get("default_expr"), str):
            yield node["default_expr"]
        for case in node.get("cases") or []:
            if isinstance(case, dict) and isinstance(case.get("expression"), str):
                yield case["expression"]


def _current_assignment_expression_corpus():
    parser = DSLParser()
    corpus = {}
    sources_by_file = []

    for path in CONFORMANCE_PY_FILES:
        sources_by_file.extend((path, source) for source in _pivotal_sources_from_python(path))
    for path in CONFORMANCE_MARKDOWN_FILES:
        sources_by_file.extend((path, source) for source in _pivotal_sources_from_markdown(path))

    for path, source in sources_by_file:
        if "=" not in source:
            continue
        parsed = parser.parse(source)
        if isinstance(parsed, dict) and "error" in parsed:
            continue
        for expression in _assignment_expressions(parsed):
            corpus.setdefault(expression, set()).add(str(path))

    return corpus


def test_current_assignment_expression_corpus_conformance():
    corpus = _current_assignment_expression_corpus()
    fallbacks = set()

    assert len(corpus) >= 100
    assert REPRESENTATIVE_CORPUS_EXPRESSIONS <= set(corpus)

    for expression in corpus:
        expression_ast = parse_expression(expression)
        if expression_ast is None:
            fallbacks.add(expression)
        else:
            json.dumps(expression_ast)

    assert fallbacks == RAW_FALLBACK_EXPRESSIONS


def test_parser_attachment_matches_standalone_parser_for_current_corpus():
    parser = DSLParser()

    for path in CONFORMANCE_PY_FILES:
        sources = _pivotal_sources_from_python(path)
        for source in sources:
            _assert_assignment_asts_match_standalone_parser(parser, source)

    for path in CONFORMANCE_MARKDOWN_FILES:
        sources = _pivotal_sources_from_markdown(path)
        for source in sources:
            _assert_assignment_asts_match_standalone_parser(parser, source)


def _assert_assignment_asts_match_standalone_parser(parser, source):
    if "=" not in source:
        return
    parsed = parser.parse(source)
    if isinstance(parsed, dict) and "error" in parsed:
        return

    for node in _assignment_nodes(parsed):
        expression = node.get("expression")
        if expression is None:
            continue
        assert "expression_ast" in node
        assert "expression_ir" in node
        assert node["expression_ast"] == parse_expression(expression)
        if node["expression_ast"] is not None:
            json.dumps(node["expression_ast"])
        if node["expression_ir"] is not None:
            json.dumps(node["expression_ir"])
