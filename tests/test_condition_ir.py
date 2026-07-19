"""Tests for Stage 7 additive condition IR metadata."""

import json

from pivotal import DSLParser
from pivotal.condition_ir import build_condition_ast, normalize_condition_ast


def column(name):
    return {"kind": "column", "name": name}


def literal(literal_type, value):
    return {"kind": "literal", "literal_type": literal_type, "value": value}


def predicate(operator, left, right):
    return {
        "kind": "predicate",
        "operator": operator,
        "left": left,
        "right": right,
    }


def test_build_condition_ast_from_public_condition_structures():
    ast = build_condition_ast(
        [
            {"column": "amount", "comparator": ">", "value": 100},
            {"column": "region", "comparator": "==", "value": "state"},
        ],
        ["and"],
    )

    assert ast == {
        "kind": "logical",
        "operator": "and",
        "left": predicate("greater_than", column("amount"), literal("integer", 100)),
        "right": predicate("equal", column("region"), column("state")),
    }
    assert normalize_condition_ast(ast) == ast
    json.dumps(ast)


def test_filter_condition_ir_is_additive_and_codegen_still_uses_existing_path():
    parser = DSLParser()
    nodes = parser.parse("with sales\nfilter amount > 100 and region == \"NSW\"\n")
    filter_node = nodes[1]

    assert filter_node["conditions"] == [
        {"column": "amount", "comparator": ">", "value": 100},
        {"column": "region", "comparator": "==", "value": "NSW"},
    ]
    assert filter_node["operators"] == ["and"]
    assert filter_node["condition_ir"]["kind"] == "logical"
    assert filter_node["condition_ir"]["left"] == predicate(
        "greater_than", column("amount"), literal("integer", 100)
    )
    assert filter_node["condition_ir"]["right"] == predicate(
        "equal", column("region"), literal("string", "NSW")
    )

    code = "\n".join(parser.generate_code(nodes))
    assert "sales = sales.query('amount > 100 and region == \"NSW\"')" in code


def test_condition_ir_attaches_to_assign_where_and_case_branches():
    parser = DSLParser()
    nodes = parser.parse(
        "with sales\n"
        "discount = price * 0.9\n"
        "    where amount >= :minimum\n"
        "tier =\n"
        "    where amount > 100: \"high\"\n"
        "    else \"low\"\n"
    )

    assign_where = nodes[1]
    case_branch = nodes[2]["cases"][0]

    assert assign_where["condition_ir"] == predicate(
        "greater_than_or_equal",
        column("amount"),
        {"kind": "runtime_reference", "name": "minimum"},
    )
    assert case_branch["condition_ir"] == predicate(
        "greater_than", column("amount"), literal("integer", 100)
    )
    assert nodes[2]["cases"][1]["type"] == "case_default"
    assert "condition_ir" not in nodes[2]["cases"][1]


def test_condition_ir_attaches_to_data_quality_condition_only():
    parser = DSLParser()
    nodes = parser.parse(
        "with sales\n"
        "check amount > 0\n"
        "assert id unique\n"
    )

    check_node = nodes[1]
    unique_node = nodes[2]

    assert check_node["condition_ir"] == predicate(
        "greater_than", column("amount"), literal("integer", 0)
    )
    assert "condition_ir" not in unique_node


def test_condition_ir_falls_back_to_none_for_unsupported_metadata():
    ast = build_condition_ast(
        [{"column": "region", "comparator": "==", "value": {"type": "unknown"}}],
        [],
    )

    assert ast is None


def test_stage_8_filter_codegen_uses_condition_ir_across_backends():
    expected = {
        "pandas": "sales = sales.query('amount > 100 and region == \"NSW\"')",
        "polars": "sales = sales.filter(((pl.col('amount') > 100)) & ((pl.col('region') == 'NSW')))",
        "duckdb": "WHERE amount > 100 AND region = 'NSW'",
        "sql": "WHERE amount > 100 AND region = 'NSW'",
    }

    for backend, fragment in expected.items():
        parser = DSLParser()
        nodes = parser.parse("with sales\nfilter amount > 100 and region == \"NSW\"\n")
        filter_node = nodes[1]
        filter_node["conditions"] = [{"column": "wrong", "comparator": ">", "value": 0}]
        filter_node["operators"] = []

        code = "\n".join(parser.generate_code(nodes, backend=backend))

        assert fragment in code
        assert "wrong" not in code


def test_stage_8_condition_codegen_falls_back_to_legacy_conditions_when_ir_is_missing():
    parser = DSLParser()
    nodes = parser.parse("with sales\nfilter amount > 100\n")
    nodes[1]["condition_ir"] = None

    code = "\n".join(parser.generate_code(nodes, backend="pandas"))

    assert "sales = sales.query('amount > 100')" in code


def test_stage_8_assign_where_codegen_uses_condition_ir():
    parser = DSLParser()
    nodes = parser.parse(
        "with sales\n"
        "flag = 1\n"
        "    where amount > 100\n"
    )
    assign = nodes[1]
    assign["conditions"] = [{"column": "wrong", "comparator": ">", "value": 0}]
    assign["operators"] = []

    code = "\n".join(parser.generate_code(nodes, backend="pandas"))

    assert "condition = sales.eval('amount > 100')" in code
    assert "wrong" not in code
