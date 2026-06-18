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


@pytest.mark.parametrize(
    ("backend", "expected_fragments"),
    [
        (
            "pandas",
            [
                "sales['clean'] = sales['name'].str.strip().str.upper()",
                "sales['amount_f'] = pd.to_numeric(sales['amount'], errors='coerce')",
                "sales['capped'] = __import__('numpy').maximum((sales['amount'] - 150), 0)",
                "sales['yr'] = sales['sale_date'].dt.year",
            ],
        ),
        (
            "polars",
            [
                "pl.col('name').str.strip_chars().str.to_uppercase()",
                "pl.col('amount').cast(pl.Float64, strict=False)",
                "pl.max_horizontal([(pl.col('amount') - pl.lit(150)), pl.lit(0)])",
                "pl.col('sale_date').dt.year()",
            ],
        ),
        (
            "duckdb",
            [
                "UPPER(TRIM(name)) AS clean",
                "TRY_CAST(amount AS DOUBLE) AS amount_f",
                "GREATEST((amount - 150), 0) AS capped",
                "YEAR(sale_date) AS yr",
            ],
        ),
        (
            "sql",
            [
                "UPPER(TRIM(name)) AS clean",
                "TRY_CAST(amount AS DOUBLE) AS amount_f",
                "GREATEST((amount - 150), 0) AS capped",
                "YEAR(sale_date) AS yr",
            ],
        ),
    ],
)
def test_scalar_and_cast_assignment_uses_expression_ir_across_backends(
    backend, expected_fragments
):
    parser = DSLParser()
    nodes = parser.parse(
        "with sales\n"
        "clean = upper(trim(name))\n"
        "amount_f = float(amount)\n"
        "capped = max(amount - 150, 0)\n"
        "yr = year(sale_date)\n"
    )

    assert nodes[1]["expression_ir"]["kind"] == "scalar_function"
    assert nodes[2]["expression_ir"]["kind"] == "cast"
    assert nodes[3]["expression_ir"]["kind"] == "scalar_function"
    code = "\n".join(parser.generate_code(nodes, backend=backend))

    for expected in expected_fragments:
        assert expected in code


def test_runtime_call_expression_ir_keeps_existing_codegen_fallback():
    parser = DSLParser()
    nodes = parser.parse("with sales\nnet = :discount(price)\n")

    assert nodes[1]["expression_ir"]["kind"] == "runtime_call"
    code = "\n".join(parser.generate_code(nodes))

    assert "sales['net'] = discount(sales['price'])" in code


@pytest.mark.parametrize(
    ("backend", "expected_fragments"),
    [
        (
            "pandas",
            [
                "data['pct'] = (data['amount'] / data['amount'].sum())",
                "data['p90'] = data['amount'].quantile(0.9)",
                "data['dev'] = (data['amount'] - ((data['amount'] * data['weight']).sum() / data['weight'].sum()))",
                "data['band'] = __import__('numpy').minimum(__import__('numpy').maximum((data['amount'] - 150), 0), (data['amount'].max() / 2))",
            ],
        ),
        (
            "polars",
            [
                "(pl.col('amount') / pl.col('amount').sum())",
                "pl.col('amount').quantile(0.9, interpolation='linear')",
                "(pl.col('amount') - ((pl.col('amount') * pl.col('weight')).sum() / pl.col('weight').sum()))",
                "pl.min_horizontal([pl.max_horizontal([(pl.col('amount') - pl.lit(150)), pl.lit(0)]), (pl.col('amount').max() / pl.lit(2))])",
            ],
        ),
        (
            "duckdb",
            [
                "(amount / SUM(amount) OVER ()) AS pct",
                "QUANTILE_CONT(amount, 0.9) OVER () AS p90",
                "(amount - (SUM(amount * weight) OVER ()) / NULLIF(SUM(weight) OVER (), 0)) AS dev",
                "LEAST(GREATEST((amount - 150), 0), (MAX(amount) OVER () / 2)) AS band",
            ],
        ),
        (
            "sql",
            [
                "(amount / SUM(amount) OVER ()) AS pct",
                "QUANTILE_CONT(amount, 0.9) OVER () AS p90",
                "(amount - (SUM(amount * weight) OVER ()) / NULLIF(SUM(weight) OVER (), 0)) AS dev",
                "LEAST(GREATEST((amount - 150), 0), (MAX(amount) OVER () / 2)) AS band",
            ],
        ),
    ],
)
def test_aggregate_assignment_uses_expression_ir_across_backends(
    backend, expected_fragments
):
    parser = DSLParser()
    nodes = parser.parse(
        "with data\n"
        "pct = amount / sum(amount)\n"
        "p90 = quantile(amount, 0.9)\n"
        "dev = amount - wavg(amount, weight)\n"
        "band = min(max(amount - 150, 0), max(amount) / 2)\n"
    )

    assert nodes[1]["expression_ir"]["kind"] == "binary"
    assert nodes[2]["expression_ir"]["kind"] == "aggregate"
    code = "\n".join(parser.generate_code(nodes, backend=backend))

    for expected in expected_fragments:
        assert expected in code


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("pandas", "data['pct'] = (data['amount'] / data.groupby(['region'])['amount'].transform('sum'))"),
        ("polars", "(pl.col('amount') / pl.col('amount').sum().over('region'))"),
        ("duckdb", "(amount / SUM(amount) OVER (PARTITION BY region)) AS pct"),
        ("sql", "(amount / SUM(amount) OVER (PARTITION BY region)) AS pct"),
    ],
)
def test_grouped_aggregate_assignment_uses_expression_ir_across_backends(
    backend, expected
):
    parser = DSLParser()
    nodes = parser.parse("with data\npct = amount / sum(amount)\n    by region\n")

    assert nodes[1]["expression_ir"]["kind"] == "binary"
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
