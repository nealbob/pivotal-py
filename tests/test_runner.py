import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pivotal
from pivotal.runner import (
    compare_pandas_to_pivotal,
    compare_pandas_to_pivotal_isolated,
    run_pivotal,
    run_pivotal_isolated,
)


def test_run_pivotal_success_returns_structured_table():
    sales = pd.DataFrame({
        "region": ["North", "South", "North"],
        "amount": [10, -5, 20],
    })

    result = run_pivotal(
        "with sales\n    filter amount > 0\n    total = amount * 2\n",
        inputs={"sales": sales},
    )

    assert result["ok"] is True
    assert result["stage"] == "execute"
    assert result["backend"] == "pandas"
    assert "filter amount > 0" not in result["generated_code"]
    assert result["warnings"] == []
    assert result["stdout"] == ""
    assert result["stderr"] == ""

    table = result["tables"]["sales"]
    assert table["type"] == "pandas.DataFrame"
    assert table["shape"] == [2, 3]
    assert [col["name"] for col in table["columns"]] == ["region", "amount", "total"]
    assert table["preview"] == [
        {"region": "North", "amount": 10, "total": 20},
        {"region": "North", "amount": 20, "total": 40},
    ]
    json.dumps(result)

    # Inputs are copied before execution, so verifier runs do not mutate callers.
    assert list(sales.columns) == ["region", "amount"]


def test_run_pivotal_parse_error_is_structured():
    result = run_pivotal("with sales\n    filter\n")

    assert result["ok"] is False
    assert result["stage"] == "parse"
    assert result["error_type"] == "Syntax Error"
    assert result["line"] is not None
    assert result["message"]


def test_run_pivotal_runtime_error_includes_generated_code_and_traceback():
    sales = pd.DataFrame({"amount": [10]})

    result = run_pivotal(
        "with sales\n    filter missing > 0\n",
        inputs={"sales": sales},
    )

    assert result["ok"] is False
    assert result["stage"] == "execute"
    assert result["backend"] == "pandas"
    assert result["generated_code"]
    assert "traceback" in result
    assert result["message"]


def test_run_pivotal_supports_runtime_variables():
    sales = pd.DataFrame({"amount": [10, 25, 50]})

    result = run_pivotal(
        "with sales\n    filter amount > :minimum\n",
        inputs={"sales": sales},
        variables={"minimum": 20},
    )

    assert result["ok"] is True
    assert result["tables"]["sales"]["preview"] == [
        {"amount": 25},
        {"amount": 50},
    ]


def test_run_pivotal_return_tables_limits_output():
    sales = pd.DataFrame({"amount": [10, 20]})

    result = run_pivotal(
        "with sales as clean\n    filter amount > 10\n",
        inputs={"sales": sales},
        return_tables=["clean"],
    )

    assert result["ok"] is True
    assert list(result["tables"]) == ["clean"]
    assert result["tables"]["clean"]["preview"] == [{"amount": 20}]


def test_run_pivotal_sql_backend_compiles_without_execution():
    result = run_pivotal(
        "with sales\n    filter amount > 0\n",
        backend="sql",
    )

    assert result["ok"] is True
    assert result["stage"] == "codegen"
    assert result["tables"] == {}
    assert "WITH" in result["generated_code"]
    json.dumps(result)


def test_run_pivotal_is_public_package_api():
    assert pivotal.run_pivotal is run_pivotal


def test_run_pivotal_isolated_success_returns_structured_table():
    sales = pd.DataFrame({"amount": [10, -5, 20]})

    result = run_pivotal_isolated(
        "with sales\n    filter amount > 0\n    doubled = amount * 2\n",
        inputs={"sales": sales},
    )

    assert result["ok"] is True
    assert result["stage"] == "execute"
    assert result["tables"]["sales"]["shape"] == [2, 2]
    assert result["tables"]["sales"]["preview"] == [
        {"amount": 10, "doubled": 20},
        {"amount": 20, "doubled": 40},
    ]
    assert result["returncode"] == 0
    json.dumps(result)

    # The child process receives a serialized copy, so parent inputs are untouched.
    assert list(sales.columns) == ["amount"]


def test_run_pivotal_isolated_timeout():
    result = run_pivotal_isolated(
        "python\nimport time\ntime.sleep(2)\nend\n",
        timeout_seconds=0.25,
    )

    assert result["ok"] is False
    assert result["stage"] == "timeout"
    assert result["error_type"] == "TimeoutExpired"
    assert "timed out" in result["message"]


def test_run_pivotal_isolated_is_public_package_api():
    assert pivotal.run_pivotal_isolated is run_pivotal_isolated


def test_compare_pandas_to_pivotal_match():
    sales = pd.DataFrame({"amount": [10, -5, 20]})
    pandas_source = (
        "result = sales.copy()\n"
        "result = result[result['amount'] > 0].copy()\n"
        "result['doubled'] = result['amount'] * 2\n"
    )
    pivotal_source = "with sales as result\n    filter amount > 0\n    doubled = amount * 2\n"

    result = compare_pandas_to_pivotal(
        pandas_source,
        pivotal_source,
        output_table="result",
        inputs={"sales": sales},
    )

    assert result["ok"] is True
    assert result["match"] is True
    assert result["differences"] == []
    assert result["pandas_table"]["preview"] == result["pivotal_table"]["preview"]
    assert list(sales.columns) == ["amount"]
    json.dumps(result)


def test_compare_pandas_to_pivotal_value_mismatch():
    sales = pd.DataFrame({"amount": [10, 20]})
    pandas_source = "result = sales.copy()\nresult['doubled'] = result['amount'] * 2\n"
    pivotal_source = "with sales as result\n    doubled = amount * 3\n"

    result = compare_pandas_to_pivotal(
        pandas_source,
        pivotal_source,
        output_table="result",
        inputs={"sales": sales},
    )

    assert result["ok"] is False
    assert result["match"] is False
    assert result["stage"] == "compare"
    assert result["differences"][0]["kind"] == "value_mismatch"
    assert result["differences"][0]["column"] == "doubled"


def test_compare_pandas_to_pivotal_isolated_match():
    sales = pd.DataFrame({"amount": [10, -5, 20]})
    pandas_source = (
        "result = sales.copy()\n"
        "result = result[result['amount'] > 0].copy()\n"
        "result['doubled'] = result['amount'] * 2\n"
    )
    pivotal_source = "with sales as result\n    filter amount > 0\n    doubled = amount * 2\n"

    result = compare_pandas_to_pivotal_isolated(
        pandas_source,
        pivotal_source,
        output_table="result",
        inputs={"sales": sales},
    )

    assert result["ok"] is True
    assert result["match"] is True
    assert result["returncode"] == 0


def test_compare_pandas_to_pivotal_is_public_package_api():
    assert pivotal.compare_pandas_to_pivotal is compare_pandas_to_pivotal
    assert pivotal.compare_pandas_to_pivotal_isolated is compare_pandas_to_pivotal_isolated
