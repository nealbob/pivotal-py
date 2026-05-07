import json
import os
import subprocess
import sys

import pandas as pd


def _run_pivotal(args, cwd):
    env = os.environ.copy()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_root if not existing else repo_root + os.pathsep + existing
    return subprocess.run(
        [sys.executable, "-m", "pivotal", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_verify_cli_json_success_with_csv_input(tmp_path):
    csv_path = tmp_path / "sales.csv"
    pivotal_path = tmp_path / "pipeline.pivotal"
    pd.DataFrame({"amount": [10, -5, 20]}).to_csv(csv_path, index=False)
    pivotal_path.write_text(
        "with sales\n    filter amount > 0\n    doubled = amount * 2\n",
        encoding="utf-8",
    )

    completed = _run_pivotal(
        [
            "--verify",
            "--json",
            "--input",
            f"sales={csv_path}",
            "--return",
            "sales",
            str(pivotal_path),
        ],
        tmp_path,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["stage"] == "execute"
    assert result["tables"]["sales"]["preview"] == [
        {"amount": 10, "doubled": 20},
        {"amount": 20, "doubled": 40},
    ]


def test_verify_cli_json_parse_error_exits_nonzero(tmp_path):
    pivotal_path = tmp_path / "bad.pivotal"
    pivotal_path.write_text("with sales\n    filter\n", encoding="utf-8")

    completed = _run_pivotal(["--verify", "--json", str(pivotal_path)], tmp_path)

    assert completed.returncode == 1
    result = json.loads(completed.stdout)
    assert result["ok"] is False
    assert result["stage"] == "parse"
    assert result["error_type"] == "Syntax Error"


def test_verify_cli_plain_success_summary(tmp_path):
    csv_path = tmp_path / "sales.csv"
    pivotal_path = tmp_path / "pipeline.pivotal"
    pd.DataFrame({"amount": [10, 20]}).to_csv(csv_path, index=False)
    pivotal_path.write_text("with sales\n    filter amount > 10\n", encoding="utf-8")

    completed = _run_pivotal(
        ["--verify", "--input", f"sales={csv_path}", str(pivotal_path)],
        tmp_path,
    )

    assert completed.returncode == 0
    assert "Pivotal verification passed" in completed.stdout
    assert "sales: 1 row(s), 1 column(s)" in completed.stdout


def test_compare_cli_json_match(tmp_path):
    csv_path = tmp_path / "sales.csv"
    pandas_path = tmp_path / "original.py"
    pivotal_path = tmp_path / "converted.pivotal"
    pd.DataFrame({"amount": [10, -5, 20]}).to_csv(csv_path, index=False)
    pandas_path.write_text(
        "result = sales.copy()\n"
        "result = result[result['amount'] > 0].copy()\n"
        "result['doubled'] = result['amount'] * 2\n",
        encoding="utf-8",
    )
    pivotal_path.write_text(
        "with sales as result\n    filter amount > 0\n    doubled = amount * 2\n",
        encoding="utf-8",
    )

    completed = _run_pivotal(
        [
            "--compare",
            "--json",
            "--pandas",
            str(pandas_path),
            "--pivotal",
            str(pivotal_path),
            "--output",
            "result",
            "--input",
            f"sales={csv_path}",
        ],
        tmp_path,
    )

    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["match"] is True
    assert result["differences"] == []


def test_compare_cli_json_mismatch_exits_nonzero(tmp_path):
    csv_path = tmp_path / "sales.csv"
    pandas_path = tmp_path / "original.py"
    pivotal_path = tmp_path / "converted.pivotal"
    pd.DataFrame({"amount": [10, 20]}).to_csv(csv_path, index=False)
    pandas_path.write_text(
        "result = sales.copy()\nresult['doubled'] = result['amount'] * 2\n",
        encoding="utf-8",
    )
    pivotal_path.write_text(
        "with sales as result\n    doubled = amount * 3\n",
        encoding="utf-8",
    )

    completed = _run_pivotal(
        [
            "--compare",
            "--json",
            "--pandas",
            str(pandas_path),
            "--pivotal",
            str(pivotal_path),
            "--output",
            "result",
            "--input",
            f"sales={csv_path}",
        ],
        tmp_path,
    )

    assert completed.returncode == 1
    result = json.loads(completed.stdout)
    assert result["ok"] is False
    assert result["match"] is False
    assert result["differences"][0]["kind"] == "value_mismatch"
