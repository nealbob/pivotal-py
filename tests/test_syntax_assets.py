import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pivotal.syntax_metadata import load_syntax_tokens


def test_syntax_tokens_include_recent_language_features():
    tokens = load_syntax_tokens()

    assert "round" in tokens["statement_keywords"]
    assert "min_periods" in tokens["clause_keywords"]
    assert "shape" in tokens["clause_keywords"]
    assert "columns" in tokens["clause_keywords"]
    assert "date_format" in tokens["builtin_functions"]
    assert "regex_extract" in tokens["builtin_functions"]


def test_generated_syntax_assets_are_current():
    result = subprocess.run(
        [sys.executable, "scripts/generate_syntax_assets.py", "--check"],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_syntax_tokens_json_is_valid():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(root, "pivotal", "syntax_tokens.json"), encoding="utf-8") as f:
        data = json.load(f)

    assert "statement_keywords" in data
    assert all(isinstance(value, list) for value in data.values())
