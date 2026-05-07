"""MCP server exposing Pivotal verification and comparison tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from .dsl_parser import DSLParser
from .runner import (
    compare_pandas_to_pivotal_isolated,
    run_pivotal_isolated,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SYNTAX_PATH = _REPO_ROOT / "PIVOTAL.md"


def _load_input_files(input_files: Optional[Mapping[str, str]]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for name, path_text in (input_files or {}).items():
        if not name.isidentifier():
            raise ValueError(f"Invalid input table name: {name}")
        path = Path(path_text)
        if not path.is_file():
            raise ValueError(f"Input file not found: {path}")

        lower = path.name.lower()
        if lower.endswith(".csv"):
            inputs[name] = pd.read_csv(path)
        elif lower.endswith(".parquet"):
            inputs[name] = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported input file type for {path}; use CSV or Parquet")
    return inputs


def _read_text_file(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_file():
        raise ValueError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def _trim_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        max_chars = 1
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip() + "\n\n[truncated]", True


def _syntax_heading_level(line: str) -> Optional[int]:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    hashes = len(stripped) - len(stripped.lstrip("#"))
    if hashes and len(stripped) > hashes and stripped[hashes] == " ":
        return hashes
    return None


def get_pivotal_syntax(topic: Optional[str] = None, max_chars: int = 12000) -> dict[str, Any]:
    """Return all or part of PIVOTAL.md for MCP syntax grounding."""
    text = _SYNTAX_PATH.read_text(encoding="utf-8")
    if not topic:
        content, truncated = _trim_text(text, max_chars)
        return {
            "ok": True,
            "topic": None,
            "source": str(_SYNTAX_PATH),
            "content": content,
            "truncated": truncated,
        }

    lines = text.splitlines()
    topic_lower = topic.lower()
    sections: list[tuple[int, int, int]] = []
    for idx, line in enumerate(lines):
        level = _syntax_heading_level(line)
        if level is not None:
            sections.append((idx, level, len(lines)))
    for pos, (start, level, _) in enumerate(sections):
        end = len(lines)
        for next_start, next_level, _ in sections[pos + 1:]:
            if next_level <= level:
                end = next_start
                break
        sections[pos] = (start, level, end)

    for start, _, end in sections:
        body = "\n".join(lines[start:end])
        if topic_lower in body.lower():
            content, truncated = _trim_text(body, max_chars)
            return {
                "ok": True,
                "topic": topic,
                "source": str(_SYNTAX_PATH),
                "content": content,
                "truncated": truncated,
            }

    return {
        "ok": False,
        "topic": topic,
        "source": str(_SYNTAX_PATH),
        "content": "",
        "truncated": False,
        "message": f"No Pivotal syntax section matched topic '{topic}'.",
    }


def compile_pivotal_source(source: str, backend: str = "pandas") -> dict[str, Any]:
    """Compile Pivotal source without executing the generated code."""
    parser = DSLParser()
    ast_list = parser.parse(source)
    if isinstance(ast_list, dict) and "error" in ast_list:
        err = ast_list["error"]
        return {
            "ok": False,
            "stage": "parse",
            "error_type": getattr(err, "error_type", "Error"),
            "message": getattr(err, "message", str(err)),
            "line": getattr(err, "line", None),
            "column": getattr(err, "column", None),
            "source_line": getattr(err, "source_line", None),
            "suggestion": getattr(err, "suggestion", None),
        }
    try:
        code_blocks = parser.generate_code(ast_list, backend=backend)
    except Exception as exc:  # noqa: BLE001 - structured tool boundary
        return {
            "ok": False,
            "stage": "codegen",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    return {
        "ok": True,
        "stage": "codegen",
        "backend": backend,
        "generated_code": "\n\n".join(code_blocks),
    }


def get_pivotal_examples(kind: Optional[str] = None) -> dict[str, Any]:
    """Return MCP tool-call examples, especially input_files shape."""
    examples = {
        "run": {
            "description": "Run Pivotal against a CSV file loaded as table 'sales'.",
            "create_fixture_powershell": (
                "Set-Content -Path .\\sales.csv -Value @'\n"
                "region,product,amount,price,quantity,cost\n"
                "North,Widget,120,10,12,80\n"
                "South,Gadget,75,15,5,40\n"
                "West,Doohickey,240,20,12,150\n"
                "North,Gizmo,180,30,6,110\n"
                "'@"
            ),
            "tool": "pivotal_run",
            "arguments": {
                "source": (
                    "with sales as report\n"
                    "    filter amount > 100\n"
                    "    revenue = price * quantity\n"
                    "    margin = revenue - cost\n"
                    "    group by region\n"
                    "        agg sum revenue as total_revenue, sum margin as total_margin\n"
                    "    sort total_revenue desc\n"
                ),
                "input_files": {"sales": "C:\\path\\to\\sales.csv"},
                "return_tables": ["report"],
                "timeout_seconds": 30,
            },
            "notes": [
                "input_files is a mapping of Pivotal table name to local CSV/Parquet file path.",
                "The key must be a table name like 'sales', not a filename like 'sales.csv'.",
                "The value must be a path to an existing .csv or .parquet file, not inline CSV text.",
            ],
        },
        "compare": {
            "description": "Compare pandas code and Pivotal code on the same input table.",
            "tool": "pivotal_compare",
            "arguments": {
                "pandas_source": (
                    "report = sales.copy()\n"
                    "report = report[report['amount'] > 100].copy()\n"
                    "report['revenue'] = report['price'] * report['quantity']\n"
                    "report['margin'] = report['revenue'] - report['cost']\n"
                    "report = report.groupby('region', as_index=False).agg(\n"
                    "    total_revenue=('revenue', 'sum'),\n"
                    "    total_margin=('margin', 'sum'),\n"
                    ")\n"
                    "report = report.sort_values('total_revenue', ascending=False).reset_index(drop=True)\n"
                ),
                "pivotal_source": (
                    "with sales as report\n"
                    "    filter amount > 100\n"
                    "    revenue = price * quantity\n"
                    "    margin = revenue - cost\n"
                    "    group by region\n"
                    "        agg sum revenue as total_revenue, sum margin as total_margin\n"
                    "    sort total_revenue desc\n"
                ),
                "output_table": "report",
                "input_files": {"sales": "C:\\path\\to\\sales.csv"},
            },
        },
    }
    if kind:
        key = kind.lower()
        if key in examples:
            return {"ok": True, "kind": key, "examples": {key: examples[key]}}
        return {
            "ok": False,
            "kind": kind,
            "message": f"Unknown example kind '{kind}'. Use one of: {', '.join(examples)}.",
            "examples": {},
        }
    return {"ok": True, "kind": None, "examples": examples}


def create_mcp_server():
    """Create the FastMCP server.

    Importing the MCP SDK is intentionally lazy because Pivotal supports Python
    3.9, while the current MCP Python SDK requires Python 3.10+.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "The Pivotal MCP server requires the optional MCP dependency. "
            "Install it with: pip install 'pivotal-lang[mcp]'"
        ) from exc

    mcp = FastMCP(
        "Pivotal",
        instructions=(
            "Tools for generating, running, and comparing Pivotal DSL code. "
            "Prefer native Pivotal syntax and use python blocks only for "
            "operations that Pivotal cannot express. For tools with input_files, "
            "pass a mapping of Pivotal table name to local CSV/Parquet file path, "
            "for example {'sales': 'C:\\\\data\\\\sales.csv'}. Do not use the "
            "filename as the key, and do not pass inline CSV text as the value."
        ),
    )

    @mcp.tool()
    def pivotal_syntax(topic: Optional[str] = None, max_chars: int = 12000) -> dict[str, Any]:
        """Return Pivotal language syntax guidance from PIVOTAL.md."""
        return get_pivotal_syntax(topic=topic, max_chars=max_chars)

    @mcp.tool()
    def pivotal_examples(kind: Optional[str] = None) -> dict[str, Any]:
        """Return working MCP call examples. kind can be 'run' or 'compare'."""
        return get_pivotal_examples(kind=kind)

    @mcp.tool()
    def pivotal_run(
        source: str,
        backend: str = "pandas",
        input_files: Optional[dict[str, str]] = None,
        return_tables: Optional[list[str]] = None,
        max_rows: int = 20,
        timeout_seconds: float = 10,
        include_generated_code: bool = True,
    ) -> dict[str, Any]:
        """Run Pivotal source and return structured results.

        input_files must map Pivotal table names to existing local CSV/Parquet
        file paths. Example: {"sales": "C:\\data\\sales.csv"}. The key is the
        table name used in Pivotal (`with sales`), not the filename, and the
        value is a file path, not inline CSV content.
        """
        try:
            inputs = _load_input_files(input_files)
            return run_pivotal_isolated(
                source,
                backend=backend,
                inputs=inputs,
                return_tables=return_tables,
                max_rows=max_rows,
                timeout_seconds=timeout_seconds,
                include_generated_code=include_generated_code,
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary
            return {
                "ok": False,
                "stage": "mcp_input",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

    @mcp.tool()
    def pivotal_compile(
        source: str,
        backend: str = "pandas",
    ) -> dict[str, Any]:
        """Compile Pivotal source to backend code without executing data pipelines."""
        return compile_pivotal_source(source, backend=backend)

    @mcp.tool()
    def pivotal_compare(
        pandas_source: str,
        pivotal_source: str,
        output_table: str,
        input_files: Optional[dict[str, str]] = None,
        backend: str = "pandas",
        max_rows: int = 20,
        timeout_seconds: float = 10,
        atol: float = 1e-9,
        rtol: float = 1e-9,
        check_dtype: bool = False,
        max_differences: int = 20,
    ) -> dict[str, Any]:
        """Compare a pandas script and Pivotal source on the same inputs.

        input_files must map table/variable names to existing local CSV/Parquet
        file paths. Example: {"sales": "C:\\data\\sales.csv"} makes a DataFrame
        named `sales` available to both pandas_source and pivotal_source.
        """
        try:
            inputs = _load_input_files(input_files)
            return compare_pandas_to_pivotal_isolated(
                pandas_source,
                pivotal_source,
                output_table=output_table,
                backend=backend,
                inputs=inputs,
                max_rows=max_rows,
                timeout_seconds=timeout_seconds,
                atol=atol,
                rtol=rtol,
                check_dtype=check_dtype,
                max_differences=max_differences,
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary
            return {
                "ok": False,
                "stage": "mcp_input",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

    @mcp.tool()
    def pivotal_compare_files(
        pandas_path: str,
        pivotal_path: str,
        output_table: str,
        input_files: Optional[dict[str, str]] = None,
        backend: str = "pandas",
        max_rows: int = 20,
        timeout_seconds: float = 10,
        atol: float = 1e-9,
        rtol: float = 1e-9,
        check_dtype: bool = False,
        max_differences: int = 20,
    ) -> dict[str, Any]:
        """Compare pandas and Pivotal files on the same inputs.

        input_files has the same format as pivotal_compare: table name keys,
        local CSV/Parquet path values.
        """
        try:
            return pivotal_compare(
                _read_text_file(pandas_path),
                _read_text_file(pivotal_path),
                output_table,
                input_files=input_files,
                backend=backend,
                max_rows=max_rows,
                timeout_seconds=timeout_seconds,
                atol=atol,
                rtol=rtol,
                check_dtype=check_dtype,
                max_differences=max_differences,
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary
            return {
                "ok": False,
                "stage": "mcp_input",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

    return mcp


def main() -> None:
    create_mcp_server().run()


if __name__ == "__main__":
    main()
