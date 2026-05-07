"""Structured execution helpers for AI and automation workflows."""
from __future__ import annotations

import contextlib
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import traceback
import warnings
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Optional, Sequence, Union

import pandas as pd
from pandas.testing import assert_frame_equal

from .dsl_parser import DSLParser
from .errors import PivotalError, _translate_runtime_error


_SUPPORTED_BACKENDS = {"pandas", "polars", "duckdb", "sql"}


def _json_safe(value: Any) -> Any:
    """Return a JSON-compatible representation of arbitrary result values."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _error_payload(
    stage: str,
    exc: Union[BaseException, PivotalError],
    *,
    traceback_text: Optional[str] = None,
) -> dict[str, Any]:
    if isinstance(exc, PivotalError):
        payload = asdict(exc)
    elif is_dataclass(exc):
        payload = asdict(exc)  # pragma: no cover - defensive for future error types
    else:
        translated = _translate_runtime_error(exc)
        if translated is not None:
            payload = asdict(translated)
            payload["raw_message"] = str(exc)
        else:
            payload = {
                "message": str(exc),
                "error_type": type(exc).__name__,
                "line": None,
                "column": None,
                "source_line": None,
                "suggestion": None,
            }

    payload["ok"] = False
    payload["stage"] = stage
    if traceback_text:
        payload["traceback"] = traceback_text
    return payload


def _copy_input(value: Any) -> Any:
    if hasattr(value, "copy"):
        try:
            return value.copy()
        except TypeError:
            return value.copy(deep=True)
    return value


def _normalise_inputs(inputs: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    for name, value in (inputs or {}).items():
        namespace[name] = _copy_input(value)
    return namespace


def _pandas_table_payload(df: pd.DataFrame, max_rows: int) -> dict[str, Any]:
    preview_json = df.head(max_rows).to_json(orient="records", date_format="iso")
    preview = json.loads(preview_json)
    return {
        "type": "pandas.DataFrame",
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [
            {"name": _json_safe(col), "dtype": str(dtype)}
            for col, dtype in df.dtypes.items()
        ],
        "preview": preview,
    }


def _polars_table_payload(df: Any, max_rows: int) -> dict[str, Any]:
    preview = df.head(max_rows).to_dicts()
    return {
        "type": "polars.DataFrame",
        "shape": [int(df.height), int(df.width)],
        "columns": [
            {"name": _json_safe(name), "dtype": str(dtype)}
            for name, dtype in zip(df.columns, df.dtypes)
        ],
        "preview": _json_safe(preview),
    }


def _table_payload(value: Any, max_rows: int) -> Optional[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return _pandas_table_payload(value, max_rows)

    # Avoid importing optional backends unless the object is already present.
    if value.__class__.__module__.startswith("polars") and hasattr(value, "to_dicts"):
        return _polars_table_payload(value, max_rows)

    return None


def _table_names_from_ast(ast_list: Sequence[Any]) -> list[str]:
    names: list[str] = []
    for node in ast_list:
        if not isinstance(node, dict):
            continue
        name = node.get("table_name")
        if isinstance(name, str) and name not in names:
            names.append(name)
    return names


def _collect_tables(
    namespace: Mapping[str, Any],
    table_names: Sequence[str],
    *,
    max_rows: int,
) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for name in table_names:
        if name not in namespace:
            continue
        payload = _table_payload(namespace[name], max_rows)
        if payload is not None:
            tables[name] = payload
    return tables


def _warning_payloads(caught: Sequence[warnings.WarningMessage]) -> list[dict[str, Any]]:
    return [
        {
            "category": item.category.__name__,
            "message": str(item.message),
        }
        for item in caught
    ]


def _run_pandas_source(
    source: str,
    *,
    inputs: Optional[Mapping[str, Any]] = None,
    variables: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    namespace = {"pd": pd}
    namespace.update(_normalise_inputs(inputs))
    namespace.update(dict(variables or {}))

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                exec(source, namespace)  # noqa: S102 - local comparison contract
        except Exception as exc:  # noqa: BLE001 - structured API boundary
            payload = _error_payload("pandas_execute", exc, traceback_text=traceback.format_exc())
            payload["stdout"] = stdout_buffer.getvalue()
            payload["stderr"] = stderr_buffer.getvalue()
            payload["warnings"] = _warning_payloads(caught)
            return {"ok": False, "result": payload, "namespace": namespace}

    return {
        "ok": True,
        "result": {
            "ok": True,
            "stage": "pandas_execute",
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "warnings": _warning_payloads(caught),
        },
        "namespace": namespace,
    }


def _run_pivotal_namespace(
    source: str,
    *,
    backend: str,
    inputs: Optional[Mapping[str, Any]] = None,
    variables: Optional[Mapping[str, Any]] = None,
    return_tables: Optional[Sequence[str]] = None,
    max_rows: int = 20,
    include_generated_code: bool = True,
) -> dict[str, Any]:
    if backend == "sql":
        result = run_pivotal(
            source,
            backend=backend,
            inputs=inputs,
            variables=variables,
            return_tables=return_tables,
            max_rows=max_rows,
            include_generated_code=include_generated_code,
        )
        return {"ok": result.get("ok", False), "result": result, "namespace": {}}

    parser = DSLParser()
    namespace = {"pd": pd}
    namespace.update(_normalise_inputs(inputs))
    namespace.update(dict(variables or {}))

    ast_list = parser.parse(source)
    if isinstance(ast_list, dict) and "error" in ast_list:
        result = _error_payload("parse", ast_list["error"])
        return {"ok": False, "result": result, "namespace": namespace}

    try:
        ast_list = parser._expand_for_loops(ast_list, namespace)
    except Exception as exc:  # noqa: BLE001 - structured API boundary
        result = _error_payload("expand", exc, traceback_text=traceback.format_exc())
        return {"ok": False, "result": result, "namespace": namespace}

    try:
        code_blocks = parser.generate_code(ast_list, backend=backend)
    except Exception as exc:  # noqa: BLE001 - structured API boundary
        result = _error_payload("codegen", exc, traceback_text=traceback.format_exc())
        return {"ok": False, "result": result, "namespace": namespace}

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                for block in code_blocks:
                    exec(block, namespace)  # noqa: S102 - Pivotal execution contract
        except Exception as exc:  # noqa: BLE001 - structured API boundary
            result = _error_payload("execute", exc, traceback_text=traceback.format_exc())
            result["backend"] = backend
            result["generated_code"] = "\n\n".join(code_blocks) if include_generated_code else None
            result["warnings"] = _warning_payloads(caught)
            result["stdout"] = stdout_buffer.getvalue()
            result["stderr"] = stderr_buffer.getvalue()
            return {"ok": False, "result": result, "namespace": namespace}

    names = list(return_tables) if return_tables is not None else _table_names_from_ast(ast_list)
    result = {
        "ok": True,
        "stage": "execute",
        "backend": backend,
        "generated_code": "\n\n".join(code_blocks) if include_generated_code else None,
        "tables": _collect_tables(namespace, names, max_rows=max_rows),
        "warnings": _warning_payloads(caught),
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
    }
    return {"ok": True, "result": result, "namespace": namespace}


def _values_equal(left: Any, right: Any, *, atol: float, rtol: float) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    try:
        if pd.api.types.is_number(left) and pd.api.types.is_number(right):
            return bool(abs(left - right) <= (atol + rtol * abs(right)))
    except TypeError:
        pass
    return bool(left == right)


def _compare_dataframes(
    pandas_df: pd.DataFrame,
    pivotal_df: pd.DataFrame,
    *,
    atol: float,
    rtol: float,
    check_dtype: bool,
    max_differences: int,
) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
    differences: list[dict[str, Any]] = []
    warnings_out: list[dict[str, Any]] = []

    pandas_shape = [int(pandas_df.shape[0]), int(pandas_df.shape[1])]
    pivotal_shape = [int(pivotal_df.shape[0]), int(pivotal_df.shape[1])]
    if pandas_shape != pivotal_shape:
        differences.append({"kind": "shape_mismatch", "pandas": pandas_shape, "pivotal": pivotal_shape})

    pandas_cols = list(pandas_df.columns)
    pivotal_cols = list(pivotal_df.columns)
    if pandas_cols != pivotal_cols:
        differences.append({
            "kind": "column_order_mismatch",
            "pandas": [_json_safe(col) for col in pandas_cols],
            "pivotal": [_json_safe(col) for col in pivotal_cols],
        })

    for col in pandas_cols:
        if col not in pivotal_df.columns:
            differences.append({"kind": "missing_column", "side": "pivotal", "column": _json_safe(col)})
    for col in pivotal_cols:
        if col not in pandas_df.columns:
            differences.append({"kind": "extra_column", "side": "pivotal", "column": _json_safe(col)})

    shared_cols = [col for col in pandas_cols if col in pivotal_df.columns]
    for col in shared_cols:
        pandas_dtype = str(pandas_df[col].dtype)
        pivotal_dtype = str(pivotal_df[col].dtype)
        if pandas_dtype != pivotal_dtype:
            payload = {
                "kind": "dtype_mismatch",
                "column": _json_safe(col),
                "pandas": pandas_dtype,
                "pivotal": pivotal_dtype,
            }
            if check_dtype:
                differences.append(payload)
            else:
                warnings_out.append(payload)

    if not differences:
        try:
            assert_frame_equal(
                pandas_df.reset_index(drop=True),
                pivotal_df.reset_index(drop=True),
                check_dtype=check_dtype,
                check_like=False,
                atol=atol,
                rtol=rtol,
            )
        except AssertionError:
            row_count = min(len(pandas_df), len(pivotal_df))
            for row_idx in range(row_count):
                if len(differences) >= max_differences:
                    break
                for col in shared_cols:
                    if len(differences) >= max_differences:
                        break
                    left = pandas_df.iloc[row_idx][col]
                    right = pivotal_df.iloc[row_idx][col]
                    if not _values_equal(left, right, atol=atol, rtol=rtol):
                        differences.append({
                            "kind": "value_mismatch",
                            "row": int(row_idx),
                            "column": _json_safe(col),
                            "pandas": _json_safe(left),
                            "pivotal": _json_safe(right),
                        })
            if len(differences) >= max_differences:
                differences.append({
                    "kind": "diff_truncated",
                    "max_differences": int(max_differences),
                })

    return not differences, differences, warnings_out


def run_pivotal(
    source: str,
    *,
    backend: str = "pandas",
    inputs: Optional[Mapping[str, Any]] = None,
    variables: Optional[Mapping[str, Any]] = None,
    return_tables: Optional[Sequence[str]] = None,
    max_rows: int = 20,
    include_generated_code: bool = True,
) -> dict[str, Any]:
    """Parse, generate, and execute Pivotal source with structured results.

    This function is intentionally JSON-friendly so it can back CLI commands,
    MCP tools, notebook widgets, and AI verifier loops without requiring callers
    to scrape printed output.
    """
    if backend not in _SUPPORTED_BACKENDS:
        return _error_payload(
            "setup",
            ValueError(
                f"Unsupported backend '{backend}'. "
                f"Expected one of: {', '.join(sorted(_SUPPORTED_BACKENDS))}"
            ),
        )

    if backend == "sql":
        parser = DSLParser()
        ast_list = parser.parse(source)
        if isinstance(ast_list, dict) and "error" in ast_list:
            return _error_payload("parse", ast_list["error"])
        try:
            code_blocks = parser.generate_code(ast_list, backend=backend)
        except Exception as exc:  # noqa: BLE001 - structured API boundary
            return _error_payload("codegen", exc, traceback_text=traceback.format_exc())
        return {
            "ok": True,
            "stage": "codegen",
            "backend": backend,
            "generated_code": "\n\n".join(code_blocks) if include_generated_code else None,
            "tables": {},
            "warnings": [],
            "stdout": "",
            "stderr": "",
        }

    parser = DSLParser()
    namespace = {"pd": pd}
    namespace.update(_normalise_inputs(inputs))
    namespace.update(dict(variables or {}))

    ast_list = parser.parse(source)
    if isinstance(ast_list, dict) and "error" in ast_list:
        return _error_payload("parse", ast_list["error"])

    try:
        ast_list = parser._expand_for_loops(ast_list, namespace)
    except Exception as exc:  # noqa: BLE001 - structured API boundary
        return _error_payload("expand", exc, traceback_text=traceback.format_exc())

    try:
        code_blocks = parser.generate_code(ast_list, backend=backend)
    except Exception as exc:  # noqa: BLE001 - structured API boundary
        return _error_payload("codegen", exc, traceback_text=traceback.format_exc())

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    warning_payloads: list[dict[str, Any]] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                for block in code_blocks:
                    exec(block, namespace)  # noqa: S102 - Pivotal execution contract
        except Exception as exc:  # noqa: BLE001 - structured API boundary
            for item in caught:
                warning_payloads.append(
                    {
                        "category": item.category.__name__,
                        "message": str(item.message),
                    }
                )
            payload = _error_payload(
                "execute",
                exc,
                traceback_text=traceback.format_exc(),
            )
            payload["backend"] = backend
            payload["generated_code"] = "\n\n".join(code_blocks) if include_generated_code else None
            payload["warnings"] = warning_payloads
            payload["stdout"] = stdout_buffer.getvalue()
            payload["stderr"] = stderr_buffer.getvalue()
            return payload

        for item in caught:
            warning_payloads.append(
                {
                    "category": item.category.__name__,
                    "message": str(item.message),
                }
            )

    names = list(return_tables) if return_tables is not None else _table_names_from_ast(ast_list)
    tables = _collect_tables(namespace, names, max_rows=max_rows)

    return {
        "ok": True,
        "stage": "execute",
        "backend": backend,
        "generated_code": "\n\n".join(code_blocks) if include_generated_code else None,
        "tables": tables,
        "warnings": warning_payloads,
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
    }


def run_pivotal_isolated(
    source: str,
    *,
    backend: str = "pandas",
    inputs: Optional[Mapping[str, Any]] = None,
    variables: Optional[Mapping[str, Any]] = None,
    return_tables: Optional[Sequence[str]] = None,
    max_rows: int = 20,
    include_generated_code: bool = True,
    timeout_seconds: float = 10,
) -> dict[str, Any]:
    """Run Pivotal in a fresh Python subprocess with a timeout.

    The subprocess boundary keeps each verifier call in a clean interpreter and
    gives callers a reliable way to stop runaway generated code. Inputs are
    pickled across the local parent/child boundary so DataFrames can be supplied
    without inventing a lossy JSON encoding for Phase 2.
    """
    request = {
        "kind": "run_pivotal",
        "source": source,
        "backend": backend,
        "inputs": dict(inputs or {}),
        "variables": dict(variables or {}),
        "return_tables": list(return_tables) if return_tables is not None else None,
        "max_rows": max_rows,
        "include_generated_code": include_generated_code,
    }

    with tempfile.TemporaryDirectory(prefix="pivotal-run-") as tmpdir:
        request_path = os.path.join(tmpdir, "request.pkl")
        result_path = os.path.join(tmpdir, "result.json")
        with open(request_path, "wb") as f:
            pickle.dump(request, f, protocol=pickle.HIGHEST_PROTOCOL)

        cmd = [
            sys.executable,
            "-m",
            "pivotal._runner_worker",
            request_path,
            result_path,
        ]
        try:
            completed = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "stage": "timeout",
                "error_type": "TimeoutExpired",
                "message": f"Pivotal execution timed out after {timeout_seconds} second(s)",
                "line": None,
                "column": None,
                "source_line": None,
                "suggestion": "Reduce the input size or inspect the generated code for long-running operations.",
                "backend": backend,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "returncode": None,
            }

        if os.path.exists(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                result.setdefault("worker_stdout", completed.stdout)
                result.setdefault("worker_stderr", completed.stderr)
                result.setdefault("returncode", completed.returncode)
                return result
            except json.JSONDecodeError as exc:
                return {
                    "ok": False,
                    "stage": "worker",
                    "error_type": "JSONDecodeError",
                    "message": f"Could not decode worker result: {exc}",
                    "line": None,
                    "column": None,
                    "source_line": None,
                    "suggestion": None,
                    "backend": backend,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "returncode": completed.returncode,
                }

        return {
            "ok": False,
            "stage": "worker",
            "error_type": "WorkerError",
            "message": "Pivotal worker exited without writing a result.",
            "line": None,
            "column": None,
            "source_line": None,
            "suggestion": None,
            "backend": backend,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        }


def compare_pandas_to_pivotal(
    pandas_source: str,
    pivotal_source: str,
    *,
    output_table: str,
    backend: str = "pandas",
    inputs: Optional[Mapping[str, Any]] = None,
    variables: Optional[Mapping[str, Any]] = None,
    max_rows: int = 20,
    include_generated_code: bool = True,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    check_dtype: bool = False,
    max_differences: int = 20,
) -> dict[str, Any]:
    """Run pandas and Pivotal pipelines and compare one output table."""
    if backend == "sql":
        return _error_payload(
            "setup",
            ValueError("Pandas comparison requires an executable Pivotal backend, not 'sql'."),
        )

    pandas_run = _run_pandas_source(pandas_source, inputs=inputs, variables=variables)
    if not pandas_run["ok"]:
        result = pandas_run["result"]
        result["pandas_ok"] = False
        result["pivotal_ok"] = None
        result["match"] = False
        result["output_table"] = output_table
        return result

    pandas_namespace = pandas_run["namespace"]
    if output_table not in pandas_namespace:
        result = _error_payload(
            "pandas_output",
            KeyError(output_table),
        )
        result["message"] = f"Pandas output table '{output_table}' was not created."
        result["pandas_ok"] = False
        result["pivotal_ok"] = None
        result["match"] = False
        result["output_table"] = output_table
        result["pandas_result"] = pandas_run["result"]
        return result

    pandas_df = pandas_namespace[output_table]
    if not isinstance(pandas_df, pd.DataFrame):
        result = _error_payload(
            "pandas_output",
            TypeError(f"Pandas output '{output_table}' is {type(pandas_df).__name__}, not a DataFrame."),
        )
        result["pandas_ok"] = False
        result["pivotal_ok"] = None
        result["match"] = False
        result["output_table"] = output_table
        result["pandas_result"] = pandas_run["result"]
        return result

    pivotal_run = _run_pivotal_namespace(
        pivotal_source,
        backend=backend,
        inputs=inputs,
        variables=variables,
        return_tables=[output_table],
        max_rows=max_rows,
        include_generated_code=include_generated_code,
    )
    if not pivotal_run["ok"]:
        result = pivotal_run["result"]
        result["pandas_ok"] = True
        result["pivotal_ok"] = False
        result["match"] = False
        result["output_table"] = output_table
        result["pandas_result"] = pandas_run["result"]
        result["pandas_table"] = _pandas_table_payload(pandas_df, max_rows)
        return result

    pivotal_namespace = pivotal_run["namespace"]
    if output_table not in pivotal_namespace:
        result = _error_payload(
            "pivotal_output",
            KeyError(output_table),
        )
        result["message"] = f"Pivotal output table '{output_table}' was not created."
        result["pandas_ok"] = True
        result["pivotal_ok"] = False
        result["match"] = False
        result["output_table"] = output_table
        result["pandas_result"] = pandas_run["result"]
        result["pivotal_result"] = pivotal_run["result"]
        result["pandas_table"] = _pandas_table_payload(pandas_df, max_rows)
        return result

    pivotal_df = pivotal_namespace[output_table]
    if not isinstance(pivotal_df, pd.DataFrame):
        result = _error_payload(
            "pivotal_output",
            TypeError(f"Pivotal output '{output_table}' is {type(pivotal_df).__name__}, not a DataFrame."),
        )
        result["pandas_ok"] = True
        result["pivotal_ok"] = False
        result["match"] = False
        result["output_table"] = output_table
        result["pandas_result"] = pandas_run["result"]
        result["pivotal_result"] = pivotal_run["result"]
        result["pandas_table"] = _pandas_table_payload(pandas_df, max_rows)
        return result

    match, differences, compare_warnings = _compare_dataframes(
        pandas_df,
        pivotal_df,
        atol=atol,
        rtol=rtol,
        check_dtype=check_dtype,
        max_differences=max_differences,
    )

    return {
        "ok": match,
        "stage": "compare",
        "backend": backend,
        "pandas_ok": True,
        "pivotal_ok": True,
        "match": match,
        "output_table": output_table,
        "differences": differences,
        "compare_warnings": compare_warnings,
        "pandas_table": _pandas_table_payload(pandas_df, max_rows),
        "pivotal_table": _pandas_table_payload(pivotal_df, max_rows),
        "pandas_result": pandas_run["result"],
        "pivotal_result": pivotal_run["result"],
    }


def compare_pandas_to_pivotal_isolated(
    pandas_source: str,
    pivotal_source: str,
    *,
    output_table: str,
    backend: str = "pandas",
    inputs: Optional[Mapping[str, Any]] = None,
    variables: Optional[Mapping[str, Any]] = None,
    max_rows: int = 20,
    include_generated_code: bool = True,
    timeout_seconds: float = 10,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    check_dtype: bool = False,
    max_differences: int = 20,
) -> dict[str, Any]:
    """Compare pandas and Pivotal pipelines in a fresh Python subprocess."""
    request = {
        "kind": "compare_pandas_to_pivotal",
        "pandas_source": pandas_source,
        "pivotal_source": pivotal_source,
        "output_table": output_table,
        "backend": backend,
        "inputs": dict(inputs or {}),
        "variables": dict(variables or {}),
        "max_rows": max_rows,
        "include_generated_code": include_generated_code,
        "atol": atol,
        "rtol": rtol,
        "check_dtype": check_dtype,
        "max_differences": max_differences,
    }

    with tempfile.TemporaryDirectory(prefix="pivotal-compare-") as tmpdir:
        request_path = os.path.join(tmpdir, "request.pkl")
        result_path = os.path.join(tmpdir, "result.json")
        with open(request_path, "wb") as f:
            pickle.dump(request, f, protocol=pickle.HIGHEST_PROTOCOL)

        cmd = [
            sys.executable,
            "-m",
            "pivotal._runner_worker",
            request_path,
            result_path,
        ]
        try:
            completed = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "stage": "timeout",
                "error_type": "TimeoutExpired",
                "message": f"Pandas/Pivotal comparison timed out after {timeout_seconds} second(s)",
                "line": None,
                "column": None,
                "source_line": None,
                "suggestion": "Reduce the input size or inspect the pipelines for long-running operations.",
                "backend": backend,
                "pandas_ok": None,
                "pivotal_ok": None,
                "match": False,
                "output_table": output_table,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "returncode": None,
            }

        if os.path.exists(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                result.setdefault("worker_stdout", completed.stdout)
                result.setdefault("worker_stderr", completed.stderr)
                result.setdefault("returncode", completed.returncode)
                return result
            except json.JSONDecodeError as exc:
                return {
                    "ok": False,
                    "stage": "worker",
                    "error_type": "JSONDecodeError",
                    "message": f"Could not decode worker result: {exc}",
                    "line": None,
                    "column": None,
                    "source_line": None,
                    "suggestion": None,
                    "backend": backend,
                    "pandas_ok": None,
                    "pivotal_ok": None,
                    "match": False,
                    "output_table": output_table,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "returncode": completed.returncode,
                }

        return {
            "ok": False,
            "stage": "worker",
            "error_type": "WorkerError",
            "message": "Pivotal comparison worker exited without writing a result.",
            "line": None,
            "column": None,
            "source_line": None,
            "suggestion": None,
            "backend": backend,
            "pandas_ok": None,
            "pivotal_ok": None,
            "match": False,
            "output_table": output_table,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        }


__all__ = [
    "run_pivotal",
    "run_pivotal_isolated",
    "compare_pandas_to_pivotal",
    "compare_pandas_to_pivotal_isolated",
]
