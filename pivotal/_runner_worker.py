"""Subprocess worker for isolated Pivotal execution."""
from __future__ import annotations

import json
import pickle
import sys
import traceback
from typing import Any

from .runner import compare_pandas_to_pivotal, run_pivotal


def _worker_error(exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "stage": "worker",
        "error_type": type(exc).__name__,
        "message": str(exc),
        "line": None,
        "column": None,
        "source_line": None,
        "suggestion": None,
        "traceback": traceback.format_exc(),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("Usage: python -m pivotal._runner_worker <request.pkl> <result.json>", file=sys.stderr)
        return 2

    request_path, result_path = args
    try:
        with open(request_path, "rb") as f:
            request = pickle.load(f)

        kind = request.get("kind", "run_pivotal")
        if kind == "run_pivotal":
            result = run_pivotal(
                request["source"],
                backend=request.get("backend", "pandas"),
                inputs=request.get("inputs"),
                variables=request.get("variables"),
                return_tables=request.get("return_tables"),
                max_rows=request.get("max_rows", 20),
                include_generated_code=request.get("include_generated_code", True),
            )
        elif kind == "compare_pandas_to_pivotal":
            result = compare_pandas_to_pivotal(
                request["pandas_source"],
                request["pivotal_source"],
                output_table=request["output_table"],
                backend=request.get("backend", "pandas"),
                inputs=request.get("inputs"),
                variables=request.get("variables"),
                max_rows=request.get("max_rows", 20),
                include_generated_code=request.get("include_generated_code", True),
                atol=request.get("atol", 1e-9),
                rtol=request.get("rtol", 1e-9),
                check_dtype=request.get("check_dtype", False),
                max_differences=request.get("max_differences", 20),
            )
        else:
            raise ValueError(f"Unknown worker request kind: {kind}")
    except BaseException as exc:  # noqa: BLE001 - worker boundary
        result = _worker_error(exc)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
