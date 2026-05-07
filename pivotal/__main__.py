"""
CLI entry point:
  python -m pivotal <file.pivotal>                     — execute a .pivotal file
  python -m pivotal --compile <file.pivotal>            — compile to a .py file
  python -m pivotal --export-py <notebook.ipynb>       — export notebook to a .py file
  python -m pivotal --export-pivotal <notebook.ipynb>  — export notebook to a .pivotal file
"""
import sys
import os
import json
import re


def _usage_verify():
    return (
        "Usage: python -m pivotal --verify [--json] [--backend <backend>] "
        "[--input <name=path>] [--return <table>] [--max-rows <n>] "
        "[--timeout <seconds>] <file.pivotal>"
    )


def _usage_compare():
    return (
        "Usage: python -m pivotal --compare --pandas <original.py> "
        "--pivotal <converted.pivotal> --output <table> [--json] "
        "[--backend <backend>] [--input <name=path>] [--max-rows <n>] "
        "[--timeout <seconds>] [--atol <n>] [--rtol <n>] "
        "[--max-differences <n>] [--check-dtype]"
    )


def _parse_verify_args(args):
    options = {
        "backend": "pandas",
        "inputs": [],
        "return_tables": [],
        "max_rows": 20,
        "timeout_seconds": 10.0,
        "json": False,
        "path": None,
    }

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--json":
            options["json"] = True
            i += 1
            continue
        if arg in ("--backend", "--input", "--return", "--max-rows", "--timeout"):
            if i + 1 >= len(args):
                raise ValueError(f"{arg} requires a value")
            value = args[i + 1]
            if arg == "--backend":
                options["backend"] = value
            elif arg == "--input":
                options["inputs"].append(value)
            elif arg == "--return":
                options["return_tables"].append(value)
            elif arg == "--max-rows":
                try:
                    options["max_rows"] = int(value)
                except ValueError as exc:
                    raise ValueError("--max-rows must be an integer") from exc
            elif arg == "--timeout":
                try:
                    options["timeout_seconds"] = float(value)
                except ValueError as exc:
                    raise ValueError("--timeout must be a number") from exc
            i += 2
            continue
        if arg.startswith("--"):
            raise ValueError(f"Unknown option: {arg}")
        if options["path"] is not None:
            raise ValueError(f"Unexpected extra argument: {arg}")
        options["path"] = arg
        i += 1

    if options["path"] is None:
        raise ValueError("Missing .pivotal file")
    return options


def _parse_compare_args(args):
    options = {
        "backend": "pandas",
        "inputs": [],
        "max_rows": 20,
        "timeout_seconds": 10.0,
        "json": False,
        "pandas_path": None,
        "pivotal_path": None,
        "output_table": None,
        "atol": 1e-9,
        "rtol": 1e-9,
        "check_dtype": False,
        "max_differences": 20,
    }

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--json":
            options["json"] = True
            i += 1
            continue
        if arg == "--check-dtype":
            options["check_dtype"] = True
            i += 1
            continue
        if arg in (
            "--backend",
            "--input",
            "--max-rows",
            "--timeout",
            "--pandas",
            "--pivotal",
            "--output",
            "--atol",
            "--rtol",
            "--max-differences",
        ):
            if i + 1 >= len(args):
                raise ValueError(f"{arg} requires a value")
            value = args[i + 1]
            if arg == "--backend":
                options["backend"] = value
            elif arg == "--input":
                options["inputs"].append(value)
            elif arg == "--max-rows":
                try:
                    options["max_rows"] = int(value)
                except ValueError as exc:
                    raise ValueError("--max-rows must be an integer") from exc
            elif arg == "--timeout":
                try:
                    options["timeout_seconds"] = float(value)
                except ValueError as exc:
                    raise ValueError("--timeout must be a number") from exc
            elif arg == "--pandas":
                options["pandas_path"] = value
            elif arg == "--pivotal":
                options["pivotal_path"] = value
            elif arg == "--output":
                options["output_table"] = value
            elif arg == "--atol":
                try:
                    options["atol"] = float(value)
                except ValueError as exc:
                    raise ValueError("--atol must be a number") from exc
            elif arg == "--rtol":
                try:
                    options["rtol"] = float(value)
                except ValueError as exc:
                    raise ValueError("--rtol must be a number") from exc
            elif arg == "--max-differences":
                try:
                    options["max_differences"] = int(value)
                except ValueError as exc:
                    raise ValueError("--max-differences must be an integer") from exc
            i += 2
            continue
        raise ValueError(f"Unknown option: {arg}")

    if options["pandas_path"] is None:
        raise ValueError("Missing --pandas <original.py>")
    if options["pivotal_path"] is None:
        raise ValueError("Missing --pivotal <converted.pivotal>")
    if options["output_table"] is None:
        raise ValueError("Missing --output <table>")
    return options


def _load_verify_inputs(input_specs):
    import pandas as pd

    inputs = {}
    for spec in input_specs:
        if "=" not in spec:
            raise ValueError("--input must be in the form <name=path>")
        name, path = spec.split("=", 1)
        if not name.isidentifier():
            raise ValueError(f"Invalid input table name: {name}")
        if not os.path.isfile(path):
            raise ValueError(f"Input file not found: {path}")

        lower = path.lower()
        if lower.endswith(".csv"):
            inputs[name] = pd.read_csv(path)
        elif lower.endswith(".parquet"):
            inputs[name] = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported input file type for {path}; use CSV or Parquet")
    return inputs


def verify_file(path, *, backend="pandas", inputs=None, return_tables=None,
                max_rows=20, timeout_seconds=10.0):
    """Verify a .pivotal file and return a structured result dictionary."""
    from .runner import run_pivotal_isolated

    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    return run_pivotal_isolated(
        source,
        backend=backend,
        inputs=inputs,
        return_tables=return_tables,
        max_rows=max_rows,
        timeout_seconds=timeout_seconds,
    )


def compare_files(pandas_path, pivotal_path, *, output_table, backend="pandas",
                  inputs=None, max_rows=20, timeout_seconds=10.0,
                  atol=1e-9, rtol=1e-9, check_dtype=False,
                  max_differences=20):
    """Compare a pandas script and a Pivotal file using structured results."""
    from .runner import compare_pandas_to_pivotal_isolated

    with open(pandas_path, "r", encoding="utf-8") as f:
        pandas_source = f.read()
    with open(pivotal_path, "r", encoding="utf-8") as f:
        pivotal_source = f.read()

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


def _load_and_parse(path):
    from .dsl_parser import DSLParser
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    if not source.endswith('\n'):
        source += '\n'
    parser = DSLParser()
    results = parser.parse(source)
    if isinstance(results, dict) and "error" in results:
        print(f"Pivotal parse error: {results['error']}", file=sys.stderr)
        sys.exit(1)
    return parser, results


def _strip_pivotal_internals(code_blocks):
    """Remove #__pivotal__ marker lines and __table_name__ assignments from compiled output."""
    cleaned = []
    for block in code_blocks:
        lines = block.splitlines()
        kept = [l for l in lines
                if '#__pivotal__' not in l and not l.strip().startswith('__table_name__')]
        cleaned.append('\n'.join(kept))
    return cleaned


def compile_to_python(path, backend='pandas'):
    """Compile a .pivotal file to a .py (or .sql) file in the same directory."""
    parser, results = _load_and_parse(path)
    code_blocks = _strip_pivotal_internals(parser.generate_code(results, backend=backend))

    ext = '.sql' if backend == 'sql' else '.py'
    out_path = os.path.splitext(path)[0] + ext
    header = f"# Generated by Pivotal from: {os.path.basename(path)} (backend: {backend})\n\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n\n".join(code_blocks) + "\n")
    print(f"Compiled to: {out_path}")


def notebook_to_python(path, backend='pandas'):
    """Export a Jupyter notebook to a .py file, converting %%pivotal cells to Python.

    Args:
        path:    Absolute path to the .ipynb file.
        backend: Code generation backend — 'pandas' (default), 'duckdb', or 'sql'.
                 'sql' cells are exported as a SQL string printed at the end of each
                 pivotal block rather than as executable Python.
    """
    from .dsl_parser import DSLParser

    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    parser = DSLParser()
    sections = []
    cell_num = 0

    for cell in nb.get('cells', []):
        if cell['cell_type'] != 'code':
            continue

        cell_num += 1
        source = ''.join(cell['source']).strip()
        if not source:
            continue

        # Pivotal GUI cells — skip entirely (they just launch interactive widgets)
        if re.match(r'^(import pivotal\s*\n\s*)?pivotal\.\w+_gui\(', source):
            continue

        # %%pivotal cell — parse and generate code for the chosen backend
        if source.startswith('%%pivotal'):
            # Strip the magic line (may contain per-cell args like backend=duckdb —
            # those are ignored here; the export dialog backend choice takes precedence)
            first_nl = source.find('\n')
            pivotal_src = (source[first_nl + 1:] if first_nl != -1 else '').strip() + '\n'

            # Mirror magic.py pre-processing: strip `delete <name>` lines and
            # generate del statements, since the parser doesn't handle them
            del_names = []
            kept_lines = []
            for line in pivotal_src.split('\n'):
                m = re.match(r'^delete\s+(\w+)\s*$', line)
                if m:
                    del_names.append(m.group(1))
                else:
                    kept_lines.append(line)
            pivotal_src = '\n'.join(kept_lines)

            results = parser.parse(pivotal_src)
            if isinstance(results, dict) and 'error' in results:
                print(f"Warning: parse error in cell {cell_num}: {results['error']}", file=sys.stderr)
                sections.append(f"# [Cell {cell_num}] Pivotal parse error — original source:\n"
                                + '\n'.join('# ' + l for l in pivotal_src.splitlines()))
            else:
                try:
                    code_blocks = parser.generate_code(results, backend=backend)
                except Exception as exc:
                    print(f"Warning: codegen error in cell {cell_num} ({backend}): {exc}", file=sys.stderr)
                    code_blocks = [f"# [Cell {cell_num}] codegen error ({backend}): {exc}\n"
                                   + '\n'.join('# ' + l for l in pivotal_src.splitlines())]
                if del_names and backend != 'sql':
                    code_blocks.append('\n'.join(f'del {n}' for n in del_names))
                cell_header = f"# [Cell {cell_num}] pivotal → {backend}"
                sections.append(cell_header + '\n' + '\n\n'.join(code_blocks))
        else:
            # Regular Python cell — include as-is (skip for sql-only export)
            if backend != 'sql':
                sections.append(f"# [Cell {cell_num}]\n{source}")

    ext = '.sql' if backend == 'sql' else '.py'
    py_path = os.path.splitext(path)[0] + ext

    if backend == 'duckdb':
        imports = (
            "import duckdb\n"
            "import pandas as pd\n"
        )
    elif backend == 'sql':
        imports = (
            "-- SQL export — paste each query into your SQL tool of choice.\n"
            "-- Python-only operations (plot, apply, etc.) are omitted.\n"
        )
    else:
        imports = "import pandas as pd\n"

    comment = '--' if backend == 'sql' else '#'
    file_header = (
        f"{comment} Generated by Pivotal from: {os.path.basename(path)}\n"
        + imports
    )
    with open(py_path, 'w', encoding='utf-8') as f:
        f.write(file_header + '\n\n' + '\n\n'.join(sections) + '\n')
    print(f"Exported ({backend}): {py_path}")


def notebook_to_pivotal(path):
    """Export a Jupyter notebook to a .pivotal file.

    %%pivotal cells are written as-is (DSL source).
    Regular Python cells are wrapped in python...end blocks.
    GUI cells are skipped.
    """
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    sections = []
    cell_num = 0

    for cell in nb.get('cells', []):
        if cell['cell_type'] != 'code':
            continue

        cell_num += 1
        source = ''.join(cell['source']).strip()
        if not source:
            continue

        # GUI cells — skip entirely
        if re.match(r'^(import pivotal\s*\n\s*)?pivotal\.\w+_gui\(', source):
            continue

        if source.startswith('%%pivotal'):
            # Strip the magic line and write DSL source as-is
            first_nl = source.find('\n')
            pivotal_src = (source[first_nl + 1:] if first_nl != -1 else '').strip()
            sections.append(f"# [Cell {cell_num}]\n{pivotal_src}")
        else:
            # Wrap Python cell in python...end block
            sections.append(f"# [Cell {cell_num}]\npython\n{source}\nend")

    pivotal_path = os.path.splitext(path)[0] + '.pivotal'
    file_header = (
        f"# Generated by Pivotal from: {os.path.basename(path)}\n"
    )
    with open(pivotal_path, 'w', encoding='utf-8') as f:
        f.write(file_header + '\n' + '\n\n'.join(sections) + '\n')
    print(f"Exported (pivotal): {pivotal_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m pivotal <file.pivotal>", file=sys.stderr)
        print("       python -m pivotal --verify [--json] <file.pivotal>", file=sys.stderr)
        print("       python -m pivotal --compare --pandas <original.py> --pivotal <converted.pivotal> --output <table>", file=sys.stderr)
        print("       python -m pivotal --compile <file.pivotal>", file=sys.stderr)
        sys.exit(1)

    if sys.argv[1] == '--compare':
        try:
            options = _parse_compare_args(sys.argv[2:])
            if not os.path.isfile(options["pandas_path"]):
                raise ValueError(f"file not found: {options['pandas_path']}")
            if not os.path.isfile(options["pivotal_path"]):
                raise ValueError(f"file not found: {options['pivotal_path']}")
            inputs = _load_verify_inputs(options["inputs"])
            result = compare_files(
                options["pandas_path"],
                options["pivotal_path"],
                output_table=options["output_table"],
                backend=options["backend"],
                inputs=inputs,
                max_rows=options["max_rows"],
                timeout_seconds=options["timeout_seconds"],
                atol=options["atol"],
                rtol=options["rtol"],
                check_dtype=options["check_dtype"],
                max_differences=options["max_differences"],
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            print(_usage_compare(), file=sys.stderr)
            sys.exit(2)

        if options["json"]:
            print(json.dumps(result, indent=2))
        elif result.get("match"):
            print(f"Pivotal comparison passed for '{options['output_table']}'")
            table = result.get("pivotal_table", {})
            shape = table.get("shape", ["?", "?"])
            print(f"  {shape[0]} row(s), {shape[1]} column(s)")
        else:
            print(f"Pivotal comparison failed at {result.get('stage')}: {result.get('message', 'outputs differ')}", file=sys.stderr)
            for diff in result.get("differences", [])[:5]:
                print(f"  {diff}", file=sys.stderr)
        sys.exit(0 if result.get("match") else 1)

    if sys.argv[1] == '--verify':
        try:
            options = _parse_verify_args(sys.argv[2:])
            path = options["path"]
            if not os.path.isfile(path):
                raise ValueError(f"file not found: {path}")
            inputs = _load_verify_inputs(options["inputs"])
            result = verify_file(
                path,
                backend=options["backend"],
                inputs=inputs,
                return_tables=options["return_tables"] or None,
                max_rows=options["max_rows"],
                timeout_seconds=options["timeout_seconds"],
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            print(_usage_verify(), file=sys.stderr)
            sys.exit(2)

        if options["json"]:
            print(json.dumps(result, indent=2))
        elif result.get("ok"):
            print(f"Pivotal verification passed ({result.get('backend', options['backend'])})")
            for name, table in result.get("tables", {}).items():
                shape = table.get("shape", ["?", "?"])
                print(f"  {name}: {shape[0]} row(s), {shape[1]} column(s)")
        else:
            print(f"Pivotal verification failed at {result.get('stage')}: {result.get('message')}", file=sys.stderr)
        sys.exit(0 if result.get("ok") else 1)

    if sys.argv[1] == '--compile':
        args = sys.argv[2:]
        backend = 'pandas'
        if '--backend' in args:
            idx = args.index('--backend')
            if idx + 1 >= len(args):
                print("Usage: python -m pivotal --compile [--backend <backend>] <file.pivotal>", file=sys.stderr)
                sys.exit(1)
            backend = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
        if not args:
            print("Usage: python -m pivotal --compile [--backend <backend>] <file.pivotal>", file=sys.stderr)
            sys.exit(1)
        path = args[0]
        if not os.path.isfile(path):
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        compile_to_python(path, backend=backend)
        return

    if sys.argv[1] == '--export-py':
        if len(sys.argv) < 3:
            print("Usage: python -m pivotal --export-py <notebook.ipynb>", file=sys.stderr)
            sys.exit(1)
        path = sys.argv[2]
        if not os.path.isfile(path):
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        notebook_to_python(path)
        return

    if sys.argv[1] == '--export-pivotal':
        if len(sys.argv) < 3:
            print("Usage: python -m pivotal --export-pivotal <notebook.ipynb>", file=sys.stderr)
            sys.exit(1)
        path = sys.argv[2]
        if not os.path.isfile(path):
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        notebook_to_pivotal(path)
        return

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    import pandas as pd  # noqa: F401  — available in exec namespace
    parser, results = _load_and_parse(path)
    code_blocks = parser.generate_code(results)

    namespace: dict = {"pd": pd}
    for block in code_blocks:
        try:
            exec(block, namespace)  # noqa: S102
        except Exception as exc:
            print(f"Pivotal execution error: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
