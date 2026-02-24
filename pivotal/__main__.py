"""
CLI entry point:
  python -m pivotal <file.pivotal>           — execute a .pivotal file
  python -m pivotal --compile <file.pivotal> — compile to a .py file in the same directory
"""
import sys
import os


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


def compile_to_python(path):
    """Compile a .pivotal file to a .py file in the same directory."""
    parser, results = _load_and_parse(path)
    code_blocks = parser.generate_code(results)

    py_path = os.path.splitext(path)[0] + '.py'
    header = (
        f"# Generated from: {os.path.basename(path)}\n"
        f"# Do not edit directly — edit the source .pivotal file instead.\n\n"
    )
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(header + "\n\n".join(code_blocks) + "\n")
    print(f"Compiled to: {py_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m pivotal <file.pivotal>", file=sys.stderr)
        print("       python -m pivotal --compile <file.pivotal>", file=sys.stderr)
        sys.exit(1)

    if sys.argv[1] == '--compile':
        if len(sys.argv) < 3:
            print("Usage: python -m pivotal --compile <file.pivotal>", file=sys.stderr)
            sys.exit(1)
        path = sys.argv[2]
        if not os.path.isfile(path):
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        compile_to_python(path)
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
