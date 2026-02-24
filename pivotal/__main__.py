"""
CLI entry point: python -m pivotal <file.pivotal>

Parses and executes a .pivotal file. Output is silent; errors are printed to stderr.
"""
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m pivotal <file.pivotal>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    from .dsl_parser import DSLParser
    import pandas as pd  # noqa: F401  — available in exec namespace

    parser = DSLParser()
    results = parser.parse(source)

    if isinstance(results, dict) and "error" in results:
        print(f"Pivotal parse error: {results['error']}", file=sys.stderr)
        sys.exit(1)

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
