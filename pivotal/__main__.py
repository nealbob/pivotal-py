"""
CLI entry point:
  python -m pivotal <file.pivotal>                  — execute a .pivotal file
  python -m pivotal --compile <file.pivotal>         — compile to a .py file
  python -m pivotal --export-py <notebook.ipynb>    — export notebook to a .py file
"""
import sys
import os
import json


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


def notebook_to_python(path):
    """Export a Jupyter notebook to a .py file, converting %%pivotal cells to Python."""
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

        # %%pivotal cell — parse and generate Python
        if source.startswith('%%pivotal'):
            pivotal_src = source[len('%%pivotal'):].strip() + '\n'

            # Mirror magic.py pre-processing: strip `delete <name>` lines and
            # generate del statements, since the parser doesn't handle them
            import re as _re
            del_names = []
            kept_lines = []
            for line in pivotal_src.split('\n'):
                m = _re.match(r'^delete\s+(\w+)\s*$', line)
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
                code_blocks = parser.generate_code(results)
                if del_names:
                    code_blocks.append('\n'.join(f'del {n}' for n in del_names))
                header = f"# [Cell {cell_num}] pivotal"
                sections.append(header + '\n' + '\n\n'.join(code_blocks))
        else:
            # Regular Python cell — include as-is
            sections.append(f"# [Cell {cell_num}]\n{source}")

    py_path = os.path.splitext(path)[0] + '.py'
    header = (
        f"# Generated from: {os.path.basename(path)}\n"
        f"# Do not edit directly — edit the source notebook instead.\n"
        f"import pandas as pd\n"
    )
    with open(py_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n\n' + '\n\n'.join(sections) + '\n')
    print(f"Exported to: {py_path}")


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
