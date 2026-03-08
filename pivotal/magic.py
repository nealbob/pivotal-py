try:
    from IPython.core.magic import Magics, magics_class, cell_magic, line_magic
    from IPython.display import display
    import pandas as pd
except ImportError:
    # Fallback if IPython is not installed
    class Magics: pass
    def magics_class(cls): return cls
    def cell_magic(func): return func
    def line_magic(func): return func
    display = print

from .dsl_parser import DSLParser


# ---------------------------------------------------------------------------
# Object Viewer comm helper
# ---------------------------------------------------------------------------

class _PivotalViewer:
    """Sends DataFrames and chart figures to the JupyterLab Object Viewer panel."""

    MAX_ROWS = 10_000

    def __init__(self, shell):
        self._shell = shell
        self._comm = None
        self._last_sent: dict = {}   # name → df, for re-send on row-limit change

    def _ensure_comm(self):
        if self._comm is not None:
            return
        try:
            from ipykernel.comm import Comm
            self._comm = Comm(target_name='pivotal_viewer')
            self._comm.on_msg(self._on_msg)
            self._comm.open()
        except Exception:
            pass

    def _on_msg(self, msg):
        """Handle re-request from panel (e.g. user changed row limit)."""
        data = msg['content']['data']
        if data.get('type') == 'request' and data.get('name') in self._last_sent:
            limit = int(data.get('limit', self.MAX_ROWS))
            self.send_dataframe(data['name'], self._last_sent[data['name']], limit=limit)

    def send_dataframe(self, name: str, df, limit: int = None):
        self._ensure_comm()
        if self._comm is None:
            return
        limit = limit or self.MAX_ROWS
        self._last_sent[name] = df
        truncated = len(df) > limit
        payload = df.head(limit)
        try:
            import json as _json
            # to_json() uses pandas' C-level serializer: fast and handles NaN→null natively.
            # orient='split' gives {columns, index, data} — compact (no repeated keys per row).
            split = _json.loads(payload.to_json(orient='split', default_handler=str))
            self._comm.send({
                'type': 'dataframe',
                'name': name,
                'columns': split['columns'],
                'data': split['data'],       # list of rows (each row is a list of values)
                'dtypes': {str(c): str(t) for c, t in payload.dtypes.items()},
                'shape': list(df.shape),
                'truncated': truncated,
            })
        except Exception:
            pass

    def send_chart(self, name: str, fig):
        self._ensure_comm()
        if self._comm is None:
            return
        import io
        import base64
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            self._comm.send({
                'type': 'chart',
                'name': name,
                'data': base64.b64encode(buf.getvalue()).decode(),
            })
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Walk AST results and send objects to viewer in execution order
# ---------------------------------------------------------------------------

def _send_results_to_viewer(viewer: _PivotalViewer, results: list, ns: dict):
    """Send the final state of each named object to the viewer.

    The viewer keeps one slot per name (latest wins), so we collect the unique
    table names touched in this cell and send each once — reflecting the state
    after all operations have run. Charts are sent last so they are visible at
    cell completion.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return

    # Collect touched table names and whether there was a plot, preserving order
    seen_tables: dict[str, bool] = {}   # name → True (ordered, dedup)
    has_plot = False
    last_plot_node = None

    for node in results:
        if not isinstance(node, dict):
            continue
        node_type = node.get('type')
        if node_type == 'plot':
            has_plot = True
            last_plot_node = node
        else:
            table_name = node.get('table_name')
            if table_name:
                seen_tables[table_name] = True

    # Send each table's current (post-execution) state
    for name in seen_tables:
        obj = ns.get(name)
        if isinstance(obj, pd.DataFrame):
            viewer.send_dataframe(name, obj)

    # Send chart last (so it's visible at cell completion)
    if has_plot and last_plot_node is not None:
        fig = plt.gcf()
        chart_name = last_plot_node.get('table_name', 'chart')
        viewer.send_chart(chart_name, fig)


# ---------------------------------------------------------------------------
# Main magic class
# ---------------------------------------------------------------------------

@magics_class
class PivotalMagics(Magics):
    def __init__(self, shell):
        super().__init__(shell)
        self.parser = DSLParser()
        self.auto_pivotal = False
        self._viewer = _PivotalViewer(shell)

        # Register input transformer
        if hasattr(shell, 'input_transformers_cleanup'):
            shell.input_transformers_cleanup.append(self.input_transform)

        # Register completer (not supported in all environments, e.g. VS Code notebooks)
        try:
            self.shell.set_hook('complete_command', self.pivotal_completer, re_key='%%pivotal')
            self.shell.set_hook('complete_command', self.pivotal_completer, str_key='%pivotal_auto')
        except Exception:
            pass

        # Try to register a custom completer for auto mode
        try:
            from IPython.core.completer import Completer
            # We can't easily inject into the main completer logic without being a proper extension
            # But we can try to add a hook that runs for all cells if auto_pivotal is on
            pass
        except ImportError:
            pass

    def pivotal_completer(self, event):
        """Custom completer for Pivotal DSL"""
        # Basic keywords list
        keywords = [
            'load', 'df', 'filter', 'select', 'sort',
            'group by', 'merge', 'pivot', 'plot', 'python', 'apply',
            'drop', 'dropna', 'fillna', 'distinct', 'concat', 'rename',
            'mean', 'min', 'max', 'sum', 'count', 'avg', 'median', 'std',
            'asc', 'desc', 'left', 'right', 'inner', 'outer',
            'from', 'as', 'on', 'rows', 'cols',
            'between', 'contains', 'not contains', 'startswith', 'endswith',
        ]

        # Add table names and columns from parser state
        if hasattr(self.parser, 'table_info'):
            for table_name, info in self.parser.table_info.items():
                keywords.append(table_name)
                if 'columns' in info:
                    # Flatten columns if they are lists (MultiIndex)
                    for col in info['columns']:
                        if isinstance(col, list):
                            keywords.extend([str(c) for c in col])
                        else:
                            keywords.append(str(col))

        # Filter matches
        return [k for k in keywords if k.startswith(event.symbol)]

    @line_magic
    def pivotal_auto(self, line):
        """Toggle automatic Pivotal parsing for all cells."""
        self.auto_pivotal = not self.auto_pivotal
        print(f"Automatic Pivotal parsing is now {'ON' if self.auto_pivotal else 'OFF'}")

    def input_transform(self, lines):
        """
        Transform input if auto_pivotal is enabled and code parses as Pivotal.
        """
        if not self.auto_pivotal:
            return lines

        # Join lines to check content
        code = ''.join(lines)

        # Ignore magics
        if code.strip().startswith('%'):
            return lines

        # Try to parse
        results = self.parser.parse(code)

        if isinstance(results, dict) and 'error' in results:
            # Parse failed, assume it's Python
            return lines

        # If successful, generate code
        try:
            python_code_list = self.parser.generate_code(results)

            # Process code to add display logic
            final_lines = []
            for block in python_code_list:
                # Split block into lines
                block_lines = block.splitlines()
                for line in block_lines:
                    if '__table_name__ =' in line:
                        # Extract table name
                        # Line format: __table_name__ = 'df'
                        try:
                            parts = line.split('=')
                            t_name = parts[1].strip().strip("'").strip('"')
                            # Add display logic
                            # We use display() if available, else print
                            display_code = f"try:\n    from IPython.display import display\n    print(f\"Table '{t_name}' shape: {{{t_name}.shape}}\")\n    display({t_name}.head())\nexcept:\n    print({t_name}.head())"
                            final_lines.extend(display_code.split('\n'))
                        except:
                            pass
                    elif '#__pivotal__' in line:
                        continue
                    else:
                        final_lines.append(line)
                final_lines.append("") # Add newline between blocks

            return [l + '\n' for l in final_lines]
        except Exception:
            return lines

    @staticmethod
    def _clean_code(code_block):
        """Strip internal markers and boilerplate for display, keeping meaningful pandas code."""
        import re
        lines = code_block.split('\n')
        result = []
        skip_pivot = False
        # Stack entries: (mode, base_indent, replacements)
        # mode='skip'   → drop all lines in block
        # mode='dedent' → emit body lines with one indent level removed
        #                  replacements: dict of str→str substitutions in body lines
        block_stack = []

        for raw_line in lines:
            lstripped = raw_line.lstrip()
            indent = len(raw_line) - len(lstripped)
            stripped = lstripped.rstrip()

            # #__pivotal__ marker blocks
            if '#__pivotal__' in raw_line:
                skip_pivot = not skip_pivot
                continue
            if skip_pivot:
                continue

            # Pop finished blocks when a non-empty line returns to base indent
            if stripped:
                while block_stack and indent <= block_stack[-1][1]:
                    block_stack.pop()

            # Apply current block mode
            if block_stack:
                mode, base_indent, replacements = block_stack[-1]
                if mode == 'skip':
                    continue
                elif mode == 'dedent':
                    body = raw_line[base_indent + 4:]
                    for old, new in replacements.items():
                        body = body.replace(old, new)
                    result.append(body)
                    continue

            # === Line-level filter rules ===

            # Known guard patterns
            if 'not in locals() and' in stripped and 'raise NameError' in stripped:
                continue
            if stripped.startswith('if not isinstance(') and 'raise TypeError' in stripped:
                continue

            # Anything referencing _pivotal_charts
            if '_pivotal_charts' in stripped:
                continue

            # import X as _Y
            if re.match(r'import\s+\S+\s+as\s+_', stripped):
                continue

            # with _X...: → show body dedented (e.g. sqlite3 connection wrapper)
            if re.match(r'with\s+_[a-zA-Z]', stripped):
                # Extract connection path to replace _conn in body lines
                replacements = {}
                m = re.search(r'connect\(([^)]+)\)', stripped)
                alias = re.search(r'\bas\s+(\w+)', stripped)
                if m and alias:
                    replacements[alias.group(1)] = m.group(1)
                block_stack.append(('dedent', indent, replacements))
                continue

            # if/elif _identifier: or if/elif '_identifier'... → skip block
            if re.match(r'(if|elif)\s+(_[a-zA-Z]|[\'"]_)', stripped):
                if stripped.endswith(':'):
                    block_stack.append(('skip', indent, {}))
                continue

            # for _identifier...: → skip block (e.g. faceted subplot loops)
            if re.match(r'for\s+_[a-zA-Z]', stripped):
                block_stack.append(('skip', indent, {}))
                continue

            # Lines whose first token starts with _ (internal variables)
            if re.match(r'_[a-zA-Z]', stripped):
                # If it's `_var = obj.method(...)`, show just the RHS
                m = re.match(r'_\w+\s*=\s*(.+)', stripped)
                if m:
                    rhs = m.group(1).strip()
                    if re.match(r'[a-zA-Z]\w*[.\[]', rhs):
                        result.append(' ' * indent + rhs)
                continue

            # Non-_ LHS assigned from a _ RHS (e.g. chart_var = _ax.get_figure())
            if re.match(r'\w+\s*=\s*_[a-zA-Z]', stripped):
                continue

            result.append(raw_line)

        return '\n'.join(result).strip()

    @cell_magic
    def pivotal(self, line, cell):
        """
        Execute Pivotal DSL code.
        Usage:
            %%pivotal
            load df "data.csv"
            filter df
                col > 5
        """
        # Ensure pandas is imported in the user namespace
        self.shell.push({'pd': pd})

        if not cell.endswith('\n'):
            cell += '\n'

        results = self.parser.parse(cell)

        if isinstance(results, dict) and 'error' in results:
            print(f"Pivotal Parse Error: {results['error']}")
            return

        python_code_list = self.parser.generate_code(results)

        combined = '\n\n'.join(python_code_list)
        result = self.shell.run_cell(combined)
        if not result.error_in_exec:
            cleaned = self._clean_code(combined)
            if cleaned:
                print(cleaned)

            # Send objects to the viewer panel (best-effort; silently skipped if comm not open)
            try:
                _send_results_to_viewer(self._viewer, results, self.shell.user_ns)
            except Exception:
                pass

        # Update autocomplete file so the next cell can offer column/table names
        self.parser.update_autocomplete_info(self.shell.user_ns)

def load_ipython_extension(ipython):
    ipython.register_magics(PivotalMagics)
