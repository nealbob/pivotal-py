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
        self._last_sent: dict = {}          # name → df, for re-send on row-limit change
        self._last_viewer_settings: dict = {}  # name → {viewer_font, viewer_num_format}

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
            vs = self._last_viewer_settings.get(data['name'], {})
            self.send_dataframe(data['name'], self._last_sent[data['name']], limit=limit,
                                viewer_font=vs.get('viewer_font'),
                                viewer_num_format=vs.get('viewer_num_format'))

    def send_dataframe(self, name: str, df, limit: int = None,
                       viewer_font: float = None, viewer_num_format: int = None):
        self._ensure_comm()
        if self._comm is None:
            return
        limit = limit or self.MAX_ROWS
        self._last_sent[name] = df
        if viewer_font is not None or viewer_num_format is not None:
            self._last_viewer_settings[name] = {
                'viewer_font': viewer_font,
                'viewer_num_format': viewer_num_format,
            }
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
                'viewer_font': viewer_font,
                'viewer_num_format': viewer_num_format,
            })
        except Exception:
            pass

    def send_chart(self, name: str, fig, canvas_meta: dict = None):
        """Render fig to PNG and send to the viewer."""
        self._ensure_comm()
        if self._comm is None:
            return
        import io, base64
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            msg = {
                'type': 'chart',
                'name': name,
                'data': base64.b64encode(buf.getvalue()).decode(),
            }
            if canvas_meta:
                msg['canvas'] = canvas_meta
            self._comm.send(msg)
        except Exception:
            pass

    def send_table(self, name: str, html: str, canvas: str = 'none'):
        """Send a rendered GT table to the viewer."""
        self._ensure_comm()
        if self._comm is None:
            return
        msg = {
            'type': 'gt_table',
            'name': name,
            'html': html,
        }
        if canvas in _PAPER_SIZES_MM:
            page_w, page_h = _PAPER_SIZES_MM[canvas]
            msg['canvas'] = {
                'page_width_mm': page_w,
                'page_height_mm': page_h,
                'margin_mm': 25.4,
                'label': _CANVAS_LABELS.get(canvas, canvas.upper()),
            }
        try:
            self._comm.send(msg)
        except Exception as e:
            print(f"[Pivotal] send_table error: {e}")


# ---------------------------------------------------------------------------
# Walk AST results and send objects to viewer in execution order
# ---------------------------------------------------------------------------

_PAPER_SIZES_MM: dict = {
    'a4':           (210.0, 297.0),
    'a4_landscape': (297.0, 210.0),
    'a3':           (297.0, 420.0),
    'a3_landscape': (420.0, 297.0),
    'letter':       (215.9, 279.4),
    'slide':        (338.7, 190.5),  # 16:9 widescreen (PPT/Beamer default)
}

_CANVAS_LABELS: dict = {
    'a4':           'A4',
    'a4_landscape': 'A4 Landscape',
    'a3':           'A3',
    'a3_landscape': 'A3 Landscape',
    'letter':       'Letter',
    'slide':        'Slide (16:9)',
}


def _build_canvas_meta(fig, settings: dict, canvas_override: str = None) -> dict | None:
    """Compute page-layout metadata from settings and the figure's aspect ratio.

    canvas_override: per-plot canvas value that takes precedence over the global setting.
    """
    canvas = canvas_override or settings.get('canvas', 'none')
    if canvas not in _PAPER_SIZES_MM:
        return None
    page_w_mm, page_h_mm = _PAPER_SIZES_MM[canvas]
    margin_mm = float(settings.get('margins', 25.4))
    usable_w = page_w_mm - 2 * margin_mm
    fraction = 0.5 if settings.get('chart_width') == 'half' else 1.0
    chart_w_mm = usable_w * fraction
    try:
        w_in, h_in = fig.get_size_inches()
        chart_h_mm = chart_w_mm * (h_in / w_in) if w_in else chart_w_mm
    except Exception:
        chart_h_mm = chart_w_mm * 0.75
    return {
        'page_width_mm':  page_w_mm,
        'page_height_mm': page_h_mm,
        'margin_mm':      margin_mm,
        'chart_width_mm': chart_w_mm,
        'chart_height_mm': chart_h_mm,
        'label': _CANVAS_LABELS.get(canvas, canvas.upper()),
    }


def _send_results_to_viewer(viewer: _PivotalViewer, results: list, ns: dict, settings: dict = None):
    """Send the final state of each named object to the viewer.

    The generated code stores each chart figure in the user namespace under
    the chart's name (e.g. `myplot = ax.get_figure()`), so we can retrieve
    it by name after run_cell() completes — even if the inline backend has
    already rendered it, the Figure object is still valid for savefig().
    """
    try:
        import pandas as pd
        import matplotlib.figure as mfig
    except ImportError:
        return

    seen_tables: dict = {}
    plot_nodes: list = []
    gt_table_nodes: list = []

    for node in results:
        if not isinstance(node, dict):
            continue
        if node.get('type') == 'plot':
            plot_nodes.append(node)
        elif node.get('type') == 'gt_table':
            gt_table_nodes.append(node)
        else:
            name = node.get('table_name')
            if name:
                seen_tables[name] = True

    # Send each table's post-execution state
    for name in seen_tables:
        obj = ns.get(name)
        if isinstance(obj, pd.DataFrame):
            viewer.send_dataframe(
                name, obj,
                viewer_font=(settings or {}).get('viewer_font'),
                viewer_num_format=(settings or {}).get('viewer_num_format'),
            )

    # Send charts: look up the figure stored in the namespace by chart name
    for node in plot_nodes:
        chart_name = node.get('name') or node.get('table_name') or 'chart'
        fig = ns.get(chart_name)
        if isinstance(fig, mfig.Figure):
            canvas_meta = _build_canvas_meta(fig, settings or {},
                                             canvas_override=node.get('canvas'))
            viewer.send_chart(chart_name, fig, canvas_meta=canvas_meta)

    # Send GT tables — canvas falls back to the global setting when not specified per-table
    global_canvas = (settings or {}).get('canvas', 'none')
    for node in gt_table_nodes:
        tbl_name = node.get('name')
        if tbl_name is None:
            continue
        entry = ns.get('_pivotal_gt_tables', {}).get(tbl_name, {})
        viewer_html = entry.get('viewer_html') or entry.get('html')
        if viewer_html:
            canvas = entry.get('canvas', 'none')
            if canvas == 'none':
                canvas = global_canvas
            viewer.send_table(tbl_name, viewer_html, canvas)


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
        self.settings = dict(self.DEFAULT_SETTINGS)

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

    # -------------------------------------------------------------------------
    # Settings
    # -------------------------------------------------------------------------

    # Defaults — can be changed via %pivotal_set or overridden per-cell
    DEFAULT_SETTINGS = {
        'output_type': 'viewer',   # viewer | inline | both
        'output_code': False,       # print generated Python code inline
        'canvas': 'none',           # none | a4 | a4_landscape | a3 | a3_landscape | letter | slide
        'margins': 25.4,            # page margin in mm (all sides) — 25.4 mm = 2.54 cm (MS Word default)
        'chart_width': 'full',      # full | half  (fraction of usable page width)
        'viewer_font': 0.8,         # em units for DataFrame viewer font size
        'viewer_num_format': 5,     # significant digits for float columns (0 = no formatting)
    }

    def _parse_line_args(self, line: str) -> dict:
        """Parse 'key=value ...' from the magic line into a settings dict."""
        overrides = {}
        for part in (line or '').split():
            k, _, v = part.partition('=')
            k, v = k.strip(), v.strip().lower()
            if k == 'output_type' and v in ('viewer', 'inline', 'both'):
                overrides['output_type'] = v
            elif k == 'output_code':
                overrides['output_code'] = v in ('true', '1', 'yes')
            elif k == 'canvas' and v in ('none', *_PAPER_SIZES_MM):
                overrides['canvas'] = v
            elif k == 'margins':
                try:
                    overrides['margins'] = float(v)
                except ValueError:
                    pass
            elif k == 'chart_width' and v in ('full', 'half'):
                overrides['chart_width'] = v
            elif k == 'viewer_font':
                try:
                    overrides['viewer_font'] = float(v)
                except ValueError:
                    pass
            elif k == 'viewer_num_format':
                try:
                    overrides['viewer_num_format'] = int(v)
                except ValueError:
                    pass
        return overrides

    def _effective_settings(self, line: str) -> dict:
        """Merge instance settings with any per-cell overrides from the magic line."""
        s = dict(self.settings)
        s.update(self._parse_line_args(line))
        return s

    @line_magic
    def pivotal_set(self, line):
        """Set persistent Pivotal output options.

        Usage:
            %pivotal_set output_type=viewer   # viewer | inline | both
            %pivotal_set output_code=true     # print generated Python code
            %pivotal_set output_type=inline output_code=false
        """
        updates = self._parse_line_args(line)
        if not updates:
            print(f"Current settings: {self.settings}")
            print("Usage: %pivotal_set output_type=viewer|inline|both output_code=true|false")
            return
        self.settings.update(updates)
        print(f"Pivotal settings: {self.settings}")

    # -------------------------------------------------------------------------
    # Cell magic
    # -------------------------------------------------------------------------

    @cell_magic
    def pivotal(self, line, cell):
        """Execute Pivotal DSL code.

        Per-cell options (override persistent settings for this cell only):
            %%pivotal output_type=inline
            %%pivotal output_code=true

        Persistent settings: use %pivotal_set
        """
        s = self._effective_settings(line)
        output_type = s['output_type']
        output_code = s['output_code']

        self.shell.push({'pd': pd})

        if not cell.endswith('\n'):
            cell += '\n'

        results = self.parser.parse(cell)

        if isinstance(results, dict) and 'error' in results:
            print(f"Pivotal Parse Error: {results['error']}")
            return

        python_code_list = self.parser.generate_code(results)
        combined = '\n\n'.join(python_code_list)

        # In viewer-only mode, close each chart figure within the cell code so the
        # inline backend's post-execute hook has nothing to render.
        if output_type == 'viewer':
            close_stmts = []
            for node in results:
                if isinstance(node, dict) and node.get('type') == 'plot':
                    chart_name = node.get('name') or node.get('table_name') or 'chart'
                    close_stmts.append(
                        f'import matplotlib.pyplot as _mpl; _mpl.close({chart_name})'
                    )
            if close_stmts:
                combined += '\n\n' + '\n'.join(close_stmts)

        result = self.shell.run_cell(combined)

        if not result.error_in_exec:
            if output_code:
                cleaned = self._clean_code(combined)
                if cleaned:
                    print(cleaned)

            if output_type in ('viewer', 'both'):
                try:
                    _send_results_to_viewer(self._viewer, results, self.shell.user_ns, settings=s)
                except Exception as e:
                    print(f"[Pivotal] viewer error: {e}")

            if output_type in ('inline', 'both'):
                _display_inline(results, self.shell.user_ns)

        self.parser.update_autocomplete_info(self.shell.user_ns)


# ---------------------------------------------------------------------------
# Inline display helper (used when output_type is 'inline' or 'both')
# ---------------------------------------------------------------------------

def _display_inline(results: list, ns: dict):
    """Display DataFrame heads inline for inline/both modes."""
    try:
        from IPython.display import display as ipy_display
        import pandas as pd
    except ImportError:
        return

    seen: set = set()
    for node in results:
        if not isinstance(node, dict) or node.get('type') == 'plot':
            continue
        name = node.get('table_name')
        if name and name not in seen:
            seen.add(name)
            obj = ns.get(name)
            if isinstance(obj, pd.DataFrame):
                print(f"'{name}'  {obj.shape[0]:,} rows × {obj.shape[1]} cols")
                ipy_display(obj.head())


def load_ipython_extension(ipython):
    ipython.register_magics(PivotalMagics)
    # Clear any stale autocomplete file from a previous session so completions
    # don't offer tables/columns that no longer exist in the fresh kernel.
    try:
        import json as _json
        from pathlib import Path as _Path
        _ac = _Path('pivotal_autocomplete.json')
        if _ac.exists():
            _ac.write_text(_json.dumps({'tables': {}, 'current_table': None}))
    except Exception:
        pass
