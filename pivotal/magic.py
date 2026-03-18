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

def _infer_col_types(payload, use_visions: bool = False) -> dict:
    """Infer semantic column types for the filter UI.

    Returns {col: 'numeric'|'categorical'|'datetime'|'boolean'|'string'}.
    Tries visions if available (and use_visions is True); falls back to
    dtype + cardinality heuristic.
    """
    import pandas as pd
    col_types = {}
    n = max(len(payload), 1)

    # Optional: visions for richer semantic inference
    if use_visions:
        try:
            import visions
            typeset = visions.StandardSet()
            inferred = visions.infer_type(payload, typeset)
            _vmap = {
                'Integer': 'numeric', 'Float': 'numeric', 'Complex': 'numeric',
                'DateTime': 'datetime', 'Date': 'datetime', 'TimeDelta': 'numeric',
                'Boolean': 'boolean', 'Categorical': 'categorical',
                'Object': 'string', 'String': 'string', 'URL': 'string',
                'EmailAddress': 'string', 'IPAddress': 'string', 'UUID': 'string',
                'Path': 'string', 'Geometry': 'string',
            }
            for col, vtype in inferred.items():
                col_types[str(col)] = _vmap.get(type(vtype).__name__, 'string')
            return col_types
        except Exception:
            pass

    # Fallback: dtype + cardinality heuristic
    for col in payload.columns:
        col_str = str(col)
        dt = str(payload[col].dtype)
        if dt.startswith(('float', 'int', 'uint')) or dt in ('Int8', 'Int16', 'Int32', 'Int64',
                                                               'UInt8', 'UInt16', 'UInt32', 'UInt64',
                                                               'Float32', 'Float64'):
            col_types[col_str] = 'numeric'
        elif dt.startswith('datetime') or dt.startswith('date') or dt == 'period':
            col_types[col_str] = 'datetime'
        elif dt == 'bool' or dt == 'boolean':
            col_types[col_str] = 'boolean'
        elif dt == 'category':
            col_types[col_str] = 'categorical'
        else:
            # object / string — check for dates first, then cardinality for categorical
            try:
                sample = payload[col].dropna().head(20)
                if len(sample) >= 3:
                    try:
                        parsed = pd.to_datetime(sample, errors='coerce', format='mixed')
                    except TypeError:  # pandas < 2.0
                        parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
                    if parsed.notna().mean() >= 0.8:
                        col_types[col_str] = 'datetime'
                        continue
                n_unique = payload[col].nunique(dropna=True)
                threshold = min(50, max(5, int(n * 0.3)))
                col_types[col_str] = 'categorical' if n_unique <= threshold else 'string'
            except Exception:
                col_types[col_str] = 'string'

    return col_types


class _PivotalViewer:
    """Sends DataFrames and chart figures to the JupyterLab Object Viewer panel."""

    MAX_ROWS = 10_000

    def __init__(self, shell):
        self._shell = shell
        self._comm = None
        self._last_sent: dict = {}          # name → df, for re-send on row-limit change
        self._last_viewer_settings: dict = {}  # name → {viewer_font, viewer_num_format}
        self._post_delete_cb = None         # called after frontend-initiated delete

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
        """Handle messages from the viewer panel."""
        data = msg['content']['data']
        if data.get('type') == 'request' and data.get('name') in self._last_sent:
            limit = int(data.get('limit', self.MAX_ROWS))
            vs = self._last_viewer_settings.get(data['name'], {})
            self.send_dataframe(data['name'], self._last_sent[data['name']], limit=limit,
                                viewer_font=vs.get('viewer_font'),
                                viewer_num_format=vs.get('viewer_num_format'),
                                use_visions=vs.get('use_visions', False))
        elif data.get('type') == 'delete':
            name = data.get('name')
            if name:
                self._shell.user_ns.pop(name, None)
                self._last_sent.pop(name, None)
                self._last_viewer_settings.pop(name, None)
                if self._post_delete_cb:
                    self._post_delete_cb()

    def send_dataframe(self, name: str, df, limit: int = None,
                       viewer_font: float = None, viewer_num_format: int = None,
                       use_visions: bool = False):
        self._ensure_comm()
        if self._comm is None:
            return
        limit = limit or self.MAX_ROWS
        self._last_sent[name] = df
        if viewer_font is not None or viewer_num_format is not None or use_visions:
            self._last_viewer_settings[name] = {
                'viewer_font': viewer_font,
                'viewer_num_format': viewer_num_format,
                'use_visions': use_visions,
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
                'col_types': _infer_col_types(payload, use_visions=use_visions),
                'shape': list(df.shape),
                'truncated': truncated,
                'viewer_font': viewer_font,
                'viewer_num_format': viewer_num_format,
            })
        except Exception:
            pass

    def send_chart(self, name: str, fig, canvas_meta: dict = None, source_df: str = None):
        """Render fig to PNG and send to the viewer."""
        self._ensure_comm()
        if self._comm is None:
            return
        import io, base64, struct
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            data = buf.getvalue()
            # bbox_inches='tight' may crop the figure, changing its pixel dimensions
            # vs what _build_canvas_meta measured via fig.get_size_inches().
            # Read actual PNG width/height from the IHDR chunk (bytes 16-23) and
            # recompute chart_height_mm so the viewer renders without squishing.
            if canvas_meta and canvas_meta.get('chart_width_mm') and len(data) >= 24:
                w_px, h_px = struct.unpack('>II', data[16:24])
                if w_px > 0:
                    canvas_meta = dict(canvas_meta)
                    canvas_meta['chart_height_mm'] = canvas_meta['chart_width_mm'] * (h_px / w_px)
            msg = {
                'type': 'chart',
                'name': name,
                'data': base64.b64encode(data).decode(),
            }
            if canvas_meta:
                msg['canvas'] = canvas_meta
            if source_df:
                msg['source_df'] = source_df
            self._comm.send(msg)
        except Exception:
            pass

    def send_delete(self, name: str):
        """Tell the frontend viewer to remove an item (Python-initiated delete)."""
        self._ensure_comm()
        if self._comm is None:
            return
        self._last_sent.pop(name, None)
        self._last_viewer_settings.pop(name, None)
        try:
            self._comm.send({'type': 'delete', 'name': name})
        except Exception:
            pass

    def send_insert_cell(self, code: str):
        """Ask the frontend to insert a new notebook cell with the given code and run it."""
        self._ensure_comm()
        if self._comm is None:
            return
        try:
            self._comm.send({'type': 'insert_cell', 'code': code})
        except Exception:
            pass

    def send_upsert_cell(self, code: str, gui_id: str):
        """Insert or overwrite the cell previously created by this GUI instance."""
        self._ensure_comm()
        if self._comm is None:
            return
        try:
            self._comm.send({'type': 'upsert_cell', 'code': code, 'gui_id': gui_id})
        except Exception:
            pass

    def send_current_table(self, name: str | None):
        """Notify the explorer which DataFrame is the current active table."""
        self._ensure_comm()
        if self._comm is None:
            return
        try:
            self._comm.send({'type': 'current_table', 'name': name})
        except Exception:
            pass

    def send_table(self, name: str, html: str, canvas: str = 'none', source_df: str = None):
        """Send a rendered GT table to the viewer."""
        self._ensure_comm()
        if self._comm is None:
            return
        msg = {
            'type': 'gt_table',
            'name': name,
            'html': html,
        }
        if source_df:
            msg['source_df'] = source_df
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
        if node.get('type') in ('plot', 'agg_plot'):
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
                use_visions=(settings or {}).get('use_visions', False),
            )

    # Send charts: look up the figure stored in the namespace by chart name
    for node in plot_nodes:
        chart_name = node.get('name') or node.get('table_name') or 'chart'
        fig = ns.get(chart_name)
        # agg_plot: send df first so chart arrives last and gets viewer focus
        if node.get('type') == 'agg_plot':
            df_name = f"{chart_name}_df"
            df = ns.get(df_name)
            if isinstance(df, pd.DataFrame):
                viewer.send_dataframe(df_name, df,
                                      viewer_font=(settings or {}).get('viewer_font'),
                                      viewer_num_format=(settings or {}).get('viewer_num_format'),
                                      use_visions=(settings or {}).get('use_visions', False))
        if isinstance(fig, mfig.Figure):
            canvas_meta = _build_canvas_meta(fig, settings or {},
                                             canvas_override=node.get('canvas'))
            viewer.send_chart(chart_name, fig, canvas_meta=canvas_meta,
                              source_df=node.get('table_name'))

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
            viewer.send_table(tbl_name, viewer_html, canvas,
                              source_df=node.get('table_name'))


# ---------------------------------------------------------------------------
# Plot GUI
# ---------------------------------------------------------------------------

def plot_gui(df_name: str = None):
    """Display an interactive widget for building %%pivotal agg plot commands."""
    try:
        import ipywidgets as widgets
        from IPython.display import display as ipy_display
        import IPython
    except ImportError:
        print('[Pivotal] ipywidgets is required for plot_gui. Install with: pip install ipywidgets')
        return

    shell = IPython.get_ipython()
    ns = shell.user_ns if shell else {}

    try:
        import pandas as pd
        df_names = [k for k, v in ns.items() if isinstance(v, pd.DataFrame) and not k.startswith('_')]
    except ImportError:
        df_names = []

    if not df_names:
        print('[Pivotal] No DataFrames found in namespace.')
        return

    # Layout constants
    _dd  = widgets.Layout(width='130px')
    _tx  = widgets.Layout(width='140px')
    _lbl = widgets.Layout(width='44px')
    _sm  = widgets.Layout(width='26px', height='24px', padding='0px')

    def _lh(text):
        return widgets.HTML(f'<b style="line-height:1.8">{text}</b>', layout=_lbl)

    def _row(*ws):
        return widgets.HBox(list(ws), layout=widgets.Layout(align_items='center', margin='2px 0'))

    # Static widgets
    df_sel       = widgets.Dropdown(options=df_names,
                       value=df_name if df_name in df_names else df_names[0],
                       description='', layout=_dd)
    chart_type   = widgets.Dropdown(options=['bar', 'line', 'scatter', 'area'],
                       value='bar', description='', layout=_dd)
    chart_name_w = widgets.Text(value='', placeholder='name', description='', layout=_tx)
    agg_func     = widgets.Dropdown(options=['mean', 'sum', 'count', 'min', 'max', 'median'],
                       value='mean', description='', layout=_dd)
    x_col        = widgets.Dropdown(options=[], description='', layout=_dd)
    x_label_w    = widgets.Text(value='', placeholder='x label', description='', layout=_tx)
    y_label_w    = widgets.Text(value='', placeholder='y label', description='', layout=_tx)
    by_col       = widgets.Dropdown(options=[], description='', layout=_dd)
    output       = widgets.Output()
    gen_btn      = widgets.Button(description='Generate', button_style='primary',
                                  layout=widgets.Layout(width='90px'))

    # Dynamic Y rows
    y_rows    = []   # list of Dropdown widgets
    y_section = widgets.VBox([])
    add_y_btn = widgets.Button(description='+', layout=_sm)

    def _make_y_dd(cols):
        return widgets.Dropdown(options=cols, description='', layout=_dd)

    def _rebuild_y():
        rows = []
        for i, dd in enumerate(y_rows):
            minus = widgets.Button(description='−', layout=_sm)
            minus.on_click(lambda _, d=dd: _remove_y(d))
            if i == 0:
                lbl  = _lh('y')
                btns = widgets.HBox(
                    [add_y_btn, minus] if len(y_rows) > 1 else [add_y_btn],
                    layout=widgets.Layout(align_items='center'))
            else:
                lbl  = widgets.HTML('', layout=_lbl)
                btns = widgets.HBox([minus], layout=widgets.Layout(align_items='center'))
            rows.append(widgets.HBox([lbl, dd, btns],
                                     layout=widgets.Layout(align_items='center', margin='2px 0')))
        # agg func + y label pinned at bottom of Y block
        rows.append(_row(widgets.HTML('', layout=_lbl), agg_func, y_label_w))
        y_section.children = tuple(rows)

    def _remove_y(dd):
        if dd in y_rows and len(y_rows) > 1:
            y_rows.remove(dd)
            _rebuild_y()

    def _add_y(_btn=None):
        y_rows.append(_make_y_dd(list(x_col.options)))
        _rebuild_y()

    add_y_btn.on_click(_add_y)

    def _get_cols(name):
        df = ns.get(name)
        try:
            return list(df.columns) if df is not None else []
        except Exception:
            return []

    def _refresh_cols(change=None):
        cols = _get_cols(df_sel.value)
        x_col.options = cols
        if cols:
            x_col.value = cols[0]
        for dd in y_rows:
            dd.options = cols
        by_col.options = ['(none)'] + cols
        by_col.value = '(none)'

    df_sel.observe(_refresh_cols, names='value')

    # Initialise with one Y row then populate column lists
    y_rows.append(_make_y_dd([]))
    _refresh_cols()
    _rebuild_y()

    def _build_dsl(_btn=None, _gui_id=''):
        table = df_sel.value
        kind  = chart_type.value
        name  = chart_name_w.value.strip() or f'{table}_plot'
        func  = agg_func.value
        x     = x_col.value
        xl    = x_label_w.value.strip()
        ys    = [dd.value for dd in y_rows if dd.value]
        yl    = y_label_w.value.strip()
        by    = by_col.value if by_col.value != '(none)' else None

        if not ys:
            with output:
                output.clear_output()
                print('Select at least one Y column.')
            return

        params = []
        params.append(f'  x {x} "{xl}"' if xl else f'  x {x}')
        y_part = ' '.join(ys)
        params.append(f'  y {func} {y_part} "{yl}"' if yl else f'  y {func} {y_part}')
        if by:
            params.append(f'  by {by}')

        dsl = f'%%pivotal\ndf {table}\nagg plot {kind} {name}\n' + '\n'.join(params)

        viewer = _get_viewer()
        if viewer is not None:
            viewer.send_upsert_cell(dsl, _gui_id)
        else:
            with output:
                output.clear_output()
                print('Generated DSL:\n' + dsl)

    _gui_id = str(id(gen_btn))

    def _build_dsl_plot(_btn=None):
        _build_dsl(_btn, _gui_id)

    gen_btn.on_click(_build_dsl_plot)

    content = widgets.VBox([
        _row(_lh('df'),   df_sel),
        _row(_lh('plot'), chart_type, chart_name_w),
        _row(_lh('x'),    x_col, x_label_w),
        y_section,
        _row(_lh('by'),   by_col),
        widgets.HBox([gen_btn], layout=widgets.Layout(margin='6px 0 2px 0')),
        output,
    ])
    accordion = widgets.Accordion(children=[content], selected_index=0)
    accordion.set_title(0, 'Pivotal Plot')
    ipy_display(accordion)


def pivot_gui(df_name: str = None):
    """Display an interactive widget for building %%pivotal pivot commands."""
    try:
        import ipywidgets as widgets
        from IPython.display import display as ipy_display
        import IPython
    except ImportError:
        print('[Pivotal] ipywidgets is required for pivot_gui. Install with: pip install ipywidgets')
        return

    shell = IPython.get_ipython()
    ns = shell.user_ns if shell else {}

    try:
        import pandas as pd
        df_names = [k for k, v in ns.items() if isinstance(v, pd.DataFrame) and not k.startswith('_')]
    except ImportError:
        df_names = []

    if not df_names:
        print('[Pivotal] No DataFrames found in namespace.')
        return

    _dd  = widgets.Layout(width='130px')
    _tx  = widgets.Layout(width='140px')
    _lbl = widgets.Layout(width='44px')
    _sm  = widgets.Layout(width='26px', height='24px', padding='0px')

    def _lh(text):
        return widgets.HTML(f'<b style="line-height:1.8">{text}</b>', layout=_lbl)

    def _row(*ws):
        return widgets.HBox(list(ws), layout=widgets.Layout(align_items='center', margin='2px 0'))

    df_sel   = widgets.Dropdown(options=df_names,
                   value=df_name if df_name in df_names else df_names[0],
                   description='', layout=_dd)
    agg_func = widgets.Dropdown(options=['mean', 'sum', 'count', 'min', 'max', 'median'],
                   value='mean', description='', layout=_dd)
    val_col  = widgets.Dropdown(options=[], description='', layout=_dd)
    output   = widgets.Output()
    gen_btn  = widgets.Button(description='Generate', button_style='primary',
                              layout=widgets.Layout(width='90px'))

    def _make_dd(cols):
        return widgets.Dropdown(options=cols, description='', layout=_dd)

    def _make_dynamic_section(label):
        """Return (item_list, section_vbox, add_btn, rebuild_fn, remove_fn, add_fn)."""
        items    = []
        section  = widgets.VBox([])
        add_btn  = widgets.Button(description='+', layout=_sm)

        def _rebuild():
            rows = []
            for i, dd in enumerate(items):
                minus = widgets.Button(description='−', layout=_sm)
                minus.on_click(lambda _, d=dd: _remove(d))
                if i == 0:
                    lbl  = _lh(label)
                    btns = widgets.HBox(
                        [add_btn, minus] if len(items) > 1 else [add_btn],
                        layout=widgets.Layout(align_items='center'))
                else:
                    lbl  = widgets.HTML('', layout=_lbl)
                    btns = widgets.HBox([minus], layout=widgets.Layout(align_items='center'))
                rows.append(widgets.HBox([lbl, dd, btns],
                                         layout=widgets.Layout(align_items='center', margin='2px 0')))
            section.children = tuple(rows)

        def _remove(dd):
            if dd in items and len(items) > 1:
                items.remove(dd)
                _rebuild()

        def _add(_btn=None):
            items.append(_make_dd(list(val_col.options)))
            _rebuild()

        add_btn.on_click(_add)
        return items, section, _rebuild

    rows_items, rows_section, _rebuild_rows = _make_dynamic_section('rows')
    cols_items, cols_section, _rebuild_cols = _make_dynamic_section('cols')

    def _get_cols(name):
        df = ns.get(name)
        try:
            return list(df.columns) if df is not None else []
        except Exception:
            return []

    def _refresh_cols(change=None):
        cols = _get_cols(df_sel.value)
        for dd in rows_items + cols_items:
            dd.options = cols
        val_col.options = cols

    df_sel.observe(_refresh_cols, names='value')

    rows_items.append(_make_dd([]))
    cols_items.append(_make_dd([]))
    _refresh_cols()
    _rebuild_rows()
    _rebuild_cols()

    def _build_dsl(_btn=None):
        table  = df_sel.value
        row_cs = [dd.value for dd in rows_items if dd.value]
        col_cs = [dd.value for dd in cols_items if dd.value]
        func   = agg_func.value
        val    = val_col.value

        if not row_cs or not col_cs or not val:
            with output:
                output.clear_output()
                print('Fill in rows, cols, and agg value columns.')
            return

        dsl = (f'%%pivotal\ndf {table}\npivot\n'
               f'  rows {" ".join(row_cs)}\n'
               f'  cols {" ".join(col_cs)}\n'
               f'  agg {func} {val}')

        viewer = _get_viewer()
        if viewer is not None:
            viewer.send_upsert_cell(dsl, _gui_id)
        else:
            with output:
                output.clear_output()
                print('Generated DSL:\n' + dsl)

    _gui_id = str(id(gen_btn))
    gen_btn.on_click(_build_dsl)

    content = widgets.VBox([
        _row(_lh('df'),  df_sel),
        _row(_lh('agg'), agg_func, val_col,
             widgets.HTML('<span style="color:var(--jp-ui-font-color2);font-size:0.85em">(values)</span>')),
        rows_section,
        cols_section,
        widgets.HBox([gen_btn], layout=widgets.Layout(margin='6px 0 2px 0')),
        output,
    ])
    accordion = widgets.Accordion(children=[content], selected_index=0)
    accordion.set_title(0, 'Pivotal Pivot')
    ipy_display(accordion)


def load_gui():
    """Display an interactive widget for building %%pivotal load commands."""
    try:
        import ipywidgets as widgets
        from IPython.display import display as ipy_display
    except ImportError:
        print('[Pivotal] ipywidgets is required for load_gui. Install with: pip install ipywidgets')
        return

    _dd  = widgets.Layout(width='200px')
    _lbl = widgets.Layout(width='44px')

    def _lh(text):
        return widgets.HTML(f'<b style="line-height:1.8">{text}</b>', layout=_lbl)

    def _row(*ws):
        return widgets.HBox(list(ws), layout=widgets.Layout(align_items='center', margin='2px 0'))

    name_w  = widgets.Text(value='', placeholder='df name', description='', layout=_dd)
    output  = widgets.Output()
    gen_btn = widgets.Button(description='Load', button_style='primary',
                             layout=widgets.Layout(width='70px'))

    from ipyfilechooser import FileChooser
    fc = FileChooser('.', show_hidden=False)
    fc.title = ''
    fc.layout.width = '100%'
    fc._select.description = '📁'
    fc._change_desc = '📁'          # keeps icon after a file is selected
    fc._filename.layout.display = 'none'  # hide the path label beside the button

    def _build_dsl(_btn=None):
        name = name_w.value.strip()
        path = fc.selected
        if not name or not path:
            with output:
                output.clear_output()
                print('Enter a df name and select a file.')
            return
        dsl = f'%%pivotal\nload {name} "{path}"'
        viewer = _get_viewer()
        if viewer is not None:
            viewer.send_insert_cell(dsl)
        else:
            with output:
                output.clear_output()
                print('Generated DSL:\n' + dsl)

    gen_btn.on_click(_build_dsl)

    content = widgets.VBox([
        _row(_lh('df'),   name_w),
        fc,
        widgets.HBox([gen_btn], layout=widgets.Layout(margin='6px 0 2px 0')),
        output,
    ], layout=widgets.Layout(width='100%'))
    accordion = widgets.Accordion(children=[content], selected_index=0,
                                  layout=widgets.Layout(width='100%'))
    accordion.set_title(0, 'Pivotal Load')
    ipy_display(accordion)


def _get_viewer() -> '_PivotalViewer | None':
    """Return the active _PivotalViewer if one exists, else None."""
    import IPython
    shell = IPython.get_ipython()
    if shell is None:
        return None
    # Search registered cell magics for a PivotalMagics instance
    for fn in shell.magics_manager.magics.get('cell', {}).values():
        obj = getattr(fn, '__self__', None)
        if obj is not None and isinstance(obj, PivotalMagics) and hasattr(obj, '_viewer'):
            return obj._viewer
    return None


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
        self._viewer._post_delete_cb = lambda: self.parser.update_autocomplete_info(shell.user_ns)
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
        'viewer_font': 1.0,         # em units for DataFrame viewer font size
        'viewer_num_format': 5,     # significant digits for float columns (0 = no formatting)
        'use_visions': False,     # True = use visions for type inference if installed
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
            elif k == 'use_visions':
                overrides['use_visions'] = v in ('true', '1', 'yes')
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

        # Pre-process top-level `del <name>` commands (not inside python: blocks)
        import re as _re
        del_names = []
        kept_lines = []
        for _line in cell.split('\n'):
            _m = _re.match(r'^delete\s+(\w+)\s*$', _line)
            if _m:
                del_names.append(_m.group(1))
            else:
                kept_lines.append(_line)
        if del_names:
            cell = '\n'.join(kept_lines)

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
                if isinstance(node, dict) and node.get('type') in ('plot', 'agg_plot'):
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
                    self._viewer.send_current_table(self.shell.user_ns.get('__table_name__'))
                except Exception as e:
                    print(f"[Pivotal] viewer error: {e}")

            if output_type in ('inline', 'both'):
                _display_inline(results, self.shell.user_ns)

        # Execute any del commands collected above
        for _name in del_names:
            self.shell.user_ns.pop(_name, None)
            self._viewer.send_delete(_name)

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


# ---------------------------------------------------------------------------
# Module-level instance reference — set by load_ipython_extension so that
# the module-level update() function can reach the active magics instance
# regardless of whether the user accesses it via `import pivotal` or the
# injected namespace object.
# ---------------------------------------------------------------------------

_active_magics: 'PivotalMagics | None' = None


def delete(name: str):
    """Delete a named object from the notebook namespace and the Pivotal viewer.

    Example::

        pivotal.delete('df_sales')
    """
    if _active_magics is None:
        print("[Pivotal] Extension not loaded. Run %load_ext pivotal first.")
        return
    m = _active_magics
    m.shell.user_ns.pop(name, None)
    m._viewer.send_delete(name)
    m.parser.update_autocomplete_info(m.shell.user_ns)


def update():
    """Send all DataFrames in the notebook namespace to the Pivotal viewer.

    Call this after creating or modifying DataFrames in pure Python cells
    to refresh the Object Viewer and the left-panel item list.

    Example::

        df_sales = pd.read_csv('sales.csv')
        pivotal.update()
    """
    if _active_magics is None:
        print("[Pivotal] Extension not loaded. Run %load_ext pivotal first.")
        return

    try:
        import pandas as pd
    except ImportError:
        print("[Pivotal] pandas not available")
        return

    m = _active_magics
    ns = m.shell.user_ns
    s = m.settings

    sent = []
    for name, obj in ns.items():
        if name.startswith('_'):
            continue
        if isinstance(obj, pd.DataFrame):
            m._viewer.send_dataframe(
                name, obj,
                viewer_font=s.get('viewer_font'),
                viewer_num_format=s.get('viewer_num_format'),
                use_visions=s.get('use_visions', False),
            )
            sent.append(name)

    m.parser.update_autocomplete_info(ns)
    m._viewer.send_current_table(m.shell.user_ns.get('__table_name__'))

    if sent:
        print(f"[Pivotal] Updated viewer: {', '.join(sent)}")
    else:
        print("[Pivotal] No DataFrames found in namespace")


def load_ipython_extension(ipython):
    global _active_magics
    magics = PivotalMagics(ipython)
    _active_magics = magics
    ipython.register_magics(magics)
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
