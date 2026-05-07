"""
pivotal.validator — semantic validation of a parsed Pivotal AST.

Runs after parsing, before code generation.  Checks that referenced
tables exist in the session namespace and that column names are valid,
tracking how each statement transforms the column set through the pipeline.
"""
from __future__ import annotations

import ast
import re
from typing import Optional
from .errors import PivotalError, _make_suggestion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_columns(table_name: str, namespace: dict) -> Optional[set]:
    """Return the column set of a DataFrame in the namespace, or None if not found."""
    obj = namespace.get(table_name)
    if obj is None:
        return None
    try:
        return {str(c) for c in obj.columns}
    except AttributeError:
        return None


def _extract_col_names(items) -> list:
    """Extract plain string column names from a list, skipping Python var dicts."""
    return [item for item in items if isinstance(item, str)]


def _available_tables(namespace: dict) -> list:
    """Return names of DataFrames in the namespace."""
    return [k for k, v in namespace.items() if hasattr(v, 'columns') and not k.startswith('_')]


def _table_error(name: str, namespace: dict, cell_defined: set) -> Optional[PivotalError]:
    """Return a PivotalError if table *name* cannot be resolved, else None."""
    if name in namespace or name in cell_defined:
        return None
    candidates = _available_tables(namespace)
    suggestion = _make_suggestion(name, candidates)
    if not suggestion and candidates:
        suggestion = f"Available tables: {', '.join(sorted(candidates))}"
    return PivotalError(
        message=f"Table '{name}' not found",
        error_type="Validation Error",
        suggestion=suggestion,
    )


def _col_errors(col_refs: list, current_cols: Optional[set],
                table_name: str, statement: str) -> list:
    """
    Check each name in col_refs exists in current_cols.
    Silently skips if current_cols is None (columns unknown).
    """
    if current_cols is None:
        return []
    errors = []
    for col in col_refs:
        if col and col not in current_cols:
            suggestion = _make_suggestion(col, list(current_cols))
            errors.append(PivotalError(
                message=f"Unknown column '{col}' in '{statement}' on table '{table_name}'",
                error_type="Validation Error",
                suggestion=suggestion,
            ))
    return errors


def _decode_pattern(pattern) -> str:
    """Decode a parser string literal enough for regex validation."""
    raw = str(pattern)
    try:
        return ast.literal_eval('"' + raw.replace('"', '\\"') + '"')
    except (SyntaxError, ValueError):
        return raw


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate(ast: list, namespace: dict, source_code: str = "") -> list:
    """
    Validate a parsed AST against *namespace*.

    Returns a (possibly empty) list of PivotalError objects.
    Errors are blocking; the caller should not generate code if any are present.
    Validation is skipped silently when the relevant table is not yet available
    (e.g. within-cell defined tables whose columns are not known until runtime).
    """
    errors: list[PivotalError] = []

    # Forward pass: collect table names that this cell itself defines so we
    # don't flag them as missing when they're referenced in later statements.
    cell_defined: set = set()
    for node in ast:
        if isinstance(node, dict) and node.get('type') in ('load_table', 'bulk_load', 'copy_table'):
            name = node.get('table_name')
            if name:
                cell_defined.add(name)

    # Main pass — walk AST in statement order.
    current_table: Optional[str] = None
    current_cols: Optional[set] = None   # None = unknown, skip column checks

    for node in ast:
        if not isinstance(node, dict):
            continue

        t = node.get('type')
        tbl = node.get('table_name', current_table) or current_table

        # ----------------------------------------------------------------
        # Table-definition nodes — set up current_table and current_cols
        # ----------------------------------------------------------------

        if t == 'load_table':
            # load sales "file.csv" — columns unknown until the file is read
            current_table = node['table_name']
            current_cols = None

        elif t == 'bulk_load':
            # bulk load reads file lists; columns are unknown until runtime.
            current_table = node.get('table_name')
            current_cols = None

        elif t == 'copy_table':
            # df summary from sales
            current_table = node['table_name']
            src = node.get('copy_from')
            if src:
                err = _table_error(src, namespace, cell_defined)
                if err:
                    errors.append(err)
                    current_cols = None
                elif src in cell_defined:
                    current_cols = None   # defined in cell; columns unknown at validate time
                else:
                    current_cols = _df_columns(src, namespace)
            else:
                current_cols = None

        elif t == 'validate_table':
            # df sales  (without "from") — reference an existing table
            current_table = node['table_name']
            err = _table_error(current_table, namespace, cell_defined)
            if err:
                errors.append(err)
                current_cols = None
            elif current_table in cell_defined:
                current_cols = None
            else:
                current_cols = _df_columns(current_table, namespace)

        # ----------------------------------------------------------------
        # Column-checking nodes — verify refs, then update column set
        # ----------------------------------------------------------------

        elif t == 'filter':
            conditions = node.get('conditions', [])
            cols = [c['column'] for c in conditions
                    if isinstance(c, dict) and isinstance(c.get('column'), str)]
            errors.extend(_col_errors(cols, current_cols, tbl, 'filter'))
            # column set unchanged

        elif t == 'data_quality':
            if node.get('rule') == 'condition':
                conditions = node.get('conditions', [])
                cols = [c['column'] for c in conditions
                        if isinstance(c, dict) and isinstance(c.get('column'), str)]
            else:
                cols = node.get('columns', [])
            errors.extend(_col_errors(cols, current_cols, tbl, node.get('mode', 'data quality')))
            # column set unchanged

        elif t == 'select':
            cols = node.get('columns', [])
            if node.get('column_match') and current_cols is not None:
                pattern = _decode_pattern(node.get('column_match'))
                cols = [c for c in current_cols if re.search(pattern, c)]
            errors.extend(_col_errors(cols, current_cols, tbl, 'select'))
            if current_cols is not None:
                renames = node.get('renames', {})
                current_cols = {renames.get(c, c) for c in cols if c in current_cols}

        elif t == 'drop':
            cols = node.get('columns', [])
            if node.get('column_match') and current_cols is not None:
                pattern = _decode_pattern(node.get('column_match'))
                cols = [c for c in current_cols if re.search(pattern, c)]
            errors.extend(_col_errors(cols, current_cols, tbl, 'drop'))
            if current_cols is not None:
                current_cols = current_cols - set(cols)

        elif t == 'rename':
            renames = node.get('renames', {})
            errors.extend(_col_errors(list(renames.keys()), current_cols, tbl, 'rename'))
            if current_cols is not None:
                for old, new in renames.items():
                    if old in current_cols:
                        current_cols = (current_cols - {old}) | {new}

        elif t == 'sort':
            cols = node.get('columns', [])
            errors.extend(_col_errors(cols, current_cols, tbl, 'sort'))

        elif t == 'cast':
            cols = node.get('columns', [])
            errors.extend(_col_errors(cols, current_cols, tbl, 'cast'))

        elif t == 'groupby':
            by_cols = _extract_col_names(node.get('by', []))
            agg_list = node.get('agg_list', [])
            agg_cols = []
            for item in agg_list:
                if not isinstance(item, dict):
                    continue
                col = item.get('column')
                if isinstance(col, str):
                    agg_cols.append(col)
                weight = item.get('weight')   # wavg weight column
                if isinstance(weight, str):
                    agg_cols.append(weight)
            errors.extend(_col_errors(by_cols + agg_cols, current_cols, tbl, 'group by / agg'))
            if current_cols is not None:
                new_cols = set(by_cols)
                for item in agg_list:
                    if not isinstance(item, dict):
                        continue
                    alias = item.get('alias') or item.get('column')
                    if isinstance(alias, str):
                        new_cols.add(alias)
                current_cols = new_cols

        elif t == 'assign':
            target = node.get('target')
            if current_cols is not None and isinstance(target, str):
                current_cols = current_cols | {target}
            # Expression contents are not statically analysed (too complex)

        elif t == 'rank':
            col = node.get('column')
            partition = _extract_col_names(node.get('partition', []))
            order_col = node.get('order_col')
            check = ([col] if isinstance(col, str) else []) + partition
            if isinstance(order_col, str):
                check.append(order_col)
            errors.extend(_col_errors(check, current_cols, tbl, 'rank'))
            if current_cols is not None:
                result_col = node.get('result_col')
                if isinstance(result_col, str):
                    current_cols = current_cols | {result_col}

        elif t == 'shift':   # lag / lead
            col = node.get('column')
            partition = _extract_col_names(node.get('partition', []))
            order_col = node.get('order_col')
            check = ([col] if isinstance(col, str) else []) + partition
            if isinstance(order_col, str):
                check.append(order_col)
            errors.extend(_col_errors(check, current_cols, tbl, node.get('func', 'lag/lead')))
            if current_cols is not None:
                result_col = node.get('result_col')
                if isinstance(result_col, str):
                    current_cols = current_cols | {result_col}

        elif t == 'cumulative':
            col = node.get('column')
            partition = _extract_col_names(node.get('partition', []))
            order_col = node.get('order_col')
            check = ([col] if isinstance(col, str) else []) + partition
            if isinstance(order_col, str):
                check.append(order_col)
            errors.extend(_col_errors(check, current_cols, tbl, node.get('func', 'cumulative')))
            if current_cols is not None:
                result_col = node.get('result_col')
                if isinstance(result_col, str):
                    current_cols = current_cols | {result_col}

        elif t == 'rolling':
            col = node.get('column')
            partition = _extract_col_names(node.get('partition', []))
            order_col = node.get('order_col')
            check = ([col] if isinstance(col, str) else []) + partition
            if isinstance(order_col, str):
                check.append(order_col)
            errors.extend(_col_errors(check, current_cols, tbl, 'rolling'))
            if current_cols is not None:
                result_col = node.get('result_col')
                if isinstance(result_col, str):
                    current_cols = current_cols | {result_col}

        elif t == 'dropna':
            cols = node.get('columns', [])
            if cols:
                errors.extend(_col_errors(cols, current_cols, tbl, 'dropna'))

        elif t == 'distinct':
            cols = node.get('columns', [])
            if cols:
                errors.extend(_col_errors(cols, current_cols, tbl, 'distinct'))
            # column set unchanged (distinct deduplicates rows, keeps all columns)

        elif t == 'fillna':
            per_col = node.get('per_col', {})
            if per_col:
                errors.extend(_col_errors(list(per_col.keys()), current_cols, tbl, 'fillna'))

        elif t == 'round':
            cols = node.get('columns', [])
            errors.extend(_col_errors(cols, current_cols, tbl, 'round'))
            if current_cols is not None:
                alias = node.get('alias')
                if isinstance(alias, str):
                    current_cols = current_cols | {alias}

        elif t == 'pivot':
            index_cols = _extract_col_names(node.get('index', []))
            col_cols = _extract_col_names(node.get('columns', []))
            agg_cols = [item['column'] for item in node.get('agg_list', [])
                        if isinstance(item, dict) and isinstance(item.get('column'), str)]
            errors.extend(_col_errors(index_cols + col_cols + agg_cols, current_cols, tbl, 'pivot'))
            current_cols = None   # output shape is not statically predictable

        elif t == 'merge':
            right_table = node.get('right_table')
            # The parser stores join keys in 'keys' (a list) and kwargs as a
            # dict.  Read inline keys directly from the node and indented
            # merge options from kwargs.
            keys = node.get('keys') or []
            kwargs = node.get('kwargs') or {}
            if not isinstance(kwargs, dict):
                kwargs = {}

            # If current_cols is unknown but the left table is in the namespace,
            # seed it now so we can validate join keys.
            left_table = node.get('table_name')
            if current_cols is None and left_table and left_table in namespace:
                current_cols = _df_columns(left_table, namespace)

            right_cols = _df_columns(right_table, namespace) if right_table else None

            # Check right table exists
            if right_table:
                err = _table_error(right_table, namespace, cell_defined)
                if err:
                    errors.append(err)

            # Check join keys against the left table (current_cols)
            if keys:
                on_list = [keys] if isinstance(keys, str) else list(keys)
                errors.extend(_col_errors(on_list, current_cols, tbl, 'merge on'))
                errors.extend(_col_errors(on_list, right_cols, right_table, 'merge on'))

            # Indented form:
            #   merge customers
            #       on region
            #       left_on region
            #       right_on cust_region
            opt_on = kwargs.get('on')
            if isinstance(opt_on, str):
                errors.extend(_col_errors([opt_on], current_cols, tbl, 'merge on'))
                errors.extend(_col_errors([opt_on], right_cols, right_table, 'merge on'))

            left_on = kwargs.get('left_on')
            if isinstance(left_on, str):
                errors.extend(_col_errors([left_on], current_cols, tbl, 'merge left_on'))

            right_on = kwargs.get('right_on')
            if isinstance(right_on, str):
                errors.extend(_col_errors([right_on], right_cols, right_table, 'merge right_on'))

            current_cols = None   # merged column set is not statically predictable

        elif t == 'concat':
            for src in node.get('tables', []):
                err = _table_error(src, namespace, cell_defined)
                if err:
                    errors.append(err)
            # column set unchanged (concat stacks rows)

        elif t == 'intersect':
            for src in node.get('tables', []):
                err = _table_error(src, namespace, cell_defined)
                if err:
                    errors.append(err)

        # Other node types (plot, table, show, save, apply, python, etc.)
        # do not reference columns in ways we can statically check — skip.

    return errors
