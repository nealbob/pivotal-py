from lark import Lark, Transformer, v_args
from lark.indenter import Indenter
from lark import Tree
from lark.lexer import Token
import pandas as pd
import json
import os
import warnings
from pathlib import Path

# All reserved words in the Pivotal grammar.  Used for collision validation.
PIVOTAL_KEYWORDS = frozenset({
    # Statement keywords (not 'df' — it is unambiguous after its own token)
    'load', 'filter', 'select', 'assign', 'sort', 'order', 'save', 'all',
    'merge', 'pivot', 'group', 'python', 'plot', 'drop', 'fillna',
    'dropna', 'distinct', 'concat', 'rename', 'apply',
    # Clause keywords
    'from', 'where', 'as', 'on', 'by', 'rows', 'cols', 'agg', 'include', 'exclude',
    # Comparators / logic
    'in', 'not', 'between', 'contains', 'startswith', 'endswith',
    'and', 'or',
    # Aggregation functions
    'mean', 'min', 'max', 'sum', 'count', 'avg', 'median', 'std',
    # Sort / merge modifiers
    'asc', 'desc', 'left', 'right', 'inner', 'outer',
    # Atoms
    'true', 'false', 'none',
})

#AGG_DICT: _NL _INDENT IDENTIFIER AGG_FUNCTION ("," AGG_FUNCTION)* _DEDENT (_NL _INDENT IDENTIFIER AGG_FUNCTION ("," AGG_FUNCTION)* _DEDENT)* _NL?
# Grammar definition using indentation
grammar_indented = r"""
    %declare _INDENT _DEDENT
    start: _NL* (_INDENT? statement _DEDENT?)+ _NL*

    statement: load_statement
               | dataframe_statement
               | assign_statement
               | filter_statement
               | select_statement
               | sort_statement
               | merge_statement
               | pivot_statement
               | groupby_statement
               | python_statement
               | plot_statement
               | drop_statement
               | fillna_statement
               | dropna_statement
               | distinct_statement
               | concat_statement
               | rename_statement
               | apply_statement
               | save_statement

    apply_statement: "apply" IDENTIFIER _NL?

    plot_statement: "plot" IDENTIFIER IDENTIFIER? (_NL | _NL _INDENT plot_params _DEDENT)?

    plot_params: plot_param+
    plot_param: "by" IDENTIFIER _NL?           -> plot_by_param
              | "cols" value _NL?              -> plot_cols_param
              | IDENTIFIER value STRING _NL?   -> plot_labeled_param
              | IDENTIFIER value _NL?          -> plot_value_param
              | IDENTIFIER "=" value _NL?      -> plot_value_param
              | IDENTIFIER "=" list_value _NL? -> plot_list_param
              | IDENTIFIER list_value _NL?     -> plot_list_param

    python_statement: "python" UNQUOTED_STRING _NL?

    groupby_statement: "group" "by" group_cols (_NL _INDENT agg_clause _DEDENT)? _NL?

    group_cols: (IDENTIFIER | PYTHON_VAR) ("," (IDENTIFIER | PYTHON_VAR))*

    agg_clause: "agg" agg_item ("," agg_item)* _NL?

    agg_item: AGG_FUNCTION (IDENTIFIER | PYTHON_VAR) ("as" IDENTIFIER)?

    merge_statement: MERGE_TYPE? "merge" RIGHT_TABLE ("on" keys)? (_NL | _NL _INDENT params _DEDENT)?
    
    MERGE_TYPE: "left" | "right" | "inner" | "outer"
    keys: IDENTIFIER ("," IDENTIFIER)*
    RIGHT_TABLE: IDENTIFIER

    load_statement: "load" table_name (STRING | PATH | PYTHON_VAR) (_NL | _NL _INDENT params _DEDENT)?
                  | "load" "all" _NL?
                  | "load" table_name _NL?

    save_statement: "save" STRING (_NL _INDENT save_params _DEDENT)? _NL?

    save_params: save_param+
    save_param: "path" (STRING | PYTHON_VAR) _NL?           -> save_path
              | "format" IDENTIFIER _NL?                     -> save_format
              | "chart_format" IDENTIFIER _NL?               -> save_chart_format
              | "include" save_id_list _NL?                  -> save_include
              | "exclude" save_id_list _NL?                  -> save_exclude

    save_id_list: IDENTIFIER ("," IDENTIFIER)*

    dataframe_statement: "df" table_name ("from" copy_table)? _NL?

    assign_statement: "assign" target "=" expression (_NL | _NL _INDENT "where" condition_list _NL _DEDENT)?

    filter_statement: "filter" condition_list  _NL?
    
    select_statement: "select" select_item ("," select_item)* _NL?
    
    select_item: (IDENTIFIER | PYTHON_VAR) ("as" IDENTIFIER)?

    pivot_statement: "pivot" _NL _INDENT pivot_args _DEDENT

    pivot_args: (agg_clause | pivot_rows | pivot_cols)+

    pivot_rows: "rows" (IDENTIFIER | PYTHON_VAR) ("," (IDENTIFIER | PYTHON_VAR))* _NL?
    pivot_cols: "cols" (IDENTIFIER | PYTHON_VAR) ("," (IDENTIFIER | PYTHON_VAR))* _NL?
   
    AGG_FUNCTION: "mean" | "min" | "max" | "sum" | "count" | "avg" | "median" | "std"

    sort_statement: ("sort" | "order" "by") (IDENTIFIER | PYTHON_VAR) SORT_TYPE? ("," (IDENTIFIER | PYTHON_VAR) SORT_TYPE?)* _NL?

    SORT_TYPE: "asc" | "desc"
    
    target: IDENTIFIER
    table_name: IDENTIFIER
    copy_table: IDENTIFIER

    expression: UNQUOTED_STRING | STRING

    condition: IDENTIFIER COMPARATOR (value | list_value)
             | IDENTIFIER "in" list_value       -> condition_in_list
             | IDENTIFIER "in" PYTHON_VAR       -> condition_in_var
             | IDENTIFIER "not" "in" list_value -> condition_not_in_list
             | IDENTIFIER "not" "in" PYTHON_VAR -> condition_not_in_var

    condition_list: condition (AOR condition)*

    COMPARATOR: "==" | "!=" | ">" | "<" | ">=" | "<=" | "between" | "contains" | "not contains" | "startswith" | "endswith"

    drop_statement: "drop" IDENTIFIER ("," IDENTIFIER)* _NL?

    fillna_statement: "fillna" value _NL?

    dropna_statement: "dropna" dropna_cols? _NL?
    dropna_cols: IDENTIFIER ("," IDENTIFIER)*

    distinct_statement: "distinct" distinct_cols? _NL?
    distinct_cols: IDENTIFIER ("," IDENTIFIER)*

    concat_statement: "concat" IDENTIFIER ("," IDENTIFIER)* _NL?

    rename_statement: "rename" rename_item ("," rename_item)* _NL?
    rename_item: IDENTIFIER "as" IDENTIFIER

    params: param+

    param: keyword_arg _NL?

    keyword_arg: IDENTIFIER value 
                | IDENTIFIER "=" value 
                | IDENTIFIER "=" list_value 
                | IDENTIFIER list_value 

    file_path: PATH

    value: BOOLEAN | NUMBER | STRING | IDENTIFIER | PATH | NONE | PYTHON_VAR
    list_value: "[" [value ("," value)*] "]" |  [value "," value ("," value)*] 

    BOOLEAN.2: "True" | "False" | "true" | "false"
    NONE.2: "None" | "none"
    AOR.2: /and/i | /or/i
    PYTHON_VAR: ":" IDENTIFIER
    IDENTIFIER: /[a-zA-Z][a-zA-Z0-9_]*/
    IDENT_LIST.2: IDENTIFIER ("," IDENTIFIER)*
    STRING: /"[^"]*"/ | /'[^']*'/
    UNQUOTED_STRING: /[^\n]+/
    PATH: /[a-zA-Z0-9_]+[:\\\/][a-zA-Z0-9_:\/\\\.\-]+|[\\\/][a-zA-Z0-9_:\/\\\.\-]+|[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+/
    NUMBER: /\d+(\.\d+)?/
    COMMENT: /#[^\n]*/ | /--[^\n]*/
    MULTILINE_COMMENT:  /\/\*[\s\S]*?\*\//

    _NL: (/\r?\n[\t ]*/)+

    %import common.WS_INLINE
    %ignore WS_INLINE
    %ignore COMMENT
    %ignore MULTILINE_COMMENT
"""

# Define the Indenter for our DSL
class DSLIndenter(Indenter):
    NL_type = '_NL'
    OPEN_PAREN_types = []
    CLOSE_PAREN_types = []
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 4

@v_args(inline=True)
class DSLTransformer(Transformer):
    """Transform parse tree into AST"""
    
    def __init__(self):
        self.statements = []
        self.current_table = None
    
    def start(self, *statements):
        return list(statements)

    #def statement_type(self, stmt_type):
    #    return stmt_type.children

    def statement(self, stmt):
        return stmt
    
    def sort_statement(self, *args):
        """Handle sort statements to sort DataFrame by columns"""
        # Parse arguments into column/sort_type pairs
        columns = []
        ascending = []
        
        i = 0
        while i < len(args):
            arg = args[i]
            # Check if this is a sort_type Tree object (skip it, as it's not a column)
            if isinstance(arg, Token) and arg.type in ['SORT', 'ORDER', 'BY']:
                i += 1
                continue
            
            # This is a column identifier or variable
            if isinstance(arg, dict) and arg.get('type') == 'var':
                column = arg
            else:
                column = str(arg)
            
            columns.append(column)
            
            # Check if next arg is a sort_type
            if i + 1 < len(args) and isinstance(args[i + 1], Token) and args[i + 1].type == 'SORT_TYPE':
                sort_type = str(args[i + 1]).lower()
                ascending.append(sort_type == 'asc')
                i += 2
            else:
                # Default to ascending if no sort_type specified
                ascending.append(True)
                i += 1
        
        ast_node = {
            'type': 'sort',
            'table_name': self.current_table,
            'columns': columns,
            'ascending': ascending
        }
        
        return ast_node

    def _keyword_arg(self, params):

        kwargs = {}
        
        for param in params:
            if isinstance(param, dict):
                kwargs.update(param)

        kwargs_str = ', '.join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in kwargs.items()])
        
        if kwargs_str:
            kwargs_str = ', ' + kwargs_str
        
        return kwargs, kwargs_str

    def load_statement(self, *args):
        """Handle all three load forms:
        - load name "path"  → load_table (existing file)
        - load all          → load_all (all tables from active package)
        - load name         → load_package_table (named table from active package)
        """
        if len(args) == 0:
            # "load all" — no named children (both "load" and "all" are anonymous terminals)
            return {'type': 'load_all'}

        if len(args) == 1:
            # "load table_name" — package table load (no source path)
            table_name_str = str(args[0])
            return {'type': 'load_package_table', 'table_name': table_name_str}

        # len(args) >= 2: "load table_name source [params]"
        table_name, source = args[0], args[1]
        params = args[2] if len(args) > 2 else None

        if isinstance(source, dict) and source.get('type') == 'var':
            source_val = source
        else:
            source_val = str(source)

        if params:
            kwargs, kwargs_str = self._keyword_arg(params)
        else:
            kwargs = {}
            kwargs_str = ''

        # Extract sql_query before building kwargs_str so it doesn't leak
        # into pandas reader calls
        sql_query = kwargs.pop('query', None) if isinstance(kwargs, dict) else None
        if sql_query is not None:
            kwargs_str = ', '.join(
                f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}"
                for k, v in kwargs.items()
            )
            if kwargs_str:
                kwargs_str = ', ' + kwargs_str

        ast_node = {
            'type': 'load_table',
            'table_name': str(table_name),
            'source': source_val,
            'kwargs': kwargs,
            'kwargs_str': kwargs_str,
            'sql_query': sql_query,
        }

        self.current_table = str(table_name)
        return ast_node

    # ------------------------------------------------------------------
    # save transformer methods
    # ------------------------------------------------------------------

    def save_statement(self, *args):
        """Handle: save "name" [params]"""
        pkg_name = str(args[0])
        params_list = args[1] if len(args) > 1 else []

        path = None
        fmt = None
        chart_fmt = None
        include = None
        exclude = []

        for item in (params_list or []):
            if not isinstance(item, dict):
                continue
            key = item.get('key')
            if key == 'path':
                path = item['value']
            elif key == 'format':
                fmt = item['value']
            elif key == 'chart_format':
                chart_fmt = item['value']
            elif key == 'include':
                include = item['value']
            elif key == 'exclude':
                exclude = item['value']

        return {
            'type': 'save',
            'name': pkg_name,
            'path': path,
            'format': fmt,
            'chart_format': chart_fmt,
            'include': include,
            'exclude': exclude,
        }

    def save_params(self, *params):
        return list(params)

    def save_path(self, val):
        if isinstance(val, dict) and val.get('type') == 'var':
            return {'key': 'path', 'value': val}
        return {'key': 'path', 'value': str(val)}

    def save_format(self, val):
        return {'key': 'format', 'value': str(val)}

    def save_chart_format(self, val):
        return {'key': 'chart_format', 'value': str(val)}

    def save_include(self, id_list):
        return {'key': 'include', 'value': id_list}

    def save_exclude(self, id_list):
        return {'key': 'exclude', 'value': id_list}

    def save_id_list(self, *ids):
        return [str(i) for i in ids]
    
    def dataframe_statement(self, *args):
        """Handle table statements with optional 'from' clause"""
        
        # Filter out the keyword if it's passed (it might be the first arg)
        # And filter out _NL if it's passed
        
        clean_args = []
        for arg in args:
            # We used to filter 'df', 'table' here, but that caused issues
            # when the table name itself was 'df'.
            # Since these are anonymous terminals in the grammar, they shouldn't appear in args anyway.
            if isinstance(arg, Token) and arg.type == '_NL':
                continue
            clean_args.append(arg)
            
        table_name = clean_args[0]
        copy_table = clean_args[1] if len(clean_args) > 1 else None
        
        table_name_str = str(table_name)

        if table_name_str.lower() in PIVOTAL_KEYWORDS:
            raise ValueError(
                f"'{table_name_str}' is a Pivotal reserved keyword and cannot be used as a table name."
            )

        if copy_table is not None:
            # Case: table new_table from existing_table
            copy_table_str = str(copy_table)
            ast_node = {
                'type': 'copy_table',
                'table_name': table_name_str,
                'copy_from': copy_table_str
            }
            
    
        else:
            # Case: table existing_table (just validate it exists)
            ast_node = {
                'type': 'validate_table',
                'table_name': table_name_str
            }
            
        self.current_table = table_name_str
        
        return ast_node
    
    def assign_statement(self, target, expression, condition_list=None):
        """Handle assign statements to create new columns with optional where clause"""
        target_str = str(target)
        expr_str = str(expression)

        if target_str.lower() in PIVOTAL_KEYWORDS:
            raise ValueError(
                f"'{target_str}' is a Pivotal reserved keyword and cannot be used as a column name."
            )
        
        # Check if there's a where clause
        conditions = []
        operators = []
        has_where = False
        
        if condition_list:
            has_where = True
            temp = self._build_conditional_statement(condition_list)
            conditions = temp['ast']['conditions']
            operators = temp['ast']['operators']
            query_str = temp['query_str']
                                                    
        
        ast_node = {
            'type': 'assign',
            'table_name': self.current_table,
            'target': target_str,
            'expression': expr_str,
            'conditions': conditions if has_where else None,
            'operators': operators if has_where else None
        }
        
        if has_where:
            python_code = f"condition = {self.current_table}.eval('{query_str}')\n"
            python_code += f"{self.current_table}.loc[condition, '{target_str}'] = {self.current_table}.eval('{expr_str}')[condition]"
        else:
            python_code = f"{self.current_table}['{target_str}'] = {self.current_table}.eval('{expr_str}')"
        
        return ast_node
    
    def _build_conditional_statement(self, condition_list):
            """
            Normalize and build a query string and AST for a list of condition/operator items.
            Lark sometimes passes a single list wrapped inside a tuple (e.g. ( [cond, 'and', cond], )),
            so unwrap that case here so the rest of the logic can iterate over the actual items.
            """
            if isinstance(condition_list, Tree):
                condition_list = condition_list.children 
            
            # Unwrap if we have a single-element tuple containing the actual list
            if isinstance(condition_list, tuple) and len(condition_list) == 1:
                condition_list = condition_list[0]

            conditions = []
            operators = []
            query_parts = []

            i = 0
            while i < len(condition_list):
                item = condition_list[i]
                
                # If it's a condition dict, process it
                if isinstance(item, dict) and 'column' in item:
                    condition = item
                    column = condition['column']
                    comparator = condition['comparator']
                    value = condition['value']
                    
                    conditions.append(condition)
                    
                    # Build query string part
                    if comparator in ['in', 'not in']:
                        # Handle list values for 'in' and 'not in'
                        if isinstance(value, list):
                            value_str = str(value)
                        else:
                            value_str = f"[{value}]"
                        query_parts.append(f"{column} {comparator} {value_str}")
                    elif isinstance(value, str):
                        query_parts.append(f"{column} {comparator} '{value}'")
                    else:
                        query_parts.append(f"{column} {comparator} {value}")
                    
                    i += 1
                # If it's an operator string (and/or), add it
                elif isinstance(item, str) and item.lower() in ['and', 'or']:
                    operators.append(item.lower())
                    query_parts.append(item.lower())
                    i += 1
                else:
                    i += 1

            # Join all parts into a single query string
            query_str = ' '.join(query_parts)

            return {
                'query_str': query_str,
                'ast': {
                    'conditions': conditions,
                    'operators': operators
                }
            }
    
    def target(self, identifier):
        return str(identifier)
    
    def expression(self, expr):
        return str(expr)
    
    def filter_statement(self, condition_list):
        """Handle filter statements with conditions"""
        # Debug: Check if condition_list is a Tree object
        
        temp = self._build_conditional_statement(condition_list)
        conditions = temp['ast']['conditions']
        operators = temp['ast']['operators']
        query_str = temp['query_str']
        
        ast_node = {
            'type': 'filter',
            'table_name': self.current_table,
            'conditions': conditions,
            'operators': operators
        }
        
        python_code = f"{self.current_table} = {self.current_table}.query('{query_str}')"
        
        return ast_node
    
    def select_statement(self, *items):
        """Handle select statements to select specific columns"""
        columns = []
        renames = {}
        
        for item in items:
            if isinstance(item, dict):
                if item.get('type') == 'var':
                    columns.append(item)
                else:
                    col = item['column']
                    columns.append(col)
                    if 'alias' in item:
                        renames[col] = item['alias']
            else:
                # Fallback if item is just a string (shouldn't happen with new grammar)
                columns.append(str(item))
        
        ast_node = {
            'type': 'select',
            'table_name': self.current_table,
            'columns': columns,
            'renames': renames
        }
        
        return ast_node

    def select_item(self, col, alias=None):
        if isinstance(col, dict) and col.get('type') == 'var':
             # Variable reference cannot have alias in this simple implementation
             # or we could support it if it's a single column
             return col
        
        if alias:
            return {'column': str(col), 'alias': str(alias)}
        return {'column': str(col)}
    
    def PYTHON_VAR(self, token):
        return {'type': 'var', 'name': str(token)[1:]}

    def drop_statement(self, *cols):
        return {
            'type': 'drop',
            'table_name': self.current_table,
            'columns': [str(c) for c in cols]
        }

    def fillna_statement(self, val):
        return {
            'type': 'fillna',
            'table_name': self.current_table,
            'value': val
        }

    def dropna_statement(self, *args):
        cols = args[0] if args and isinstance(args[0], list) else []
        return {
            'type': 'dropna',
            'table_name': self.current_table,
            'columns': cols
        }

    def dropna_cols(self, *cols):
        return [str(c) for c in cols]

    def distinct_statement(self, *args):
        cols = args[0] if args and isinstance(args[0], list) else []
        return {
            'type': 'distinct',
            'table_name': self.current_table,
            'columns': cols
        }

    def distinct_cols(self, *cols):
        return [str(c) for c in cols]

    def concat_statement(self, *tables):
        return {
            'type': 'concat',
            'table_name': self.current_table,
            'tables': [str(t) for t in tables]
        }

    def rename_statement(self, *items):
        renames = {}
        for item in items:
            if isinstance(item, dict):
                renames.update(item)
        return {
            'type': 'rename',
            'table_name': self.current_table,
            'renames': renames
        }

    def rename_item(self, old, new):
        return {str(old): str(new)}

    def apply_statement(self, func):
        return {
            'type': 'apply',
            'table_name': self.current_table,
            'func': str(func)
        }

    def merge_statement(self, *args):
        """Handle merge statements"""
        
        # First arg is always merge_type (Token or None)
        merge_type = args[0] if args and isinstance(args[0], Token) else 'inner'
        
        # Second arg is always right_table (Token)
        right_table = args[1] if len(args) > 1 else None
        
        # Remaining args are optional: keys (Tree with data='keys') or params (list)
        keys = None
        params = None
        
        for arg in args[2:]:
            if isinstance(arg, Tree) and arg.data == 'keys':
                keys = arg
            elif isinstance(arg, list):
                params = arg
        
        if keys:
            keys = keys.children
            key_list = [str(col) for col in keys]
        else:
            key_list = ''
        
        if params:
            kwargs, kwargs_str = self._keyword_arg(params)
        else:
            kwargs = ''
            kwargs_str = ''
        
        ast_node = {
            'type' : 'merge',
            'how': str(merge_type),
            'table_name': self.current_table,
            'right_table': str(right_table),
            'keys': key_list,
            'kwargs': kwargs,
            'kwargs_str': kwargs_str
        }
        
        return ast_node
   
    
    def pivot_statement(self, *args):
        """Handle pivot statements to create pivot tables"""
        row_columns = []
        col_columns = []
        agg_list = []
        
        pivot_args_result = args[0]
        
        for arg in pivot_args_result:
            if isinstance(arg, list):
                # This is likely agg_clause result (list of dicts)
                for item in arg:
                    if isinstance(item, dict) and 'type' not in item:
                         agg_list.append(item)
            elif isinstance(arg, dict):
                if arg.get('type') == 'rows':
                    row_columns = arg['columns']
                elif arg.get('type') == 'cols':
                    col_columns = arg['columns']
        
        ast_node = {
            'type': 'pivot',
            'table_name': self.current_table,
            'index': row_columns,
            'columns': col_columns,
            'agg_list': agg_list
        }
        
        return ast_node

    def pivot_args(self, *args):
        return list(args)
    
    def pivot_rows(self, *columns):
        """Handle pivot rows specification"""
        cols = []
        for col in columns:
            if isinstance(col, dict) and col.get('type') == 'var':
                cols.append(col)
            else:
                cols.append(str(col))
        return {'type': 'rows', 'columns': cols}
    
    def pivot_cols(self, *columns):
        """Handle pivot columns specification"""
        cols = []
        for col in columns:
            if isinstance(col, dict) and col.get('type') == 'var':
                cols.append(col)
            else:
                cols.append(str(col))
        return {'type': 'cols', 'columns': cols}

    def groupby_statement(self, *args):
        """Handle groupby statements"""
        # args[0] is group_cols (list of strings)
        group_cols = args[0]
        
        agg_list = []
        
        # Search for agg_clause result in args
        for arg in args[1:]:
            if isinstance(arg, list):
                # Check if it's a list of dicts (agg_clause result)
                is_agg_list = True
                for item in arg:
                    if not isinstance(item, dict):
                        is_agg_list = False
                        break
                
                if is_agg_list and len(arg) > 0:
                    agg_list.extend(arg)
        
        ast_node = {
            'type': 'groupby',
            'table_name': self.current_table,
            'by': group_cols,
            'agg_list': agg_list
        }
        
        return ast_node

    def group_cols(self, *columns):
        cols = []
        for col in columns:
            if isinstance(col, dict) and col.get('type') == 'var':
                cols.append(col)
            else:
                cols.append(str(col))
        return cols

    def agg_clause(self, *items):
        # Filter out tokens like _NL
        return [item for item in items if isinstance(item, dict)]

    def agg_item(self, func, col, alias=None):
        if isinstance(col, dict) and col.get('type') == 'var':
            res = {'column': col, 'func': str(func)}
        else:
            res = {'column': str(col), 'func': str(func)}
            
        if alias:
            res['alias'] = str(alias)
        return res
    
    def agg_s(self, *functions):
        """Handle aggregation functions"""
        return [str(func) for func in functions]
    
    def AGG_FUNCTION(self, token):
        """Handle aggregation function tokens"""
        return str(token)
    
    def python_statement(self, code):
        raw = str(code).strip()
        blocks = getattr(self, '_python_blocks', {})
        resolved = blocks.get(raw, raw)
        return {
            'type': 'python',
            'code': resolved,
            'table_name': self.current_table
        }

    def plot_statement(self, *args):
        name = None
        kind = None
        kwargs = {}
        kwargs_str = ""

        identifiers = []
        for arg in args:
            if isinstance(arg, Token) and arg.type != '_NL':
                identifiers.append(str(arg))
            elif isinstance(arg, str):
                identifiers.append(arg)
            elif isinstance(arg, list):
                # plot_params returns list of dicts with key/value/label
                kwargs, kwargs_str = self._plot_kwargs(arg)

        if len(identifiers) == 1:
            name = identifiers[0]
        elif len(identifiers) >= 2:
            kind = identifiers[0]
            name = identifiers[1]

        # Extract structural params so they don't get forwarded to df.plot()
        by_col = kwargs.pop('by', None)
        n_cols = kwargs.pop('cols', None)
        style = kwargs.pop('style', None)

        # Rebuild kwargs_str if any structural params were removed
        if any(p is not None for p in [by_col, n_cols, style]):
            parts = []
            for k, v in kwargs.items():
                if isinstance(v, dict) and v.get('type') == 'var':
                    parts.append(f"{k}={v['name']}")
                elif isinstance(v, str):
                    parts.append(f"{k}={repr(v)}")
                else:
                    parts.append(f"{k}={v}")
            kwargs_str = ', '.join(parts)

        return {
            'type': 'plot',
            'table_name': self.current_table,
            'name': name,
            'kind': kind,
            'kwargs': kwargs,
            'kwargs_str': kwargs_str,
            'by': by_col,
            'cols': n_cols,
            'style': style,
        }

    def plot_params(self, *params):
        return list(params)

    def plot_by_param(self, col):
        return {'key': 'by', 'value': str(col), 'label': None}

    def plot_cols_param(self, val):
        return {'key': 'cols', 'value': self._convert_value(val), 'label': None}

    def plot_labeled_param(self, key, val, label):
        return {'key': str(key), 'value': self._convert_value(val), 'label': str(label).strip('"').strip("'")}

    def plot_value_param(self, key, val):
        return {'key': str(key), 'value': self._convert_value(val), 'label': None}

    def plot_list_param(self, key, val):
        return {'key': str(key), 'value': val if isinstance(val, list) else self._convert_value(val), 'label': None}

    def _plot_kwargs(self, plot_params_list):
        """Build kwargs dict and string from a list of plot_param dicts."""
        kwargs = {}
        label_kwargs = {}

        for p in plot_params_list:
            if not isinstance(p, dict):
                continue
            key = p['key']
            val = p['value']
            label = p.get('label')
            kwargs[key] = val
            if label is not None:
                # x → xlabel, y → ylabel; other keys get a best-effort {key}label
                label_key = {'x': 'xlabel', 'y': 'ylabel'}.get(key, f'{key}label')
                label_kwargs[label_key] = label

        kwargs.update(label_kwargs)

        parts = []
        for k, v in kwargs.items():
            if isinstance(v, dict) and v.get('type') == 'var':
                parts.append(f"{k}={v['name']}")
            elif isinstance(v, str):
                parts.append(f"{k}={repr(v)}")
            else:
                parts.append(f"{k}={v}")
        kwargs_str = ', '.join(parts)
        return kwargs, kwargs_str

    def python_block(self, *lines):
        return "\n".join(lines)

    def python_line(self, code):
        return str(code)

    def condition(self, column, comparator, value):
        """Handle individual filter conditions."""
        return {
            'column': str(column),
            'comparator': str(comparator),
            'value': self._convert_value(value) if not isinstance(value, list) else value
        }

    def condition_in_list(self, column, lst):
        return {'column': str(column), 'comparator': 'in', 'value': lst}

    def condition_in_var(self, column, var):
        return {'column': str(column), 'comparator': 'in', 'value': var}

    def condition_not_in_list(self, column, lst):
        return {'column': str(column), 'comparator': 'not in', 'value': lst}

    def condition_not_in_var(self, column, var):
        return {'column': str(column), 'comparator': 'not in', 'value': var}
    
    def AOR(self, token):
        return str(token)
    
    def COMPARATOR(self, token):
        return str(token)
    
    def params(self, *params):
        return list(params)
    
    def param(self, param_content):
        return param_content
    
    def file_path(self, path):
        return {'file_path': str(path).strip()}
    
    def keyword_arg(self, key, val):
        return {str(key): self._convert_value(val)}
    
    def value(self, val):
        return val
    
    def table_name(self, name):
        return str(name)
    
    def copy_table(self, name):
        return str(name)
    
    #def comparator(self, token):
    #    return str(token)

    def IDENTIFIER(self, token):
        return str(token)
    
    def list_value(self, *items):
        return [self._convert_value(item) for item in items] if items else []

    def PATH(self, token):
        return str(token)
    
    def STRING(self, token):
        # Remove quotes
        return str(token)[1:-1]
    
    def UNQUOTED_STRING(self, token):
        return str(token).strip()
    
    def BOOLEAN(self, token):
        s = str(token).lower()
        return s == 'true'
    
    def NUMBER(self, token):
        s = str(token)
        return int(s) if '.' not in s else float(s)
    
    def _convert_value(self, val):
        """Convert parsed value to appropriate Python type"""
        if isinstance(val, (int, float, list, dict)) or val is None:
            return val
        val_str = str(val)
        # Try to convert to number
        try:
            if '.' in val_str:
                return float(val_str)
            return int(val_str)
        except ValueError:
            # Remove quotes if string
            if val_str.startswith('"') or val_str.startswith("'"):
                return val_str[1:-1]
            return val_str
    

class CodeGenerator:
    """Separate code generation logic from parsing"""
    
    def __init__(self, backend="pandas"):
        self.backend = backend
    
    def generate(self, ast_node):
        """Generate code for an AST node"""
        statement_type = ast_node['type']
        method_name = f"generate_{statement_type}_{self.backend}"
        
        if hasattr(self, method_name):
            return getattr(self, method_name)(ast_node)
        else:
            raise NotImplementedError(f"No generator for {statement_type} with {self.backend} backend")
    
    # Pandas code generators
    def generate_sort_pandas(self, ast_node):
        columns = ast_node['columns']
        ascending = ast_node['ascending']
        
        # Check if we have any variable references
        has_vars = any(isinstance(col, dict) and col.get('type') == 'var' for col in columns)
        
        if has_vars:
            # Generate code to construct the column list and ascending list dynamically
            col_list_code = "[]"
            asc_list_code = "[]"
            
            for col, asc in zip(columns, ascending):
                if isinstance(col, dict) and col.get('type') == 'var':
                    var_name = col['name']
                    # Handle both list and single item
                    # If it's a list, we assume ascending applies to all (or default True)
                    # If the user specified 'desc' for a list variable, we apply it to all
                    asc_val = str(asc)
                    col_list_code += f" + ({var_name} if isinstance({var_name}, list) else [{var_name}])"
                    asc_list_code += f" + ([{asc_val}] * len({var_name}) if isinstance({var_name}, list) else [{asc_val}])"
                else:
                    col_list_code += f" + ['{col}']"
                    asc_list_code += f" + [{asc}]"
            
            return f"{ast_node['table_name']} = {ast_node['table_name']}.sort_values({col_list_code}, ascending={asc_list_code})"
        else:
            return f"{ast_node['table_name']} = {ast_node['table_name']}.sort_values({columns}, ascending={ascending})"
    
    def _reader_for_source(self, source_str):
        """Return the appropriate pandas reader function for a file path string."""
        ext = source_str.rsplit('.', 1)[-1].lower() if '.' in source_str else ''
        if ext in ('xlsx', 'xls'):
            return 'pd.read_excel'
        if ext == 'parquet':
            return 'pd.read_parquet'
        return 'pd.read_csv'

    def generate_load_table_pandas(self, ast_node):
        source = ast_node['source']
        table_name_marker = f"#__pivotal__\n__table_name__ = '{ast_node['table_name']}'\n#__pivotal__"

        if isinstance(source, dict) and source.get('type') == 'var':
            # Runtime format detection for variable file paths
            var = source['name']
            kw = ast_node['kwargs_str']
            tname = ast_node['table_name']
            sql_query = ast_node.get('sql_query') or f"SELECT * FROM {tname}"
            load_table = (
                f"_src = {var}\n"
                f"_ext = _src.rsplit('.', 1)[-1].lower() if '.' in _src else ''\n"
                f"if _ext in ('xlsx', 'xls'):\n"
                f"    {tname} = pd.read_excel(_src{kw})\n"
                f"elif _ext == 'parquet':\n"
                f"    {tname} = pd.read_parquet(_src{kw})\n"
                f"elif _ext in ('sqlite', 'db', 'sqlite3'):\n"
                f"    import sqlite3 as _sqlite3\n"
                f"    with _sqlite3.connect(_src) as _conn:\n"
                f"        {tname} = pd.read_sql({repr(sql_query)}, _conn)\n"
                f"else:\n"
                f"    {tname} = pd.read_csv(_src{kw})"
            )
        else:
            source_str = str(source)
            ext = source_str.rsplit('.', 1)[-1].lower() if '.' in source_str else ''
            if ext in ('sqlite', 'db', 'sqlite3'):
                tname = ast_node['table_name']
                sql_query = ast_node.get('sql_query') or f"SELECT * FROM {tname}"
                load_table = (
                    f"import sqlite3 as _sqlite3\n"
                    f"with _sqlite3.connect('{source_str}') as _conn:\n"
                    f"    {tname} = pd.read_sql({repr(sql_query)}, _conn)"
                )
            else:
                reader = self._reader_for_source(source_str)
                load_table = f"{ast_node['table_name']} = {reader}('{source}'{ast_node['kwargs_str']})"

        kw_set = repr(PIVOTAL_KEYWORDS)
        tname = ast_node['table_name']
        kw_check = (
            f"_kw_cols = [c for c in {tname}.columns if c.lower() in {kw_set}]\n"
            f"if _kw_cols:\n"
            f"    import warnings\n"
            f"    warnings.warn(\n"
            f"        f\"Table '{tname}' has columns that are Pivotal keywords: {{_kw_cols}}. \"\n"
            f"        \"Use a 'python' block to reference them.\",\n"
            f"        UserWarning, stacklevel=2)"
        )
        return f"{load_table}\n{kw_check}\n{table_name_marker}"

    def generate_load_table_polars(self, ast_node):
        source = ast_node['source']
        table_name_marker = f"#__pivotal__\n__table_name__ = '{ast_node['table_name']}'\n#__pivotal__"

        if isinstance(source, dict) and source.get('type') == 'var':
            source_code = source['name']
            load_table = f"{ast_node['table_name']} = pl.read_csv({source_code}{ast_node['kwargs_str']})"
        else:
            ext = str(source).rsplit('.', 1)[-1].lower() if '.' in str(source) else ''
            if ext in ('xlsx', 'xls'):
                reader = 'pl.read_excel'
            elif ext == 'parquet':
                reader = 'pl.read_parquet'
            else:
                reader = 'pl.read_csv'
            load_table = f"{ast_node['table_name']} = {reader}('{source}'{ast_node['kwargs_str']})"

        return f"{load_table}\n{table_name_marker}"

    def generate_copy_table_pandas(self, ast_node):
        # Validation + copy
        # We check if the source table exists and is a dataframe
        validation = f"if '{ast_node['copy_from']}' not in locals() and '{ast_node['copy_from']}' not in globals(): raise NameError(\"name '{ast_node['copy_from']}' is not defined\")"
        validation += f"\nif not isinstance({ast_node['copy_from']}, pd.DataFrame): raise TypeError('{ast_node['copy_from']} is not a pandas DataFrame')"
        
        copy_code = f"{ast_node['table_name']} = {ast_node['copy_from']}.copy()"
        table_name = f"#__pivotal__\n__table_name__ = '{ast_node['table_name']}'\n#__pivotal__"
        return f"{validation}\n{copy_code}\n{table_name}"
    
    def generate_validate_table_pandas(self, ast_node):
        # Just set the table name for tracking, don't validate existence yet as it might be created in this block
        # Actually, validate_table is used when we say "df existing_table"
        # So we should check if it exists in globals/locals
        
        validation = f"if '{ast_node['table_name']}' not in locals() and '{ast_node['table_name']}' not in globals(): raise NameError(\"name '{ast_node['table_name']}' is not defined\")"
        validation += f"\nif not isinstance({ast_node['table_name']}, pd.DataFrame): raise TypeError('{ast_node['table_name']} is not a pandas DataFrame')"
        
        table_name = f"#__pivotal__\n__table_name__ = '{ast_node['table_name']}'\n#__pivotal__"
        return f"{validation}\n{table_name}"
    
    # Built-in function names reserved for future string function support.
    # User-defined functions must not use these names.
    _BUILTIN_FUNCS = frozenset({
        'upper', 'lower', 'trim', 'ltrim', 'rtrim',
        'left', 'right', 'substr', 'len', 'replace',
    })

    def _parse_user_func_call(self, expr):
        """If expr matches 'func(col)' and func is not a built-in, return (func, col).
        Otherwise return None."""
        import re
        m = re.fullmatch(r'([a-zA-Z][a-zA-Z0-9_]*)\(([a-zA-Z][a-zA-Z0-9_]*)\)', expr.strip())
        if m and m.group(1) not in self._BUILTIN_FUNCS:
            return m.group(1), m.group(2)
        return None

    @staticmethod
    def _is_scalar_expr(expr):
        """Return True if expr is a literal scalar (number, quoted string, bool, None)."""
        import re
        s = expr.strip()
        if re.match(r'^\d+(\.\d+)?$', s):
            return True
        if re.match(r'^"[^"]*"$', s) or re.match(r"^'[^']*'$", s):
            return True
        if s in ('True', 'False', 'true', 'false', 'None', 'none'):
            return True
        return False

    def generate_assign_pandas(self, ast_node):
        table = ast_node['table_name']
        target = ast_node['target']
        expr = ast_node['expression']

        user_call = self._parse_user_func_call(expr)

        if ast_node['conditions']:
            query_str, _ = self._build_query_string(ast_node['conditions'], ast_node['operators'])
            if user_call:
                func, col = user_call
                return (f"condition = {table}.eval('{query_str}')\n"
                        f"{table}.loc[condition, '{target}'] = "
                        f"{func}({table}['{col}'])[condition]")
            if self._is_scalar_expr(expr):
                rhs = expr
            else:
                rhs = f"{table}.eval('{expr}')[condition]"
            return (f"condition = {table}.eval('{query_str}')\n"
                    f"{table}.loc[condition, '{target}'] = {rhs}")
        else:
            if user_call:
                func, col = user_call
                return f"{table}['{target}'] = {func}({table}['{col}'])"
            return f"{table}['{target}'] = {table}.eval('{expr}')"

    def generate_apply_pandas(self, ast_node):
        table = ast_node['table_name']
        func = ast_node['func']
        return f"{table} = {func}({table})"

    def generate_apply_polars(self, ast_node):
        table = ast_node['table_name']
        func = ast_node['func']
        return f"{table} = {func}({table})"
    
    def generate_filter_pandas(self, ast_node):
        query_str, needs_python_engine = self._build_query_string(ast_node['conditions'], ast_node['operators'])
        engine = ", engine='python'" if needs_python_engine else ""
        return f"{ast_node['table_name']} = {ast_node['table_name']}.query('{query_str}'{engine})"
    
    def generate_select_pandas(self, ast_node):
        columns = ast_node['columns']
        renames = ast_node.get('renames', {})
        
        # Check if we have any variable references
        has_vars = any(isinstance(col, dict) and col.get('type') == 'var' for col in columns)
        
        if has_vars:
            # Generate code to construct the column list dynamically
            col_list_code = "[]"
            for col in columns:
                if isinstance(col, dict) and col.get('type') == 'var':
                    var_name = col['name']
                    # Handle both list and single item
                    col_list_code += f" + ({var_name} if isinstance({var_name}, list) else [{var_name}])"
                else:
                    col_list_code += f" + ['{col}']"
            
            code = f"{ast_node['table_name']} = {ast_node['table_name']}.loc[:, {col_list_code}]"
        else:
            code = f"{ast_node['table_name']} = {ast_node['table_name']}.loc[:, {columns}]"
            
        if renames:
            code += f".rename(columns={renames})"
        return code
    
    def generate_merge_pandas(self, ast_node):
        if ast_node['keys'] == '':
            return f"{ast_node['table_name']} = {ast_node['table_name']}.merge({ast_node['right_table']}, how='{ast_node['how']}'{ast_node['kwargs_str']})"
        else:
            return f"{ast_node['table_name']} = {ast_node['table_name']}.merge({ast_node['right_table']}, on={ast_node['keys']}, how='{ast_node['how']}', {ast_node['kwargs_str']})"
    
    def generate_pivot_pandas(self, ast_node):
        """Generate pandas pivot_table code"""
        table_name = ast_node['table_name']
        index = ast_node['index']
        columns = ast_node['columns']
        agg_list = ast_node.get('agg_list', [])
        
        # Helper to process list/var/string arguments
        def process_arg(arg):
            if not arg:
                return None
            
            # Check if it's a variable reference
            if isinstance(arg, dict) and arg.get('type') == 'var':
                return arg['name']
            
            # Check if it's a list containing variable references
            if isinstance(arg, list):
                has_vars = any(isinstance(item, dict) and item.get('type') == 'var' for item in arg)
                if has_vars:
                    code = "[]"
                    for item in arg:
                        if isinstance(item, dict) and item.get('type') == 'var':
                            var_name = item['name']
                            code += f" + ({var_name} if isinstance({var_name}, list) else [{var_name}])"
                        else:
                            code += f" + ['{item}']"
                    return code
                elif len(arg) > 1:
                    return str(arg)
                elif len(arg) == 1:
                    return f"'{arg[0]}'"
                else:
                    return None
            
            return f"'{arg}'"

        index_str = process_arg(index)
        columns_str = process_arg(columns)
        
        # Process agg_list to build values and aggfunc
        has_vars_in_agg = any(isinstance(item['column'], dict) and item['column'].get('type') == 'var' for item in agg_list)
        
        code_lines = []
        
        if has_vars_in_agg:
            code_lines.append("_aggfunc = {}")
            for item in agg_list:
                col = item['column']
                func = item['func']
                if isinstance(col, dict) and col.get('type') == 'var':
                    var_name = col['name']
                    code_lines.append(f"_cols = {var_name} if isinstance({var_name}, list) else [{var_name}]")
                    code_lines.append(f"for c in _cols: _aggfunc[c] = '{func}'")
                else:
                    code_lines.append(f"_aggfunc['{col}'] = '{func}'")
            
            aggfunc_str = "_aggfunc"
            values_str = "list(_aggfunc.keys())"
            
        else:
             agg_dict = {}
             for item in agg_list:
                col = item['column']
                func = item['func']
                if col not in agg_dict:
                    agg_dict[col] = []
                agg_dict[col].append(func)
            
             for k, v in agg_dict.items():
                if len(v) == 1:
                    agg_dict[k] = v[0]
             
             values = list(agg_dict.keys())
             values_str = str(values) if len(values) > 1 else f"'{values[0]}'"
             aggfunc_str = str(agg_dict)

        # Build pivot_table call
        pivot_args = []
        pivot_args.append(f"values={values_str}")
        
        if index_str:
            pivot_args.append(f"index={index_str}")
        
        if columns_str:
            pivot_args.append(f"columns={columns_str}")
            
        pivot_args.append(f"aggfunc={aggfunc_str}")
        
        pivot_call = f"{table_name} = pd.pivot_table({table_name}, {', '.join(pivot_args)}).reset_index()"

        if has_vars_in_agg:
            return "\n".join(code_lines + [pivot_call])
        else:
            return pivot_call

    def generate_groupby_pandas(self, ast_node):
        by = ast_node['by']
        agg_list = ast_node.get('agg_list', [])
        
        # Handle 'by' argument which can be a list, a variable, or a list containing variables
        if isinstance(by, dict) and by.get('type') == 'var':
            by_code = by['name']
        elif isinstance(by, list):
            has_vars = any(isinstance(item, dict) and item.get('type') == 'var' for item in by)
            if has_vars:
                by_code = "[]"
                for item in by:
                    if isinstance(item, dict) and item.get('type') == 'var':
                        var_name = item['name']
                        by_code += f" + ({var_name} if isinstance({var_name}, list) else [{var_name}])"
                    else:
                        by_code += f" + ['{item}']"
            else:
                by_code = str(by)
        else:
            by_code = f"'{by}'"

        if agg_list:
            # Check if any aliases exist
            has_aliases = any('alias' in item for item in agg_list)
            
            if has_aliases:
                # Use named aggregation syntax
                # agg(alias=('col', 'func'))
                agg_args = []
                for item in agg_list:
                    col = item['column']
                    func = item['func']
                    alias = item.get('alias', None)
                    
                    if isinstance(col, dict) and col.get('type') == 'var':
                        col_code = col['name']
                        if not alias:
                             alias = f"agg_{func}" 
                    else:
                        col_code = f"'{col}'"
                        if not alias:
                            alias = f"{col}_{func}"
                    
                    agg_args.append(f"{alias}=({col_code}, '{func}')")
                
                agg_str = ", ".join(agg_args)
                return f"{ast_node['table_name']} = {ast_node['table_name']}.groupby({by_code}).agg({agg_str}).reset_index()"
            else:
                # Old style dict aggregation
                has_vars_in_agg = any(isinstance(item['column'], dict) and item['column'].get('type') == 'var' for item in agg_list)
                
                if has_vars_in_agg:
                    code_lines = []
                    code_lines.append("_agg_dict = {}")
                    for item in agg_list:
                        col = item['column']
                        func = item['func']
                        if isinstance(col, dict) and col.get('type') == 'var':
                            var_name = col['name']
                            code_lines.append(f"_cols = {var_name} if isinstance({var_name}, list) else [{var_name}]")
                            code_lines.append(f"for c in _cols: _agg_dict[c] = '{func}'")
                        else:
                            code_lines.append(f"_agg_dict['{col}'] = '{func}'")
                    
                    agg_dict_str = "_agg_dict"
                    groupby_call = f"{ast_node['table_name']} = {ast_node['table_name']}.groupby({by_code}).agg({agg_dict_str}).reset_index()"
                    return "\n".join(code_lines + [groupby_call])
                else:
                    agg_dict = {}
                    for item in agg_list:
                        agg_dict[item['column']] = item['func']
                    return f"{ast_node['table_name']} = {ast_node['table_name']}.groupby({by_code}).agg({agg_dict}).reset_index()"
        else:
            return f"{ast_node['table_name']} = {ast_node['table_name']}.groupby({by_code}).sum().reset_index()"
    
    def generate_drop_pandas(self, ast_node):
        return f"{ast_node['table_name']} = {ast_node['table_name']}.drop(columns={ast_node['columns']})"

    def generate_fillna_pandas(self, ast_node):
        val = ast_node['value']
        val_code = f"'{val}'" if isinstance(val, str) else str(val)
        return f"{ast_node['table_name']} = {ast_node['table_name']}.fillna({val_code})"

    def generate_dropna_pandas(self, ast_node):
        cols = ast_node['columns']
        if cols:
            return f"{ast_node['table_name']} = {ast_node['table_name']}.dropna(subset={cols})"
        return f"{ast_node['table_name']} = {ast_node['table_name']}.dropna()"

    def generate_distinct_pandas(self, ast_node):
        cols = ast_node['columns']
        if cols:
            return f"{ast_node['table_name']} = {ast_node['table_name']}.drop_duplicates(subset={cols})"
        return f"{ast_node['table_name']} = {ast_node['table_name']}.drop_duplicates()"

    def generate_concat_pandas(self, ast_node):
        others = ', '.join(ast_node['tables'])
        return f"{ast_node['table_name']} = pd.concat([{ast_node['table_name']}, {others}], ignore_index=True)"

    def generate_rename_pandas(self, ast_node):
        return f"{ast_node['table_name']} = {ast_node['table_name']}.rename(columns={ast_node['renames']})"

    def generate_python_pandas(self, ast_node):
        return ast_node['code']

    def generate_plot_pandas(self, ast_node):
        kind = ast_node['kind']
        kwargs_str = ast_node['kwargs_str']
        table = ast_node['table_name']
        chart_key = ast_node['name']
        by_col = ast_node.get('by')
        n_cols = int(ast_node.get('cols') or 2)
        style = ast_node.get('style')

        args_str = ""
        if kind:
            args_str += f"kind='{kind}'"
        if kwargs_str:
            args_str = f"{args_str}, {kwargs_str}" if args_str else kwargs_str

        lines = ["import matplotlib.pyplot as plt"]

        # Style file: look for <name>.mplstyle locally, then styles/<name>.mplstyle,
        # otherwise pass the name directly to plt.style.use() for built-in styles.
        # Custom keys not supported by matplotlib (e.g. xtick.labelrotation) are
        # extracted and applied manually after plotting.
        _CUSTOM_STYLE_KEYS = frozenset((
            'xtick.labelrotation', 'ytick.labelrotation',
            'xtick.labelalignment', 'ytick.labelalignment',
        ))
        if style:
            lines += [
                f"_style_candidates = [{repr(style + '.mplstyle')}, {repr('styles/' + style + '.mplstyle')}]",
                f"_style_path = next((_p for _p in _style_candidates if __import__('os').path.exists(_p)), {repr(style)})",
                f"_custom_style = {{}}",
                f"_PKEYS = {_CUSTOM_STYLE_KEYS!r}",
                f"_custom_style.update({{_k.strip(): _v.strip() for _ln in (open(_style_path).readlines() if __import__('os').path.isfile(_style_path) else []) if (_ln.strip() and not _ln.strip().startswith('#') and ':' in _ln) for _k, _, _v in [_ln.partition(':')]  if _k.strip() in _PKEYS}})",
                f"import warnings as _warnings",
                f"_warnings.filterwarnings('ignore', message='Bad key')",
                f"plt.style.use(_style_path)",
                f"_warnings.filterwarnings('default', message='Bad key')",
            ]

        if not by_col:
            # Simple plot — existing behaviour
            lines += [
                f"_ax = {table}.plot({args_str})",
                f"if '_pivotal_charts' not in globals(): globals()['_pivotal_charts'] = {{}}",
                f"globals()['_pivotal_charts'][{repr(chart_key)}] = {{'fig': _ax.get_figure(), 'data': {table}.copy()}}",
                f"{chart_key} = _ax.get_figure()",
                f"plt.close({chart_key})",  # suppress inline cell output (viewer is the display)
            ]
        else:
            # Faceted subplots: one per unique value of by_col
            lines += [
                f"_by_vals = {table}[{repr(by_col)}].unique()",
                f"_n_cols = {n_cols}",
                f"_n_rows = -(-len(_by_vals) // _n_cols)",
                f"_fig, _axes = plt.subplots(_n_rows, _n_cols, figsize=(7 * _n_cols, 5 * _n_rows))",
                f"_axes = _axes.flatten() if hasattr(_axes, 'flatten') else [_axes]",
                f"for _i, _val in enumerate(_by_vals):",
                f"    {table}[{table}[{repr(by_col)}] == _val].plot({args_str}, ax=_axes[_i], title=str(_val))",
                f"for _ax in _axes[len(_by_vals):]:",
                f"    _ax.set_visible(False)",
                f"plt.tight_layout()",
                f"if '_pivotal_charts' not in globals(): globals()['_pivotal_charts'] = {{}}",
                f"globals()['_pivotal_charts'][{repr(chart_key)}] = {{'fig': _fig, 'data': {table}.copy()}}",
                f"{chart_key} = _fig",
                f"plt.close({chart_key})",  # suppress inline cell output (viewer is the display)
            ]

        # Apply custom style keys that matplotlib doesn't support natively
        if style:
            lines += [
                f"for _a in plt.gcf().get_axes():",
                f"    if 'xtick.labelrotation' in _custom_style: _a.tick_params(axis='x', labelrotation=float(_custom_style['xtick.labelrotation']))",
                f"    if 'ytick.labelrotation' in _custom_style: _a.tick_params(axis='y', labelrotation=float(_custom_style['ytick.labelrotation']))",
                f"    if 'xtick.labelalignment' in _custom_style: plt.setp(_a.xaxis.get_majorticklabels(), ha=_custom_style['xtick.labelalignment'])",
                f"    if 'ytick.labelalignment' in _custom_style: plt.setp(_a.yaxis.get_majorticklabels(), ha=_custom_style['ytick.labelalignment'])",
            ]

        return "\n".join(lines)

    def _build_query_string(self, conditions, operators):
        """Build query string from conditions and operators.

        Returns:
            (query_str, needs_python_engine) — the second flag signals that
            pandas must use engine='python' (e.g. for str accessor methods).
        """
        query_parts = []
        needs_python_engine = False

        for i, condition in enumerate(conditions):
            column = condition['column']
            comparator = condition['comparator']
            value = condition['value']

            if comparator == 'between':
                lo, hi = value
                query_parts.append(f"{column} >= {lo} and {column} <= {hi}")
            elif comparator == 'contains':
                query_parts.append(f'{column}.str.contains("{value}")')
                needs_python_engine = True
            elif comparator == 'not contains':
                query_parts.append(f'not {column}.str.contains("{value}")')
                needs_python_engine = True
            elif comparator == 'startswith':
                query_parts.append(f'{column}.str.startswith("{value}")')
                needs_python_engine = True
            elif comparator == 'endswith':
                query_parts.append(f'{column}.str.endswith("{value}")')
                needs_python_engine = True
            elif isinstance(value, dict) and value.get('type') == 'var':
                value_str = f"@{value['name']}"
                query_parts.append(f"{column} {comparator} {value_str}")
            elif comparator in ['in', 'not in']:
                if isinstance(value, list):
                    # Use double-quoted strings so they don't clash with the
                    # outer single-quoted .query('...') call.
                    items = ', '.join(f'"{v}"' if isinstance(v, str) else str(v) for v in value)
                    value_str = f"[{items}]"
                else:
                    value_str = f'["{value}"]' if isinstance(value, str) else f"[{value}]"
                query_parts.append(f"{column} {comparator} {value_str}")
            elif isinstance(value, str):
                query_parts.append(f'{column} {comparator} "{value}"')
            else:
                query_parts.append(f"{column} {comparator} {value}")

            if i < len(operators):
                query_parts.append(operators[i])

        return ' '.join(query_parts), needs_python_engine
    
    # Future: Add SQL generators
    def generate_sort_sql(self, ast_node):
        order_clause = ', '.join([f"{col} {'ASC' if asc else 'DESC'}" 
                                 for col, asc in zip(ast_node['columns'], ast_node['ascending'])])
        return f"SELECT * FROM {ast_node['table_name']} ORDER BY {order_clause}"
    
    # Future: Add Spark generators (commented out to avoid import issues)
    # def generate_sort_spark(self, ast_node):
    #     from pyspark.sql import functions as F
    #     order_cols = [F.col(col).asc() if asc else F.col(col).desc()
    #                  for col, asc in zip(ast_node['columns'], ast_node['ascending'])]
    #     return f"{ast_node['table_name']} = {ast_node['table_name']}.orderBy({order_cols})"

    # ------------------------------------------------------------------
    # save / load_all / load_package_table generators
    # ------------------------------------------------------------------

    def generate_save_pandas(self, ast_node):
        name = ast_node['name']
        path = ast_node.get('path')
        fmt = ast_node.get('format') or 'csv'
        chart_fmt = ast_node.get('chart_format') or 'png'
        include = ast_node.get('include')
        exclude = ast_node.get('exclude') or []

        if isinstance(path, dict) and path.get('type') == 'var':
            path_arg = f", path={path['name']}"
        elif path:
            path_arg = f", path={repr(path)}"
        else:
            path_arg = ""

        include_arg = f", include={repr(include)}" if include is not None else ""
        exclude_arg = f", exclude={repr(exclude)}" if exclude else ""

        return (
            f"from pivotal.package import Package as _PivotalPackage\n"
            f"_PivotalPackage.export({repr(name)}, globals(){path_arg}, fmt={repr(fmt)}"
            f", chart_fmt={repr(chart_fmt)}{include_arg}{exclude_arg})"
        )

    def generate_save_polars(self, ast_node):
        return self.generate_save_pandas(ast_node)

    def generate_load_all_pandas(self, ast_node):
        return (
            f"globals().update(_pivotal_pkg.load_all())\n"
            f"print(f\"Loaded {{len(_pivotal_pkg.load_all())}} table(s) from '{{_pivotal_pkg.name}}'\")"
        )

    def generate_load_all_polars(self, ast_node):
        return self.generate_load_all_pandas(ast_node)

    def generate_load_package_table_pandas(self, ast_node):
        name = ast_node['table_name']
        return f"{name} = _pivotal_pkg.load_table({repr(name)})"

    def generate_load_package_table_polars(self, ast_node):
        return self.generate_load_package_table_pandas(ast_node)


class DSLParser:
    def __init__(self, backend="pandas"):
        self._transformer = DSLTransformer()
        self.parser = Lark(
            grammar_indented,
            parser='lalr',
            postlex=DSLIndenter(),
            transformer=self._transformer
        )
        self.code_generator = CodeGenerator(backend)
        self.autocomplete_file = Path('pivotal_autocomplete.json')
        self.table_info = {}
        
    def update_autocomplete_info(self, globals_dict=None):
        """Update the autocomplete JSON file with current table information"""
        if globals_dict is None:
            globals_dict = {}
            
        table_info = {}
        
        # Scan for pandas DataFrames in the globals
        for name, obj in globals_dict.items():
            if isinstance(obj, pd.DataFrame):
                # Handle nested columns (MultiIndex columns from pivot operations)
                if isinstance(obj.columns, pd.MultiIndex):
                    columns = [list(col) if isinstance(col, tuple) else col for col in obj.columns]
                else:
                    columns = list(obj.columns)
                
                table_info[name] = {
                    'columns': columns,
                    'shape': list(obj.shape),
                    'dtypes': {str(col): str(dtype) for col, dtype in obj.dtypes.items()},
                    'has_multiindex_columns': isinstance(obj.columns, pd.MultiIndex)
                }
        
        # Check if table info has actually changed to avoid unnecessary file I/O
        if table_info == self.table_info:
            return  # No changes, skip file write
            
        # Update our internal tracking
        self.table_info = table_info
        
        # Write to file for VS Code extension only when there are changes
        try:
            with open(self.autocomplete_file, 'w') as f:
                json.dump({
                    'tables': table_info,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'current_table' : globals_dict.get('__table_name__')
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not update autocomplete file: {e}")
    
    def get_table_columns(self, table_name=None):
        """Get columns for a specific table or all tables"""
        if table_name:
            return self.table_info.get(table_name, {}).get('columns', [])
        return self.table_info
        
    @staticmethod
    def _strip_line_comment(line):
        """Remove a trailing -- or # comment from a single line.

        Respects double-quoted strings so that ``--`` or ``#`` inside a string
        literal is preserved.  Returns the line with the comment (and any
        trailing whitespace before it) removed.
        """
        in_string = False
        i = 0
        while i < len(line):
            ch = line[i]
            if in_string:
                if ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == '#':
                return line[:i]
            elif ch == '-' and i + 1 < len(line) and line[i + 1] == '-':
                return line[:i]
            i += 1
        return line

    def preprocess_code(self, code):
        """Preprocess DSL code to handle whitespace issues.

        Single-line comments (``--`` and ``#``) are stripped here rather than
        relying solely on Lark's ``%ignore COMMENT``.  When lark ignores a
        comment token it still leaves the surrounding newline characters in the
        token stream, which can split what should be a single ``_NL`` token
        into two, causing unexpected-token parse errors after indented blocks.
        """
        import re

        # Normalize Windows line endings so regexes always see \n.
        code = code.replace('\r\n', '\n').replace('\r', '\n')

        # Strip multi-line comments (/* ... */) preserving line count.
        def _replace_multiline(m):
            return '\n' * m.group(0).count('\n')
        code = re.sub(r'/\*[\s\S]*?\*/', _replace_multiline, code)

        # Extract python...end blocks before comment stripping so that
        # '#' or '--' inside Python code is not incorrectly removed.
        python_blocks = {}

        def _extract_python_block(m):
            indent = m.group(1)
            content = m.group(2)
            import textwrap
            content = textwrap.dedent(content)
            key = f'__PYBLOCK_{len(python_blocks)}__'
            python_blocks[key] = content
            return f'{indent}python {key}'

        # Match: <indent>python<optional spaces>\n<content>\n<same-indent>end
        code = re.sub(
            r'^([ \t]*)python[ \t]*\n(.*?)\n\1end[ \t]*(?=\n|$)',
            _extract_python_block,
            code,
            flags=re.MULTILINE | re.DOTALL,
        )

        self._transformer._python_blocks = python_blocks

        # Strip single-line comments line-by-line (respects string literals).
        lines = [self._strip_line_comment(ln) for ln in code.split('\n')]
        code = '\n'.join(lines)

        # Strip leading and trailing whitespace
        code = code.strip()

        # Ensure the file ends with a newline if it's not empty
        if code and not code.endswith('\n'):
            code += '\n'

        return code
    
    def parse(self, code):
        """Parse DSL code and return AST + Python code"""
        try:
            # Preprocess the code to handle whitespace issues
            processed_code = self.preprocess_code(code)
            result = self.parser.parse(processed_code)
            return result
        except ValueError:
            raise  # Keyword-collision and other validation errors propagate
        except Exception as e:
            return {'error': str(e)}
    
    def parse_file(self, filepath):
        """Parse a DSL file and return AST + Python code"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            return self.parse(code)
        except FileNotFoundError:
            return {'error': f"File not found: {filepath}"}
        except Exception as e:
            return {'error': f"Error reading file {filepath}: {str(e)}"}

    def generate_code(self, ast_list, backend="pandas"):
        """Generate code for a list of AST nodes"""
        if backend and backend != self.code_generator.backend:
            # Create new generator if backend changed
            self.code_generator = CodeGenerator(backend)
        
        python_code = []
        for ast_node in ast_list:
            code = self.code_generator.generate(ast_node)
            python_code.append(code)
        
        return python_code

    def export(self, code):
        """Parse DSL code and return generated Python code as a string
        
        Args:
            code: DSL code string to parse
            
        Returns:
            String containing all generated Python code, or None if parse error
        """
        results = self.parse(code)

        if isinstance(results, dict) and 'error' in results:
            print(f"Parse error: {results['error']}")
            return None

        code = self.generate_code(results)
        
        # Collect all Python code
        python_lines = ["import pandas as pd", ""]
        
        for c in code:
            
            python_code = c
            
            # Remove lines between #__pivotal__ markers
            lines = python_code.split('\n')
            filtered_lines = []
            skip = False
            
            for line in lines:
                if '#__pivotal__' in line:
                    skip = not skip
                    continue
                if not skip:
                    filtered_lines.append(line)
            
            cleaned_code = '\n'.join(filtered_lines).strip()
            if cleaned_code:
                python_lines.append(cleaned_code)
                python_lines.append("")  # Add blank line between statements
        
        return '\n'.join(python_lines)
   
    def execute(self, code, globals_dict, backend="pandas", verbose=True):
        """Parse and execute the DSL code
        
        Args:
            code: DSL code string to parse and execute
            globals_dict: Namespace to execute in (typically pass globals())
            
        Returns:
            Dictionary of executed table names -> DataFrames
        """
        # Ensure pandas is available in the namespace
        if 'pd' not in globals_dict:
            globals_dict['pd'] = pd
        
        results = self.parse(code)

        if isinstance(results, dict) and 'error' in results:
            print(f"Parse error: {results['error']}")
            return None

        python_code_list = self.generate_code(results, backend=backend)
        
        i = 0 
        for python_code in python_code_list:
            print(f"Executing: {python_code}")
            try:
                exec(python_code, globals_dict)
                table_name = results[i].get('table_name')
                if verbose and table_name and table_name in globals_dict:
                    df = globals_dict[table_name]
                    print(f"\nTable '{table_name}':")
                    print(f"Shape: {df.shape}\n")
                    print(df.head())
            except Exception as e:
                print(f"Execution error: {e}")
            i+=1
        
        # Update autocomplete info after execution
        self.update_autocomplete_info(globals_dict)
        
        # Collect tables
        tables = {}
        for res in results:
            if 'table_name' in res:
                name = res['table_name']
                if name in globals_dict:
                    tables[name] = globals_dict[name]
        return tables
        

# Example usage
if __name__ == "__main__":
    import csv
    import os
    
    # Create dummy data.csv
    print("Creating dummy CSV files...")
    
    with open('data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'product', 'price', 'quantity', 'category'])
        writer.writerow([1, 'Laptop', 999.99, 5, 'Electronics'])
        writer.writerow([2, 'Mouse', 25.50, 150, 'Electronics'])
        writer.writerow([3, 'Desk', 299.00, 20, 'Furniture'])
        writer.writerow([4, 'Chair', 159.99, 45, 'Furniture'])
        writer.writerow([5, 'Monitor', 399.00, 30, 'Electronics'])
        writer.writerow([6, 'Keyboard', 79.99, 80, 'Electronics'])
    
    # Create dummy users.csv
    with open('users.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'name', 'email', 'age', 'city'])
        writer.writerow([101, 'Alice Johnson', 'alice@example.com', 28, 'New York'])
        writer.writerow([102, 'Bob Smith', 'bob@example.com', 34, 'Los Angeles'])
        writer.writerow([103, 'Carol White', 'carol@example.com', 25, 'Chicago'])
        writer.writerow([104, 'David Brown', 'david@example.com', 42, 'Houston'])
        writer.writerow([105, 'Eve Davis', 'eve@example.com', 31, 'Phoenix'])
    
    print("✓ Created data.csv and users.csv\n")
    
    dsl_code = """table testdf from csv
    data.csv
    
table users from csv
    users.csv
"""
    
    parser = DSLParser()
    
    print("=" * 60)
    print("Parsing DSL Code:")
    print("=" * 60)
    print(dsl_code)
    print()
    
    results = parser.parse(dsl_code)
    
    if isinstance(results, dict) and 'error' in results:
        print(f"Error: {results['error']}")
    else:
        for i, result in enumerate(results, 1):
            print(f"\n--- Statement {i} ---")
            print(f"AST: {result['ast']}")
            print(f"Python: {result['python']}")
    
    print("\n" + "=" * 60)
    print("Executing DSL code:")
    print("=" * 60)
    
    tables = parser.execute(dsl_code)
    
    if tables:
        for name, df in tables.items():
            print(f"\nTable '{name}':")
            print(df.to_string())
            print(f"Shape: {df.shape}")
    
    # Clean up
    print("\n" + "=" * 60)
    print("Cleaning up CSV files...")
    os.remove('data.csv')
    os.remove('users.csv')
    print("✓ Removed data.csv and users.csv")