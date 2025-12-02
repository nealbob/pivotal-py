from lark import Lark, Transformer, v_args
from lark.indenter import Indenter
from lark import Tree
from lark.lexer import Token
import pandas as pd
import json
import os
from pathlib import Path

#AGG_DICT: _NL _INDENT IDENTIFIER AGG_FUNCTION ("," AGG_FUNCTION)* _DEDENT (_NL _INDENT IDENTIFIER AGG_FUNCTION ("," AGG_FUNCTION)* _DEDENT)* _NL?
# Grammar definition using indentation
grammar_indented = r"""
    %declare _INDENT _DEDENT
    start: _NL* (_INDENT? statement _DEDENT?)+ _NL*

    statement: load_statement 
               | table_statement  
               | set_statement  
               | filter_statement  
               | select_statement   
               | sort_statement
               | merge_statement
               | pivot_statement
               | groupby_statement

    groupby_statement: "groupby" group_cols (_NL _INDENT agg_clause _DEDENT)? _NL?

    group_cols: IDENTIFIER ("," IDENTIFIER)*

    agg_clause: "agg" agg_item ("," agg_item)* _NL?

    agg_item: IDENTIFIER AGG_FUNCTION

    merge_statement: MERGE_TYPE? ("merge" | "join") RIGHT_TABLE ("on" keys)? (_NL | _NL _INDENT params _DEDENT)?
    
    MERGE_TYPE: "left" | "right" | "inner" | "outer"
    keys: IDENTIFIER ("," IDENTIFIER)*
    RIGHT_TABLE: IDENTIFIER

    load_statement: "load" table_name (STRING | PATH) (_NL | _NL _INDENT params _DEDENT)?

    table_statement: "table" table_name ("from" copy_table)? _NL?

    set_statement: "set" target "=" expression (_NL | _NL _INDENT "where" condition_list _NL _DEDENT)?

    filter_statement: "filter" condition_list  _NL?
    
    select_statement: "select" IDENTIFIER ("," IDENTIFIER)* _NL?

    pivot_statement: "pivot" "on" pivot_values _NL _INDENT pivot_rows _NL pivot_cols (_NL _DEDENT | _NL agg_s _DEDENT) 

    pivot_values: IDENTIFIER ("," IDENTIFIER)* 
    pivot_rows: "rows" IDENTIFIER ("," IDENTIFIER)*   
    pivot_cols: "cols" IDENTIFIER ("," IDENTIFIER)*  
    agg_s:  "agg" AGG_FUNCTION ("," AGG_FUNCTION)* _NL?
   
    AGG_FUNCTION: "mean" | "min" | "max" | "sum" | "count" | "avg" | "median" | "std"

    sort_statement: "sort" IDENTIFIER SORT_TYPE? ("," IDENTIFIER SORT_TYPE?)* _NL?

    SORT_TYPE: "asc" | "desc"
    
    target: IDENTIFIER
    table_name: IDENTIFIER
    copy_table: IDENTIFIER

    expression: UNQUOTED_STRING | STRING

    condition: IDENTIFIER COMPARATOR (value | list_value)

    condition_list: condition (AOR condition)*

    COMPARATOR: "==" | "!=" | ">" | "<" | ">=" | "<=" | "in" | "not in"

    params: param+

    param: keyword_arg _NL?

    keyword_arg: IDENTIFIER value 
                | IDENTIFIER "=" value 
                | IDENTIFIER "=" list_value 
                | IDENTIFIER list_value 

    file_path: PATH

    value: BOOLEAN | NUMBER | STRING | IDENTIFIER | PATH | NONE 
    list_value: "[" [value ("," value)*] "]" |  [value "," value ("," value)*] 

    BOOLEAN.2: "True" | "False" | "true" | "false"
    NONE.2: "None" | "none"
    AOR.2: /and/i | /or/i
    IDENTIFIER: /[a-zA-Z][a-zA-Z0-9_]*/
    IDENT_LIST.2: IDENTIFIER ("," IDENTIFIER)*
    STRING: /"[^"]*"/ | /'[^']*'/
    UNQUOTED_STRING: /[^\n]+/
    PATH: /[a-zA-Z0-9_]*[:\\\/][a-zA-Z0-9_:\/\\\.\-]+|[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+/
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
            if isinstance(arg, Token):
                i += 1
                continue
            
            # This is a column identifier
            column = str(arg)
            columns.append(column)
            
            # Check if next arg is a sort_type
            if i + 1 < len(args) and isinstance(args[i + 1], Token):
                # Safely access children with bounds checking
                if len(args[i + 1]) > 0:
                    sort_type = str(args[i + 1]).lower()
                    ascending.append(sort_type == 'asc')
                else:
                    ascending.append(True)
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

    def load_statement(self, table_name,  source, params=None):

        source_str = str(source)

        if params:
            kwargs, kwargs_str = self._keyword_arg(params)
        else:
            kwargs = ''
            kwargs_str = ''
        
        # Source file case
        ast_node = {
            'type': 'load_table',
            'table_name': str(table_name),
            'source': source_str,
            'kwargs': kwargs,
            'kwargs_str': kwargs_str
        }
        
        self.current_table = str(table_name)
        
        return ast_node
    
    def table_statement(self, table_name, copy_table=None):
        """Handle table statements with optional 'from' clause"""
        table_name_str = str(table_name)
        
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
    
    def set_statement(self, target, expression, condition_list=None):
        """Handle set statements to create new columns with optional where clause"""
        target_str = str(target)
        expr_str = str(expression)
        
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
            'type': 'set',
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
    
    def select_statement(self, *columns):
        """Handle select statements to select specific columns"""
        column_list = [str(col) for col in columns]
        
        ast_node = {
            'type': 'select',
            'table_name': self.current_table,
            'columns': column_list
        }
        
        # Generate Python code: table.loc[:, ['col1', 'col2', ...]]
        columns_str = str(column_list)
        python_code = f"{self.current_table} = {self.current_table}.loc[:, {columns_str}]"
        
        return ast_node

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
   
    
    def pivot_statement(self, values_columns, pivot_rows, pivot_cols, agg):
        """Handle pivot statements to create pivot tables"""
        # Extract values columns from values_columns
        values_column_list = []
        if isinstance(values_columns, Tree):
            values_column_list = [str(col) for col in values_columns.children]
        elif isinstance(values_columns, list):
            values_column_list = [str(col) for col in values_columns]
        else:
            values_column_list = [str(values_columns)]
        
        # Extract row columns from pivot_rows
        row_columns = []
        if isinstance(pivot_rows, Tree):
            row_columns = [str(col) for col in pivot_rows.children]
        elif isinstance(pivot_rows, list):
            row_columns = [str(col) for col in pivot_rows]
        else:
            row_columns = [str(pivot_rows)]
        
        # Extract column columns from pivot_cols
        col_columns = []
        if isinstance(pivot_cols, Tree):
            col_columns = [str(col) for col in pivot_cols.children]
        elif isinstance(pivot_cols, list):
            col_columns = [str(col) for col in pivot_cols]
        else:
            col_columns = [str(pivot_cols)]
        
        # Extract aggregation functions
        agg_functions = []
        if isinstance(agg, Tree):
            agg_functions = agg.children 
        elif isinstance(agg, list):
            agg_functions = [str(func) for func in agg]
        else:
            agg_functions = [str(agg)]
        
        # Default to 'sum' if no aggregation specified
        if not agg_functions:
            agg_functions = ['sum']
        
        ast_node = {
            'type': 'pivot',
            'table_name': self.current_table,
            'values': values_column_list,
            'index': row_columns,
            'columns': col_columns,
            'aggfunc': agg_functions[0] if len(agg_functions) == 1 else agg_functions
        }
        
        return ast_node
    
    def pivot_values(self, *columns):
        """Handle pivot rows specification"""
        return [str(col) for col in columns]
    
    def pivot_rows(self, *columns):
        """Handle pivot rows specification"""
        return [str(col) for col in columns]
    
    def pivot_cols(self, *columns):
        """Handle pivot columns specification"""
        return [str(col) for col in columns]

    def groupby_statement(self, *args):
        """Handle groupby statements"""
        # args[0] is group_cols (list of strings)
        group_cols = args[0]
        
        agg_dict = {}
        
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
                    for item in arg:
                        agg_dict.update(item)
        
        ast_node = {
            'type': 'groupby',
            'table_name': self.current_table,
            'by': group_cols,
            'agg': agg_dict
        }
        
        return ast_node

    def group_cols(self, *columns):
        return [str(col) for col in columns]

    def agg_clause(self, *items):
        # Filter out tokens like _NL
        return [item for item in items if isinstance(item, dict)]

    def agg_item(self, col, func):
        return {str(col): str(func)}
    
    def agg_s(self, *functions):
        """Handle aggregation functions"""
        return [str(func) for func in functions]
    
    def AGG_FUNCTION(self, token):
        """Handle aggregation function tokens"""
        return str(token)
    
    def condition(self, column, comparator, value):
        """Handle individual filter conditions"""
        return {
            'column': str(column),
            'comparator': str(comparator),
            'value': self._convert_value(value) if not isinstance(value, list) else value
        }
    
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
        if isinstance(val, (int, float, list)) or val is None:
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
        return f"{ast_node['table_name']} = {ast_node['table_name']}.sort_values({ast_node['columns']}, ascending={ast_node['ascending']})"
    
    def generate_load_table_pandas(self, ast_node):
        load_table = f"{ast_node['table_name']} = pd.read_csv('{ast_node['source']}'{ast_node['kwargs_str']})"
        table_name = f"#__pivotal__\n__table_name__ = '{ast_node['table_name']}'\n#__pivotal__"
        return f"{load_table}\n{table_name}" 

    def generate_load_table_polars(self, ast_node):
        load_table = f"{ast_node['table_name']} = pl.read_csv('{ast_node['source']}'{ast_node['kwargs_str']})"
        table_name = f"#__pivotal__\n__table_name__ = '{ast_node['table_name']}'\n#__pivotal__"
        return f"{load_table}\n{table_name}"

    def generate_copy_table_pandas(self, ast_node):
        # Validation + copy
        validation = f"if not isinstance({ast_node['copy_from']}, pd.DataFrame): raise TypeError('{ast_node['copy_from']} is not a pandas DataFrame')"
        copy_code = f"{ast_node['table_name']} = {ast_node['copy_from']}.copy()"
        table_name = f"#__pivotal__\n__table_name__ = '{ast_node['table_name']}'\n#__pivotal__"
        return f"{validation}\n{copy_code}\n{table_name}"
    
    def generate_validate_table_pandas(self, ast_node):
        table_name = f"#__pivotal__\n__table_name__ = '{ast_node['table_name']}'\n#__pivotal__"
        validation - f"if not isinstance({ast_node['table_name']}, pd.DataFrame): raise TypeError('{ast_node['table_name']} is not a pandas DataFrame')"
        return f"{validation}\n{table_name}"
    
    def generate_set_pandas(self, ast_node):
        if ast_node['conditions']:
            query_str = self._build_query_string(ast_node['conditions'], ast_node['operators'])
            return (f"condition = {ast_node['table_name']}.eval('{query_str}')\n"
                   f"{ast_node['table_name']}.loc[condition, '{ast_node['target']}'] = "
                   f"{ast_node['table_name']}.eval('{ast_node['expression']}')[condition]")
        else:
            return f"{ast_node['table_name']}['{ast_node['target']}'] = {ast_node['table_name']}.eval('{ast_node['expression']}')"
    
    def generate_filter_pandas(self, ast_node):
        query_str = self._build_query_string(ast_node['conditions'], ast_node['operators'])
        return f"{ast_node['table_name']} = {ast_node['table_name']}.query('{query_str}')"
    
    def generate_select_pandas(self, ast_node):
        return f"{ast_node['table_name']} = {ast_node['table_name']}.loc[:, {ast_node['columns']}]"
    
    def generate_merge_pandas(self, ast_node):
        if ast_node['keys'] == '':
            return f"{ast_node['table_name']} = {ast_node['table_name']}.merge({ast_node['right_table']}, how='{ast_node['how']}'{ast_node['kwargs_str']})"
        else:
            return f"{ast_node['table_name']} = {ast_node['table_name']}.merge({ast_node['right_table']}, on={ast_node['keys']}, how='{ast_node['how']}', {ast_node['kwargs_str']})"
    
    def generate_pivot_pandas(self, ast_node):
        """Generate pandas pivot_table code"""
        table_name = ast_node['table_name']
        values = ast_node['values']
        index = ast_node['index']
        columns = ast_node['columns']
        aggfunc = ast_node['aggfunc']
        
        # Format index and columns as lists if they have multiple elements
        index_str = str(index) if len(index) > 1 else f"'{index[0]}'" if index else None
        columns_str = str(columns) if len(columns) > 1 else f"'{columns[0]}'" if columns else None
        
        # Build the pivot_table call
        pivot_args = []
        pivot_args.append(f"values={values}")
        
        if index:
            pivot_args.append(f"index={index_str}")
        
        if columns:
            pivot_args.append(f"columns={columns_str}")
        
        if aggfunc:
            if isinstance(aggfunc, list):
                aggfunc_str = str(aggfunc)
                if len(aggfunc) == len(values):
                    aggfunc_str = str({val: func for val, func in zip(values, aggfunc)})
                else:
                    aggfunc_str = str(aggfunc)
            else:
                aggfunc_str = f"'{aggfunc}'"
            pivot_args.append(f"aggfunc={aggfunc_str}")
        
        pivot_args_str = ', '.join(pivot_args)
        
        return f"{table_name} = pd.pivot_table({table_name}, {pivot_args_str})"

    def generate_groupby_pandas(self, ast_node):
        by = ast_node['by']
        agg = ast_node['agg']
        
        if agg:
            return f"{ast_node['table_name']} = {ast_node['table_name']}.groupby({by}).agg({agg}).reset_index()"
        else:
            return f"{ast_node['table_name']} = {ast_node['table_name']}.groupby({by}).sum().reset_index()"
    
    def _build_query_string(self, conditions, operators):
        """Build query string from conditions and operators"""
        query_parts = []
        
        for i, condition in enumerate(conditions):
            column = condition['column']
            comparator = condition['comparator']
            value = condition['value']
            
            # Build query string part
            if comparator in ['in', 'not in']:
                if isinstance(value, list):
                    value_str = str(value)
                else:
                    value_str = f"[{value}]"
                query_parts.append(f"{column} {comparator} {value_str}")
            elif isinstance(value, str):
                query_parts.append(f"{column} {comparator} '{value}'")
            else:
                query_parts.append(f"{column} {comparator} {value}")
            
            # Add operator if not the last condition
            if i < len(operators):
                query_parts.append(operators[i])
        
        return ' '.join(query_parts)
    
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


class DSLParser:
    def __init__(self, backend="pandas"):
        self.parser = Lark(
            grammar_indented, 
            parser='lalr', 
            postlex=DSLIndenter(),
            transformer=DSLTransformer()
        )
        self.code_generator = CodeGenerator(backend)
        self.autocomplete_file = Path('.pivotal_autocomplete.json')
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
        
    def preprocess_code(self, code):
        """Preprocess DSL code to handle whitespace issues"""
        # Strip leading and trailing whitespace
        code = code.strip()
        
        # Ensure the file ends with a newline if it's not empty
        if code and not code.endswith('\n'):
            code += '\n'
        
        # Handle multiple consecutive newlines
        lines = code.split('\n')
        processed_lines = []
        
        for line in lines:
            # Keep the line as-is, but ensure consistent spacing
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def parse(self, code):
        """Parse DSL code and return AST + Python code"""
        try:
            # Preprocess the code to handle whitespace issues
            processed_code = self.preprocess_code(code)
            result = self.parser.parse(processed_code)
            return result
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
        code = self.generate_code(results)
        
        if isinstance(results, dict) and 'error' in results:
            print(f"Parse error: {results['error']}")
            return None
        
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
        python_code_list = self.generate_code(results, backend=backend)
        
        if isinstance(results, dict) and 'error' in results:
            print(f"Parse error: {results['error']}")
            return None
        
        i = 0 
        for python_code in python_code_list:
            print(f"Executing: {python_code}")
            try:
                exec(python_code, globals_dict)
                table_name = results[i]['table_name']
                if verbose:
                    df = globals_dict[table_name]
                    print(f"\nTable '{table_name}':")
                    print(f"Shape: {df.shape}\n")
                    print(df.head())
            except Exception as e:
                print(f"Execution error: {e}")
            i+=1
        
        # Update autocomplete info after execution
        self.update_autocomplete_info(globals_dict)
        

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