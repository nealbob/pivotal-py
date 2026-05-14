from lark import Lark, Transformer, v_args
from lark.indenter import Indenter
from lark import Tree
from lark.lexer import Token
from lark.exceptions import UnexpectedToken, UnexpectedCharacters, UnexpectedEOF, VisitError
import pandas as pd
import ast
import copy
import json
import os
import re
import warnings
from pathlib import Path
from types import SimpleNamespace
from .errors import PivotalError, _make_suggestion

_AGG_CALL_RE = re.compile(
    r'\b(mean|avg|sum|min|max|count|std|median|var|nunique|first|last)'
    r'\(([a-zA-Z_][a-zA-Z0-9_]*)\)'
)
_QUANTILE_CALL_RE = re.compile(
    r'\b(quantile|percentile)'
    r'\(([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)'
)
_WAVG_CALL_RE = re.compile(
    r'\b(?:wavg|wmean)\(([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)\)'
)

# All reserved words in the Pivotal grammar.  Used for collision validation.
PIVOTAL_KEYWORDS = frozenset({
    # Statement keywords
    'with', 'load', 'bulk', 'filter', 'assert', 'check', 'select', 'sort', 'order', 'save', 'all', 'delete',
    'for', 'function', 'return', 'list', 'scalar', 'dict',
    'merge', 'pivot', 'unpivot', 'group', 'python', 'plot', 'drop', 'fillna',
    'dropna', 'distinct', 'concat', 'rename', 'round', 'apply', 'table',
    'rank', 'lag', 'lead', 'cumsum', 'cummean', 'cummin', 'cummax', 'rolling', 'agg',
    'intersect', 'exclude', 'cast',
    # Clause keywords
    'from', 'where', 'as', 'on', 'by', 'rows', 'cols', 'include', 'exclude',
    # Comparators / logic
    'in', 'not', 'between', 'contains', 'startswith', 'endswith',
    'and', 'or',
    # Aggregation functions
    'mean', 'min', 'max', 'sum', 'count', 'avg', 'median', 'std', 'wavg', 'wmean',
    'quantile', 'percentile',
    # Sort / merge modifiers
    'asc', 'desc', 'left', 'right', 'inner', 'outer',
    # Atoms
    'true', 'false', 'none',
})

# Inline helper emitted into generated code to sanitise DataFrame column names.
# Spaces and special chars → underscore; leading digit → prefixed; duplicates suffixed.
_SANITISE_HELPER = (
    "def _pvt_sanitise(cols):\n"
    "    import re as _re\n"
    "    seen, renamed, new = {}, {}, []\n"
    "    for c in cols:\n"
    "        s = str(c)\n"
    "        n = _re.sub(r'_+', '_', _re.sub(r'[^a-zA-Z0-9_]', '_', s)).strip('_') or 'col'\n"
    "        if n[0].isdigit(): n = '_' + n\n"
    "        cnt = seen.get(n, 0); seen[n] = cnt + 1\n"
    "        if cnt: n = f'{n}_{cnt}'\n"
    "        new.append(n)\n"
    "        if n != s: renamed[s] = n\n"
    "    return new, renamed\n"
)

#AGG_DICT: _NL _INDENT IDENTIFIER AGG_FUNCTION ("," AGG_FUNCTION)* _DEDENT (_NL _INDENT IDENTIFIER AGG_FUNCTION ("," AGG_FUNCTION)* _DEDENT)* _NL?
# Grammar definition using indentation
grammar_indented = r"""
    %declare _INDENT _DEDENT
    start: _NL* (_INDENT? statement _DEDENT?)+ _NL*

    statement: bulk_load_statement
               | function_definition
               | return_statement
               | list_statement
               | scalar_statement
               | dict_statement
               | load_statement
               | from_statement
               | dataframe_statement
               | assign_statement
               | assert_statement
               | check_statement
               | filter_statement
               | select_statement
               | sort_statement
               | merge_statement
               | pivot_statement
               | unpivot_statement
               | rank_statement
               | shift_statement
               | cumulative_statement
               | rolling_statement
               | groupby_statement
               | agg_statement
               | python_statement
               | agg_plot_statement
               | plot_statement
               | drop_statement
               | fillna_statement
               | dropna_statement
               | distinct_statement
               | concat_statement
               | intersect_statement
               | exclude_statement
               | cast_statement
               | round_statement
               | rename_statement
               | apply_statement
               | table_statement
               | save_statement
               | show_statement
               | delete_statement
               | for_statement
               | function_call_statement

    function_definition: "function" IDENTIFIER "(" function_params? ")" _NL _INDENT function_body_statement+ _DEDENT
    function_params: function_param ("," function_param)*
    function_param: IDENTIFIER -> function_param_required
                  | IDENTIFIER "=" function_arg_value -> function_param_default
    function_body_statement: _INDENT? statement
    return_statement: "return" IDENTIFIER ("," IDENTIFIER)* _NL?

    list_statement: "list" IDENTIFIER "=" list_items _NL?
    list_items: function_arg_value ("," function_arg_value)*

    scalar_statement: "scalar" IDENTIFIER "=" function_arg_value _NL?

    dict_statement: "dict" IDENTIFIER "=" function_arg_value _NL? -> dict_assignment
                  | "dict" IDENTIFIER "from" (STRING | PATH) _NL? -> dict_from_file
                  | "dict" IDENTIFIER _NL _INDENT dict_entries _DEDENT
    dict_entries: dict_entry+
    dict_entry: dict_key "=" dict_entry_items _NL? -> dict_value_entry
              | dict_key ":" dict_entry_items _NL? -> dict_value_entry
              | dict_key _NL _INDENT dict_entries _DEDENT -> dict_nested_entry
    dict_key: IDENTIFIER | NUMBER | SIGNED_NUMBER | STRING
    dict_entry_items: function_arg_value ("," function_arg_value)*

    function_call_statement: FUNCTION_CALL_NAME "(" function_args? ")" _NL?
    function_args: function_arg ("," function_arg)*
    function_arg: function_kw_arg
                | function_arg_value
    function_kw_arg.2: IDENTIFIER "=" function_arg_value
    function_arg_value: function_inline_list
                      | value
    function_inline_list: "(" function_inline_items ")"
    function_inline_items: value ("," value)+

    for_statement: "for" IDENTIFIER "in" loop_source _NL _INDENT for_body_statement+ _DEDENT
    loop_source: PYTHON_VAR -> loop_var_source
               | loop_cols
    loop_cols: IDENTIFIER ("," IDENTIFIER)*
    for_body_statement: for_assign_statement
                      | cast_statement
                      | fillna_statement
                      | drop_statement
                      | dropna_statement
                      | round_statement
                      | rank_statement
                      | shift_statement
                      | cumulative_statement
                      | rolling_statement
    for_assign_statement: for_simple_target "=" expression (_NL | _NL _INDENT assign_opts _DEDENT)?
                        | for_simple_target "=" _NL _INDENT case_list _DEDENT
                        | target_expr "=" expression (_NL | _NL _INDENT assign_opts _DEDENT)?
                        | target_expr "=" _NL _INDENT case_list _DEDENT
    for_simple_target: IDENTIFIER
    target_expr: target_expr_part ("+" target_expr_part)+
    target_expr_part: IDENTIFIER | STRING

    delete_statement: "delete" IDENTIFIER _NL?

    show_statement: "show" SHOW_MODE? _NL?
    SHOW_MODE.2: "head" | "summary" | "shape" | "columns"

    apply_statement: "apply" (PYTHON_VAR | IDENTIFIER) _NL?

    agg_plot_statement: ("pivot" "plot" | "agg" "plot") IDENTIFIER IDENTIFIER? (_NL | _NL _INDENT agg_plot_params _DEDENT)?

    agg_plot_params: agg_plot_param+
    agg_plot_y_item: IDENTIFIER IDENTIFIER
    agg_plot_param: "x" IDENTIFIER STRING _NL?                                  -> agg_plot_x_labeled
                  | "x" IDENTIFIER _NL?                                         -> agg_plot_x
                  | "y" agg_plot_y_item ("," agg_plot_y_item)* STRING _NL?     -> agg_plot_y_labeled
                  | "y" agg_plot_y_item ("," agg_plot_y_item)* _NL?            -> agg_plot_y
                  | "by" IDENTIFIER _NL?                                        -> agg_plot_by
                  | "cols" NUMBER _NL?                                          -> agg_plot_cols
                  | "canvas" IDENTIFIER _NL?                                    -> agg_plot_canvas
                  | "show" _NL?                                                 -> agg_plot_show

    plot_statement: "plot" IDENTIFIER IDENTIFIER? plot_on? (_NL | _NL _INDENT plot_params _DEDENT)?
    plot_on: "on" IDENTIFIER

    plot_params: plot_param+
    plot_param: "by" IDENTIFIER _NL?           -> plot_by_param
              | "cols" value _NL?              -> plot_cols_param
              | IDENTIFIER value STRING _NL?   -> plot_labeled_param
              | IDENTIFIER value _NL?          -> plot_value_param
              | IDENTIFIER "=" value _NL?      -> plot_value_param
              | IDENTIFIER "=" list_value _NL? -> plot_list_param
              | IDENTIFIER list_value _NL?     -> plot_list_param
              | "show" _NL?                   -> plot_show

    table_statement: "table" IDENTIFIER (_NL | _NL _INDENT table_params _DEDENT)?

    table_params: table_param+

    table_param: "title"    STRING         _NL?                                  -> table_title
               | "subtitle" STRING         _NL?                                  -> table_subtitle
               | "font" "size" NUMBER      _NL?                                  -> table_font_size
               | "font" STRING             _NL?                                  -> table_font_family
               | "stub" IDENTIFIER                              _NL?             -> table_stub
               | "stub" IDENTIFIER STRING                       _NL?             -> table_stub_labeled
               | "stub" IDENTIFIER "," IDENTIFIER              _NL?             -> table_stub_grouped
               | "stub" IDENTIFIER "," IDENTIFIER STRING       _NL?             -> table_stub_grouped_labeled
               | "stripe"                  _NL?                                  -> table_stripe
               | "canvas" IDENTIFIER       _NL?                                  -> table_canvas
               | "label" label_col_spec ("," label_col_spec)* _NL?              -> table_label_line
               | "format" IDENTIFIER "as" fmt_type _NL?                         -> table_format_col
               | "format" fmt_type                     _NL?                      -> table_format_all
               | "style"  STRING                       _NL?                      -> table_style
               | "summary" summary_spec ("," summary_spec)* _NL?                -> table_summary_line
               | "spanner" spanner_cols STRING _NL?                              -> table_spanner_line
               | "auto" "spanner" _NL?                                           -> table_auto_spanner
               | "show" _NL?                                                      -> table_show

    spanner_cols: IDENTIFIER ("," IDENTIFIER)*

    summary_spec: IDENTIFIER "as" STRING    -> summary_spec_labeled
                | IDENTIFIER                -> summary_spec_bare

    label_col_spec: IDENTIFIER "as" STRING

    fmt_type: "number" NUMBER    -> fmt_number
            | "number"           -> fmt_number_noprec
            | "integer"          -> fmt_integer
            | "currency" IDENTIFIER -> fmt_currency
            | "currency"         -> fmt_currency_nocode
            | "percent" NUMBER   -> fmt_percent
            | "percent"          -> fmt_percent_noprec
            | "date"             -> fmt_date

    python_statement: "python" UNQUOTED_STRING _NL?

    groupby_statement: "group" "by" group_cols (_NL _INDENT agg_clause _DEDENT)? _NL?

    group_cols: (IDENTIFIER | PYTHON_VAR | COMPILE_REF) ("," (IDENTIFIER | PYTHON_VAR | COMPILE_REF))*

    agg_clause: agg_line+

    agg_line: "agg" agg_item ("," agg_item)* _NL?

    agg_item: QUANTILE_FUNCTION "(" (IDENTIFIER | PYTHON_VAR) "," quantile_value ")" ("as" IDENTIFIER)? -> quantile_bracket_item
            | QUANTILE_FUNCTION (IDENTIFIER | PYTHON_VAR) quantile_value ("as" IDENTIFIER)? -> quantile_space_item
            | AGG_FUNCTION "(" (IDENTIFIER | PYTHON_VAR) ")" ("as" IDENTIFIER)?
            | AGG_FUNCTION (IDENTIFIER | PYTHON_VAR) ("as" IDENTIFIER)?
            | PYTHON_VAR "(" custom_agg_args? ")" "as" IDENTIFIER -> custom_agg_bracket_alias_item
            | PYTHON_VAR "(" custom_agg_args? ")" -> custom_agg_bracket_item
            | PYTHON_VAR custom_agg_cols "as" IDENTIFIER -> custom_agg_alias_item
            | PYTHON_VAR custom_agg_cols -> custom_agg_item
            | ("wmean" | "wavg") "(" (IDENTIFIER | PYTHON_VAR) "," (IDENTIFIER | PYTHON_VAR) ")" ("as" IDENTIFIER)? -> wmean_bracket_item
            | ("wmean" | "wavg") (IDENTIFIER | PYTHON_VAR) (IDENTIFIER | PYTHON_VAR) "as" IDENTIFIER -> wmean_alias_item
            | ("wmean" | "wavg") (IDENTIFIER | PYTHON_VAR) (IDENTIFIER | PYTHON_VAR)+ -> wmean_cols_item

    custom_agg_cols: (IDENTIFIER | PYTHON_VAR)+
    custom_agg_args: custom_agg_arg ("," custom_agg_arg)*
    custom_agg_arg: custom_agg_kw_arg
                  | IDENTIFIER -> custom_agg_col_arg
                  | PYTHON_VAR -> custom_agg_col_arg
    custom_agg_kw_arg.2: IDENTIFIER "=" custom_agg_value
    custom_agg_value: value | "[" [value ("," value)*] "]" -> custom_agg_list_value
    quantile_value: SIGNED_NUMBER | NUMBER

    merge_statement: MERGE_TYPE? "merge" RIGHT_TABLE ("on" keys)? (_NL | _NL _INDENT params _DEDENT)?
    
    MERGE_TYPE.2: "left" | "right" | "inner" | "outer"
    keys: IDENTIFIER ("," IDENTIFIER)*
    RIGHT_TABLE: IDENTIFIER

    load_statement: "load" (STRING | PATH | PYTHON_VAR) "as" table_name (_NL | _NL _INDENT params _DEDENT)?
                  | "load" "all" _NL?
                  | "load" table_name _NL?

    bulk_load_statement: "bulk" "load" bulk_sources "as" bulk_aliases (_NL | _NL _INDENT bulk_load_params _DEDENT)?
    bulk_sources: PYTHON_VAR -> bulk_source_var
                | bulk_file_source ("," bulk_file_source)* -> bulk_source_list
    bulk_file_source: STRING | PATH
    bulk_aliases: PYTHON_VAR -> bulk_alias_var
                | table_name ("," table_name)* -> bulk_alias_names
    bulk_load_params: bulk_load_param+
    bulk_load_param: "source" "column" IDENTIFIER _NL? -> bulk_source_column
                   | "source" "value" SOURCE_VALUE _NL? -> bulk_source_value
                   | param
    SOURCE_VALUE: "path" | "filename" | "stem"

    from_statement: "from" (STRING | PATH | PYTHON_VAR) _NL _INDENT from_body+ _DEDENT
    from_body: from_load_line | from_query_line
    from_load_line:  "load"  from_item ("," from_item)* _NL
    from_query_line: "query" STRING "as" table_name _NL
    from_item: table_name "as" table_name

    save_statement: "save" STRING (_NL _INDENT save_params _DEDENT)? _NL?

    save_params: save_param+
    save_param: "path" (STRING | PYTHON_VAR) _NL?           -> save_path
              | "format" IDENTIFIER _NL?                     -> save_format
              | "chart_format" IDENTIFIER _NL?               -> save_chart_format
              | "include" save_id_list _NL?                  -> save_include
              | "exclude" save_id_list _NL?                  -> save_exclude

    save_id_list: IDENTIFIER ("," IDENTIFIER)*

    dataframe_statement: "with" table_name ("as" copy_table)? _NL?

    assign_statement: target "=" expression (_NL | _NL _INDENT assign_opts _DEDENT)?
                    | target "=" _NL _INDENT case_list _DEDENT

    assign_opts: assign_opt+
    assign_opt: "where" condition_list _NL? -> assign_where
              | "by"    IDENTIFIER ("," IDENTIFIER)* _NL? -> assign_by
              | "else" expression _NL? -> assign_else

    case_list: case_branch+ case_default?
    case_branch: "where" condition_list (";" | ":") CASE_BRANCH_EXPR _NL
    case_default: "else" expression _NL?
                | CASE_DEFAULT_EXPR _NL
    CASE_BRANCH_EXPR: /[^\n]+/
    CASE_DEFAULT_EXPR: /(?!where\b)[^\n]+/

    filter_statement: "filter" condition_list  _NL?

    assert_statement: "assert" quality_rule _NL?
    check_statement: "check" quality_rule _NL?
    quality_rule: quality_not_null
                | quality_unique
                | condition_list         -> quality_condition
    quality_unique.2: dq_cols "unique"
    quality_not_null.2: dq_cols NOT_NULL
    dq_cols: IDENTIFIER ("," IDENTIFIER)*
    
    select_statement: "select" select_item ("," select_item)* _NL?
                    | "select" "matches" STRING _NL? -> select_matches_statement

    select_item: (IDENTIFIER | PYTHON_VAR | COMPILE_REF) ("as" IDENTIFIER)?

    agg_statement: "agg" agg_item ("," agg_item)* _NL?

    pivot_statement: "pivot" _NL _INDENT pivot_args _DEDENT

    pivot_args: (agg_clause | pivot_rows | pivot_cols)+

    pivot_rows: "rows" (IDENTIFIER | PYTHON_VAR | COMPILE_REF) ("," (IDENTIFIER | PYTHON_VAR | COMPILE_REF))* _NL?
    pivot_cols: "cols" (IDENTIFIER | PYTHON_VAR | COMPILE_REF) ("," (IDENTIFIER | PYTHON_VAR | COMPILE_REF))* _NL?

    unpivot_statement: "unpivot" _NL _INDENT unpivot_args _DEDENT

    unpivot_args: unpivot_arg+
    unpivot_arg: "id"     (IDENTIFIER | PYTHON_VAR | COMPILE_REF) ("," (IDENTIFIER | PYTHON_VAR | COMPILE_REF))* _NL? -> unpivot_id
               | "cols"   (IDENTIFIER | PYTHON_VAR | COMPILE_REF) ("," (IDENTIFIER | PYTHON_VAR | COMPILE_REF))* _NL? -> unpivot_cols
               | "variable" STRING _NL?                                                     -> unpivot_name
               | "value"  STRING _NL?                                                       -> unpivot_value_name

    AGG_FUNCTION.2: "mean" | "min" | "max" | "sum" | "count" | "avg" | "median" | "std" | "nunique"
    QUANTILE_FUNCTION.2: "quantile" | "percentile"

    rank_statement: "rank" IDENTIFIER SORT_TYPE? RANK_PCT? "as" IDENTIFIER (_NL | _NL _INDENT window_opts _DEDENT)?
    RANK_PCT: "pct"

    shift_statement: SHIFT_FUNC IDENTIFIER NUMBER "as" IDENTIFIER (_NL | _NL _INDENT window_opts _DEDENT)?
    SHIFT_FUNC.2: "lag" | "lead"

    cumulative_statement: CUM_FUNC IDENTIFIER "as" IDENTIFIER (_NL | _NL _INDENT window_opts _DEDENT)?
    CUM_FUNC.2: "cumsum" | "cummean" | "cummin" | "cummax"

    rolling_statement: "rolling" AGG_FUNCTION IDENTIFIER NUMBER "as" IDENTIFIER (_NL | _NL _INDENT window_opts _DEDENT)?

    window_opts: window_opt+
    window_opt: "by"    IDENTIFIER ("," IDENTIFIER)* _NL? -> window_by
              | "order" IDENTIFIER                  _NL? -> window_order
              | "min_periods" "="? NUMBER           _NL? -> window_min_periods

    sort_statement: ("sort" | "order" "by") (IDENTIFIER | PYTHON_VAR | COMPILE_REF) SORT_TYPE? ("," (IDENTIFIER | PYTHON_VAR | COMPILE_REF) SORT_TYPE?)* _NL?

    SORT_TYPE.2: "asc" | "desc"

    // Lower priority (-1) ensures keywords always win over assign target
    ASSIGN_TARGET.-1: /[a-zA-Z][a-zA-Z0-9_]*/

    target: ASSIGN_TARGET
          | IDENTIFIER
    table_name: IDENTIFIER
    copy_table: IDENTIFIER

    expression: UNQUOTED_STRING

    condition: IDENTIFIER COMPARATOR (value | list_value)
             | IDENTIFIER "in" list_value       -> condition_in_list
             | IDENTIFIER "in" PYTHON_VAR       -> condition_in_var
             | IDENTIFIER "in" COMPILE_REF      -> condition_in_ref
             | IDENTIFIER "in" IDENTIFIER       -> condition_in_name
             | IDENTIFIER "not" "in" list_value -> condition_not_in_list
             | IDENTIFIER "not" "in" PYTHON_VAR -> condition_not_in_var
             | IDENTIFIER "not" "in" COMPILE_REF -> condition_not_in_ref
             | IDENTIFIER "not" "in" IDENTIFIER -> condition_not_in_name

    condition_list: condition (AOR condition)*

    COMPARATOR: "==" | "!=" | ">" | "<" | ">=" | "<=" | "between" | "contains" | "not contains" | "matches" | "not matches" | "startswith" | "endswith"

    drop_statement: "drop" IDENTIFIER ("," IDENTIFIER)* _NL?
                  | "drop" "matches" STRING _NL? -> drop_matches_statement

    fillna_statement: "fillna" value _NL?
                    | "fillna" _NL _INDENT fillna_col_params _DEDENT
                    | "fillna" fillna_col_list _NL?

    fillna_col_list: fillna_col_item ("," fillna_col_item)*
    fillna_col_item: IDENTIFIER value

    fillna_col_params: fillna_col_param+
    fillna_col_param: IDENTIFIER "=" value _NL?

    dropna_statement: "dropna" dropna_cols? _NL?
    dropna_cols: IDENTIFIER ("," IDENTIFIER)*

    distinct_statement: "distinct" distinct_cols? _NL?
    distinct_cols: IDENTIFIER ("," IDENTIFIER)*

    concat_statement: "concat" IDENTIFIER ("," IDENTIFIER)* _NL?
    intersect_statement: "intersect" IDENTIFIER ("," IDENTIFIER)* _NL?
    exclude_statement: "exclude" IDENTIFIER ("," IDENTIFIER)* _NL?

    cast_statement: "cast" IDENTIFIER ("," IDENTIFIER)* "as" CAST_TYPE CAST_STRICT? _NL?
    CAST_TYPE.2: "int" | "integer" | "float" | "string" | "str" | "bool" | "boolean" | "datetime"
    CAST_STRICT.2: "strict"

    round_statement: "round" IDENTIFIER ("," IDENTIFIER)* NUMBER ("as" IDENTIFIER)? _NL?

    rename_statement: "rename" rename_item ("," rename_item)* _NL?
    rename_item: IDENTIFIER "as" IDENTIFIER

    params: param+

    param: keyword_arg _NL?

    keyword_arg: IDENTIFIER value 
                | IDENTIFIER "=" value 
                | IDENTIFIER "=" list_value 
                | IDENTIFIER list_value 

    file_path: PATH

    value: BOOLEAN | SIGNED_NUMBER | NUMBER | STRING | COMPILE_REF | IDENTIFIER | PATH | NONE | PYTHON_VAR
    list_value: "[" [value ("," value)*] "]"
              | "(" value "," value ("," value)* ")"
              | [value "," value ("," value)*]

    BOOLEAN.2: "True" | "False" | "true" | "false"
    NONE.2: "None" | "none"
    AOR.2: /and/i | /or/i
    NOT_NULL.3: /not[ \t]+null/i
    PYTHON_VAR: ":" IDENTIFIER
    COMPILE_REF.3: /[a-zA-Z_][a-zA-Z0-9_]*(?:(?:\.[a-zA-Z_][a-zA-Z0-9_]*)|(?:\[-?\d+\]))+/
    FUNCTION_CALL_NAME.3: /[a-zA-Z][a-zA-Z0-9_]*(?=\()/
    IDENTIFIER: /[a-zA-Z][a-zA-Z0-9_]*/
    IDENT_LIST.2: IDENTIFIER ("," IDENTIFIER)*
    STRING: /"[^"]*"/ | /'[^']*'/
    UNQUOTED_STRING: /[^\n]+/
    PATH: /[a-zA-Z0-9_]+[:\\\/][a-zA-Z0-9_:\/\\\.\-]+|[\\\/][a-zA-Z0-9_:\/\\\.\-]+|[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+/
    SIGNED_NUMBER.1: /-\d+(\.\d+)?/
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

class _LiteralStr(str):
    """A quoted string literal value — distinct from an unquoted identifier (column reference)."""
    pass


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

    def function_definition(self, name, params=None, *body):
        self.current_table = None
        return {
            'type': 'function_def',
            'name': str(name),
            'params': params or [],
            'body': list(body),
        }

    def function_params(self, *params):
        return list(params)

    def function_param_required(self, name):
        return {'name': str(name), 'default': None, 'has_default': False}

    def function_param_default(self, name, default):
        return {'name': str(name), 'default': default, 'has_default': True}

    def function_body_statement(self, stmt):
        return stmt

    def return_statement(self, *names):
        return {'type': 'return', 'names': [str(name) for name in names]}

    def list_statement(self, name, items):
        return {'type': 'list_def', 'name': str(name), 'items': items}

    def list_items(self, *items):
        return list(items)

    def scalar_statement(self, name, value):
        return {'type': 'scalar_def', 'name': str(name), 'value': value}

    def dict_from_file(self, name, path):
        return {'type': 'dict_def', 'name': str(name), 'source': str(path)}

    def dict_assignment(self, name, value):
        return {'type': 'dict_def', 'name': str(name), 'value': value}

    def dict_statement(self, name, entries):
        return {'type': 'dict_def', 'name': str(name), 'items': entries}

    def dict_entries(self, *entries):
        return dict(entries)

    def dict_key(self, key):
        if isinstance(key, Token) and key.type == 'STRING':
            return ast.literal_eval(str(key))
        return str(key)

    def dict_value_entry(self, key, items):
        return (str(key), items)

    def dict_nested_entry(self, key, entries):
        return (str(key), entries)

    def dict_entry_items(self, *items):
        converted = [self._convert_value(item) for item in items]
        return converted[0] if len(converted) == 1 else converted

    def function_call_statement(self, name, args=None):
        return {'type': 'function_call', 'name': str(name), 'args': args or []}

    def function_args(self, *args):
        return list(args)

    def function_kw_arg(self, name, value):
        return {'kind': 'kwarg', 'name': str(name), 'value': value}

    def function_arg(self, value):
        return value

    def function_arg_value(self, value):
        return value if isinstance(value, list) else self._convert_value(value)

    def function_inline_list(self, items):
        return items

    def function_inline_items(self, *items):
        return [self._convert_value(item) for item in items]
    
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
        - load "path" as name  → load_table (existing file)
        - load all             → load_all (all tables from active package)
        - load name            → load_package_table (named table from active package)
        """
        if len(args) == 0:
            # "load all" — no named children (both "load" and "all" are anonymous terminals)
            return {'type': 'load_all'}

        if len(args) == 1:
            # "load table_name" — package table load (no source path)
            table_name_str = str(args[0])
            return {'type': 'load_package_table', 'table_name': table_name_str}

        # len(args) >= 2: "load source as table_name [params]"
        source, table_name = args[0], args[1]
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

        ast_node = {
            'type': 'load_table',
            'table_name': str(table_name),
            'source': source_val,
            'kwargs': kwargs,
            'kwargs_str': kwargs_str,
        }

        self.current_table = str(table_name)
        return ast_node

    def bulk_file_source(self, source):
        return str(source)

    def bulk_source_var(self, var):
        return {'kind': 'var', 'value': var}

    def bulk_source_list(self, *sources):
        return {'kind': 'list', 'value': [str(s) for s in sources]}

    def bulk_alias_var(self, var):
        return {'kind': 'var', 'value': var}

    def bulk_alias_names(self, *aliases):
        return {'kind': 'names', 'value': [str(a) for a in aliases]}

    def bulk_load_params(self, *params):
        return list(params)

    def bulk_source_column(self, name):
        return {'bulk_option': 'source_column', 'value': str(name)}

    def bulk_source_value(self, value):
        return {'bulk_option': 'source_value', 'value': str(value)}

    def SOURCE_VALUE(self, token):
        return str(token)

    def bulk_load_statement(self, sources, aliases, params=None):
        params = params or []
        kwargs = {}
        source_column = 'source'
        source_value = 'filename'

        for param in params:
            if not isinstance(param, dict):
                continue
            opt = param.get('bulk_option')
            if opt == 'source_column':
                source_column = param['value']
            elif opt == 'source_value':
                source_value = param['value']
            else:
                kwargs.update(param)

        kwargs_str = ', '.join(
            [f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in kwargs.items()]
        )
        if kwargs_str:
            kwargs_str = ', ' + kwargs_str

        alias_kind = aliases['kind']
        alias_value = aliases['value']
        mode = 'concat' if alias_kind == 'names' and len(alias_value) == 1 else 'separate'
        table_name = alias_value[0] if mode == 'concat' else (
            alias_value[-1] if alias_kind == 'names' and alias_value else None
        )

        if table_name:
            self.current_table = table_name

        return {
            'type': 'bulk_load',
            'mode': mode,
            'table_name': table_name,
            'sources': sources,
            'aliases': aliases,
            'source_column': source_column,
            'source_value': source_value,
            'kwargs': kwargs,
            'kwargs_str': kwargs_str,
        }

    # ------------------------------------------------------------------
    # from_statement transformer methods
    # ------------------------------------------------------------------

    def from_item(self, *args):
        return {'sql_table': str(args[0]), 'alias': str(args[1])}

    def from_load_line(self, *args):
        return {'kind': 'load', 'items': list(args)}

    def from_query_line(self, *args):
        return {'kind': 'query', 'sql': str(args[0]), 'alias': str(args[1])}

    def from_body(self, *args):
        return args[0]

    def from_statement(self, *args):
        source = args[0]
        bodies = list(args[1:])
        # Set current_table to last alias defined in the block
        all_aliases = []
        for b in bodies:
            if b['kind'] == 'load':
                all_aliases += [i['alias'] for i in b['items']]
            else:
                all_aliases.append(b['alias'])
        if all_aliases:
            self.current_table = all_aliases[-1]
        source_val = source if isinstance(source, dict) and source.get('type') == 'var' else str(source)
        return {
            'type': 'from_db',
            'source': source_val,
            'bodies': bodies,
        }

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
        """Handle 'with' statements:
        - with source          → operate on existing table
        - with source as new   → copy source to new table
        """
        clean_args = []
        for arg in args:
            if isinstance(arg, Token) and arg.type == '_NL':
                continue
            clean_args.append(arg)

        source_table = clean_args[0]
        new_table = clean_args[1] if len(clean_args) > 1 else None

        source_table_str = str(source_table)

        if source_table_str.lower() in PIVOTAL_KEYWORDS:
            raise ValueError(
                f"'{source_table_str}' is a Pivotal reserved keyword and cannot be used as a table name."
            )

        if new_table is not None:
            # Case: with source as new_table — copy
            new_table_str = str(new_table)
            ast_node = {
                'type': 'copy_table',
                'table_name': new_table_str,
                'copy_from': source_table_str,
            }
            self.current_table = new_table_str
        else:
            # Case: with source — operate on existing
            ast_node = {
                'type': 'validate_table',
                'table_name': source_table_str,
            }
            self.current_table = source_table_str

        return ast_node

    def loop_cols(self, *cols):
        return [str(c) for c in cols]

    def loop_var_source(self, var):
        return var

    def loop_source(self, source):
        return source

    def for_body_statement(self, stmt):
        return stmt

    def target(self, target):
        return str(target)

    def for_simple_target(self, target):
        return target

    def target_expr_part(self, part):
        if isinstance(part, Token) and part.type == 'STRING':
            return {'type': 'literal', 'value': str(part)[1:-1]}
        return {'type': 'identifier', 'value': str(part)}

    def target_expr(self, *parts):
        return {'type': 'target_expr', 'parts': list(parts)}

    def for_statement(self, loop_var, columns, *body):
        """Handle column-loop statements before parser-level lowering."""
        loop_var_str = str(loop_var)

        if loop_var_str.lower() in PIVOTAL_KEYWORDS:
            raise ValueError(
                f"'{loop_var_str}' is a Pivotal reserved keyword and cannot be used as a loop variable."
            )

        return {
            'type': 'for',
            'var': loop_var_str,
            'columns': list(columns) if isinstance(columns, list) else columns,
            'body': list(body),
        }

    def assign_where(self, condition_list):
        temp = self._build_conditional_statement(condition_list)
        return {'type': 'assign_where',
                'conditions': temp['ast']['conditions'],
                'operators': temp['ast']['operators'],
                'query_str': temp['query_str']}

    def assign_by(self, *cols):
        return {'type': 'assign_by', 'cols': [str(c) for c in cols]}

    def assign_else(self, expression):
        return {'type': 'assign_else', 'expression': str(expression).strip()}

    def assign_opts(self, *args):
        return list(args)

    def case_branch(self, condition_list, expression):
        temp = self._build_conditional_statement(condition_list)
        # CASE_BRANCH_EXPR captures raw text including quotes, e.g. '"premium"' or 'price * 1.1'
        return {
            'type': 'case_branch',
            'query_str': temp['query_str'],
            'conditions': temp['ast']['conditions'],
            'operators': temp['ast']['operators'],
            'expression': str(expression).strip(),
        }

    def case_default(self, token):
        expr = str(token).strip()
        # Strip a leading explicit else keyword, with or without colon
        if expr.lower().startswith('else:'):
            expr = expr[5:].strip()
        elif expr.lower().startswith('else '):
            expr = expr[5:].strip()
        return {'type': 'case_default', 'expression': expr}

    def case_list(self, *args):
        return list(args)

    def assign_statement(self, target, *rest):
        """Handle assign statements: simple, conditional, or multi-case."""
        return self._build_assign_node(target, *rest)

    def for_assign_statement(self, target, *rest):
        """Handle assignment statements inside column loops."""
        return self._build_assign_node(target, *rest)

    def _build_assign_node(self, target, *rest):
        target_str = target if isinstance(target, dict) else str(target)
        if isinstance(target_str, str) and target_str.lower() in PIVOTAL_KEYWORDS:
            raise ValueError(
                f"'{target_str}' is a Pivotal reserved keyword and cannot be used as a column name."
            )

        # Multi-case form: second arg is a list of case_branch/case_default dicts
        if rest and isinstance(rest[0], list):
            return {
                'type': 'assign',
                'table_name': self.current_table,
                'target': target_str,
                'expression': None,
                'conditions': None,
                'operators': None,
                'query_str': None,
                'by_cols': [],
                'default_expr': None,
                'cases': rest[0],
            }

        # Simple / with opts form
        expr_str = str(rest[0])
        opts = rest[1] if len(rest) > 1 else []
        conditions = None
        operators = None
        
        query_str = None
        by_cols = []

        default_expr = None
        for opt in (opts or []):
            if opt['type'] == 'assign_where':
                conditions = opt['conditions']
                operators = opt['operators']
                query_str = opt['query_str']
            elif opt['type'] == 'assign_by':
                by_cols = opt['cols']
            elif opt['type'] == 'assign_else':
                default_expr = opt['expression']

        return {
            'type': 'assign',
            'table_name': self.current_table,
            'target': target_str,
            'expression': expr_str,
            'conditions': conditions,
            'operators': operators,
            'query_str': query_str,
            'by_cols': by_cols,
            'default_expr': default_expr,
            'cases': None,
        }
    
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
                    elif isinstance(value, _LiteralStr):
                        query_parts.append(f'{column} {comparator} "{value}"')
                    elif isinstance(value, str):
                        # Unquoted identifier — treat as column reference (no quotes)
                        query_parts.append(f"{column} {comparator} {value}")
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

    def dq_cols(self, *cols):
        return [str(col) for col in cols]

    def quality_condition(self, condition_list):
        temp = self._build_conditional_statement(condition_list)
        return {
            'rule': 'condition',
            'conditions': temp['ast']['conditions'],
            'operators': temp['ast']['operators'],
            'query_str': temp['query_str'],
        }

    def quality_rule(self, quality):
        return quality

    def quality_unique(self, cols):
        return {'rule': 'unique', 'columns': cols}

    def quality_not_null(self, cols, *_):
        return {'rule': 'not_null', 'columns': cols}

    def _data_quality_statement(self, mode, quality):
        return {
            'type': 'data_quality',
            'table_name': self.current_table,
            'mode': mode,
            **quality,
        }

    def assert_statement(self, quality):
        return self._data_quality_statement('assert', quality)

    def check_statement(self, quality):
        return self._data_quality_statement('check', quality)
    
    def select_statement(self, *items):
        """Handle select statements to select specific columns"""
        columns = []
        renames = {}

        for item in items:
            if isinstance(item, dict):
                if item.get('type') in ('var', 'compile_ref'):
                    columns.append(item)
                else:
                    col = item['column']
                    columns.append(col)
                    if 'alias' in item:
                        renames[col] = item['alias']
            else:
                columns.append(str(item))

        return {
            'type': 'select',
            'table_name': self.current_table,
            'columns': columns,
            'renames': renames,
        }

    def select_matches_statement(self, pattern):
        return {
            'type': 'select',
            'table_name': self.current_table,
            'columns': [],
            'renames': {},
            'column_match': pattern,
        }

    def agg_statement(self, *items):
        """Whole-table aggregation with no group-by columns (standalone agg)."""
        agg_list = []
        for item in items:
            if isinstance(item, dict) and 'func' in item:
                agg_list.append(item)
            elif isinstance(item, list):
                agg_list.extend(d for d in item if isinstance(d, dict) and 'func' in d)
        return {
            'type': 'groupby',
            'table_name': self.current_table,
            'by': [],
            'agg_list': agg_list,
        }

    def select_item(self, col, alias=None):
        if isinstance(col, dict) and col.get('type') == 'var':
             # Variable reference cannot have alias in this simple implementation
             # or we could support it if it's a single column
             return col
        if isinstance(col, dict) and col.get('type') == 'compile_ref' and not alias:
            return col
        
        if alias:
            return {'column': col if isinstance(col, dict) else str(col), 'alias': str(alias)}
        return {'column': col if isinstance(col, dict) else str(col)}
    
    def PYTHON_VAR(self, token):
        return {'type': 'var', 'name': str(token)[1:]}

    def COMPILE_REF(self, token):
        return {'type': 'compile_ref', 'path': str(token)}

    def drop_statement(self, *cols):
        return {
            'type': 'drop',
            'table_name': self.current_table,
            'columns': [str(c) for c in cols]
        }

    def drop_matches_statement(self, pattern):
        return {
            'type': 'drop',
            'table_name': self.current_table,
            'columns': [],
            'column_match': pattern,
        }

    def fillna_statement(self, *args):
        # Three forms: 
        # 1. fillna value (fill all columns)
        # 2. fillna \n indent col=value... (old indented syntax)
        # 3. fillna col1 val1, col2 val2... (new comma-separated syntax)
        if len(args) == 1:
            if not isinstance(args[0], list):
                # Case 1: fillna value
                return {'type': 'fillna', 'table_name': self.current_table, 'value': args[0], 'per_col': {}}
            else:
                # Case 2: fillna \n indent col_params (args[0] is list from fillna_col_params)
                col_params = args[0]
                return {'type': 'fillna', 'table_name': self.current_table, 'value': None, 'per_col': dict(col_params)}
        elif len(args) == 2:
            # Case 3: fillna col_list (args[1] is list from fillna_col_list)
            col_params = args[1] if len(args) > 1 else []
            return {'type': 'fillna', 'table_name': self.current_table, 'value': None, 'per_col': dict(col_params)}
        else:
            # Fallback
            return {'type': 'fillna', 'table_name': self.current_table, 'value': None, 'per_col': {}}

    def fillna_col_list(self, *items):
        return list(items)

    def fillna_col_item(self, col, val):
        return (str(col), val)

    def fillna_col_params(self, *params):
        return list(params)

    def fillna_col_param(self, col, val):
        return (str(col), val)

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
        return {'type': 'concat', 'table_name': self.current_table, 'tables': [str(t) for t in tables]}

    def intersect_statement(self, *tables):
        return {'type': 'intersect', 'table_name': self.current_table, 'tables': [str(t) for t in tables]}

    def exclude_statement(self, *tables):
        return {'type': 'exclude', 'table_name': self.current_table, 'tables': [str(t) for t in tables]}

    def round_statement(self, *args):
        cols = []
        decimals = None
        alias = None
        for a in args:
            if isinstance(a, (int, float)):
                decimals = int(a)
            elif decimals is None:
                cols.append(str(a))
            else:
                alias = str(a)
        if alias and len(cols) != 1:
            raise ValueError("'round ... as ...' can only be used with a single source column.")
        return {
            'type': 'round',
            'table_name': self.current_table,
            'columns': cols,
            'decimals': decimals,
            'alias': alias,
        }

    def cast_statement(self, *args):
        # args: IDENTIFIER... CAST_TYPE [CAST_STRICT]
        cols = []
        cast_type = None
        strict = False
        for a in args:
            s = str(a)
            if s == 'strict':
                strict = True
            elif s in ('int', 'integer', 'float', 'string', 'str', 'bool', 'boolean', 'datetime'):
                cast_type = s
            else:
                cols.append(s)
        return {
            'type': 'cast',
            'table_name': self.current_table,
            'columns': cols,
            'cast_type': cast_type,
            'strict': strict,
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
        if not isinstance(func, dict) or func.get('type') != 'var':
            raise ValueError(
                "Python function used with apply must be prefixed with ':'. "
                "Use apply :function_name."
            )
        return {
            'type': 'apply',
            'table_name': self.current_table,
            'func': func['name']
        }

    def delete_statement(self, *args):
        name = str(args[0])
        return {'type': 'delete', 'name': name}

    def show_statement(self, *args):
        mode = 'df'
        for arg in args:
            if isinstance(arg, Token) and arg.type == 'SHOW_MODE':
                mode = str(arg)  # 'head', 'summary', 'shape', or 'columns'
        return {
            'type': 'show',
            'table_name': self.current_table,
            'mode': mode,
        }

    def plot_show(self):
        return {'key': 'show'}

    def plot_on(self, target):
        return {'type': 'plot_on', 'target': str(target)}

    def agg_plot_show(self):
        return {'key': 'show'}

    def table_show(self):
        return {'key': 'show'}

    def merge_statement(self, *args):
        """Handle merge statements"""

        # MERGE_TYPE is optional and comes before "merge".  Distinguish it from
        # RIGHT_TABLE by checking the Token's type attribute.
        idx = 0
        merge_type = 'inner'
        if args and isinstance(args[0], Token) and args[0].type == 'MERGE_TYPE':
            merge_type = str(args[0])
            idx = 1

        # Next token is RIGHT_TABLE
        right_table = args[idx] if idx < len(args) else None
        idx += 1

        # Remaining args are optional: keys (Tree with data='keys') or params (list)
        keys = None
        params = None

        for arg in args[idx:]:
            if isinstance(arg, Tree) and str(arg.data) == 'keys':
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

    def unpivot_statement(self, *args):
        id_vars = []
        value_vars = []
        var_name = 'variable'
        value_name = 'value'
        for arg in args[0]:
            if isinstance(arg, dict):
                t = arg['type']
                if t == 'id':
                    id_vars = arg['columns']
                elif t == 'cols':
                    value_vars = arg['columns']
                elif t == 'name':
                    var_name = arg['name']
                elif t == 'value_name':
                    value_name = arg['name']
        return {
            'type': 'unpivot',
            'table_name': self.current_table,
            'id_vars': id_vars,
            'value_vars': value_vars,
            'var_name': var_name,
            'value_name': value_name,
        }

    def unpivot_args(self, *args):
        return list(args)

    def unpivot_id(self, *cols):
        return {'type': 'id', 'columns': [str(c) for c in cols]}

    def unpivot_cols(self, *cols):
        return {'type': 'cols', 'columns': [str(c) for c in cols]}

    def unpivot_name(self, name):
        return {'type': 'name', 'name': str(name).strip('"').strip("'")}

    def unpivot_value_name(self, name):
        return {'type': 'value_name', 'name': str(name).strip('"').strip("'")}

    # -------------------------------------------------------------------------
    # Shared window helpers
    # -------------------------------------------------------------------------

    def window_by(self, *cols):
        return {'type': 'window_by', 'cols': [str(c) for c in cols]}

    def window_order(self, col):
        return {'type': 'window_order', 'col': str(col)}

    def window_min_periods(self, value):
        return {'type': 'window_min_periods', 'value': int(value)}

    def window_opts(self, *args):
        return list(args)

    def _extract_window_opts(self, args):
        """Pop trailing window_opts list from args; return (remaining, opts)."""
        args = list(args)
        opts = args.pop() if args and isinstance(args[-1], list) else []
        return args, opts

    def _parse_window_common(self, opts):
        """Extract partition cols and order col from a window_opts list."""
        partition = []
        order_col = None
        min_periods = None
        for item in opts:
            if isinstance(item, dict) and item.get('type') == 'window_by':
                partition = item['cols']
            elif isinstance(item, dict) and item.get('type') == 'window_order':
                order_col = item['col']
            elif isinstance(item, dict) and item.get('type') == 'window_min_periods':
                min_periods = item['value']
        return partition, order_col, min_periods

    # -------------------------------------------------------------------------
    # rank
    # -------------------------------------------------------------------------

    def rank_statement(self, *args):
        col = str(args[0])
        remaining, opts = self._extract_window_opts(args[1:])
        ascending = True
        pct = False
        if remaining and hasattr(remaining[0], 'type') and remaining[0].type == 'SORT_TYPE':
            ascending = str(remaining[0]) == 'asc'
            remaining = list(remaining[1:])
        if remaining and hasattr(remaining[0], 'type') and remaining[0].type == 'RANK_PCT':
            pct = True
            remaining = list(remaining[1:])
        result_col = str(remaining[0])
        partition, _, _ = self._parse_window_common(opts)
        return {
            'type': 'rank',
            'table_name': self.current_table,
            'column': col,
            'ascending': ascending,
            'pct': pct,
            'partition': partition,
            'result_col': result_col,
        }

    # -------------------------------------------------------------------------
    # lag / lead
    # -------------------------------------------------------------------------

    def shift_statement(self, *args):
        func = str(args[0])
        col = str(args[1])
        periods = int(args[2])
        remaining, opts = self._extract_window_opts(args[3:])
        result_col = str(remaining[0])
        partition, order_col, _ = self._parse_window_common(opts)
        return {
            'type': 'shift',
            'table_name': self.current_table,
            'func': func,
            'column': col,
            'periods': periods,
            'partition': partition,
            'order_col': order_col,
            'result_col': result_col,
        }

    # -------------------------------------------------------------------------
    # cumulative functions
    # -------------------------------------------------------------------------

    def cumulative_statement(self, *args):
        func = str(args[0])
        col = str(args[1])
        remaining, opts = self._extract_window_opts(args[2:])
        result_col = str(remaining[0])
        partition, order_col, _ = self._parse_window_common(opts)
        return {
            'type': 'cumulative',
            'table_name': self.current_table,
            'func': func,
            'column': col,
            'partition': partition,
            'order_col': order_col,
            'result_col': result_col,
        }

    # -------------------------------------------------------------------------
    # rolling
    # -------------------------------------------------------------------------

    def rolling_statement(self, *args):
        func = str(args[0])
        col = str(args[1])
        window = int(args[2])
        remaining, opts = self._extract_window_opts(args[3:])
        result_col = str(remaining[0])
        partition, order_col, min_periods = self._parse_window_common(opts)
        if min_periods is not None and (min_periods < 0 or min_periods > window):
            raise ValueError("rolling min_periods must be between 0 and the rolling window size.")
        return {
            'type': 'rolling',
            'table_name': self.current_table,
            'func': func,
            'column': col,
            'window': window,
            'min_periods': min_periods,
            'partition': partition,
            'order_col': order_col,
            'result_col': result_col,
        }

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

    def agg_line(self, *items):
        """Single agg line: 'agg func col [as name], ...' — returns list of agg_item dicts."""
        result = []
        for item in items:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, list):
                result.extend(d for d in item if isinstance(d, dict))
        return result

    def agg_clause(self, *lines):
        """Flatten agg_line results into a single list of agg_item dicts."""
        result = []
        for line in lines:
            if isinstance(line, list):
                result.extend(line)
            elif isinstance(line, dict):
                result.append(line)
        return result

    def wmean_bracket_item(self, col, weight, alias=None):
        """wmean(col, weight) — bracket form, col-first for backward compat with wavg."""
        res = {'func': 'wmean', 'column': str(col), 'weight': str(weight)}
        if alias:
            res['alias'] = str(alias)
        return res

    def wmean_alias_item(self, weight, col, alias):
        """wmean weight col as alias — space form, single col with alias."""
        return {'func': 'wmean', 'column': str(col), 'weight': str(weight), 'alias': str(alias)}

    def wmean_cols_item(self, *args):
        """wmean weight col1 col2 ... — space form, one or more cols, no alias.
        Lark passes (IDENTIFIER | PYTHON_VAR)+ as a single list argument."""
        # Lark collapses (TOKEN)+ into one list passed as a single arg
        items = args[0] if (len(args) == 1 and isinstance(args[0], list)) else list(args)
        weight = str(items[0])
        return [{'func': 'wmean', 'column': str(c), 'weight': weight} for c in items[1:]]

    @staticmethod
    def _quantile_probability(func, q):
        q_value = float(q)
        if str(func) == 'percentile':
            q_value = q_value / 100.0
        if not 0 <= q_value <= 1:
            label = 'percentile' if str(func) == 'percentile' else 'quantile'
            upper = 100 if label == 'percentile' else 1
            raise ValueError(f"{label} must be between 0 and {upper}.")
        return q_value

    def quantile_bracket_item(self, func, col, q, alias=None):
        res = {
            'func': str(func),
            'column': col if isinstance(col, dict) and col.get('type') == 'var' else str(col),
            'q': self._quantile_probability(func, q),
        }
        if alias:
            res['alias'] = str(alias)
        return res

    def quantile_space_item(self, func, col, q, alias=None):
        return self.quantile_bracket_item(func, col, q, alias)

    def custom_agg_cols(self, *columns):
        return list(columns)

    def custom_agg_item(self, func, columns):
        """Custom Python aggregation: agg :func col1 col2 ..."""
        func_name = func['name'] if isinstance(func, dict) and func.get('type') == 'var' else str(func)
        res = {'func': func_name, 'columns': columns, 'custom': True}
        return res

    def custom_agg_alias_item(self, func, columns, alias):
        res = self.custom_agg_item(func, columns)
        if alias:
            res['alias'] = str(alias)
        return res

    def custom_agg_col_arg(self, col):
        return {'kind': 'column', 'value': col}

    def custom_agg_kw_arg(self, name, value):
        return {'kind': 'kw', 'name': str(name), 'value': self._convert_value(value)}

    def custom_agg_arg(self, arg):
        return arg

    def custom_agg_value(self, value):
        return self._convert_value(value)

    def custom_agg_list_value(self, *items):
        return [self._convert_value(item) for item in items] if items else []

    def quantile_value(self, value):
        return self._convert_value(value)

    def custom_agg_args(self, *args):
        return list(args)

    def custom_agg_bracket_item(self, func, args=None):
        """Custom Python aggregation: agg :func(col1, col2, key=value)."""
        args = args or []
        columns = [arg['value'] for arg in args if arg.get('kind') == 'column']
        kwargs = {arg['name']: arg['value'] for arg in args if arg.get('kind') == 'kw'}
        res = self.custom_agg_item(func, columns)
        if kwargs:
            res['kwargs'] = kwargs
        return res

    def custom_agg_bracket_alias_item(self, func, args, alias):
        res = self.custom_agg_bracket_item(func, args)
        if alias:
            res['alias'] = str(alias)
        return res

    # Keep wavg_item as an alias so old parsed ASTs / direct calls still work
    def wavg_item(self, col, weight, alias=None):
        return self.wmean_bracket_item(col, weight, alias)

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

    def QUANTILE_FUNCTION(self, token):
        """Handle quantile/percentile aggregation function tokens."""
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
        on = None
        kwargs = {}
        kwargs_str = ""

        identifiers = []
        show = False
        for arg in args:
            if isinstance(arg, dict) and arg.get('type') == 'plot_on':
                on = arg['target']
            elif isinstance(arg, Token) and arg.type != '_NL':
                identifiers.append(str(arg))
            elif isinstance(arg, str):
                identifiers.append(arg)
            elif isinstance(arg, list):
                # Extract 'show' pseudo-param before passing to _plot_kwargs
                filtered = []
                for p in arg:
                    if isinstance(p, dict) and p.get('key') == 'show' and 'value' not in p:
                        show = True
                    else:
                        filtered.append(p)
                kwargs, kwargs_str = self._plot_kwargs(filtered)

        if on:
            # 'plot line on myplot' — the single identifier is the kind, not a chart name
            if len(identifiers) >= 1:
                kind = identifiers[0]
        elif len(identifiers) == 1:
            name = identifiers[0]
        elif len(identifiers) >= 2:
            kind = identifiers[0]
            name = identifiers[1]

        # Extract structural params so they don't get forwarded to df.plot()
        by_col = kwargs.pop('by', None)
        n_cols = kwargs.pop('cols', None)
        style = kwargs.pop('style', None)
        canvas = kwargs.pop('canvas', None)  # per-plot canvas override

        # Rebuild kwargs_str if any structural params were removed
        if any(p is not None for p in [by_col, n_cols, style, canvas]):
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
            'on': on,
            'kwargs': kwargs,
            'kwargs_str': kwargs_str,
            'by': by_col,
            'cols': n_cols,
            'style': style,
            'canvas': canvas,  # None means inherit global setting
            'show': show,
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

    def agg_plot_statement(self, *args):
        kind = None
        name = None
        params = []
        identifiers = []
        for arg in args:
            if isinstance(arg, Token) and arg.type != '_NL':
                identifiers.append(str(arg))
            elif isinstance(arg, str):
                identifiers.append(arg)
            elif isinstance(arg, list):
                params = arg
        if len(identifiers) == 1:
            name = identifiers[0]
        elif len(identifiers) >= 2:
            kind = identifiers[0]
            name = identifiers[1]

        x_col = None
        x_label = None
        y_items = []   # list of {func, col}
        y_label = None
        by_col = None
        n_cols = None
        canvas = None
        show = False
        for p in params:
            k = p.get('key')
            if k == 'x':
                x_col = p['col']
                x_label = p.get('label')
            elif k == 'y':
                y_items.extend(p['items'])
                if p.get('label'):
                    y_label = p['label']
            elif k == 'by':
                by_col = p['col']
            elif k == 'cols':
                n_cols = p['value']
            elif k == 'canvas':
                canvas = p['value']
            elif k == 'show':
                show = True

        return {
            'type': 'pivot_plot',
            'table_name': self.current_table,
            'name': name,
            'kind': kind or 'line',
            'x': x_col,
            'x_label': x_label,
            'y_items': y_items,   # list of {func, col}
            'y_label': y_label,
            'by': by_col,
            'cols': n_cols,
            'canvas': canvas,
            'show': show,
        }

    def agg_plot_params(self, *params):
        return list(params)

    def agg_plot_x(self, col):
        return {'key': 'x', 'col': str(col), 'label': None}

    def agg_plot_x_labeled(self, col, label):
        return {'key': 'x', 'col': str(col), 'label': str(label).strip('"').strip("'")}

    def agg_plot_y_item(self, func, col):
        return {'func': str(func), 'col': str(col)}

    def agg_plot_y(self, *args):
        # args: agg_plot_y_item dicts (commas filtered by Lark)
        items = [a for a in args if isinstance(a, dict)]
        return {'key': 'y', 'items': items, 'label': None}

    def agg_plot_y_labeled(self, *args):
        # args: agg_plot_y_item dicts, then STRING label
        items = [a for a in args if isinstance(a, dict)]
        labels = [a for a in args if not isinstance(a, dict)]
        label = str(labels[0]).strip('"').strip("'") if labels else None
        return {'key': 'y', 'items': items, 'label': label}

    def agg_plot_by(self, col):
        return {'key': 'by', 'col': str(col)}

    def agg_plot_cols(self, val):
        return {'key': 'cols', 'value': int(val)}

    def agg_plot_canvas(self, val):
        return {'key': 'canvas', 'value': str(val)}

    def table_statement(self, *args):
        # @v_args(inline=True) unpacks tree children as individual args:
        #   no params:   args = ('summary',)            ← bare str
        #   with params: args = ('summary', [dict, ...])  ← str + list
        name = None
        params = []
        for arg in args:
            if isinstance(arg, str):
                name = arg
            elif isinstance(arg, list):
                params = arg
        node = {
            'type': 'gt_table',
            'name': name,
            'table_name': self.current_table,
            'title': None,
            'subtitle': None,
            'font_size': None,
            'font_family': None,
            'stub': None,
            'stub_group': None,
            'stub_label': None,
            'stripe': False,
            'canvas': 'none',
            'labels': [],
            'formats': [],
            'summary': [],
            'spanners': [],
            'style_file': None,
            'show': False,
        }
        for p in params:
            if not isinstance(p, dict):
                continue
            k = p.get('key')
            if k == 'title':       node['title'] = p['value']
            elif k == 'subtitle':  node['subtitle'] = p['value']
            elif k == 'font_size': node['font_size'] = p['value']
            elif k == 'font_family': node['font_family'] = p['value']
            elif k == 'stub':
                node['stub'] = p['value']
                node['stub_group'] = p.get('group')
                node['stub_label'] = p.get('label')
            elif k == 'stripe':    node['stripe'] = True
            elif k == 'canvas':    node['canvas'] = p['value']
            elif k == 'style':        node['style_file'] = p['value']
            elif k == 'show':         node['show'] = True
            elif k == 'summary_line': node['summary'].extend(p.get('specs', []))
            elif k == 'label_line':   node['labels'].extend(p.get('specs', []))
            elif k == 'format_line': node['formats'].append({kk: vv for kk, vv in p.items() if kk != 'key'})
            elif k == 'spanner':   node['spanners'].append(p)
            elif k == 'auto_spanner': node['spanners'].append({'type': 'auto'})
        return node

    def table_params(self, *params):
        return list(params)

    def table_title(self, s):
        return {'key': 'title', 'value': str(s).strip('"').strip("'")}

    def table_subtitle(self, s):
        return {'key': 'subtitle', 'value': str(s).strip('"').strip("'")}

    def table_font_size(self, n):
        return {'key': 'font_size', 'value': float(n)}

    def table_font_family(self, s):
        return {'key': 'font_family', 'value': str(s).strip('"').strip("'")}

    def table_stub(self, col):
        return {'key': 'stub', 'value': str(col)}

    def table_stub_labeled(self, col, label):
        return {'key': 'stub', 'value': str(col),
                'label': str(label).strip('"').strip("'")}

    def table_stub_grouped(self, col, group):
        return {'key': 'stub', 'value': str(col), 'group': str(group)}

    def table_stub_grouped_labeled(self, col, group, label):
        return {'key': 'stub', 'value': str(col), 'group': str(group),
                'label': str(label).strip('"').strip("'")}

    def spanner_cols(self, *cols):
        return [str(c) for c in cols]

    def table_spanner_line(self, cols, label):
        return {'key': 'spanner', 'type': 'manual',
                'label': str(label).strip('"').strip("'"),
                'columns': list(cols)}

    def table_auto_spanner(self):
        return {'key': 'auto_spanner', 'type': 'auto'}

    def table_stripe(self):
        return {'key': 'stripe', 'value': True}

    def table_canvas(self, val):
        return {'key': 'canvas', 'value': str(val)}

    def table_style(self, path):
        return {'key': 'style', 'value': str(path).strip('"').strip("'")}

    # Default labels for each aggregation function
    _SUMMARY_LABELS = {
        'sum': 'Total', 'mean': 'Mean', 'min': 'Min',
        'max': 'Max', 'median': 'Median', 'count': 'Count',
    }

    def summary_spec_labeled(self, fn, label):
        return {'fn': str(fn), 'label': str(label).strip('"').strip("'")}

    def summary_spec_bare(self, fn):
        fn = str(fn)
        return {'fn': fn, 'label': self._SUMMARY_LABELS.get(fn, fn.capitalize())}

    def table_summary_line(self, *specs):
        return {'key': 'summary_line', 'specs': list(specs)}

    def label_col_spec(self, col, label):
        return {'col': str(col), 'label': str(label).strip('"').strip("'")}

    def table_label_line(self, *specs):
        return {'key': 'label_line', 'specs': list(specs)}

    def fmt_number(self, decimals):
        return {'fmt': 'number', 'decimals': float(decimals)}

    def fmt_number_noprec(self):
        return {'fmt': 'number', 'decimals': 2}

    def fmt_integer(self):
        return {'fmt': 'integer'}

    def fmt_currency(self, code):
        return {'fmt': 'currency', 'code': str(code)}

    def fmt_currency_nocode(self):
        return {'fmt': 'currency', 'code': 'USD'}

    def fmt_percent(self, decimals):
        return {'fmt': 'percent', 'decimals': float(decimals)}

    def fmt_percent_noprec(self):
        return {'fmt': 'percent', 'decimals': 1}

    def fmt_date(self):
        return {'fmt': 'date'}

    def table_format_col(self, col, fmt_dict):
        return {'key': 'format_line', 'col': str(col), **fmt_dict}

    def table_format_all(self, fmt_dict):
        return {'key': 'format_line', 'col': None, **fmt_dict}

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

    def condition_in_ref(self, column, ref):
        return {'column': str(column), 'comparator': 'in', 'value': ref}

    def condition_in_name(self, column, name):
        return {'column': str(column), 'comparator': 'in', 'value': {'type': 'list_ref', 'name': str(name)}}

    def condition_not_in_list(self, column, lst):
        return {'column': str(column), 'comparator': 'not in', 'value': lst}

    def condition_not_in_var(self, column, var):
        return {'column': str(column), 'comparator': 'not in', 'value': var}

    def condition_not_in_ref(self, column, ref):
        return {'column': str(column), 'comparator': 'not in', 'value': ref}

    def condition_not_in_name(self, column, name):
        return {'column': str(column), 'comparator': 'not in', 'value': {'type': 'list_ref', 'name': str(name)}}
    
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
        # Remove quotes; wrap in _LiteralStr so it can be distinguished from
        # unquoted IDENTIFIER tokens (column references) when building queries.
        return _LiteralStr(str(token)[1:-1])
    
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
        if isinstance(val, (int, float, list, tuple, dict)) or val is None:
            return val
        # Preserve _LiteralStr (quoted string) — don't try numeric conversion
        if isinstance(val, _LiteralStr):
            return val
        val_str = str(val)
        # Try to convert to number (handles both NUMBER and SIGNED_NUMBER tokens)
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

    def _bulk_sources_code(self, ast_node):
        sources = ast_node['sources']
        if sources['kind'] == 'var':
            return sources['value']['name']
        return repr(sources['value'])

    def _bulk_aliases_code(self, ast_node):
        aliases = ast_node['aliases']
        if aliases['kind'] == 'var':
            return aliases['value']['name']
        return repr(aliases['value'])

    def _bulk_source_label_code(self, ast_node):
        source_value = ast_node.get('source_value', 'filename')
        if source_value == 'path':
            return "str(_pvt_src)"
        if source_value == 'stem':
            return "os.path.splitext(os.path.basename(str(_pvt_src)))[0]"
        return "os.path.basename(str(_pvt_src))"

    def _bulk_read_pandas_lines(self, indent=""):
        kw = getattr(self, '_bulk_current_kwargs_str', '')
        return [
            f"{indent}def _pvt_read_file(_pvt_src):",
            f"{indent}    _pvt_src = str(_pvt_src)",
            f"{indent}    _pvt_ext = _pvt_src.rsplit('.', 1)[-1].lower() if '.' in _pvt_src else ''",
            f"{indent}    if _pvt_ext == 'parquet':",
            f"{indent}        return pd.read_parquet(_pvt_src{kw})",
            f"{indent}    if _pvt_ext == 'csv':",
            f"{indent}        return pd.read_csv(_pvt_src{kw})",
            f"{indent}    raise ValueError(f\"bulk load supports CSV and Parquet files, got '{{_pvt_src}}'\")",
        ]

    def _bulk_validate_sources_lines(self):
        return [
            "_pvt_sources = list(_pvt_sources)",
            "if not _pvt_sources:",
            "    raise ValueError('bulk load requires at least one file')",
        ]

    def _bulk_expand_sources_lines(self):
        return [
            "def _pvt_expand_sources(_pvt_raw_sources):",
            "    if isinstance(_pvt_raw_sources, (str, os.PathLike)):",
            "        _pvt_iter = [_pvt_raw_sources]",
            "    else:",
            "        _pvt_iter = list(_pvt_raw_sources)",
            "    _pvt_expanded = []",
            "    _pvt_supported_exts = {'csv', 'parquet'}",
            "    for _pvt_item in _pvt_iter:",
            "        _pvt_path = os.fspath(_pvt_item)",
            "        if os.path.isdir(_pvt_path):",
            "            _pvt_names = sorted(os.listdir(_pvt_path))",
            "            if not _pvt_names:",
            "                raise ValueError(f'bulk load folder is empty: {_pvt_path}')",
            "            _pvt_dir_files = []",
            "            _pvt_dir_exts = set()",
            "            for _pvt_name in _pvt_names:",
            "                _pvt_file = os.path.join(_pvt_path, _pvt_name)",
            "                if not os.path.isfile(_pvt_file):",
            "                    raise ValueError(f'bulk load folder contains non-file entry: {_pvt_file}')",
            "                _pvt_ext = _pvt_file.rsplit('.', 1)[-1].lower() if '.' in _pvt_file else ''",
            "                if _pvt_ext not in _pvt_supported_exts:",
            "                    raise ValueError(f'bulk load folder supports only CSV and Parquet files, got: {_pvt_file}')",
            "                _pvt_dir_exts.add(_pvt_ext)",
            "                _pvt_dir_files.append(_pvt_file)",
            "            if len(_pvt_dir_exts) > 1:",
            "                _pvt_formats = ', '.join(sorted(_pvt_dir_exts))",
            "                raise ValueError(f'bulk load folder must contain one file format, got: {_pvt_formats}')",
            "            _pvt_expanded.extend(_pvt_dir_files)",
            "        else:",
            "            _pvt_ext = _pvt_path.rsplit('.', 1)[-1].lower() if '.' in _pvt_path else ''",
            "            if _pvt_ext not in _pvt_supported_exts:",
            "                raise ValueError(f'bulk load supports CSV and Parquet files, got: {_pvt_path}')",
            "            _pvt_expanded.append(_pvt_path)",
            "    return _pvt_expanded",
        ]

    def _bulk_validate_aliases_lines(self):
        return [
            "_pvt_aliases = list(_pvt_aliases)",
            "if len(_pvt_sources) != len(_pvt_aliases):",
            "    raise ValueError(f'bulk load source/alias count mismatch: {len(_pvt_sources)} files, {len(_pvt_aliases)} aliases')",
            "for _pvt_alias in _pvt_aliases:",
            "    if not isinstance(_pvt_alias, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', _pvt_alias):",
            "        raise ValueError(f\"Invalid bulk load table alias '{_pvt_alias}'. Use a valid Python identifier.\")",
        ]

    def generate_bulk_load_pandas(self, ast_node):
        self._bulk_current_kwargs_str = ast_node.get('kwargs_str', '')
        t = ast_node.get('table_name')
        src_expr = self._bulk_sources_code(ast_node)
        source_col = ast_node.get('source_column', 'source')
        label_expr = self._bulk_source_label_code(ast_node)

        lines = [
            "import os",
            "import re",
            *self._bulk_expand_sources_lines(),
            f"_pvt_sources = {src_expr}",
            "_pvt_sources = _pvt_expand_sources(_pvt_sources)",
            *self._bulk_validate_sources_lines(),
            *self._bulk_read_pandas_lines(),
        ]

        if ast_node['mode'] == 'concat':
            marker = f"#__pivotal__\n__table_name__ = '{t}'\n#__pivotal__"
            lines.extend([
                "_pvt_frames = []",
                "for _pvt_src in _pvt_sources:",
                "    _pvt_df = _pvt_read_file(_pvt_src)",
                f"    _pvt_df[{source_col!r}] = {label_expr}",
                "    _pvt_frames.append(_pvt_df)",
                f"{t} = pd.concat(_pvt_frames, ignore_index=True, sort=False)",
                f"{_SANITISE_HELPER}"
                f"_pvt_new_cols, _pvt_renamed = _pvt_sanitise({t}.columns.tolist())",
                f"{t}.columns = _pvt_new_cols",
                "if _pvt_renamed:",
                "    import warnings",
                f"    warnings.warn(f\"[Pivotal] Column(s) renamed in '{t}': {{_pvt_renamed}}\", UserWarning, stacklevel=2)",
                marker,
            ])
            return "\n".join(lines)

        alias_expr = self._bulk_aliases_code(ast_node)
        lines.extend([
            f"_pvt_aliases = {alias_expr}",
            *self._bulk_validate_aliases_lines(),
            f"{_SANITISE_HELPER}"
            "for _pvt_src, _pvt_alias in zip(_pvt_sources, _pvt_aliases):",
            "    _pvt_df = _pvt_read_file(_pvt_src)",
            "    _pvt_new_cols, _pvt_renamed = _pvt_sanitise(_pvt_df.columns.tolist())",
            "    _pvt_df.columns = _pvt_new_cols",
            "    if _pvt_renamed:",
            "        import warnings",
            "        warnings.warn(f\"[Pivotal] Column(s) renamed in '{_pvt_alias}': {_pvt_renamed}\", UserWarning, stacklevel=2)",
            "    globals()[_pvt_alias] = _pvt_df",
        ])
        return "\n".join(lines)

    def generate_load_table_pandas(self, ast_node):
        source = ast_node['source']
        table_name_marker = f"#__pivotal__\n__table_name__ = '{ast_node['table_name']}'\n#__pivotal__"

        if isinstance(source, dict) and source.get('type') == 'var':
            # Runtime format detection for variable file paths
            var = source['name']
            kw = ast_node['kwargs_str']
            tname = ast_node['table_name']
            load_table = (
                f"_src = {var}\n"
                f"_ext = _src.rsplit('.', 1)[-1].lower() if '.' in _src else ''\n"
                f"if _ext in ('xlsx', 'xls'):\n"
                f"    {tname} = pd.read_excel(_src{kw})\n"
                f"elif _ext == 'parquet':\n"
                f"    {tname} = pd.read_parquet(_src{kw})\n"
                f"else:\n"
                f"    {tname} = pd.read_csv(_src{kw})"
            )
        else:
            source_str = str(source)
            reader = self._reader_for_source(source_str)
            load_table = f"{ast_node['table_name']} = {reader}({repr(source_str)}{ast_node['kwargs_str']})"

        kw_set = repr(PIVOTAL_KEYWORDS)
        tname = ast_node['table_name']
        sanitise_code = (
            f"{_SANITISE_HELPER}"
            f"_pvt_new_cols, _pvt_renamed = _pvt_sanitise({tname}.columns.tolist())\n"
            f"{tname}.columns = _pvt_new_cols\n"
            f"if _pvt_renamed:\n"
            f"    import warnings\n"
            f"    warnings.warn(f\"[Pivotal] Column(s) renamed in '{tname}': {{_pvt_renamed}}\", UserWarning, stacklevel=2)\n"
        )
        kw_check = (
            f"_kw_cols = [c for c in {tname}.columns if c.lower() in {kw_set}]\n"
            f"if _kw_cols:\n"
            f"    import warnings\n"
            f"    warnings.warn(\n"
            f"        f\"Table '{tname}' has columns that are Pivotal keywords: {{_kw_cols}}. \"\n"
            f"        \"Use a 'python' block to reference them.\",\n"
            f"        UserWarning, stacklevel=2)"
        )
        return f"{load_table}\n{sanitise_code}\n{kw_check}\n{table_name_marker}"

    # ------------------------------------------------------------------
    # from_db code generators (pandas / polars / duckdb / sql)
    # ------------------------------------------------------------------

    def _from_db_source_type(self, source):
        """Return ('sqlite'|'duckdb'|'uri'|'unknown', source_str)."""
        if isinstance(source, dict) and source.get('type') == 'var':
            return 'var', source['name']
        s = str(source)
        ext = s.rsplit('.', 1)[-1].lower() if '.' in s and '://' not in s else ''
        if ext in ('sqlite', 'db', 'sqlite3'):
            return 'sqlite', s
        if ext in ('ddb', 'duckdb'):
            return 'duckdb', s
        if '://' in s:
            return 'uri', s
        return 'unknown', s

    def _from_db_sanitise_pandas(self, alias):
        kw_set = repr(PIVOTAL_KEYWORDS)
        return (
            f"{_SANITISE_HELPER}"
            f"_pvt_new_cols, _pvt_renamed = _pvt_sanitise({alias}.columns.tolist())\n"
            f"{alias}.columns = _pvt_new_cols\n"
            f"if _pvt_renamed:\n"
            f"    import warnings\n"
            f"    warnings.warn(f\"[Pivotal] Column(s) renamed in '{alias}': {{_pvt_renamed}}\", UserWarning, stacklevel=2)\n"
            f"_kw_cols = [c for c in {alias}.columns if c.lower() in {kw_set}]\n"
            f"if _kw_cols:\n"
            f"    import warnings\n"
            f"    warnings.warn(f\"Table '{alias}' has columns that are Pivotal keywords: {{_kw_cols}}. Use a 'python' block to reference them.\", UserWarning, stacklevel=2)\n"
        )

    def _from_db_sanitise_polars(self, alias):
        return (
            f"{_SANITISE_HELPER}"
            f"_, _pvt_renamed = _pvt_sanitise({alias}.columns)\n"
            f"if _pvt_renamed:\n"
            f"    {alias} = {alias}.rename(_pvt_renamed)\n"
        )

    def _from_db_items(self, bodies):
        """Flatten bodies into list of (sql, alias) tuples."""
        items = []
        for b in bodies:
            if b['kind'] == 'load':
                for it in b['items']:
                    items.append((f"SELECT * FROM {it['sql_table']}", it['alias']))
            else:
                items.append((b['sql'], b['alias']))
        return items

    def generate_from_db_pandas(self, ast_node):
        source = ast_node['source']
        bodies = ast_node['bodies']
        items = self._from_db_items(bodies)
        src_type, src_val = self._from_db_source_type(source)
        lines = []
        markers = []

        if src_type == 'sqlite':
            lines.append(f"import sqlite3 as _sqlite3")
            lines.append(f"with _sqlite3.connect({repr(src_val)}) as _conn:")
            for sql, alias in items:
                lines.append(f"    {alias} = pd.read_sql({repr(sql)}, _conn)")
        elif src_type == 'duckdb':
            lines.append(f"import duckdb as _duckdb")
            lines.append(f"with _duckdb.connect({repr(src_val)}) as _dconn:")
            for sql, alias in items:
                lines.append(f"    {alias} = _dconn.execute({repr(sql)}).df()")
        elif src_type == 'uri':
            lines.append(
                f"try:\n"
                f"    from sqlalchemy import create_engine as _sa_engine\n"
                f"    _engine = _sa_engine({repr(src_val)})\n"
            )
            for sql, alias in items:
                lines.append(f"    {alias} = pd.read_sql({repr(sql)}, _engine)")
            lines.append(
                f"except ImportError:\n"
                f"    raise ImportError(\"SQLAlchemy is required for database URIs: pip install sqlalchemy\")"
            )
        else:
            # var — runtime detection
            lines.append(f"_src = {src_val}")
            lines.append(f"_ext = _src.rsplit('.', 1)[-1].lower() if '.' in _src else ''")
            lines.append(f"if _ext in ('sqlite', 'db', 'sqlite3'):")
            lines.append(f"    import sqlite3 as _sqlite3")
            lines.append(f"    with _sqlite3.connect(_src) as _conn:")
            for sql, alias in items:
                lines.append(f"        {alias} = pd.read_sql({repr(sql)}, _conn)")
            lines.append(f"elif _ext in ('ddb', 'duckdb'):")
            lines.append(f"    import duckdb as _duckdb")
            lines.append(f"    with _duckdb.connect(_src) as _dconn:")
            for sql, alias in items:
                lines.append(f"        {alias} = _dconn.execute({repr(sql)}).df()")
            lines.append(f"elif '://' in _src:")
            lines.append(f"    try:")
            lines.append(f"        from sqlalchemy import create_engine as _sa_engine")
            lines.append(f"        _engine = _sa_engine(_src)")
            for sql, alias in items:
                lines.append(f"        {alias} = pd.read_sql({repr(sql)}, _engine)")
            lines.append(f"    except ImportError:")
            lines.append(f"        raise ImportError(\"SQLAlchemy is required for database URIs: pip install sqlalchemy\")")
            lines.append(f"else:")
            lines.append(f"    raise ValueError(f\"Unsupported database source: {{_src}}\")")

        code = '\n'.join(lines)
        for sql, alias in items:
            san = self._from_db_sanitise_pandas(alias)
            marker = f"#__pivotal__\n__table_name__ = '{alias}'\n#__pivotal__"
            markers.append(f"{san}{marker}")
        return code + '\n' + '\n'.join(markers)

    def generate_from_db_polars(self, ast_node):
        source = ast_node['source']
        bodies = ast_node['bodies']
        items = self._from_db_items(bodies)
        src_type, src_val = self._from_db_source_type(source)
        lines = []

        if src_type == 'sqlite':
            lines.append(f"import sqlite3 as _sqlite3; import pandas as _pd")
            lines.append(f"with _sqlite3.connect({repr(src_val)}) as _conn:")
            for sql, alias in items:
                lines.append(f"    {alias} = pl.from_pandas(_pd.read_sql({repr(sql)}, _conn))")
        elif src_type == 'duckdb':
            lines.append(f"import duckdb as _duckdb")
            lines.append(f"with _duckdb.connect({repr(src_val)}) as _dconn:")
            for sql, alias in items:
                lines.append(f"    {alias} = pl.from_pandas(_dconn.execute({repr(sql)}).df())")
        elif src_type == 'uri':
            lines.append(
                f"try:\n"
                f"    import connectorx as _cx\n"
                f"    _cx_avail = True\n"
                f"except ImportError:\n"
                f"    _cx_avail = False\n"
            )
            for sql, alias in items:
                lines.append(
                    f"if _cx_avail:\n"
                    f"    {alias} = pl.read_database({repr(sql)}, {repr(src_val)})\n"
                    f"else:\n"
                    f"    try:\n"
                    f"        from sqlalchemy import create_engine as _sa_engine; import pandas as _pd\n"
                    f"        _engine = _sa_engine({repr(src_val)})\n"
                    f"        {alias} = pl.from_pandas(_pd.read_sql({repr(sql)}, _engine))\n"
                    f"    except ImportError:\n"
                    f"        raise ImportError(\"Install connectorx or sqlalchemy: pip install connectorx sqlalchemy\")"
                )
        else:
            lines.append(f"_src = {src_val}")
            lines.append(f"_ext = _src.rsplit('.', 1)[-1].lower() if '.' in _src else ''")
            lines.append(f"import sqlite3 as _sqlite3; import pandas as _pd")
            lines.append(f"if _ext in ('sqlite', 'db', 'sqlite3'):")
            lines.append(f"    with _sqlite3.connect(_src) as _conn:")
            for sql, alias in items:
                lines.append(f"        {alias} = pl.from_pandas(_pd.read_sql({repr(sql)}, _conn))")
            lines.append(f"elif _ext in ('ddb', 'duckdb'):")
            lines.append(f"    import duckdb as _duckdb")
            lines.append(f"    with _duckdb.connect(_src) as _dconn:")
            for sql, alias in items:
                lines.append(f"        {alias} = pl.from_pandas(_dconn.execute({repr(sql)}).df())")
            lines.append(f"else:")
            lines.append(f"    raise ValueError(f\"Unsupported database source: {{_src}}\")")

        code = '\n'.join(lines)
        post = []
        for sql, alias in items:
            san = self._from_db_sanitise_polars(alias)
            marker = f"#__pivotal__\n__table_name__ = '{alias}'\n#__pivotal__"
            post.append(f"{san}{marker}")
        return code + '\n' + '\n'.join(post)

    def generate_from_db_duckdb(self, ast_node):
        source = ast_node['source']
        bodies = ast_node['bodies']
        items = self._from_db_items(bodies)
        src_type, src_val = self._from_db_source_type(source)
        lines = []

        if src_type == 'sqlite':
            # Read via sqlite3, register as DuckDB tables
            lines.append(f"import sqlite3 as _sqlite3")
            lines.append(f"with _sqlite3.connect({repr(src_val)}) as _conn:")
            for sql, alias in items:
                lines.append(f"    _df_tmp = pd.read_sql({repr(sql)}, _conn)")
                lines.append(f"    _pvt.register('_load_tmp', _df_tmp)")
                lines.append(f"    _pvt.execute('CREATE OR REPLACE TABLE {alias} AS SELECT * FROM _load_tmp')")
        elif src_type == 'duckdb':
            lines.append(f"_ext_conn = _pvt.cursor()")
            lines.append(f"_ext_conn.execute(\"ATTACH {repr(src_val)} AS _ext_db (READ_ONLY)\")")
            for sql, alias in items:
                lines.append(f"_pvt.execute(\"CREATE OR REPLACE TABLE {alias} AS ({sql})\")")
            lines.append(f"_ext_conn.execute(\"DETACH _ext_db\")")
        elif src_type == 'uri':
            # Use DuckDB's extension for the URI type (postgres_scanner etc.)
            lines.append(f"_pvt.execute(\"INSTALL postgres_scanner; LOAD postgres_scanner\" )")
            lines.append(f"_pvt.execute(\"ATTACH {repr(src_val)} AS _ext_db\")")
            for sql, alias in items:
                lines.append(f"_pvt.execute(\"CREATE OR REPLACE TABLE {alias} AS ({sql})\")")
            lines.append(f"_pvt.execute(\"DETACH _ext_db\")")
        else:
            lines.append(f"_src = {src_val}")
            lines.append(f"_ext = _src.rsplit('.', 1)[-1].lower() if '.' in _src else ''")
            lines.append(f"if _ext in ('sqlite', 'db', 'sqlite3'):")
            lines.append(f"    import sqlite3 as _sqlite3")
            lines.append(f"    with _sqlite3.connect(_src) as _conn:")
            for sql, alias in items:
                lines.append(f"        _df_tmp = pd.read_sql({repr(sql)}, _conn)")
                lines.append(f"        _pvt.register('_load_tmp', _df_tmp)")
                lines.append(f"        _pvt.execute('CREATE OR REPLACE TABLE {alias} AS SELECT * FROM _load_tmp')")
            lines.append(f"elif _ext in ('ddb', 'duckdb'):")
            for sql, alias in items:
                lines.append(f"    _pvt.execute(\"CREATE OR REPLACE TABLE {alias} AS ({sql})\")")
            lines.append(f"else:")
            lines.append(f"    raise ValueError(f\"Unsupported database source: {{_src}}\")")

        code = '\n'.join(lines)
        san_lines = []
        for sql, alias in items:
            san = (
                f"{_SANITISE_HELPER}"
                f"_pvt_info = [row[0] for row in _pvt.execute('PRAGMA table_info({alias})').fetchall()]\n"
                f"_, _pvt_renamed = _pvt_sanitise(_pvt_info)\n"
                f"if _pvt_renamed:\n"
                f"    _pvt.execute('ALTER TABLE {alias} RENAME COLUMN ' + ' ; ALTER TABLE {alias} RENAME COLUMN '.join(f'\"{{o}}\" TO \"{{n}}\"' for o,n in _pvt_renamed.items()))\n"
            )
            marker = f"#__pivotal__\n__table_name__ = '{alias}'\n#__pivotal__"
            san_lines.append(f"{san}{marker}")
        return code + '\n' + '\n'.join(san_lines)

    def generate_from_db_sql(self, ast_node):
        # SQL/CTE backend — emit a comment pointing to the source
        source = ast_node['source']
        bodies = ast_node['bodies']
        items = self._from_db_items(bodies)
        src_type, src_val = self._from_db_source_type(source)
        lines = [f"-- [from database: {src_val}]"]
        for sql, alias in items:
            lines.append(f"-- {alias}: {sql}")
        return '\n'.join(lines)

    def generate_load_table_polars(self, ast_node):
        source = ast_node['source']
        table_name_marker = f"#__pivotal__\n__table_name__ = '{ast_node['table_name']}'\n#__pivotal__"

        if isinstance(source, dict) and source.get('type') == 'var':
            var = source['name']
            kw = ast_node['kwargs_str']
            tname = ast_node['table_name']
            load_table = (
                f"_src = {var}\n"
                f"_ext = _src.rsplit('.', 1)[-1].lower() if '.' in _src else ''\n"
                f"if _ext in ('xlsx', 'xls'):\n"
                f"    {tname} = pl.read_excel(_src{kw})\n"
                f"elif _ext == 'parquet':\n"
                f"    {tname} = pl.read_parquet(_src{kw})\n"
                f"else:\n"
                f"    {tname} = pl.read_csv(_src{kw})"
            )
        else:
            source_str = str(source)
            ext = source_str.rsplit('.', 1)[-1].lower() if '.' in source_str else ''
            tname = ast_node['table_name']
            if ext in ('xlsx', 'xls'):
                load_table = f"{tname} = pl.read_excel({repr(source_str)}{ast_node['kwargs_str']})"
            elif ext == 'parquet':
                load_table = f"{tname} = pl.read_parquet({repr(source_str)}{ast_node['kwargs_str']})"
            else:
                load_table = f"{tname} = pl.read_csv({repr(source_str)}{ast_node['kwargs_str']})"

        tname = ast_node['table_name']
        sanitise_code = (
            f"{_SANITISE_HELPER}"
            f"_, _pvt_renamed = _pvt_sanitise({tname}.columns)\n"
            f"if _pvt_renamed:\n"
            f"    {tname} = {tname}.rename(_pvt_renamed)\n"
            f"    import warnings\n"
            f"    warnings.warn(f\"[Pivotal] Column(s) renamed in '{tname}': {{_pvt_renamed}}\", UserWarning, stacklevel=2)\n"
        )
        return f"{load_table}\n{sanitise_code}\n{table_name_marker}"

    def _bulk_read_polars_lines(self):
        kw = getattr(self, '_bulk_current_kwargs_str', '')
        return [
            "def _pvt_read_file(_pvt_src):",
            "    _pvt_src = str(_pvt_src)",
            "    _pvt_ext = _pvt_src.rsplit('.', 1)[-1].lower() if '.' in _pvt_src else ''",
            "    if _pvt_ext == 'parquet':",
            f"        return pl.read_parquet(_pvt_src{kw})",
            "    if _pvt_ext == 'csv':",
            f"        return pl.read_csv(_pvt_src{kw})",
            "    raise ValueError(f\"bulk load supports CSV and Parquet files, got '{_pvt_src}'\")",
        ]

    def generate_bulk_load_polars(self, ast_node):
        self._bulk_current_kwargs_str = ast_node.get('kwargs_str', '')
        t = ast_node.get('table_name')
        src_expr = self._bulk_sources_code(ast_node)
        source_col = ast_node.get('source_column', 'source')
        label_expr = self._bulk_source_label_code(ast_node)

        lines = [
            "import os",
            "import re",
            *self._bulk_expand_sources_lines(),
            f"_pvt_sources = {src_expr}",
            "_pvt_sources = _pvt_expand_sources(_pvt_sources)",
            *self._bulk_validate_sources_lines(),
            *self._bulk_read_polars_lines(),
        ]

        if ast_node['mode'] == 'concat':
            marker = f"#__pivotal__\n__table_name__ = '{t}'\n#__pivotal__"
            lines.extend([
                "_pvt_frames = []",
                "for _pvt_src in _pvt_sources:",
                "    _pvt_df = _pvt_read_file(_pvt_src)",
                f"    _pvt_df = _pvt_df.with_columns(pl.lit({label_expr}).alias({source_col!r}))",
                "    _pvt_frames.append(_pvt_df)",
                f"{t} = pl.concat(_pvt_frames, how='diagonal')",
                f"{_SANITISE_HELPER}"
                f"_, _pvt_renamed = _pvt_sanitise({t}.columns)",
                "if _pvt_renamed:",
                f"    {t} = {t}.rename(_pvt_renamed)",
                "    import warnings",
                f"    warnings.warn(f\"[Pivotal] Column(s) renamed in '{t}': {{_pvt_renamed}}\", UserWarning, stacklevel=2)",
                marker,
            ])
            return "\n".join(lines)

        alias_expr = self._bulk_aliases_code(ast_node)
        lines.extend([
            f"_pvt_aliases = {alias_expr}",
            *self._bulk_validate_aliases_lines(),
            f"{_SANITISE_HELPER}"
            "for _pvt_src, _pvt_alias in zip(_pvt_sources, _pvt_aliases):",
            "    _pvt_df = _pvt_read_file(_pvt_src)",
            "    _, _pvt_renamed = _pvt_sanitise(_pvt_df.columns)",
            "    if _pvt_renamed:",
            "        _pvt_df = _pvt_df.rename(_pvt_renamed)",
            "        import warnings",
            "        warnings.warn(f\"[Pivotal] Column(s) renamed in '{_pvt_alias}': {_pvt_renamed}\", UserWarning, stacklevel=2)",
            "    globals()[_pvt_alias] = _pvt_df",
        ])
        return "\n".join(lines)

    def generate_copy_table_pandas(self, ast_node):
        copy_code = f"{ast_node['table_name']} = {ast_node['copy_from']}.copy()"
        table_name = f"#__pivotal__\n__table_name__ = '{ast_node['table_name']}'\n#__pivotal__"
        return f"{copy_code}\n{table_name}"

    def generate_validate_table_pandas(self, ast_node):
        table_name = f"#__pivotal__\n__table_name__ = '{ast_node['table_name']}'\n#__pivotal__"
        return table_name
    
    # Built-in function names reserved for future string function support.
    # User-defined functions must not use these names.
    _BUILTIN_FUNCS = frozenset({
        'upper', 'lower', 'trim', 'ltrim', 'rtrim',
        'left', 'right', 'substr', 'len', 'replace',
        'regex_extract', 'regex_replace',
        # date functions
        'year', 'month', 'day', 'quarter', 'dayofweek',
        'hour', 'minute', 'date_format', 'to_date',
        'date_diff', 'date_add',
        # cast functions (inline)
        'int', 'integer', 'float', 'string', 'str',
        'bool', 'boolean', 'datetime',
    })

    _CAST_FUNCS = frozenset({
        'int', 'integer', 'float', 'string', 'str',
        'bool', 'boolean', 'datetime',
    })

    _DATE_FUNCS = frozenset({
        'year', 'month', 'day', 'quarter', 'dayofweek',
        'hour', 'minute', 'date_format', 'to_date',
    })
    _DATE_TWO_ARG = frozenset({'date_diff', 'date_add'})

    def _parse_python_func_call(self, expr):
        """If expr matches ':func(col)' and func is not a built-in, return (func, col).
        Otherwise return None."""
        import re
        m = re.fullmatch(r'[:@]([a-zA-Z][a-zA-Z0-9_]*)\(([a-zA-Z][a-zA-Z0-9_]*)\)', expr.strip())
        if m and m.group(1) not in self._BUILTIN_FUNCS:
            return m.group(1), m.group(2)
        return None

    def _bare_unknown_func_call(self, expr):
        """Return function name for bare non-built-in func(col), else None."""
        import re
        m = re.fullmatch(r'([a-zA-Z][a-zA-Z0-9_]*)\(([a-zA-Z][a-zA-Z0-9_]*)\)', expr.strip())
        if m and m.group(1) not in self._BUILTIN_FUNCS:
            return m.group(1)
        return None

    def _raise_bare_python_func_call(self, func):
        raise ValueError(
            f"Python function '{func}' must be called with ':' in a column expression. "
            f"Use :{func}(column) for a Python runtime function, or use a built-in Pivotal "
            "function without ':'."
        )

    # -------------------------------------------------------------------------
    # String expression helpers
    # -------------------------------------------------------------------------

    def _parse_string_expr(self, expr, table):
        """Return a pandas code string if expr is a string function call or
        string concatenation, otherwise return None (fall through to eval)."""
        import re
        expr = expr.strip()
        cast_result = self._try_cast_func(expr, table)
        if cast_result is not None:
            return cast_result
        date_result = self._try_date_func(expr, table)
        if date_result is not None:
            return date_result
        result = self._try_string_func(expr, table)
        if result is not None:
            return result
        if '+' in expr:
            # Always try string concat when a quoted literal is present
            if '"' in expr or "'" in expr:
                return self._try_string_concat(expr, table)
            # Also try when all tokens are bare identifiers and no other
            # arithmetic operators are present — df['a'] + df['b'] works for
            # both string and numeric columns so this is safe in all cases
            if not re.search(r'[-*/]', expr):
                tokens = [t.strip() for t in expr.split('+')]
                if all(re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]*', t) for t in tokens if t):
                    return self._try_string_concat(expr, table)
        return None

    def _try_string_func(self, expr, table):
        """Parse STRING_FUNC(col, ...) and return pandas .str code, or None."""
        import re
        m = re.fullmatch(r'([a-zA-Z][a-zA-Z0-9_]*)\s*\((.+)\)\s*', expr.strip(), re.DOTALL)
        if not m or m.group(1) not in self._BUILTIN_FUNCS:
            return None
        func = m.group(1)
        args = self._split_func_args(m.group(2))
        if not args:
            return None
        first = args[0].strip()
        # First arg may itself be a nested string function call
        nested = self._try_string_func(first, table)
        base = nested if nested is not None else f"{table}['{first}']"
        rest = [a.strip() for a in args[1:]]
        _simple = {'upper': 'upper', 'lower': 'lower',
                   'trim': 'strip', 'ltrim': 'lstrip', 'rtrim': 'rstrip'}
        if func in _simple:
            return f"{base}.str.{_simple[func]}()"
        if func == 'len':
            return f"{base}.str.len()"
        if func == 'left' and len(rest) == 1:
            return f"{base}.str[:{rest[0]}]"
        if func == 'right' and len(rest) == 1:
            return f"{base}.str[-{rest[0]}:]"
        if func == 'substr' and len(rest) == 2:
            s, n = rest
            return f"{base}.str[{s}:{s}+{n}]"
        if func == 'replace' and len(rest) == 2:
            a = self._decode_string_arg(rest[0])
            b = self._decode_string_arg(rest[1])
            return f"{base}.str.replace({repr(a)}, {repr(b)}, regex=False)"
        if func == 'regex_extract' and len(rest) in (1, 2):
            pattern = self._decode_string_arg(rest[0])
            group = int(rest[1]) if len(rest) == 2 else 0
            if group == 0:
                return f"{base}.astype('string').str.extract({self._py_literal(f'({pattern})')}, expand=False)"
            return f"{base}.astype('string').str.extract({self._py_literal(pattern)}, expand=True).iloc[:, {group - 1}]"
        if func == 'regex_replace' and len(rest) == 2:
            pattern = self._decode_string_arg(rest[0])
            repl = self._decode_string_arg(rest[1])
            return f"{base}.astype('string').str.replace({self._py_literal(pattern)}, {self._py_literal(repl)}, regex=True)"
        return None

    def _try_date_func(self, expr, table):
        """Parse a date function call and return pandas .dt.* code, or None."""
        import re
        m = re.fullmatch(r'([a-zA-Z][a-zA-Z0-9_]*)\s*\((.+)\)\s*', expr.strip(), re.DOTALL)
        if not m:
            return None
        func = m.group(1)
        if func not in self._DATE_FUNCS and func not in self._DATE_TWO_ARG:
            return None
        args = self._split_func_args(m.group(2))
        if not args:
            return None
        col = args[0].strip()
        base = f"{table}['{col}']"
        _simple = {
            'year': 'year', 'month': 'month', 'day': 'day',
            'quarter': 'quarter', 'dayofweek': 'dayofweek',
            'hour': 'hour', 'minute': 'minute',
        }
        if func in _simple:
            return f"{base}.dt.{_simple[func]}"
        if func == 'date_format' and len(args) == 2:
            fmt = args[1].strip()
            return f"{base}.dt.strftime({fmt})"
        if func == 'to_date':
            return f"pd.to_datetime({base})"
        if func == 'date_diff' and len(args) == 2:
            start = args[1].strip()
            return f"({base} - {table}['{start}']).dt.days"
        if func == 'date_add' and len(args) == 2:
            n = args[1].strip()
            if n.startswith(':'):
                var = n[1:]
                return f"{base} + pd.to_timedelta({var}, unit='d')"
            return f"{base} + pd.Timedelta(days={n})"
        return None

    def _try_cast_func(self, expr, table):
        """Parse an inline cast call int(col)/float(col)/etc and return pandas code, or None."""
        import re
        m = re.fullmatch(r'([a-zA-Z][a-zA-Z0-9_]*)\s*\((.+)\)\s*', expr.strip(), re.DOTALL)
        if not m or m.group(1) not in self._CAST_FUNCS:
            return None
        func = m.group(1)
        args = self._split_func_args(m.group(2))
        if not args:
            return None
        col = args[0].strip()
        base = f"{table}['{col}']"
        if func in ('int', 'integer'):
            return f"pd.to_numeric({base}, errors='coerce').astype('Int64')"
        if func == 'float':
            return f"pd.to_numeric({base}, errors='coerce')"
        if func in ('str', 'string'):
            return f"{base}.astype(str)"
        if func in ('bool', 'boolean'):
            return f"{base}.astype(bool)"
        if func == 'datetime':
            return f"pd.to_datetime({base}, errors='coerce')"
        return None

    def _split_func_args(self, args_str):
        """Split comma-separated args, respecting quoted strings and nested parens."""
        args, depth, in_quote, current = [], 0, None, []
        for ch in args_str:
            if in_quote:
                current.append(ch)
                if ch == in_quote:
                    in_quote = None
            elif ch in ('"', "'"):
                in_quote = ch
                current.append(ch)
            elif ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            args.append(''.join(current).strip())
        return [a for a in args if a]

    @staticmethod
    def _decode_string_arg(arg):
        """Decode a quoted Pivotal string argument, preserving regex escapes."""
        if isinstance(arg, _LiteralStr):
            try:
                return ast.literal_eval('"' + str(arg).replace('"', '\\"') + '"')
            except (SyntaxError, ValueError):
                return str(arg)
        arg = str(arg).strip()
        if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in ('"', "'"):
            try:
                return ast.literal_eval(arg)
            except (SyntaxError, ValueError):
                return arg[1:-1]
        return arg

    @staticmethod
    def _py_literal(value):
        """Return a Python string literal that is safe inside generated code."""
        return json.dumps(str(value))

    @staticmethod
    def _sql_literal(value):
        """Return a single-quoted SQL string literal."""
        return "'" + str(value).replace("'", "''") + "'"

    @staticmethod
    def _quantile_alias(col, func, q):
        q_text = ("%g" % float(q)).replace('.', 'p')
        return f"{col}_{func}_{q_text}"

    def _parse_scalar_minmax_call(self, expr):
        """Return (func, args) for scalar min/max(expr1, expr2, ...), else None."""
        import re
        m = re.fullmatch(r'([a-zA-Z][a-zA-Z0-9_]*)\s*\((.+)\)\s*', expr.strip(), re.DOTALL)
        if not m:
            return None
        func = m.group(1).lower()
        if func not in {'min', 'max'}:
            return None
        args = self._split_func_args(m.group(2))
        if len(args) < 2:
            return None
        return func, args

    def _try_scalar_minmax_pandas(self, expr, table):
        """Translate scalar min/max(expr1, expr2, ...) to vectorised numpy code."""
        parsed = self._parse_scalar_minmax_call(expr)
        if parsed is None:
            return None

        func, args = parsed
        np_func = 'maximum' if func == 'max' else 'minimum'

        def _arg_code(arg):
            arg = arg.strip()
            nested = self._try_scalar_minmax_pandas(arg, table)
            if nested is not None:
                return nested
            string_code = self._parse_string_expr(arg, table)
            if string_code is not None:
                return string_code
            if self._is_scalar_expr(arg):
                return arg
            return f"{table}.eval({arg!r})"

        code = _arg_code(args[0])
        for arg in args[1:]:
            code = f"np.{np_func}({code}, {_arg_code(arg)})"
        return code

    def _try_sql_scalar_minmax(self, expr):
        """Translate scalar min/max(expr1, expr2, ...) to LEAST/GREATEST."""
        parsed = self._parse_scalar_minmax_call(expr)
        if parsed is None:
            return None, False

        func, args = parsed
        sql_func = 'GREATEST' if func == 'max' else 'LEAST'
        translated = []
        uses_pyvar = False

        for arg in args:
            arg = arg.strip()
            nested_sql, nested_pyvar = self._try_sql_scalar_minmax(arg)
            if nested_sql is not None:
                translated.append(nested_sql)
                uses_pyvar = uses_pyvar or nested_pyvar
                continue

            sql = self._try_sql_cast_func(arg)
            if sql is None:
                sql, upv = self._try_sql_date_func(arg)
                uses_pyvar = uses_pyvar or upv
            if sql is None:
                sql = self._try_sql_string_func(arg)
            if sql is None:
                sql = self._try_sql_string_concat(arg)
            if sql is None:
                sql, upv = self._translate_assign_expr_to_sql(arg)
                uses_pyvar = uses_pyvar or upv
            translated.append(sql)

        return f"{sql_func}({', '.join(translated)})", uses_pyvar

    def _try_string_concat(self, expr, table):
        """Parse col + "lit" + col concatenation. Returns pandas code or None."""
        import re
        tokens = self._split_on_plus(expr)
        if len(tokens) < 2:
            return None
        parts = []
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                return None
            # Quoted literal
            if (tok.startswith('"') and tok.endswith('"')) or \
               (tok.startswith("'") and tok.endswith("'")):
                parts.append(repr(tok[1:-1]))
            # @varname — Python variable reference (from :varname → @varname substitution)
            elif re.fullmatch(r'@[a-zA-Z_][a-zA-Z0-9_]*', tok):
                parts.append(tok[1:])  # strip @ → bare Python variable name
            # Nested string function
            elif re.match(r'[a-zA-Z][a-zA-Z0-9_]*\s*\(', tok):
                nested = self._try_string_func(tok, table)
                if nested is None:
                    return None
                parts.append(nested)
            # Bare column identifier
            elif re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]*', tok):
                parts.append(f"{table}['{tok}']")
            else:
                return None  # unrecognised token — fall back to arithmetic
        return ' + '.join(parts)

    def _split_on_plus(self, s):
        """Split on + while respecting quoted strings and parentheses."""
        tokens, depth, in_quote, current = [], 0, None, []
        for ch in s:
            if in_quote:
                current.append(ch)
                if ch == in_quote:
                    in_quote = None
            elif ch in ('"', "'"):
                in_quote = ch
                current.append(ch)
            elif ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == '+' and depth == 0:
                tokens.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            tokens.append(''.join(current))
        return tokens

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

    def _generate_case_assign_pandas(self, ast_node):
        table = ast_node['table_name']
        target = ast_node['target']
        cases = ast_node['cases']

        branches = [c for c in cases if c['type'] == 'case_branch']
        defaults = [c for c in cases if c['type'] == 'case_default']

        def _eval_expr(expr):
            if self._is_scalar_expr(expr):
                return expr
            return f"{table}.eval({expr!r})"

        conds = ', '.join(
            self._pandas_condition_eval(table, b['conditions'], b['operators'])
            for b in branches
        )
        choices = ', '.join(_eval_expr(b['expression']) for b in branches)

        if defaults:
            default_str = _eval_expr(defaults[0]['expression'])
        else:
            default_str = 'None'

        return (f"import numpy as np\n"
                f"{table}[{target!r}] = np.select(\n"
                f"    [{conds}],\n"
                f"    [{choices}],\n"
                f"    default={default_str},\n"
                f")")

    def _pandas_condition_eval(self, table, conditions, operators):
        query_str, needs_python_engine = self._build_query_string(conditions, operators)
        engine = ", engine='python'" if needs_python_engine else ""
        return f"{table}.eval({query_str!r}{engine})"

    def _substitute_agg_calls(self, expr, table, by_cols):
        """Replace agg(col), quantile(col, q), and wavg(col, wt) calls with @_agg_N locals."""
        preamble = []
        counter = [0]

        def replace_wavg(m):
            col, wt = m.group(1), m.group(2)
            var = f'_agg_{counter[0]}'
            counter[0] += 1
            if by_cols:
                preamble.append(
                    f"_wsum = {table}.groupby({by_cols!r})[{col!r}].transform("
                    f"lambda g: (g * {table}.loc[g.index, {wt!r}]).sum())"
                )
                preamble.append(
                    f"_wtot = {table}.groupby({by_cols!r})[{wt!r}].transform('sum')"
                )
                preamble.append(f"{var} = _wsum / _wtot")
            else:
                preamble.append(
                    f"{var} = ({table}[{col!r}] * {table}[{wt!r}]).sum() / {table}[{wt!r}].sum()"
                )
            return f'@{var}'

        def replace_quantile(m):
            func, col, q = m.group(1), m.group(2), float(m.group(3))
            if func == 'percentile':
                q = q / 100.0
            var = f'_agg_{counter[0]}'
            counter[0] += 1
            if by_cols:
                preamble.append(
                    f"{var} = {table}.groupby({by_cols!r})[{col!r}].transform(lambda _s: _s.quantile({q!r}))"
                )
            else:
                preamble.append(f"{var} = {table}[{col!r}].quantile({q!r})")
            return f'@{var}'

        def replace_agg(m):
            func = m.group(1)
            col = m.group(2)
            pandas_func = 'mean' if func == 'avg' else func
            var = f'_agg_{counter[0]}'
            counter[0] += 1
            if by_cols:
                preamble.append(
                    f"{var} = {table}.groupby({by_cols!r})[{col!r}].transform({pandas_func!r})"
                )
            else:
                preamble.append(f"{var} = {table}[{col!r}].{pandas_func}()")
            return f'@{var}'

        new_expr = _WAVG_CALL_RE.sub(replace_wavg, expr)
        new_expr = _QUANTILE_CALL_RE.sub(replace_quantile, new_expr)
        new_expr = _AGG_CALL_RE.sub(replace_agg, new_expr)
        return preamble, new_expr

    def generate_assign_pandas(self, ast_node):
        table = ast_node['table_name']
        target = ast_node['target']

        if ast_node.get('cases'):
            return self._generate_case_assign_pandas(ast_node)

        expr = ast_node['expression']
        by_cols = ast_node.get('by_cols', [])
        conditions = ast_node.get('conditions')
        operators = ast_node.get('operators', [])
        default_expr = ast_node.get('default_expr')

        # Convert :varname Python variable refs to @varname for df.eval()
        import re as _re_pa
        expr = _re_pa.sub(r':([a-zA-Z_][a-zA-Z0-9_]*)', r'@\1', expr)
        if default_expr is not None:
            default_expr = _re_pa.sub(r':([a-zA-Z_][a-zA-Z0-9_]*)', r'@\1', default_expr)

        def _python_ref_name(expr_value):
            match = _re_pa.fullmatch(r'\s*@([a-zA-Z_][a-zA-Z0-9_]*)\s*', expr_value or '')
            return match.group(1) if match else None

        def _default_rhs(expr_value):
            if expr_value is None:
                return None
            ref_name = _python_ref_name(expr_value)
            if ref_name is not None:
                return ref_name
            string_code = self._parse_string_expr(expr_value, table)
            if string_code is not None:
                return f"({string_code})"
            scalar_minmax = self._try_scalar_minmax_pandas(expr_value, table)
            if scalar_minmax is not None:
                return f"({scalar_minmax})"
            user_call = self._parse_python_func_call(expr_value)
            if user_call:
                func, col = user_call
                return f"{func}({table}['{col}'])"
            bare_func = self._bare_unknown_func_call(expr_value)
            if bare_func:
                self._raise_bare_python_func_call(bare_func)
            if self._is_scalar_expr(expr_value):
                return expr_value
            return f"{table}.eval('{expr_value}')"

        default_rhs = _default_rhs(default_expr)

        # Detect aggregate function calls — substitute before other processing
        preamble, subst_expr = self._substitute_agg_calls(expr, table, by_cols)
        if preamble:
            lines = preamble
            scalar_code = self._try_scalar_minmax_pandas(subst_expr, table)
            if scalar_code is not None:
                lines.insert(0, "import numpy as np")
                if conditions:
                    lines.append(f"_cond = {self._pandas_condition_eval(table, conditions, operators)}")
                    lines.append(
                        f"{table}.loc[_cond, {target!r}] = ({scalar_code})"
                    )
                    if default_rhs is not None:
                        lines.append(
                            f"{table}.loc[~_cond, {target!r}] = {default_rhs}"
                        )
                else:
                    lines.append(f"{table}[{target!r}] = {scalar_code}")
                return '\n'.join(lines)
            if conditions:
                lines.append(f"_cond = {self._pandas_condition_eval(table, conditions, operators)}")
                lines.append(
                    f"{table}.loc[_cond, {target!r}] = {table}.eval({subst_expr!r})"
                )
                if default_rhs is not None:
                    lines.append(
                        f"{table}.loc[~_cond, {target!r}] = {default_rhs}"
                    )
            else:
                lines.append(f"{table}[{target!r}] = {table}.eval({subst_expr!r})")
            return '\n'.join(lines)

        # String function / concatenation — takes priority over eval
        string_code = self._parse_string_expr(expr, table)
        if string_code is not None:
            if conditions:
                condition_expr = self._pandas_condition_eval(table, conditions, operators)
                code = (f"condition = {condition_expr}\n"
                        f"{table}.loc[condition, '{target}'] = ({string_code})")
                if default_rhs is not None:
                    code += f"\n{table}.loc[~condition, '{target}'] = {default_rhs}"
                return code
            return f"{table}['{target}'] = {string_code}"

        scalar_minmax = self._try_scalar_minmax_pandas(expr, table)
        if scalar_minmax is not None:
            if conditions:
                condition_expr = self._pandas_condition_eval(table, conditions, operators)
                code = (f"import numpy as np\n"
                        f"condition = {condition_expr}\n"
                        f"{table}.loc[condition, '{target}'] = ({scalar_minmax})")
                if default_rhs is not None:
                    code += f"\n{table}.loc[~condition, '{target}'] = {default_rhs}"
                return code
            return f"import numpy as np\n{table}['{target}'] = {scalar_minmax}"

        user_call = self._parse_python_func_call(expr)
        bare_func = self._bare_unknown_func_call(expr)
        if bare_func:
            self._raise_bare_python_func_call(bare_func)

        if conditions:
            condition_expr = self._pandas_condition_eval(table, conditions, operators)
            if user_call:
                func, col = user_call
                code = (f"condition = {condition_expr}\n"
                        f"{table}.loc[condition, '{target}'] = "
                        f"{func}({table}['{col}'])")
            else:
                ref_name = _python_ref_name(expr)
                if ref_name is not None:
                    rhs = ref_name
                elif self._is_scalar_expr(expr):
                    rhs = expr
                else:
                    rhs = f"{table}.eval('{expr}')"
                code = (f"condition = {condition_expr}\n"
                        f"{table}.loc[condition, '{target}'] = {rhs}")
            if default_rhs is not None:
                code += f"\n{table}.loc[~condition, '{target}'] = {default_rhs}"
            return code
        else:
            if user_call:
                func, col = user_call
                return f"{table}['{target}'] = {func}({table}['{col}'])"
            ref_name = _python_ref_name(expr)
            if ref_name is not None:
                return f"{table}['{target}'] = {ref_name}"
            if self._is_scalar_expr(expr):
                return f"{table}['{target}'] = {expr}"
            return f"{table}['{target}'] = {table}.eval('{expr}')"

    def generate_apply_pandas(self, ast_node):
        table = ast_node['table_name']
        func = ast_node['func']
        return f"{table} = {func}({table})"

    def generate_apply_polars(self, ast_node):
        table = ast_node['table_name']
        func = ast_node['func']
        return f"{table} = {func}({table})"

    # ------------------------------------------------------------------
    # Polars Phase 1 generators — core pipeline
    # ------------------------------------------------------------------

    def generate_copy_table_polars(self, ast_node):
        src = ast_node['copy_from']
        tgt = ast_node['table_name']
        copy_code = f"{tgt} = {src}.clone()"
        marker = f"#__pivotal__\n__table_name__ = '{tgt}'\n#__pivotal__"
        return f"{copy_code}\n{marker}"

    def generate_validate_table_polars(self, ast_node):
        tbl = ast_node['table_name']
        marker = f"#__pivotal__\n__table_name__ = '{tbl}'\n#__pivotal__"
        return marker

    def generate_filter_polars(self, ast_node):
        expr = self._build_polars_filter(ast_node['conditions'], ast_node['operators'])
        tbl = ast_node['table_name']
        return f"{tbl} = {tbl}.filter({expr})"

    def generate_data_quality_polars(self, ast_node):
        tbl = ast_node['table_name']
        rule = ast_node.get('rule')
        if rule == 'condition':
            expr = self._build_polars_filter(ast_node['conditions'], ast_node['operators'])
            lines = [f"_pvt_dq_bad = {tbl}.filter(~({expr})).height"]
        elif rule == 'unique':
            cols = ast_node['columns']
            lines = [f"_pvt_dq_bad = int({tbl}.select({cols!r}).is_duplicated().sum())"]
        elif rule == 'not_null':
            null_exprs = ', '.join(f"pl.col({col!r}).is_null()" for col in ast_node['columns'])
            lines = [f"_pvt_dq_bad = {tbl}.filter(pl.any_horizontal([{null_exprs}])).height"]
        else:
            raise ValueError(f"Unknown data quality rule: {rule}")
        lines.append(self._dq_action_code(ast_node))
        return "\n".join(lines)

    def generate_select_polars(self, ast_node):
        columns = ast_node['columns']
        renames = ast_node.get('renames', {})
        tbl = ast_node['table_name']

        if ast_node.get('column_match'):
            pattern = self._decode_string_arg(ast_node['column_match'])
            return (
                f"_pvt_re = __import__('re')\n"
                f"_pvt_cols = [c for c in {tbl}.columns if _pvt_re.search({self._py_literal(pattern)}, str(c))]\n"
                f"{tbl} = {tbl}.select(_pvt_cols)"
            )

        has_vars = any(isinstance(col, dict) and col.get('type') == 'var' for col in columns)

        if has_vars:
            col_list_code = "[]"
            for col in columns:
                if isinstance(col, dict) and col.get('type') == 'var':
                    var_name = col['name']
                    col_list_code += f" + ({var_name} if isinstance({var_name}, list) else [{var_name}])"
                else:
                    col_list_code += f" + ['{col}']"
            code = f"{tbl} = {tbl}.select({col_list_code})"
            if renames:
                code += f"\n{tbl} = {tbl}.rename({renames})"
        else:
            if renames:
                select_exprs = []
                for col in columns:
                    col_str = str(col)
                    if col_str in renames:
                        select_exprs.append(f"pl.col('{col_str}').alias('{renames[col_str]}')")
                    else:
                        select_exprs.append(f"'{col_str}'")
                code = f"{tbl} = {tbl}.select([{', '.join(select_exprs)}])"
            else:
                col_list = [str(c) for c in columns]
                code = f"{tbl} = {tbl}.select({col_list})"

        return code

    def generate_rename_polars(self, ast_node):
        tbl = ast_node['table_name']
        return f"{tbl} = {tbl}.rename({ast_node['renames']})"

    def generate_round_polars(self, ast_node):
        tbl = ast_node['table_name']
        decimals = ast_node['decimals']
        alias = ast_node.get('alias')
        exprs = []
        for col in ast_node['columns']:
            target = alias or col
            exprs.append(f"pl.col({col!r}).round({decimals}).alias({target!r})")
        return f"{tbl} = {tbl}.with_columns([{', '.join(exprs)}])"

    def generate_drop_polars(self, ast_node):
        tbl = ast_node['table_name']
        if ast_node.get('column_match'):
            pattern = self._decode_string_arg(ast_node['column_match'])
            return (
                f"_pvt_re = __import__('re')\n"
                f"_pvt_cols = [c for c in {tbl}.columns if _pvt_re.search({self._py_literal(pattern)}, str(c))]\n"
                f"{tbl} = {tbl}.drop(_pvt_cols)"
            )
        return f"{tbl} = {tbl}.drop({ast_node['columns']})"

    def generate_cast_polars(self, ast_node):
        table = ast_node['table_name']
        cols = ast_node['columns']
        cast_type = ast_node['cast_type']
        strict = ast_node.get('strict', False)
        _POLARS_TYPES = {
            'int': 'pl.Int64', 'integer': 'pl.Int64',
            'float': 'pl.Float64',
            'str': 'pl.Utf8', 'string': 'pl.Utf8',
            'bool': 'pl.Boolean', 'boolean': 'pl.Boolean',
            'datetime': 'pl.Datetime',
        }
        pl_type = _POLARS_TYPES.get(cast_type, 'pl.Utf8')
        strict_flag = '' if strict else ', strict=False'
        lines = []
        for col in cols:
            lines.append(
                f"{table} = {table}.with_columns("
                f"pl.col('{col}').cast({pl_type}{strict_flag}))"
            )
        return '\n'.join(lines)

    def generate_sort_polars(self, ast_node):
        columns = ast_node['columns']
        ascending = ast_node['ascending']
        tbl = ast_node['table_name']

        has_vars = any(isinstance(col, dict) and col.get('type') == 'var' for col in columns)

        if has_vars:
            col_list_code = "[]"
            desc_list_code = "[]"
            for col, asc in zip(columns, ascending):
                if isinstance(col, dict) and col.get('type') == 'var':
                    var_name = col['name']
                    desc_val = str(not asc)
                    col_list_code += f" + ({var_name} if isinstance({var_name}, list) else [{var_name}])"
                    desc_list_code += (
                        f" + ([{desc_val}] * (len({var_name}) if isinstance({var_name}, list) else 1))"
                    )
                else:
                    col_list_code += f" + ['{col}']"
                    desc_list_code += f" + [{not asc}]"
            return f"{tbl} = {tbl}.sort({col_list_code}, descending={desc_list_code})"
        else:
            descending = [not a for a in ascending]
            return f"{tbl} = {tbl}.sort({columns}, descending={descending})"

    def generate_distinct_polars(self, ast_node):
        cols = ast_node['columns']
        tbl = ast_node['table_name']
        if cols:
            return f"{tbl} = {tbl}.unique(subset={cols})"
        return f"{tbl} = {tbl}.unique()"

    def generate_concat_polars(self, ast_node):
        others = ', '.join(ast_node['tables'])
        tbl = ast_node['table_name']
        return f"{tbl} = pl.concat([{tbl}, {others}])"

    def generate_fillna_polars(self, ast_node):
        tbl = ast_node['table_name']
        per_col = ast_node.get('per_col', {})
        if per_col:
            lines = [f"{tbl} = {tbl}.with_columns(["]
            for col, val in per_col.items():
                if isinstance(val, _LiteralStr):
                    val_code = repr(str(val))
                elif isinstance(val, str):
                    val_code = f"pl.col('{val}')"
                else:
                    val_code = str(val)
                lines.append(f"    pl.col('{col}').fill_null({val_code}),")
            lines.append("])")
            return '\n'.join(lines)
        val = ast_node['value']
        val_code = repr(str(val)) if isinstance(val, _LiteralStr) else (f"'{val}'" if isinstance(val, str) else str(val))
        return f"{tbl} = {tbl}.fill_null({val_code})"

    def generate_intersect_polars(self, ast_node):
        tbl = ast_node['table_name']
        others = ast_node['tables']
        result = tbl
        for other in others:
            result = f"{result}.join({other}, on={result}.columns, how='inner')"
        return f"{tbl} = {result}.unique()"

    def generate_exclude_polars(self, ast_node):
        tbl = ast_node['table_name']
        others = ast_node['tables']
        lines = [f"{tbl} = {tbl}.unique()"]
        for other in others:
            lines.append(f"{tbl} = {tbl}.join({other}.unique(), on={tbl}.columns, how='anti')")
        return '\n'.join(lines)

    def generate_dropna_polars(self, ast_node):
        cols = ast_node['columns']
        tbl = ast_node['table_name']
        if cols:
            return f"{tbl} = {tbl}.drop_nulls(subset={cols})"
        return f"{tbl} = {tbl}.drop_nulls()"

    # ------------------------------------------------------------------
    # Polars Phase 2 generators — assign + merge
    # ------------------------------------------------------------------

    def _expr_to_polars(self, expr, by_cols=None):
        """Convert a simple arithmetic/agg expression string to a Polars Expr string.

        Handles column references, numeric/string literals, arithmetic operators,
        and aggregate function calls (sum, mean, etc.) with optional .over().
        """
        import re
        AGG_FUNCS = frozenset({
            'sum', 'mean', 'avg', 'min', 'max', 'count',
            'std', 'median', 'var', 'nunique', 'first', 'last',
        })
        AGG_RENAME = {'avg': 'mean'}

        result = []
        i = 0
        expr = expr.strip()
        n = len(expr)

        while i < n:
            ch = expr[i]

            if ch.isspace():
                result.append(ch)
                i += 1
                continue

            # Quoted string literal
            if ch in ('"', "'"):
                j = i + 1
                while j < n:
                    if expr[j] == '\\':
                        j += 2
                        continue
                    if expr[j] == ch:
                        break
                    j += 1
                s_content = expr[i + 1:j]
                result.append(f"pl.lit('{s_content}')")
                i = j + 1
                continue

            # Number (integer or float, including negative handled by operator)
            m = re.match(r'\d+(?:\.\d+)?', expr[i:])
            if m:
                result.append(m.group())
                i += len(m.group())
                continue

            # Identifier — column ref or agg/wavg function call
            m = re.match(r'[a-zA-Z_][a-zA-Z0-9_]*', expr[i:])
            if m:
                name = m.group()
                end = i + len(name)
                # Peek past whitespace for opening paren
                rest = expr[end:].lstrip()

                if rest.startswith('(') and name in ('wavg', 'wmean'):
                    paren_pos = expr.index('(', end)
                    depth, j = 1, paren_pos + 1
                    while j < n and depth > 0:
                        depth += (expr[j] == '(') - (expr[j] == ')')
                        j += 1
                    inner = expr[paren_pos + 1:j - 1].strip()
                    parts = [p.strip() for p in inner.split(',', 1)]
                    col, wt = parts[0], parts[1]
                    if by_cols:
                        by_repr = repr(by_cols[0]) if len(by_cols) == 1 else repr(by_cols)
                        agg_expr = (
                            f"(pl.col('{col}') * pl.col('{wt}')).sum().over({by_repr})"
                            f" / pl.col('{wt}').sum().over({by_repr})"
                        )
                    else:
                        agg_expr = (
                            f"(pl.col('{col}') * pl.col('{wt}')).sum()"
                            f" / pl.col('{wt}').sum()"
                        )
                    result.append(agg_expr)
                    i = j
                    continue

                if rest.startswith('(') and name in ('quantile', 'percentile'):
                    paren_pos = expr.index('(', end)
                    depth, j = 1, paren_pos + 1
                    while j < n and depth > 0:
                        depth += (expr[j] == '(') - (expr[j] == ')')
                        j += 1
                    inner = expr[paren_pos + 1:j - 1].strip()
                    parts = self._split_func_args(inner)
                    if len(parts) == 2:
                        col, q = parts[0], float(parts[1])
                        if name == 'percentile':
                            q = q / 100.0
                        agg_expr = f"pl.col('{col}').quantile({q!r}, interpolation='linear')"
                        if by_cols:
                            by_repr = repr(by_cols[0]) if len(by_cols) == 1 else repr(by_cols)
                            agg_expr += f".over({by_repr})"
                        result.append(agg_expr)
                        i = j
                        continue

                if rest.startswith('(') and name in {'min', 'max'}:
                    paren_pos = expr.index('(', end)
                    depth, j = 1, paren_pos + 1
                    while j < n and depth > 0:
                        depth += (expr[j] == '(') - (expr[j] == ')')
                        j += 1
                    inner = expr[paren_pos + 1:j - 1].strip()
                    parts = self._split_func_args(inner)
                    if len(parts) >= 2:
                        horizontal = 'max_horizontal' if name == 'max' else 'min_horizontal'
                        arg_exprs = [self._expr_to_polars(p, by_cols) for p in parts]
                        result.append(f"pl.{horizontal}([{', '.join(arg_exprs)}])")
                        i = j
                        continue

                if rest.startswith('(') and name in AGG_FUNCS:
                    paren_pos = expr.index('(', end)
                    depth, j = 1, paren_pos + 1
                    while j < n and depth > 0:
                        depth += (expr[j] == '(') - (expr[j] == ')')
                        j += 1
                    arg = expr[paren_pos + 1:j - 1].strip()
                    polars_func = AGG_RENAME.get(name, name)
                    agg_expr = f"pl.col('{arg}').{polars_func}()"
                    if by_cols:
                        by_repr = repr(by_cols[0]) if len(by_cols) == 1 else repr(by_cols)
                        agg_expr += f".over({by_repr})"
                    result.append(agg_expr)
                    i = j
                    continue

                # Regular identifier → column reference
                result.append(f"pl.col('{name}')")
                i = end
                continue

            # Python variable reference (:varname) → bare variable name
            if ch == ':':
                m = re.match(r'[a-zA-Z_][a-zA-Z0-9_]*', expr[i + 1:])
                if m:
                    result.append(m.group())
                    i += 1 + len(m.group())
                    continue

            # Operator or punctuation — pass through
            result.append(ch)
            i += 1

        return ''.join(result)

    def _try_string_func_polars(self, expr):
        """Parse STRING_FUNC(col, ...) and return a Polars Expr string, or None."""
        import re
        m = re.fullmatch(r'([a-zA-Z][a-zA-Z0-9_]*)\s*\((.+)\)\s*', expr.strip(), re.DOTALL)
        if not m or m.group(1) not in self._BUILTIN_FUNCS:
            return None
        func = m.group(1)
        args = self._split_func_args(m.group(2))
        if not args:
            return None
        first = args[0].strip()
        # First arg may itself be a nested string function
        nested = self._try_string_func_polars(first)
        base = nested if nested is not None else f"pl.col('{first}')"
        rest = [a.strip() for a in args[1:]]

        if func == 'upper':
            return f"{base}.str.to_uppercase()"
        if func == 'lower':
            return f"{base}.str.to_lowercase()"
        if func == 'trim':
            return f"{base}.str.strip_chars()"
        if func == 'ltrim':
            return f"{base}.str.strip_chars_start()"
        if func == 'rtrim':
            return f"{base}.str.strip_chars_end()"
        if func == 'len':
            return f"{base}.str.len_chars()"
        if func == 'left' and len(rest) == 1:
            return f"{base}.str.slice(0, {rest[0]})"
        if func == 'right' and len(rest) == 1:
            return f"{base}.str.slice(-{rest[0]})"
        if func == 'substr' and len(rest) == 2:
            s, length = rest
            return f"{base}.str.slice({s}, {length})"
        if func == 'replace' and len(rest) == 2:
            a = self._decode_string_arg(rest[0])
            b = self._decode_string_arg(rest[1])
            return f"{base}.str.replace_all({repr(a)}, {repr(b)}, literal=True)"
        if func == 'regex_extract' and len(rest) in (1, 2):
            pattern = self._decode_string_arg(rest[0])
            group = int(rest[1]) if len(rest) == 2 else 0
            return f"{base}.str.extract({self._py_literal(pattern)}, group_index={group})"
        if func == 'regex_replace' and len(rest) == 2:
            pattern = self._decode_string_arg(rest[0])
            repl = self._decode_string_arg(rest[1])
            return f"{base}.str.replace_all({self._py_literal(pattern)}, {self._py_literal(repl)})"
        return None

    def _try_date_func_polars(self, expr):
        """Parse a date function call and return a Polars Expr string, or None."""
        import re
        m = re.fullmatch(r'([a-zA-Z][a-zA-Z0-9_]*)\s*\((.+)\)\s*', expr.strip(), re.DOTALL)
        if not m:
            return None
        func = m.group(1)
        if func not in self._DATE_FUNCS and func not in self._DATE_TWO_ARG:
            return None
        args = self._split_func_args(m.group(2))
        if not args:
            return None
        col = args[0].strip()
        base = f"pl.col('{col}')"
        _simple = {
            'year': 'year', 'month': 'month', 'day': 'day',
            'quarter': 'quarter', 'hour': 'hour', 'minute': 'minute',
        }
        if func in _simple:
            return f"{base}.dt.{_simple[func]}()"
        if func == 'dayofweek':
            return f"{base}.dt.weekday()"
        if func == 'date_format' and len(args) == 2:
            fmt = args[1].strip()
            return f"{base}.dt.strftime({fmt})"
        if func == 'to_date':
            return f"{base}.cast(pl.Date)"
        if func == 'date_diff' and len(args) == 2:
            start = args[1].strip()
            return f"({base} - pl.col('{start}')).dt.total_days()"
        if func == 'date_add' and len(args) == 2:
            n = args[1].strip()
            if n.startswith(':'):
                var = n[1:]
                return f"({base} + pl.duration(days={var}))"
            return f"({base} + pl.duration(days={n}))"
        return None

    def _try_cast_func_polars(self, expr):
        """Parse an inline cast call int(col)/float(col)/etc and return Polars Expr code, or None."""
        import re
        m = re.fullmatch(r'([a-zA-Z][a-zA-Z0-9_]*)\s*\((.+)\)\s*', expr.strip(), re.DOTALL)
        if not m or m.group(1) not in self._CAST_FUNCS:
            return None
        func = m.group(1)
        args = self._split_func_args(m.group(2))
        if not args:
            return None
        col = args[0].strip()
        _POLARS_TYPES = {
            'int': 'pl.Int64', 'integer': 'pl.Int64',
            'float': 'pl.Float64',
            'str': 'pl.Utf8', 'string': 'pl.Utf8',
            'bool': 'pl.Boolean', 'boolean': 'pl.Boolean',
            'datetime': 'pl.Datetime',
        }
        pl_type = _POLARS_TYPES.get(func)
        if pl_type is None:
            return None
        return f"pl.col('{col}').cast({pl_type}, strict=False)"

    def _try_string_concat_polars(self, expr):
        """Parse col + 'lit' + col concatenation. Returns pl.concat_str(...) code or None."""
        import re
        tokens = self._split_on_plus(expr)
        if len(tokens) < 2:
            return None
        parts = []
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                return None
            if (tok.startswith('"') and tok.endswith('"')) or \
               (tok.startswith("'") and tok.endswith("'")):
                parts.append(f"pl.lit({tok})")
            # :varname — Python variable reference
            elif re.fullmatch(r':[a-zA-Z_][a-zA-Z0-9_]*', tok):
                parts.append(f"pl.lit({tok[1:]})")  # pl.lit(var)
            elif re.match(r'[a-zA-Z][a-zA-Z0-9_]*\s*\(', tok):
                nested = self._try_string_func_polars(tok)
                if nested is None:
                    return None
                parts.append(nested)
            elif re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]*', tok):
                parts.append(f"pl.col('{tok}')")
            else:
                return None
        return f"pl.concat_str([{', '.join(parts)}])"

    def _parse_string_expr_polars(self, expr):
        """Return a Polars Expr string if expr is a string func or concat, else None.

        Unlike pandas, pl.concat_str() always casts to string, so we only trigger
        string concat when a quoted literal is present in the expression. A bare
        col + col expression may be numeric addition and is left to _expr_to_polars.
        """
        import re
        expr = expr.strip()
        cast_result = self._try_cast_func_polars(expr)
        if cast_result is not None:
            return cast_result
        date_result = self._try_date_func_polars(expr)
        if date_result is not None:
            return date_result
        result = self._try_string_func_polars(expr)
        if result is not None:
            return result
        if '+' in expr and ('"' in expr or "'" in expr):
            return self._try_string_concat_polars(expr)
        return None

    def _generate_case_assign_polars(self, ast_node):
        """Generate pl.when().then().when().then().otherwise() for multi-case assign."""
        import re
        table = ast_node['table_name']
        target = ast_node['target']
        cases = ast_node['cases']

        branches = [c for c in cases if c['type'] == 'case_branch']
        defaults = [c for c in cases if c['type'] == 'case_default']

        def _polars_val(expr):
            expr = expr.strip()
            if self._is_scalar_expr(expr):
                # Numbers stay as numbers; quoted strings need pl.lit
                if re.match(r'^\d+(?:\.\d+)?$', expr):
                    return f"pl.lit({expr})"
                return f"pl.lit({expr})"
            s = self._try_string_func_polars(expr)
            if s:
                return s
            return self._expr_to_polars(expr)

        chain_parts = []
        for i, branch in enumerate(branches):
            filter_expr = self._build_polars_filter(branch['conditions'], branch['operators'])
            val = _polars_val(branch['expression'])
            if i == 0:
                chain_parts.append(f"pl.when({filter_expr}).then({val})")
            else:
                chain_parts.append(f"    .when({filter_expr}).then({val})")

        if defaults:
            chain_parts.append(f"    .otherwise({_polars_val(defaults[0]['expression'])})")
        else:
            chain_parts.append("    .otherwise(None)")

        chain = '\n'.join(chain_parts)
        return f"{table} = {table}.with_columns(\n    ({chain})\n    .alias('{target}')\n)"

    def generate_assign_polars(self, ast_node):
        table = ast_node['table_name']
        target = ast_node['target']

        if ast_node.get('cases'):
            return self._generate_case_assign_polars(ast_node)

        expr = ast_node['expression']
        by_cols = ast_node.get('by_cols', [])
        conditions = ast_node.get('conditions')
        operators = ast_node.get('operators') or []
        default_expr = ast_node.get('default_expr')

        def _polars_default(expr_value):
            expr_value = expr_value.strip()
            if self._is_scalar_expr(expr_value):
                return f"pl.lit({expr_value})"
            string_code = self._parse_string_expr_polars(expr_value)
            if string_code is not None:
                return string_code
            user_call = self._parse_python_func_call(expr_value)
            if user_call:
                func, col = user_call
                return f"pl.col('{col}').map_batches({func})"
            bare_func = self._bare_unknown_func_call(expr_value)
            if bare_func:
                self._raise_bare_python_func_call(bare_func)
            return self._expr_to_polars(expr_value, by_cols)

        def _conditional_wrap(polars_expr):
            filter_expr = self._build_polars_filter(conditions, operators)
            if default_expr is not None:
                otherwise = _polars_default(default_expr)
            else:
                otherwise = (
                    f"pl.col('{target}') if '{target}' in {table}.columns else pl.lit(None)"
                )
            return (
                f"{table} = {table}.with_columns(\n"
                f"    pl.when({filter_expr})\n"
                f"    .then({polars_expr})\n"
                f"    .otherwise({otherwise})\n"
                f"    .alias('{target}')\n"
                f")"
            )

        # Agg function calls — handle inline in Polars (no preamble needed)
        if _WAVG_CALL_RE.search(expr) or _QUANTILE_CALL_RE.search(expr) or _AGG_CALL_RE.search(expr):
            polars_expr = self._expr_to_polars(expr, by_cols)
            if conditions:
                return _conditional_wrap(polars_expr)
            return f"{table} = {table}.with_columns(({polars_expr}).alias('{target}'))"

        # String function / concatenation
        string_code = self._parse_string_expr_polars(expr)
        if string_code is not None:
            if conditions:
                return _conditional_wrap(string_code)
            return f"{table} = {table}.with_columns({string_code}.alias('{target}'))"

        # Python runtime function call: :func(col)
        user_call = self._parse_python_func_call(expr)
        if user_call:
            func, col = user_call
            if conditions:
                filter_expr = self._build_polars_filter(conditions, operators)
                otherwise = (
                    f"pl.col('{target}') if '{target}' in {table}.columns else pl.lit(None)"
                )
                return (
                    f"{table} = {table}.with_columns(\n"
                    f"    pl.when({filter_expr})\n"
                    f"    .then(pl.col('{col}').map_batches({func}))\n"
                    f"    .otherwise({otherwise})\n"
                    f"    .alias('{target}')\n"
                    f")"
                )
            return f"{table} = {table}.with_columns(pl.col('{col}').map_batches({func}).alias('{target}'))"

        bare_func = self._bare_unknown_func_call(expr)
        if bare_func:
            self._raise_bare_python_func_call(bare_func)

        # General arithmetic / scalar expression
        polars_expr = self._expr_to_polars(expr, by_cols)
        # If the entire expression reduced to a bare scalar (no pl.col / pl.lit
        # already present), wrap it so that .alias() can be called on it.
        # e.g. `wins = 1`  →  _expr_to_polars returns '1'
        #      but (1).alias('wins') fails; pl.lit(1).alias('wins') works.
        import re as _re
        if 'pl.' not in polars_expr and (
            _re.fullmatch(r'\s*-?\s*\d+(?:\.\d+)?\s*', polars_expr) or
            _re.fullmatch(r'\s*[a-zA-Z_][a-zA-Z0-9_]*\s*', polars_expr)
        ):
            polars_expr = f"pl.lit({polars_expr.strip()})"
        if conditions:
            return _conditional_wrap(polars_expr)
        return f"{table} = {table}.with_columns(({polars_expr}).alias('{target}'))"

    def generate_merge_polars(self, ast_node):
        tbl = ast_node['table_name']
        right = ast_node['right_table']
        how = ast_node['how']
        keys = ast_node['keys']
        kwargs = ast_node.get('kwargs') or {}
        if isinstance(kwargs, str):
            kwargs = {}

        # Polars uses 'full' where Pivotal says 'outer'
        how_map = {'inner': 'inner', 'left': 'left', 'right': 'right', 'outer': 'full'}
        polars_how = how_map.get(how, how)

        left_on = kwargs.get('left_on')
        right_on = kwargs.get('right_on')

        if left_on and right_on:
            return (
                f"{tbl} = {tbl}.join({right}, "
                f"left_on='{left_on}', right_on='{right_on}', "
                f"how='{polars_how}', coalesce=True)"
            )

        if keys and keys != '':
            key_repr = repr(keys[0]) if len(keys) == 1 else repr(keys)
            return f"{tbl} = {tbl}.join({right}, on={key_repr}, how='{polars_how}', coalesce=True)"

        # No explicit keys — join on all common columns (natural join)
        return (
            f"_common = [c for c in {tbl}.columns if c in {right}.columns]\n"
            f"{tbl} = {tbl}.join({right}, on=_common, how='{polars_how}', coalesce=True)"
        )

    # ------------------------------------------------------------------
    # Polars Phase 3 generators — aggregation and reshape
    # ------------------------------------------------------------------

    # Mapping from Pivotal agg function names to Polars Series method names
    _POLARS_AGG_MAP = {
        'sum': 'sum', 'mean': 'mean', 'avg': 'mean',
        'min': 'min', 'max': 'max', 'count': 'count',
        'std': 'std', 'median': 'median', 'var': 'var',
        'nunique': 'n_unique', 'first': 'first', 'last': 'last',
    }
    # Polars pivot uses slightly different names for aggregate_function
    _POLARS_PIVOT_AGG_MAP = {
        'sum': 'sum', 'mean': 'mean', 'avg': 'mean',
        'min': 'min', 'max': 'max', 'count': 'len',
        'std': 'std', 'median': 'median', 'first': 'first', 'last': 'last',
    }

    def _polars_by_code(self, by):
        """Convert a 'by' field to a Polars group_by argument string."""
        if isinstance(by, dict) and by.get('type') == 'var':
            return by['name']
        if isinstance(by, list):
            has_vars = any(isinstance(i, dict) and i.get('type') == 'var' for i in by)
            if has_vars:
                code = "[]"
                for item in by:
                    if isinstance(item, dict) and item.get('type') == 'var':
                        v = item['name']
                        code += f" + ({v} if isinstance({v}, list) else [{v}])"
                    else:
                        code += f" + ['{item}']"
                return code
            return str(by)
        return f"'{by}'"

    def generate_groupby_polars(self, ast_node):
        tbl = ast_node['table_name']
        by = ast_node['by']
        agg_list = ast_node.get('agg_list', [])
        if any(item.get('custom') for item in agg_list):
            pandas_ast = dict(ast_node)
            pandas_ast['table_name'] = '_pivotal_polars_custom_pdf'
            pandas_code = self.generate_groupby_pandas(pandas_ast)
            return '\n'.join([
                "# Pivotal: custom Python aggregations on Polars use a pandas fallback.",
                f"_pivotal_polars_custom_pdf = {tbl}.to_pandas()",
                pandas_code,
                f"{tbl} = pl.from_pandas(_pivotal_polars_custom_pdf)",
            ])

        # Whole-table aggregation (no group-by columns)
        if by == []:
            if agg_list:
                exprs = []
                for item in agg_list:
                    col = item['column']
                    func = item['func']
                    col_code = col['name'] if isinstance(col, dict) and col.get('type') == 'var' else f"'{col}'"
                    if func in ('wavg', 'wmean'):
                        wt = item['weight']
                        alias = item.get('alias') or f"wmean_{col}"
                        exprs.append(
                            f"((pl.col({col_code}) * pl.col('{wt}')).sum()"
                            f" / pl.col('{wt}').sum()).alias('{alias}')"
                        )
                    elif func in ('quantile', 'percentile'):
                        alias = item.get('alias') or self._quantile_alias(col, func, item['q'])
                        exprs.append(f"pl.col({col_code}).quantile({item['q']!r}, interpolation='linear').alias('{alias}')")
                    else:
                        alias = item.get('alias') or f"{col}_{func}"
                        polars_func = self._POLARS_AGG_MAP.get(func, func)
                        exprs.append(f"pl.col({col_code}).{polars_func}().alias('{alias}')")
                return f"{tbl} = {tbl}.select([{', '.join(exprs)}])"
            else:
                return f"{tbl} = {tbl}.select(pl.all().sum())"

        by_code = self._polars_by_code(by)

        if not agg_list:
            return f"{tbl} = {tbl}.group_by({by_code}).sum()"

        # Check whether any agg column is a runtime variable
        has_var_cols = any(
            isinstance(item['column'], dict) and item['column'].get('type') == 'var'
            for item in agg_list
        )

        if has_var_cols:
            # Build the agg list dynamically at runtime
            lines = ["_agg_exprs = []"]
            for item in agg_list:
                func = item['func']
                col = item['column']
                alias = item.get('alias')
                polars_func = self._POLARS_AGG_MAP.get(func, func)

                if isinstance(col, dict) and col.get('type') == 'var':
                    var_name = col['name']
                    if func in ('quantile', 'percentile'):
                        q_text = ("%g" % float(item['q'])).replace('.', 'p')
                        alias_expr = f"f'{{c}}_{func}_{q_text}'" if not alias else repr(alias)
                        lines.append(
                            f"for c in ({var_name} if isinstance({var_name}, list) else [{var_name}]):\n"
                            f"    _agg_exprs.append(pl.col(c).quantile({item['q']!r}, interpolation='linear').alias({alias_expr}))"
                        )
                    else:
                        alias_expr = f"f'{{c}}_{func}'" if not alias else repr(alias)
                        lines.append(
                            f"for c in ({var_name} if isinstance({var_name}, list) else [{var_name}]):\n"
                            f"    _agg_exprs.append(pl.col(c).{polars_func}().alias({alias_expr}))"
                        )
                elif func in ('wavg', 'wmean'):
                    wt = item['weight']
                    col_str = str(col)
                    alias_val = alias or f'wmean_{col_str}'
                    lines.append(
                        f"_agg_exprs.append("
                        f"((pl.col('{col_str}') * pl.col('{wt}')).sum()"
                        f" / pl.col('{wt}').sum()).alias('{alias_val}'))"
                    )
                elif func in ('quantile', 'percentile'):
                    col_str = str(col)
                    alias_val = alias or self._quantile_alias(col_str, func, item['q'])
                    lines.append(
                        f"_agg_exprs.append(pl.col('{col_str}').quantile({item['q']!r}, interpolation='linear').alias('{alias_val}'))"
                    )
                else:
                    col_str = str(col)
                    expr = f"pl.col('{col_str}').{polars_func}()"
                    if alias:
                        expr += f".alias('{alias}')"
                    lines.append(f"_agg_exprs.append({expr})")
            lines.append(f"{tbl} = {tbl}.group_by({by_code}).agg(_agg_exprs)")
            return '\n'.join(lines)

        # Static columns — build agg expressions directly
        agg_exprs = []
        for item in agg_list:
            func = item['func']
            col = str(item['column'])
            alias = item.get('alias')

            if func in ('wavg', 'wmean'):
                wt = item['weight']
                alias_val = alias or f'wmean_{col}'
                expr = (
                    f"((pl.col('{col}') * pl.col('{wt}')).sum()"
                    f" / pl.col('{wt}').sum()).alias('{alias_val}')"
                )
            elif func in ('quantile', 'percentile'):
                alias_val = alias or self._quantile_alias(col, func, item['q'])
                expr = f"pl.col('{col}').quantile({item['q']!r}, interpolation='linear').alias('{alias_val}')"
            else:
                polars_func = self._POLARS_AGG_MAP.get(func, func)
                expr = f"pl.col('{col}').{polars_func}()"
                if alias:
                    expr += f".alias('{alias}')"

            agg_exprs.append(expr)

        return f"{tbl} = {tbl}.group_by({by_code}).agg([{', '.join(agg_exprs)}])"

    def generate_pivot_polars(self, ast_node):
        tbl = ast_node['table_name']
        index = ast_node['index']
        columns = ast_node['columns']
        agg_list = ast_node.get('agg_list', [])
        if any(item.get('custom') for item in agg_list):
            return "raise NotImplementedError('Custom aggregation functions are currently supported for whole-table and group by aggregations only')"

        def _process_arg(arg):
            if not arg:
                return None
            if isinstance(arg, dict) and arg.get('type') == 'var':
                return arg['name']
            if isinstance(arg, list):
                has_vars = any(isinstance(i, dict) and i.get('type') == 'var' for i in arg)
                if has_vars:
                    code = "[]"
                    for item in arg:
                        if isinstance(item, dict) and item.get('type') == 'var':
                            v = item['name']
                            code += f" + ({v} if isinstance({v}, list) else [{v}])"
                        else:
                            code += f" + ['{item}']"
                    return code
                if len(arg) == 1:
                    return repr(str(arg[0]))
                return repr([str(a) for a in arg])
            return repr(str(arg))

        index_str = _process_arg(index)
        columns_str = _process_arg(columns)

        if not agg_list:
            return (
                f"{tbl} = {tbl}.pivot(values=None, index={index_str}, "
                f"on={columns_str}, aggregate_function='first')"
            )

        # Extract value columns and functions (skip variable columns for now)
        static_items = [i for i in agg_list if not (isinstance(i['column'], dict) and i['column'].get('type') == 'var')]
        values = [str(i['column']) for i in static_items]
        funcs = [i['func'] for i in static_items]

        # Use first function for all values (Polars pivot takes single aggregate_function)
        polars_func = self._POLARS_PIVOT_AGG_MAP.get(funcs[0], funcs[0]) if funcs else 'sum'
        values_str = repr(values[0]) if len(values) == 1 else repr(values)

        return (
            f"{tbl} = {tbl}.pivot(values={values_str}, index={index_str}, "
            f"on={columns_str}, aggregate_function='{polars_func}')"
        )

    def generate_unpivot_polars(self, ast_node):
        tbl = ast_node['table_name']
        id_vars = ast_node['id_vars']
        value_vars = ast_node['value_vars']
        var_name = ast_node['var_name']
        value_name = ast_node['value_name']

        # Polars 1.0+: .unpivot(index=, on=, variable_name=, value_name=)
        args = [f"index={id_vars!r}"]
        if value_vars:
            args.append(f"on={value_vars!r}")
        args.append(f"variable_name={var_name!r}")
        args.append(f"value_name={value_name!r}")
        return f"{tbl} = {tbl}.unpivot({', '.join(args)})"

    # -------------------------------------------------------------------------
    # Polars Phase 4 generators — window functions
    # -------------------------------------------------------------------------

    def generate_rank_polars(self, ast_node):
        tbl = ast_node['table_name']
        col = ast_node['column']
        ascending = ast_node['ascending']
        pct = ast_node.get('pct', False)
        partition = ast_node['partition']
        result_col = ast_node['result_col']

        # Polars rank method: 'ordinal' by default; for pct we divide by n
        if pct:
            rank_expr = f"pl.col({col!r}).rank(method='ordinal') / pl.col({col!r}).count()"
        else:
            rank_expr = f"pl.col({col!r}).rank(method='ordinal')"

        if not ascending:
            # Negate rank for descending: rank of highest value becomes 1
            rank_expr = f"(pl.col({col!r}).count() + 1 - pl.col({col!r}).rank(method='ordinal'))"
            if pct:
                rank_expr = f"(pl.col({col!r}).count() + 1 - pl.col({col!r}).rank(method='ordinal')) / pl.col({col!r}).count()"

        if partition:
            if isinstance(partition, list):
                part_str = repr(partition)
            else:
                part_str = repr([partition])
            rank_expr = f"({rank_expr}).over({part_str})"

        return f"{tbl} = {tbl}.with_columns(({rank_expr}).alias({result_col!r}))"

    def generate_shift_polars(self, ast_node):
        tbl = ast_node['table_name']
        col = ast_node['column']
        periods = ast_node['periods']
        func = ast_node['func']
        partition = ast_node['partition']
        order_col = ast_node['order_col']
        result_col = ast_node['result_col']
        n = periods if func == 'lag' else -periods

        lines = []
        if order_col:
            lines.append(f"{tbl} = {tbl}.sort({order_col!r})")

        shift_expr = f"pl.col({col!r}).shift({n})"
        if partition:
            if isinstance(partition, list):
                part_str = repr(partition)
            else:
                part_str = repr([partition])
            shift_expr = f"pl.col({col!r}).shift({n}).over({part_str})"

        lines.append(f"{tbl} = {tbl}.with_columns({shift_expr}.alias({result_col!r}))")
        return '\n'.join(lines)

    def generate_cumulative_polars(self, ast_node):
        tbl = ast_node['table_name']
        func = ast_node['func']   # cumsum | cummean | cummin | cummax
        col = ast_node['column']
        partition = ast_node['partition']
        order_col = ast_node['order_col']
        result_col = ast_node['result_col']

        _CUM_MAP = {
            'cumsum': 'cum_sum',
            'cummin': 'cum_min',
            'cummax': 'cum_max',
        }

        lines = []
        if order_col:
            lines.append(f"{tbl} = {tbl}.sort({order_col!r})")

        if func == 'cummean':
            cum_expr = f"pl.col({col!r}).cum_sum() / pl.col({col!r}).cum_count()"
        else:
            polars_method = _CUM_MAP.get(func, func)
            cum_expr = f"pl.col({col!r}).{polars_method}()"

        if partition:
            if isinstance(partition, list):
                part_str = repr(partition)
            else:
                part_str = repr([partition])
            cum_expr = f"({cum_expr}).over({part_str})"

        lines.append(f"{tbl} = {tbl}.with_columns({cum_expr}.alias({result_col!r}))")
        return '\n'.join(lines)

    def generate_rolling_polars(self, ast_node):
        tbl = ast_node['table_name']
        func = ast_node['func']
        col = ast_node['column']
        window = ast_node['window']
        min_periods = ast_node.get('min_periods')
        partition = ast_node['partition']
        order_col = ast_node['order_col']
        result_col = ast_node['result_col']

        _ROLLING_MAP = {
            'mean': 'rolling_mean',
            'sum': 'rolling_sum',
            'min': 'rolling_min',
            'max': 'rolling_max',
            'std': 'rolling_std',
        }
        polars_method = _ROLLING_MAP.get(func, f'rolling_{func}')

        lines = []
        if order_col:
            lines.append(f"{tbl} = {tbl}.sort({order_col!r})")

        min_arg = "" if min_periods is None else f", min_samples={min_periods}"
        roll_expr = f"pl.col({col!r}).{polars_method}({window}{min_arg})"
        if partition:
            if isinstance(partition, list):
                part_str = repr(partition)
            else:
                part_str = repr([partition])
            roll_expr = f"pl.col({col!r}).{polars_method}({window}{min_arg}).over({part_str})"

        lines.append(f"{tbl} = {tbl}.with_columns({roll_expr}.alias({result_col!r}))")
        return '\n'.join(lines)

    # -------------------------------------------------------------------------
    # Polars Phase 5 generators — output
    # -------------------------------------------------------------------------

    def generate_python_polars(self, ast_node):
        return ast_node['code']

    def generate_show_polars(self, ast_node):
        table = ast_node['table_name']
        mode = ast_node.get('mode', 'df')
        lines = ["from IPython.display import display as _ipyd"]
        if mode == 'head':
            lines.append(f"_ipyd({table}.head(5))")
        elif mode == 'summary':
            lines.append(f"_ipyd({table}.to_pandas().describe())")
        elif mode == 'shape':
            lines.append(f"_ipyd({table}.shape)")
        elif mode == 'columns':
            lines.append(f"_ipyd(list({table}.columns))")
        else:
            lines.append(f"_ipyd({table})")
        return "\n".join(lines)

    def generate_plot_polars(self, ast_node):
        # Convert to pandas first, then delegate to the pandas plot generator
        table = ast_node['table_name']
        pd_var = f"_{table}_pd"
        lines = [f"{pd_var} = {table}.to_pandas()"]
        # Patch table_name in a copy of the node so pandas generator uses the temp var
        pandas_node = dict(ast_node, table_name=pd_var)
        lines.append(self.generate_plot_pandas(pandas_node))
        return "\n".join(lines)

    def generate_pivot_plot_polars(self, ast_node):
        # Convert to pandas first, then delegate to the pandas pivot_plot generator
        table = ast_node['table_name']
        pd_var = f"_{table}_pd"
        lines = [f"{pd_var} = {table}.to_pandas()"]
        pandas_node = dict(ast_node, table_name=pd_var)
        lines.append(self.generate_pivot_plot_pandas(pandas_node))
        return "\n".join(lines)

    generate_agg_plot_polars = generate_pivot_plot_polars

    def generate_filter_pandas(self, ast_node):
        query_str, needs_python_engine = self._build_query_string(ast_node['conditions'], ast_node['operators'])
        engine = ", engine='python'" if needs_python_engine else ""
        return f"{ast_node['table_name']} = {ast_node['table_name']}.query('{query_str}'{engine})"

    def _dq_message(self, ast_node):
        mode = ast_node.get('mode', 'assert')
        table = ast_node['table_name']
        rule = ast_node.get('rule')
        if rule == 'unique':
            cols = ', '.join(ast_node.get('columns', []))
            return f"[Pivotal] {mode} failed on {table}: {cols} must be unique"
        if rule == 'not_null':
            cols = ', '.join(ast_node.get('columns', []))
            return f"[Pivotal] {mode} failed on {table}: {cols} must not be null"
        query = ast_node.get('query_str') or ''
        return f"[Pivotal] {mode} failed on {table}: expected {query}"

    def _dq_action_code(self, ast_node, count_var='_pvt_dq_bad'):
        message = self._dq_message(ast_node)
        if ast_node.get('mode') == 'check':
            return (
                f"if {count_var}:\n"
                f"    import warnings\n"
                f"    warnings.warn(f\"{message}: {{{count_var}}} row(s)\", UserWarning, stacklevel=2)"
            )
        return f"if {count_var}: raise AssertionError(f\"{message}: {{{count_var}}} row(s)\")"

    def generate_data_quality_pandas(self, ast_node):
        table = ast_node['table_name']
        rule = ast_node.get('rule')
        if rule == 'condition':
            query_str, needs_python_engine = self._build_query_string(
                ast_node['conditions'], ast_node['operators'])
            engine = ", engine='python'" if needs_python_engine else ""
            lines = [f"_pvt_dq_bad = len({table}.query('not ({query_str})'{engine}))"]
        elif rule == 'unique':
            cols = ast_node['columns']
            lines = [f"_pvt_dq_bad = int({table}.duplicated(subset={cols!r}).sum())"]
        elif rule == 'not_null':
            cols = ast_node['columns']
            lines = [f"_pvt_dq_bad = int({table}[{cols!r}].isna().any(axis=1).sum())"]
        else:
            raise ValueError(f"Unknown data quality rule: {rule}")
        lines.append(self._dq_action_code(ast_node))
        return "\n".join(lines)
    
    def generate_select_pandas(self, ast_node):
        table = ast_node['table_name']
        if ast_node.get('column_match'):
            pattern = self._decode_string_arg(ast_node['column_match'])
            return (
                f"{table} = {table}.loc[:, "
                f"{table}.columns.to_series().astype(str).str.contains({self._py_literal(pattern)}, regex=True).to_numpy()]"
            )

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
            
            code = f"{table} = {table}.loc[:, {col_list_code}]"
        else:
            code = f"{table} = {table}.loc[:, {columns}]"
            
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

    def generate_unpivot_pandas(self, ast_node):
        """Generate pandas melt code"""
        table = ast_node['table_name']
        id_vars = ast_node['id_vars']
        value_vars = ast_node['value_vars']
        var_name = ast_node['var_name']
        value_name = ast_node['value_name']

        args = [f"id_vars={id_vars!r}"]
        if value_vars:
            args.append(f"value_vars={value_vars!r}")
        args.append(f"var_name={var_name!r}")
        args.append(f"value_name={value_name!r}")
        return f"{table} = {table}.melt({', '.join(args)})"

    def generate_rank_pandas(self, ast_node):
        table = ast_node['table_name']
        col = ast_node['column']
        ascending = ast_node['ascending']
        pct = ast_node.get('pct', False)
        partition = ast_node['partition']
        result_col = ast_node['result_col']
        kwargs = f"ascending={ascending}, pct={pct}"
        if partition:
            return f"{table}[{result_col!r}] = {table}.groupby({partition!r})[{col!r}].rank({kwargs})"
        return f"{table}[{result_col!r}] = {table}[{col!r}].rank({kwargs})"

    def generate_shift_pandas(self, ast_node):
        table = ast_node['table_name']
        col = ast_node['column']
        periods = ast_node['periods']
        func = ast_node['func']
        partition = ast_node['partition']
        order_col = ast_node['order_col']
        result_col = ast_node['result_col']
        n = periods if func == 'lag' else -periods
        lines = []
        if order_col:
            lines.append(f"{table} = {table}.sort_values({order_col!r})")
        if partition:
            lines.append(f"{table}[{result_col!r}] = {table}.groupby({partition!r})[{col!r}].shift({n})")
        else:
            lines.append(f"{table}[{result_col!r}] = {table}[{col!r}].shift({n})")
        return '\n'.join(lines)

    def generate_cumulative_pandas(self, ast_node):
        table = ast_node['table_name']
        func = ast_node['func']   # cumsum | cummean | cummin | cummax
        col = ast_node['column']
        partition = ast_node['partition']
        order_col = ast_node['order_col']
        result_col = ast_node['result_col']
        lines = []
        if order_col:
            lines.append(f"{table} = {table}.sort_values({order_col!r})")
        if func == 'cummean':
            if partition:
                lines.append(f"{table}[{result_col!r}] = {table}.groupby({partition!r})[{col!r}].transform(lambda x: x.expanding().mean())")
            else:
                lines.append(f"{table}[{result_col!r}] = {table}[{col!r}].expanding().mean()")
        else:
            pandas_method = func  # cumsum, cummin, cummax all exist on pandas
            if partition:
                lines.append(f"{table}[{result_col!r}] = {table}.groupby({partition!r})[{col!r}].{pandas_method}()")
            else:
                lines.append(f"{table}[{result_col!r}] = {table}[{col!r}].{pandas_method}()")
        return '\n'.join(lines)

    def generate_rolling_pandas(self, ast_node):
        table = ast_node['table_name']
        func = ast_node['func']
        col = ast_node['column']
        window = ast_node['window']
        min_periods = ast_node.get('min_periods')
        partition = ast_node['partition']
        order_col = ast_node['order_col']
        result_col = ast_node['result_col']
        lines = []
        if order_col:
            lines.append(f"{table} = {table}.sort_values({order_col!r})")
        rolling_args = str(window)
        if min_periods is not None:
            rolling_args += f", min_periods={min_periods}"
        if partition:
            lines.append(f"{table}[{result_col!r}] = {table}.groupby({partition!r})[{col!r}].transform(lambda x: x.rolling({rolling_args}).{func}())")
        else:
            lines.append(f"{table}[{result_col!r}] = {table}[{col!r}].rolling({rolling_args}).{func}()")
        return '\n'.join(lines)

    def generate_groupby_pandas(self, ast_node):
        by = ast_node['by']
        agg_list = ast_node.get('agg_list', [])
        custom_items = [i for i in agg_list if i.get('custom')]
        regular_items = [i for i in agg_list if not i.get('custom')]

        def _custom_alias(item):
            if item.get('alias'):
                return item['alias']
            cols = []
            for col in item.get('columns', []):
                cols.append(col['name'] if isinstance(col, dict) and col.get('type') == 'var' else str(col))
            return f"{item['func']}_{'_'.join(cols)}" if cols else item['func']

        def _custom_arg(table, col):
            if isinstance(col, dict) and col.get('type') == 'var':
                return f"{table}[{col['name']}]"
            return f"{table}[{str(col)!r}]"

        def _custom_value_code(value):
            if isinstance(value, dict) and value.get('type') == 'var':
                return value['name']
            if isinstance(value, list):
                return '[' + ', '.join(_custom_value_code(v) for v in value) + ']'
            return repr(value)

        def _custom_call(func, args, kwargs):
            parts = list(args)
            parts.extend(f"{name}={_custom_value_code(value)}" for name, value in (kwargs or {}).items())
            return f"{func}({', '.join(parts)})"

        def _custom_scalar_helper():
            return (
                "def _pivotal_custom_scalar(_value, _func_name):\n"
                "    import pandas as _pd\n"
                "    if not _pd.api.types.is_scalar(_value):\n"
                "        raise ValueError(\n"
                "            f\"Custom aggregation function '{_func_name}' must return a scalar value; \"\n"
                "            f\"got {type(_value).__name__}. Use assignment syntax for row-wise transforms.\"\n"
                "        )\n"
                "    return _value"
            )

        # Whole-table aggregation (no group-by columns)
        if by == []:
            table = ast_node['table_name']
            if agg_list:
                parts = []
                for item in regular_items:
                    col = item['column']
                    func = item['func']
                    alias = item.get('alias') or (
                        f"wmean_{col}" if func in ('wavg', 'wmean')
                        else self._quantile_alias(col, func, item.get('q')) if func in ('quantile', 'percentile')
                        else f"{col}_{func}"
                    )
                    col_code = col['name'] if isinstance(col, dict) and col.get('type') == 'var' else f"'{col}'"
                    if func in ('wavg', 'wmean'):
                        wt = item['weight']
                        parts.append(
                            f"'{alias}': [({table}[{col_code}] * {table}['{wt}']).sum()"
                            f" / {table}['{wt}'].sum()]"
                        )
                    elif func in ('quantile', 'percentile'):
                        parts.append(f"'{alias}': [{table}[{col_code}].quantile({item['q']!r})]")
                    else:
                        pandas_func = 'mean' if func == 'avg' else func
                        parts.append(f"'{alias}': [{table}[{col_code}].{pandas_func}()]")
                for item in custom_items:
                    alias = _custom_alias(item)
                    args = ', '.join(_custom_arg(table, col) for col in item['columns'])
                    call = _custom_call(item['func'], [args] if args else [], item.get('kwargs'))
                    parts.append(
                        f"'{alias}': [_pivotal_custom_scalar({call}, {item['func']!r})]"
                    )
                code = f"{table} = __import__('pandas').DataFrame({{{', '.join(parts)}}})"
                if custom_items:
                    return '\n'.join([_custom_scalar_helper(), code])
                return code
            else:
                return f"{table} = {table}.agg('sum').to_frame().T.reset_index(drop=True)"

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

        if custom_items:
            table = ast_node['table_name']
            lines = [
                f"_pivotal_custom_source = {table}.copy()",
                f"_pivotal_custom_by = {by_code} if isinstance({by_code}, list) else [{by_code}]",
            ]
            if regular_items:
                regular_ast = dict(ast_node)
                regular_ast['agg_list'] = regular_items
                lines.append(self.generate_groupby_pandas(regular_ast))
            else:
                lines.append(
                    f"{table} = _pivotal_custom_source.groupby(_pivotal_custom_by, dropna=False)"
                    ".size().reset_index(name='_pivotal_size').drop(columns=['_pivotal_size'])"
                )
            for idx, item in enumerate(custom_items):
                alias = _custom_alias(item)
                arg_list = [_custom_arg('_g', col) for col in item['columns']]
                call = _custom_call(item['func'], arg_list, item.get('kwargs'))
                cols_code = "[]"
                for col in item['columns']:
                    if isinstance(col, dict) and col.get('type') == 'var':
                        var_name = col['name']
                        cols_code += f" + ({var_name} if isinstance({var_name}, list) else [{var_name}])"
                    else:
                        cols_code += f" + [{str(col)!r}]"
                lines.append(f"_pivotal_custom_cols_{idx} = {cols_code}")
                lines.append(
                    f"_pivotal_custom_{idx} = _pivotal_custom_source.groupby(_pivotal_custom_by, dropna=False)"
                    f"[_pivotal_custom_cols_{idx}].apply("
                    f"lambda _g: _pivotal_custom_scalar({call}, {item['func']!r})"
                    f").reset_index(name={alias!r})"
                )
                lines.append(
                    f"{table} = {table}.merge(_pivotal_custom_{idx}, on=_pivotal_custom_by, how='left')"
                )
            return '\n'.join([_custom_scalar_helper()] + lines)

        if agg_list:
            table = ast_node['table_name']
            wavg_items = [i for i in agg_list if i['func'] in ('wavg', 'wmean')]
            quantile_items = [i for i in agg_list if i['func'] in ('quantile', 'percentile')]
            regular_items = [i for i in agg_list if i['func'] not in ('wavg', 'wmean', 'quantile', 'percentile')]

            # wmean requires named agg with a lambda — force that path
            if wavg_items or quantile_items:
                agg_args = []
                for item in regular_items:
                    col = item['column']
                    func = item['func']
                    alias = item.get('alias', f"{col}_{func}")
                    pandas_func = 'mean' if func == 'avg' else func
                    agg_args.append(f"{alias}=('{col}', '{pandas_func}')")
                for item in wavg_items:
                    col = item['column']
                    wt = item['weight']
                    alias = item.get('alias', f"wmean_{col}")
                    lam = (f"lambda x: (x * {table}.loc[x.index, {wt!r}]).sum()"
                           f" / {table}.loc[x.index, {wt!r}].sum()")
                    agg_args.append(f"{alias}=('{col}', {lam})")
                for item in quantile_items:
                    col = item['column']
                    alias = item.get('alias', self._quantile_alias(col, item['func'], item['q']))
                    agg_args.append(f"{alias}=('{col}', lambda x: x.quantile({item['q']!r}))")
                agg_str = ', '.join(agg_args)
                return f"{table} = {table}.groupby({by_code}).agg({agg_str}).reset_index()"

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
        if ast_node.get('column_match'):
            table = ast_node['table_name']
            pattern = self._decode_string_arg(ast_node['column_match'])
            return f"{table} = {table}.drop(columns={table}.filter(regex={self._py_literal(pattern)}).columns)"
        return f"{ast_node['table_name']} = {ast_node['table_name']}.drop(columns={ast_node['columns']})"

    def generate_round_pandas(self, ast_node):
        table = ast_node['table_name']
        decimals = ast_node['decimals']
        alias = ast_node.get('alias')
        lines = []
        for col in ast_node['columns']:
            target = alias or col
            lines.append(f"{table}[{target!r}] = {table}[{col!r}].round({decimals})")
        return '\n'.join(lines)

    def generate_cast_pandas(self, ast_node):
        table = ast_node['table_name']
        cols = ast_node['columns']
        cast_type = ast_node['cast_type']
        strict = ast_node.get('strict', False)
        lines = []
        for col in cols:
            c = f"{table}['{col}']"
            if cast_type in ('int', 'integer'):
                if strict:
                    lines.append(f"{c} = {c}.astype(int)")
                else:
                    lines.append(f"{c} = pd.to_numeric({c}, errors='coerce').astype('Int64')")
            elif cast_type == 'float':
                if strict:
                    lines.append(f"{c} = {c}.astype(float)")
                else:
                    lines.append(f"{c} = pd.to_numeric({c}, errors='coerce')")
            elif cast_type in ('str', 'string'):
                lines.append(f"{c} = {c}.astype(str)")
            elif cast_type in ('bool', 'boolean'):
                lines.append(f"{c} = {c}.astype(bool)")
            elif cast_type == 'datetime':
                if strict:
                    lines.append(f"{c} = pd.to_datetime({c})")
                else:
                    lines.append(f"{c} = pd.to_datetime({c}, errors='coerce')")
        return '\n'.join(lines)

    def generate_fillna_pandas(self, ast_node):
        t = ast_node['table_name']
        per_col = ast_node.get('per_col', {})
        if per_col:
            def _fill_value_code(v):
                if isinstance(v, _LiteralStr):
                    return repr(str(v))
                if isinstance(v, str):
                    return f"{t}[{v!r}]"
                return str(v)
            fill_dict = {col: _fill_value_code(v) for col, v in per_col.items()}
            fill_str = '{' + ', '.join(f"'{c}': {v}" for c, v in fill_dict.items()) + '}'
            return f"{t} = {t}.fillna({fill_str})"
        val = ast_node['value']
        val_code = repr(str(val)) if isinstance(val, _LiteralStr) else (f"'{val}'" if isinstance(val, str) else str(val))
        return f"{t} = {t}.fillna({val_code})"

    def generate_intersect_pandas(self, ast_node):
        t = ast_node['table_name']
        others = ast_node['tables']
        result = t
        for other in others:
            result = f"__import__('pandas').merge({result}, {other}, how='inner')"
        return f"{t} = {result}.drop_duplicates().reset_index(drop=True)"

    def generate_exclude_pandas(self, ast_node):
        t = ast_node['table_name']
        others = ast_node['tables']
        lines = [f"{t} = {t}.drop_duplicates()"]
        for other in others:
            lines.append(f"{t} = {t}.merge({other}.drop_duplicates(), how='left', indicator=True)")
            lines.append(f"{t} = {t}[{t}['_merge'] == 'left_only'].drop(columns='_merge').reset_index(drop=True)")
        return '\n'.join(lines)

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

    def generate_delete_pandas(self, ast_node):
        name = ast_node['name']
        return f"try:\n    del {name}\nexcept NameError:\n    pass"

    def generate_show_pandas(self, ast_node):
        table = ast_node['table_name']
        mode = ast_node.get('mode', 'df')
        lines = ["from IPython.display import display as _ipyd"]
        if mode == 'head':
            lines.append(f"_ipyd({table}.head())")
        elif mode == 'summary':
            lines.append(f"_ipyd({table}.describe())")
        elif mode == 'shape':
            lines.append(f"_ipyd({table}.shape)")
        elif mode == 'columns':
            lines.append(f"_ipyd(list({table}.columns))")
        else:
            lines.append(f"_ipyd({table})")
        return "\n".join(lines)

    def _format_plot_kwarg_value(self, value):
        """Render an already-resolved plot kwarg value as Python source."""
        if isinstance(value, dict) and value.get('type') == 'var':
            return value['name']
        return repr(value)

    def generate_plot_pandas(self, ast_node):
        kind = ast_node['kind']
        kwargs = ast_node.get('kwargs', {})
        table = ast_node['table_name']
        chart_key = ast_node['name']
        on = ast_node.get('on')
        by_col = ast_node.get('by')
        n_cols = int(ast_node.get('cols') or 2)
        style = ast_node.get('style')

        arg_parts = []
        if kind:
            arg_parts.append(f"kind={kind!r}")
        arg_parts.extend(
            f"{key}={self._format_plot_kwarg_value(value)}"
            for key, value in kwargs.items()
        )
        args_str = ', '.join(arg_parts)

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

        if on:
            # Layer onto an existing single-axis figure
            preserve_xlabel = 'xlabel' not in kwargs
            preserve_ylabel = 'ylabel' not in kwargs
            lines += [
                f"if '_pivotal_charts' not in globals() or {on!r} not in globals()['_pivotal_charts']:",
                f"    raise KeyError(\"plot 'on' target {on!r} not found - make sure it is created first\")",
                f"_ax = globals()['_pivotal_charts'][{on!r}]['fig'].axes[0]",
            ]
            if preserve_xlabel:
                lines.append(f"_prev_xlabel = _ax.get_xlabel()")
            if preserve_ylabel:
                lines.append(f"_prev_ylabel = _ax.get_ylabel()")
            lines.append(f"{table}.plot({args_str}, ax=_ax)")
            if preserve_xlabel:
                lines.append(f"_ax.set_xlabel(_prev_xlabel)")
            if preserve_ylabel:
                lines.append(f"_ax.set_ylabel(_prev_ylabel)")
        elif not by_col:
            # Simple plot — existing behaviour
            lines += [
                f"_ax = {table}.plot({args_str})",
                f"if '_pivotal_charts' not in globals(): globals()['_pivotal_charts'] = {{}}",
                f"globals()['_pivotal_charts'][{repr(chart_key)}] = {{'fig': _ax.get_figure(), 'data': {table}.copy()}}",
                f"{chart_key} = _ax.get_figure()",
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

        if ast_node.get('show'):
            lines += [
                "from IPython.display import display as _ipyd",
                f"_ipyd({chart_key})",
            ]

        return "\n".join(lines)

    def generate_pivot_plot_pandas(self, ast_node):
        table   = ast_node['table_name']
        name    = ast_node['name']
        kind    = ast_node['kind']
        x_col   = ast_node['x']
        x_label = ast_node.get('x_label')
        y_items = ast_node.get('y_items') or []
        y_label = ast_node.get('y_label')
        by_col  = ast_node.get('by')
        n_cols  = int(ast_node.get('cols') or 2)
        df_name = f"{name}_df"

        lines = ["import matplotlib.pyplot as plt"]

        y_cols   = [item['col'] for item in y_items]
        agg_dict = {item['col']: item['func'] for item in y_items}

        if not by_col:
            # No faceting — groupby x, aggregate each y col (possibly with different funcs)
            lines += [
                f"{df_name} = {table}.groupby({x_col!r})[{y_cols!r}].agg({agg_dict!r}).reset_index()",
                f"{df_name}.columns = {df_name}.columns.astype(str)",
                f"_ax = {df_name}.plot(x={x_col!r}, y={y_cols!r}, kind={kind!r})",
                f"{name} = _ax.get_figure()",
            ]
            if x_label: lines.append(f"_ax.set_xlabel({x_label!r})")
            if y_label: lines.append(f"_ax.set_ylabel({y_label!r})")
        elif len(y_items) == 1:
            # Single y + by → pivot so each by-value becomes a column
            y_col    = y_items[0]['col']
            agg_func = y_items[0]['func']
            lines += [
                f"{df_name} = {table}.pivot_table(index={x_col!r}, columns={by_col!r}, values={y_col!r}, aggfunc={agg_func!r})",
                f"{df_name}.columns = [str(c) for c in {df_name}.columns]",
                f"{df_name} = {df_name}.reset_index()",
                f"_pivot_y = [c for c in {df_name}.columns if c != {x_col!r}]",
                f"_ax = {df_name}.plot(x={x_col!r}, y=_pivot_y, kind={kind!r})",
                f"_ax.set_ylabel({(y_label or y_col)!r})",
                f"{name} = _ax.get_figure()",
            ]
            if x_label: lines.append(f"_ax.set_xlabel({x_label!r})")
        else:
            # Multiple y cols + by → faceted subplots per by value
            lines += [
                f"{df_name} = {table}.groupby([{x_col!r}, {by_col!r}])[{y_cols!r}].agg({agg_dict!r}).reset_index()",
                f"_by_vals = {df_name}[{by_col!r}].unique()",
                f"_n_cols = {n_cols}",
                f"_n_rows = -(-len(_by_vals) // _n_cols)",
                f"_fig, _axes = plt.subplots(_n_rows, _n_cols, figsize=(7 * _n_cols, 5 * _n_rows))",
                f"_axes = _axes.flatten() if hasattr(_axes, 'flatten') else [_axes]",
                f"for _i, _val in enumerate(_by_vals):",
                f"    _sub = {df_name}[{df_name}[{by_col!r}] == _val].plot(x={x_col!r}, y={y_cols!r}, kind={kind!r}, ax=_axes[_i], title=str(_val))",
            ]
            if x_label: lines.append(f"    _sub.set_xlabel({x_label!r})")
            if y_label: lines.append(f"    _sub.set_ylabel({y_label!r})")
            lines += [
                f"for _ax in _axes[len(_by_vals):]:",
                f"    _ax.set_visible(False)",
                f"plt.tight_layout()",
                f"{name} = _fig",
            ]

        lines += [
            f"if '_pivotal_charts' not in globals(): globals()['_pivotal_charts'] = {{}}",
            f"globals()['_pivotal_charts'][{name!r}] = {{'fig': {name}, 'data': {df_name}.copy()}}",
        ]

        if ast_node.get('show'):
            lines += [
                "from IPython.display import display as _ipyd",
                f"_ipyd({name})",
            ]

        return "\n".join(lines)

    # Backward-compat alias — old serialised AST nodes used type 'agg_plot'
    generate_agg_plot_pandas = generate_pivot_plot_pandas

    def generate_gt_table_pandas(self, ast_node):
        table = ast_node['table_name']
        name = ast_node['name']
        title = ast_node.get('title')
        subtitle = ast_node.get('subtitle')
        font_size = ast_node.get('font_size')
        font_family = ast_node.get('font_family')
        stub = ast_node.get('stub')
        stub_group = ast_node.get('stub_group')
        stub_label = ast_node.get('stub_label')
        stripe = ast_node.get('stripe', False)
        canvas = ast_node.get('canvas', 'none')
        labels      = ast_node.get('labels', [])
        formats     = ast_node.get('formats', [])
        summary     = ast_node.get('summary', [])
        spanners    = ast_node.get('spanners', [])
        style_file  = ast_node.get('style_file')

        lines = ["import great_tables as _gt_mod"]

        # Build the optional constructor keyword args (stub, group) as a snippet
        ctor_extra = ""
        if stub:
            ctor_extra += f", rowname_col={stub!r}"
        if stub_group:
            ctor_extra += f", groupname_col={stub_group!r}"

        has_auto_spanner = any(sp.get('type') == 'auto' for sp in spanners)

        # Constructor — when auto spanner is requested we must flatten MultiIndex
        # columns first because GT does not support them directly.
        if has_auto_spanner:
            lines.append(f"if isinstance({table}.columns, pd.MultiIndex):")
            lines.append(f"    _gt_orig_cols = list({table}.columns)")
            # Flatten: join non-empty levels with '|'; fall back to first level if only one
            lines.append(f"    _gt_flat = ['|'.join(str(p) for p in c if str(p)) or str(c[0]) for c in _gt_orig_cols]")
            lines.append(f"    _gt_df = {table}.copy()")
            lines.append(f"    _gt_df.columns = _gt_flat")
            lines.append(f"    _gt = _gt_mod.GT(_gt_df{ctor_extra})")
            # Auto spanners: one per top-level group that contains >1 column
            lines.append(f"    for _gt_l0 in {table}.columns.get_level_values(0).unique():")
            lines.append(f"        _gt_span = [f for c, f in zip(_gt_orig_cols, _gt_flat) if c[0] == _gt_l0 and len([p for p in c if str(p)]) > 1]")
            lines.append(f"        if _gt_span: _gt = _gt.tab_spanner(label=str(_gt_l0), columns=_gt_span)")
            lines.append(f"else:")
            lines.append(f"    _gt = _gt_mod.GT({table}{ctor_extra})")
        else:
            lines.append(f"_gt = _gt_mod.GT({table}{ctor_extra})")

        # Header
        if title or subtitle:
            args = []
            if title: args.append(f"title={title!r}")
            if subtitle: args.append(f"subtitle={subtitle!r}")
            lines.append(f"_gt = _gt.tab_header({', '.join(args)})")

        # Stub header label (column label above the stub)
        if stub_label:
            lines.append(f"_gt = _gt.tab_stubhead(label={stub_label!r})")

        # Font family — opt_table_font() only accepts font/stack, not size
        if font_family:
            lines.append(f"_gt = _gt.opt_table_font(font={font_family!r})")

        # Grand summary rows.
        # fmt_* methods don't reach grand summary cells — pass fmt= directly.
        # Use the first blanket format (col=None) found in formats as the summary fmt.
        if summary:
            _PANDAS_FNS = {
                'sum': 'sum', 'mean': 'mean', 'min': 'min',
                'max': 'max', 'median': 'median', 'count': 'count',
            }
            fns_parts = []
            for s in summary:
                fn = s['fn']
                label = s['label']
                pandas_fn = _PANDAS_FNS.get(fn, fn)
                fns_parts.append(
                    f"    {label!r}: lambda _df: _df.select_dtypes('number').{pandas_fn}()"
                )
            # Derive fmt= from the first blanket format so summary cells match body formatting
            blanket = next((f for f in formats if f.get('col') is None), None)
            if blanket:
                lines.append("import great_tables.vals as _gt_vals")
                fmt_code = self._summary_fmt_code(blanket)
            else:
                fmt_code = None
            fmt_arg = f",\n    fmt={fmt_code}" if fmt_code else ""
            lines.append(
                "_gt = _gt.grand_summary_rows(fns={\n" +
                ",\n".join(fns_parts) + "\n}" + fmt_arg + ")"
            )

        # Import style helpers whenever tab_style calls are needed
        if font_size or stub:
            lines.append("import great_tables.style as _gt_style")
            lines.append("import great_tables.loc as _gt_loc")

        # Font size — applied to body, stub, column labels, header, and grand summary rows
        if font_size:
            size_str = f"'{int(font_size)}pt'"
            lines.append(
                f"_gt = _gt.tab_style("
                f"style=_gt_style.text(size={size_str}), "
                f"locations=[_gt_loc.body(), _gt_loc.stub(), _gt_loc.stubhead(), "
                f"_gt_loc.column_labels(), _gt_loc.header(), "
                f"_gt_loc.grand_summary(), _gt_loc.grand_summary_stub()]"
                f")"
            )

        # Prevent stub text from wrapping across lines (body + grand summary stub)
        if stub:
            lines.append(
                "_gt = _gt.tab_style("
                "style=_gt_style.text(whitespace='nowrap'), "
                "locations=[_gt_loc.stub(), _gt_loc.stubhead(), _gt_loc.grand_summary_stub()]"
                ")"
            )

        # Stripe
        if stripe:
            lines.append("_gt = _gt.opt_row_striping()")

        # Column labels
        if labels:
            kwargs = ', '.join(f"{c['col']}={c['label']!r}" for c in labels)
            lines.append(f"_gt = _gt.cols_label({kwargs})")

        # Manual spanners (auto spanners are handled in the constructor block above)
        for sp in spanners:
            if sp.get('type') == 'manual':
                cols_repr = repr(sp['columns'])
                lines.append(f"_gt = _gt.tab_spanner(label={sp['label']!r}, columns={cols_repr})")

        # Format methods.
        # When col=None (blanket format), restrict to appropriate dtypes so that
        # string/object columns are silently skipped rather than raising an error.
        numeric_sel = f"{table}.select_dtypes(include='number').columns.tolist()"
        date_sel    = f"{table}.select_dtypes(include='datetime').columns.tolist()"

        for f in formats:
            fmt = f.get('fmt')
            col = f.get('col')
            if fmt == 'number':
                if col:
                    lines.append(f"_gt = _gt.fmt_number(columns={col!r}, decimals={int(f.get('decimals', 2))})")
                else:
                    lines.append(f"_gt = _gt.fmt_number(columns={numeric_sel}, decimals={int(f.get('decimals', 2))})")
            elif fmt == 'integer':
                if col:
                    lines.append(f"_gt = _gt.fmt_integer(columns={col!r})")
                else:
                    lines.append(f"_gt = _gt.fmt_integer(columns={numeric_sel})")
            elif fmt == 'currency':
                if col:
                    lines.append(f"_gt = _gt.fmt_currency(columns={col!r}, currency={f.get('code', 'USD')!r})")
                else:
                    lines.append(f"_gt = _gt.fmt_currency(columns={numeric_sel}, currency={f.get('code', 'USD')!r})")
            elif fmt == 'percent':
                if col:
                    lines.append(f"_gt = _gt.fmt_percent(columns={col!r}, decimals={int(f.get('decimals', 1))})")
                else:
                    lines.append(f"_gt = _gt.fmt_percent(columns={numeric_sel}, decimals={int(f.get('decimals', 1))})")
            elif fmt == 'date':
                if col:
                    lines.append(f"_gt = _gt.fmt_date(columns={col!r})")
                else:
                    lines.append(f"_gt = _gt.fmt_date(columns={date_sel})")

        # Style file — apply(gt) function from an external Python file
        if style_file:
            lines.append("import importlib.util as _gt_ilu")
            lines.append(f"_gt_spec = _gt_ilu.spec_from_file_location('_pv_style', {style_file!r})")
            lines.append("_gt_mod2 = _gt_ilu.module_from_spec(_gt_spec); _gt_spec.loader.exec_module(_gt_mod2)")
            lines.append("_gt = _gt_mod2.apply(_gt)")

        # Convert all px values in GT's inline CSS to pt so that:
        #   - Word import reads physical units correctly (no 96/72 DPI inflation)
        #   - Browser print renders at true physical size
        # Conversion: 1px = 72/96 pt = 0.75 pt
        # The browser renders pt→px internally so the viewer appearance is unchanged.
        lines.append("import re as _gt_re")
        lines.append(
            "def _gt_px_to_pt(h): "
            r"return _gt_re.sub(r'(\d+(?:\.\d+)?)px', "
            r"lambda _m: f'{float(_m.group(1)) * 0.75:.4g}pt', h)"
        )
        lines.append("_gt_viewer_html = _gt_px_to_pt(_gt.as_raw_html(inline_css=True))")
        lines.append("_gt_export_html = _gt_px_to_pt(_gt.as_raw_html(make_page=True, inline_css=True))")

        # Inject @page CSS into the export HTML for direct browser printing at
        # the correct physical page size.  Only added when canvas is explicit.
        _PAPER_SIZES_MM = {
            'a4': (210.0, 297.0), 'a4_landscape': (297.0, 210.0),
            'a3': (297.0, 420.0), 'a3_landscape': (420.0, 297.0),
            'letter': (215.9, 279.4), 'slide': (338.7, 190.5),
        }
        # Word-specific CSS to suppress paragraph spacing Word adds inside
        # table cells (from its "Normal" paragraph style, typically 8pt space-after).
        # mso-* properties are ignored by browsers so the viewer is unaffected.
        word_css = (
            '<style>'
            'p{margin-top:0;margin-bottom:0;}'
            'p.MsoNormal,li.MsoNormal,div.MsoNormal{margin:0;}'
            'td{mso-line-height-rule:exactly;}'
            'td p,th p{margin:0;mso-line-height-rule:exactly;}'
            '</style>'
        )
        lines.append(
            f"_gt_export_html = _gt_export_html.replace('</head>', {word_css!r} + '</head>', 1)"
        )

        if canvas in _PAPER_SIZES_MM:
            pw, ph = _PAPER_SIZES_MM[canvas]
            margin = 25.4
            uw = pw - 2 * margin
            page_css = (
                f'<style>'
                f'@page{{size:{pw}mm {ph}mm;margin:{margin}mm}}'
                f'body{{width:{uw:.2f}mm;margin:0 auto}}'
                f'</style>'
            )
            lines.append(
                f"_gt_export_html = _gt_export_html.replace('</head>', {page_css!r} + '</head>', 1)"
            )

        lines.append("if '_pivotal_gt_tables' not in globals(): globals()['_pivotal_gt_tables'] = {}")
        lines.append(
            f"globals()['_pivotal_gt_tables'][{name!r}] = {{"
            f"'viewer_html': _gt_viewer_html, "
            f"'html': _gt_export_html, "
            f"'canvas': {canvas!r}}}"
        )

        if ast_node.get('show'):
            lines += [
                "from IPython.display import display as _ipyd, HTML as _ipyHTML",
                f"_ipyd(_ipyHTML(_gt_viewer_html))",
            ]

        return "\n".join(lines)

    def generate_gt_table_polars(self, ast_node):
        # great_tables dispatches grand_summary_rows fns based on the underlying
        # DataFrame type.  When the data is a Polars DataFrame it expects Polars
        # expressions, not the pandas-style lambdas we generate.  Converting to
        # pandas first keeps all GT logic on the pandas path (same strategy as
        # generate_plot_polars / generate_agg_plot_polars).
        tbl = ast_node['table_name']
        pd_var = f"_{tbl}_gt_pd"
        pandas_node = dict(ast_node, table_name=pd_var)
        pandas_code = self.generate_gt_table_pandas(pandas_node)
        return f"{pd_var} = {tbl}.to_pandas()\n{pandas_code}"

    def _summary_fmt_code(self, fmt_dict: dict) -> str:
        """Return a lambda string for the fmt= arg of grand_summary_rows."""
        fmt = fmt_dict.get('fmt')
        if fmt == 'number':
            d = int(fmt_dict.get('decimals', 2))
            return f"lambda x: _gt_vals.fmt_number(x, decimals={d})"
        elif fmt == 'integer':
            return "lambda x: _gt_vals.fmt_integer(x)"
        elif fmt == 'currency':
            code = fmt_dict.get('code', 'USD')
            return f"lambda x: _gt_vals.fmt_currency(x, currency={code!r})"
        elif fmt == 'percent':
            d = int(fmt_dict.get('decimals', 1))
            return f"lambda x: _gt_vals.fmt_percent(x, decimals={d})"
        elif fmt == 'date':
            return "lambda x: _gt_vals.fmt_date(x)"
        return None

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
            elif comparator == 'matches':
                pattern = self._decode_string_arg(value)
                query_parts.append(f'{column}.str.contains({self._py_literal(pattern)}, regex=True, na=False)')
                needs_python_engine = True
            elif comparator == 'not matches':
                pattern = self._decode_string_arg(value)
                query_parts.append(f'not {column}.str.contains({self._py_literal(pattern)}, regex=True, na=False)')
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
            elif isinstance(value, _LiteralStr):
                query_parts.append(f'{column} {comparator} "{value}"')
            elif isinstance(value, str):
                # Unquoted identifier — treat as column reference (no quotes)
                query_parts.append(f"{column} {comparator} {value}")
            else:
                query_parts.append(f"{column} {comparator} {value}")

            if i < len(operators):
                query_parts.append(operators[i])

        return ' '.join(query_parts), needs_python_engine

    # ------------------------------------------------------------------
    # DuckDB backend helpers
    # ------------------------------------------------------------------

    def duckdb_preamble(self):
        """Return Python code that sets up the persistent DuckDB connection."""
        return (
            "import duckdb as _ddb\n"
            "import pandas as pd\n"
            "if '_pivotal_ddb' not in globals(): globals()['_pivotal_ddb'] = _ddb.connect()\n"
            "_pvt = globals()['_pivotal_ddb']"
        )

    def polars_preamble(self):
        """Return Python code that imports Polars (and matplotlib for plots)."""
        return (
            "try:\n"
            "    import polars as pl\n"
            "except RuntimeError as _e:\n"
            "    if any(_k in str(_e).lower() for _k in ('feature flag', 'sse', 'avx')):\n"
            "        raise RuntimeError(\n"
            "            'Polars failed to load due to a CPU compatibility issue. '\n"
            "            'Try: pip install polars-lts-cpu'\n"
            "        ) from _e\n"
            "    raise\n"
            "import matplotlib.pyplot as plt"
        )

    # ------------------------------------------------------------------
    # Polars filter expression builder
    # ------------------------------------------------------------------

    def _build_polars_filter(self, conditions, operators):
        """Build a Polars filter expression string from conditions and operators.

        Returns a Python expression string that evaluates to a ``pl.Expr``.
        """
        parts = []
        for condition in conditions:
            column = condition['column']
            comparator = condition['comparator']
            value = condition['value']

            if comparator == 'between':
                lo, hi = value
                parts.append(f"pl.col('{column}').is_between({lo}, {hi})")
            elif comparator == 'contains':
                parts.append(f"pl.col('{column}').str.contains('{value}')")
            elif comparator == 'not contains':
                parts.append(f"~pl.col('{column}').str.contains('{value}')")
            elif comparator == 'matches':
                pattern = self._decode_string_arg(value)
                parts.append(f"pl.col('{column}').str.contains({self._py_literal(pattern)})")
            elif comparator == 'not matches':
                pattern = self._decode_string_arg(value)
                parts.append(f"~pl.col('{column}').str.contains({self._py_literal(pattern)})")
            elif comparator == 'startswith':
                parts.append(f"pl.col('{column}').str.starts_with('{value}')")
            elif comparator == 'endswith':
                parts.append(f"pl.col('{column}').str.ends_with('{value}')")
            elif isinstance(value, dict) and value.get('type') == 'var':
                var_name = value['name']
                if comparator in ('in', 'not in'):
                    expr = f"pl.col('{column}').is_in({var_name})"
                    if comparator == 'not in':
                        expr = f"~{expr}"
                    parts.append(expr)
                else:
                    parts.append(f"(pl.col('{column}') {comparator} {var_name})")
            elif comparator in ('in', 'not in'):
                if isinstance(value, list):
                    items = ', '.join(f"'{v}'" if isinstance(v, str) else str(v) for v in value)
                    value_str = f"[{items}]"
                else:
                    value_str = f"['{value}']" if isinstance(value, str) else f"[{value}]"
                expr = f"pl.col('{column}').is_in({value_str})"
                if comparator == 'not in':
                    expr = f"~{expr}"
                parts.append(expr)
            elif isinstance(value, _LiteralStr):
                parts.append(f"(pl.col('{column}') {comparator} '{value}')")
            elif isinstance(value, str):
                # Unquoted identifier — treat as column reference
                parts.append(f"(pl.col('{column}') {comparator} pl.col('{value}'))")
            else:
                parts.append(f"(pl.col('{column}') {comparator} {value})")

        if len(parts) == 1:
            return parts[0]

        result = parts[0]
        for i, op in enumerate(operators):
            polars_op = '&' if op == 'and' else '|'
            result = f"({result}) {polars_op} ({parts[i + 1]})"
        return result

    def _build_sql_where(self, conditions, operators):
        """Build a SQL WHERE clause from filter conditions.

        Returns:
            (where_clause, preamble_lines, use_fstring)
            where_clause   — SQL fragment (may contain {placeholder} if use_fstring)
            preamble_lines — Python lines to emit before the execute call
            use_fstring    — if True, wrap the SQL string in f"..." for runtime injection
        """
        parts = []
        preamble = []
        use_fstring = False
        sql_op_map = {'==': '=', '!=': '<>', '<': '<', '>': '>', '<=': '<=', '>=': '>='}

        for i, cond in enumerate(conditions):
            col = cond['column']
            comparator = cond['comparator']
            val = cond['value']
            # Quote column names that contain spaces
            sql_col = f'"{col}"' if ' ' in str(col) else col

            if comparator == 'between':
                lo, hi = val
                parts.append(f"{sql_col} BETWEEN {lo} AND {hi}")
            elif comparator == 'contains':
                parts.append(f"{sql_col} LIKE '%{val}%'")
            elif comparator == 'not contains':
                parts.append(f"{sql_col} NOT LIKE '%{val}%'")
            elif comparator == 'matches':
                pattern = self._decode_string_arg(val)
                parts.append(f"REGEXP_MATCHES({sql_col}, {self._sql_literal(pattern)})")
            elif comparator == 'not matches':
                pattern = self._decode_string_arg(val)
                parts.append(f"NOT REGEXP_MATCHES({sql_col}, {self._sql_literal(pattern)})")
            elif comparator == 'startswith':
                parts.append(f"{sql_col} LIKE '{val}%'")
            elif comparator == 'endswith':
                parts.append(f"{sql_col} LIKE '%{val}'")
            elif isinstance(val, dict) and val.get('type') == 'var':
                var_name = val['name']
                if comparator in ('in', 'not in'):
                    tmp = f"_ddb_in_{i}"
                    preamble.append(f"{tmp} = ', '.join(repr(v) for v in {var_name})")
                    sql_in_op = 'NOT IN' if comparator == 'not in' else 'IN'
                    parts.append(f"{sql_col} {sql_in_op} ({{{tmp}}})")
                    use_fstring = True
                else:
                    sql_op = sql_op_map.get(comparator, comparator)
                    parts.append(f"{sql_col} {sql_op} {{{var_name}}}")
                    use_fstring = True
            elif comparator in ('in', 'not in'):
                sql_in_op = 'NOT IN' if comparator == 'not in' else 'IN'
                if isinstance(val, list):
                    items = ', '.join(repr(v) for v in val)
                else:
                    items = repr(val)
                parts.append(f"{sql_col} {sql_in_op} ({items})")
            elif isinstance(val, _LiteralStr):
                sql_op = sql_op_map.get(comparator, comparator)
                parts.append(f"{sql_col} {sql_op} '{val}'")
            elif isinstance(val, str):
                sql_op = sql_op_map.get(comparator, comparator)
                parts.append(f"{sql_col} {sql_op} {val}")
            else:
                sql_op = sql_op_map.get(comparator, comparator)
                parts.append(f"{sql_col} {sql_op} {val}")

            if i < len(operators):
                parts.append(operators[i].upper())

        return ' '.join(parts), preamble, use_fstring

    # ------------------------------------------------------------------
    # DuckDB code generators — Phase 1
    # ------------------------------------------------------------------

    def generate_validate_table_duckdb(self, ast_node):
        t = ast_node['table_name']
        marker = f"#__pivotal__\n__table_name__ = '{t}'\n#__pivotal__"
        check = (
            f"_pvt_tables = [r[0] for r in _pvt.execute('SHOW TABLES').fetchall()]\n"
            f"if '{t}' not in _pvt_tables: raise NameError(\"DuckDB table '{t}' does not exist\")"
        )
        return f"{check}\n{marker}"

    def generate_copy_table_duckdb(self, ast_node):
        t = ast_node['table_name']
        src = ast_node['copy_from']
        marker = f"#__pivotal__\n__table_name__ = '{t}'\n#__pivotal__"
        return f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS SELECT * FROM {src}")\n{marker}'

    def generate_load_table_duckdb(self, ast_node):
        t = ast_node['table_name']
        source = ast_node['source']
        marker = f"#__pivotal__\n__table_name__ = '{t}'\n#__pivotal__"

        if isinstance(source, dict) and source.get('type') == 'var':
            var = source['name']
            load_code = (
                f"_src = {var}.replace('\\\\', '/')\n"
                f"_ext = _src.rsplit('.', 1)[-1].lower() if '.' in _src else ''\n"
                f"if _ext in ('xlsx', 'xls'):\n"
                f"    _df_tmp = pd.read_excel(_src)\n"
                f"    _pvt.register('_load_tmp', _df_tmp)\n"
                f"    _pvt.execute('CREATE OR REPLACE TABLE {t} AS SELECT * FROM _load_tmp')\n"
                f"elif _ext == 'parquet':\n"
                f"    _pvt.execute(f\"CREATE OR REPLACE TABLE {t} AS SELECT * FROM read_parquet('{{_src}}')\")\n"
                f"else:\n"
                f"    _pvt.execute(f\"CREATE OR REPLACE TABLE {t} AS SELECT * FROM read_csv('{{_src}}')\")"
            )
        else:
            # Normalise to forward slashes so Windows paths are safe inside SQL strings
            source_str = str(source).replace('\\', '/')
            ext = source_str.rsplit('.', 1)[-1].lower() if '.' in source_str else ''
            if ext in ('xlsx', 'xls'):
                load_code = (
                    f"_df_tmp = pd.read_excel('{source_str}')\n"
                    f"_pvt.register('_load_tmp', _df_tmp)\n"
                    f"_pvt.execute('CREATE OR REPLACE TABLE {t} AS SELECT * FROM _load_tmp')"
                )
            elif ext == 'parquet':
                load_code = f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS SELECT * FROM read_parquet(\'{source_str}\')")'
            else:
                load_code = f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS SELECT * FROM read_csv(\'{source_str}\')")'

        sanitise_code = (
            f"{_SANITISE_HELPER}"
            f"_pvt_info = [r[1] for r in _pvt.execute('PRAGMA table_info({t})').fetchall()]\n"
            f"_, _pvt_renamed = _pvt_sanitise(_pvt_info)\n"
            f"for _pvt_old, _pvt_new in _pvt_renamed.items():\n"
            f"    _pvt.execute(f'ALTER TABLE {t} RENAME COLUMN \"{{_pvt_old}}\" TO \"{{_pvt_new}}\"')\n"
            f"if _pvt_renamed:\n"
            f"    import warnings\n"
            f"    warnings.warn(f\"[Pivotal] Column(s) renamed in '{t}': {{_pvt_renamed}}\", UserWarning, stacklevel=2)\n"
        )
        return f"{load_code}\n{sanitise_code}\n{marker}"

    def generate_bulk_load_duckdb(self, ast_node):
        pandas_code = self.generate_bulk_load_pandas(ast_node)
        lines = [pandas_code]

        if ast_node['mode'] == 'concat':
            t = ast_node['table_name']
            lines.extend([
                f"_pvt.register('_bulk_tmp', {t})",
                f"_pvt.execute('CREATE OR REPLACE TABLE {t} AS SELECT * FROM _bulk_tmp')",
            ])
            return "\n".join(lines)

        alias_expr = self._bulk_aliases_code(ast_node)
        lines.extend([
            f"_pvt_aliases = list({alias_expr})",
            "for _pvt_alias in _pvt_aliases:",
            "    _pvt.register('_bulk_tmp', globals()[_pvt_alias])",
            "    _pvt.execute(f'CREATE OR REPLACE TABLE {_pvt_alias} AS SELECT * FROM _bulk_tmp')",
        ])
        return "\n".join(lines)

    def generate_filter_duckdb(self, ast_node):
        t = ast_node['table_name']
        where, preamble, use_fstring = self._build_sql_where(
            ast_node['conditions'], ast_node['operators']
        )
        sql = f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM {t} WHERE {where}"
        lines = list(preamble)
        if use_fstring:
            lines.append(f'_pvt.execute(f{sql!r})')
        else:
            lines.append(f'_pvt.execute({sql!r})')
        return '\n'.join(lines)

    def generate_data_quality_duckdb(self, ast_node):
        t = ast_node['table_name']
        rule = ast_node.get('rule')
        preamble = []
        use_fstring = False

        if rule == 'condition':
            where, preamble, use_fstring = self._build_sql_where(
                ast_node['conditions'], ast_node['operators']
            )
            sql = f"SELECT count(*) FROM {t} WHERE NOT ({where})"
        elif rule == 'unique':
            cols = ast_node['columns']
            partition = ', '.join(cols)
            sql = (
                f"SELECT count(*) FROM ("
                f"SELECT row_number() OVER (PARTITION BY {partition}) AS _pvt_rn FROM {t}"
                f") WHERE _pvt_rn > 1"
            )
        elif rule == 'not_null':
            where = ' OR '.join(f"{col} IS NULL" for col in ast_node['columns'])
            sql = f"SELECT count(*) FROM {t} WHERE {where}"
        else:
            raise ValueError(f"Unknown data quality rule: {rule}")

        lines = list(preamble)
        if use_fstring:
            lines.append(f"_pvt_dq_bad = _pvt.execute(f{sql!r}).fetchone()[0]")
        else:
            lines.append(f"_pvt_dq_bad = _pvt.execute({sql!r}).fetchone()[0]")
        lines.append(self._dq_action_code(ast_node))
        return '\n'.join(lines)

    def generate_select_duckdb(self, ast_node):
        t = ast_node['table_name']
        columns = ast_node['columns']
        renames = ast_node.get('renames', {})

        if ast_node.get('column_match'):
            pattern = self._decode_string_arg(ast_node['column_match'])
            return "\n".join([
                "_pvt_re = __import__('re')",
                f"_cols = [r[0] for r in _pvt.execute('DESCRIBE {t}').fetchall() if _pvt_re.search({self._py_literal(pattern)}, str(r[0]))]",
                "if not _cols: raise ValueError('select matches did not match any columns')",
                "_sel = ', '.join(_cols)",
                f'_pvt.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT {{_sel}} FROM {t}")',
            ])

        has_vars = any(isinstance(col, dict) and col.get('type') == 'var' for col in columns)

        if has_vars:
            col_list_code = '[]'
            for col in columns:
                if isinstance(col, dict) and col.get('type') == 'var':
                    v = col['name']
                    col_list_code += f" + ({v} if isinstance({v}, list) else [{v}])"
                else:
                    col_list_code += f" + ['{col}']"
            if renames:
                lines = [
                    f"_cols = {col_list_code}",
                    f"_rename = {renames!r}",
                    "_sel = ', '.join(f'{c} AS {_rename[c]}' if c in _rename else c for c in _cols)",
                    f'_pvt.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT {{_sel}} FROM {t}")',
                ]
            else:
                lines = [
                    f"_cols = {col_list_code}",
                    "_sel = ', '.join(_cols)",
                    f'_pvt.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT {{_sel}} FROM {t}")',
                ]
            return '\n'.join(lines)

        # Static column list
        if renames:
            sel_parts = [f"{col} AS {renames[col]}" if col in renames else col for col in columns]
        else:
            sel_parts = list(columns)
        sel = ', '.join(sel_parts)
        return f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS SELECT {sel} FROM {t}")'

    def generate_rename_duckdb(self, ast_node):
        t = ast_node['table_name']
        renames = ast_node['renames']
        lines = [
            f"_cols = [r[0] for r in _pvt.execute('DESCRIBE {t}').fetchall()]",
            f"_rename = {renames!r}",
            "_sel = ', '.join(f'{c} AS {_rename[c]}' if c in _rename else c for c in _cols)",
            f'_pvt.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT {{_sel}} FROM {t}")',
        ]
        return '\n'.join(lines)

    def generate_round_duckdb(self, ast_node):
        t = ast_node['table_name']
        decimals = ast_node['decimals']
        alias = ast_node.get('alias')
        exprs = [
            f"ROUND({col}, {decimals}) AS {alias or col}"
            for col in ast_node['columns']
        ]
        if alias:
            return f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS SELECT *, {", ".join(exprs)} FROM {t}")'
        excl = ', '.join(ast_node['columns'])
        return f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS SELECT * EXCLUDE ({excl}), {", ".join(exprs)} FROM {t}")'

    def generate_sort_duckdb(self, ast_node):
        t = ast_node['table_name']
        columns = ast_node['columns']
        ascending = ast_node['ascending']
        has_vars = any(isinstance(col, dict) and col.get('type') == 'var' for col in columns)

        if has_vars:
            parts_code = '[]'
            for col, asc in zip(columns, ascending):
                direction = 'ASC' if asc else 'DESC'
                if isinstance(col, dict) and col.get('type') == 'var':
                    v = col['name']
                    parts_code += (
                        f" + [f'{{c}} {direction}' for c in "
                        f"({v} if isinstance({v}, list) else [{v}])]"
                    )
                else:
                    parts_code += f" + ['{col} {direction}']"
            lines = [
                f"_order = {parts_code}",
                f'_pvt.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM {t} ORDER BY {{\\", \\".join(_order)}}")',
            ]
            return '\n'.join(lines)

        order = ', '.join(
            f"{col} {'ASC' if asc else 'DESC'}"
            for col, asc in zip(columns, ascending)
        )
        return f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS SELECT * FROM {t} ORDER BY {order}")'

    def generate_drop_duckdb(self, ast_node):
        t = ast_node['table_name']
        if ast_node.get('column_match'):
            pattern = self._decode_string_arg(ast_node['column_match'])
            return "\n".join([
                "_pvt_re = __import__('re')",
                f"_cols = [r[0] for r in _pvt.execute('DESCRIBE {t}').fetchall() if _pvt_re.search({self._py_literal(pattern)}, str(r[0]))]",
                "_excl = ', '.join(_cols)",
                f'_pvt.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * EXCLUDE ({{_excl}}) FROM {t}" if _cols else "CREATE OR REPLACE TABLE {t} AS SELECT * FROM {t}")',
            ])
        excl = ', '.join(ast_node['columns'])
        return f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS SELECT * EXCLUDE ({excl}) FROM {t}")'

    def generate_cast_duckdb(self, ast_node):
        t = ast_node['table_name']
        cols = ast_node['columns']
        cast_type = ast_node['cast_type']
        strict = ast_node.get('strict', False)
        _SQL_TYPES = {
            'int': 'INTEGER', 'integer': 'INTEGER',
            'float': 'DOUBLE',
            'str': 'VARCHAR', 'string': 'VARCHAR',
            'bool': 'BOOLEAN', 'boolean': 'BOOLEAN',
            'datetime': 'TIMESTAMP',
        }
        sql_type = _SQL_TYPES.get(cast_type, 'VARCHAR')
        cast_fn = 'CAST' if strict else 'TRY_CAST'
        lines = []
        for col in cols:
            expr = f"{cast_fn}({col} AS {sql_type}) AS {col}"
            lines.append(
                f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS '
                f'SELECT * REPLACE ({expr}) FROM {t}")'
            )
        return '\n'.join(lines)

    def generate_distinct_duckdb(self, ast_node):
        t = ast_node['table_name']
        cols = ast_node['columns']
        if cols:
            sel = ', '.join(cols)
            return f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS SELECT DISTINCT {sel} FROM {t}")'
        return f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS SELECT DISTINCT * FROM {t}")'

    def generate_concat_duckdb(self, ast_node):
        t = ast_node['table_name']
        others = ast_node['tables']
        union = ' UNION ALL '.join(
            [f"SELECT * FROM {t}"] + [f"SELECT * FROM {o}" for o in others]
        )
        return f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS {union}")'

    def generate_merge_duckdb(self, ast_node):
        t = ast_node['table_name']
        right = ast_node['right_table']
        keys = ast_node['keys']
        how = ast_node['how']
        join_map = {
            'inner': 'INNER JOIN',
            'left': 'LEFT JOIN',
            'right': 'RIGHT JOIN',
            'outer': 'FULL OUTER JOIN',
        }
        join_type = join_map.get(how, 'INNER JOIN')

        kwargs = ast_node.get('kwargs') or {}
        if isinstance(kwargs, str):
            kwargs = {}
        left_on = kwargs.get('left_on')
        right_on = kwargs.get('right_on')

        if left_on and right_on:
            sql = (f"CREATE OR REPLACE TABLE {t} AS SELECT {t}.*, {right}.* "
                   f"FROM {t} {join_type} {right} ON {t}.{left_on} = {right}.{right_on}")
        elif not keys or keys == '':
            sql = f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM {t} NATURAL {join_type} {right}"
        else:
            if isinstance(keys, list):
                using = ', '.join(keys)
            else:
                using = str(keys).strip("[]'\"")
            sql = f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM {t} {join_type} {right} USING ({using})"
        return f'_pvt.execute("{sql}")'

    # ------------------------------------------------------------------
    # DuckDB code generators — Phase 2
    # ------------------------------------------------------------------

    _DDB_AGG_MAP = {
        'sum': 'SUM', 'mean': 'AVG', 'avg': 'AVG', 'count': 'COUNT',
        'min': 'MIN', 'max': 'MAX', 'median': 'MEDIAN', 'std': 'STDDEV',
        'nunique': 'COUNT_DISTINCT',  # placeholder — handled explicitly below
    }

    @staticmethod
    def _ddb_agg_expr(col, func, alias, weight=None, q=None):
        """Return a SQL agg expression string for one agg_list item."""
        if func in ('wavg', 'wmean'):
            return f"SUM({col} * {weight}) / NULLIF(SUM({weight}), 0) AS {alias}"
        if func in ('quantile', 'percentile'):
            return f"QUANTILE_CONT({col}, {q}) AS {alias}"
        if func == 'nunique':
            return f"COUNT(DISTINCT {col}) AS {alias}"
        sql_func = CodeGenerator._DDB_AGG_MAP.get(func, func.upper())
        return f"{sql_func}({col}) AS {alias}"

    @staticmethod
    def _ddb_by_parts(by):
        """Return (by_list, has_vars, by_list_code).

        by_list      — list of static column names (None if has_vars)
        has_vars     — True when any column is a Python var reference
        by_list_code — Python expression that evaluates to the by list at runtime
        """
        if isinstance(by, dict) and by.get('type') == 'var':
            v = by['name']
            return None, True, f"({v} if isinstance({v}, list) else [{v}])"
        if isinstance(by, list):
            has_vars = any(isinstance(item, dict) and item.get('type') == 'var'
                           for item in by)
            if has_vars:
                code = '[]'
                for item in by:
                    if isinstance(item, dict) and item.get('type') == 'var':
                        v = item['name']
                        code += f" + ({v} if isinstance({v}, list) else [{v}])"
                    else:
                        code += f" + ['{item}']"
                return None, True, code
            return list(by), False, repr(list(by))
        col = str(by)
        return [col], False, repr([col])

    def generate_groupby_duckdb(self, ast_node):
        t = ast_node['table_name']
        by = ast_node['by']
        agg_list = ast_node.get('agg_list', [])
        if any(item.get('custom') for item in agg_list):
            return "raise NotImplementedError('Custom aggregation functions are currently supported only by the pandas backend')"

        # Whole-table aggregation (no group-by columns)
        if by == []:
            if agg_list:
                sel_parts = []
                for item in agg_list:
                    col = item['column']
                    func = item['func']
                    alias = item.get('alias') or (col if isinstance(col, str) else f"agg_{func}")
                    sel_parts.append(self._ddb_agg_expr(col, func, alias, item.get('weight'), item.get('q')))
                sel = ', '.join(sel_parts)
                return f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS SELECT {sel} FROM {t}")'
            else:
                return f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS SELECT * FROM {t} LIMIT 0")'

        by_list, has_var_by, by_list_code = self._ddb_by_parts(by)
        agg_has_vars = any(
            isinstance(item.get('column'), dict) and item['column'].get('type') == 'var'
            for item in agg_list
        )
        needs_runtime = has_var_by or agg_has_vars

        # ── Static path ────────────────────────────────────────────────
        if not needs_runtime:
            group_by_sql = ', '.join(by_list)

            if agg_list:
                sel_parts = list(by_list)
                for item in agg_list:
                    col = item['column']
                    func = item['func']
                    alias = item.get('alias') or col
                    sel_parts.append(self._ddb_agg_expr(col, func, alias,
                                                         item.get('weight'), item.get('q')))
                sel = ', '.join(sel_parts)
                sql = f"CREATE OR REPLACE TABLE {t} AS SELECT {sel} FROM {t} GROUP BY {group_by_sql}"
                return f'_pvt.execute("{sql}")'
            else:
                # No agg → SUM all non-by numeric columns; column names/types unknown at codegen
                _num_types = "{'INTEGER','BIGINT','DOUBLE','FLOAT','DECIMAL','HUGEINT','SMALLINT','TINYINT','REAL','UBIGINT','UINTEGER','USMALLINT','UTINYINT','FLOAT4','FLOAT8'}"
                lines = [
                    f"_ddb_desc = _pvt.execute('DESCRIBE {t}').fetchall()",
                    f"_ddb_by   = {by_list!r}",
                    f"_ddb_vals = [r[0] for r in _ddb_desc if r[0] not in _ddb_by"
                    f"             and r[1].upper().split('(')[0] in {_num_types}]",
                    f"_ddb_sel  = ', '.join(_ddb_by) + (', ' + ', '.join("
                    f"f'SUM({{c}}) AS {{c}}' for c in _ddb_vals) if _ddb_vals else '')",
                    f"_ddb_grp  = ', '.join(_ddb_by)",
                    f'_pvt.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT {{_ddb_sel}} '
                    f'FROM {t} GROUP BY {{_ddb_grp}}")',
                ]
                return '\n'.join(lines)

        # ── Runtime path (variable by or agg columns) ──────────────────
        lines = [f"_ddb_by = {by_list_code}"]
        if agg_list:
            lines.append("_ddb_sel_parts = list(_ddb_by)")
            for item in agg_list:
                col = item['column']
                func = item['func']
                alias = item.get('alias') or (f"agg_{func}" if isinstance(col, dict) else col)
                if isinstance(col, dict) and col.get('type') == 'var':
                    v = col['name']
                    if func in ('wavg', 'wmean'):
                        wt = item.get('weight', '')
                        lines.append(
                            f"_ddb_sel_parts.append("
                            f"f'SUM({{{v}}} * {wt}) / NULLIF(SUM({wt}), 0) AS {alias}')"
                        )
                    elif func == 'nunique':
                        lines.append(
                            f"_ddb_sel_parts.append(f'COUNT(DISTINCT {{{v}}}) AS {alias}')"
                        )
                    elif func in ('quantile', 'percentile'):
                        lines.append(
                            f"_ddb_sel_parts.append(f'QUANTILE_CONT({{{v}}}, {item.get('q')}) AS {alias}')"
                        )
                    else:
                        sql_func = self._DDB_AGG_MAP.get(func, func.upper())
                        lines.append(
                            f"_ddb_sel_parts.append(f'{sql_func}({{{v}}}) AS {alias}')"
                        )
                else:
                    expr = self._ddb_agg_expr(col, func, alias, item.get('weight'), item.get('q'))
                    lines.append(f"_ddb_sel_parts.append({expr!r})")
            lines.append("_ddb_sel = ', '.join(_ddb_sel_parts)")
        else:
            _num_types = "{'INTEGER','BIGINT','DOUBLE','FLOAT','DECIMAL','HUGEINT','SMALLINT','TINYINT','REAL','UBIGINT','UINTEGER','USMALLINT','UTINYINT','FLOAT4','FLOAT8'}"
            lines += [
                f"_ddb_desc = _pvt.execute('DESCRIBE {t}').fetchall()",
                f"_ddb_vals = [r[0] for r in _ddb_desc if r[0] not in _ddb_by"
                f"             and r[1].upper().split('(')[0] in {_num_types}]",
                "_ddb_sel  = ', '.join(_ddb_by) + (', ' + ', '.join("
                "f'SUM({c}) AS {c}' for c in _ddb_vals) if _ddb_vals else '')",
            ]
        lines += [
            "_ddb_grp = ', '.join(_ddb_by)",
            f'_pvt.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT {{_ddb_sel}} '
            f'FROM {t} GROUP BY {{_ddb_grp}}")',
        ]
        return '\n'.join(lines)

    def generate_pivot_duckdb(self, ast_node):
        t = ast_node['table_name']
        index = ast_node['index']    # rows → GROUP BY
        columns = ast_node['columns']  # pivot column → ON
        agg_list = ast_node.get('agg_list', [])
        if any(item.get('custom') for item in agg_list):
            return "raise NotImplementedError('Custom aggregation functions are currently supported for whole-table and group by aggregations only')"

        def _static_cols(arg):
            """Return list of strings if arg is fully static, else None."""
            if not arg:
                return []
            if isinstance(arg, dict) and arg.get('type') == 'var':
                return None
            if isinstance(arg, list):
                if any(isinstance(i, dict) and i.get('type') == 'var' for i in arg):
                    return None
                return [str(a) for a in arg]
            return [str(arg)]

        def _runtime_col_code(arg):
            """Return Python expression that evaluates to a list at runtime."""
            if isinstance(arg, dict) and arg.get('type') == 'var':
                v = arg['name']
                return f"({v} if isinstance({v}, list) else [{v}])"
            if isinstance(arg, list):
                code = '[]'
                for item in arg:
                    if isinstance(item, dict) and item.get('type') == 'var':
                        v = item['name']
                        code += f" + ({v} if isinstance({v}, list) else [{v}])"
                    else:
                        code += f" + ['{item}']"
                return code
            return repr([str(arg)])

        index_cols = _static_cols(index)
        on_cols    = _static_cols(columns)
        agg_has_vars = any(isinstance(item.get('column'), dict) for item in agg_list)
        needs_runtime = (index_cols is None) or (on_cols is None) or agg_has_vars

        # Build static USING clause when possible
        using_parts = []
        if not agg_has_vars and agg_list:
            for item in agg_list:
                col  = item['column']
                func = item['func']
                alias = item.get('alias')  # None if not explicitly specified
                sql_func = self._DDB_AGG_MAP.get(func, func.upper())
                expr = f"{sql_func}({col})"
                if alias:
                    expr += f" AS {alias}"
                using_parts.append(expr)

        if not needs_runtime:
            on_col    = on_cols[0] if on_cols else ''
            group_by  = ', '.join(index_cols) if index_cols else ''
            using     = ', '.join(using_parts) if using_parts else ''

            parts = [f"PIVOT {t}"]
            if on_col:
                parts.append(f"ON {on_col}")
            if using:
                parts.append(f"USING {using}")
            if group_by:
                parts.append(f"GROUP BY {group_by}")

            sql = f"CREATE OR REPLACE TABLE {t} AS {' '.join(parts)}"
            return f'_pvt.execute("{sql}")'

        # ── Runtime path ───────────────────────────────────────────────
        lines = []
        if on_cols is None:
            lines.append(f"_ddb_on = {_runtime_col_code(columns)}[0]")
            on_expr = '{_ddb_on}'
        else:
            on_expr = on_cols[0] if on_cols else ''

        if index_cols is None:
            lines.append(f"_ddb_idx = {_runtime_col_code(index)}")
            grp_expr = "{', '.join(_ddb_idx)}"
        else:
            grp_expr = ', '.join(index_cols) if index_cols else ''

        if agg_has_vars:
            lines.append("_ddb_using_parts = []")
            for item in agg_list:
                col  = item['column']
                func = item['func']
                alias = item.get('alias')  # None if not explicitly specified
                sql_func = self._DDB_AGG_MAP.get(func, func.upper())
                if isinstance(col, dict) and col.get('type') == 'var':
                    v = col['name']
                    base = f"f'{sql_func}({{{v}}})"
                    if alias:
                        base += f" AS {alias}"
                    lines.append(f"_ddb_using_parts.append({base}')")
                else:
                    base_expr = f"{sql_func}({col})"
                    if alias:
                        base_expr += f" AS {alias}"
                    lines.append(f"_ddb_using_parts.append({base_expr!r})")
            using_expr = "{', '.join(_ddb_using_parts)}"
        else:
            using_expr = ', '.join(using_parts) if using_parts else ''

        pivot_sql = f"CREATE OR REPLACE TABLE {t} AS PIVOT {t}"
        if on_expr:
            pivot_sql += f" ON {on_expr}"
        if using_expr:
            pivot_sql += f" USING {using_expr}"
        if grp_expr:
            pivot_sql += f" GROUP BY {grp_expr}"
        lines.append(f'_pvt.execute(f"{pivot_sql}")')
        return '\n'.join(lines)

    def generate_unpivot_duckdb(self, ast_node):
        t         = ast_node['table_name']
        id_vars   = ast_node['id_vars']   # list of strings
        value_vars = ast_node['value_vars']  # list of strings or empty
        var_name  = ast_node['var_name']
        value_name = ast_node['value_name']

        if value_vars:
            on_cols = ', '.join(value_vars)
            sql = (f"CREATE OR REPLACE TABLE {t} AS "
                   f"UNPIVOT {t} ON {on_cols} "
                   f"INTO NAME '{var_name}' VALUE '{value_name}'")
            return f'_pvt.execute("{sql}")'

        # value_vars not specified → melt all non-id columns (runtime DESCRIBE)
        lines = [
            f"_ddb_cols = [r[0] for r in _pvt.execute('DESCRIBE {t}').fetchall()]",
            f"_ddb_id   = {id_vars!r}",
            f"_ddb_vals = [c for c in _ddb_cols if c not in _ddb_id]",
            f"_ddb_on   = ', '.join(_ddb_vals)",
            f'_pvt.execute(f"CREATE OR REPLACE TABLE {t} AS '
            f"UNPIVOT {t} ON {{_ddb_on}} "
            f"INTO NAME '{var_name}' VALUE '{value_name}'\")",
        ]
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # DuckDB code generators — Phase 3
    # ------------------------------------------------------------------

    # ── Expression / string helpers ────────────────────────────────────

    @staticmethod
    def _translate_assign_expr_to_sql(expr):
        """Minimal translation of a pandas-eval expression to SQL.
        - Converts double-quoted string literals to single-quoted.
        - Converts :varname Python variable references to {varname} f-string placeholders.

        Returns (sql_str, uses_pyvar) where uses_pyvar is True if any :varname was found.
        """
        import re
        result = re.sub(r'"([^"]*)"', lambda m: "'" + m.group(1) + "'", expr)
        result, n = re.subn(r':([a-zA-Z_][a-zA-Z0-9_]*)', r'{\1}', result)
        return result, n > 0

    def _try_sql_string_concat(self, expr):
        """Translate col + "lit" + col concatenation to SQL col || 'lit' || col.
        Tokens may be string literals, bare column names, or string function calls.
        Returns the SQL string, or None if not recognised as string concat."""
        if '+' not in expr or ('"' not in expr and "'" not in expr):
            return None
        tokens = self._split_on_plus(expr)
        if len(tokens) < 2:
            return None
        parts = []
        for tok in tokens:
            tok = tok.strip()
            if (tok.startswith('"') and tok.endswith('"')) or \
               (tok.startswith("'") and tok.endswith("'")):
                parts.append("'" + tok[1:-1] + "'")
            elif re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*', tok):
                parts.append(tok)   # bare column name in SQL
            else:
                sql = self._try_sql_string_func(tok)
                if sql is not None:
                    parts.append(sql)  # e.g. left(season,4) → LEFT(season, 4)
                else:
                    return None        # unrecognised token — fall through
        return ' || '.join(parts)

    def _try_sql_cast_func(self, expr):
        """Translate an inline cast call int(col)/float(col)/etc to SQL, or return None."""
        m = re.fullmatch(r'([a-zA-Z][a-zA-Z0-9_]*)\s*\((.+)\)\s*', expr.strip(), re.DOTALL)
        if not m or m.group(1) not in self._CAST_FUNCS:
            return None
        func = m.group(1)
        args = self._split_func_args(m.group(2))
        if not args:
            return None
        col = args[0].strip()
        _SQL_TYPES = {
            'int': 'INTEGER', 'integer': 'INTEGER',
            'float': 'DOUBLE',
            'str': 'VARCHAR', 'string': 'VARCHAR',
            'bool': 'BOOLEAN', 'boolean': 'BOOLEAN',
            'datetime': 'TIMESTAMP',
        }
        sql_type = _SQL_TYPES.get(func)
        if sql_type is None:
            return None
        return f"TRY_CAST({col} AS {sql_type})"

    def _try_sql_date_func(self, expr):
        """Translate a date function call to SQL, or return None."""
        m = re.fullmatch(r'([a-zA-Z][a-zA-Z0-9_]*)\s*\((.+)\)\s*', expr.strip(), re.DOTALL)
        if not m:
            return None, False
        func = m.group(1).lower()
        if func not in self._DATE_FUNCS and func not in self._DATE_TWO_ARG:
            return None, False
        args = self._split_func_args(m.group(2))
        if not args:
            return None, False
        col = args[0].strip()
        _simple_sql = {
            'year': 'YEAR', 'month': 'MONTH', 'day': 'DAY',
            'quarter': 'QUARTER', 'dayofweek': 'DAYOFWEEK',
            'hour': 'HOUR', 'minute': 'MINUTE',
        }
        if func in _simple_sql:
            return f"{_simple_sql[func]}({col})", False
        if func == 'date_format' and len(args) == 2:
            fmt_inner = args[1].strip().strip('"\'')
            return f"STRFTIME({col}, '{fmt_inner}')", False
        if func == 'to_date':
            return f"CAST({col} AS DATE)", False
        if func == 'date_diff' and len(args) == 2:
            start = args[1].strip()
            return f"DATE_DIFF('day', {start}, {col})", False
        if func == 'date_add' and len(args) == 2:
            n = args[1].strip()
            if n.startswith(':'):
                var = n[1:]
                return f"({col} + INTERVAL {{{var}}} DAY)", True  # needs f-string
            return f"({col} + INTERVAL {n} DAY)", False
        return None, False

    def _try_sql_string_func(self, expr):
        """Translate a string-function call to SQL, or return None."""
        m = re.fullmatch(r'([a-zA-Z][a-zA-Z0-9_]*)\s*\((.+)\)\s*', expr.strip(), re.DOTALL)
        if not m:
            return None
        func = m.group(1).lower()
        args = self._split_func_args(m.group(2))
        if not args:
            return None
        first = args[0].strip()
        rest  = [a.strip() for a in args[1:]]
        _simple = {'upper': 'UPPER', 'lower': 'LOWER', 'trim': 'TRIM',
                   'ltrim': 'LTRIM', 'rtrim': 'RTRIM', 'len': 'LENGTH'}
        if func in _simple:
            return f"{_simple[func]}({first})"
        if func == 'left'   and len(rest) == 1: return f"LEFT({first}, {rest[0]})"
        if func == 'right'  and len(rest) == 1: return f"RIGHT({first}, {rest[0]})"
        if func == 'substr' and len(rest) == 2: return f"SUBSTR({first}, {rest[0]}, {rest[1]})"
        if func == 'replace' and len(rest) == 2:
            a = self._decode_string_arg(rest[0]); b = self._decode_string_arg(rest[1])
            return f"REPLACE({first}, {self._sql_literal(a)}, {self._sql_literal(b)})"
        if func == 'regex_extract' and len(rest) in (1, 2):
            pattern = self._decode_string_arg(rest[0])
            group = int(rest[1]) if len(rest) == 2 else 0
            return f"REGEXP_EXTRACT({first}, {self._sql_literal(pattern)}, {group})"
        if func == 'regex_replace' and len(rest) == 2:
            pattern = self._decode_string_arg(rest[0])
            repl = self._decode_string_arg(rest[1])
            return f"REGEXP_REPLACE({first}, {self._sql_literal(pattern)}, {self._sql_literal(repl)}, 'g')"
        return None

    def _substitute_agg_calls_sql(self, expr, by_cols):
        """Replace aggregate calls in expr with SQL window function calls."""
        if by_cols:
            part_str = ', '.join(by_cols) if isinstance(by_cols, list) else str(by_cols)
            over = f" OVER (PARTITION BY {part_str})"
        else:
            over = ' OVER ()'

        def replace_wavg(m):
            col, wt = m.group(1), m.group(2)
            return f"(SUM({col} * {wt}){over}) / NULLIF(SUM({wt}){over}, 0)"

        def replace_quantile(m):
            func, col, q = m.group(1).lower(), m.group(2), float(m.group(3))
            if func == 'percentile':
                q = q / 100.0
            return f"QUANTILE_CONT({col}, {q!r}){over}"

        def replace_agg(m):
            func = m.group(1).lower()
            col  = m.group(2)
            sf   = self._DDB_AGG_MAP.get(func, func.upper())
            return f"{sf}({col}){over}"

        new_expr = _WAVG_CALL_RE.sub(replace_wavg, expr)
        new_expr = _QUANTILE_CALL_RE.sub(replace_quantile, new_expr)
        new_expr = _AGG_CALL_RE.sub(replace_agg, new_expr)
        return new_expr

    @staticmethod
    def _ddb_window(partition, order_col, frame=None):
        """Build a SQL OVER (...) clause string (without OVER keyword)."""
        parts = []
        if partition:
            part_str = ', '.join(partition) if isinstance(partition, list) else str(partition)
            parts.append(f"PARTITION BY {part_str}")
        if order_col:
            parts.append(f"ORDER BY {order_col}")
        if frame:
            parts.append(frame)
        return ' '.join(parts)

    # ── Shared helper: add/replace a column in a DuckDB table ─────────

    @staticmethod
    def _ddb_upsert_col_lines(t, result_col, sql_expr, use_fstring=False):
        """Return Python code lines that add or replace `result_col` in table `t`
        using the given SQL expression (already a valid SQL fragment).
        Uses runtime DESCRIBE so it works whether the column is new or existing."""
        lines = [
            f"_ddb_desc = [r[0] for r in _pvt.execute('DESCRIBE {t}').fetchall()]",
            f"_ddb_sel  = ', '.join(c for c in _ddb_desc if c != {result_col!r})",
        ]
        sql_expr_template = sql_expr if use_fstring else sql_expr.replace('{', '{{').replace('}', '}}')
        sql = f"CREATE OR REPLACE TABLE {t} AS SELECT {{_ddb_sel}}, {sql_expr_template} AS {result_col} FROM {t}"
        lines.append(f'_pvt.execute(f{sql!r})')
        return lines

    # ── assign ────────────────────────────────────────────────────────

    def generate_assign_duckdb(self, ast_node):
        t      = ast_node['table_name']
        target = ast_node['target']

        # ── Multi-case (CASE WHEN … THEN … ELSE …) ─────────────────
        if ast_node.get('cases'):
            cases    = ast_node['cases']
            branches = [c for c in cases if c['type'] == 'case_branch']
            defaults = [c for c in cases if c['type'] == 'case_default']

            when_parts = []
            all_preamble = []
            uses_fstr = False
            for b in branches:
                where, preamble, uf = self._build_sql_where(b['conditions'], b['operators'])
                all_preamble.extend(preamble)
                uses_fstr = uses_fstr or uf
                sql_expr, upv = self._translate_assign_expr_to_sql(b['expression'])
                uses_fstr = uses_fstr or upv
                when_parts.append(f"WHEN {where} THEN {sql_expr}")

            if defaults:
                else_expr, upv = self._translate_assign_expr_to_sql(defaults[0]['expression'])
                uses_fstr = uses_fstr or upv
                case_sql = f"CASE {' '.join(when_parts)} ELSE {else_expr} END"
            else:
                case_sql = f"CASE {' '.join(when_parts)} ELSE NULL END"

            lines = (all_preamble +
                     self._ddb_upsert_col_lines(t, target, case_sql, uses_fstr))
            return '\n'.join(lines)

        expr      = ast_node['expression']
        by_cols   = ast_node.get('by_cols', [])
        conditions = ast_node.get('conditions')
        operators  = ast_node.get('operators')
        default_expr = ast_node.get('default_expr')

        # ── By-clause agg calls → window functions ─────────────────
        sql_expr_with_agg = self._substitute_agg_calls_sql(expr, by_cols)
        if sql_expr_with_agg != expr:
            sql_expr_translated, _ = self._try_sql_scalar_minmax(sql_expr_with_agg)
            if sql_expr_translated is None:
                sql_expr_translated, _ = self._translate_assign_expr_to_sql(sql_expr_with_agg)
            return '\n'.join(self._ddb_upsert_col_lines(t, target, sql_expr_translated))

        # ── Translate expression to SQL ────────────────────────────
        sql_str = self._try_sql_cast_func(expr)
        uses_pyvar_expr = False
        if sql_str is None:
            sql_str, uses_pyvar_expr = self._try_sql_date_func(expr)
        if sql_str is None:
            sql_str = self._try_sql_string_func(expr)
        if sql_str is None:
            sql_str = self._try_sql_string_concat(expr)
        if sql_str is None:
            sql_str, uses_pyvar_expr = self._try_sql_scalar_minmax(expr)
        if sql_str is None:
            sql_str, uses_pyvar_expr = self._translate_assign_expr_to_sql(expr)

        # ── Conditional (where clause) → CASE WHEN ─────────────────
        if conditions:
            where, preamble, use_fstring = self._build_sql_where(conditions, operators)
            if default_expr is not None:
                else_sql, uses_pyvar_expr2 = self._translate_assign_expr_to_sql(default_expr)
                uses_pyvar_expr = uses_pyvar_expr or uses_pyvar_expr2
                case_when_else = f"CASE WHEN {where} THEN {sql_str} ELSE {else_sql} END"
                case_sql_existing = case_sql_new = case_when_else
            else:
                case_sql_existing = f"CASE WHEN {where} THEN {sql_str} ELSE {target} END"
                case_sql_new      = f"CASE WHEN {where} THEN {sql_str} ELSE NULL END"
            desc_line = f"_ddb_desc = [r[0] for r in _pvt.execute('DESCRIBE {t}').fetchall()]"
            sel_line  = f"_ddb_sel  = ', '.join(c for c in _ddb_desc if c != {target!r})"
            # Use f-strings at runtime if the SQL contains Python variable placeholders
            if use_fstring or uses_pyvar_expr:
                choose_sql = (
                    f"_ddb_case = ("
                    f'f"{case_sql_existing}" if {target!r} in _ddb_desc '
                    f'else f"{case_sql_new}")'
                )
            else:
                choose_sql = (
                    f"_ddb_case = ("
                    f'"{case_sql_existing}" if {target!r} in _ddb_desc '
                    f'else "{case_sql_new}")'
                )
            exec_line = f'_pvt.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT {{_ddb_sel}}, {{_ddb_case}} AS {target} FROM {t}")'
            lines = list(preamble) + [desc_line, sel_line, choose_sql, exec_line]
            return '\n'.join(lines)

        # ── Simple expression ───────────────────────────────────────
        return '\n'.join(self._ddb_upsert_col_lines(t, target, sql_str, uses_pyvar_expr))

    # ── rank ──────────────────────────────────────────────────────────

    def generate_rank_duckdb(self, ast_node):
        t          = ast_node['table_name']
        col        = ast_node['column']
        ascending  = ast_node['ascending']
        pct        = ast_node.get('pct', False)
        partition  = ast_node['partition']
        result_col = ast_node['result_col']

        rank_func  = 'PERCENT_RANK()' if pct else 'RANK()'
        order_dir  = 'ASC' if ascending else 'DESC'
        win = self._ddb_window(partition, f"{col} {order_dir}")
        sql_expr = f"{rank_func} OVER ({win})"
        return '\n'.join(self._ddb_upsert_col_lines(t, result_col, sql_expr))

    # ── lag / lead ────────────────────────────────────────────────────

    def generate_shift_duckdb(self, ast_node):
        t          = ast_node['table_name']
        col        = ast_node['column']
        periods    = ast_node['periods']
        func       = ast_node['func']
        partition  = ast_node['partition']
        order_col  = ast_node['order_col']
        result_col = ast_node['result_col']

        sql_func = 'LAG' if func == 'lag' else 'LEAD'
        win = self._ddb_window(partition, order_col)
        sql_expr = f"{sql_func}({col}, {periods}) OVER ({win})"
        return '\n'.join(self._ddb_upsert_col_lines(t, result_col, sql_expr))

    # ── cumulative ────────────────────────────────────────────────────

    def generate_cumulative_duckdb(self, ast_node):
        t          = ast_node['table_name']
        func       = ast_node['func']
        col        = ast_node['column']
        partition  = ast_node['partition']
        order_col  = ast_node['order_col']
        result_col = ast_node['result_col']

        _cum_map   = {'cumsum': 'SUM', 'cummean': 'AVG', 'cummin': 'MIN', 'cummax': 'MAX'}
        sql_func   = _cum_map.get(func, 'SUM')
        frame      = 'ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW'
        win = self._ddb_window(partition, order_col, frame)
        sql_expr = f"{sql_func}({col}) OVER ({win})"
        return '\n'.join(self._ddb_upsert_col_lines(t, result_col, sql_expr))

    # ── rolling ───────────────────────────────────────────────────────

    def generate_rolling_duckdb(self, ast_node):
        t          = ast_node['table_name']
        func       = ast_node['func']
        col        = ast_node['column']
        window     = ast_node['window']
        min_periods = ast_node.get('min_periods')
        partition  = ast_node['partition']
        order_col  = ast_node['order_col']
        result_col = ast_node['result_col']

        _roll_map = {'sum': 'SUM', 'mean': 'AVG', 'avg': 'AVG',
                     'min': 'MIN', 'max': 'MAX', 'std': 'STDDEV'}
        sql_func  = _roll_map.get(func, func.upper())
        frame     = f'ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW'
        win = self._ddb_window(partition, order_col, frame)
        sql_expr = f"{sql_func}({col}) OVER ({win})"
        if min_periods is not None:
            count_expr = f"COUNT({col}) OVER ({win})"
            sql_expr = f"CASE WHEN {count_expr} >= {min_periods} THEN {sql_expr} ELSE NULL END"
        return '\n'.join(self._ddb_upsert_col_lines(t, result_col, sql_expr))

    # ── fillna ────────────────────────────────────────────────────────

    def generate_fillna_duckdb(self, ast_node):
        t = ast_node['table_name']
        per_col = ast_node.get('per_col', {})

        if per_col:
            # Build SELECT replacing only specified cols; keep others as-is via star + override
            per_entries = []
            for c, v in per_col.items():
                if isinstance(v, _LiteralStr):
                    per_entries.append(repr(c) + ': ' + repr(('literal', str(v))))
                elif isinstance(v, str):
                    per_entries.append(repr(c) + ': ' + repr(('column', v)))
                else:
                    per_entries.append(repr(c) + ': ' + repr(('scalar', v)))
            lines = [
                f"_ddb_all_cols = [r[0] for r in _pvt.execute('DESCRIBE {t}').fetchall()]",
                f"_ddb_per = {{{', '.join(per_entries)}}}",
                "_ddb_sel_parts = []",
                "for _c in _ddb_all_cols:",
                "    if _c not in _ddb_per:",
                "        _ddb_sel_parts.append(_c)",
                "    else:",
                "        _kind, _val = _ddb_per[_c]",
                "        _fill = repr(_val) if _kind == 'literal' else str(_val)",
                "        _ddb_sel_parts.append(f'COALESCE({_c}, {_fill}) AS {_c}')",
                "_ddb_sel = ', '.join(_ddb_sel_parts)",
                f'_pvt.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT {{_ddb_sel}} FROM {t}")',
            ]
            return '\n'.join(lines)

        val = ast_node['value']
        if isinstance(val, _LiteralStr):
            fill_val_line = f"_ddb_fillval = {repr(str(val))}"
            sel_line = "_ddb_sel = ', '.join(f\"COALESCE({c}, '{_ddb_fillval}') AS {c}\" for c in _ddb_cols)"
        elif isinstance(val, str):
            fill_val_line = f"_ddb_fillval = {repr(str(val))}"
            sel_line = "_ddb_sel = ', '.join(f\"COALESCE({c}, '{_ddb_fillval}') AS {c}\" for c in _ddb_cols)"
        elif val is None:
            fill_val_line = None
            sel_line = "_ddb_sel = ', '.join(f'COALESCE({c}, NULL) AS {c}' for c in _ddb_cols)"
        else:
            fill_val_line = f"_ddb_fillval = {val}"
            sel_line = "_ddb_sel = ', '.join(f'COALESCE({c}, {_ddb_fillval}) AS {c}' for c in _ddb_cols)"

        lines = []
        if fill_val_line:
            lines.append(fill_val_line)
        lines += [
            f"_ddb_cols = [r[0] for r in _pvt.execute('DESCRIBE {t}').fetchall()]",
            sel_line,
            f'_pvt.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT {{_ddb_sel}} FROM {t}")',
        ]
        return '\n'.join(lines)

    def generate_intersect_duckdb(self, ast_node):
        t = ast_node['table_name']
        others = ast_node['tables']
        parts = [f"SELECT * FROM {t}"] + [f"SELECT * FROM {o}" for o in others]
        sql = ' INTERSECT '.join(parts)
        return f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS {sql}")'

    def generate_exclude_duckdb(self, ast_node):
        t = ast_node['table_name']
        others = ast_node['tables']
        parts = [f"SELECT * FROM {t}"] + [f"SELECT * FROM {o}" for o in others]
        sql = ' EXCEPT '.join(parts)
        return f'_pvt.execute("CREATE OR REPLACE TABLE {t} AS {sql}")'

    # ── dropna ────────────────────────────────────────────────────────

    def generate_dropna_duckdb(self, ast_node):
        t    = ast_node['table_name']
        cols = ast_node.get('columns', [])

        if cols:
            not_null = ' AND '.join(f"{c} IS NOT NULL" for c in cols)
            sql = f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM {t} WHERE {not_null}"
            return f'_pvt.execute("{sql}")'

        # No columns specified — drop any row with a NULL in any column (runtime DESCRIBE)
        lines = [
            f"_ddb_cols = [r[0] for r in _pvt.execute('DESCRIBE {t}').fetchall()]",
            f"_ddb_where = ' AND '.join(f'{{c}} IS NOT NULL' for c in _ddb_cols)",
            f'_pvt.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM {t} WHERE {{_ddb_where}}")',
        ]
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # DuckDB code generators — Phase 4
    # (show / plot / gt_table materialise the table to pandas first)
    # ------------------------------------------------------------------

    @staticmethod
    def _ddb_materialize(t):
        """Return (lines, df_var_name): lines that fetch a DuckDB table into pandas."""
        df_var = f"_df_{t}"
        lines = [f"{df_var} = _pvt.execute('SELECT * FROM {t}').df()"]
        return lines, df_var

    def generate_python_duckdb(self, ast_node):
        """Pass raw Python code through verbatim (user handles _pvt if needed)."""
        return ast_node['code']

    def generate_apply_duckdb(self, ast_node):
        """Materialise to pandas, apply user function, re-register as DuckDB table."""
        t    = ast_node['table_name']
        func = ast_node['func']
        mat_lines, df_var = self._ddb_materialize(t)
        lines = mat_lines + [
            f"{df_var} = {func}({df_var})",
            f"_pvt.register('_tmp_{t}', {df_var})",
            f"_pvt.execute('CREATE OR REPLACE TABLE {t} AS SELECT * FROM _tmp_{t}')",
        ]
        return '\n'.join(lines)

    def generate_show_duckdb(self, ast_node):
        """Materialise table then display using the pandas show generator."""
        t = ast_node['table_name']
        mat_lines, df_var = self._ddb_materialize(t)
        pandas_code = self.generate_show_pandas(dict(ast_node, table_name=df_var))
        return '\n'.join(mat_lines) + '\n' + pandas_code

    def generate_plot_duckdb(self, ast_node):
        """Materialise table then plot using the pandas plot generator."""
        t = ast_node['table_name']
        mat_lines, df_var = self._ddb_materialize(t)
        pandas_code = self.generate_plot_pandas(dict(ast_node, table_name=df_var))
        return '\n'.join(mat_lines) + '\n' + pandas_code

    def generate_pivot_plot_duckdb(self, ast_node):
        """Materialise table then pivot-plot using the pandas pivot_plot generator."""
        t = ast_node['table_name']
        mat_lines, df_var = self._ddb_materialize(t)
        pandas_code = self.generate_pivot_plot_pandas(dict(ast_node, table_name=df_var))
        return '\n'.join(mat_lines) + '\n' + pandas_code

    generate_agg_plot_duckdb = generate_pivot_plot_duckdb

    def generate_gt_table_duckdb(self, ast_node):
        """Materialise table then build GT table using the pandas gt_table generator."""
        t = ast_node['table_name']
        mat_lines, df_var = self._ddb_materialize(t)
        pandas_code = self.generate_gt_table_pandas(dict(ast_node, table_name=df_var))
        return '\n'.join(mat_lines) + '\n' + pandas_code

    # ------------------------------------------------------------------
    # SQL CTE backend
    # Each generator returns one of:
    #   None         — skip silently
    #   str "-- ..." — emit as comment, no CTE created
    #   str (other)  — SELECT body; caller wraps in "alias AS (...)"
    # ------------------------------------------------------------------

    def _sql_current(self, table_name):
        """Return the current CTE alias for table_name (or the real table if first seen)."""
        return self._sql_state.get(table_name, table_name)

    def generate_validate_table_sql(self, ast_node):
        t = ast_node['table_name']
        return f"SELECT * FROM {t}"

    def generate_copy_table_sql(self, ast_node):
        src = ast_node['copy_from']
        return f"SELECT * FROM {src}"

    def generate_load_table_sql(self, ast_node):
        source = ast_node['source']
        if isinstance(source, dict) and source.get('type') == 'var':
            return f"-- [skipped: load from Python variable not supported in SQL CTE mode]"
        source_str = str(source).replace('\\', '/')
        ext = source_str.rsplit('.', 1)[-1].lower() if '.' in source_str else ''
        if ext == 'parquet':
            return f"SELECT * FROM read_parquet('{source_str}')"
        if ext in ('xlsx', 'xls'):
            return f"-- [skipped: Excel load requires Python runtime]"
        return f"SELECT * FROM read_csv('{source_str}')"

    def generate_bulk_load_sql(self, ast_node):
        return "-- [skipped: bulk load requires pandas, polars, or duckdb backend]"

    def generate_filter_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        where, preamble, use_fstring = self._build_sql_where(
            ast_node['conditions'], ast_node['operators'])
        if use_fstring or preamble:
            return f"-- [skipped: filter with Python variable reference]"
        return f"SELECT * FROM {from_alias} WHERE {where}"

    def generate_data_quality_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        mode = ast_node.get('mode', 'assert')
        rule = ast_node.get('rule')
        if rule == 'condition':
            where, preamble, use_fstring = self._build_sql_where(
                ast_node['conditions'], ast_node['operators'])
            if use_fstring or preamble:
                detail = 'condition with Python variable reference'
            else:
                detail = f"{where}"
        elif rule in ('unique', 'not_null'):
            detail = ', '.join(ast_node.get('columns', []))
        else:
            detail = rule or 'unknown'
        return f"-- [skipped: {mode} {rule} on {from_alias}: {detail}]"

    def generate_select_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        columns = ast_node['columns']
        renames = ast_node.get('renames', {})
        if ast_node.get('column_match'):
            pattern = self._decode_string_arg(ast_node['column_match'])
            return f"SELECT COLUMNS({self._sql_literal(pattern)}) FROM {from_alias}"
        has_vars = any(isinstance(c, dict) and c.get('type') == 'var' for c in columns)
        if has_vars:
            return f"-- [skipped: select with Python variable column list]"
        if renames:
            sel_parts = [f"{c} AS {renames[c]}" if c in renames else c for c in columns]
        else:
            sel_parts = list(columns)
        return f"SELECT {', '.join(sel_parts)} FROM {from_alias}"

    def generate_rename_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        renames = ast_node['renames']  # {old: new, ...}
        excl = ', '.join(renames.keys())
        new_cols = ', '.join(f"{old} AS {new}" for old, new in renames.items())
        return f"SELECT * EXCLUDE ({excl}), {new_cols} FROM {from_alias}"

    def generate_round_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        decimals = ast_node['decimals']
        alias = ast_node.get('alias')
        exprs = [
            f"ROUND({col}, {decimals}) AS {alias or col}"
            for col in ast_node['columns']
        ]
        if alias:
            return f"SELECT *, {', '.join(exprs)} FROM {from_alias}"
        return f"SELECT * EXCLUDE ({', '.join(ast_node['columns'])}), {', '.join(exprs)} FROM {from_alias}"

    def generate_sort_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        columns = ast_node['columns']
        ascending = ast_node['ascending']
        has_vars = any(isinstance(c, dict) and c.get('type') == 'var' for c in columns)
        if has_vars:
            return f"-- [skipped: sort with Python variable columns]"
        order = ', '.join(
            f"{col} {'ASC' if asc else 'DESC'}"
            for col, asc in zip(columns, ascending)
        )
        return f"SELECT * FROM {from_alias} ORDER BY {order}"

    def generate_drop_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        if ast_node.get('column_match'):
            pattern = self._decode_string_arg(ast_node['column_match'])
            return f"SELECT COLUMNS(c -> NOT REGEXP_MATCHES(c, {self._sql_literal(pattern)})) FROM {from_alias}"
        excl = ', '.join(ast_node['columns'])
        return f"SELECT * EXCLUDE ({excl}) FROM {from_alias}"

    def generate_cast_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        cols = ast_node['columns']
        cast_type = ast_node['cast_type']
        _SQL_TYPES = {
            'int': 'INTEGER', 'integer': 'INTEGER',
            'float': 'DOUBLE',
            'str': 'VARCHAR', 'string': 'VARCHAR',
            'bool': 'BOOLEAN', 'boolean': 'BOOLEAN',
            'datetime': 'TIMESTAMP',
        }
        sql_type = _SQL_TYPES.get(cast_type, 'VARCHAR')
        # Standard SQL has no TRY_CAST — always use CAST regardless of strict flag
        replaces = ', '.join(f"CAST({col} AS {sql_type}) AS {col}" for col in cols)
        return f"SELECT * REPLACE ({replaces}) FROM {from_alias}"

    def generate_distinct_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        cols = ast_node['columns']
        if cols:
            return f"SELECT DISTINCT {', '.join(cols)} FROM {from_alias}"
        return f"SELECT DISTINCT * FROM {from_alias}"

    def generate_concat_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        others = ast_node['tables']
        union_parts = [f"SELECT * FROM {from_alias}"] + [f"SELECT * FROM {o}" for o in others]
        return ' UNION ALL '.join(union_parts)

    def generate_merge_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        right = ast_node['right_table']
        keys = ast_node['keys']
        how = ast_node['how']
        join_map = {
            'inner': 'INNER JOIN', 'left': 'LEFT JOIN',
            'right': 'RIGHT JOIN', 'outer': 'FULL OUTER JOIN',
        }
        join_type = join_map.get(how, 'INNER JOIN')
        if not keys or keys == '':
            return f"SELECT * FROM {from_alias} NATURAL {join_type} {right}"
        using = ', '.join(keys) if isinstance(keys, list) else str(keys).strip("[]'\"")
        return f"SELECT * FROM {from_alias} {join_type} {right} USING ({using})"

    def generate_groupby_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        by = ast_node['by']
        agg_list = ast_node.get('agg_list', [])
        if any(item.get('custom') for item in agg_list):
            return f"-- [skipped: custom aggregation functions are supported only by the pandas backend]"

        # Whole-table aggregation (no group-by columns)
        if by == []:
            if not agg_list:
                return f"-- [skipped: summarise without agg list]"
            sel_parts = []
            for item in agg_list:
                col = item['column']
                func = item['func']
                alias = item.get('alias', f"{col}_{func}")
                sel_parts.append(self._ddb_agg_expr(col, func, alias, item.get('weight'), item.get('q')))
            return f"SELECT {', '.join(sel_parts)} FROM {from_alias}"

        by_list, has_var_by, _ = self._ddb_by_parts(by)
        if has_var_by:
            return f"-- [skipped: groupby with Python variable columns]"
        group_by_sql = ', '.join(by_list)
        if not agg_list:
            return f"-- [skipped: groupby without explicit agg_list requires runtime schema]"
        agg_has_vars = any(
            isinstance(item.get('column'), dict) and item['column'].get('type') == 'var'
            for item in agg_list
        )
        if agg_has_vars:
            return f"-- [skipped: groupby with Python variable agg columns]"
        sel_parts = list(by_list)
        for item in agg_list:
            col = item['column']
            func = item['func']
            alias = item.get('alias', f"{col}_{func}")
            sel_parts.append(self._ddb_agg_expr(col, func, alias, item.get('weight'), item.get('q')))
        return f"SELECT {', '.join(sel_parts)} FROM {from_alias} GROUP BY {group_by_sql}"

    def generate_pivot_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        index = ast_node['index']
        columns = ast_node['columns']
        agg_list = ast_node.get('agg_list', [])
        if any(item.get('custom') for item in agg_list):
            return f"-- [skipped: custom aggregation functions are supported only by pandas group by/whole-table agg]"
        # Only handle static (non-variable) args
        def _as_str_list(arg):
            if not arg:
                return []
            if isinstance(arg, dict) and arg.get('type') == 'var':
                return None
            if isinstance(arg, list):
                if any(isinstance(i, dict) for i in arg):
                    return None
                return [str(a) for a in arg]
            return [str(arg)]
        index_cols = _as_str_list(index)
        on_cols    = _as_str_list(columns)
        if index_cols is None or on_cols is None:
            return f"-- [skipped: pivot with Python variable columns]"
        using_parts = []
        for item in agg_list:
            col = item['column']
            if isinstance(col, dict):
                return f"-- [skipped: pivot with Python variable agg column]"
            func = item['func']
            alias = item.get('alias')
            sql_func = self._DDB_AGG_MAP.get(func, func.upper())
            expr = f"{sql_func}({col})"
            if alias:
                expr += f" AS {alias}"
            using_parts.append(expr)
        using_sql  = ', '.join(using_parts) if using_parts else 'COUNT(*)'
        on_sql     = ', '.join(on_cols) if on_cols else ''
        group_sql  = f" GROUP BY {', '.join(index_cols)}" if index_cols else ''
        pivot_body = f"SELECT * FROM PIVOT {from_alias}"
        if on_sql:
            pivot_body += f" ON {on_sql}"
        pivot_body += f" USING {using_sql}{group_sql}"
        return pivot_body

    def generate_unpivot_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        value_vars = ast_node.get('value_vars', [])
        var_name   = ast_node['var_name']
        value_name = ast_node['value_name']
        if not value_vars:
            return f"-- [skipped: unpivot without explicit value_vars requires runtime schema]"
        on_cols = ', '.join(value_vars)
        return (f"SELECT * FROM UNPIVOT {from_alias} ON {on_cols} "
                f"INTO NAME '{var_name}' VALUE '{value_name}'")

    def generate_assign_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        target = ast_node['target']
        if ast_node.get('cases'):
            cases    = ast_node['cases']
            branches = [c for c in cases if c['type'] == 'case_branch']
            defaults = [c for c in cases if c['type'] == 'case_default']
            when_parts = []
            for b in branches:
                where, preamble, uf = self._build_sql_where(b['conditions'], b['operators'])
                if uf or preamble:
                    return f"-- [skipped: assign with Python variable in condition]"
                sql_expr, upv = self._translate_assign_expr_to_sql(b['expression'])
                if upv:
                    return f"-- [skipped: assign with Python variable in expression]"
                when_parts.append(f"WHEN {where} THEN {sql_expr}")
            if defaults:
                else_expr, upv = self._translate_assign_expr_to_sql(defaults[0]['expression'])
                if upv:
                    return f"-- [skipped: assign with Python variable in expression]"
            else:
                else_expr = 'NULL'
            case_sql = f"CASE {' '.join(when_parts)} ELSE {else_expr} END"
            return f"SELECT *, {case_sql} AS {target} FROM {from_alias}"
        expr     = ast_node['expression']
        by_cols  = ast_node.get('by_cols', [])
        conditions = ast_node.get('conditions')
        operators  = ast_node.get('operators')
        default_expr = ast_node.get('default_expr')
        sql_expr = self._substitute_agg_calls_sql(expr, by_cols)
        sql_str = self._try_sql_cast_func(sql_expr)
        if sql_str is None:
            sql_str, upv = self._try_sql_date_func(sql_expr)
            if upv:
                return f"-- [skipped: assign with Python variable in expression]"
        if sql_str is None:
            sql_str = self._try_sql_string_func(sql_expr)
        if sql_str is None:
            sql_str = self._try_sql_string_concat(sql_expr)
        if sql_str is None:
            sql_str, upv = self._try_sql_scalar_minmax(sql_expr)
            if upv:
                return f"-- [skipped: assign with Python variable in expression]"
        if sql_str is None:
            sql_str, upv = self._translate_assign_expr_to_sql(sql_expr)
            if upv:
                return f"-- [skipped: assign with Python variable in expression]"
        if conditions:
            where, preamble, use_fstring = self._build_sql_where(conditions, operators)
            if use_fstring or preamble:
                return f"-- [skipped: assign with Python variable in condition]"
            if default_expr is not None:
                else_expr, upv = self._translate_assign_expr_to_sql(default_expr)
                if upv:
                    return f"-- [skipped: assign with Python variable in expression]"
            else:
                else_expr = 'NULL'
            case_sql = f"CASE WHEN {where} THEN {sql_str} ELSE {else_expr} END"
            return f"SELECT *, {case_sql} AS {target} FROM {from_alias}"
        return f"SELECT *, {sql_str} AS {target} FROM {from_alias}"

    def generate_rank_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        col        = ast_node['column']
        ascending  = ast_node['ascending']
        pct        = ast_node.get('pct', False)
        partition  = ast_node['partition']
        result_col = ast_node['result_col']
        rank_func  = 'PERCENT_RANK()' if pct else 'RANK()'
        order_dir  = 'ASC' if ascending else 'DESC'
        win = self._ddb_window(partition, f"{col} {order_dir}")
        return f"SELECT *, {rank_func} OVER ({win}) AS {result_col} FROM {from_alias}"

    def generate_shift_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        col        = ast_node['column']
        periods    = ast_node['periods']
        func       = ast_node['func']
        partition  = ast_node['partition']
        order_col  = ast_node['order_col']
        result_col = ast_node['result_col']
        sql_func   = 'LAG' if func == 'lag' else 'LEAD'
        win = self._ddb_window(partition, order_col)
        return f"SELECT *, {sql_func}({col}, {periods}) OVER ({win}) AS {result_col} FROM {from_alias}"

    def generate_cumulative_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        func       = ast_node['func']
        col        = ast_node['column']
        partition  = ast_node['partition']
        order_col  = ast_node['order_col']
        result_col = ast_node['result_col']
        _cum_map   = {'cumsum': 'SUM', 'cummean': 'AVG', 'cummin': 'MIN', 'cummax': 'MAX'}
        sql_func   = _cum_map.get(func, 'SUM')
        frame      = 'ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW'
        win = self._ddb_window(partition, order_col, frame)
        return f"SELECT *, {sql_func}({col}) OVER ({win}) AS {result_col} FROM {from_alias}"

    def generate_rolling_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        func       = ast_node['func']
        col        = ast_node['column']
        window     = ast_node['window']
        min_periods = ast_node.get('min_periods')
        partition  = ast_node['partition']
        order_col  = ast_node['order_col']
        result_col = ast_node['result_col']
        _roll_map  = {'sum': 'SUM', 'mean': 'AVG', 'avg': 'AVG',
                      'min': 'MIN', 'max': 'MAX', 'std': 'STDDEV'}
        sql_func   = _roll_map.get(func, func.upper())
        frame      = f'ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW'
        win = self._ddb_window(partition, order_col, frame)
        sql_expr = f"{sql_func}({col}) OVER ({win})"
        if min_periods is not None:
            count_expr = f"COUNT({col}) OVER ({win})"
            sql_expr = f"CASE WHEN {count_expr} >= {min_periods} THEN {sql_expr} ELSE NULL END"
        return f"SELECT *, {sql_expr} AS {result_col} FROM {from_alias}"

    def generate_fillna_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        per_col = ast_node.get('per_col', {})
        if per_col:
            parts = ', '.join(
                f"COALESCE({c}, {repr(str(v)) if isinstance(v, _LiteralStr) else v}) AS {c}"
                for c, v in per_col.items()
            )
            return f"SELECT *, {parts} FROM {from_alias}"
        return f"-- [skipped: fillna without column list requires runtime schema in SQL CTE mode]"

    def generate_intersect_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        others = ast_node['tables']
        parts = [f"SELECT * FROM {from_alias}"] + [f"SELECT * FROM {o}" for o in others]
        return ' INTERSECT '.join(parts)

    def generate_exclude_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        others = ast_node['tables']
        parts = [f"SELECT * FROM {from_alias}"] + [f"SELECT * FROM {o}" for o in others]
        return ' EXCEPT '.join(parts)

    def generate_dropna_sql(self, ast_node):
        t = ast_node['table_name']
        from_alias = self._sql_current(t)
        cols = ast_node.get('columns', [])
        if not cols:
            return f"-- [skipped: dropna without explicit column list requires runtime schema]"
        not_null = ' AND '.join(f"{c} IS NOT NULL" for c in cols)
        return f"SELECT * FROM {from_alias} WHERE {not_null}"

    # Python-only operations — skip in SQL CTE mode
    def generate_python_sql(self, ast_node):
        return f"-- [skipped: python block not supported in SQL CTE mode]"

    def generate_apply_sql(self, ast_node):
        return f"-- [skipped: apply not supported in SQL CTE mode]"

    def generate_show_sql(self, ast_node):
        return None  # silent skip

    def generate_plot_sql(self, ast_node):
        return None

    def generate_pivot_plot_sql(self, ast_node):
        return None

    generate_agg_plot_sql = generate_pivot_plot_sql

    def generate_gt_table_sql(self, ast_node):
        return None


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

    def generate_save_duckdb(self, ast_node):
        return self.generate_save_pandas(ast_node)

    def generate_load_all_pandas(self, ast_node):
        return (
            f"globals().update(_pivotal_pkg.load_all())\n"
            f"print(f\"Loaded {{len(_pivotal_pkg.load_all())}} table(s) from '{{_pivotal_pkg.name}}'\")"
        )

    def generate_load_all_polars(self, ast_node):
        return self.generate_load_all_pandas(ast_node)

    def generate_load_all_duckdb(self, ast_node):
        return self.generate_load_all_pandas(ast_node)

    def generate_load_package_table_pandas(self, ast_node):
        name = ast_node['table_name']
        return f"{name} = _pivotal_pkg.load_table({repr(name)})"

    def generate_load_package_table_polars(self, ast_node):
        return self.generate_load_package_table_pandas(ast_node)

    def generate_load_package_table_duckdb(self, ast_node):
        return self.generate_load_package_table_pandas(ast_node)


# ---------------------------------------------------------------------------
# Friendly parse-error helpers
# ---------------------------------------------------------------------------

# Human-readable names for Lark terminal symbols that appear in error messages.
_TERMINAL_NAMES = {
    'IDENTIFIER':    'a name (column or table)',
    'PYTHON_VAR':    'a Python variable reference (:name)',
    'STRING':        'a quoted string ("...")',
    'NUMBER':        'a number',
    'AGG_FUNCTION':  'an aggregation function (sum, mean, count, ...)',
    'MERGE_TYPE':    'a join type (left, right, inner, outer)',
    'SHOW_MODE':     'head, summary, shape, or columns',
    '_NL':           'end of line',
    'EQUAL':         "'='",
    'RIGHT_TABLE':   'a table name',
    'PATH':          'a file path',
    'LPAR':          "'('",
    'RPAR':          "')'",
    'COMMA':         "','",
}

# Statement keywords used for "did you mean?" suggestions on unknown words.
_STATEMENT_KEYWORDS = [
    'load', 'df', 'for', 'filter', 'assert', 'check', 'select', 'drop', 'distinct', 'assign',
    'cast', 'rename', 'sort', 'group', 'agg', 'merge', 'pivot',
    'unpivot', 'rank', 'lag', 'lead', 'cumsum', 'rolling', 'fillna',
    'dropna', 'concat', 'python', 'apply', 'show', 'plot', 'table', 'save',
]


def _describe_expected(expected: set) -> str:
    """Convert a set of Lark terminal names into a readable phrase."""
    meaningful = {t for t in expected if not t.startswith('__') and t != '_NL'}
    if not meaningful:
        return "end of line"
    readable = sorted(_TERMINAL_NAMES.get(t, t.lower()) for t in meaningful)
    if len(readable) == 1:
        return readable[0]
    if len(readable) == 2:
        return f"{readable[0]} or {readable[1]}"
    return ", ".join(readable[:-1]) + f", or {readable[-1]}"


def _source_line(source: str, line: int) -> str:
    """Return the (1-based) line from source, or empty string."""
    lines = source.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1]
    return ""


def _active_keyword(source: str, line: int) -> str:
    """Return the first word on the given source line (the active statement keyword)."""
    ln = _source_line(source, line).strip()
    return ln.split()[0] if ln.split() else ""


def _friendly_parse_error(exc: Exception, source: str) -> PivotalError:
    """Convert a Lark parse exception into a PivotalError with a user-friendly message."""

    # VisitError wraps a transformer exception — unwrap and re-classify.
    if isinstance(exc, VisitError):
        inner = exc.orig_exc
        if isinstance(inner, ValueError):
            # e.g. keyword-collision raised during transformation
            return PivotalError(
                message=str(inner),
                error_type="Error",
            )
        return PivotalError(
            message=f"Internal error while processing statement: {inner}",
            error_type="Error",
        )

    # UnexpectedEOF — input ended mid-grammar.
    if isinstance(exc, UnexpectedEOF):
        return PivotalError(
            message="Unexpected end of input — is a statement incomplete?",
            error_type="Syntax Error",
            suggestion=f"Expected {_describe_expected(set(exc.expected))}",
        )

    # UnexpectedCharacters — lexer hit an unrecognised character.
    if isinstance(exc, UnexpectedCharacters):
        ln = getattr(exc, 'line', None)
        col = getattr(exc, 'column', None)
        char = _source_line(source, ln)[col - 1] if ln and col else '?'
        return PivotalError(
            message=f"Unrecognised character '{char}'",
            error_type="Syntax Error",
            line=ln,
            column=col,
            source_line=_source_line(source, ln) if ln else None,
        )

    # UnexpectedToken — the most common case; discriminate on token type.
    if isinstance(exc, UnexpectedToken):
        ln = getattr(exc, 'line', None)
        col = getattr(exc, 'column', None)
        tok = exc.token
        tok_type = getattr(tok, 'type', '')
        tok_str = str(tok).strip()
        expected = set(exc.expected)
        src_ln = _source_line(source, ln) if ln else None

        # Token is a newline → statement cut short, or extra token at end of line.
        if tok_type == '_NL':
            keyword = _active_keyword(source, ln) if ln else ''

            # If expected only contains EQUAL, the parser tried to treat the
            # first word on the line as an assign target.  Two cases:
            #   1. "selects revenue" — first word is a misspelled keyword
            #   2. "group by country_id poo" — first word is valid, last word is extra
            if expected == {'EQUAL'} or expected == {'EQUAL', '_NL'}:
                words = (src_ln or '').split()
                first_word = words[0].lower() if words else ''
                # Case 1: first word looks like a misspelled statement keyword
                # (but is not already a valid keyword — those go to case 2)
                if first_word and first_word not in _STATEMENT_KEYWORDS:
                    kw_suggestion = _make_suggestion(first_word, _STATEMENT_KEYWORDS)
                    if kw_suggestion:
                        msg = f"Unknown statement '{words[0]}'"
                        return PivotalError(
                            message=msg,
                            error_type="Syntax Error",
                            line=ln,
                            column=1,
                            source_line=src_ln,
                            suggestion=kw_suggestion,
                        )
                # Case 2: valid keyword on line but extra unrecognised word at end
                if keyword in _STATEMENT_KEYWORDS and len(words) > 1:
                    msg = f"Unexpected extra text '{words[-1]}' at end of '{keyword}' line"
                    return PivotalError(
                        message=msg,
                        error_type="Syntax Error",
                        line=ln,
                        column=col,
                        source_line=src_ln,
                    )

            desc = _describe_expected(expected)
            if keyword:
                msg = f"Incomplete '{keyword}' statement - expected {desc}"
            else:
                msg = f"Incomplete statement - expected {desc}"
            return PivotalError(
                message=msg,
                error_type="Syntax Error",
                line=ln,
                column=col,
                source_line=src_ln,
            )

        # Token is CASE_DEFAULT_EXPR → unrecognised text (garbage / bad chars).
        if tok_type == 'CASE_DEFAULT_EXPR':
            display = tok_str[:30] + ('...' if len(tok_str) > 30 else '')
            return PivotalError(
                message=f"Unexpected text '{display}' - check for typos or unsupported syntax",
                error_type="Syntax Error",
                line=ln,
                column=col,
                source_line=src_ln,
            )

        # Token looks like an unknown keyword/identifier.
        if tok_str:
            desc = _describe_expected(expected)
            msg = f"Unexpected '{tok_str}' - expected {desc}"

            # Check token_history for a preceding ASSIGN_TARGET that looks like
            # a misspelled keyword (e.g. "sekect country_id" → token_history has
            # Token('ASSIGN_TARGET', 'sekect')).
            suggestion = None
            history = getattr(exc, 'token_history', None)
            if history:
                for hist_tok in history:
                    if getattr(hist_tok, 'type', '') == 'ASSIGN_TARGET':
                        suggestion = _make_suggestion(str(hist_tok).lower(), _STATEMENT_KEYWORDS)
                        if suggestion:
                            break
            # Fall back to suggesting a correction for the token itself.
            if not suggestion:
                suggestion = _make_suggestion(tok_str.lower(), _STATEMENT_KEYWORDS)

            return PivotalError(
                message=msg,
                error_type="Syntax Error",
                line=ln,
                column=col,
                source_line=src_ln,
                suggestion=suggestion,
            )

    # Fallback for any other exception type.
    return PivotalError(
        message=str(exc),
        error_type="Error",
    )


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
        self.autocomplete_snapshot = None
        self._pivotal_registry_name = '_pivotal_values'

    def update_autocomplete_info(self, globals_dict=None):
        """Update the autocomplete JSON file with current table/value information."""
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
        
        value_info = self._build_autocomplete_value_info(self._get_pivotal_values(globals_dict))
        payload = {
            'tables': table_info,
            'values': value_info,
            'timestamp': pd.Timestamp.now().isoformat(),
            'current_table': globals_dict.get('__table_name__')
        }

        # Check if autocomplete metadata has actually changed to avoid unnecessary file I/O
        snapshot = {k: v for k, v in payload.items() if k != 'timestamp'}
        if snapshot == self.autocomplete_snapshot:
            return

        # Update our internal tracking
        self.table_info = table_info
        self.autocomplete_snapshot = snapshot
        
        # Write to file for VS Code extension only when there are changes
        try:
            with open(self.autocomplete_file, 'w') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not update autocomplete file: {e}")
    
    def get_table_columns(self, table_name=None):
        """Get columns for a specific table or all tables"""
        if table_name:
            return self.table_info.get(table_name, {}).get('columns', [])
        return self.table_info

    def _summarize_pivotal_value(self, value, depth=0, *, include_non_identifier_keys=False,
                                 include_list_children=False, max_depth=5, max_children=25):
        """Serialize a native Pivotal value for editor tooling."""
        def preview_scalar(obj):
            if obj is None:
                return 'None'
            if isinstance(obj, str):
                text = repr(obj)
            elif isinstance(obj, bool):
                text = 'True' if obj else 'False'
            else:
                text = repr(obj)
            if len(text) > 40:
                text = text[:37] + '...'
            return text

        if depth >= max_depth:
            return {
                'kind': 'scalar',
                'value_type': type(value).__name__,
                'preview': preview_scalar(value),
            }

        if isinstance(value, dict):
            children = {}
            count = 0
            for key, child in value.items():
                key_text = str(key)
                if (not include_non_identifier_keys
                        and not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', key_text)):
                    continue
                children[key_text] = self._summarize_pivotal_value(
                    child,
                    depth + 1,
                    include_non_identifier_keys=include_non_identifier_keys,
                    include_list_children=include_list_children,
                    max_depth=max_depth,
                    max_children=max_children,
                )
                count += 1
                if count >= max_children:
                    break
            return {
                'kind': 'dict',
                'children': children,
                'size': len(value),
            }

        if isinstance(value, (list, tuple)):
            summary = {
                'kind': 'list',
                'length': len(value),
                'item_type': type(value[0]).__name__ if value else None,
            }
            if include_list_children:
                children = {}
                for idx, child in enumerate(value[:max_children]):
                    children[f'[{idx}]'] = self._summarize_pivotal_value(
                        child,
                        depth + 1,
                        include_non_identifier_keys=include_non_identifier_keys,
                        include_list_children=include_list_children,
                        max_depth=max_depth,
                        max_children=max_children,
                    )
                summary['children'] = children
            return summary

        return {
            'kind': 'scalar',
            'value_type': type(value).__name__,
            'preview': preview_scalar(value),
        }

    def _build_autocomplete_value_info(self, values):
        """Serialize named Pivotal values for editor autocomplete."""
        return {
            name: self._summarize_pivotal_value(value)
            for name, value in values.items()
        }

    def build_explorer_value_info(self, values):
        """Serialize named Pivotal values for the object explorer."""
        return {
            name: self._summarize_pivotal_value(
                value,
                include_non_identifier_keys=True,
                include_list_children=True,
            )
            for name, value in values.items()
        }

    def _get_pivotal_values(self, namespace=None):
        if namespace is None:
            return {}
        return namespace.setdefault(self._pivotal_registry_name, {})

    def _runtime_var_expr(self, var_node):
        expr = var_node['name']
        for part in var_node.get('path', []):
            if isinstance(part, int):
                expr += f'[{part}]'
            else:
                expr += f'[{part!r}]'
        return expr

    def _resolve_runtime_var(self, var_node, namespace):
        if namespace is None:
            raise ValueError(f"Python variable ':{var_node['name']}' requires a runtime namespace.")
        name = var_node['name']
        if name not in namespace:
            raise ValueError(f"Python variable ':{name}' was not found.")
        value = namespace[name]
        for part in var_node.get('path', []):
            try:
                value = value[part]
            except Exception as exc:
                raise ValueError(
                    f"Could not resolve Python variable reference ':{self._runtime_var_expr(var_node)}': {exc}"
                )
        return value

    _PYTHON_INDEXED_REF_RE = re.compile(
        r':[a-zA-Z_][a-zA-Z0-9_]*(?:\[(?:-?\d+|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')\])+'
    )

    def _parse_runtime_ref(self, ref):
        name_match = re.match(r':([a-zA-Z_][a-zA-Z0-9_]*)', ref)
        if not name_match:
            raise ValueError(f"Invalid Python variable reference '{ref}'.")
        name = name_match.group(1)
        rest = ref[name_match.end():]
        path = []
        while rest:
            if not rest.startswith('['):
                raise ValueError(f"Invalid Python variable reference '{ref}'.")
            end = rest.find(']')
            if end == -1:
                raise ValueError(f"Invalid Python variable reference '{ref}'.")
            token = rest[1:end].strip()
            if token.startswith(("'", '"')):
                path.append(ast.literal_eval(token))
            else:
                path.append(int(token))
            rest = rest[end + 1:]
        return {'type': 'var', 'name': name, 'path': path}

    def _rewrite_runtime_refs(self, code, namespace=None):
        if namespace is None:
            return code, []

        result = []
        created = []
        i = 0
        quote = None
        while i < len(code):
            ch = code[i]
            if quote:
                result.append(ch)
                if ch == '\\' and i + 1 < len(code):
                    i += 1
                    result.append(code[i])
                elif ch == quote:
                    quote = None
                i += 1
                continue

            if ch in ('"', "'"):
                quote = ch
                result.append(ch)
                i += 1
                continue

            match = self._PYTHON_INDEXED_REF_RE.match(code, i)
            if match:
                ref = match.group(0)
                temp_name = f"pvt_runtime_ref_{len(created)}"
                namespace[temp_name] = self._resolve_runtime_var(self._parse_runtime_ref(ref), namespace)
                created.append(temp_name)
                result.append(f":{temp_name}")
                i = match.end()
                continue

            result.append(ch)
            i += 1

        return ''.join(result), created

    def _store_pivotal_values(self, namespace, values):
        if namespace is None:
            return
        registry = self._get_pivotal_values(namespace)
        for name, value in values.items():
            namespace[name] = copy.deepcopy(value)
            registry[name] = copy.deepcopy(value)
        
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
        self._validate_inline_python_statements(code)

        # Allow compact aggregation syntax such as:
        #     agg mean colA, colB
        # by expanding it to:
        #     agg mean colA, mean colB
        # Existing multi-function syntax like "agg sum x, max y" is left alone.
        agg_funcs = 'mean|avg|sum|min|max|count|std|median|var|nunique|first|last|quantile|percentile'

        def _split_agg_parts(body):
            parts = []
            start = 0
            depth = 0
            quote = None
            i = 0
            while i < len(body):
                ch = body[i]
                if quote:
                    if ch == '\\':
                        i += 2
                        continue
                    if ch == quote:
                        quote = None
                    i += 1
                    continue
                if ch in ("'", '"'):
                    quote = ch
                elif ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth = max(0, depth - 1)
                elif ch == ',' and depth == 0:
                    parts.append(body[start:i].strip())
                    start = i + 1
                i += 1
            parts.append(body[start:].strip())
            return parts

        def _expand_agg_line(m):
            indent, body = m.group(1), m.group(2)
            parts = _split_agg_parts(body)
            expanded = []
            current_func = None
            for part in parts:
                if not part:
                    continue
                func_match = re.match(rf'^({agg_funcs})\b', part, flags=re.IGNORECASE)
                special_match = re.match(r'^(wmean|wavg)\b', part, flags=re.IGNORECASE)
                custom_match = re.match(r'^:[a-zA-Z_][a-zA-Z0-9_]*\b', part)
                if func_match:
                    current_func = func_match.group(1)
                    expanded.append(part)
                elif special_match or custom_match:
                    current_func = None
                    expanded.append(part)
                elif current_func:
                    expanded.append(f'{current_func} {part}')
                else:
                    expanded.append(part)
            return f"{indent}agg {', '.join(expanded)}"

        code = re.sub(r'^([ \t]*)agg[ \t]+([^\n]+)$', _expand_agg_line, code, flags=re.MULTILINE)

        # Strip leading and trailing whitespace
        code = code.strip()

        # Ensure the file ends with a newline if it's not empty
        if code and not code.endswith('\n'):
            code += '\n'

        return code

    def _validate_inline_python_statements(self, code):
        """Catch incomplete single-line Python before backend codegen."""
        for line in code.splitlines():
            match = re.match(r'^[ \t]*python[ \t]+(.+?)\s*$', line)
            if not match:
                continue
            source = match.group(1)
            if source.startswith('__PYBLOCK_') and source.endswith('__'):
                continue
            try:
                ast.parse(source, mode='exec')
            except SyntaxError as exc:
                raise ValueError(
                    "Invalid inline Python statement. Use a python/end block for "
                    f"multi-line Python such as function definitions: {exc.msg}"
                ) from exc

    def _replace_loop_identifier(self, text, loop_var, column):
        """Replace a bare loop variable in expression text, preserving strings."""
        if not isinstance(text, str) or not text:
            return text

        result = []
        i = 0
        quote = None
        ident_re = re.compile(r'[a-zA-Z][a-zA-Z0-9_]*')

        while i < len(text):
            ch = text[i]
            if quote:
                result.append(ch)
                if ch == '\\' and i + 1 < len(text):
                    i += 1
                    result.append(text[i])
                elif ch == quote:
                    quote = None
                i += 1
                continue

            if ch in ('"', "'"):
                quote = ch
                result.append(ch)
                i += 1
                continue

            match = ident_re.match(text, i)
            if match:
                token = match.group(0)
                prev = text[i - 1] if i > 0 else ''
                result.append(column if token == loop_var and prev != ':' else token)
                i = match.end()
                continue

            result.append(ch)
            i += 1

        return ''.join(result)

    def _replace_loop_column_ref(self, value, loop_var, column):
        return column if isinstance(value, str) and value == loop_var else value

    def _replace_bound_identifier(self, text, bindings):
        """Replace bound identifiers inside expression text, preserving strings."""
        if not isinstance(text, str) or not text:
            return text

        result = []
        i = 0
        quote = None
        ident_re = re.compile(r'[a-zA-Z][a-zA-Z0-9_]*')

        while i < len(text):
            ch = text[i]
            if quote:
                result.append(ch)
                if ch == '\\' and i + 1 < len(text):
                    i += 1
                    result.append(text[i])
                elif ch == quote:
                    quote = None
                i += 1
                continue

            if ch in ('"', "'"):
                quote = ch
                result.append(ch)
                i += 1
                continue

            match = ident_re.match(text, i)
            if match:
                token = match.group(0)
                prev = text[i - 1] if i > 0 else ''
                has_bound = token in bindings
                bound = bindings.get(token)
                if has_bound and prev != ':' and not isinstance(bound, list):
                    if isinstance(bound, _LiteralStr):
                        result.append(repr(str(bound)))
                    else:
                        result.append(str(bound))
                else:
                    result.append(token)
                i = match.end()
                continue

            result.append(ch)
            i += 1

        return ''.join(result)

    _COMPILE_REF_RE = re.compile(
        r'[a-zA-Z_][a-zA-Z0-9_]*(?:(?:\.[a-zA-Z_][a-zA-Z0-9_]*)|(?:\[-?\d+\]))+'
    )

    def _parse_compile_ref_path(self, path):
        parts = []
        i = 0
        while i < len(path):
            match = re.match(r'[a-zA-Z_][a-zA-Z0-9_]*', path[i:])
            if not match:
                raise ValueError(f"Invalid compile-time lookup '{path}'.")
            parts.append(match.group(0))
            i += len(match.group(0))
            while i < len(path):
                if path[i] == '.':
                    i += 1
                    break
                if path[i] == '[':
                    end = path.find(']', i)
                    if end == -1:
                        raise ValueError(f"Invalid compile-time lookup '{path}'.")
                    idx_text = path[i + 1:end].strip()
                    try:
                        parts.append(int(idx_text))
                    except ValueError:
                        raise ValueError(
                            f"List indexes in compile-time lookup '{path}' must be integers."
                        )
                    i = end + 1
                    continue
                raise ValueError(f"Invalid compile-time lookup '{path}'.")
            else:
                break
        return parts

    def _resolve_compile_ref(self, path, values):
        parts = self._parse_compile_ref_path(path)
        root = parts[0]
        if root not in values:
            raise ValueError(f"Compile-time value '{root}' was not defined.")
        current = copy.deepcopy(values[root])
        for part in parts[1:]:
            if isinstance(part, int):
                if not isinstance(current, list):
                    raise ValueError(f"Cannot index non-list value in compile-time lookup '{path}'.")
                try:
                    current = current[part]
                except IndexError:
                    raise ValueError(f"List index {part} is out of range in compile-time lookup '{path}'.")
            else:
                if not isinstance(current, dict):
                    raise ValueError(f"Cannot access key '{part}' on non-dict value in compile-time lookup '{path}'.")
                if part not in current:
                    raise ValueError(f"Key '{part}' was not found in compile-time lookup '{path}'.")
                current = current[part]
        return copy.deepcopy(current)

    def _compile_literal_for_expression(self, value):
        if isinstance(value, _LiteralStr):
            return repr(str(value))
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return repr([str(v) if isinstance(v, _LiteralStr) else v for v in value])
        if isinstance(value, dict):
            return repr(value)
        return repr(value)

    def _replace_compile_identifiers_in_text(self, text, values):
        """Replace bare scalar/config identifiers in expression text, preserving strings."""
        if not isinstance(text, str) or not text:
            return text

        result = []
        i = 0
        quote = None
        ident_re = re.compile(r'[a-zA-Z][a-zA-Z0-9_]*')

        while i < len(text):
            ch = text[i]
            if quote:
                result.append(ch)
                if ch == '\\' and i + 1 < len(text):
                    i += 1
                    result.append(text[i])
                elif ch == quote:
                    quote = None
                i += 1
                continue

            if ch in ('"', "'"):
                quote = ch
                result.append(ch)
                i += 1
                continue

            match = ident_re.match(text, i)
            if match:
                token = match.group(0)
                prev = text[i - 1] if i > 0 else ''
                value = values.get(token)
                if (
                    token in values
                    and prev != ':'
                    and isinstance(value, (int, float, bool, str, _LiteralStr))
                ):
                    result.append(self._compile_literal_for_expression(value))
                else:
                    result.append(token)
                i = match.end()
                continue

            result.append(ch)
            i += 1

        return ''.join(result)

    def _replace_compile_refs_in_text(self, text, values):
        if not isinstance(text, str) or not text:
            return text

        result = []
        i = 0
        quote = None
        while i < len(text):
            ch = text[i]
            if quote:
                result.append(ch)
                if ch == '\\' and i + 1 < len(text):
                    i += 1
                    result.append(text[i])
                elif ch == quote:
                    quote = None
                i += 1
                continue

            if ch in ('"', "'"):
                quote = ch
                result.append(ch)
                i += 1
                continue

            match = self._COMPILE_REF_RE.match(text, i)
            if match and (i == 0 or text[i - 1] != ':'):
                ref = match.group(0)
                value = self._resolve_compile_ref(ref, values)
                result.append(self._compile_literal_for_expression(value))
                i = match.end()
                continue

            result.append(ch)
            i += 1

        return ''.join(result)

    def _normalise_compile_value(self, value, values, namespace=None):
        if isinstance(value, dict):
            node_type = value.get('type')
            if node_type == 'var' and 'name' in value:
                return copy.deepcopy(self._resolve_runtime_var(value, namespace))
            if node_type == 'compile_ref' and 'path' in value:
                return self._resolve_compile_ref(value['path'], values)
            if node_type == 'list_ref' and 'name' in value:
                return self._resolve_compile_value(value, values)
            return {k: self._normalise_compile_value(v, values, namespace) for k, v in value.items()}
        if isinstance(value, list):
            return [self._normalise_compile_value(item, values, namespace) for item in value]
        if isinstance(value, str) and not isinstance(value, _LiteralStr) and value in values:
            return copy.deepcopy(values[value])
        return copy.deepcopy(value)

    def _mark_loaded_strings_literal(self, value):
        if isinstance(value, dict):
            return {str(k): self._mark_loaded_strings_literal(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._mark_loaded_strings_literal(item) for item in value]
        if isinstance(value, str):
            return _LiteralStr(value)
        return value

    def _load_compile_dict(self, source):
        path = Path(source)
        suffix = path.suffix.lower()
        try:
            text = path.read_text(encoding='utf-8')
        except OSError as exc:
            raise ValueError(f"Could not read dict file '{source}': {exc}")

        if suffix == '.json':
            data = json.loads(text)
        elif suffix in ('.yaml', '.yml'):
            try:
                import yaml
            except ImportError:
                raise ValueError("Loading YAML dict files requires PyYAML to be installed.")
            data = yaml.safe_load(text)
        else:
            raise ValueError("Dict files must use .json, .yaml, or .yml extension.")

        if not isinstance(data, dict):
            raise ValueError(f"Dict file '{source}' must contain an object/mapping at the top level.")
        return self._mark_loaded_strings_literal(data)

    def _resolve_compile_value(self, value, lists):
        if isinstance(value, dict) and value.get('type') == 'compile_ref' and 'path' in value:
            return self._resolve_compile_ref(value['path'], lists)
        if isinstance(value, dict) and value.get('type') == 'list_ref' and 'name' in value:
            name = value.get('name')
            if name not in lists:
                raise ValueError(f"Compile-time value '{name}' was not defined.")
            return copy.deepcopy(lists[name])
        if isinstance(value, str) and value in lists:
            return copy.deepcopy(lists[value])
        return value

    def _substitute_compile_values(self, value, bindings, lists, field=None):
        if isinstance(value, dict):
            if value.get('type') == 'var' and 'name' in value:
                return copy.deepcopy(value)
            if value.get('type') == 'compile_ref' and 'path' in value:
                return self._resolve_compile_ref(value['path'], lists)
            if value.get('type') == 'list_ref' and 'name' in value:
                return self._resolve_compile_value(value, lists)
            result = {}
            for key, child in value.items():
                if key == 'type':
                    result[key] = child
                elif key in ('expression', 'query_str') and isinstance(child, str):
                    exact = bindings.get(child)
                    if exact is not None:
                        result[key] = copy.deepcopy(exact)
                    else:
                        replaced = self._replace_bound_identifier(child, bindings)
                        replaced = self._replace_compile_refs_in_text(replaced, lists)
                        result[key] = self._replace_compile_identifiers_in_text(replaced, lists)
                else:
                    result[key] = self._substitute_compile_values(child, bindings, lists, field=key)
            return result

        if isinstance(value, list):
            expanded = []
            for item in value:
                resolved = self._substitute_compile_values(item, bindings, lists, field=field)
                if isinstance(resolved, list):
                    expanded.extend(resolved)
                else:
                    expanded.append(resolved)
            return expanded

        if isinstance(value, str):
            if value in bindings:
                return copy.deepcopy(bindings[value])
            return self._resolve_compile_value(value, lists)

        return copy.deepcopy(value)

    def _collect_compile_defs(self, ast_list, namespace=None):
        functions = {}
        lists = copy.deepcopy(self._get_pivotal_values(namespace))
        executable = []
        defined_values = {}
        last_function = None

        for node in ast_list:
            if not isinstance(node, dict):
                executable.append(node)
                continue
            node_type = node.get('type')
            if node_type == 'list_def':
                resolved = self._normalise_compile_value(node.get('items', []), lists, namespace)
                if len(resolved) == 1 and isinstance(resolved[0], list):
                    resolved = resolved[0]
                lists[node['name']] = resolved
                defined_values[node['name']] = copy.deepcopy(resolved)
            elif node_type == 'scalar_def':
                resolved = self._normalise_compile_value(node.get('value'), lists, namespace)
                lists[node['name']] = resolved
                defined_values[node['name']] = copy.deepcopy(resolved)
            elif node_type == 'dict_def':
                if node.get('source') is not None:
                    resolved = self._normalise_compile_value(
                        self._load_compile_dict(node['source']),
                        lists,
                        namespace,
                    )
                else:
                    raw_value = node.get('value', node.get('items', {}))
                    resolved = self._normalise_compile_value(raw_value, lists, namespace)
                if not isinstance(resolved, dict):
                    raise ValueError(f"Dict '{node['name']}' must resolve to a dictionary.")
                lists[node['name']] = resolved
                defined_values[node['name']] = copy.deepcopy(resolved)
            elif node_type == 'function_def':
                name = node['name']
                if name in PIVOTAL_KEYWORDS:
                    raise ValueError(f"'{name}' is a Pivotal reserved keyword and cannot be used as a function name.")
                if name in functions:
                    raise ValueError(f"Function '{name}' is already defined.")
                functions[name] = node
                last_function = node
            elif node_type == 'return':
                if last_function is not None:
                    last_function.setdefault('body', []).append(node)
                continue
            else:
                last_function = None
                executable.append(node)

        return executable, functions, lists, defined_values

    def _bind_function_args(self, fn, call, lists):
        params = fn.get('params', [])
        args = call.get('args', [])
        positional = []
        keywords = {}

        for arg in args:
            if isinstance(arg, dict) and arg.get('kind') == 'kwarg':
                name = arg['name']
                if name in keywords:
                    raise ValueError(f"Function '{fn['name']}' received keyword argument '{name}' more than once.")
                keywords[name] = self._resolve_compile_value(arg['value'], lists)
            else:
                positional.append(self._resolve_compile_value(arg, lists))

        if len(positional) > len(params):
            raise ValueError(f"Function '{fn['name']}' expected at most {len(params)} arguments, got {len(positional)}.")

        bindings = {}
        for param, arg in zip(params, positional):
            bindings[param['name']] = arg

        param_names = {param['name'] for param in params}
        for name, value in keywords.items():
            if name not in param_names:
                raise ValueError(f"Function '{fn['name']}' has no parameter named '{name}'.")
            if name in bindings:
                raise ValueError(f"Function '{fn['name']}' received multiple values for parameter '{name}'.")
            bindings[name] = value

        for param in params:
            name = param['name']
            if name not in bindings:
                if param.get('has_default'):
                    bindings[name] = self._resolve_compile_value(param.get('default'), lists)
                else:
                    raise ValueError(f"Function '{fn['name']}' missing required argument '{name}'.")

        return bindings

    def _expand_function_calls(self, ast_list, functions, lists, stack=None):
        stack = stack or []
        expanded = []

        for node in ast_list:
            if not isinstance(node, dict):
                expanded.append(node)
                continue

            node_type = node.get('type')
            if node_type in ('function_def', 'list_def', 'return'):
                continue

            if node_type != 'function_call':
                expanded.append(self._substitute_compile_values(node, {}, lists))
                continue

            name = node.get('name')
            if name not in functions:
                raise ValueError(f"Function '{name}' was not defined.")
            if name in stack:
                chain = ' -> '.join(stack + [name])
                raise ValueError(f"Recursive function calls are not supported: {chain}.")

            fn = functions[name]
            bindings = self._bind_function_args(fn, node, lists)
            body = []
            for child in fn.get('body', []):
                if isinstance(child, dict) and child.get('type') == 'return':
                    continue
                if isinstance(child, dict) and child.get('type') == 'function_def':
                    raise ValueError("Nested function definitions are not supported.")
                body.append(self._substitute_compile_values(child, bindings, lists))
            expanded.extend(self._expand_function_calls(body, functions, lists, stack + [name]))

        return expanded

    def _expand_compile_time_features(self, ast_list, namespace=None):
        executable, functions, lists, _ = self._collect_compile_defs(ast_list, namespace)
        return self._expand_function_calls(executable, functions, lists)

    def _resolve_loop_target(self, target, loop_var, column):
        if isinstance(target, dict) and target.get('type') == 'target_expr':
            parts = []
            for part in target.get('parts', []):
                if part.get('type') == 'identifier' and part.get('value') == loop_var:
                    parts.append(column)
                else:
                    parts.append(str(part.get('value', '')))
            resolved = ''.join(parts)
        else:
            resolved = self._replace_loop_column_ref(target, loop_var, column)

        if not isinstance(resolved, str) or not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]*', resolved):
            raise ValueError(
                f"Loop assignment target '{resolved}' is not a valid Pivotal column name."
            )
        if resolved.lower() in PIVOTAL_KEYWORDS:
            raise ValueError(
                f"'{resolved}' is a Pivotal reserved keyword and cannot be used as a column name."
            )
        return resolved

    def _replace_loop_columns(self, values, loop_var, column):
        return [
            self._replace_loop_column_ref(value, loop_var, column)
            for value in (values or [])
        ]

    def _expand_for_assign(self, node, loop_var, column):
        expanded = copy.deepcopy(node)
        expanded['target'] = self._resolve_loop_target(expanded.get('target'), loop_var, column)

        for key in ('expression', 'query_str', 'default_expr'):
            if key in expanded:
                expanded[key] = self._replace_loop_identifier(expanded.get(key), loop_var, column)

        expanded['by_cols'] = self._replace_loop_columns(expanded.get('by_cols', []), loop_var, column)

        for condition in expanded.get('conditions') or []:
            if isinstance(condition, dict):
                condition['column'] = self._replace_loop_column_ref(
                    condition.get('column'), loop_var, column
                )

        for case in expanded.get('cases') or []:
            if not isinstance(case, dict):
                continue
            if 'expression' in case:
                case['expression'] = self._replace_loop_identifier(
                    case.get('expression'), loop_var, column
                )
            if 'query_str' in case:
                case['query_str'] = self._replace_loop_identifier(
                    case.get('query_str'), loop_var, column
                )
            for condition in case.get('conditions') or []:
                if isinstance(condition, dict):
                    condition['column'] = self._replace_loop_column_ref(
                        condition.get('column'), loop_var, column
                    )

        return expanded

    def _expand_for_generic_node(self, node, loop_var, column):
        expanded = copy.deepcopy(node)
        node_type = expanded.get('type')

        if node_type == 'assign':
            return self._expand_for_assign(expanded, loop_var, column)

        if node_type in ('cast', 'drop', 'dropna', 'round'):
            expanded['columns'] = self._replace_loop_columns(expanded.get('columns', []), loop_var, column)
            if expanded.get('alias'):
                expanded['alias'] = self._replace_loop_column_ref(expanded.get('alias'), loop_var, column)
            return expanded

        if node_type == 'fillna':
            per_col = {}
            for key, value in expanded.get('per_col', {}).items():
                per_col[self._replace_loop_column_ref(key, loop_var, column)] = value
            expanded['per_col'] = per_col
            return expanded

        if node_type in ('rank', 'shift', 'cumulative', 'rolling'):
            for key in ('column', 'order_col', 'result_col'):
                if key in expanded:
                    expanded[key] = self._replace_loop_column_ref(expanded.get(key), loop_var, column)
            expanded['partition'] = self._replace_loop_columns(expanded.get('partition', []), loop_var, column)
            return expanded

        raise ValueError(f"Column for-loops do not currently support '{node_type}' statements.")

    def _resolve_loop_columns(self, source, namespace=None):
        if isinstance(source, dict) and source.get('type') == 'var':
            if namespace is None:
                return None
            var_name = source.get('name')
            if var_name not in namespace:
                raise ValueError(f"Loop column list variable ':{var_name}' was not found.")
            value = namespace[var_name]
            if isinstance(value, str) or not hasattr(value, '__iter__'):
                raise ValueError(f"Loop column list variable ':{var_name}' must be a list or iterable of column names.")
            return [str(item) for item in value]
        return list(source or [])

    def _expand_for_loops(self, ast_list, namespace=None):
        """Lower column for-loops into repeated AST nodes when column names are known."""
        expanded = []
        for node in ast_list:
            if not isinstance(node, dict) or node.get('type') != 'for':
                expanded.append(node)
                continue

            loop_var = node.get('var')
            body = node.get('body', [])
            columns = self._resolve_loop_columns(node.get('columns'), namespace)
            if columns is None:
                expanded.append(node)
                continue

            for column in columns:
                for child in body:
                    if not isinstance(child, dict):
                        raise ValueError("Column for-loops only support Pivotal statements.")
                    expanded.append(self._expand_for_generic_node(child, loop_var, column))

        return expanded

    def parse(self, code, namespace=None):
        """Parse DSL code and return AST or {'error': PivotalError}."""
        try:
            processed_code = self.preprocess_code(code)
            processed_code, _ = self._rewrite_runtime_refs(processed_code, namespace)
            result = self.parser.parse(processed_code)
            result = self._expand_compile_time_features(result, namespace)
            return self._expand_for_loops(result, namespace)
        except ValueError as e:
            # Keyword-collision errors raised by the transformer — wrap cleanly.
            return {'error': PivotalError(message=str(e), error_type="Error")}
        except (UnexpectedToken, UnexpectedCharacters, UnexpectedEOF, VisitError) as e:
            return {'error': _friendly_parse_error(e, code)}
        except Exception as e:
            return {'error': PivotalError(message=str(e), error_type="Error")}
    
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

    def parse_definitions(self, code, namespace=None):
        """Parse compile-time list/function definitions without expanding them."""
        try:
            processed_code = self.preprocess_code(code)
            processed_code, _ = self._rewrite_runtime_refs(processed_code, namespace)
            result = self.parser.parse(processed_code)
            executable, functions, lists, defined_values = self._collect_compile_defs(result, namespace)
            return {'executable': executable, 'functions': functions, 'lists': lists, 'defined_values': defined_values}
        except ValueError as e:
            return {'error': PivotalError(message=str(e), error_type="Error")}
        except (UnexpectedToken, UnexpectedCharacters, UnexpectedEOF, VisitError) as e:
            return {'error': _friendly_parse_error(e, code)}
        except Exception as e:
            return {'error': PivotalError(message=str(e), error_type="Error")}

    def generate_code(self, ast_list, backend="pandas"):
        """Generate code for a list of AST nodes"""
        if backend and backend != self.code_generator.backend:
            # Create new generator if backend changed
            self.code_generator = CodeGenerator(backend)

        if any(isinstance(node, dict) and node.get('type') == 'for' for node in ast_list):
            if backend == 'sql':
                raise ValueError(
                    "Python variable column for-loops are not supported for the SQL backend. "
                    "Use an explicit column list instead."
                )
            raise ValueError(
                "Python variable column for-loops require runtime execution with a namespace "
                "and cannot be generated or exported before the list is resolved."
            )

        if backend == 'sql':
            return self._generate_code_sql(ast_list)

        python_code = []
        if backend == 'duckdb':
            python_code.append(self.code_generator.duckdb_preamble())
        elif backend == 'polars':
            python_code.append(self.code_generator.polars_preamble())

        for ast_node in ast_list:
            code = self.code_generator.generate(ast_node)
            python_code.append(code)

        return python_code

    def _generate_code_sql(self, ast_list):
        """Assemble a SQL CTE chain from ast_list. Returns [sql_string]."""
        cg = self.code_generator
        # Per-call state initialised on the generator object
        cg._sql_state   = {}   # table_name -> current CTE alias
        cg._sql_counter = [0]  # mutable int wrapped in list

        cte_pairs = []   # [(alias, select_body), ...]
        comments  = []   # comment lines emitted above the WITH block

        for ast_node in ast_list:
            if not isinstance(ast_node, dict):
                continue  # skip error strings or non-dict entries
            stmt_type  = ast_node.get('type', '')
            table_name = ast_node.get('table_name', '_unknown')
            method     = f"generate_{stmt_type}_sql"

            if hasattr(cg, method):
                result = getattr(cg, method)(ast_node)
            else:
                result = f"-- [skipped: no SQL generator for '{stmt_type}']"

            if result is None:
                continue
            if isinstance(result, str) and result.startswith('--'):
                comments.append(result)
                continue
            # result is a SELECT body — wrap it in a CTE alias
            idx   = cg._sql_counter[0]
            cg._sql_counter[0] += 1
            alias = f"_cte_{idx}_{table_name}"
            cte_pairs.append((alias, result))
            cg._sql_state[table_name] = alias

        if not cte_pairs:
            return ['\n'.join(comments) or '-- (no SQL output)']

        cte_parts = [f"{alias} AS (\n  {body}\n)" for alias, body in cte_pairs]
        last_alias = cte_pairs[-1][0]
        sql = "WITH\n" + ",\n".join(cte_parts) + f"\nSELECT * FROM {last_alias}"
        if comments:
            sql = '\n'.join(comments) + '\n' + sql
        return [sql]

    def export(self, code, backend='pandas'):
        """Parse DSL code and return generated Python code as a string

        Args:
            code: DSL code string to parse
            backend: Code generation backend — 'pandas' (default), 'duckdb', or 'sql'

        Returns:
            String containing all generated Python code, or None if parse error
        """
        results = self.parse(code)

        if isinstance(results, dict) and 'error' in results:
            print(f"Parse error: {results['error']}")
            return None

        try:
            code = self.generate_code(results, backend=backend)
        except ValueError as e:
            print(f"Code generation error: {e}")
            return None

        # Build preamble based on backend
        cg = self.code_generator
        if backend == 'duckdb':
            preamble = cg.duckdb_preamble()
        elif backend == 'polars':
            preamble = cg.polars_preamble()
        else:
            preamble = "import pandas as pd"

        # Collect all Python code
        python_lines = [preamble, ""]
        
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
        
        results = self.parse(code, globals_dict)

        if isinstance(results, dict) and 'error' in results:
            print(f"Parse error: {results['error']}")
            return None

        defs = self.parse_definitions(code, globals_dict)
        if isinstance(defs, dict) and 'error' in defs:
            print(f"Parse error: {defs['error']}")
            return None
        self._store_pivotal_values(globals_dict, defs.get('defined_values', {}))

        try:
            if backend != 'sql':
                results = self._expand_for_loops(results, globals_dict)
        except ValueError as e:
            print(f"Loop expansion error: {e}")
            return None

        try:
            python_code_list = self.generate_code(results, backend=backend)
        except ValueError as e:
            print(f"Code generation error: {e}")
            return None

        # For DuckDB/Polars the first element is the backend preamble — exec it
        # separately so the results[i] index stays aligned with the statements.
        if backend in ('duckdb', 'polars') and python_code_list:
            preamble = python_code_list[0]
            python_code_list = python_code_list[1:]
            try:
                exec(preamble, globals_dict)
            except Exception as e:
                print(f"{backend} preamble error: {e}")

        for i, python_code in enumerate(python_code_list):
            if verbose:
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
                if isinstance(e, AssertionError):
                    raise
                print(f"Execution error: {e}")
        
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


class _PivotalFunction:
    def __init__(self, name, definition, functions, lists, backend="pandas"):
        self.name = name
        self.definition = definition
        self.functions = functions
        self.lists = lists
        self.backend = backend
        self.parser = DSLParser(backend=backend)

    def _return_names(self):
        for node in self.definition.get('body', []):
            if isinstance(node, dict) and node.get('type') == 'return':
                return node.get('names', [])
        return []

    @staticmethod
    def _is_dataframe(value):
        return hasattr(value, 'columns')

    @staticmethod
    def _is_list_like(value):
        return isinstance(value, (list, tuple))

    def _bind_python_value(self, param_name, value, namespace):
        if self._is_dataframe(value):
            table_name = f"_pvt_arg_{self.name}_{param_name}"
            namespace[table_name] = value
            return table_name
        if self._is_list_like(value):
            return [str(item) if not isinstance(item, (int, float, bool, _LiteralStr)) else item for item in value]
        return value

    def _build_call_args(self, args, kwargs, namespace):
        params = self.definition.get('params', [])
        param_names = [param['name'] for param in params]
        return_params = set(self._return_names()) & set(param_names)
        call_values = {}

        target_params = [p for p in param_names if p not in return_params]
        if len(args) > len(target_params):
            target_params = param_names
        if len(args) > len(target_params):
            raise TypeError(f"{self.name}() expected at most {len(target_params)} positional arguments, got {len(args)}")

        for param_name, value in zip(target_params, args):
            call_values[param_name] = self._bind_python_value(param_name, value, namespace)

        for key, value in kwargs.items():
            if key not in param_names:
                raise TypeError(f"{self.name}() got an unexpected keyword argument '{key}'")
            if key in call_values:
                raise TypeError(f"{self.name}() got multiple values for argument '{key}'")
            call_values[key] = self._bind_python_value(key, value, namespace)

        for name in return_params:
            if name not in call_values:
                call_values[name] = f"_pvt_return_{self.name}_{name}"

        call_args = []
        for param in params:
            name = param['name']
            if name in call_values:
                call_args.append(call_values[name])
            elif param.get('has_default'):
                call_args.append(self.parser._resolve_compile_value(param.get('default'), self.lists))
            else:
                raise TypeError(f"{self.name}() missing required argument '{name}'")
        return call_args

    def __call__(self, *args, **kwargs):
        namespace = {'pd': pd}
        call_args = self._build_call_args(args, kwargs, namespace)
        call_node = {'type': 'function_call', 'name': self.name, 'args': call_args}
        ast_list = self.parser._expand_function_calls([call_node], self.functions, self.lists)
        if self.backend != 'sql':
            ast_list = self.parser._expand_for_loops(ast_list, namespace)
        python_code_list = self.parser.generate_code(ast_list, backend=self.backend)
        if self.backend in ('duckdb', 'polars') and python_code_list:
            preamble = python_code_list[0]
            python_code_list = python_code_list[1:]
            exec(preamble, namespace)
        for python_code in python_code_list:
            exec(python_code, namespace)

        returns = self._return_names()
        if not returns:
            return None
        values = [namespace.get(f"_pvt_return_{self.name}_{name}", namespace.get(name)) for name in returns]
        if len(values) == 1:
            return values[0]
        return tuple(values)


class PivotalFunctionLibrary(SimpleNamespace):
    pass


def load_functions(path_or_code, backend="pandas"):
    """Load Pivotal function definitions as Python callables."""
    text = str(path_or_code)
    source_path = Path(text)
    if '\n' not in text and source_path.exists():
        code = source_path.read_text(encoding='utf-8')
    else:
        code = text

    parser = DSLParser(backend=backend)
    parsed = parser.parse_definitions(code)
    if isinstance(parsed, dict) and 'error' in parsed:
        raise ValueError(str(parsed['error']))

    functions = parsed['functions']
    lists = parsed['lists']
    library = PivotalFunctionLibrary()
    for name, definition in functions.items():
        setattr(library, name, _PivotalFunction(name, definition, functions, lists, backend=backend))
    return library
        

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
