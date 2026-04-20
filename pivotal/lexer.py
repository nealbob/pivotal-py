"""Pygments lexer for the Pivotal DSL.

Registered as a pygments.lexers entry point so that any Pygments-powered
tool (MkDocs, GitHub Linguist, Sphinx, etc.) picks it up automatically
when pivotal is installed.
"""
from pygments.lexer import RegexLexer, words
from pygments.token import (
    Comment, Keyword, Name, Number, Operator, Punctuation, String, Text,
)


class PivotalLexer(RegexLexer):
    """Lexer for the Pivotal data transformation DSL."""

    name = 'Pivotal'
    aliases = ['pivotal']
    filenames = ['*.pivotal']
    mimetypes = ['text/x-pivotal']

    # Primary DSL verbs
    KEYWORDS = (
        'with', 'load', 'from', 'delete', 'save', 'concat',
        'filter', 'select', 'drop', 'distinct', 'rename',
        'sort', 'order', 'group', 'by', 'agg',
        'merge', 'pivot', 'unpivot', 'apply',
        'intersect', 'cast', 'python', 'end', 'query',
        'rank', 'lag', 'lead',
        'cumsum', 'cummean', 'cummin', 'cummax',
        'rolling', 'fillna', 'dropna',
        'show', 'plot', 'table', 'all',
    )

    # Clause keywords, modifiers, and sub-options
    MODIFIERS = (
        'as', 'on', 'where', 'else', 'rows', 'cols',
        'left', 'right', 'inner', 'outer',
        'asc', 'desc', 'pct',
        'head', 'summary',
        'variable', 'value', 'id',
        'left_on', 'right_on', 'suffixes',
        'header', 'names', 'path', 'format', 'include', 'exclude',
        'chart_format',
        'title', 'subtitle', 'font', 'size', 'stub', 'spanner', 'auto',
        'label', 'stripe', 'canvas', 'style',
        'kind', 'colormap', 'legend', 'x', 'y', 'c',
        'number', 'integer', 'currency', 'percent', 'date',
        'int', 'float', 'string', 'str', 'bool', 'boolean', 'datetime', 'strict',
    )

    # Word-form logical and comparison operators
    WORD_OPS = (
        'and', 'or', 'not', 'in', 'between',
        'contains', 'startswith', 'endswith',
    )

    # Built-in aggregation and string functions
    BUILTINS = (
        'sum', 'mean', 'count', 'min', 'max', 'median', 'std', 'nunique',
        'wmean', 'wavg', 'avg',
        'upper', 'lower', 'trim', 'ltrim', 'rtrim',
        'left', 'right', 'substr', 'len', 'replace',
    )

    tokens = {
        'root': [
            # Comments
            (r'#[^\n]*',              Comment.Single),
            (r'--[^\n]*',             Comment.Single),
            (r'/\*',                  Comment.Multiline, 'block-comment'),

            # Strings
            (r'"[^"]*"',              String.Double),
            (r"'[^']*'",              String.Single),

            # Python variable references  :varname
            (r':[a-zA-Z_]\w*',        Name.Variable),

            # Numbers
            (r'\d+\.\d+',             Number.Float),
            (r'\d+',                  Number.Integer),

            # Symbol operators
            (r'(==|!=|>=|<=|>|<)',    Operator),
            (r'[+\-*/=]',             Operator),

            # Keywords — must precede the general identifier rule
            (words(KEYWORDS,  suffix=r'\b'), Keyword),
            (words(MODIFIERS, suffix=r'\b'), Keyword.Declaration),
            (words(WORD_OPS,  suffix=r'\b'), Operator.Word),
            (words(BUILTINS,  suffix=r'\b'), Name.Builtin),

            # Boolean / None literals
            (words(('True', 'False', 'true', 'false', 'None', 'none'), suffix=r'\b'),
             Keyword.Constant),

            # Identifiers (table names, column names, etc.)
            (r'[a-zA-Z_]\w*',         Name),

            # Punctuation
            (r'[\[\](),;]',            Punctuation),

            # Whitespace
            (r'\s+',                  Text),
        ],
        'block-comment': [
            (r'\*/',      Comment.Multiline, '#pop'),
            (r'[^*/]+',   Comment.Multiline),
            (r'[*/]',     Comment.Multiline),
        ],
    }
