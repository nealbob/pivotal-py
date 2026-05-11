"""Pygments lexer for the Pivotal DSL.

Registered as a pygments.lexers entry point so that any Pygments-powered
tool (MkDocs, GitHub Linguist, Sphinx, etc.) picks it up automatically
when pivotal is installed.
"""
from pygments.lexer import RegexLexer, words
from pygments.token import (
    Comment, Keyword, Name, Number, Operator, Punctuation, String, Text,
)

from .syntax_metadata import token_category


class PivotalLexer(RegexLexer):
    """Lexer for the Pivotal data transformation DSL."""

    name = 'Pivotal'
    aliases = ['pivotal']
    filenames = ['*.pivotal']
    mimetypes = ['text/x-pivotal']

    KEYWORDS = tuple(token_category("statement_keywords"))
    MODIFIERS = tuple(
        token_category("clause_keywords")
        + token_category("merge_modifiers")
        + token_category("sort_modifiers")
        + token_category("format_types")
    )
    WORD_OPS = tuple(token_category("word_operators"))
    BUILTINS = tuple(token_category("builtin_functions"))
    CONSTANTS = tuple(token_category("constants"))

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
            (words(CONSTANTS, suffix=r'\b'), Keyword.Constant),

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
