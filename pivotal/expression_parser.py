"""Standalone parser for additive assignment expression ASTs.

This module intentionally does not participate in backend code generation yet.
Unsupported legacy expressions return ``None`` so the public raw expression
string remains the authoritative fallback during the migration.
"""

import ast
from typing import Optional

from lark import Lark, Transformer, v_args
from lark.exceptions import LarkError


_EXPRESSION_GRAMMAR = r"""
    ?start: expression

    ?expression: sum
    ?sum: sum "+" product        -> add
        | sum "-" product        -> subtract
        | product
    ?product: product "*" unary  -> multiply
            | product "/" unary  -> divide
            | product "%" unary  -> modulo
            | unary
    ?unary: "+" unary            -> positive
          | "-" unary            -> negative
          | power
    ?power: atom "**" unary      -> power
          | atom

    ?atom: literal
         | runtime_call
         | runtime_reference
         | function_call
         | IDENTIFIER            -> column
         | "(" expression ")"

    function_call: IDENTIFIER "(" [arguments] ")"
    runtime_call: ":" IDENTIFIER "(" [arguments] ")"
    runtime_reference: ":" IDENTIFIER
    arguments: expression ("," expression)*

    ?literal: FLOAT              -> float_literal
            | INTEGER            -> integer_literal
            | STRING             -> string_literal
            | TRUE               -> true_literal
            | FALSE              -> false_literal
            | NULL               -> null_literal

    TRUE.2: /true/i
    FALSE.2: /false/i
    NULL.2: /null|none/i
    FLOAT: /(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+/
    INTEGER: /\d+/
    STRING: /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/
    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/

    %import common.WS_INLINE
    %ignore WS_INLINE
"""


def _literal(literal_type, value):
    return {"kind": "literal", "literal_type": literal_type, "value": value}


def _binary(operator, left, right):
    return {
        "kind": "binary",
        "operator": operator,
        "left": left,
        "right": right,
    }


@v_args(inline=True)
class _ExpressionTransformer(Transformer):
    def column(self, name):
        return {"kind": "column", "name": str(name)}

    def integer_literal(self, token):
        return _literal("integer", int(token))

    def float_literal(self, token):
        return _literal("float", float(token))

    def string_literal(self, token):
        return _literal("string", ast.literal_eval(str(token)))

    def true_literal(self, _token):
        return _literal("boolean", True)

    def false_literal(self, _token):
        return _literal("boolean", False)

    def null_literal(self, _token):
        return _literal("null", None)

    def arguments(self, *arguments):
        return list(arguments)

    def function_call(self, name, arguments=None):
        return {
            "kind": "call",
            "name": str(name),
            "arguments": arguments or [],
        }

    def runtime_reference(self, name):
        return {"kind": "runtime_reference", "name": str(name)}

    def runtime_call(self, name, arguments=None):
        return {
            "kind": "runtime_call",
            "name": str(name),
            "arguments": arguments or [],
        }

    def positive(self, operand):
        return {"kind": "unary", "operator": "positive", "operand": operand}

    def negative(self, operand):
        return {"kind": "unary", "operator": "negative", "operand": operand}

    def add(self, left, right):
        return _binary("add", left, right)

    def subtract(self, left, right):
        return _binary("subtract", left, right)

    def multiply(self, left, right):
        return _binary("multiply", left, right)

    def divide(self, left, right):
        return _binary("divide", left, right)

    def modulo(self, left, right):
        return _binary("modulo", left, right)

    def power(self, left, right):
        return _binary("power", left, right)


_PARSER = Lark(
    _EXPRESSION_GRAMMAR,
    parser="lalr",
    transformer=_ExpressionTransformer(),
)


def parse_expression(source: str) -> Optional[dict]:
    """Return a JSON-serializable expression AST, or ``None`` when unsupported."""
    if not isinstance(source, str) or not source.strip():
        return None

    try:
        return _PARSER.parse(source.strip())
    except (LarkError, ValueError, SyntaxError):
        return None

