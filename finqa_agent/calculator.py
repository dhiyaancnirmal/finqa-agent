"""A calculator tool: evaluates arithmetic safely via the AST, no eval()."""

from __future__ import annotations

import ast
import math
import operator
import re

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"unsupported expression: {ast.dump(node)[:80]}")


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result as text.

    Accepts numbers with thousands separators ("1,234.5") and a trailing "%"
    on a number ("12%" means 0.12). Errors come back as text so the model can fix them.
    """
    expr = expression.replace("$", "").replace("^", "**")
    expr = _strip_thousands(expr)
    expr = _percent_literals(expr)
    try:
        value = _eval(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero"
    except (SyntaxError, ValueError, TypeError, OverflowError) as e:
        return f"Error: {e}"
    if isinstance(value, float):
        return repr(round(value, 10))
    return str(value)


def _strip_thousands(expr: str) -> str:

    # "1,234,567.8" -> "1234567.8"; leaves "max(1, 2)" alone (space after comma).
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expr)


def _percent_literals(expr: str) -> str:

    return re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
