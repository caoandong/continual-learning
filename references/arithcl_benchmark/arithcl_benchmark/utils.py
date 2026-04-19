from __future__ import annotations

import ast
import math
import operator
import random
from typing import Iterable, List, Sequence


def encode_base(n: int, base: int = 10, alphabet: str | None = None) -> str:
    if base < 2:
        raise ValueError("base must be >= 2")
    if alphabet is None:
        alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if base > len(alphabet):
        raise ValueError("alphabet too short for base")
    if n == 0:
        return alphabet[0]
    digits: List[str] = []
    value = n
    while value > 0:
        value, r = divmod(value, base)
        digits.append(alphabet[r])
    return "".join(reversed(digits))


def safe_eval_arithmetic(expr: str) -> int:
    allowed_ops = {
        ast.Add: operator.add,
        ast.Mult: operator.mul,
        ast.USub: operator.neg,
    }

    def _eval(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, int):
                raise ValueError("non-int constant")
            return int(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError(f"unsupported expression: {ast.dump(node)}")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree)


def force_digit_length(rng: random.Random, min_digits: int, max_digits: int) -> int:
    if min_digits < 1 or max_digits < min_digits:
        raise ValueError("invalid digit bounds")
    return rng.randint(min_digits, max_digits)


def sample_int_with_digits(rng: random.Random, min_digits: int, max_digits: int) -> int:
    digits = force_digit_length(rng, min_digits, max_digits)
    lo = 10 ** (digits - 1)
    hi = (10 ** digits) - 1
    if digits == 1:
        lo = 0
    return rng.randint(lo, hi)


def batch_iterable(items: Sequence, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def normalize_answer(text: str) -> str:
    return " ".join(str(text).strip().split())


def accuracy(predictions: Sequence[str], gold: Sequence[str]) -> float:
    if len(predictions) != len(gold):
        raise ValueError("length mismatch")
    if not predictions:
        return 0.0
    correct = sum(int(normalize_answer(p) == normalize_answer(g)) for p, g in zip(predictions, gold))
    return correct / len(predictions)
