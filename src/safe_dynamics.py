"""Restricted array-expression evaluator for editable neural dynamics."""
import ast
import operator

import numpy as np


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    'tanh': np.tanh,
    'maximum': np.maximum,
    'minimum': np.minimum,
    'clip': np.clip,
    'abs': np.abs,
}
_NAMES = {'h', 'recurrent', 'sensory', 'gain', 'leak', 'bias'}


class DynamicsExpression:
    """Compile and evaluate a small, non-Python neural update expression."""

    def __init__(self, expression):
        self.expression = str(expression).strip()
        if not self.expression:
            raise ValueError('dynamics equation cannot be empty')
        try:
            self.tree = ast.parse(self.expression, mode='eval').body
        except SyntaxError as exc:
            raise ValueError(f'invalid dynamics equation: {exc.msg}') from exc
        self._validate(self.tree)

    def _validate(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError('equation constants must be numbers')
            return
        if isinstance(node, ast.Name):
            if node.id not in _NAMES:
                raise ValueError(f'equation name is not allowed: {node.id}')
            return
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            self._validate(node.left)
            self._validate(node.right)
            return
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            self._validate(node.operand)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _FUNCTIONS or node.keywords:
                raise ValueError(f'equation function is not allowed: {node.func.id}')
            for argument in node.args:
                self._validate(argument)
            return
        raise ValueError(f'equation syntax is not allowed: {type(node).__name__}')

    def _evaluate(self, node, values):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](
                self._evaluate(node.left, values), self._evaluate(node.right, values))
        if isinstance(node, ast.UnaryOp):
            return _UNARY[type(node.op)](self._evaluate(node.operand, values))
        if isinstance(node, ast.Call):
            arguments = [self._evaluate(item, values) for item in node.args]
            return _FUNCTIONS[node.func.id](*arguments)
        raise AssertionError(type(node).__name__)

    def evaluate(self, **values):
        missing = _NAMES.difference(values)
        if missing:
            raise ValueError(f'equation values missing: {sorted(missing)}')
        result = np.asarray(self._evaluate(self.tree, values), dtype=np.float32)
        expected = np.shape(values['h'])
        if result.shape != expected:
            raise ValueError(f'equation result shape {result.shape} does not match state {expected}')
        if not np.isfinite(result).all():
            raise FloatingPointError('dynamics equation produced NaN or infinity')
        return result
