"""Regression contracts for eval_all TensorBoard scalar filtering.

These tests pin two Oracle findings:
  * ``tools/test.py`` must route eval_all TensorBoard writes through a
    scalar-filter helper instead of calling ``tb_log.add_scalar`` for every
    value returned by the evaluator,
  * that helper must accept numeric scalars and skip lists/dicts/strings.

The routing contract is source/AST based. The helper behavior test is dynamic
and skips when the helper module cannot be imported without OpenPCDet/torch.
"""

import ast
import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_PY_PATH = REPO_ROOT / 'tools' / 'test.py'


def _parse(path):
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _find_function(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError('function %r not found' % name)


def _call_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _add_scalar_loops(function):
    loops = []
    for node in ast.walk(function):
        if not isinstance(node, ast.For):
            continue
        if any(
            isinstance(child, ast.Call) and _call_name(child.func) == 'add_scalar'
            for child in ast.walk(node)
        ):
            loops.append(node)
    return loops


def _helper_call_in_iter(loop):
    for child in ast.walk(loop.iter):
        if not isinstance(child, ast.Call):
            continue
        if _call_name(child.func) == 'add_scalar':
            continue
        if any(isinstance(arg, ast.Name) and arg.id == 'tb_dict' for arg in child.args):
            return child
    return None


def _discover_helper_name():
    repeat = _find_function(_parse(TEST_PY_PATH), 'repeat_eval_ckpt')
    loops = _add_scalar_loops(repeat)
    if len(loops) != 1:
        return None
    helper_call = _helper_call_in_iter(loops[0])
    if helper_call is None:
        return None
    return _call_name(helper_call.func)


def _find_helper_module(helper_name):
    for path in sorted((REPO_ROOT / 'tools').rglob('*.py')):
        try:
            tree = _parse(path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == helper_name:
                relative = path.relative_to(REPO_ROOT).with_suffix('')
                return '.'.join(relative.parts)
    return None


def test_eval_all_tensorboard_writes_go_through_scalar_filter_helper():
    repeat = _find_function(_parse(TEST_PY_PATH), 'repeat_eval_ckpt')
    loops = _add_scalar_loops(repeat)

    assert len(loops) == 1, 'expected one add_scalar loop in repeat_eval_ckpt'
    helper_call = _helper_call_in_iter(loops[0])
    assert helper_call is not None, (
        'eval_all must iterate a scalar-filter helper(tb_dict), not tb_dict.items()'
    )
    assert _call_name(helper_call.func) != 'items'


def test_scalar_filter_helper_accepts_numbers_and_skips_containers():
    helper_name = _discover_helper_name()
    assert helper_name is not None, (
        'tools/test.py does not route eval_all TensorBoard writes through a helper'
    )

    module_name = _find_helper_module(helper_name)
    assert module_name is not None, (
        'scalar-filter helper %r is not defined under tools/' % helper_name
    )

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip('helper module %s is not importable without OpenPCDet: %s' % (module_name, exc))
        return

    helper = getattr(module, helper_name, None)
    assert helper is not None, 'helper %r not found in %s' % (helper_name, module_name)

    mixed = {
        'numeric_float': 1.5,
        'numeric_int': 3,
        'list_value': [1, 2],
        'dict_value': {'a': 1},
        'string_value': 'text',
    }

    filtered = dict(helper(mixed))

    assert set(filtered) == {'numeric_float', 'numeric_int'}
    assert filtered['numeric_float'] == 1.5
    assert filtered['numeric_int'] == 3
