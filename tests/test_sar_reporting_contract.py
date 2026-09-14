"""Regression contracts for SAR evaluation reporting.

These source/AST tests pin the intended reporting contract that the SAR
evaluator must satisfy:
  * the independent SAR evaluator must not accumulate ``statistics_info``
    recall metrics from the adaptation forward,
  * the SAR result finalizer must not create ``recall/*`` result keys,
  * the finalizer must keep dataset mAP/NDS evaluation and prediction
    diagnostics,
  * aggregate SAR metadata must report batch/logit finite/nan/inf/nonfinite
    counts and ratios.

The tests read the production sources as text so they stay collectible on
hosts without torch/OpenPCDet installed. Expected metadata keys must combine
a scope token (``batch``/``logit``), a kind token (``finite``/``nan``/``inf``/
``nonfinite``) and a ``count`` or ``ratio`` token, for example
``sar/batch_finite_count`` and ``sar/logit_nan_ratio``.
"""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SAR_EVAL_UTILS_PATH = REPO_ROOT / 'tools' / 'eval_utils' / 'sar_eval_utils.py'
SAR_EVAL_RESULTS_PATH = REPO_ROOT / 'tools' / 'eval_utils' / 'sar_eval_results.py'

REQUIRED_METADATA_SCOPES = ('batch', 'logit')
REQUIRED_METADATA_KINDS = ('finite', 'nan', 'inf', 'nonfinite')


def _parse(path):
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _find_function(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError('function %r not found' % name)


def _call_names(node):
    names = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = getattr(child.func, 'id', None) or getattr(child.func, 'attr', None)
        if name is not None:
            names.add(name)
    return names


def _string_constants(node):
    return {
        child.value for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _assigned_targets(node):
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, ast.AugAssign):
        return [node.target]
    return []


def _result_assignment_keys(function):
    keys = []
    for node in ast.walk(function):
        for target in _assigned_targets(node):
            if not isinstance(target, ast.Subscript):
                continue
            if ast.unparse(target.value) != 'result':
                continue
            keys.append(ast.unparse(target.slice))
    return keys


def _result_update_literal_keys(function):
    keys = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != 'update':
            continue
        if ast.unparse(func.value) != 'result':
            continue
        for arg in node.args:
            if not isinstance(arg, ast.Dict):
                continue
            keys.extend(
                key.value for key in arg.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
    return keys


def _dict_literal_keys(function, names):
    keys = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key in node.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.append(key.value)
            elif key is not None:
                keys.append(ast.unparse(key))
    return keys


def _dict_subscript_keys(function, names):
    keys = []
    for node in ast.walk(function):
        for target in _assigned_targets(node):
            if not isinstance(target, ast.Subscript):
                continue
            if ast.unparse(target.value) not in names:
                continue
            if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                keys.append(target.slice.value)
    return keys


def test_sar_evaluator_does_not_call_statistics_info():
    evaluator = _find_function(_parse(SAR_EVAL_UTILS_PATH), 'eval_sar_one_epoch')

    assert 'statistics_info' not in _call_names(evaluator)


def test_sar_finalizer_creates_no_recall_result_keys():
    finalizer = _find_function(_parse(SAR_EVAL_RESULTS_PATH), '_finalize_sar_results')
    keys = _result_assignment_keys(finalizer) + _result_update_literal_keys(finalizer)

    recall_keys = sorted(key for key in keys if 'recall/' in key)
    assert recall_keys == []


def test_sar_finalizer_keeps_dataset_metrics_and_prediction_diagnostics():
    finalizer = _find_function(_parse(SAR_EVAL_RESULTS_PATH), '_finalize_sar_results')
    calls = _call_names(finalizer)
    literals = _string_constants(finalizer)

    assert 'evaluation' in calls
    assert 'update' in calls
    assert any(value.startswith('diag/') for value in literals)


def test_sar_metadata_reports_finite_nan_inf_nonfinite_counts_and_ratios():
    evaluator = _find_function(_parse(SAR_EVAL_UTILS_PATH), 'eval_sar_one_epoch')
    names = ('metadata', 'aggregate')
    keys = _dict_literal_keys(evaluator, names) + _dict_subscript_keys(evaluator, names)

    missing = []
    for scope in REQUIRED_METADATA_SCOPES:
        for kind in REQUIRED_METADATA_KINDS:
            matching = [
                key for key in keys
                if scope in key and kind in key
                and not (kind == 'finite' and 'nonfinite' in key)
            ]
            if not any('count' in key for key in matching):
                missing.append('%s %s count' % (scope, kind))
            if not any('ratio' in key for key in matching):
                missing.append('%s %s ratio' % (scope, kind))

    assert missing == []
