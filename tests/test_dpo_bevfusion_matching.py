import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / 'pcdet' / 'tta_methods' / 'dpo_bevfusion_utils.py'


def _tree():
    assert UTILS_PATH.is_file(), 'DPO matcher utilities are missing: %s' % UTILS_PATH
    return ast.parse(UTILS_PATH.read_text(encoding='utf-8'), filename=str(UTILS_PATH))


def _function(name):
    matches = [
        node for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, 'function %s must exist exactly once' % name
    return matches[0]


def _class(name):
    matches = [
        node for node in _tree().body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    assert len(matches) == 1, 'class %s must exist exactly once' % name
    return matches[0]


def _method(class_name, method_name):
    matches = [
        node for node in _class(class_name).body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    assert len(matches) == 1, '%s.%s must exist exactly once' % (class_name, method_name)
    return matches[0]


def _source(node):
    return ast.unparse(node)


def test_score_filtering_has_high_medium_low_actions_without_gt_access():
    # Given clean Forward A predictions and scalar repository thresholds.
    function = _function('build_clean_pseudo_targets')
    source = _source(function)

    # When score actions are inspected.
    assert 'score_thresh' in source and 'neg_thresh' in source
    assert 'PseudoAction.HIGH' in source
    assert 'PseudoAction.MEDIUM' in source
    assert 'PseudoAction.LOW' in source
    threshold_source = _source(_function('_thresholds_for_labels'))
    assert 'labels - 1' in threshold_source

    # Then LOW is omitted and no ground-truth field can influence pseudo labels.
    assert 'gt_boxes' not in source
    assert 'ground_truth' not in source
    assert 'pred_boxes' in source and 'pred_scores' in source and 'pred_labels' in source


def test_matcher_is_same_class_one_to_one_and_uses_actual_pairs():
    # Given the matcher implementation.
    function = _function('_hungarian_same_class_matches')
    source = _source(function)

    # When class partitioning and assignment are inspected.
    assert 'linear_sum_assignment' in source
    assert 'clean_labels' in source and 'disturbed_labels' in source
    assert 'matched_rows' in source and 'matched_cols' in source

    # Then pair costs use both returned indices, never a row-wise minimum.
    subscripts = [
        ast.unparse(node) for node in ast.walk(function)
        if isinstance(node, ast.Subscript)
    ]
    assert any('matched_rows' in item and 'matched_cols' in item for item in subscripts)
    assert '.min(dim=-1)' not in source and '.min(dim=1)' not in source


def test_cost_is_iou_plus_twice_l1_on_first_seven_box_values():
    # Given the pairwise cost helper.
    source = _source(_function('_pairwise_cost'))

    # When its geometry operands are inspected.
    assert source.count(':7') >= 2
    assert 'boxes_iou3d_gpu' in source
    assert 'torch.cdist' in source and 'p=1' in source

    # Then no score, velocity, or full-box distance enters the cost.
    assert '-iou' in source.replace(' ', '')
    assert '2.0 * l1' in source or '2 * l1' in source


def test_unmatched_clean_boxes_are_low_and_never_enter_history():
    # Given refinement after class-gated matching.
    source = _source(_function('match_refined_targets'))

    # When unmatched rows and finite pair costs are handled.
    assert 'PseudoAction.LOW' in source
    assert 'matched_costs' in source
    assert 'torch.isfinite' in source

    # Then history receives matched finite costs rather than all clean rows.
    append_calls = [
        ast.unparse(node) for node in ast.walk(_function('match_refined_targets'))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in {'append_batch', 'extend'}
    ]
    assert append_calls and all('matched_costs' in call or 'finite_costs' in call for call in append_calls)


def test_global_history_appends_batch_before_current_inclusive_quantiles():
    # Given one mutable stream-wide cost history.
    append_batch = _method('DPOCostHistory', 'append_batch')
    thresholds = _method('DPOCostHistory', 'thresholds')

    # When storage and threshold equations are inspected.
    assert 'isfinite' in _source(append_batch)
    threshold_source = _source(thresholds)
    assert 'quantile' in threshold_source
    assert 'self.alpha' in threshold_source
    assert '1.0 - self.alpha' in threshold_source or '1 - self.alpha' in threshold_source

    # Then current costs are appended once before thresholds are read by refinement.
    refinement = _source(_function('match_refined_targets'))
    assert refinement.index('append_batch') < refinement.index('thresholds')


def test_cutoff_is_optional_ema_and_latches_future_batches():
    # Given the optional early-cutoff stream state.
    update = _method('DPOCutoffState', 'update')
    source = _source(update)

    # When a finite matched batch mean arrives.
    assert 'self.gamma' in source
    assert '1.0 - self.gamma' in source or '1 - self.gamma' in source
    assert 'self.c_stop' in source

    # Then the stop state is persistent and null C_STOP cannot trigger it.
    assert 'self.stopped' in source
    assert 'is None' in source
    assert '<= self.c_stop' in source
