import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HEAD_PATH = REPO_ROOT / 'pcdet' / 'models' / 'dense_heads' / 'transfusion_head.py'
TREE = ast.parse(HEAD_PATH.read_text(encoding='utf-8'), filename=str(HEAD_PATH))


def _head_class():
    matches = [
        node for node in TREE.body
        if isinstance(node, ast.ClassDef) and node.name == 'TransFusionHead'
    ]
    assert len(matches) == 1, 'TransFusionHead class not found exactly once'
    return matches[0]


def _method(name):
    matches = [
        node for node in _head_class().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, 'method %s not found exactly once' % name
    return matches[0]


def _call(node, name):
    matches = [
        child for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and (
            isinstance(child.func, ast.Name) and child.func.id == name
            or isinstance(child.func, ast.Attribute) and child.func.attr == name
        )
    ]
    assert matches, 'call %s not found' % name
    return matches[0]


def _keyword(call, name):
    matches = [keyword.value for keyword in call.keywords if keyword.arg == name]
    assert len(matches) == 1, 'keyword %s not found exactly once' % name
    return matches[0]


def _optional_parameter(method, name):
    positional = method.args.args
    defaults = [None] * (len(positional) - len(method.args.defaults)) + method.args.defaults
    return any(
        argument.arg == name
        and isinstance(default, ast.Constant)
        and default.value is None
        for argument, default in zip(positional, defaults)
    )


def _assignment(method, target_name, value_fragment=None):
    matches = [
        child for child in ast.walk(method)
        if isinstance(child, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == target_name for target in child.targets)
        and (value_fragment is None or value_fragment in ast.unparse(child.value))
    ]
    assert len(matches) == 1, 'assignment to %s not found exactly once' % target_name
    return matches[0]


def _integer(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -node.operand.value
    return None


def _action_values(method):
    values = set()
    for comparison in (
        node for node in ast.walk(method)
        if isinstance(node, ast.Compare) and 'pseudo_actions' in ast.unparse(node)
    ):
        for candidate in (comparison.left, *comparison.comparators):
            value = _integer(candidate)
            if value is not None:
                values.add(value)
    return values


def _action_filtered_pos_name(method, value):
    for assignment in (node for node in ast.walk(method) if isinstance(node, ast.Assign)):
        if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
            continue
        expression = ast.unparse(assignment.value)
        if expression.startswith('pos_inds[') and value in _action_values(assignment.value):
            return assignment.targets[0].id
    raise AssertionError('missing pos_inds filter for pseudo action %d' % value)


def _path_to(root, target):
    if root is target:
        return [root]
    for child in ast.iter_child_nodes(root):
        path = _path_to(child, target)
        if path:
            return [root, *path]
    return []


def test_pseudo_weights_only_reweight_matched_query_losses():
    # Given the existing pseudo-weight branch in target assignment.
    method = _method('get_targets_single')
    branches = [
        node for node in ast.walk(method)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == 'pseudo_weights is not None and pseudo_weights.numel() > 0'
    ]
    # When its complete subtree is inspected.
    branch_source = ast.unparse(branches[0])
    # Then weights can scale matched cls/reg losses, but not semantic effects.
    assert 'label_weights[pos_inds]' in branch_source
    assert 'bbox_weights[pos_inds, :]' in branch_source
    assert 'draw_gaussian_to_heatmap' not in branch_source
    assert 'mean_iou' not in branch_source
    assert 'return' not in branch_source


def test_actions_propagate_optionally_through_the_training_call_chain():
    # Given every training-layer entry point.
    forward = _method('forward')
    methods = [_method(name) for name in ('loss', 'get_targets', 'get_targets_single')]
    # When signatures and calls are inspected.
    action_read = _assignment(forward, 'pseudo_actions')
    loss_call = _call(forward, 'loss')
    target_call = _call(methods[0], 'get_targets')
    single_call = _call(methods[1], 'get_targets_single')
    # Then the absent-by-default sidecar reaches assignment without affecting inference.
    assert ast.unparse(action_read.value) == "batch_dict.get('tta_pseudo_actions', None)"
    assert all(_optional_parameter(method, 'pseudo_actions') for method in methods)
    assert ast.unparse(_keyword(loss_call, 'pseudo_actions')) == 'pseudo_actions'
    assert ast.unparse(_keyword(target_call, 'pseudo_actions')) == 'pseudo_actions'
    assert ast.unparse(_keyword(single_call, 'pseudo_actions')) == 'gt_pseudo_actions'
    training_guard = next(
        node for node in ast.walk(forward)
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'not self.training'
    )
    assert any(action_read in ast.walk(statement) for statement in training_guard.orelse)


def test_actions_use_the_same_valid_indices_as_boxes_and_weights():
    # Given per-batch filtering before single-sample assignment.
    method = _method('get_targets')
    single_call = _call(method, 'get_targets_single')
    # When aligned sidecars and target arguments are inspected.
    action_slice = _assignment(method, 'gt_pseudo_actions', '[batch_idx][valid_idx]')
    weight_slice = _assignment(method, 'gt_pseudo_weights', '[batch_idx][valid_idx]')
    reg_weight_slice = _assignment(method, 'gt_pseudo_reg_weights', '[batch_idx][valid_idx]')
    # Then every optional row uses the exact valid_idx applied to boxes and labels.
    assert ast.unparse(action_slice.value) == 'pseudo_actions[batch_idx][valid_idx]'
    assert ast.unparse(weight_slice.value) == 'pseudo_weights[batch_idx][valid_idx]'
    assert ast.unparse(reg_weight_slice.value) == 'pseudo_reg_weights[batch_idx][valid_idx]'
    assert ast.unparse(single_call.args[0]) == 'gt_bboxes[valid_idx]'
    assert ast.unparse(single_call.args[1]) == 'gt_labels_3d[batch_idx][valid_idx]'


def test_high_and_medium_actions_have_distinct_query_and_dense_semantics():
    # Given valid HIGH=1 and MEDIUM=-1 targets presented to Hungarian assignment.
    method = _method('get_targets_single')
    source = ast.unparse(method)
    assign_call = _call(method, 'assign')
    # When action-filtered positive query sets are inspected.
    high_pos = _action_filtered_pos_name(method, 1)
    medium_pos = _action_filtered_pos_name(method, -1)

    # Then both actions are assigned as ordinary nonnegative GT classes first.
    assert _action_values(method) == {-1, 1}
    assert [ast.unparse(arg) for arg in assign_call.args[:3]] == ['bboxes_tensor', 'gt_bboxes_tensor', 'gt_labels_3d']
    assert 'gt_labels[i] >= 0' in ast.unparse(_method('get_targets'))
    assert 'labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]' in source
    assert 'bbox_weights[pos_inds, :] = 1.0' in source
    assert 'label_weights[pos_inds] = 1.0' in source

    # Then MEDIUM matched queries contribute neither classification nor regression.
    assert 'label_weights[%s] = 0.0' % medium_pos in source
    assert 'bbox_weights[%s, :] = 0.0' % medium_pos in source

    # Then only HIGH positives contribute positive counts and matched-IoU diagnostics.
    mean_iou = _assignment(method, 'mean_iou')
    final_return = max(
        (node for node in ast.walk(method) if isinstance(node, ast.Return)),
        key=lambda node: node.lineno,
    )
    assert isinstance(final_return.value, ast.Tuple)
    assert high_pos in ast.unparse(mean_iou.value)
    assert high_pos in ast.unparse(final_return.value.elts[4])

    # Then only HIGH targets produce dense heatmap Gaussians.
    draw_call = _call(method, 'draw_gaussian_to_heatmap')
    high_guards = [
        node for node in _path_to(method, draw_call)
        if isinstance(node, ast.If)
        and 'pseudo_actions' in ast.unparse(node.test)
        and 1 in {_integer(child) for child in ast.walk(node.test)}
    ]
    assert high_guards


def test_absent_actions_default_valid_targets_to_high_without_relabeling():
    # Given legacy normal, SAR, Tent, and CoTTA calls with no action sidecar.
    method = _method('get_targets_single')
    absent_branches = [
        node for node in ast.walk(method)
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'pseudo_actions is None'
    ]
    assert len(absent_branches) == 1, 'missing absent-action compatibility branch'
    absent_branch = absent_branches[0]

    # When the compatibility branch initializes actions.
    branch_source = ast.unparse(absent_branch)

    # Then all valid targets become HIGH and actions never become class IDs.
    assert 'pseudo_actions' in branch_source
    assert any(name in branch_source for name in ('new_ones', 'ones_like'))
    relabeling = []
    for node in ast.walk(method):
        if not isinstance(node, (ast.Assign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if (
            any('labels' in ast.unparse(target) for target in targets)
            and 'pseudo_actions' in ast.unparse(node.value)
        ):
            relabeling.append(ast.unparse(node))
    assert relabeling == []


def test_action_extension_is_confined_to_the_optional_target_path():
    # Given the complete TransFusionHead source tree.
    methods = [node for node in _head_class().body if isinstance(node, ast.FunctionDef)]

    # When methods mentioning the optional sidecar are enumerated.
    touched = {node.name for node in methods if 'pseudo_actions' in ast.unparse(node)}

    # Then no unrelated prediction, decoding, or adaptation path depends on actions.
    assert touched == {'forward', 'loss', 'get_targets', 'get_targets_single'}
