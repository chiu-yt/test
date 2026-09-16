import ast
import inspect
import textwrap
from pathlib import Path

from pcdet.tta_methods.sar import SAR
from pcdet.tta_methods.tent_hooks import TransFusionLogitCapture
from tools.eval_utils.sar_eval_utils import eval_sar_one_epoch


def _function_tree(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function)))


def _call_name(node):
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _calls(tree, name):
    return sorted(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.Call) and _call_name(node) == name),
        key=lambda node: node.lineno,
    )


def test_evaluator_accumulates_only_detached_pre_adaptation_predictions():
    # Given the production evaluator syntax tree.
    tree = _function_tree(eval_sar_one_epoch)
    model_calls = _calls(tree, 'model')
    captures = _calls(tree, 'TransFusionLogitCapture')
    detach_calls = _calls(tree, '_detach_tensor_tree')
    annotation_calls = _calls(tree, 'generate_prediction_dicts')
    adapt_calls = _calls(tree, 'adapt')
    step_calls = _calls(tree, 'SARStepInput')
    accumulations = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == 'det_annos'
    ]
    captured_forwards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and _call_name(item.context_expr.func) == 'TransFusionLogitCapture'
            for item in node.items
        )
        and _calls(node, 'model')
    ]
    prediction_contexts = [
        _call_name(item.context_expr.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
        and _call_name(item.context_expr.func) in {'enable_grad', 'no_grad'}
        and _calls(node, 'TransFusionLogitCapture')
    ]

    # When prediction, annotation, and adaptation sites are ordered.
    ordered_lines = [
        captures[0].lineno, model_calls[0].lineno, detach_calls[0].lineno,
        annotation_calls[0].lineno, accumulations[0].lineno,
        adapt_calls[0].lineno,
    ]

    first_entropy_arguments = [
        keyword.value
        for call in step_calls
        for keyword in call.keywords
        if keyword.arg == 'first_entropy'
    ]
    detached_first_entropy = [
        call for call in _calls(tree, 'detach')
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == 'first_entropy'
    ]
    deleted_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Delete)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    # Then one graph-connected official prediction is detached for output before adapt.
    assert ordered_lines == sorted(ordered_lines)
    assert len(model_calls) == 1
    assert len(detach_calls) == 1
    assert isinstance(detach_calls[0].args[0], ast.Name)
    assert detach_calls[0].args[0].id == 'pred_dicts'
    assert len(annotation_calls) == 1
    assert len(accumulations) == 1
    assert len(adapt_calls) == 1
    assert len(step_calls) == 1
    assert len(captured_forwards) == 1
    assert prediction_contexts == ['enable_grad']
    assert len(first_entropy_arguments) == 1
    assert isinstance(first_entropy_arguments[0], ast.Name)
    assert first_entropy_arguments[0].id == 'first_entropy'
    assert detached_first_entropy == []
    assert {
        'step', 'first_entropy', 'logits', 'capture', 'prediction_batch',
    }.issubset(deleted_names)


def test_transfusion_capture_exposes_detached_top_proposal_ids():
    # Given the TransFusion proposal producer and capture wrapper source trees.
    head_path = (
        Path(__file__).resolve().parents[1]
        / 'pcdet' / 'models' / 'dense_heads' / 'transfusion_head.py'
    )
    head_tree = ast.parse(head_path.read_text())
    capture_tree = _function_tree(TransFusionLogitCapture)

    # When assignments to the proposal identity channel are inspected.
    head_assignments = [
        node for node in ast.walk(head_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == 'last_top_proposals'
            for target in node.targets
        )
    ]
    capture_assignments = [
        node for node in ast.walk(capture_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == 'proposal_ids'
            for target in node.targets
        )
    ]

    # Then global Top-K IDs are detached at the head and exposed by the hook.
    assert len(head_assignments) == 1
    assert _calls(head_assignments[0], 'detach')
    assert len(capture_assignments) >= 2
    assert any(_calls(assignment, 'detach') for assignment in capture_assignments)


def test_sar_aligns_entropy_by_proposal_identity_not_shape_only():
    # Given the SAR adaptation state machine.
    tree = _function_tree(SAR.adapt)
    attribute_names = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    # When its proposal alignment operations are inspected.
    strict_alignment = _calls(tree, 'align_entropy_strict')
    selected_alignment = _calls(tree, 'align_selected_entropy')

    # Then official entropy stays graph-connected and perturbed entropy aligns by ID.
    assert 'proposal_ids' in attribute_names
    assert strict_alignment == []
    assert len(selected_alignment) == 1
