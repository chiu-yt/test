import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest


torch = pytest.importorskip('torch')
import torch.nn as nn

from pcdet.tta_methods.cotta_utils import (
    filter_predictions_to_targets,
    initialize_cotta_models,
    stochastic_restore,
    transform_boxes_between_views,
    update_ema_teacher,
)
from tools.eval_utils.cotta_eval_utils import (
    _ScaleViewBuilder,
    _is_gt_field,
    eval_cotta_one_epoch,
)
from tools.eval_utils.cotta_eval_results import _ema_parameter_delta


class TinyCottaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)


def _trainable_names(model):
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _function_tree(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function)))


def _call_lines(tree, name):
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    )


def _attribute_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_name(node.value)
        return '%s.%s' % (parent, node.attr) if parent else node.attr
    return None


def _subscript_slice(node):
    slice_node = node.slice
    if type(slice_node).__name__ == 'Index':
        return getattr(slice_node, 'value')
    return slice_node


def test_initialize_cotta_models_preserves_source_state_and_student_trainable_scope():
    # Given a source detector with only its head selected for adaptation.
    model = TinyCottaModel()
    model.backbone.requires_grad_(False)
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    original_trainable_names = _trainable_names(model)

    # When the independent source, EMA teacher, and student are initialized.
    source_anchor, teacher, student = initialize_cotta_models(model)

    # Then modes, gradients, values, and model independence match CoTTA semantics.
    assert source_anchor is not model
    assert teacher is not model
    assert student is not model
    assert source_anchor.training is False
    assert teacher.training is False
    assert student.training is True
    assert not any(parameter.requires_grad for parameter in source_anchor.parameters())
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    assert _trainable_names(student) == original_trainable_names
    for initialized_model in (source_anchor, teacher, student):
        for name, value in initialized_model.state_dict().items():
            assert torch.equal(value, original_state[name])
    with torch.no_grad():
        student.head.weight.add_(1.0)
    assert torch.equal(source_anchor.head.weight, original_state['head.weight'])
    assert torch.equal(teacher.head.weight, original_state['head.weight'])


def test_update_ema_teacher_uses_teacher_weighted_moving_average():
    # Given teacher and student parameters with known, distinct values.
    teacher = nn.Linear(2, 1)
    student = nn.Linear(2, 1)
    with torch.no_grad():
        teacher.weight.zero_()
        teacher.bias.fill_(2.0)
        student.weight.fill_(8.0)
        student.bias.fill_(6.0)

    # When the teacher receives an EMA update with alpha 0.75.
    update_ema_teacher(teacher, student, alpha=0.75)

    # Then each teacher parameter is alpha * teacher + (1 - alpha) * student.
    torch.testing.assert_close(teacher.weight, torch.full_like(teacher.weight, 2.0))
    torch.testing.assert_close(teacher.bias, torch.full_like(teacher.bias, 3.0))
    torch.testing.assert_close(student.weight, torch.full_like(student.weight, 8.0))
    assert not any(parameter.requires_grad for parameter in teacher.parameters())


def test_update_ema_teacher_copies_discrete_parameters_without_casting():
    # Given teacher and student models with floating and discrete parameters.
    teacher = TinyCottaModel()
    student = TinyCottaModel()
    teacher.step = nn.Parameter(torch.tensor(1, dtype=torch.long), requires_grad=False)
    student.step = nn.Parameter(torch.tensor(7, dtype=torch.long), requires_grad=False)
    with torch.no_grad():
        for parameter in teacher.parameters():
            if parameter.is_floating_point():
                parameter.zero_()
        for parameter in student.parameters():
            if parameter.is_floating_point():
                parameter.fill_(8.0)

    # When the pre-update delta is measured and the teacher receives an EMA update.
    delta = _ema_parameter_delta(teacher, student, alpha=0.75)
    update_ema_teacher(teacher, student, alpha=0.75)

    # Then floating parameters use EMA while the discrete parameter is copied exactly.
    assert delta == pytest.approx(2.0)
    assert teacher.step.item() == 7
    torch.testing.assert_close(teacher.head.weight, torch.full_like(teacher.head.weight, 2.0))


@pytest.mark.parametrize(
    ('probability', 'expected_restored'),
    [(0.0, 0), (1.0, 4)],
)
def test_stochastic_restore_respects_scope_and_probability_boundaries(
        probability, expected_restored):
    # Given a changed student and a source anchor where only head.weight is eligible.
    source_anchor = TinyCottaModel()
    student = TinyCottaModel()
    with torch.no_grad():
        for parameter in source_anchor.parameters():
            parameter.fill_(1.0)
        for parameter in student.parameters():
            parameter.fill_(9.0)
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
        if name != 'head.weight'
    }

    # When restoration is requested at either probability boundary.
    restored_count, eligible_count = stochastic_restore(
        student,
        source_anchor,
        trainable_names=['head.weight'],
        probability=probability,
        generator=torch.Generator().manual_seed(7),
    )

    # Then only eligible elements may change and actual/eligible counts are reported.
    assert restored_count == expected_restored
    assert eligible_count == student.head.weight.numel()
    expected_head_value = 1.0 if probability == 1.0 else 9.0
    torch.testing.assert_close(
        student.head.weight,
        torch.full_like(student.head.weight, expected_head_value),
    )
    for name, parameter in student.named_parameters():
        if name in frozen_before:
            assert torch.equal(parameter, frozen_before[name])


def test_transform_boxes_between_scale_views_includes_dimensions_and_velocities():
    # Given a 9D nuScenes box expressed in a weak view scaled by 2.
    weak_boxes = torch.tensor([
        [20.0, -8.0, 2.0, 4.0, 8.0, 3.0, 0.3, 6.0, -4.0],
    ])
    weak_matrix = torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0]))
    strong_matrix = torch.diag(torch.tensor([0.5, 0.5, 0.5, 1.0]))

    # When it is mapped weak -> canonical -> strong using scale-only matrices.
    strong_boxes = transform_boxes_between_views(
        weak_boxes,
        source_matrix=weak_matrix,
        target_matrix=strong_matrix,
    )

    # Then center, dimensions, and velocity scale while heading stays unchanged.
    expected = torch.tensor([
        [5.0, -2.0, 0.5, 1.0, 2.0, 0.75, 0.3, 1.5, -1.0],
    ])
    torch.testing.assert_close(strong_boxes, expected)
    canonical_boxes = transform_boxes_between_views(
        weak_boxes,
        source_matrix=weak_matrix,
        target_matrix=torch.eye(4),
    )
    torch.testing.assert_close(
        canonical_boxes,
        torch.tensor([[10.0, -4.0, 1.0, 2.0, 4.0, 1.5, 0.3, 3.0, -2.0]]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_transform_boxes_routes_small_linalg_operations_to_cpu(monkeypatch):
    # Given CUDA boxes and wrappers that record every solve/inverse input device.
    boxes = torch.tensor([
        [20.0, -8.0, 2.0, 4.0, 8.0, 3.0, 0.3, 6.0, -4.0],
    ], device='cuda')
    source_matrix = torch.diag(torch.tensor(
        [2.0, 2.0, 2.0, 1.0], device='cuda'
    ))
    target_matrix = torch.diag(torch.tensor(
        [0.5, 0.5, 0.5, 1.0], device='cuda'
    ))
    linalg_devices = []
    original_solve = torch.linalg.solve
    original_inv = torch.linalg.inv

    def checked_solve(matrix, right_hand_side):
        linalg_devices.append(matrix.device.type)
        return original_solve(matrix, right_hand_side)

    def checked_inv(matrix):
        linalg_devices.append(matrix.device.type)
        return original_inv(matrix)

    monkeypatch.setattr(torch.linalg, 'solve', checked_solve)
    monkeypatch.setattr(torch.linalg, 'inv', checked_inv)

    # When weak-view boxes are transformed into the strong view.
    transformed = transform_boxes_between_views(
        boxes, source_matrix=source_matrix, target_matrix=target_matrix
    )

    # Then MAGMA is bypassed while output placement and values are preserved.
    assert linalg_devices == ['cpu', 'cpu']
    assert transformed.device == boxes.device
    assert transformed.dtype == boxes.dtype
    torch.testing.assert_close(
        transformed.cpu(),
        torch.tensor([[5.0, -2.0, 0.5, 1.0, 2.0, 0.75, 0.3, 1.5, -1.0]]),
    )


def test_filter_predictions_builds_padded_native_targets_and_aligned_weights():
    # Given two batches containing overlaps, low scores, and invalid 9D boxes.
    pred_dicts = [
        {
            'pred_boxes': torch.tensor([
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 1.0, -1.0],
                [0.1, 0.1, 0.0, 2.0, 2.0, 2.0, 0.0, 2.0, -2.0],
                [5.0, 5.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                [float('nan'), 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0],
            ]),
            'pred_scores': torch.tensor([0.9, 0.8, 0.7, 0.99]),
            'pred_labels': torch.tensor([1, 1, 2, 1]),
        },
        {
            'pred_boxes': torch.tensor([
                [10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.1, 3.0, 4.0],
                [20.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.2, 5.0, 6.0],
                [30.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.3, 7.0, 8.0],
            ]),
            'pred_scores': torch.tensor([0.6, 0.95, 0.99]),
            'pred_labels': torch.tensor([1, 2, 2]),
        },
    ]

    # When class thresholds, per-class NMS, and output limits are applied.
    targets, confidence_weights, class_counts = filter_predictions_to_targets(
        pred_dicts,
        score_thresholds=[0.5, 0.75],
        nms_thresholds=[0.1, 0.1],
        nms_pre_maxsize=32,
        nms_post_maxsize=16,
    )

    # Then targets are padded 9D-box-plus-class rows with aligned confidences/counts.
    assert targets.shape == (2, 2, 10)
    assert confidence_weights.shape == (2, 2)
    assert class_counts.shape == (2, 2)
    torch.testing.assert_close(class_counts, torch.tensor([[1, 0], [1, 1]]))
    torch.testing.assert_close(targets[0, 0, :9], pred_dicts[0]['pred_boxes'][0])
    assert targets[0, 0, 9].item() == 1.0
    assert confidence_weights[0, 0].item() == pytest.approx(0.9)
    assert torch.count_nonzero(targets[0, 1]).item() == 0
    assert confidence_weights[0, 1].item() == 0.0
    assert set(targets[1, :, 9].tolist()) == {1.0, 2.0}
    assert sorted(confidence_weights[1].tolist()) == pytest.approx([0.6, 0.95])
    assert torch.isfinite(targets).all()
    assert (targets[:, :, 9] >= 0).all()


@pytest.mark.parametrize(
    ('field_name', 'is_gt'),
    [
        ('gt_boxes', True),
        ('gt_names', True),
        ('ground_truth_boxes', True),
        ('points', False),
        ('camera_imgs', False),
        ('lidar_aug_matrix', False),
    ],
)
def test_production_gt_field_filter_strips_only_target_annotations(field_name, is_gt):
    assert _is_gt_field(field_name) is is_gt


def test_production_scale_view_composes_sampled_scale_after_existing_matrix():
    tree = _function_tree(_ScaleViewBuilder.build)
    append_call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'append'
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 'lidar_aug_matrices'
    )

    composition = append_call.args[0]
    assert isinstance(composition, ast.BinOp)
    assert isinstance(composition.op, ast.MatMult)
    assert isinstance(composition.left, ast.Name)
    assert composition.left.id == 'sampled_matrix'
    assert isinstance(composition.right, ast.Subscript)
    matrix_index = _subscript_slice(composition.right)
    assert isinstance(matrix_index, ast.Name)
    assert matrix_index.id == 'batch_index'
    original_matrix = composition.right.value
    assert isinstance(original_matrix, ast.Subscript)
    assert isinstance(original_matrix.value, ast.Name)
    assert original_matrix.value.id == 'original_batch'
    assert getattr(_subscript_slice(original_matrix), 'value', None) == 'lidar_aug_matrix'


def test_production_evaluator_initializes_models_once_before_stream():
    tree = _function_tree(eval_cotta_one_epoch)
    initialization_lines = _call_lines(tree, 'initialize_cotta_models')
    stream_loop = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and [
            element.id for element in node.target.elts
            if isinstance(element, ast.Name)
        ] == ['batch_index', 'original_batch']
    )
    call_lines = {
        _attribute_name(node.func): node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    assert len(initialization_lines) == 1
    assert initialization_lines[0] < stream_loop.lineno
    assert call_lines['model.cpu'] < initialization_lines[0]
    assert initialization_lines[0] < call_lines['source_anchor.cpu']
    assert initialization_lines[0] < call_lines['teacher.to']
    assert initialization_lines[0] < call_lines['student.to']


def test_production_evaluator_rejects_distributed_execution_unconditionally():
    cfg = SimpleNamespace(TTA=SimpleNamespace(COTTA={}))
    args = SimpleNamespace(infer_time=False)

    with pytest.raises(RuntimeError, match='dist_test=False'):
        eval_cotta_one_epoch(cfg, args, None, None, None, None, dist_test=True)


def test_production_evaluator_rejects_inference_timing_mode():
    cfg = SimpleNamespace(TTA=SimpleNamespace(COTTA={}))
    args = SimpleNamespace(infer_time=True)

    with pytest.raises(RuntimeError, match='does not support infer_time'):
        eval_cotta_one_epoch(cfg, args, None, None, None, None)


def test_production_evaluator_accepts_only_adam_optimizer_name():
    cfg = SimpleNamespace(TTA=SimpleNamespace(COTTA={'OPTIMIZER': 'AdamW'}))
    args = SimpleNamespace(infer_time=False)

    with pytest.raises(RuntimeError, match='OPTIMIZER == Adam'):
        eval_cotta_one_epoch(cfg, args, None, None, None, None)


@pytest.mark.parametrize('optimizer_name', ['Adam', 'adam', 'ADAM'])
def test_production_evaluator_accepts_adam_case_insensitively(optimizer_name):
    cfg = SimpleNamespace(TTA=SimpleNamespace(COTTA={'OPTIMIZER': optimizer_name}))
    args = SimpleNamespace(infer_time=False)

    with pytest.raises(RuntimeError, match='requires a result_dir'):
        eval_cotta_one_epoch(cfg, args, None, None, None, None)


def test_production_evaluator_preserves_predict_then_adapt_update_order():
    tree = _function_tree(eval_cotta_one_epoch)
    teacher_lines = _call_lines(tree, 'teacher')
    annotation_line = _call_lines(tree, 'generate_prediction_dicts')[0]
    filter_line = _call_lines(tree, 'filter_predictions_to_targets')[0]
    student_line = _call_lines(tree, 'student')[0]
    backward_line = _call_lines(tree, 'backward')[0]
    optimizer_line = _call_lines(tree, 'step')[0]
    ema_line = _call_lines(tree, 'update_ema_teacher')[0]
    restore_line = _call_lines(tree, 'stochastic_restore')[0]

    assert teacher_lines[0] < annotation_line < teacher_lines[1] < filter_line
    assert filter_line < student_line
    assert student_line < backward_line < optimizer_line < ema_line < restore_line


def test_production_evaluator_releases_forward_trees_before_student_forward():
    tree = _function_tree(eval_cotta_one_epoch)
    deleted_names = {
        target.id: node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Delete)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    annotation_line = _call_lines(tree, 'generate_prediction_dicts')[0]
    filter_line = _call_lines(tree, 'filter_predictions_to_targets')[0]
    student_line = _call_lines(tree, 'student')[0]

    assert annotation_line < deleted_names['evaluation_batch'] < filter_line
    assert annotation_line < deleted_names['prediction_dicts'] < filter_line
    assert filter_line < deleted_names['weak_predictions'] < student_line
    assert filter_line < deleted_names['weak_batch'] < student_line
    assert filter_line < deleted_names['original_batch'] < student_line
