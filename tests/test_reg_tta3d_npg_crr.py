import ast
import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / 'pcdet' / 'tta_methods' / 'reg_tta3d_utils.py'


def _tree():
    assert UTILS_PATH.is_file(), 'Reg-TTA3D utility module is missing: %s' % UTILS_PATH
    return ast.parse(UTILS_PATH.read_text(encoding='utf-8'), filename=str(UTILS_PATH))


def _definition(tree, name):
    matches = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, '%s must be defined exactly once' % name
    return matches[0]


def _function(tree, name):
    definition = _definition(tree, name)
    assert isinstance(definition, ast.FunctionDef), '%s must be a function' % name
    return definition


def _runtime_utils():
    getattr(importlib.import_module('pytest'), 'importorskip')('torch')
    return importlib.import_module('pcdet.tta_methods.reg_tta3d_utils')


def test_npg_static_contract_locks_equation_threshold_and_query_scope():
    # Given the planned dependency-free Reg-TTA3D utility source.
    tree = _tree()
    reliability = _function(tree, 'compute_npg_reliability')
    deletion = _function(tree, 'npg_deletion_mask')
    perturbation = _function(tree, 'perturb_npg_query_boxes')
    noise_config = _definition(tree, 'NPGNoiseConfig')
    cbu = _function(tree, 'update_cbu_regression_teacher')

    # When the NPG definitions are inspected without importing torch.
    reliability_source = ast.unparse(reliability)
    deletion_source = ast.unparse(deletion)
    perturbation_source = ast.unparse(perturbation)
    config_source = ast.unparse(noise_config)
    argument_names = {argument.arg for argument in perturbation.args.args}

    # Then Eq. (2), strict R > TAU deletion, and query-box-only noise are explicit.
    for token in ('score', 'score_noisy', 'iou3d', 'eps', 'abs', 'max'):
        assert token in reliability_source
    assert '1.5' in deletion_source
    assert '>' in deletion_source
    config_fields = {
        node.target.id for node in noise_config.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert {'distribution', 'dimension_magnitude', 'yaw_magnitude', 'eps'} <= config_fields
    assert '3:6' in perturbation_source and '6' in perturbation_source
    assert not ({'points', 'images', 'camera_imgs', 'features', 'labels', 'scores', 'batch_dict'} & argument_names)
    for forbidden in ('points', 'images', 'camera_imgs', 'spatial_features', 'pred_labels', 'pred_scores'):
        assert forbidden not in perturbation_source
    assert all(token in ast.unparse(cbu) for token in ('alpha', '0.99', '0.999', 'regression'))


def test_npg_runtime_equation_and_query_box_perturbation_scope():
    # Given query boxes and a deliberately non-default, configurable NPG noise policy.
    utils = _runtime_utils()
    torch = importlib.import_module('torch')
    boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25]])
    config = utils.NPGNoiseConfig(
        distribution='uniform', dimension_magnitude=0.25, yaw_magnitude=0.5, eps=1e-4,
    )

    # When reliability and a noisy query-box prediction are computed.
    reliability = utils.compute_npg_reliability(
        torch.tensor([0.8, 0.2]), torch.tensor([0.5, 0.2]), torch.tensor([0.6, 0.0]), config.eps,
    )
    noisy_boxes = utils.perturb_npg_query_boxes(
        boxes, config, torch.Generator().manual_seed(7),
    )

    # Then R is exactly 1 - abs(score - score_noisy) / max(IoU3D, eps), and only l/w/h/yaw move.
    torch.testing.assert_close(
        reliability,
        torch.tensor([1.0 - 0.3 / 0.6, 1.0]),
    )
    torch.testing.assert_close(noisy_boxes[:, :3], boxes[:, :3])
    assert not torch.equal(noisy_boxes[:, 3:7], boxes[:, 3:7])


def test_npg_runtime_deletion_uses_strict_default_tau():
    # Given reliability values around the default threshold.
    utils = _runtime_utils()
    torch = importlib.import_module('torch')
    reliability = torch.tensor([1.4999, 1.5, 1.5001])

    # When NPG selects predictions for deletion without an override.
    deleted = utils.npg_deletion_mask(reliability)

    # Then only R > 1.5 is deleted; equality remains retained.
    assert torch.equal(deleted, torch.tensor([False, False, True]))


def test_crr_static_contract_is_dimension_only_with_paper_hyperparameters():
    # Given the planned CRR definition.
    tree = _tree()
    crr = _definition(tree, 'compute_crr_dimension_loss')

    # When its source is inspected without importing runtime dependencies.
    source = ast.unparse(crr)

    # Then the top ratio, margin, and weight defaults are fixed while geometry is l/w/h only.
    assert all(value in source for value in ('0.2', '0.1', '1.0'))
    assert '3:6' in source
    assert 'labels' in source and 'mean' in source and 'square' in source
    assert 'detach' in source
    assert all(token not in source for token in ('[:, :3]', '[:, 6]', 'yaw', 'center'))


def test_crr_runtime_uses_classwise_high_score_mean_for_low_score_sizes_only():
    # Given two classes with high-score size anchors and lower-score student boxes.
    utils = _runtime_utils()
    torch = importlib.import_module('torch')
    boxes = torch.tensor([
        [0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0],
        [99.0, -40.0, 3.0, 2.0, 3.0, 4.0, 2.5],
        [1.0, 1.0, 1.0, 5.0, 6.0, 7.0, 0.2],
        [8.0, 9.0, 2.0, 5.0, 6.0, 7.0, -2.0],
    ])
    scores = torch.tensor([0.9, 0.1, 0.8, 0.2])
    labels = torch.tensor([1, 1, 2, 2])

    # When CRR is evaluated before and after changing only low-score dimensions.
    unchanged_loss = utils.compute_crr_dimension_loss(boxes, scores, labels)
    boxes[1, 3:6] += 0.2
    changed_loss = utils.compute_crr_dimension_loss(boxes, scores, labels)

    # Then centers/yaw contribute nothing, while low-score l/w/h beyond p are constrained.
    torch.testing.assert_close(unchanged_loss, torch.zeros_like(unchanged_loss))
    assert changed_loss.item() > 0.0


def test_crr_runtime_stops_gradient_through_high_confidence_size_anchor():
    # Given one high-confidence anchor and one inconsistent low-confidence box.
    utils = _runtime_utils()
    torch = importlib.import_module('torch')
    boxes = torch.tensor([
        [0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0],
    ], requires_grad=True)

    # When class-wise CRR backpropagates from the low-confidence box.
    loss = utils.compute_crr_dimension_loss(
        boxes, torch.tensor([0.9, 0.1]), torch.tensor([1, 1])
    )
    loss.backward()

    # Then supervision flows only into the lower-confidence dimensions.
    torch.testing.assert_close(boxes.grad[0, 3:6], torch.zeros(3))
    assert bool((boxes.grad[1, 3:6] != 0).all().item())


def test_cbu_runtime_validates_alpha_and_updates_only_regression_branches():
    # Given teacher and student modules with regression and classification branches at distinct values.
    utils = _runtime_utils()
    torch = importlib.import_module('torch')
    nn = importlib.import_module('torch.nn')
    pytest = importlib.import_module('pytest')

    class TinyPredictionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.dim = nn.Linear(1, 1, bias=False)

    class TinyDenseHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.prediction_head = TinyPredictionHead()
            self.heatmap_head = nn.Linear(1, 1, bias=False)

    class TinyTeacherStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense_head = TinyDenseHead()

    teacher = TinyTeacherStudent()
    student = TinyTeacherStudent()
    with torch.no_grad():
        teacher.dense_head.prediction_head.dim.weight.fill_(1.0)
        teacher.dense_head.heatmap_head.weight.fill_(2.0)
        student.dense_head.prediction_head.dim.weight.fill_(5.0)
        student.dense_head.heatmap_head.weight.fill_(9.0)

    # When CBU updates at the lower valid alpha boundary and receives invalid alpha values.
    utils.update_cbu_regression_teacher(teacher, student, alpha=0.99)

    # Then only regression uses EMA and alpha remains constrained to [0.99, 0.999].
    torch.testing.assert_close(
        teacher.dense_head.prediction_head.dim.weight, torch.tensor([[1.04]])
    )
    torch.testing.assert_close(teacher.dense_head.heatmap_head.weight, torch.tensor([[2.0]]))
    with getattr(pytest, 'raises')(ValueError):
        utils.update_cbu_regression_teacher(teacher, student, alpha=0.989)
    with getattr(pytest, 'raises')(ValueError):
        utils.update_cbu_regression_teacher(teacher, student, alpha=0.9991)
