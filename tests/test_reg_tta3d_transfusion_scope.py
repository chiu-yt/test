import ast
import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = REPO_ROOT / 'pcdet' / 'tta_methods' / 'reg_tta3d.py'
EXPECTED_BRANCHES = ('center', 'height', 'dim', 'rot', 'vel')


def _tree():
    assert CORE_PATH.is_file(), 'Reg-TTA3D core module is missing: %s' % CORE_PATH
    return ast.parse(CORE_PATH.read_text(encoding='utf-8'), filename=str(CORE_PATH))


def _source(node):
    source = ast.get_source_segment(CORE_PATH.read_text(encoding='utf-8'), node)
    assert source is not None
    return source


def _function(name):
    matches = [
        node for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, '%s must be defined exactly once' % name
    return matches[0]


def _runtime_core():
    getattr(importlib.import_module('pytest'), 'importorskip')('torch')
    return importlib.import_module('pcdet.tta_methods.reg_tta3d')


def test_scope_static_contract_names_only_transfusion_regression_branches():
    # Given the planned student configuration boundary.
    function = _function('configure_reg_tta3d_student')

    # When its implementation is inspected without importing torch.
    source = _source(function)

    # Then all five regression branches are selected and classification paths are absent.
    assert all(branch in source for branch in EXPECTED_BRANCHES)
    assert 'prediction_head' in source
    for forbidden in (
            'heatmap_head', 'class_encoding', 'query', 'decoder', 'shared_conv',
            'backbone', 'image_backbone', 'vtransform', 'fuser'):
        assert forbidden not in source


def test_initialization_static_contract_separates_teacher_and_student():
    # Given the planned teacher/student initialization boundary.
    function = _function('initialize_reg_tta3d_models')

    # When clone, mode, and gradient operations are inspected.
    source = _source(function)

    # Then teacher/student are independent and teacher remains gradient-free.
    assert 'deepcopy' in source
    assert 'teacher' in source and 'student' in source
    assert 'requires_grad_' in source
    assert '.eval()' in source
    assert 'configure_reg_tta3d_student' in source


def test_runtime_scope_freezes_every_nonregression_parameter():
    # Given a detector-shaped module with regression and forbidden branches.
    core = _runtime_core()
    torch = importlib.import_module('torch')
    nn = importlib.import_module('torch.nn')

    class TinyPredictionHead(nn.Module):
        def __init__(self):
            super().__init__()
            for branch in EXPECTED_BRANCHES:
                setattr(self, branch, nn.Linear(1, 1))

    class TinyDenseHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.prediction_head = TinyPredictionHead()
            self.heatmap_head = nn.Linear(1, 1)
            self.decoder = nn.Linear(1, 1)

    class TinyDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone_2d = nn.Sequential(nn.BatchNorm1d(1), nn.Linear(1, 1))
            self.dense_head = TinyDenseHead()

    model = TinyDetector()

    # When Reg-TTA3D configures the student update scope.
    _, names, _, _ = core.configure_reg_tta3d_student(model)

    # Then only exact TransFusion regression prefixes are trainable and BN stays frozen.
    expected_prefixes = tuple(
        'dense_head.prediction_head.%s.' % branch for branch in EXPECTED_BRANCHES
    )
    actual_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert names == actual_names
    assert actual_names and all(name.startswith(expected_prefixes) for name in actual_names)
    assert model.backbone_2d[0].training is False
    assert torch.equal(model.backbone_2d[0].running_mean, torch.zeros(1))
