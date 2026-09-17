import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DISPATCH_PATH = REPO_ROOT / 'tools' / 'eval_utils' / 'eval_utils.py'
EVALUATOR_PATH = REPO_ROOT / 'tools' / 'eval_utils' / 'reg_tta3d_eval_utils.py'
CONFIG_PATH = REPO_ROOT / 'tools' / 'cfgs' / 'nuscenes_models' / 'bevfusion_reg_tta3d.yaml'
TRANSFUSION_PATH = REPO_ROOT / 'pcdet' / 'models' / 'dense_heads' / 'transfusion_head.py'


def _parse(path):
    assert path.is_file(), 'Required Reg-TTA3D file is missing: %s' % path
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _source(path, node):
    source = ast.get_source_segment(path.read_text(encoding='utf-8'), node)
    assert source is not None
    return source


def _function(tree, name):
    matches = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, '%s must be defined exactly once' % name
    return matches[0]


def _method(tree, class_name, method_name):
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(classes) == 1, '%s must be defined exactly once' % class_name
    methods = [
        node for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    assert len(methods) == 1, '%s.%s must be defined exactly once' % (
        class_name, method_name
    )
    return methods[0]


def _call_positions(function, terminal_name):
    positions = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, 'id', getattr(node.func, 'attr', None))
        if name == terminal_name:
            positions.append((node.lineno, name))
    return sorted(positions)


def test_dispatch_selects_independent_reg_tta3d_evaluator():
    # Given the shared evaluator dispatcher.
    function = _function(_parse(DISPATCH_PATH), 'eval_one_epoch')

    # When its routing-bearing source is inspected.
    source = _source(DISPATCH_PATH, function)

    # Then Reg-TTA3D has an explicit gated lazy route.
    assert "'reg_tta3d'" in source
    assert "'REG_TTA3D'" in source
    assert 'reg_tta3d_eval_utils.eval_reg_tta3d_one_epoch' in source


def test_evaluator_records_preupdate_teacher_prediction_before_adaptation():
    # Given the independent online evaluator.
    function = _function(_parse(EVALUATOR_PATH), 'eval_reg_tta3d_one_epoch')

    # When prediction ownership and update calls are ordered by source line.
    prediction_calls = _call_positions(function, 'generate_prediction_dicts')
    adaptation_calls = _call_positions(function, 'adapt')

    # Then the sole official prediction is recorded before the online update.
    assert len(prediction_calls) == 1
    assert len(adaptation_calls) == 1
    assert prediction_calls[0][0] < adaptation_calls[0][0]
    source = _source(EVALUATOR_PATH, function)
    assert 'detach' in source
    assert 'dist_test' in source and 'infer_time' in source and 'STEPS' in source


def test_transfusion_query_capture_is_explicit_and_predecode():
    # Given the TransFusion forward boundary.
    function = _method(_parse(TRANSFUSION_PATH), 'TransFusionHead', 'forward')

    # When the optional query capture and decoder call are ordered.
    source = _source(TRANSFUSION_PATH, function)

    # Then Reg-TTA3D clones query predictions before get_bboxes mutates center coordinates.
    assert 'reg_tta3d_capture_queries' in source
    assert 'reg_tta3d_query_predictions' in source
    assert '.clone()' in source
    assert source.index('reg_tta3d_query_predictions') < source.index('get_bboxes')


def test_adaptation_batch_removes_target_annotations():
    # Given the adaptation-batch boundary.
    function = _function(_parse(EVALUATOR_PATH), '_build_adaptation_batch')

    # When its explicit exclusion policy is inspected.
    source = _source(EVALUATOR_PATH, function)

    # Then target annotations cannot cross into adaptation.
    for key in (
            'gt_boxes', 'gt_names', 'ground_truth', 'annotations', 'annos',
            'sample_annotation_tokens'):
        assert key in source


def test_config_is_neutral_single_gpu_density_s5():
    # Given the canonical Reg-TTA3D config.
    assert CONFIG_PATH.is_file(), 'Reg-TTA3D config is missing: %s' % CONFIG_PATH

    # When dependency-free routing and protocol tokens are read.
    config = CONFIG_PATH.read_text(encoding='utf-8')

    # Then it selects the neutral BEVFusion S5 protocol and paper-defined optimizer values.
    for token in (
            '_BASE_CONFIG_: cfgs/nuscenes_models/bevfusion.yaml',
            'METHOD: reg_tta3d', 'REG_TTA3D:', 'ENABLED: True', 'STEPS: 1',
            'OPTIMIZER: Adam', 'LR: 0.001', 'MODE: density_dec_global',
            'SEVERITY: 5', 'ALLOW_DDP: False',
            'POINT_CLOUD_RANGE: [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]'):
        assert token in config
    assert 'bevfusion_mos.yaml' not in config


def test_reg_tta3d_sources_do_not_route_through_other_adaptation_methods():
    # Given all independent Reg-TTA3D production sources that exist.
    paths = [
        REPO_ROOT / 'pcdet' / 'tta_methods' / 'reg_tta3d.py',
        REPO_ROOT / 'pcdet' / 'tta_methods' / 'reg_tta3d_utils.py',
        REPO_ROOT / 'pcdet' / 'tta_methods' / 'reg_tta3d_geometry.py',
        REPO_ROOT / 'pcdet' / 'tta_methods' / 'reg_tta3d_npg.py',
        EVALUATOR_PATH,
    ]
    assert all(path.is_file() for path in paths)

    # When their machine-consumed imports and identifiers are combined.
    source = '\n'.join(path.read_text(encoding='utf-8').lower() for path in paths)

    # Then no pre-existing adaptation route is reused.
    for forbidden in (
            'spcra', 'rg_plm', 'sg_dfa', 'self_train', 'mos-main',
            'reliability.py', 'memory_ensemble', 'v2x', 'roi_pool'):
        assert forbidden not in source
