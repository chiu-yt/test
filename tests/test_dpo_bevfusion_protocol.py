import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DISPATCH_PATH = REPO_ROOT / 'tools' / 'eval_utils' / 'eval_utils.py'
EVALUATOR_PATH = REPO_ROOT / 'tools' / 'eval_utils' / 'dpo_bevfusion_eval_utils.py'
ADAPTER_PATH = REPO_ROOT / 'pcdet' / 'tta_methods' / 'dpo_bevfusion.py'
CONFIG_PATH = REPO_ROOT / 'tools' / 'cfgs' / 'nuscenes_models' / 'bevfusion_dpo.yaml'


def _required_text(path, purpose):
    assert path.is_file(), '%s is missing: %s' % (purpose, path)
    return path.read_text(encoding='utf-8')


def _required_tree(path, purpose):
    source = _required_text(path, purpose)
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError as error:
        raise AssertionError('%s is not valid Python: %s' % (purpose, error)) from error


def _function(tree, name):
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, 'function %s must exist exactly once' % name
    return matches[0]


def _adapt_method(tree):
    matches = [
        child for node in tree.body if isinstance(node, ast.ClassDef)
        for child in node.body
        if isinstance(child, ast.FunctionDef) and child.name == 'adapt'
    ]
    assert len(matches) == 1, 'DPO evaluator must define exactly one adapter.adapt method'
    return matches[0]


def _call_name(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _calls(node, name):
    return sorted(
        [child for child in ast.walk(node) if isinstance(child, ast.Call) and _call_name(child) == name],
        key=lambda child: child.lineno,
    )


def _assigned_name(node, call):
    matches = [
        target.id for assignment in ast.walk(node)
        if isinstance(assignment, ast.Assign) and call in ast.walk(assignment.value)
        for target in assignment.targets if isinstance(target, ast.Name)
    ]
    assert len(matches) == 1, 'call result must be assigned to one named value'
    return matches[0]


def _det_anno_writes(function):
    writes = [
        node for node in ast.walk(function)
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name) and node.target.id == 'det_annos'
        ) or (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'det_annos'
            and node.func.attr in {'append', 'extend'}
        )
    ]
    return sorted(writes, key=lambda node: node.lineno)


def _scalar(raw_value):
    value = raw_value.split(' #', 1)[0].strip()
    lowered = value.lower()
    if lowered in {'true', 'false'}:
        return lowered == 'true'
    if lowered in {'null', 'none', '~'}:
        return None
    if value[:1] in {'\'', '"', '[', '('}:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _yaml_contract():
    source = _required_text(CONFIG_PATH, 'planned DPO-BEVFusion config')
    stack = []
    scalars = {}
    sections = set()
    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('- '):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        key, separator, raw_value = stripped.partition(':')
        assert separator, 'invalid YAML mapping line %d in %s' % (line_number, CONFIG_PATH)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = tuple(item[1] for item in stack) + (key.strip(),)
        if raw_value.strip():
            scalars[path] = _scalar(raw_value)
        else:
            sections.add(path)
            stack.append((indent, key.strip()))
    return scalars, sections


def _value(scalars, path):
    assert path in scalars, 'DPO config must define %s' % '.'.join(path)
    return scalars[path]


def test_eval_dispatch_routes_enabled_dpo_bevfusion():
    # Given the generic evaluation dispatcher without importing OpenPCDet.
    tree = _required_tree(DISPATCH_PATH, 'generic evaluator dispatch')
    evaluator = _function(tree, 'eval_one_epoch')

    # When DPO dispatch calls and guards are inspected.
    calls = _calls(evaluator, 'eval_dpo_bevfusion_one_epoch')
    dispatches = [
        node for node in ast.walk(evaluator)
        if isinstance(node, ast.If) and _calls(node, 'eval_dpo_bevfusion_one_epoch')
    ]

    # Then METHOD and method-specific ENABLED jointly select one evaluator.
    assert len(calls) == 1, 'eval_one_epoch must call eval_dpo_bevfusion_one_epoch exactly once'
    assert len(dispatches) == 1, 'DPO-BEVFusion dispatch must have one dedicated guard'
    guard = ast.unparse(dispatches[0].test)
    assert "'dpo_bevfusion'" in guard, 'dispatch must match TTA.METHOD=dpo_bevfusion'
    assert all(token in guard for token in ('TTA', 'METHOD', 'DPO_BEVFUSION', 'ENABLED')), (
        'dispatch must require TTA.ENABLED and TTA.DPO_BEVFUSION.ENABLED'
    )


def test_forward_a_is_the_only_owner_of_official_evaluation_output():
    # Given the planned DPO evaluator syntax tree.
    tree = _required_tree(EVALUATOR_PATH, 'planned DPO-BEVFusion evaluator')
    evaluator = _function(tree, 'eval_dpo_bevfusion_one_epoch')

    # When official prediction, detachment, annotation, and adaptation sites are ordered.
    forwards = _calls(evaluator, 'model')
    detaches = _calls(evaluator, '_detach_tensor_tree')
    annotations = _calls(evaluator, 'generate_prediction_dicts')
    adaptations = _calls(evaluator, 'adapt')
    writes = _det_anno_writes(evaluator)

    # Then exactly one detached pre-update Forward A is recorded before adaptation.
    assert len(forwards) == 1, 'evaluator must execute exactly one official Forward A'
    assert len(detaches) == 1, 'Forward A predictions must be detached exactly once'
    assert len(annotations) == 1, 'only Forward A may call generate_prediction_dicts'
    assert len(writes) == 1, 'only Forward A annotations may feed det_annos'
    assert len(adaptations) == 1, 'each batch must invoke adapter.adapt exactly once'
    assert forwards[0].lineno < detaches[0].lineno < annotations[0].lineno < writes[0].lineno < adaptations[0].lineno
    detached_name = _assigned_name(evaluator, detaches[0])
    annotation_name = _assigned_name(evaluator, annotations[0])
    assert detached_name in ast.unparse(annotations[0]), 'generate_prediction_dicts must consume detached Forward A output'
    assert annotation_name in ast.unparse(writes[0]), 'det_annos must consume only Forward A annotations'


def test_adapter_locks_four_forwards_two_backwards_and_one_final_update():
    # Given evaluator Forward A and the adapter's B/C/D adaptation transaction.
    tree = _required_tree(EVALUATOR_PATH, 'planned DPO-BEVFusion evaluator')
    adapter_tree = _required_tree(ADAPTER_PATH, 'planned DPO-BEVFusion adapter')
    evaluator = _function(tree, 'eval_dpo_bevfusion_one_epoch')
    adaptation = _adapt_method(adapter_tree)

    # When model, backward, and optimizer-step sites are counted and ordered.
    evaluator_forwards = _calls(evaluator, 'model')
    adaptation_forwards = _calls(adaptation, 'model')
    backwards = _calls(adaptation, 'backward')
    updates = _calls(adaptation, 'step')

    # Then A/B/C/D markers describe four forwards, B/D backward, and one post-D update.
    identifiers = {
        node.id.lower() for syntax_tree in (tree, adapter_tree) for node in ast.walk(syntax_tree) if isinstance(node, ast.Name)
    } | {
        node.attr.lower() for syntax_tree in (tree, adapter_tree) for node in ast.walk(syntax_tree) if isinstance(node, ast.Attribute)
    }
    for marker in ('forward_a', 'forward_b', 'forward_c', 'forward_d'):
        assert any(marker in identifier for identifier in identifiers), 'missing protocol marker %s' % marker
    assert len(evaluator_forwards) == 1, 'Forward A must occur once outside adaptation'
    assert len(adaptation_forwards) == 3, 'adapter.adapt must execute Forward B, C, and D exactly once'
    assert len(backwards) == 2, 'only clean B and refined D losses may call backward'
    assert len(updates) == 1, 'DPO adaptation must perform one update per batch'
    order = [
        adaptation_forwards[0].lineno, backwards[0].lineno,
        adaptation_forwards[1].lineno, adaptation_forwards[2].lineno,
        backwards[1].lineno, updates[0].lineno,
    ]
    assert order == sorted(order), 'required order is B/backward, C, D/backward, then sole update'


def test_adaptation_batch_strips_every_ground_truth_and_annotation_key():
    # Given the planned adaptation-batch helper.
    tree = _required_tree(EVALUATOR_PATH, 'planned DPO-BEVFusion evaluator')
    helper = _function(tree, '_build_adaptation_batch')

    # When its key-filter predicates are inspected.
    source = ast.unparse(helper)
    literals = {
        node.value.lower() for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    # Then gt_boxes, every gt_* field, ground-truth aliases, and annotations are removed.
    assert "startswith('gt_')" in source, 'adaptation batch must remove gt_boxes and every gt_* key'
    assert 'ground_truth' in literals, 'adaptation batch must remove ground_truth key variants'
    assert {'gt', 'annotations', 'annos', 'sample_annotation_tokens'} <= literals, (
        'adaptation batch must explicitly exclude annotation keys'
    )


def test_evaluator_rejects_distributed_timing_and_non_single_step_modes():
    # Given startup guards in the planned evaluator.
    tree = _required_tree(EVALUATOR_PATH, 'planned DPO-BEVFusion evaluator')
    evaluator = _function(tree, 'eval_dpo_bevfusion_one_epoch')
    guards = [
        node for node in ast.walk(evaluator)
        if isinstance(node, ast.If) and any(isinstance(child, ast.Raise) for child in ast.walk(node))
    ]
    tests = [ast.unparse(node.test) for node in guards]

    # When unsupported modes and adaptation cardinality are inspected.
    assert any(test == 'dist_test' for test in tests), 'DPO evaluator must reject distributed execution'
    assert any('infer_time' in test for test in tests), 'DPO evaluator must reject infer_time mode'
    assert any('STEPS' in test and '1' in test for test in tests), 'DPO evaluator must require one step per batch'


def test_config_uses_neutral_clean_camera_density_s5_protocol():
    # Given the dedicated DPO-BEVFusion config without a YAML dependency.
    scalars, _ = _yaml_contract()

    # When its base, corruption, and competing-method settings are resolved as text.
    assert _value(scalars, ('_BASE_CONFIG_',)) == 'cfgs/nuscenes_models/bevfusion.yaml'
    assert _value(scalars, ('DATA_CONFIG', 'CAMERA_CONFIG', 'USE_CAMERA')) is True
    lidar = ('DATA_CONFIG', 'CORRUPTION', 'LIDAR_SPARSITY')
    assert _value(scalars, lidar + ('ENABLED',)) is True
    assert _value(scalars, lidar + ('MODE',)) == 'density_dec_global'
    assert _value(scalars, lidar + ('SEVERITY',)) == 5
    for corruption in ('LIDAR_FOG', 'IMAGE_FOG', 'IMAGE_STYLE', 'IMAGE_GEOMETRY'):
        path = ('DATA_CONFIG', 'CORRUPTION', corruption, 'ENABLED')
        assert _value(scalars, path) is False, '%s must stay disabled' % corruption

    # Then DPO alone is enabled on the neutral BEVFusion profile.
    assert _value(scalars, ('TTA', 'ENABLED')) is True
    assert _value(scalars, ('TTA', 'METHOD')) == 'dpo_bevfusion'
    assert _value(scalars, ('TTA', 'DPO_BEVFUSION', 'ENABLED')) is True
    assert _value(scalars, ('TTA', 'DPO_BEVFUSION', 'STEPS')) == 1
    for method in ('TENT', 'COTTA', 'SAR', 'DPO_MATCHER', 'SPCRA', 'FREEZE'):
        assert _value(scalars, ('TTA', method, 'ENABLED')) is False, '%s must stay disabled' % method


def test_config_defaults_to_paper_profile_and_keeps_repo_profile_explicit():
    # Given paper-faithful and repository-reference DPO profiles.
    scalars, sections = _yaml_contract()
    root = ('TTA', 'DPO_BEVFUSION')
    paper = root + ('PROFILES', 'DPO_PAPER')
    repo = root + ('PROFILES', 'DPO_REPO')

    # When profile selection and global pseudo-label thresholds are inspected.
    assert _value(scalars, root + ('PROFILE',)) == 'dpo_paper'
    assert paper in sections, 'DPO config must define a PAPER profile'
    assert repo in sections, 'DPO config must retain an explicit REPO profile'
    assert _value(scalars, paper + ('SCORE_THRESH',)) == 0.5
    assert _value(scalars, paper + ('NEG_THRESH',)) == 0.2
    assert _value(scalars, paper + ('SCORE_THRESHOLDS',)) == [0.5] * 10
    assert _value(scalars, paper + ('NEG_THRESHOLDS',)) == [0.2] * 10

    # Then paper mode has no class-tuned thresholds or invented early cutoff.
    paper_values = {path: value for path, value in scalars.items() if path[:len(paper)] == paper}
    cutoff_disabled = any(
        value is False and ('CUTOFF' in path[-1] or 'CUTOFF' in path[:-1])
        for path, value in paper_values.items()
    )
    assert cutoff_disabled, 'PAPER profile must explicitly disable cutoff'
    assert _value(scalars, paper + ('C_STOP',)) is None, 'PAPER C_STOP must remain null'
    assert isinstance(paper_values[paper + ('SCORE_THRESH',)], float), 'SCORE_THRESH must be one global scalar'
    assert isinstance(paper_values[paper + ('NEG_THRESH',)], float), 'NEG_THRESH must be one global scalar'
