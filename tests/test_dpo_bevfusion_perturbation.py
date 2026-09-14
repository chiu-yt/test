import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PERTURB_PATH = REPO_ROOT / 'pcdet' / 'tta_methods' / 'dpo_bevfusion_perturb.py'


def _tree():
    assert PERTURB_PATH.is_file(), 'DPO perturbation utilities are missing: %s' % PERTURB_PATH
    return ast.parse(PERTURB_PATH.read_text(encoding='utf-8'), filename=str(PERTURB_PATH))


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


def test_weight_epsilon_uses_one_global_trainable_gradient_norm():
    # Given all eligible parameter gradients.
    source = ast.unparse(_function('compute_parameter_epsilons'))

    # When the perturbation vector is built.
    assert 'rho_w' in source
    assert 'torch.stack' in source
    assert 'torch.norm' in source or 'vector_norm' in source
    assert '1e-12' in source

    # Then every epsilon shares one denominator and no parameter magnitude scaling exists.
    assert '* scale' in source
    assert 'adaptive' not in source


def test_feature_epsilon_normalizes_the_entire_batched_fused_gradient():
    # Given the retained gradient of one batched fused BEV tensor.
    source = ast.unparse(_function('compute_feature_epsilon'))

    # When epsilon_z is computed.
    assert 'rho_z' in source and '1e-12' in source
    assert 'grad' in source
    assert 'torch.norm' in source or 'vector_norm' in source

    # Then no sample, channel, camera, or modality-specific loop changes normalization.
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(_function('compute_feature_epsilon')))


def test_parameter_transaction_restores_before_the_only_optimizer_step():
    # Given the dual-perturbation parameter transaction.
    apply_method = _method('DPOParameterPerturbation', 'apply')
    restore_method = _method('DPOParameterPerturbation', 'restore')
    step_method = _method('DPOParameterPerturbation', 'step')

    # When lifecycle methods are inspected.
    assert 'old_parameters' in ast.unparse(apply_method)
    assert 'copy_' in ast.unparse(restore_method)
    step_source = ast.unparse(step_method)

    # Then base values are restored before ordinary SGD consumes D gradients.
    assert step_source.index('restore') < step_source.index('optimizer.step')


def test_fused_bev_scope_uses_a_forward_hook_not_method_replacement():
    # Given the scoped feature capture/injection resource.
    enter = _method('DPOFusedBEVPerturbation', '__enter__')
    hook = _method('DPOFusedBEVPerturbation', '_hook')
    exit_method = _method('DPOFusedBEVPerturbation', '__exit__')

    # When hook registration and cleanup are inspected.
    assert 'register_forward_hook' in ast.unparse(enter)
    assert '.remove()' in ast.unparse(exit_method)
    hook_source = ast.unparse(hook)

    # Then only the post-ConvFuser fused spatial_features tensor is captured/replaced.
    assert "['spatial_features']" in hook_source
    for forbidden in ('spatial_features_img', 'camera_imgs', "['points']", "['voxels']"):
        assert forbidden not in hook_source
    assert '.forward =' not in ast.unparse(_tree())


def test_feature_hook_reuses_one_detached_epsilon_for_c_and_d():
    # Given capture and perturb modes in the hook object.
    source = ast.unparse(_class('DPOFusedBEVPerturbation'))

    # When epsilon is stored for later forwards.
    assert 'epsilon_z' in source
    assert '.detach()' in source
    assert 'retain_grad' in source

    # Then capture is clean and perturbation is a single fused residual addition.
    assert "batch_dict['spatial_features'] + self.epsilon_z" in source


def test_nonfinite_perturbations_fail_closed_and_cleanup_is_finally_safe():
    # Given perturbation construction and the adapter transaction API.
    parameter_source = ast.unparse(_function('compute_parameter_epsilons'))
    feature_source = ast.unparse(_function('compute_feature_epsilon'))
    class_source = ast.unparse(_class('DPOParameterPerturbation'))

    # When non-finite gradients or transaction exits occur.
    assert 'isfinite' in parameter_source
    assert 'isfinite' in feature_source
    assert '__exit__' in class_source

    # Then invalid epsilon construction has an explicit no-update result.
    assert 'None' in parameter_source
    assert 'None' in feature_source
