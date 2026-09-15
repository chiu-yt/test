import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from pcdet.tta_methods.sar import SAR, SARStepInput
from pcdet.tta_methods.sar_optimizer import SAM
from tools.eval_utils.eval_utils import eval_one_epoch
from tools.eval_utils.sar_eval_utils import (
    _build_adaptation_batch,
    eval_sar_one_epoch,
)


def _function_tree(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function)))


def _call_name(node):
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _calls(tree, name):
    return sorted(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.Call) and _call_name(node) == name),
        key=lambda node: node.lineno,
    )


def _has_early_exit(nodes):
    return any(isinstance(node, (ast.Return, ast.Continue)) for node in nodes)


def _ast_tokens(node):
    return {
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    } | {
        child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)
    } | {
        child.value for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return '%s.%s' % (parent, node.attr) if parent else node.attr
    return None


def _sar_cfg(**overrides):
    values = {'OPTIMIZER': 'SGD', 'STEPS': 1}
    values.update(overrides)
    return SimpleNamespace(TTA=SimpleNamespace(SAR=values))


def test_wrapper_contract_is_owned_by_sar_module():
    # Given the independent SAR wrapper imported through its production path.
    wrapper_module = SAR.__module__

    # When its public adaptation entry point is inspected.
    adaptation_entry = SAR.adapt

    # Then the wrapper has one named home and one explicit per-batch entry point.
    assert wrapper_module == 'pcdet.tta_methods.sar'
    assert callable(adaptation_entry)


def test_second_empty_rolls_back_without_computing_or_reporting_nan():
    class TinyDenseHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.1))

        def predict(self, _inputs):
            self.last_top_proposals = torch.tensor([[10]])
            return {'heatmap': self.weight * torch.zeros(1, 2, 1)}

    class TinyDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense_head = TinyDenseHead()

        def forward(self, batch):
            return self.dense_head.predict(batch), {}

    # Given detached selected prediction entropy and high-entropy adaptation logits.
    model = TinyDetector()
    optimizer = SAM(model.parameters(), torch.optim.SGD, lr=0.1, rho=0.05)
    sar = SAR(model, optimizer, {'RELIABLE_MARGIN_NORM': 0.4})
    sar.ema = 0.25
    before = model.dense_head.weight.detach().clone()
    step = SARStepInput(
        {}, torch.tensor([[0.01]]), torch.tensor([[10]]), 1.0, 7
    )

    # When one real SAR transaction reaches an empty second reliable set.
    result = sar.adapt(step)

    # Then rollback is exact and no NaN-derived state is constructed or reported.
    assert result.loss_second is None
    assert torch.equal(model.dense_head.weight, before)
    assert not any('old_p' in state for state in optimizer.state.values())
    assert sar.ema == 0.25
    assert result.recovered is False
    assert result.skip_reason == 'second_empty_or_nonfinite'


def test_evaluator_rejects_distributed_execution_unconditionally():
    # Given otherwise valid SAR startup configuration.
    cfg = _sar_cfg()
    args = SimpleNamespace(infer_time=False)

    # When distributed evaluation is requested.
    with pytest.raises(RuntimeError):
        eval_sar_one_epoch(cfg, args, None, None, None, None, dist_test=True)

    # Then rejection occurs before any detector, loader, or result path is used.


def test_evaluator_rejects_inference_timing_mode():
    # Given single-process SAR with inference timing enabled.
    cfg = _sar_cfg()
    args = SimpleNamespace(infer_time=True)

    # When evaluator startup validates the mode.
    with pytest.raises(RuntimeError):
        eval_sar_one_epoch(cfg, args, None, None, None, None)

    # Then rejection occurs before runtime resources are required.


@pytest.mark.parametrize(
    ('override', 'value'),
    [('STEPS', 0), ('STEPS', 2), ('OPTIMIZER', 'Adam')],
)
def test_evaluator_allows_exactly_one_step_and_sgd(override, value):
    # Given a SAR setting that violates one startup invariant.
    cfg = _sar_cfg(**{override: value})
    args = SimpleNamespace(infer_time=False)

    # When startup validates the adaptation protocol.
    with pytest.raises(RuntimeError):
        eval_sar_one_epoch(cfg, args, None, None, None, None)

    # Then invalid steps and non-SGD optimizers fail before runtime setup.


def test_adaptation_batch_is_shallow_and_excludes_all_annotations():
    # Given sensor inputs, calibration, identity metadata, and forbidden labels.
    retained = {
        key: [key]
        for key in (
            'points', 'images', 'camera_imgs', 'camera_intrinsics',
            'camera2lidar', 'img_aug_matrix', 'lidar_aug_matrix',
            'metadata', 'frame_id',
        )
    }
    forbidden = {
        key: [key]
        for key in (
            'gt_boxes', 'gt_names', 'gt_boxes2d', 'ground_truth_boxes',
            'annotations', 'annos', 'sample_annotation_tokens',
        )
    }
    original_batch = {**retained, **forbidden}

    # When the evaluator creates the adaptation-only batch view.
    adaptation_batch = _build_adaptation_batch(original_batch)

    # Then the mapping is new, sensor values are shared, and labels are absent.
    assert adaptation_batch is not original_batch
    assert set(adaptation_batch) == set(retained)
    assert all(adaptation_batch[key] is value for key, value in retained.items())
    assert not set(forbidden).intersection(adaptation_batch)


def test_first_empty_exits_before_perturbation_second_forward_or_state_updates():
    # Given the wrapper's one-batch adaptation state machine.
    tree = _function_tree(SAR.adapt)
    first_step_line = _calls(tree, 'first_step')[0].lineno
    guards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and node.lineno < first_step_line
        and 'first_selected' in _ast_tokens(node.test)
        and _has_early_exit(ast.walk(node))
    ]

    # When the first reliable set is empty.
    guard_nodes = list(ast.walk(guards[0]))
    guard_calls = {_call_name(node) for node in guard_nodes if isinstance(node, ast.Call)}

    # Then the guard exits without opening any SAR transaction or state update.
    assert guard_calls.isdisjoint({
        'first_step', 'model', 'update_sar_ema', 'should_recover_sar', 'reset',
    })


def test_second_empty_or_nonfinite_rolls_back_without_update_ema_or_recovery():
    # Given SAR after the first backward and SAM perturbation.
    tree = _function_tree(SAR.adapt)
    invalid_guards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and 'second_selected' in _ast_tokens(node.test)
        and 'isfinite' in _ast_tokens(node.test)
        and _calls(node, 'rollback')
    ]

    # When second-pass selection is empty or its entropy loss is non-finite.
    guard_nodes = list(ast.walk(invalid_guards[0]))
    guard_calls = {_call_name(node) for node in guard_nodes if isinstance(node, ast.Call)}
    guarded_losses = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'loss_second'
                for target in node.targets)
        and isinstance(node.value, ast.IfExp)
    ]

    # Then every perturbation rolls back and no base step or state update occurs.
    assert _has_early_exit(guard_nodes)
    assert len(guarded_losses) == 1
    assert not _calls(guarded_losses[0].body, 'mean')
    assert len(_calls(guarded_losses[0].orelse, 'mean')) == 1
    assert guard_calls.isdisjoint({
        'second_step', 'update_sar_ema', 'should_recover_sar', 'reset',
    })


def test_success_path_uses_nested_filters_then_updates_and_checks_recovery():
    # Given the wrapper's successful two-pass path.
    tree = _function_tree(SAR.adapt)
    backwards = _calls(tree, 'backward')
    filters = _calls(tree, 'normalized_reliable_mask')
    first_step = _calls(tree, 'first_step')[0]
    forwards = _calls(tree, 'model')
    second_step = _calls(tree, 'second_step')[0]
    ema = _calls(tree, 'update_sar_ema')[0]
    recovery = _calls(tree, 'should_recover_sar')[0]

    # When successful operations are read in execution order.
    ordered_lines = [
        filters[0].lineno, forwards[0].lineno, backwards[0].lineno,
        first_step.lineno, forwards[1].lineno, filters[1].lineno, backwards[1].lineno,
        second_step.lineno, ema.lineno, recovery.lineno,
    ]

    # Then SAR performs nested reliable filtering and updates only after pass two.
    assert ordered_lines == sorted(ordered_lines)
    assert len(filters) == 2
    assert len(forwards) == 2
    assert len(backwards) == 2


def test_recovery_restores_one_time_full_snapshot_and_clears_ema():
    # Given wrapper construction and its recovery operation.
    class_tree = _function_tree(SAR)
    init_tree = _function_tree(SAR.__init__)
    reset_tree = _function_tree(SAR.reset)
    snapshots = _calls(class_tree, 'snapshot_sar_state')
    restores = _calls(reset_tree, 'restore_sar_state')
    reset_assignments = [
        node for node in ast.walk(reset_tree)
        if isinstance(node, ast.Assign)
        and any(_dotted_name(target) == 'self.ema' for target in node.targets)
    ]

    # When the model-recovery contract is inspected.
    snapshot_args = [_dotted_name(arg) for arg in _calls(init_tree, 'snapshot_sar_state')[0].args]
    restore_args = [_dotted_name(arg) for arg in restores[0].args]

    # Then one source snapshot restores model plus complete SAM state and clears EMA.
    assert len(snapshots) == 1
    assert snapshot_args == ['self.model', 'self.optimizer']
    assert restore_args == [
        'self.model', 'self.optimizer', 'self.model_state', 'self.optimizer_state',
    ]
    assert len(reset_assignments) == 1
    assert isinstance(reset_assignments[0].value, ast.Constant)
    assert reset_assignments[0].value.value is None


def test_adaptation_source_has_no_forbidden_supervision_or_auxiliary_methods():
    # Given parsed identifiers from the adaptation implementation, excluding prose.
    tree = _function_tree(SAR.adapt)
    identifiers = {
        node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr.lower() for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    forbidden_fragments = (
        'pseudo', 'detector_loss', 'bbox', 'regression', 'iou',
        'spcra', 'rg_plm', 'sg_dfa',
    )
    forbidden_names = {'get_loss', 'get_training_loss', 'loss_bbox', 'loss_reg'}

    # When machine-consumed names and attributes are checked.
    violations = sorted(
        name for name in identifiers
        if name in forbidden_names
        or any(fragment in name for fragment in forbidden_fragments)
    )

    # Then SAR depends only on entropy and optimizer state, never forbidden signals.
    assert violations == []


def test_eval_dispatch_recognizes_enabled_sar_method():
    # Given the generic evaluator dispatch syntax tree.
    tree = _function_tree(eval_one_epoch)
    constants = {
        node.value.lower() for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    # When dispatch calls and routing constants are inspected.
    sar_calls = _calls(tree, 'eval_sar_one_epoch')
    sar_dispatches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and _calls(node, 'eval_sar_one_epoch')
        and all(token in _ast_tokens(node.test) for token in ('METHOD', 'SAR', 'ENABLED'))
    ]

    # Then enabled TTA can route METHOD=sar to the dedicated evaluator.
    assert 'sar' in constants
    assert len(sar_calls) == 1
    assert len(sar_dispatches) == 1
