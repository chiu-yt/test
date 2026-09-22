import copy

import numpy as np
import pytest
import torch

from figure6_runtime_harness import Config, batch, engine, mos_namespace
from test_figure6_runtime import collector


def test_first_prediction_when_aggregation_and_training_follow():
    """Given mutable detector outputs, when MOS optimizes then first detection owns Final."""
    observer = collector()
    instance, namespace, aggregate = engine(observer)
    inputs = batch()
    instance.optimize(inputs, data_already_on_gpu=True)
    record, = observer.finalize()
    assert instance.model.calls == 3
    assert aggregate.calls == 2
    assert record.stages['final_detection'].arrays['pred_boxes'][0, 0] == 1
    np.testing.assert_array_equal(record.stages['final_detection'].arrays['pred_labels'], [1, 2])
    current = record.stages['spcra.current_pre_update']
    aggregated = record.stages['spcra.aggregated_pseudo_source']
    assert current.status.owner == 'current_pre_update'
    assert aggregated.status.owner == 'aggregated_pseudo_source'
    np.testing.assert_array_equal(current.arrays['spcra_reliability'], [0., .2])
    assert current.arrays['pred_boxes'][0, 0] == 2
    assert aggregated.arrays['pred_boxes'][0, 0] == 12
    assert record.protocol['final_pseudo_source'] == 'aggregated_pseudo_source'
    before = record.stages['rg_plm_before.aggregated_pseudo_source'].arrays
    after = record.stages['rg_plm_after.aggregated_pseudo_source'].arrays
    np.testing.assert_allclose(before['gt_boxes'][:, 8], [.2, .8])
    np.testing.assert_allclose(after['gt_boxes'][:, 8], [.1, .48])
    np.testing.assert_array_equal(after['gt_boxes'][:, 7], [-1, 2])
    np.testing.assert_array_equal(record.stages['injection'].arrays['gt_boxes'], inputs['gt_boxes'][1].numpy())
    assert '_figure6_adapter_request' not in inputs
    assert namespace['NEW_PSEUDO_LABELS']['selected']['gt_boxes'][0, 7] == -1


def test_behavior_when_capture_off_matches_capture_on():
    """Given identical runtime state, when toggling capture then training is unchanged."""
    observer = collector()
    captured, captured_ns, captured_aggregate = engine(observer)
    plain, plain_ns, plain_aggregate = engine()
    plain.model.load_state_dict(captured.model.state_dict())
    plain_aggregate.load_state_dict(captured_aggregate.state_dict())
    inputs, control = batch(), batch()
    rng = torch.get_rng_state().clone()
    captured_result = captured.optimize(inputs, data_already_on_gpu=True)
    after_capture = torch.get_rng_state().clone()
    plain_result = plain.optimize(control, data_already_on_gpu=True)
    assert captured_result == plain_result
    assert torch.equal(rng, after_capture) and torch.equal(rng, torch.get_rng_state())
    assert captured.model.calls == plain.model.calls == 3
    assert captured_aggregate.calls == plain_aggregate.calls == 2
    for left, right in zip(captured.model.parameters(), plain.model.parameters()):
        assert (left.grad is None and right.grad is None) or torch.equal(left.grad, right.grad)
    for key, value in captured_ns['NEW_PSEUDO_LABELS']['selected'].items():
        np.testing.assert_equal(value, plain_ns['NEW_PSEUDO_LABELS']['selected'][key])
    assert set(inputs) == set(control)


@pytest.mark.parametrize('missing', [True, False])
def test_injection_skip_when_ground_truth_is_present(missing):
    """Given missing or empty pseudo labels, when injecting then original GT is not captured."""
    observer = collector()
    instance, namespace, _ = engine(observer)
    inputs = batch()
    original = inputs['gt_boxes']
    if not missing:
        namespace['NEW_PSEUDO_LABELS'].update({fid: {'gt_boxes': np.zeros((0, 9))} for fid in inputs['frame_id']})
    instance._inject_pseudo_labels(inputs)
    record, = observer.finalize()
    assert inputs['gt_boxes'] is original
    assert not record.stages['injection'].arrays
    assert record.stages['injection'].status.detail == ('missing_frame_ids' if missing else 'empty_pseudo_labels')


@pytest.mark.parametrize('weights', [True, False])
def test_injection_weights_when_actually_assigned(weights):
    """Given signed pseudo boxes, when injecting then normalized boxes and assigned weights are captured."""
    observer = collector()
    instance, namespace, _ = engine(observer)
    namespace['cfg'].SELF_TRAIN['HARD_PSEUDO_MINING'] = Config(ENABLED=weights)
    raw = np.zeros((2, 9), dtype=np.float32)
    raw[:, 7] = [-1, 2]
    for fid in ('other', 'selected'):
        namespace['NEW_PSEUDO_LABELS'][fid] = {'gt_boxes': raw.copy(), 'pseudo_cls_weights': np.array([.3, 1.]),
                                             'pseudo_reg_weights': np.array([.1, 1.])}
    inputs = batch()
    inputs['tta_pseudo_weights'] = torch.full((2, 2), 99.)
    instance._inject_pseudo_labels(inputs)
    record, = observer.finalize()
    arrays = record.stages['injection'].arrays
    np.testing.assert_array_equal(arrays['gt_boxes'][:, 9], [-1, 2])
    assert ('tta_pseudo_weights' in arrays) == weights
    if weights:
        np.testing.assert_allclose(arrays['tta_pseudo_weights'], [.3, 1.])
        np.testing.assert_allclose(arrays['tta_pseudo_reg_weights'], [.1, 1.])


def test_effective_values_when_memory_ensemble_changes_labels():
    """Given memory fusion after RG-PLM, when saving then effective capture reflects the merge."""
    observer = collector()
    namespace = mos_namespace()
    namespace['cfg'].SELF_TRAIN['MEMORY_ENSEMBLE'] = Config(ENABLED=True, NAME='fake')
    memory = namespace['memory_ensemble_utils']
    memory.fake = lambda: None

    def merge(previous, current, config, function):
        result = copy.deepcopy(current)
        result['gt_boxes'][:, 8] = .95
        return result

    memory.memory_ensemble = merge
    namespace['PSEUDO_LABELS']['selected'] = {}
    predictions = [{'pred_boxes': torch.ones(2, 7), 'pred_scores': torch.tensor([.2, .8]),
                    'pred_labels': torch.tensor([1, 2]), 'spcra_reliability': np.array([0., .2])} for _ in range(2)]
    namespace['save_pseudo_label_batch'](batch(), predictions, need_update=True, observer=observer)
    record, = observer.finalize()
    np.testing.assert_allclose(record.stages['rg_plm_after.current_pre_update'].arrays['gt_boxes'][:, 8], [.1, .48])
    np.testing.assert_allclose(record.stages['effective_pseudo.current_pre_update'].arrays['gt_boxes'][:, 8], [.95, .95])
