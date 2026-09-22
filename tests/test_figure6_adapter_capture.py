import copy

import numpy as np
import pytest
import torch

from figure6_runtime_harness import adapter
from test_figure6_runtime import collector
from pcdet.utils.figure6_schema import ArtifactError, StageState


def test_output_and_gradients_when_capture_enabled():
    """Given identical adapters, when capturing then outputs, RNG and gradients match."""
    plain = adapter()
    captured = copy.deepcopy(plain)
    source = torch.arange(16.).reshape(2, 2, 2, 2)
    first = source.clone().requires_grad_()
    second = source.clone().requires_grad_()
    density = torch.arange(8.).reshape(2, 1, 2, 2)
    observer = collector()
    batch = {'spatial_features': second, 'tta_density_map': density}
    rng = torch.get_rng_state().clone()
    expected = plain({'spatial_features': first, 'tta_density_map': density})['spatial_features']
    with observer.first_forward(batch):
        actual = captured(batch)['spatial_features']
        assert set(batch['_figure6_adapter_request'].responses) == {1}
    expected.sum().backward()
    actual.sum().backward()
    assert torch.equal(expected, actual)
    assert torch.equal(first.grad, second.grad)
    for left, right in zip(plain.parameters(), captured.parameters()):
        assert (left.grad is None and right.grad is None) or torch.equal(left.grad, right.grad)
    assert torch.equal(rng, torch.get_rng_state())
    record, = observer.finalize()
    arrays = record.stages['sg_dfa'].arrays
    np.testing.assert_array_equal(arrays['delta'], (actual - second)[1].detach().numpy())
    assert arrays['shared_gate'].shape == (2, 2, 2)
    assert arrays['density_map'].shape == (1, 2, 2)
    assert arrays['residual_scale'].shape == ()
    assert not arrays['delta'].flags.writeable
    assert all(isinstance(value, np.ndarray) for value in arrays.values())
    assert '_figure6_adapter_request' not in batch


def test_first_response_when_later_forwards_run():
    """Given one armed forward, when later inputs change then first response survives."""
    module = adapter()
    observer = collector()
    batch = {'spatial_features': torch.ones(2, 2, 2, 2), 'tta_density_map': torch.ones(2, 1, 1, 1)}
    with observer.first_forward(batch):
        module(batch)
        first_response = batch['_figure6_adapter_request'].responses[1]
        batch['tta_density_map'].fill_(20)
        module(batch)
        assert batch['_figure6_adapter_request'].responses[1] is first_response
    module(batch)
    record, = observer.finalize()
    np.testing.assert_array_equal(record.stages['sg_dfa'].arrays['density_map'], np.ones((1, 2, 2)))


@pytest.mark.parametrize('enabled, density', [(False, True), (True, False)])
def test_missing_stage_when_density_branch_inactive(enabled, density):
    """Given disabled/missing density, when capturing then no zero response is fabricated."""
    module = adapter(enabled)
    observer = collector()
    batch = {'spatial_features': torch.ones(2, 2, 2, 2)}
    if density:
        batch['tta_density_map'] = torch.ones(2, 1, 1, 1)
    with observer.first_forward(batch):
        module(batch)
    record, = observer.finalize()
    assert record.stages['sg_dfa'].status.state is StageState.MISSING
    assert not record.stages['sg_dfa'].arrays


def test_request_removed_when_forward_raises():
    """Given a failed forward, when unwinding then capture cannot leak into another pass."""
    observer = collector()
    batch = {}
    with pytest.raises(ArtifactError):
        with observer.first_forward(batch):
            raise ArtifactError('model failure')
    assert batch == {}


def test_nonfinite_values_when_adapter_observation_is_nonfinite():
    """Given a non-finite runtime density, when captured then NaNs remain observable."""
    module = adapter()
    observer = collector()
    batch = {'spatial_features': torch.ones(2, 2, 2, 2),
             'tta_density_map': torch.full((2, 1, 1, 1), float('nan'))}
    with observer.first_forward(batch):
        module(batch)
    record, = observer.finalize()
    assert np.isnan(record.stages['sg_dfa'].arrays['density_map']).all()
    assert np.isnan(record.stages['sg_dfa'].arrays['delta']).all()
