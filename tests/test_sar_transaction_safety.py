import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from pcdet.tta_methods.sar import SAR, SARStepInput
from pcdet.tta_methods.sar_optimizer import SAM


class _Detector(nn.Module):
    def __init__(self, dense_head):
        super().__init__()
        self.dense_head = dense_head

    def forward(self, batch):
        return self.dense_head.predict(batch), {}


class _FailingDenseHead(nn.Module):
    def __init__(self, expected_error):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.expected_error = expected_error

    def predict(self, _inputs):
        raise self.expected_error


class _NonFiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter):
        return parameter.new_tensor([[[10.0], [-10.0]]])

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.new_full((), float('nan'))


class _NonFiniteGradientDenseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))

    def predict(self, _inputs):
        return {'heatmap': _NonFiniteGradient.apply(self.weight)}


class _FiniteEntropyNonFiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter):
        return parameter.new_tensor([[0.1]])

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.new_full((), float('nan'))


class _ConstantLogitsDenseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))

    def predict(self, _inputs):
        return {'heatmap': self.weight.new_tensor([[[10.0], [-10.0]]])}


class _PartialFirstStepSAM(SAM):
    def __init__(self, params, base_optimizer, expected_error, **kwargs):
        self.expected_error = expected_error
        super().__init__(params, base_optimizer, **kwargs)

    def first_step(self, zero_grad=False):
        parameter = self.param_groups[0]['params'][0]
        with torch.no_grad():
            self.state[parameter]['old_p'] = parameter.detach().clone()
            parameter.add_(1.0)
        raise self.expected_error


def _build_sar(model):
    optimizer = SAM(
        model.parameters(), torch.optim.SGD,
        lr=0.1, momentum=0.9, rho=0.05,
    )
    return optimizer, SAR(model, optimizer, {'RELIABLE_MARGIN_NORM': 0.4})


def test_adapt_restores_transaction_when_second_forward_raises():
    # Given a first-pass gradient that opens SAM before the model raises.
    expected_error = RuntimeError('second forward failed')
    model = _Detector(_FailingDenseHead(expected_error))
    optimizer, sar = _build_sar(model)
    before = model.dense_head.weight.detach().clone()
    step = SARStepInput(
        {}, model.dense_head.weight.square().reshape(1, 1), 1.0, 4,
    )

    # When the perturbed second forward fails.
    with pytest.raises(RuntimeError) as raised:
        sar.adapt(step)

    # Then the original error escapes after exact restoration and cleanup.
    assert raised.value is expected_error
    assert torch.equal(model.dense_head.weight, before)
    assert optimizer.transaction_active is False
    assert not any('old_p' in state for state in optimizer.state.values())
    assert model.dense_head.weight.grad is None


def test_adapt_rolls_back_nonfinite_second_gradient_without_updates():
    # Given a finite low-entropy second pass whose backward emits NaN.
    model = _Detector(_NonFiniteGradientDenseHead())
    optimizer, sar = _build_sar(model)
    sar.ema = 0.25
    before = model.dense_head.weight.detach().clone()
    step = SARStepInput(
        {}, model.dense_head.weight.square().reshape(1, 1), 1.0, 9,
    )

    # When SAR computes the non-finite second gradient norm.
    result = sar.adapt(step)

    # Then rollback skips SGD, EMA, and recovery while reporting the reason.
    assert torch.equal(model.dense_head.weight, before)
    assert optimizer.transaction_active is False
    assert not any('old_p' in state for state in optimizer.state.values())
    assert 'momentum_buffer' not in optimizer.state[model.dense_head.weight]
    assert model.dense_head.weight.grad is None
    assert result.second_grad_norm is not None
    assert math.isfinite(result.second_grad_norm) is False
    assert result.finite is False
    assert result.skip_reason == 'second_gradient_nonfinite'
    assert sar.second_skip_count == 1
    assert sar.ema == 0.25
    assert sar.recovery_count == 0
    assert sar.recovery_batch_indices == []


def test_adapt_skips_first_step_when_first_gradient_is_nonfinite():
    # Given a finite first entropy whose backward emits a NaN gradient.
    model = _Detector(_ConstantLogitsDenseHead())
    optimizer, sar = _build_sar(model)
    sar.ema = 0.25
    before = model.dense_head.weight.detach().clone()
    step = SARStepInput(
        {},
        _FiniteEntropyNonFiniteGradient.apply(model.dense_head.weight),
        1.0,
        3,
    )

    # When SAR inspects the first gradient before opening any transaction.
    result = sar.adapt(step)

    # Then no transaction or update occurs and the nonfinite reason is reported.
    assert torch.equal(model.dense_head.weight, before)
    assert optimizer.transaction_active is False
    assert not any('old_p' in state for state in optimizer.state.values())
    assert 'momentum_buffer' not in optimizer.state[model.dense_head.weight]
    assert model.dense_head.weight.grad is None
    assert sar.first_skip_count == 1
    assert sar.second_skip_count == 0
    assert sar.ema == 0.25
    assert sar.recovery_count == 0
    assert sar.recovery_batch_indices == []
    assert result.finite is False
    assert result.skip_reason == 'first_gradient_nonfinite'
    assert result.first_grad_norm is not None
    assert math.isfinite(result.first_grad_norm) is False
    assert result.perturb_norm is None
    assert result.loss_second is None
    assert result.second_grad_norm is None


def test_adapt_restores_partial_transaction_when_first_step_raises():
    # Given a first_step that opens a transaction and then raises.
    expected_error = RuntimeError('first step failed mid-transaction')
    model = _Detector(_ConstantLogitsDenseHead())
    optimizer = _PartialFirstStepSAM(
        model.parameters(), torch.optim.SGD, expected_error,
        lr=0.1, momentum=0.9, rho=0.05,
    )
    sar = SAR(model, optimizer, {'RELIABLE_MARGIN_NORM': 0.4})
    before = model.dense_head.weight.detach().clone()
    step = SARStepInput(
        {}, model.dense_head.weight.square().reshape(1, 1), 1.0, 5,
    )

    # When the perturbation raises after partially mutating the model.
    with pytest.raises(RuntimeError) as raised:
        sar.adapt(step)

    # Then SAR restores the partial transaction and re-raises the original error.
    assert raised.value is expected_error
    assert torch.equal(model.dense_head.weight, before)
    assert optimizer.transaction_active is False
    assert not any('old_p' in state for state in optimizer.state.values())
    assert model.dense_head.weight.grad is None


@pytest.mark.parametrize('momentum', [-0.01, 1.01])
def test_sar_constructor_rejects_ema_momentum_outside_unit_interval(momentum):
    # Given a model, optimizer, and an out-of-range EMA momentum.
    model = _Detector(_ConstantLogitsDenseHead())
    optimizer = SAM(
        model.parameters(), torch.optim.SGD, lr=0.1, momentum=0.9, rho=0.05,
    )

    # When SAR is constructed with the invalid momentum.
    with pytest.raises(ValueError):
        SAR(model, optimizer, {'EMA_MOMENTUM': momentum})


def test_sar_yaml_declares_actual_entropy_source_literal():
    # Given the production SAR config.
    config_path = (
        Path(__file__).resolve().parents[1]
        / 'tools' / 'cfgs' / 'nuscenes_models' / 'bevfusion_sar.yaml'
    )
    config = yaml.safe_load(config_path.read_text())

    # When the declared entropy source is read.
    entropy_source = config['TTA']['SAR']['ENTROPY_SOURCE']

    # Then the config names the actual TransFusion dense-head heatmap source.
    assert entropy_source == 'transfusion_dense_head_heatmap'


def test_sar_accepts_actual_entropy_source_literal():
    # Given SAR configured with the actual TransFusion heatmap literal.
    model = _Detector(_ConstantLogitsDenseHead())
    optimizer = SAM(
        model.parameters(), torch.optim.SGD, lr=0.1, momentum=0.9, rho=0.05,
    )

    # When SAR validates the declared entropy source.
    SAR(model, optimizer, {'ENTROPY_SOURCE': 'transfusion_dense_head_heatmap'})

    # Then the actual literal is accepted without error.


def test_sar_rejects_stale_entropy_source_literal():
    # Given SAR configured with a stale entropy source literal.
    model = _Detector(_ConstantLogitsDenseHead())
    optimizer = SAM(
        model.parameters(), torch.optim.SGD, lr=0.1, momentum=0.9, rho=0.05,
    )

    # When SAR validates the declared entropy source.
    with pytest.raises(ValueError):
        SAR(model, optimizer, {'ENTROPY_SOURCE': 'first_stage_proposal_logits'})

    # Then the stale literal is rejected before adaptation runs.
