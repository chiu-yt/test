from pathlib import Path

import torch
import torch.nn as nn
import yaml

from pcdet.tta_methods.tent_entropy import entropy_loss_from_logits
from pcdet.tta_methods.tent_hooks import TransFusionLogitCapture
from pcdet.tta_methods.tent_utils import (
    build_tent_optimizer,
    changed_parameter_names,
    clone_named_parameters,
    configure_model_for_tent,
)
from tools.eval_utils.tent_eval_utils import (
    _tent_step_indices,
    _tent_update_norm_stats_enabled,
    _tent_updates_enabled,
)


class DummyCfg(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class TinyTentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.head = nn.Linear(4, 2)

    def forward(self, inputs):
        feats = self.bn(self.conv(inputs)).mean(dim=(2, 3))
        return self.head(feats)


class FakeDenseHead:
    def __init__(self):
        self.expected_logits = torch.randn(2, 3, 5, requires_grad=True)

    def predict(self, inputs):
        return {'heatmap': self.expected_logits, 'center': inputs}


class FakeDetector:
    def __init__(self):
        self.dense_head = FakeDenseHead()


def test_configure_model_for_tent_allows_only_bn_affine_params():
    model = TinyTentModel()
    params, names, bn_count, trainable_count, total_count = configure_model_for_tent(model, DummyCfg())

    assert bn_count == 1
    assert names == ['bn.weight', 'bn.bias']
    assert trainable_count == model.bn.weight.numel() + model.bn.bias.numel()
    assert trainable_count < total_count
    assert params[0] is model.bn.weight
    assert params[1] is model.bn.bias
    assert all(name in names for name, p in model.named_parameters() if p.requires_grad)


def test_entropy_loss_uses_finite_bernoulli_terms():
    logits = torch.tensor([[[0.0, 1.0], [float('nan'), -1.0]]])
    loss, diag = entropy_loss_from_logits(logits, entropy_mode='bernoulli', min_valid_terms=1, use_sigmoid=True)

    assert loss is not None
    assert torch.isfinite(loss)
    assert diag['valid_terms'] == 3
    assert diag['finite'] is True
    assert diag['shape'] == (1, 2, 2)
    assert diag['mode'] == 'bernoulli'


def test_softmax_entropy_is_invariant_to_uniform_negative_shift():
    logits = torch.linspace(-2.0, 2.0, steps=4000).reshape(2, 10, 200)

    loss, _ = entropy_loss_from_logits(logits, entropy_mode='softmax')
    shifted_loss, _ = entropy_loss_from_logits(logits - 20.0, entropy_mode='softmax')

    assert loss is not None
    assert shifted_loss is not None
    assert torch.allclose(loss, shifted_loss, atol=1e-6)


def test_bernoulli_entropy_decreases_under_uniform_negative_shift():
    logits = torch.linspace(-2.0, 2.0, steps=4000).reshape(2, 10, 200)

    loss, _ = entropy_loss_from_logits(logits, entropy_mode='bernoulli')
    shifted_loss, _ = entropy_loss_from_logits(logits - 20.0, entropy_mode='bernoulli')

    assert loss is not None
    assert shifted_loss is not None
    assert shifted_loss < loss


def test_transfusion_logit_capture_reads_pre_nms_heatmap():
    model = FakeDetector()
    with TransFusionLogitCapture(model) as capture:
        result = model.dense_head.predict(torch.zeros(1))

    assert capture.logits is model.dense_head.expected_logits
    assert result['heatmap'] is model.dense_head.expected_logits


def test_tent_step_changes_only_bn_affine_params():
    model = TinyTentModel()
    params, names, _, _, _ = configure_model_for_tent(model, DummyCfg())
    optimizer = build_tent_optimizer(params, DummyCfg({'LR': 1e-2, 'WEIGHT_DECAY': 0.0}))
    before = clone_named_parameters(model)

    outputs = model(torch.randn(2, 3, 4, 4))
    loss, diag = entropy_loss_from_logits(outputs[:, :, None], entropy_mode='softmax', use_sigmoid=False)
    assert loss is not None and diag['finite'] is True
    loss.backward()
    optimizer.step()

    changed = changed_parameter_names(before, model)
    assert changed
    assert set(changed).issubset(set(names))


def test_zero_tent_steps_skips_optimizer_update():
    model = TinyTentModel()
    params, _, _, _, _ = configure_model_for_tent(model, DummyCfg())
    optimizer = build_tent_optimizer(params, DummyCfg({'LR': 1e-2, 'WEIGHT_DECAY': 0.0}))
    before = clone_named_parameters(model)

    for _ in _tent_step_indices(0):
        outputs = model(torch.randn(2, 3, 4, 4))
        loss, diag = entropy_loss_from_logits(outputs[:, :, None], entropy_mode='softmax', use_sigmoid=False)
        assert loss is not None and diag['finite'] is True
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    assert changed_parameter_names(before, model) == []


def test_zero_tent_steps_disable_grad_updates():
    assert _tent_updates_enabled(0) is False
    assert list(_tent_step_indices(0)) == []
    assert _tent_updates_enabled(1) is True
    assert list(_tent_step_indices(1)) == [0]


def test_zero_tent_steps_disable_norm_stat_updates():
    assert _tent_update_norm_stats_enabled(DummyCfg({'UPDATE_NORM_STATS': True}), steps=0) is False
    assert _tent_update_norm_stats_enabled(DummyCfg({'UPDATE_NORM_STATS': True}), steps=1) is True


def test_zero_tent_steps_preserve_source_bn_output():
    model = TinyTentModel().eval()
    inputs = torch.randn(2, 3, 4, 4)
    source_output = model(inputs).detach()
    update_norm_stats = _tent_update_norm_stats_enabled(
        DummyCfg({'UPDATE_NORM_STATS': True}), steps=0
    )

    configure_model_for_tent(model, DummyCfg({'UPDATE_NORM_STATS': update_norm_stats}))
    tent_output = model(inputs).detach()

    assert torch.equal(tent_output, source_output)


def test_bevfusion_tent_config_uses_stable_defaults():
    config_path = Path(__file__).parents[1] / 'tools/cfgs/nuscenes_models/bevfusion_tent.yaml'

    with config_path.open(encoding='utf-8') as config_file:
        tent_cfg = yaml.safe_load(config_file)['TTA']['TENT']

    assert tent_cfg['ENTROPY_MODE'] == 'softmax'
    assert tent_cfg['UPDATE_NORM_STATS'] is False
    assert tent_cfg['LR'] == 0.0001
