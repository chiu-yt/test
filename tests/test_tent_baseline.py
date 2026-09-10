import torch
import torch.nn as nn

from pcdet.tta_methods.tent_entropy import entropy_loss_from_logits
from pcdet.tta_methods.tent_hooks import TransFusionLogitCapture
from pcdet.tta_methods.tent_utils import (
    build_tent_optimizer,
    changed_parameter_names,
    clone_named_parameters,
    configure_model_for_tent,
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
