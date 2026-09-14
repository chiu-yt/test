import math

import torch
import torch.nn as nn

from pcdet.tta_methods.sar_optimizer import SAM
from pcdet.tta_methods.sar_utils import (
    configure_model_for_sar,
    normalized_reliable_mask,
    restore_sar_state,
    should_recover_sar,
    snapshot_sar_state,
    update_sar_ema,
)
from pcdet.tta_methods.tent_entropy import extract_detection_entropy


class TinySarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.bn1d = nn.BatchNorm1d(2)
        self.bn2d = nn.BatchNorm2d(2)
        self.bn3d = nn.BatchNorm3d(2)
        self.sync_bn = nn.SyncBatchNorm(2)
        self.gn = nn.GroupNorm(1, 2)
        self.ln = nn.LayerNorm(2)
        self.normalizer = nn.BatchNorm1d(2)
        self.image_backbone = nn.BatchNorm1d(2)
        self.backbone_3d = nn.GroupNorm(1, 2)
        self.fuser = nn.LayerNorm(2)
        self.dense_head = nn.BatchNorm1d(2)
        self.layer4 = nn.BatchNorm1d(2)
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
        self.blocks[9] = nn.BatchNorm1d(2)
        self.blocks[10] = nn.GroupNorm(1, 2)
        self.blocks[11] = nn.LayerNorm(2)
        self.norm = nn.BatchNorm1d(2)
        self.branch = nn.Module()
        self.branch.norm = nn.Module()
        self.branch.norm.affine = nn.BatchNorm1d(2)


def test_extract_detection_entropy_uses_class_mean_for_sigmoid_logits():
    # Given proposal logits whose Bernoulli probabilities are known exactly.
    logits = torch.tensor([[[0.0, math.log(3.0)], [0.0, -math.log(3.0)]]])
    expected = torch.tensor([[math.log(2.0), -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))]])

    # When proposal entropy is extracted in sigmoid mode.
    entropy, hmax = extract_detection_entropy(logits, mode='sigmoid')

    # Then class entropy is averaged per proposal and bounded by ln(2).
    torch.testing.assert_close(entropy, expected)
    assert entropy.shape == (1, 2)
    assert hmax == math.log(2.0)


def test_extract_detection_entropy_uses_class_axis_for_softmax_logits():
    # Given three-class proposal logits with uniform and non-uniform columns.
    logits = torch.tensor([[[0.0, math.log(2.0)], [0.0, 0.0], [0.0, 0.0]]])
    expected_second = -(0.5 * math.log(0.5) + 2.0 * 0.25 * math.log(0.25))

    # When proposal entropy is extracted in softmax mode.
    entropy, hmax = extract_detection_entropy(logits, mode='softmax')

    # Then softmax is taken over classes and its maximum is ln(C).
    torch.testing.assert_close(entropy, torch.tensor([[math.log(3.0), expected_second]]))
    assert entropy.shape == (1, 2)
    assert hmax == math.log(3.0)


def test_normalized_reliable_mask_is_strict_finite_and_empty_safe():
    # Given normalized entropies below, at, and above the threshold plus non-finite values.
    hmax = math.log(2.0)
    entropy = torch.tensor([[0.39 * hmax, 0.4 * hmax, 0.41 * hmax, float('nan'), float('inf')]])
    empty_logits = torch.empty(2, 3, 0)

    # When reliability and empty-proposal entropy are computed.
    mask = normalized_reliable_mask(entropy, hmax=hmax, threshold=0.4)
    empty_entropy, empty_hmax = extract_detection_entropy(empty_logits, mode='softmax')
    empty_mask = normalized_reliable_mask(empty_entropy, hmax=empty_hmax, threshold=0.4)

    # Then only finite values strictly below 0.4 pass and empty shape is preserved.
    assert torch.equal(mask, torch.tensor([[True, False, False, False, False]]))
    assert empty_entropy.shape == (2, 0)
    assert empty_mask.dtype == torch.bool
    assert empty_mask.shape == (2, 0)
    assert empty_hmax == math.log(3.0)


def test_configure_model_for_sar_selects_only_unexcluded_norm_affine_params():
    # Given every supported normalization family, official exclusions, and BEVFusion names.
    model = TinySarModel()
    expected_names = [
        'bn1d.weight', 'bn1d.bias', 'bn2d.weight', 'bn2d.bias',
        'bn3d.weight', 'bn3d.bias', 'sync_bn.weight', 'sync_bn.bias',
        'gn.weight', 'gn.bias', 'ln.weight', 'ln.bias',
        'normalizer.weight', 'normalizer.bias',
        'image_backbone.weight', 'image_backbone.bias',
        'backbone_3d.weight', 'backbone_3d.bias', 'fuser.weight', 'fuser.bias',
        'dense_head.weight', 'dense_head.bias',
    ]

    # When SAR configures the model and collects adaptation parameters.
    params, names, _, trainable_count, total_count = configure_model_for_sar(model)

    # Then only affine parameters from approved, unexcluded normalization modules train.
    assert names == expected_names
    assert [name for name, parameter in model.named_parameters() if parameter.requires_grad] == expected_names
    named_parameters = dict(model.named_parameters())
    assert all(parameter is named_parameters[name] for parameter, name in zip(params, expected_names))
    assert trainable_count == sum(parameter.numel() for parameter in params)
    assert trainable_count < total_count


def test_configure_model_for_sar_keeps_top_and_dropout_eval_but_bn_batch_adaptive():
    # Given a tiny detector initially in training mode.
    model = TinySarModel()
    model.train()

    # When SAR configures adaptation mode.
    configure_model_for_sar(model)

    # Then the top model and dropout stay eval while every BN uses batch statistics.
    assert model.training is False
    assert model.dropout.training is False
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            assert module.training is True
            assert module.track_running_stats is False
            assert module.running_mean is None
            assert module.running_var is None


def test_sam_sgd_perturbs_then_updates_from_the_unperturbed_weight():
    # Given one scalar parameter and SGD-backed SAM with a known gradient direction.
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = SAM([parameter], torch.optim.SGD, lr=0.1, momentum=0.0, rho=0.05)
    parameter.square().sum().backward()

    # When SAM takes its ascent step and its descent step.
    optimizer.first_step(zero_grad=True)
    perturbed = parameter.detach().clone()
    parameter.square().sum().backward()
    optimizer.second_step(zero_grad=True)

    # Then rho controls the perturbation and SGD updates the restored base weight.
    torch.testing.assert_close(perturbed, torch.tensor([1.05]))
    torch.testing.assert_close(parameter, torch.tensor([0.79]))
    assert parameter.grad is None


def test_sam_rollback_explicitly_restores_weight_and_accepts_zero_rho():
    # Given separate parameters for rollback and the rho-zero boundary.
    rollback_parameter = nn.Parameter(torch.tensor([1.0]))
    rollback_optimizer = SAM([rollback_parameter], torch.optim.SGD, lr=0.1, rho=0.05)
    rollback_parameter.square().sum().backward()
    zero_parameter = nn.Parameter(torch.tensor([1.0]))
    zero_optimizer = SAM([zero_parameter], torch.optim.SGD, lr=0.1, rho=0.0)
    zero_parameter.square().sum().backward()

    # When one perturbation is rolled back and zero-rho SAM completes both steps.
    rollback_optimizer.first_step(zero_grad=True)
    rollback_optimizer.rollback(zero_grad=True)
    zero_optimizer.first_step(zero_grad=True)
    zero_parameter.square().sum().backward()
    zero_optimizer.second_step(zero_grad=True)

    # Then rollback is exact and rho zero behaves as an unperturbed SGD update.
    torch.testing.assert_close(rollback_parameter, torch.tensor([1.0]))
    assert rollback_parameter.grad is None
    torch.testing.assert_close(zero_parameter, torch.tensor([0.8]))


def test_update_sar_ema_uses_fixed_point_nine_history_weight():
    # Given both an empty EMA and an existing EMA with a new scalar observation.
    previous = 10.0
    current = 20.0

    # When the SAR EMA values are updated.
    initialized = update_sar_ema(None, current)
    updated = update_sar_ema(previous, current)

    # Then the first value initializes it and later history receives exactly 0.9.
    assert initialized == current
    assert updated == 11.0


def test_should_recover_sar_uses_strict_normalized_threshold():
    # Given the normalized recovery threshold and values at every boundary class.
    threshold = 0.02895
    below_threshold = math.nextafter(threshold, 0.0)

    # When SAR evaluates whether each EMA requires recovery.
    missing_recovers = should_recover_sar(None, threshold=threshold)
    nan_recovers = should_recover_sar(float('nan'), threshold=threshold)
    boundary_recovers = should_recover_sar(threshold, threshold=threshold)
    below_recovers = should_recover_sar(below_threshold, threshold=threshold)

    # Then only a finite EMA strictly below the normalized threshold recovers.
    assert missing_recovers is False
    assert nan_recovers is False
    assert boundary_recovers is False
    assert below_recovers is True


def test_snapshot_restore_sar_state_restores_sam_and_base_sgd_state():
    # Given SAM after a complete two-pass update with populated SGD momentum.
    model = nn.Linear(2, 1, bias=False)
    optimizer = SAM(
        model.parameters(), torch.optim.SGD,
        lr=0.2, momentum=0.9, weight_decay=0.01, rho=0.05,
    )
    model(torch.ones(1, 2)).sum().backward()
    optimizer.first_step(zero_grad=True)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.second_step(zero_grad=True)
    model_state, optimizer_state = snapshot_sar_state(model, optimizer)
    expected_weight = model.weight.detach().clone()
    expected_momentum = optimizer.state[model.weight]['momentum_buffer'].detach().clone()
    assert not any('old_p' in state for state in optimizer_state['state'].values())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.first_step(zero_grad=True)
    with torch.no_grad():
        model.weight.add_(5.0)
        optimizer.state[model.weight]['momentum_buffer'].zero_()
    optimizer.param_groups[0]['lr'] = 3.0
    assert any('old_p' in state for state in optimizer.state.values())

    # When the complete model and SAM snapshot is restored.
    restore_sar_state(model, optimizer, model_state, optimizer_state)

    # Then model, SAM groups, base-SGD momentum, and transaction state match the snapshot.
    torch.testing.assert_close(model.weight, expected_weight)
    assert optimizer.param_groups[0]['lr'] == 0.2
    assert optimizer.param_groups[0]['momentum'] == 0.9
    assert optimizer.param_groups[0]['weight_decay'] == 0.01
    torch.testing.assert_close(optimizer.state[model.weight]['momentum_buffer'], expected_momentum)
    assert optimizer.base_optimizer.param_groups[0]['lr'] == 0.2
    torch.testing.assert_close(
        optimizer.base_optimizer.state[model.weight]['momentum_buffer'],
        expected_momentum,
    )
    assert not any('old_p' in state for state in optimizer.state.values())
