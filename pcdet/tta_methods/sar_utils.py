import copy
import math

import torch
import torch.nn as nn

from pcdet.tta_methods.sar_optimizer import SAM
from pcdet.tta_methods.tent_utils import count_parameters, unwrap_model


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
NORM_TYPES = BN_TYPES + (nn.GroupNorm, nn.LayerNorm)
EXCLUDED_NAME_PARTS = ('layer4', 'blocks.9', 'blocks.10', 'blocks.11', 'norm.')


class InvalidSARConfigurationError(ValueError):
    """Raised when SAR configuration values violate its core contract."""


def configure_model_for_sar(model, logger=None):
    """Freeze a detector and expose only SAR-approved normalization affine parameters."""
    model.eval()
    model.requires_grad_(False)
    base_model = unwrap_model(model)

    params = []
    names = []
    norm_counts = {
        'BatchNorm1d': 0,
        'BatchNorm2d': 0,
        'BatchNorm3d': 0,
        'SyncBatchNorm': 0,
        'GroupNorm': 0,
        'LayerNorm': 0,
        'selected': 0,
        'excluded': 0,
    }

    for module_name, module in base_model.named_modules():
        if not isinstance(module, NORM_TYPES):
            continue
        norm_counts[type(module).__name__] += 1

        if isinstance(module, BN_TYPES):
            module.train()
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

        excluded = module_name == 'norm' or any(
            part in module_name for part in EXCLUDED_NAME_PARTS
        )
        if excluded:
            norm_counts['excluded'] += 1
            continue

        norm_counts['selected'] += 1
        for param_name, parameter in module.named_parameters(recurse=False):
            if param_name not in ('weight', 'bias'):
                continue
            parameter.requires_grad_(True)
            params.append(parameter)
            names.append('%s.%s' % (module_name, param_name))

    trainable_count = sum(parameter.numel() for parameter in params)
    total_count = count_parameters(base_model)
    if logger is not None:
        logger.info('[SAR] normalization modules: %s' % norm_counts)
        logger.info('[SAR] trainable parameters: %d / %d (%.6f)' % (
            trainable_count,
            total_count,
            trainable_count / max(total_count, 1),
        ))
        for name in names:
            logger.info('[SAR] trainable: %s' % name)

    return params, names, norm_counts, trainable_count, total_count


def build_sar_optimizer(params, sar_cfg):
    """Build SAR's required SGD-backed SAM optimizer."""
    optimizer_name = str(sar_cfg.get('OPTIMIZER', 'SGD')).lower()
    if optimizer_name != 'sgd':
        raise InvalidSARConfigurationError('SAR requires OPTIMIZER == SGD')
    return SAM(
        params,
        torch.optim.SGD,
        lr=float(sar_cfg.get('LR', 2.5e-4)),
        momentum=float(sar_cfg.get('MOMENTUM', 0.9)),
        weight_decay=float(sar_cfg.get('WEIGHT_DECAY', 0.0)),
        rho=float(sar_cfg.get('RHO', 0.05)),
        adaptive=bool(sar_cfg.get('ADAPTIVE_SAM', False)),
    )


def normalized_reliable_mask(entropy, hmax, threshold):
    """Select finite proposal entropies strictly below normalized SAR margin."""
    normalized_entropy = entropy / hmax
    return torch.isfinite(normalized_entropy) & (normalized_entropy < threshold)


def update_sar_ema(ema, new_value, alpha=0.9):
    """Update SAR's entropy moving average, initializing from the first value."""
    if not 0.0 <= alpha <= 1.0:
        raise InvalidSARConfigurationError('SAR EMA alpha must be in [0, 1]')
    if ema is None:
        return new_value
    with torch.no_grad():
        return alpha * ema + (1.0 - alpha) * new_value


def should_recover_sar(ema, threshold=0.2):
    """Return whether a finite SAR EMA crossed the model-recovery threshold."""
    return ema is not None and math.isfinite(float(ema)) and float(ema) < threshold


def snapshot_sar_state(model, optimizer):
    """Deep-copy complete model and optimizer state for later recovery."""
    return (
        copy.deepcopy(unwrap_model(model).state_dict()),
        copy.deepcopy(optimizer.state_dict()),
    )


def restore_sar_state(model, optimizer, model_state, optimizer_state):
    """Strictly restore model parameters, buffers, and complete optimizer state."""
    unwrap_model(model).load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
