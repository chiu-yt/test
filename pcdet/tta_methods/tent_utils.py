import copy

import torch
import torch.nn as nn


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def configure_model_for_tent(model, tent_cfg, logger=None):
    base_model = unwrap_model(model)
    base_model.eval()
    base_model.requires_grad_(False)

    reset_bn_stats = bool(tent_cfg.get('RESET_BN_STATS', False))
    update_norm_stats = bool(tent_cfg.get('UPDATE_NORM_STATS', True))
    params = []
    names = []
    bn_count = 0

    for module_name, module in base_model.named_modules():
        if not isinstance(module, BN_TYPES):
            continue
        bn_count += 1
        module.train(update_norm_stats)
        if update_norm_stats:
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        elif reset_bn_stats:
            module.reset_running_stats()

        for param_name, param in module.named_parameters(recurse=False):
            if param_name not in ['weight', 'bias']:
                continue
            param.requires_grad_(True)
            params.append(param)
            names.append('%s.%s' % (module_name, param_name))

    trainable_params = sum(p.numel() for p in params)
    total_params = count_parameters(base_model)
    if logger is not None:
        logger.info('[Tent] detected BN-family modules: %d' % bn_count)
        logger.info('[Tent] trainable parameters: %d / %d (%.6f)' % (
            trainable_params, total_params, trainable_params / max(total_params, 1)
        ))
        for name in names:
            logger.info('[Tent] trainable: %s' % name)

    return params, names, bn_count, trainable_params, total_params


def build_tent_optimizer(params, tent_cfg):
    optimizer_name = str(tent_cfg.get('OPTIMIZER', 'Adam')).lower()
    lr = float(tent_cfg.get('LR', 1e-3))
    weight_decay = float(tent_cfg.get('WEIGHT_DECAY', 0.0))
    if optimizer_name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == 'sgd':
        momentum = float(tent_cfg.get('MOMENTUM', 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise NotImplementedError('Unsupported Tent optimizer: %s' % optimizer_name)


def snapshot_tent_state(model, optimizer):
    return copy.deepcopy(unwrap_model(model).state_dict()), copy.deepcopy(optimizer.state_dict())


def restore_tent_state(model, optimizer, model_state, optimizer_state):
    unwrap_model(model).load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def trainable_parameter_names(model):
    base_model = unwrap_model(model)
    return [name for name, param in base_model.named_parameters() if param.requires_grad]


def clone_named_parameters(model):
    base_model = unwrap_model(model)
    return {name: param.detach().clone() for name, param in base_model.named_parameters()}


def changed_parameter_names(before, model):
    base_model = unwrap_model(model)
    changed = []
    for name, param in base_model.named_parameters():
        if name not in before:
            continue
        if not torch.equal(before[name], param.detach()):
            changed.append(name)
    return changed
