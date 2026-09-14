from dataclasses import dataclass

import torch
import torch.nn as nn

from pcdet.tta_methods.dpo_bevfusion_perturb import (
    DPOFusedBEVPerturbation,
    DPOParameterPerturbation,
    compute_feature_epsilon,
    compute_parameter_epsilons,
)
from pcdet.tta_methods.dpo_bevfusion_utils import (
    DPOCostHistory,
    DPOCutoffState,
    build_clean_pseudo_targets,
    match_refined_targets,
    targets_to_training_tensors,
)
from pcdet.tta_methods.tent_utils import unwrap_model


BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


class DPOConfigurationError(ValueError):
    """Raised when a DPO profile violates the supported online protocol."""


class DPOUpdateSkipped(RuntimeError):
    """Signals a fail-closed batch without changing detector parameters."""


@dataclass(frozen=True, slots=True)
class DPOStepResult:
    updated: bool
    skip_reason: str | None
    high_count: int
    medium_count: int
    low_count: int
    matched_count: int
    history_count: int
    c1: float | None
    c2: float | None
    ema_cost: float | None
    cutoff_stopped: bool
    epsilon_w_norm: float | None
    epsilon_z_norm: float | None
    clean_loss: float | None
    refined_loss: float | None


def configure_model_for_dpo(model, profile_cfg):
    base_model = unwrap_model(model)
    original_trainable = {
        name for name, parameter in base_model.named_parameters()
        if parameter.requires_grad
    }
    update_scope = str(profile_cfg.get('UPDATE_SCOPE', 'full')).lower()
    if update_scope == 'full':
        parameters = [
            parameter for name, parameter in base_model.named_parameters()
            if name in original_trainable
        ]
        names = [name for name, _ in base_model.named_parameters() if name in original_trainable]
    elif update_scope == 'bn_affine':
        base_model.requires_grad_(False)
        parameters = []
        names = []
        for module_name, module in base_model.named_modules():
            if not isinstance(module, BN_TYPES):
                continue
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if parameter_name not in ('weight', 'bias'):
                    continue
                parameter.requires_grad_(True)
                parameters.append(parameter)
                names.append('%s.%s' % (module_name, parameter_name))
    else:
        raise DPOConfigurationError('DPO UPDATE_SCOPE must be full or bn_affine')
    base_model.eval()
    total_count = sum(parameter.numel() for parameter in base_model.parameters())
    trainable_count = sum(parameter.numel() for parameter in parameters)
    return parameters, names, trainable_count, total_count


def build_dpo_optimizer(parameters, profile_cfg):
    optimizer_name = str(profile_cfg.get('OPTIMIZER', 'SGD')).lower()
    if optimizer_name != 'sgd':
        raise DPOConfigurationError('DPO requires SGD')
    return torch.optim.SGD(
        parameters,
        lr=float(profile_cfg.LR),
        momentum=float(profile_cfg.MOMENTUM),
        weight_decay=float(profile_cfg.WEIGHT_DECAY),
    )


class DPOBEVFusion:
    """Maintains one online DPO stream and its persistent matching state."""

    def __init__(self, model, optimizer, parameters, profile_cfg):
        self.model = model
        self.base_model = unwrap_model(model)
        self.optimizer = optimizer
        self.parameters = list(parameters)
        self.profile_cfg = profile_cfg
        self.update_scope = str(profile_cfg.get('UPDATE_SCOPE', 'full')).lower()
        self.history = DPOCostHistory(float(profile_cfg.ALPHA))
        self.cutoff = DPOCutoffState(
            gamma=float(profile_cfg.get('GAMMA', 0.5)),
            c_stop=profile_cfg.get('C_STOP', None),
            enabled=bool(profile_cfg.get('CUTOFF_ENABLED', False)),
        )

    def _set_prediction_mode(self):
        self.base_model.eval()
        if self.update_scope == 'bn_affine':
            for module in self.base_model.modules():
                if isinstance(module, BN_TYPES):
                    module.train()
                    module.track_running_stats = False

    def _set_loss_mode(self):
        self._set_prediction_mode()
        self.base_model.training = True
        self.base_model.dense_head.training = True

    @staticmethod
    def _training_batch(batch, targets):
        training_batch = dict(batch)
        gt_boxes, actions = targets_to_training_tensors(targets)
        training_batch['gt_boxes'] = gt_boxes
        training_batch['tta_pseudo_actions'] = actions
        return training_batch

    @staticmethod
    def _action_counts(targets, source_count):
        actions = [target['actions'] for target in targets if target['actions'].numel() > 0]
        if not actions:
            return 0, 0, source_count
        joined = torch.cat(actions)
        high = int((joined == 1).sum().item())
        medium = int((joined == -1).sum().item())
        return high, medium, source_count - high - medium

    def _result(self, **values):
        defaults = {
            'updated': False, 'skip_reason': None, 'high_count': 0,
            'medium_count': 0, 'low_count': 0, 'matched_count': 0,
            'history_count': len(self.history), 'c1': None, 'c2': None,
            'ema_cost': self.cutoff.ema, 'cutoff_stopped': self.cutoff.stopped,
            'epsilon_w_norm': None, 'epsilon_z_norm': None,
            'clean_loss': None, 'refined_loss': None,
        }
        defaults.update(values)
        return DPOStepResult(**defaults)

    def adapt(self, batch, forward_a_predictions, batch_idx):
        del batch_idx
        if self.cutoff.stopped:
            return self._result(skip_reason='early_cutoff')
        clean_targets = build_clean_pseudo_targets(
            forward_a_predictions,
            self.profile_cfg.get('SCORE_THRESHOLDS', self.profile_cfg.SCORE_THRESH),
            self.profile_cfg.get('NEG_THRESHOLDS', self.profile_cfg.NEG_THRESH),
        )
        source_count = sum(target['source_count'] for target in clean_targets)
        clean_high, clean_medium, clean_low = self._action_counts(clean_targets, source_count)
        self.optimizer.zero_grad()
        history_size = len(self.history.values)
        parameter_perturbation = None
        try:
            self._set_loss_mode()
            forward_b = self._training_batch(batch, clean_targets)
            with DPOFusedBEVPerturbation(self.base_model.fuser, capture=True) as capture:
                clean_result, _, _ = self.model(forward_b)
            clean_loss = clean_result['loss'].mean()
            if not bool(torch.isfinite(clean_loss).item()):
                raise DPOUpdateSkipped('nonfinite_clean_loss')
            clean_loss.backward()
            epsilons_w = compute_parameter_epsilons(self.parameters, self.profile_cfg.RHO_W)
            fused_grad = None if capture.fused_bev is None else capture.fused_bev.grad
            epsilon_z = compute_feature_epsilon(fused_grad, self.profile_cfg.RHO_Z)
            if epsilons_w is None or epsilon_z is None:
                raise DPOUpdateSkipped('invalid_clean_gradient')
            epsilon_w_norm = float(torch.norm(torch.stack([
                torch.norm(epsilon, p=2) for epsilon in epsilons_w.values()
            ]), p=2).item())
            epsilon_z_norm = float(torch.norm(epsilon_z, p=2).item())
            self.optimizer.zero_grad()
            parameter_perturbation = DPOParameterPerturbation(self.optimizer, epsilons_w)
            parameter_perturbation.apply()

            self._set_prediction_mode()
            forward_c = dict(batch)
            with torch.no_grad(), DPOFusedBEVPerturbation(self.base_model.fuser, epsilon_z=epsilon_z):
                disturbed_predictions, _ = self.model(forward_c)
            refined_targets, matched_costs, thresholds = match_refined_targets(
                clean_targets, disturbed_predictions, self.history
            )

            self._set_loss_mode()
            forward_d = self._training_batch(batch, refined_targets)
            with DPOFusedBEVPerturbation(self.base_model.fuser, epsilon_z=epsilon_z):
                refined_result, _, _ = self.model(forward_d)
            refined_loss = refined_result['loss'].mean()
            if not bool(torch.isfinite(refined_loss).item()):
                raise DPOUpdateSkipped('nonfinite_refined_loss')
            refined_loss.backward()
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in self.parameters
            ):
                raise DPOUpdateSkipped('nonfinite_refined_gradient')
            parameter_perturbation.step()
            parameter_perturbation = None
            self.optimizer.zero_grad()
            self.cutoff.update(matched_costs)
            high, medium, low = self._action_counts(refined_targets, source_count)
            c1 = None if thresholds is None else float(thresholds[0].item())
            c2 = None if thresholds is None else float(thresholds[1].item())
            return self._result(
                updated=True, high_count=high, medium_count=medium,
                low_count=low, matched_count=int(matched_costs.numel()),
                history_count=len(self.history), c1=c1, c2=c2,
                ema_cost=self.cutoff.ema, cutoff_stopped=self.cutoff.stopped,
                epsilon_w_norm=epsilon_w_norm, epsilon_z_norm=epsilon_z_norm,
                clean_loss=float(clean_loss.detach().item()),
                refined_loss=float(refined_loss.detach().item()),
            )
        except DPOUpdateSkipped as error:
            del self.history.values[history_size:]
            return self._result(
                skip_reason=str(error), high_count=clean_high,
                medium_count=clean_medium, low_count=clean_low,
            )
        finally:
            if parameter_perturbation is not None:
                parameter_perturbation.restore()
            self.optimizer.zero_grad()
            self.base_model.eval()
