import copy
from dataclasses import dataclass
import math

import torch

from .reg_tta3d_utils import (
    compute_crr_dimension_loss,
    decoded_query_boxes,
    query_scores,
    update_cbu_regression_teacher,
)


class RegTTA3DConfigurationError(ValueError):
    pass


def _base_model(model):
    return model.module if hasattr(model, 'module') else model


def configure_reg_tta3d_student(model):
    base_model = _base_model(model)
    base_model.eval()
    base_model.requires_grad_(False)
    base_model.training = True
    base_model.dense_head.training = True

    params = []
    names = []
    for name, parameter in base_model.named_parameters():
        if name.startswith(tuple(
                'dense_head.prediction_head.%s.' % branch
                for branch in ('center', 'height', 'dim', 'rot', 'vel'))):
            parameter.requires_grad_(True)
            params.append(parameter)
            names.append(name)

    trainable_count = sum(parameter.numel() for parameter in params)
    total_count = sum(parameter.numel() for parameter in base_model.parameters())
    return params, names, trainable_count, total_count


def initialize_reg_tta3d_models(model):
    base_model = _base_model(model)
    teacher = copy.deepcopy(base_model)
    student = copy.deepcopy(base_model)
    teacher.eval()
    teacher.requires_grad_(False)
    params, names, trainable_count, total_count = configure_reg_tta3d_student(student)
    return teacher, student, params, names, trainable_count, total_count


def build_reg_tta3d_optimizer(parameters, method_cfg):
    optimizer_name = str(method_cfg.get('OPTIMIZER', 'Adam')).lower()
    if optimizer_name != 'adam':
        raise RegTTA3DConfigurationError('Reg-TTA3D requires Adam')
    return torch.optim.Adam(
        parameters,
        lr=float(method_cfg.get('LR', 1e-3)),
        weight_decay=float(method_cfg.get('WEIGHT_DECAY', 0.0)),
    )


class RegressionPredictionCapture:
    def __init__(self, model):
        self.output = None
        self.handle = _base_model(model).dense_head.prediction_head.register_forward_hook(
            self._capture
        )

    def _capture(self, _module, _inputs, output):
        self.output = output

    def close(self):
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()


@dataclass(frozen=True, slots=True)
class RegTTA3DStepResult:
    updated: bool
    skip_reason: str | None
    pseudo_count: int
    crr_count: int
    loss_native: float | None
    loss_crr: float | None
    loss_total: float | None
    grad_norm: float | None
    cbu_alpha: float
    teacher_delta: float


def _gradient_norm(parameters):
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(squared_norm)


class RegTTA3D:
    def __init__(self, teacher, student, parameters, optimizer, method_cfg):
        self.teacher = teacher
        self.student = student
        self.parameters = parameters
        self.optimizer = optimizer
        self.method_cfg = method_cfg

    @torch.enable_grad()
    def adapt(self, student_batch, cbu_alpha):
        pseudo_count = int((student_batch['gt_boxes'][..., -1] > 0).sum().item())
        if pseudo_count == 0:
            return self._skipped('empty_pseudo_labels', cbu_alpha, pseudo_count)

        self.optimizer.zero_grad()
        with RegressionPredictionCapture(self.student) as capture:
            training_result, _, _ = self.student(student_batch)
        if capture.output is None:
            return self._skipped('missing_query_predictions', cbu_alpha, pseudo_count)

        raw_predictions = capture.output
        student_boxes = decoded_query_boxes(raw_predictions)
        scores, labels = query_scores(
            raw_predictions,
            _base_model(self.student).dense_head.query_labels,
            _base_model(self.student).dense_head.num_classes,
        )
        crr_loss = compute_crr_dimension_loss(
            student_boxes.reshape(-1, student_boxes.shape[-1]),
            scores.reshape(-1),
            labels.reshape(-1),
            top_ratio=float(self.method_cfg.get('CRR_TOP_RATIO', 0.2)),
            margin=float(self.method_cfg.get('CRR_MARGIN', 0.1)),
            weight=float(self.method_cfg.get('CRR_WEIGHT', 1.0)),
        )
        native_loss = training_result['loss'].mean()
        total_loss = native_loss + crr_loss
        if not bool(torch.isfinite(total_loss).item()):
            self.optimizer.zero_grad()
            return self._skipped('nonfinite_loss', cbu_alpha, pseudo_count)

        total_loss.backward()
        grad_norm = _gradient_norm(self.parameters)
        if not math.isfinite(grad_norm):
            self.optimizer.zero_grad()
            return self._skipped('nonfinite_gradient', cbu_alpha, pseudo_count)
        grad_clip = float(self.method_cfg.get('GRAD_NORM_CLIP', 0.0))
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters, grad_clip)
        parameter_snapshot = [
            parameter.detach().clone() for parameter in self.parameters
        ]
        optimizer_snapshot = copy.deepcopy(self.optimizer.state_dict())
        self.optimizer.step()
        parameters_finite = all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in self.parameters
        )
        if not parameters_finite:
            with torch.no_grad():
                for parameter, before in zip(self.parameters, parameter_snapshot):
                    parameter.copy_(before)
            self.optimizer.load_state_dict(optimizer_snapshot)
            self.optimizer.zero_grad()
            return self._skipped('nonfinite_parameter', cbu_alpha, pseudo_count)
        teacher_delta = update_cbu_regression_teacher(
            self.teacher, self.student, cbu_alpha
        )
        self.optimizer.zero_grad()
        return RegTTA3DStepResult(
            True, None, pseudo_count, int(scores.numel()),
            float(native_loss.detach().item()), float(crr_loss.detach().item()),
            float(total_loss.detach().item()), grad_norm, cbu_alpha, teacher_delta,
        )

    @staticmethod
    def _skipped(reason, cbu_alpha, pseudo_count):
        return RegTTA3DStepResult(
            False, reason, pseudo_count, 0, None, None, None, None,
            cbu_alpha, 0.0,
        )
