from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from pcdet.tta_methods.sar_utils import (
    InvalidSARConfigurationError,
    normalized_reliable_mask,
    restore_sar_state,
    should_recover_sar,
    snapshot_sar_state,
    update_sar_ema,
)
from pcdet.tta_methods.tent_entropy import extract_detection_entropy
from pcdet.tta_methods.tent_hooks import TransFusionLogitCapture


@dataclass(frozen=True)  # noqa: SLOTS_OK - Required by the server's pre-3.10 dataclasses.
class SARStepInput:
    """Inputs retained from the evaluator's pre-adaptation forward."""

    batch: Mapping
    first_entropy: torch.Tensor
    hmax: float
    batch_idx: int


@dataclass(frozen=True)  # noqa: SLOTS_OK - Required by the server's pre-3.10 dataclasses.
class SARStepResult:
    """Observable diagnostics for one independent SAR transaction."""

    batch_idx: int
    proposal_count: int
    first_selected: int
    second_candidates: int
    second_selected: int
    first_selected_ratio: float
    second_selected_ratio: float
    first_entropy_raw_mean: float | None
    first_entropy_norm_mean: float | None
    second_entropy_raw_mean: float | None
    second_entropy_norm_mean: float | None
    loss_first: float | None
    loss_second: float | None
    first_grad_norm: float | None
    second_grad_norm: float | None
    perturb_norm: float | None
    ema: float | None
    ema_norm: float | None
    recovered: bool
    finite: bool
    first_entropy_shape: tuple[int, ...]
    second_logits_shape: tuple[int, ...] | None
    second_entropy_shape: tuple[int, ...] | None
    skip_reason: str | None


class SAR:
    """Mutable per-stream SAR state with independent per-batch transactions."""

    def __init__(self, model, optimizer, sar_cfg, use_sigmoid=True):
        self.model = model
        self.optimizer = optimizer
        self.entropy_source = str(sar_cfg.get(
            'ENTROPY_SOURCE', 'transfusion_dense_head_heatmap'
        ))
        if self.entropy_source != 'transfusion_dense_head_heatmap':
            raise InvalidSARConfigurationError(
                'SAR requires ENTROPY_SOURCE == transfusion_dense_head_heatmap'
            )
        configured_mode = str(sar_cfg.get('ENTROPY_MODE', 'auto')).lower()
        self.entropy_mode = (
            'sigmoid' if configured_mode == 'auto' and use_sigmoid
            else 'softmax' if configured_mode == 'auto'
            else configured_mode
        )
        self.use_sigmoid = bool(use_sigmoid)
        self.reliable_margin_norm = float(
            sar_cfg.get('RELIABLE_MARGIN_NORM', 0.4)
        )
        self.ema_momentum = float(sar_cfg.get('EMA_MOMENTUM', 0.9))
        if not 0.0 <= self.ema_momentum <= 1.0:
            raise InvalidSARConfigurationError(
                'SAR EMA_MOMENTUM must be in [0, 1]'
            )
        self.recovery = bool(sar_cfg.get('RECOVERY', True))
        self.recovery_threshold_norm = float(
            sar_cfg.get('RECOVERY_THRESHOLD_NORM', 0.02895)
        )
        self.ema = None
        self.first_skip_count = 0
        self.second_skip_count = 0
        self.recovery_count = 0
        self.recovery_batch_indices = []
        self.model_state, self.optimizer_state = snapshot_sar_state(
            self.model, self.optimizer
        )

    @torch.enable_grad()
    def adapt(self, step: SARStepInput) -> SARStepResult:
        first_mask = normalized_reliable_mask(
            step.first_entropy, step.hmax, self.reliable_margin_norm
        )
        proposal_count = int(step.first_entropy.numel())
        first_selected = int(first_mask.sum().item())
        first_entropy_finite = bool(torch.isfinite(step.first_entropy).all().item())
        first_raw_mean = (
            float(step.first_entropy[first_mask].detach().mean().item())
            if first_selected else None
        )
        first_norm_mean = (
            first_raw_mean / step.hmax if first_raw_mean is not None else None
        )
        if first_selected == 0:
            self.first_skip_count += 1
            return SARStepResult(
                step.batch_idx, proposal_count, 0, 0, 0, 0.0, 0.0,
                None, None, None, None, None, None, None, None, None,
                self.ema, self.ema / step.hmax if self.ema is not None else None,
                False, first_entropy_finite, tuple(step.first_entropy.shape), None, None,
                'first_empty',
            )

        self.optimizer.zero_grad()
        loss_first = step.first_entropy[first_mask].mean()
        loss_first.backward()
        first_grad_norm_tensor = self.optimizer._grad_norm()
        first_grad_norm = float(first_grad_norm_tensor.item())
        if not bool(torch.isfinite(first_grad_norm_tensor).item()):
            self.optimizer.zero_grad()
            self.first_skip_count += 1
            return SARStepResult(
                step.batch_idx, proposal_count, first_selected, 0, 0,
                first_selected / proposal_count, 0.0,
                first_raw_mean, first_norm_mean, None, None,
                float(loss_first.detach().item()), None,
                first_grad_norm, None, None, self.ema,
                self.ema / step.hmax if self.ema is not None else None,
                False, False, tuple(step.first_entropy.shape), None, None,
                'first_gradient_nonfinite',
            )

        try:
            first_step_diagnostics = self.optimizer.first_step(zero_grad=True)
            with TransFusionLogitCapture(self.model) as capture:
                self.model(dict(step.batch))
            second_logits = capture.logits
            assert second_logits is not None
            second_entropy, second_hmax = extract_detection_entropy(
                second_logits, mode=self.entropy_mode
            )
            second_entropy_finite = bool(torch.isfinite(second_entropy).all().item())
            second_subset = second_entropy[first_mask]
            second_mask = normalized_reliable_mask(
                second_subset, second_hmax, self.reliable_margin_norm
            )
            second_selected = int(second_mask.sum().item())
            loss_second = (
                second_subset.new_zeros(())
                if second_selected == 0
                else second_subset[second_mask].mean()
            )
            if second_selected == 0 or not torch.isfinite(loss_second):
                self.optimizer.rollback(zero_grad=True)
                self.second_skip_count += 1
                return SARStepResult(
                    step.batch_idx, proposal_count, first_selected,
                    int(second_subset.numel()), second_selected,
                    first_selected / proposal_count,
                    second_selected / first_selected,
                    first_raw_mean, first_norm_mean, None, None,
                    float(loss_first.detach().item()), None,
                    first_step_diagnostics['grad_norm'], None,
                    first_step_diagnostics['perturb_norm'], self.ema,
                    self.ema / step.hmax if self.ema is not None else None,
                    False, first_entropy_finite and second_entropy_finite,
                    tuple(step.first_entropy.shape),
                    tuple(second_logits.shape), tuple(second_entropy.shape),
                    'second_empty_or_nonfinite',
                )

            second_raw_mean = float(loss_second.detach().item())
            second_norm_mean = second_raw_mean / second_hmax
            loss_second.backward()
            second_grad_norm_tensor = self.optimizer._grad_norm()
            second_grad_norm = float(second_grad_norm_tensor.item())
            if not bool(torch.isfinite(second_grad_norm_tensor).item()):
                self.optimizer.rollback(zero_grad=True)
                self.second_skip_count += 1
                return SARStepResult(
                    step.batch_idx, proposal_count, first_selected,
                    int(second_subset.numel()), second_selected,
                    first_selected / proposal_count,
                    second_selected / first_selected,
                    first_raw_mean, first_norm_mean,
                    second_raw_mean, second_norm_mean,
                    float(loss_first.detach().item()), second_raw_mean,
                    first_step_diagnostics['grad_norm'], second_grad_norm,
                    first_step_diagnostics['perturb_norm'], self.ema,
                    self.ema / step.hmax if self.ema is not None else None,
                    False, False, tuple(step.first_entropy.shape),
                    tuple(second_logits.shape), tuple(second_entropy.shape),
                    'second_gradient_nonfinite',
                )

            self.optimizer.second_step(zero_grad=True)
            updated_ema = update_sar_ema(
                self.ema, second_raw_mean, alpha=self.ema_momentum
            )
            self.ema = updated_ema
            ema_norm = updated_ema / step.hmax
            recovered = self.recovery and should_recover_sar(
                ema_norm, threshold=self.recovery_threshold_norm
            )
            if recovered:
                self.reset()
                self.recovery_count += 1
                self.recovery_batch_indices.append(step.batch_idx)

            return SARStepResult(
                step.batch_idx, proposal_count, first_selected,
                int(second_subset.numel()), second_selected,
                first_selected / proposal_count, second_selected / first_selected,
                first_raw_mean, first_norm_mean, second_raw_mean, second_norm_mean,
                float(loss_first.detach().item()), second_raw_mean,
                first_step_diagnostics['grad_norm'], second_grad_norm,
                first_step_diagnostics['perturb_norm'], updated_ema,
                ema_norm, recovered, first_entropy_finite and second_entropy_finite,
                tuple(step.first_entropy.shape), tuple(second_logits.shape),
                tuple(second_entropy.shape), None,
            )
        finally:
            if self.optimizer.transaction_active:
                self.optimizer.rollback(zero_grad=True)

    def reset(self) -> None:
        restore_sar_state(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )
        self.ema = None
