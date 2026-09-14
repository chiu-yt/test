"""Diagnostic log formatting for the SAR evaluator.

These helpers own only the textual per-batch and aggregate diagnostic log
lines. The evaluator keeps all aggregate bookkeeping and metadata assembly.
"""


def _optional_number(value):
    return float('nan') if value is None else value


def log_sar_batch_diagnostics(logger, batch_idx, step_result, adapter, entropy_mean,
                              trainable_count, total_count, logits, finite_count,
                              nan_count, inf_count, nonfinite_count, allocated_mb,
                              peak_mb, batch_time):
    selected_entropy_mean = _optional_number(step_result.first_entropy_raw_mean)
    skip_reason = step_result.skip_reason or 'none'
    logger.info(
        '[SAR] batch=%d entropy_mean=%.6f selected_entropy_mean=%.6f proposals=%d '
        'first_selected=%d second_candidates=%d second_selected=%d first_ratio=%.6f '
        'second_ratio=%.6f loss_first=%.6f loss_second=%.6f first_grad_norm=%.6f '
        'second_grad_norm=%.6f perturb_norm=%.6f trainable=%d/%d '
        'first_raw=%.6f first_norm=%.6f second_raw=%.6f second_norm=%.6f '
        'ema_raw=%.6f ema_norm=%.6f recovered=%s recovery_count=%d skip_reason=%s '
        'finite=%s finite_count=%d nan_count=%d inf_count=%d nonfinite_count=%d logits_shape=%s '
        'Nproposal=%d C=%d cuda_alloc_mb=%.1f cuda_peak_mb=%.1f batch_time=%.3f' % (
            batch_idx, entropy_mean, selected_entropy_mean, step_result.proposal_count,
            step_result.first_selected, step_result.second_candidates,
            step_result.second_selected, step_result.first_selected_ratio,
            step_result.second_selected_ratio, _optional_number(step_result.loss_first),
            _optional_number(step_result.loss_second),
            _optional_number(step_result.first_grad_norm),
            _optional_number(step_result.second_grad_norm),
            _optional_number(step_result.perturb_norm), trainable_count, total_count,
            _optional_number(step_result.first_entropy_raw_mean),
            _optional_number(step_result.first_entropy_norm_mean),
            _optional_number(step_result.second_entropy_raw_mean),
            _optional_number(step_result.second_entropy_norm_mean),
            _optional_number(step_result.ema), _optional_number(step_result.ema_norm),
            step_result.recovered, adapter.recovery_count, skip_reason,
            step_result.finite, finite_count, nan_count, inf_count, nonfinite_count,
            tuple(logits.shape), int(logits.shape[2]), int(logits.shape[1]),
            allocated_mb, peak_mb, batch_time,
        )
    )


def log_sar_logit_summary(logger, aggregate):
    logger.info(
        '[SAR] first-forward prediction logits: total=%d finite=%d nan=%d inf=%d '
        'nonfinite=%d finite_ratio=%.6f nan_ratio=%.6f inf_ratio=%.6f '
        'nonfinite_ratio=%.6f' % (
            aggregate['logit_total'], aggregate['logit_finite'], aggregate['logit_nan'],
            aggregate['logit_inf'], aggregate['logit_nonfinite'],
            aggregate['logit_finite'] / max(aggregate['logit_total'], 1),
            aggregate['logit_nan'] / max(aggregate['logit_total'], 1),
            aggregate['logit_inf'] / max(aggregate['logit_total'], 1),
            aggregate['logit_nonfinite'] / max(aggregate['logit_total'], 1),
        )
    )


def log_sar_batch_summary(logger, aggregate):
    logger.info(
        '[SAR] batches: total=%d fully_finite=%d contains_nan=%d contains_inf=%d '
        'contains_nonfinite=%d fully_finite_ratio=%.6f contains_nan_ratio=%.6f '
        'contains_inf_ratio=%.6f contains_nonfinite_ratio=%.6f' % (
            aggregate['batch_count'], aggregate['batch_finite'], aggregate['batch_nan'],
            aggregate['batch_inf'], aggregate['batch_nonfinite'],
            aggregate['batch_finite'] / max(aggregate['batch_count'], 1),
            aggregate['batch_nan'] / max(aggregate['batch_count'], 1),
            aggregate['batch_inf'] / max(aggregate['batch_count'], 1),
            aggregate['batch_nonfinite'] / max(aggregate['batch_count'], 1),
        )
    )
