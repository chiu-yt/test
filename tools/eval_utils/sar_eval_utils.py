# noqa: SIZE_OK - ordered online evaluation keeps timing and SAM boundaries auditable.
import time

import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.tta_methods.sar import SAR, SARStepInput
from pcdet.tta_methods.sar_proposals import SARProposalAlignmentError
from pcdet.tta_methods.sar_utils import (
    EXCLUDED_NAME_PARTS,
    build_sar_optimizer,
    configure_model_for_sar,
)
from pcdet.tta_methods.tent_entropy import extract_detection_entropy
from pcdet.tta_methods.tent_hooks import TransFusionLogitCapture
from pcdet.tta_methods.tent_utils import unwrap_model
from pcdet.utils.efficiency_profiler import EfficiencyProfiler, EfficiencyProfileRun
from .eval_utils import (
    _init_class_counter,
    _update_pred_class_counter_from_annos,
)
from .sar_eval_logging import (
    log_sar_batch_diagnostics,
    log_sar_batch_summary,
    log_sar_logit_summary,
)
from .sar_eval_results import _finalize_sar_results


_ANNOTATION_KEYS = {
    'gt', 'annotations', 'annos', 'sample_annotation_tokens',
}


class SAREvaluationError(RuntimeError):
    """Raised when the independent SAR evaluation protocol is violated."""


def _build_adaptation_batch(batch_dict):
    """Return a shallow detector input view with every annotation field removed."""
    return {
        key: value
        for key, value in batch_dict.items()
        if key.lower() not in _ANNOTATION_KEYS
        and not key.lower().startswith('gt_')
        and 'ground_truth' not in key.lower()
    }


def _detach_tensor_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, list):
        return [_detach_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_tensor_tree(item) for item in value)
    if isinstance(value, dict):
        return {key: _detach_tensor_tree(item) for key, item in value.items()}
    return value


def _head_uses_sigmoid(model, cfg):
    base_model = unwrap_model(model)
    dense_head = getattr(base_model, 'dense_head', None)
    if dense_head is not None and hasattr(dense_head, 'use_sigmoid_cls'):
        return bool(dense_head.use_sigmoid_cls)
    loss_cfg = cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_CLS
    return bool(loss_cfg.get('use_sigmoid', False))


def eval_sar_one_epoch(cfg, args, model, dataloader, epoch_id, logger,
                       dist_test=False, result_dir=None):
    sar_cfg = cfg.TTA.SAR
    if dist_test:
        raise SAREvaluationError('SAR online eval requires dist_test=False')
    if getattr(args, 'infer_time', False):
        raise SAREvaluationError('SAR online eval does not support infer_time')
    if int(sar_cfg.get('STEPS', 1)) != 1:
        raise SAREvaluationError('SAR requires TTA.SAR.STEPS == 1')
    if str(sar_cfg.get('OPTIMIZER', 'SGD')).lower() != 'sgd':
        raise SAREvaluationError('SAR requires TTA.SAR.OPTIMIZER == SGD')
    if result_dir is None:
        raise SAREvaluationError('SAR eval requires a result_dir')

    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    pred_class_counter = _init_class_counter(class_names)

    logger.info('*************** EPOCH %s SAR EVALUATION *****************' % epoch_id)
    params, names, norm_counts, trainable_count, total_count = configure_model_for_sar(
        model, logger
    )
    if not params:
        raise SAREvaluationError('SAR found no eligible normalization affine parameters')
    optimizer = build_sar_optimizer(params, sar_cfg)
    use_sigmoid = _head_uses_sigmoid(model, cfg)
    adapter = SAR(model, optimizer, sar_cfg, use_sigmoid=use_sigmoid)
    profiler = EfficiencyProfiler(
        cfg.TTA.get('EFFICIENCY_PROFILE', {}),
        EfficiencyProfileRun(
            method='sar',
            entrypoint='test.py',
            config=str(cfg.TAG),
            boundary=(
                'prediction forward plus adapter.adapt SAM/update online work; '
                'excludes dataloader iteration, load_data_to_gpu, prediction '
                'serialization, progress/logging, pickle, and dataset evaluation'
            ),
            output_dir=result_dir,
            model_parameters=model.parameters(),
            optimizer=optimizer,
        ),
        logger=logger,
    )
    logger.info('[SAR] parameters total=%d trainable=%d percentage=%.6f%%' % (
        total_count, trainable_count, 100.0 * trainable_count / max(total_count, 1)
    ))
    logger.info('[SAR] normalization counts: %s' % norm_counts)
    logger.info(
        '[SAR] entropy logits source: dense_head.predict()["heatmap"] '
        '(post-decoder pre-NMS proposal classification logits)'
    )
    logger.info('[SAR] entropy mode: %s (use_sigmoid_cls=%s)' % (
        adapter.entropy_mode, use_sigmoid
    ))
    logger.info(
        '[SAR] thresholds reliable_margin_norm=%.6f recovery_threshold_norm=%.6f '
        'recovery=%s ema_momentum=%.6f' % (
            adapter.reliable_margin_norm, adapter.recovery_threshold_norm,
            adapter.recovery, adapter.ema_momentum,
        )
    )
    logger.info('[SAR] optimizer=SGD lr=%.8f rho=%.6f momentum=%.6f weight_decay=%.8f' % (
        float(sar_cfg.get('LR', 1e-3)), float(sar_cfg.get('RHO', 0.05)),
        float(sar_cfg.get('MOMENTUM', 0.9)), float(sar_cfg.get('WEIGHT_DECAY', 0.0)),
    ))
    logger.info('[SAR] normalization exclusions: %s' % (EXCLUDED_NAME_PARTS,))

    progress_bar = None
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(
            total=len(dataloader), leave=True, desc='sar_eval', dynamic_ncols=True
        )
    torch.cuda.reset_peak_memory_stats()
    aggregate = {
        'proposal_count': 0, 'first_selected': 0,
        'second_candidates': 0, 'second_selected': 0,
        'batch_count': 0, 'batch_finite': 0, 'batch_nan': 0,
        'batch_inf': 0, 'batch_nonfinite': 0,
        'logit_total': 0, 'logit_finite': 0, 'logit_nan': 0,
        'logit_inf': 0, 'logit_nonfinite': 0,
    }
    skip_reason_counts = {}
    start_time = time.time()

    for batch_idx, original_batch in enumerate(dataloader):
        batch_start = time.time()
        load_data_to_gpu(original_batch)
        prediction_batch = _build_adaptation_batch(original_batch)
        profiler.begin_batch(original_batch['batch_size'])
        profiler.begin_segment()
        with torch.enable_grad():
            with TransFusionLogitCapture(model) as capture:
                pred_dicts, _ = model(prediction_batch)
            pred_dicts = _detach_tensor_tree(pred_dicts)
            logits = capture.logits
            assert logits is not None
            first_entropy, hmax = extract_detection_entropy(
                logits, mode=adapter.entropy_mode
            )
        profiler.end_segment()
        annos = dataset.generate_prediction_dicts(
            original_batch, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None,
        )
        det_annos += annos
        display = {}
        _update_pred_class_counter_from_annos(annos, pred_class_counter)

        profiler.begin_segment()
        proposal_ids = capture.proposal_ids
        if proposal_ids is None:
            raise SARProposalAlignmentError(
                'SAR prediction forward did not expose proposal IDs'
            )
        detached_entropy = first_entropy.detach()
        finite_entropy = detached_entropy[torch.isfinite(detached_entropy)]
        entropy_mean = (
            float(finite_entropy.detach().mean().item())
            if finite_entropy.numel() else float('nan')
        )
        step = SARStepInput(
            batch=_build_adaptation_batch(original_batch),
            first_entropy=first_entropy,
            proposal_ids=proposal_ids.detach(),
            hmax=hmax,
            batch_idx=batch_idx,
        )
        step_result = adapter.adapt(step)
        profiler.end_segment()
        profiler.end_batch()

        for key in ('proposal_count', 'first_selected', 'second_candidates', 'second_selected'):
            aggregate[key] += getattr(step_result, key)
        skip_reason = step_result.skip_reason or 'none'
        skip_reason_counts[skip_reason] = skip_reason_counts.get(skip_reason, 0) + 1
        logit_total = int(logits.numel())
        finite_count = int(torch.isfinite(logits).sum().item())
        nan_count = int(torch.isnan(logits).sum().item())
        inf_count = int(torch.isinf(logits).sum().item())
        nonfinite_count = logit_total - finite_count
        aggregate['batch_count'] += 1
        aggregate['logit_total'] += logit_total
        aggregate['logit_finite'] += finite_count
        aggregate['logit_nan'] += nan_count
        aggregate['logit_inf'] += inf_count
        aggregate['logit_nonfinite'] += nonfinite_count
        if nonfinite_count == 0:
            aggregate['batch_finite'] += 1
        if nan_count > 0:
            aggregate['batch_nan'] += 1
        if inf_count > 0:
            aggregate['batch_inf'] += 1
        if nonfinite_count > 0:
            aggregate['batch_nonfinite'] += 1
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        log_sar_batch_diagnostics(
            logger, batch_idx, step_result, adapter, entropy_mean,
            trainable_count, total_count, logits, finite_count, nan_count,
            inf_count, nonfinite_count, allocated_mb, peak_mb,
            time.time() - batch_start,
        )
        display.update({
            'sar_entropy': '%.4f' % entropy_mean,
            'sar_selected': step_result.first_selected,
        })
        if progress_bar is not None:
            progress_bar.set_postfix(display)
            progress_bar.update()
        del step, first_entropy, logits, capture, prediction_batch

    profiler.finalize()
    if progress_bar is not None:
        progress_bar.close()
    log_sar_logit_summary(logger, aggregate)
    log_sar_batch_summary(logger, aggregate)
    metadata = {
        'sar/trainable_params': trainable_count,
        'sar/total_params': total_count,
        'sar/trainable_ratio': trainable_count / max(total_count, 1),
        'sar/trainable_names': names,
        'sar/norm_counts': norm_counts,
        'sar/first_skip_count': adapter.first_skip_count,
        'sar/second_skip_count': adapter.second_skip_count,
        'sar/skip_count': adapter.first_skip_count + adapter.second_skip_count,
        'sar/skip_reason_counts': skip_reason_counts,
        'sar/recovery_count': adapter.recovery_count,
        'sar/recovery_batch_indices': list(adapter.recovery_batch_indices),
        'sar/proposal_count': aggregate['proposal_count'],
        'sar/first_selected': aggregate['first_selected'],
        'sar/second_candidates': aggregate['second_candidates'],
        'sar/second_selected': aggregate['second_selected'],
        'sar/first_selection_ratio': aggregate['first_selected'] / max(aggregate['proposal_count'], 1),
        'sar/second_selection_ratio': aggregate['second_selected'] / max(aggregate['first_selected'], 1),
        'sar/batch_total_count': aggregate['batch_count'],
        'sar/batch_finite_count': aggregate['batch_finite'],
        'sar/batch_nan_count': aggregate['batch_nan'],
        'sar/batch_inf_count': aggregate['batch_inf'],
        'sar/batch_nonfinite_count': aggregate['batch_nonfinite'],
        'sar/batch_finite_ratio': aggregate['batch_finite'] / max(aggregate['batch_count'], 1),
        'sar/batch_nan_ratio': aggregate['batch_nan'] / max(aggregate['batch_count'], 1),
        'sar/batch_inf_ratio': aggregate['batch_inf'] / max(aggregate['batch_count'], 1),
        'sar/batch_nonfinite_ratio': aggregate['batch_nonfinite'] / max(aggregate['batch_count'], 1),
        'sar/logit_total_count': aggregate['logit_total'],
        'sar/logit_finite_count': aggregate['logit_finite'],
        'sar/logit_nan_count': aggregate['logit_nan'],
        'sar/logit_inf_count': aggregate['logit_inf'],
        'sar/logit_nonfinite_count': aggregate['logit_nonfinite'],
        'sar/logit_finite_ratio': aggregate['logit_finite'] / max(aggregate['logit_total'], 1),
        'sar/logit_nan_ratio': aggregate['logit_nan'] / max(aggregate['logit_total'], 1),
        'sar/logit_inf_ratio': aggregate['logit_inf'] / max(aggregate['logit_total'], 1),
        'sar/logit_nonfinite_ratio': aggregate['logit_nonfinite'] / max(aggregate['logit_total'], 1),
        'sar/finite_batches': aggregate['batch_finite'],
        'sar/finite_batch_ratio': aggregate['batch_finite'] / max(aggregate['batch_count'], 1),
    }
    return _finalize_sar_results(
        cfg, dataset, result_dir, final_output_dir, logger, det_annos,
        pred_class_counter, metadata, epoch_id, start_time,
    )
