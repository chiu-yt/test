import pickle
import time

import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.tta_methods.dpo_bevfusion import (
    DPOBEVFusion,
    build_dpo_optimizer,
    configure_model_for_dpo,
)
from .eval_utils import _init_class_counter, _update_pred_class_counter_from_annos


_ANNOTATION_KEYS = {'gt', 'annotations', 'annos', 'sample_annotation_tokens'}


class DPOBEVFusionEvaluationError(RuntimeError):
    """Raised when DPO-BEVFusion online evaluation cannot preserve its protocol."""


def _build_adaptation_batch(batch_dict):
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


def _profile_config(dpo_cfg):
    profile_name = str(dpo_cfg.get('PROFILE', 'dpo_paper')).upper()
    if profile_name not in dpo_cfg.PROFILES:
        raise DPOBEVFusionEvaluationError('Unknown DPO-BEVFusion profile: %s' % profile_name)
    return profile_name, dpo_cfg.PROFILES[profile_name]


def _finalize_results(cfg, dataset, result_dir, final_output_dir, logger,
                      det_annos, pred_class_counter, metadata, epoch_id, start_time):
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    elapsed = time.time() - start_time
    logger.info('Generate label finished(sec_per_example: %.4f second).' % (
        elapsed / max(len(dataset), 1)
    ))
    if cfg.LOCAL_RANK != 0:
        return {}
    result = {}
    pred_total = max(sum(pred_class_counter.values()), 1)
    for name in dataset.class_names:
        count = pred_class_counter[name]
        logger.info('DPO Pred[%s]: %d / %.4f' % (name, count, count / pred_total))
        result['diag/pred_count_%s' % name] = count
    with open(result_dir / 'result.pkl', 'wb') as result_file:
        pickle.dump(det_annos, result_file)
    result_string, dataset_result = dataset.evaluation(
        det_annos, dataset.class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir,
    )
    logger.info(result_string)
    result.update(dataset_result)
    result.update(metadata)
    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************DPO-BEVFusion evaluation done.*****************')
    return result


def eval_dpo_bevfusion_one_epoch(cfg, args, model, dataloader, epoch_id, logger,
                                 dist_test=False, result_dir=None):
    dpo_cfg = cfg.TTA.DPO_BEVFUSION
    if dist_test:
        raise DPOBEVFusionEvaluationError('DPO-BEVFusion online eval requires dist_test=False')
    if getattr(args, 'infer_time', False):
        raise DPOBEVFusionEvaluationError('DPO-BEVFusion does not support infer_time')
    if int(dpo_cfg.get('STEPS', 1)) != 1:
        raise DPOBEVFusionEvaluationError('DPO-BEVFusion requires STEPS == 1')
    if result_dir is None:
        raise DPOBEVFusionEvaluationError('DPO-BEVFusion eval requires result_dir')

    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    pred_class_counter = _init_class_counter(class_names)
    profile_name, profile_cfg = _profile_config(dpo_cfg)
    parameters, names, trainable_count, total_count = configure_model_for_dpo(
        model, profile_cfg
    )
    if not parameters:
        raise DPOBEVFusionEvaluationError('DPO-BEVFusion found no trainable parameters')
    optimizer = build_dpo_optimizer(parameters, profile_cfg)
    adapter = DPOBEVFusion(model, optimizer, parameters, profile_cfg)
    logger.info('[DPO] profile=%s trainable=%d/%d ratio=%.6f' % (
        profile_name, trainable_count, total_count,
        trainable_count / max(total_count, 1),
    ))
    logger.info('[DPO] trainable names: %s' % names)
    logger.info('[DPO] Z=ConvFuser output spatial_features; forwards=A/B/C/D; backwards=B/D')

    progress_bar = None
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(
            total=len(dataloader), leave=True, desc='dpo_bevfusion_eval', dynamic_ncols=True
        )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    aggregate = {
        'updated': 0, 'skipped': 0, 'high': 0, 'medium': 0,
        'low': 0, 'matched': 0,
    }
    skip_reasons = {}

    for batch_idx, original_batch in enumerate(dataloader):
        load_data_to_gpu(original_batch)
        adapter._set_prediction_mode()
        forward_a = _build_adaptation_batch(original_batch)
        with torch.no_grad():
            forward_a_predictions, _ = model(forward_a)
        detached_predictions = _detach_tensor_tree(forward_a_predictions)
        annos = dataset.generate_prediction_dicts(
            original_batch, detached_predictions, class_names,
            output_path=final_output_dir if args.save_to_file else None,
        )
        det_annos += annos
        _update_pred_class_counter_from_annos(annos, pred_class_counter)
        with torch.enable_grad():
            step_result = adapter.adapt(
                _build_adaptation_batch(original_batch), detached_predictions, batch_idx
            )
        aggregate['updated'] += int(step_result.updated)
        aggregate['skipped'] += int(not step_result.updated)
        aggregate['high'] += step_result.high_count
        aggregate['medium'] += step_result.medium_count
        aggregate['low'] += step_result.low_count
        aggregate['matched'] += step_result.matched_count
        reason = step_result.skip_reason or 'none'
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        logger.info(
            '[DPO] batch=%d updated=%s pseudo=%d/%d/%d matched=%d history=%d '
            'C1=%s C2=%s Ew=%s Ez=%s losses=%s/%s ema=%s cutoff=%s skip=%s' % (
                batch_idx, step_result.updated, step_result.high_count,
                step_result.medium_count, step_result.low_count,
                step_result.matched_count, step_result.history_count,
                step_result.c1, step_result.c2, step_result.epsilon_w_norm,
                step_result.epsilon_z_norm, step_result.clean_loss,
                step_result.refined_loss, step_result.ema_cost,
                step_result.cutoff_stopped, reason,
            )
        )
        if progress_bar is not None:
            progress_bar.set_postfix({'dpo_updated': aggregate['updated']})
            progress_bar.update()

    if progress_bar is not None:
        progress_bar.close()
    metadata = {
        'dpo/profile': profile_name,
        'dpo/trainable_params': trainable_count,
        'dpo/total_params': total_count,
        'dpo/trainable_ratio': trainable_count / max(total_count, 1),
        'dpo/updated_batches': aggregate['updated'],
        'dpo/skipped_batches': aggregate['skipped'],
        'dpo/high_count': aggregate['high'],
        'dpo/medium_count': aggregate['medium'],
        'dpo/low_count': aggregate['low'],
        'dpo/matched_count': aggregate['matched'],
        'dpo/history_count': len(adapter.history),
        'dpo/ema_cost': adapter.cutoff.ema,
        'dpo/cutoff_stopped': adapter.cutoff.stopped,
        'dpo/skip_reason_counts': skip_reasons,
    }
    if torch.cuda.is_available():
        metadata['dpo/peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return _finalize_results(
        cfg, dataset, result_dir, final_output_dir, logger, det_annos,
        pred_class_counter, metadata, epoch_id, start_time,
    )
