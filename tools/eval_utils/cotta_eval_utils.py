import copy
import math
import time

import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.tta_methods.cotta_utils import (
    filter_predictions_to_targets,
    initialize_cotta_models,
    stochastic_restore,
    transform_boxes_between_views,
    update_ema_teacher,
)
from .cotta_eval_results import _ema_parameter_delta, _finalize_results, _gradient_norm
from .cotta_scale_view import (
    CottaEvaluationError,
    _ScaleViewBuilder,
    _is_gt_field,
)
from .eval_utils import (
    _init_class_counter,
    _update_gt_class_counter_from_batch,
    _update_pred_class_counter_from_annos,
    statistics_info,
)


_DEFAULT_SCORE_THRESHOLDS = [0.28, 0.28, 0.28, 0.20, 0.20, 0.24, 0.20, 0.18, 0.24, 0.34]
_DEFAULT_NMS_THRESHOLDS = [0.1] * 10


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


def _map_pseudo_targets(targets, weak_matrices, strong_matrices):
    mapped_targets = targets.clone()
    for batch_index in range(targets.shape[0]):
        valid = targets[batch_index, :, -1] > 0
        mapped_targets[batch_index, valid, :9] = transform_boxes_between_views(
            targets[batch_index, valid, :9],
            weak_matrices[batch_index],
            strong_matrices[batch_index],
        )
    return mapped_targets


def _loss_value(tb_dict, key):
    value = tb_dict.get(key, float('nan'))
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    return float(value)


def eval_cotta_one_epoch(cfg, args, model, dataloader, epoch_id, logger,
                         dist_test=False, result_dir=None):
    cotta_cfg = cfg.TTA.COTTA
    if dist_test:
        raise CottaEvaluationError('CoTTA online eval requires dist_test=False')
    if getattr(args, 'infer_time', False):
        raise CottaEvaluationError('CoTTA online eval does not support infer_time')
    optimizer_name = cotta_cfg.get('OPTIMIZER', 'Adam')
    if not isinstance(optimizer_name, str) or optimizer_name.lower() != 'adam':
        raise CottaEvaluationError('CoTTA-3OD requires TTA.COTTA.OPTIMIZER == Adam')
    if result_dir is None:
        raise CottaEvaluationError('CoTTA eval requires a result_dir')
    steps = int(cotta_cfg.get('STEPS', 1))
    if steps != 1:
        raise CottaEvaluationError(
            'The CoTTA-3OD baseline requires TTA.COTTA.STEPS == 1'
        )

    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataloader.dataset
    class_names = dataset.class_names
    metric = {'gt_num': 0}
    for threshold in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % threshold] = 0
        metric['recall_rcnn_%s' % threshold] = 0
    det_annos = []
    gt_class_counter = _init_class_counter(class_names)
    pred_class_counter = _init_class_counter(class_names)

    logger.info('*************** EPOCH %s COTTA EVALUATION *****************' % epoch_id)
    model_parameters = list(model.parameters())
    if not model_parameters:
        raise CottaEvaluationError('CoTTA requires a detector with parameters')
    model_device = model_parameters[0].device
    if model_device.type != 'cuda':
        raise CottaEvaluationError('CoTTA online eval requires the detector on CUDA')
    model.cpu()
    torch.cuda.empty_cache()
    source_anchor, teacher, student = initialize_cotta_models(model)
    source_anchor.cpu()
    teacher.to(model_device)
    student.to(model_device)
    trainable_names = [
        name for name, parameter in student.named_parameters() if parameter.requires_grad
    ]
    trainable_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise CottaEvaluationError('CoTTA found no source-trainable detector parameters')
    total_count = sum(parameter.numel() for parameter in student.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    logger.info('[CoTTA] parameters trainable=%d total=%d ratio=%.6f' % (
        trainable_count, total_count, trainable_count / max(total_count, 1)
    ))
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=float(cotta_cfg.get('LR', 1e-4)),
        weight_decay=float(cotta_cfg.get('WEIGHT_DECAY', 0.0)),
    )
    weak_builder = _ScaleViewBuilder(
        dataset, cotta_cfg.get('WEAK_SCALE_RANGE', [0.95, 1.05])
    )
    strong_builder = _ScaleViewBuilder(
        dataset, cotta_cfg.get('STRONG_SCALE_RANGE', [0.90, 1.10])
    )
    ema_alpha = float(cotta_cfg.get('EMA_ALPHA', 0.999))
    restore_probability = float(cotta_cfg.get('RESTORE_PROBABILITY', 0.01))
    log_interval = int(cotta_cfg.get('LOG_INTERVAL', 50))
    grad_clip = float(cotta_cfg.get('GRAD_NORM_CLIP', 0.0))

    progress_bar = None
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(
            total=len(dataloader), leave=True, desc='cotta_eval', dynamic_ncols=True
        )
    torch.cuda.reset_peak_memory_stats(model_device)
    start_time = time.time()

    for batch_index, original_batch in enumerate(dataloader):
        batch_start = time.time()
        load_data_to_gpu(original_batch)
        _update_gt_class_counter_from_batch(original_batch, class_names, gt_class_counter)

        teacher.eval()
        with torch.no_grad():
            evaluation_batch = copy.deepcopy(original_batch)
            prediction_dicts, recall_dict = teacher(evaluation_batch)
            prediction_dicts = _detach_tensor_tree(prediction_dicts)
        annotations = dataset.generate_prediction_dicts(
            original_batch, prediction_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None,
        )
        del evaluation_batch
        del prediction_dicts

        weak_batch = weak_builder.build(original_batch)
        weak_matrices = weak_batch['lidar_aug_matrix'].clone()
        with torch.no_grad():
            weak_predictions, _ = teacher(weak_batch)
        pseudo_targets, pseudo_weights, class_counts = filter_predictions_to_targets(
            weak_predictions,
            cotta_cfg.get('SCORE_THRESHOLDS', _DEFAULT_SCORE_THRESHOLDS),
            cotta_cfg.get('NMS_THRESHOLDS', _DEFAULT_NMS_THRESHOLDS),
            int(cotta_cfg.get('NMS_PRE_MAXSIZE', 4096)),
            int(cotta_cfg.get('NMS_POST_MAXSIZE', 500)),
        )
        del weak_predictions
        del weak_batch
        strong_batch = strong_builder.build(original_batch)
        strong_batch['gt_boxes'] = _map_pseudo_targets(
            pseudo_targets, weak_matrices, strong_batch['lidar_aug_matrix']
        )
        del weak_matrices
        strong_batch['tta_pseudo_weights'] = pseudo_weights
        del original_batch

        with torch.enable_grad():
            student.train()
            optimizer.zero_grad()
            training_result, loss_metrics, _ = student(strong_batch)
            loss = training_result['loss'].mean()
            finite = bool(torch.isfinite(loss).item())
            if not finite:
                logger.info('[CoTTA] batch=%d finite=False total_loss=%s' % (
                    batch_index, float(loss.detach().item())
                ))
                raise FloatingPointError('CoTTA encountered a non-finite native detector loss')
            loss.backward()
            grad_norm = _gradient_norm(trainable_parameters)
            if not math.isfinite(grad_norm):
                raise FloatingPointError('CoTTA encountered non-finite student gradients')
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip)
            optimizer.step()
        total_loss = float(loss.detach().item())
        loss_breakdown = {
            name: _loss_value(loss_metrics, name)
            for name in ('loss_heatmap', 'loss_cls', 'loss_bbox')
        }
        del training_result
        del loss_metrics
        del strong_batch
        del loss
        ema_delta = _ema_parameter_delta(teacher, student, ema_alpha)
        update_ema_teacher(teacher, student, ema_alpha)
        restored_count, eligible_count = stochastic_restore(
            student, source_anchor, trainable_names, restore_probability
        )

        display = {}
        statistics_info(cfg, recall_dict, metric, display)
        _update_pred_class_counter_from_annos(annotations, pred_class_counter)
        det_annos += annotations
        valid_pseudo = pseudo_targets[:, :, -1] > 0
        pseudo_total = int(valid_pseudo.sum().item())
        mean_confidence = float(pseudo_weights[valid_pseudo].mean().item()) if pseudo_total else 0.0
        restoration_ratio = restored_count / max(eligible_count, 1)
        display.update({'cotta_loss': '%.4f' % total_loss, 'cotta_pseudo': pseudo_total})

        if batch_index == 0 or (log_interval > 0 and batch_index % log_interval == 0):
            allocated_memory = torch.cuda.memory_allocated(model_device) / (1024 ** 2)
            peak_memory = torch.cuda.max_memory_allocated(model_device) / (1024 ** 2)
            per_class = {
                name: int(count)
                for name, count in zip(class_names, class_counts.sum(dim=0).tolist())
            }
            logger.info(
                '[CoTTA] batch=%d pseudo_total=%d pseudo_per_class=%s pseudo_mean_conf=%.6f '
                'loss_heatmap=%.6f loss_cls=%.6f loss_bbox=%.6f total=%.6f grad_norm=%.6f '
                'ema_delta=%.9f restoration_ratio=%.9f finite=%s cuda_alloc_mb=%.1f '
                'cuda_max_mb=%.1f step_time=%.3f' % (
                    batch_index, pseudo_total, per_class, mean_confidence,
                    loss_breakdown['loss_heatmap'], loss_breakdown['loss_cls'],
                    loss_breakdown['loss_bbox'], total_loss, grad_norm,
                    ema_delta, restoration_ratio, finite, allocated_memory,
                    peak_memory, time.time() - batch_start,
                )
            )
        if progress_bar is not None:
            progress_bar.set_postfix(display)
            progress_bar.update()

    if progress_bar is not None:
        progress_bar.close()
    return _finalize_results(
        cfg, dataset, result_dir, final_output_dir, logger, metric, det_annos,
        gt_class_counter, pred_class_counter, trainable_count, total_count,
        epoch_id, start_time,
    )
