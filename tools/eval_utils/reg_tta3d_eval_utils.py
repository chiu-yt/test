import pickle
import time

import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.tta_methods.reg_tta3d import (
    RegTTA3D,
    build_reg_tta3d_optimizer,
    initialize_reg_tta3d_models,
)
from pcdet.tta_methods.reg_tta3d_geometry import RegTTA3DViewBuilder
from pcdet.tta_methods.reg_tta3d_npg import build_npg_pseudo_targets
from pcdet.tta_methods.reg_tta3d_utils import cbu_alpha_from_labels
from .eval_utils import _init_class_counter, _update_pred_class_counter_from_annos


class RegTTA3DEvaluationError(RuntimeError):
    pass


def _build_adaptation_batch(batch_dict):
    annotation_keys = {
        'gt', 'gt_boxes', 'gt_names', 'ground_truth', 'annotations', 'annos',
        'sample_annotation_tokens',
    }
    return {
        key: value for key, value in batch_dict.items()
        if key.lower() not in annotation_keys
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


def _validate_method_config(method_cfg):
    if str(method_cfg.get('CBU_VARIANCE_MAPPING', '')).lower() != 'theoretical_max':
        raise RegTTA3DEvaluationError(
            'Reg-TTA3D supports CBU_VARIANCE_MAPPING == theoretical_max'
        )
    alpha_min = float(method_cfg.get('CBU_ALPHA_MIN', 0.99))
    alpha_max = float(method_cfg.get('CBU_ALPHA_MAX', 0.999))
    if not 0.99 <= alpha_min <= alpha_max <= 0.999:
        raise RegTTA3DEvaluationError(
            'Reg-TTA3D CBU alpha range must remain within [0.99, 0.999]'
        )


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
        logger.info('Reg-TTA3D Pred[%s]: %d / %.4f' % (
            name, count, count / pred_total
        ))
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
    logger.info('****************Reg-TTA3D evaluation done.*****************')
    return result


def eval_reg_tta3d_one_epoch(cfg, args, model, dataloader, epoch_id, logger,
                             dist_test=False, result_dir=None):
    method_cfg = cfg.TTA.REG_TTA3D
    if dist_test:
        raise RegTTA3DEvaluationError('Reg-TTA3D requires dist_test=False')
    if getattr(args, 'infer_time', False):
        raise RegTTA3DEvaluationError('Reg-TTA3D does not support infer_time')
    if int(method_cfg.get('STEPS', 1)) != 1:
        raise RegTTA3DEvaluationError('Reg-TTA3D requires STEPS == 1')
    if bool(method_cfg.get('ALLOW_DDP', False)):
        raise RegTTA3DEvaluationError('Reg-TTA3D ALLOW_DDP must be False')
    if result_dir is None:
        raise RegTTA3DEvaluationError('Reg-TTA3D requires result_dir')
    _validate_method_config(method_cfg)

    model_parameters = list(model.parameters())
    if not model_parameters or model_parameters[0].device.type != 'cuda':
        raise RegTTA3DEvaluationError('Reg-TTA3D requires a CUDA detector')
    model_device = model_parameters[0].device
    model.cpu()
    torch.cuda.empty_cache()
    teacher, student, _, names, trainable_count, total_count = \
        initialize_reg_tta3d_models(model)
    teacher.to(model_device)
    student.to(model_device)
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not parameters:
        raise RegTTA3DEvaluationError('Reg-TTA3D found no regression parameters')
    optimizer = build_reg_tta3d_optimizer(parameters, method_cfg)
    adapter = RegTTA3D(teacher, student, parameters, optimizer, method_cfg)

    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataloader.dataset
    class_names = dataset.class_names
    view_builder = RegTTA3DViewBuilder(dataset, method_cfg)
    det_annos = []
    pred_class_counter = _init_class_counter(class_names)
    logger.info('[Reg-TTA3D Inspection] regression branches=%s' % (
        ['center', 'height', 'dim', 'rot', 'vel']
    ))
    logger.info('[Reg-TTA3D Inspection] trainable=%d total=%d ratio=%.6f names=%s' % (
        trainable_count, total_count, trainable_count / max(total_count, 1), names
    ))
    logger.info('[Reg-TTA3D] BEVFusion-specific NPG approximation; '
                'TransFusion has no proposal-conditioned score reevaluation')

    progress_bar = None
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(
            total=len(dataloader), leave=True, desc='reg_tta3d_eval', dynamic_ncols=True
        )
    torch.cuda.reset_peak_memory_stats(model_device)
    start_time = time.time()
    aggregate = {
        'updated': 0, 'skipped': 0, 'source': 0, 'deleted': 0,
        'confidence_filtered': 0, 'kept': 0, 'inactive': 0,
    }
    skip_reasons = {}

    for batch_index, original_batch in enumerate(dataloader):
        load_data_to_gpu(original_batch)
        teacher.eval()
        evaluation_batch = _build_adaptation_batch(original_batch)
        evaluation_batch['reg_tta3d_capture_queries'] = True
        with torch.no_grad():
            prediction_dicts, _ = teacher(evaluation_batch)
        detached_predictions = _detach_tensor_tree(prediction_dicts)
        annotations = dataset.generate_prediction_dicts(
            original_batch, detached_predictions, class_names,
            output_path=final_output_dir if args.save_to_file else None,
        )
        det_annos += annotations
        _update_pred_class_counter_from_annos(annotations, pred_class_counter)

        raw_predictions = _detach_tensor_tree(
            evaluation_batch.pop('reg_tta3d_query_predictions', None)
        )
        if raw_predictions is None:
            raise RegTTA3DEvaluationError('Teacher query predictions were not captured')
        npg_result = build_npg_pseudo_targets(
            raw_predictions, teacher.dense_head, method_cfg
        )
        cbu_alpha = cbu_alpha_from_labels(
            npg_result.labels, teacher.dense_head.num_classes,
            alpha_min=float(method_cfg.get('CBU_ALPHA_MIN', 0.99)),
            alpha_max=float(method_cfg.get('CBU_ALPHA_MAX', 0.999)),
        )
        student_batch = view_builder.build(original_batch, npg_result.pseudo_targets)
        step_result = adapter.adapt(student_batch, cbu_alpha)

        aggregate['updated'] += int(step_result.updated)
        aggregate['skipped'] += int(not step_result.updated)
        aggregate['source'] += npg_result.source_count
        aggregate['deleted'] += npg_result.deleted_count
        aggregate['confidence_filtered'] += npg_result.confidence_filtered_count
        aggregate['kept'] += npg_result.kept_count
        aggregate['inactive'] += int(npg_result.inactive_threshold)
        reason = step_result.skip_reason or 'none'
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        log_interval = int(method_cfg.get('LOG_INTERVAL', 50))
        if batch_index == 0 or (log_interval > 0 and batch_index % log_interval == 0):
            logger.info(
                '[Reg-TTA3D] batch=%d updated=%s NPG=%d/%d/%d inactive_tau=%s '
                'CRR=%d loss=%s/%s/%s grad=%s CBU=%.6f delta=%.9f skip=%s' % (
                    batch_index, step_result.updated, npg_result.source_count,
                    npg_result.deleted_count, npg_result.kept_count,
                    npg_result.inactive_threshold, step_result.crr_count,
                    step_result.loss_native, step_result.loss_crr,
                    step_result.loss_total, step_result.grad_norm,
                    step_result.cbu_alpha, step_result.teacher_delta, reason,
                )
            )
        if progress_bar is not None:
            progress_bar.set_postfix({'reg_updated': aggregate['updated']})
            progress_bar.update()

    if progress_bar is not None:
        progress_bar.close()
    metadata = {
        'reg_tta3d/trainable_params': trainable_count,
        'reg_tta3d/total_params': total_count,
        'reg_tta3d/trainable_ratio': trainable_count / max(total_count, 1),
        'reg_tta3d/updated_batches': aggregate['updated'],
        'reg_tta3d/skipped_batches': aggregate['skipped'],
        'reg_tta3d/npg_source': aggregate['source'],
        'reg_tta3d/npg_deleted': aggregate['deleted'],
        'reg_tta3d/npg_confidence_filtered': aggregate['confidence_filtered'],
        'reg_tta3d/npg_kept': aggregate['kept'],
        'reg_tta3d/npg_inactive_batches': aggregate['inactive'],
        'reg_tta3d/skip_reason_counts': skip_reasons,
        'reg_tta3d/peak_memory_mb': torch.cuda.max_memory_allocated(model_device) / (1024 ** 2),
    }
    return _finalize_results(
        cfg, dataset, result_dir, final_output_dir, logger, det_annos,
        pred_class_counter, metadata, epoch_id, start_time,
    )
