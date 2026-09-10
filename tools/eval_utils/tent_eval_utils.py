import pickle
import time

import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.tta_methods.tent_entropy import entropy_loss_from_logits
from pcdet.tta_methods.tent_hooks import TransFusionLogitCapture
from pcdet.tta_methods.tent_utils import (
    build_tent_optimizer,
    changed_parameter_names,
    clone_named_parameters,
    configure_model_for_tent,
    restore_tent_state,
    snapshot_tent_state,
    trainable_parameter_names,
    unwrap_model,
)
from .eval_utils import (
    _init_class_counter,
    _update_gt_class_counter_from_batch,
    _update_pred_class_counter_from_annos,
    statistics_info,
)


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


def _grad_norm(parameters):
    total = None
    for param in parameters:
        if param.grad is None:
            continue
        norm = param.grad.detach().data.norm(2)
        total = norm.pow(2) if total is None else total + norm.pow(2)
    if total is None:
        return 0.0
    return float(total.sqrt().item())


def _log_param_debug(logger, before_params, model, trainable_names):
    changed = changed_parameter_names(before_params, model)
    illegal_changed = [name for name in changed if name not in set(trainable_names)]
    logger.info('[Tent] changed params after debug step: %s' % changed)
    if illegal_changed:
        raise AssertionError('Tent changed non-BN-affine parameters: %s' % illegal_changed)


def eval_tent_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    tent_cfg = cfg.TTA.TENT
    if dist_test and not bool(tent_cfg.get('ALLOW_DDP', False)):
        raise NotImplementedError('Tent online eval is single-process by default; set TTA.TENT.ALLOW_DDP True only after validating rank-local ordering')
    if result_dir is None:
        raise RuntimeError('Tent eval requires a result_dir')

    result_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {'gt_num': 0}
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    gt_class_counter = _init_class_counter(class_names)
    pred_class_counter = _init_class_counter(class_names)

    logger.info('*************** EPOCH %s TENT EVALUATION *****************' % epoch_id)
    params, param_names, bn_count, trainable_count, total_count = configure_model_for_tent(model, tent_cfg, logger)
    if not params:
        raise RuntimeError('Tent found no BN-family affine parameters to optimize')
    optimizer = build_tent_optimizer(params, tent_cfg)
    model_state = optimizer_state = None
    if bool(tent_cfg.get('EPISODIC', False)):
        model_state, optimizer_state = snapshot_tent_state(model, optimizer)

    use_sigmoid = _head_uses_sigmoid(model, cfg)
    logger.info('[Tent] entropy logits source: dense_head.predict()["heatmap"]')
    logger.info('[Tent] entropy mode: %s (use_sigmoid_cls=%s)' % (tent_cfg.get('ENTROPY_MODE', 'auto'), use_sigmoid))

    progress_bar = None
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='tent_eval', dynamic_ncols=True)

    start_time = time.time()
    steps = int(tent_cfg.get('STEPS', 1))
    log_interval = int(tent_cfg.get('LOG_INTERVAL', 50))
    min_valid_terms = int(tent_cfg.get('MIN_VALID_TERMS', 1))
    debug_param_check = bool(tent_cfg.get('DEBUG_PARAM_CHECK', False))

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        _update_gt_class_counter_from_batch(batch_dict, class_names, gt_class_counter)
        if model_state is not None and optimizer_state is not None:
            restore_tent_state(model, optimizer, model_state, optimizer_state)

        before_params = clone_named_parameters(model) if debug_param_check and i == 0 else None
        grad_norm_value = 0.0

        optimizer.zero_grad()
        with torch.enable_grad():
            with TransFusionLogitCapture(model) as capture:
                pred_dicts, ret_dict = model(batch_dict)
            pred_dicts_for_eval = _detach_tensor_tree(pred_dicts)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts_for_eval, class_names,
                output_path=final_output_dir if args.save_to_file else None
            )

            loss = None
            loss_diag = {'valid_terms': 0, 'finite': False, 'shape': None, 'mode': tent_cfg.get('ENTROPY_MODE', 'auto')}
            for _ in range(max(steps, 1)):
                loss, loss_diag = entropy_loss_from_logits(
                    capture.logits,
                    entropy_mode=tent_cfg.get('ENTROPY_MODE', 'auto'),
                    min_valid_terms=min_valid_terms,
                    use_sigmoid=use_sigmoid,
                )
                if loss is None or not loss_diag['finite']:
                    break
                loss.backward(retain_graph=False)
                grad_norm_value = _grad_norm(params)
                clip_val = float(tent_cfg.get('GRAD_NORM_CLIP', 0.0))
                if clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(params, clip_val)
                optimizer.step()
                optimizer.zero_grad()
                break

        if before_params is not None:
            _log_param_debug(logger, before_params, model, trainable_parameter_names(model))

        disp_dict = {}
        statistics_info(cfg, ret_dict, metric, disp_dict)
        _update_pred_class_counter_from_annos(annos, pred_class_counter)
        det_annos += annos

        loss_value = float(loss.detach().item()) if loss is not None and loss_diag['finite'] else float('nan')
        disp_dict.update({
            'tent_entropy': '%.4f' % loss_value,
            'tent_terms': int(loss_diag['valid_terms']),
            'tent_grad': '%.4f' % grad_norm_value,
        })
        if i == 0 or (log_interval > 0 and i % log_interval == 0):
            logger.info(
                '[Tent] batch=%d entropy=%s terms=%s grad_norm=%.6f trainable=%d/%d bn=%d finite=%s logits_shape=%s'
                % (i, loss_value, loss_diag['valid_terms'], grad_norm_value, trainable_count, total_count, bn_count, loss_diag['finite'], loss_diag['shape'])
            )

        if progress_bar is not None:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if progress_bar is not None:
        progress_bar.close()

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = sum(anno['name'].__len__() for anno in det_annos)
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    gt_total = max(sum(gt_class_counter.values()), 1)
    pred_total = max(sum(pred_class_counter.values()), 1)
    logger.info('-------- D1 GT Class Distribution (count / ratio) --------')
    for name in class_names:
        c = gt_class_counter[name]
        logger.info(f'GT[{name}]: {c} / {c / gt_total:.4f}')

    logger.info('-------- D2 Pred Class Distribution (count / ratio) ------')
    for name in class_names:
        c = pred_class_counter[name]
        logger.info(f'Pred[{name}]: {c} / {c / pred_total:.4f}')

    for name in class_names:
        ret_dict[f'diag/gt_count_{name}'] = gt_class_counter[name]
        ret_dict[f'diag/pred_count_{name}'] = pred_class_counter[name]

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )
    logger.info(result_str)
    ret_dict.update(result_dict)
    ret_dict['tent/bn_modules'] = bn_count
    ret_dict['tent/trainable_params'] = trainable_count
    ret_dict['tent/trainable_ratio'] = trainable_count / max(total_count, 1)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Tent evaluation done.*****************')
    return ret_dict
