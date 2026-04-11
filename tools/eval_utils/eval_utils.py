import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def _init_class_counter(class_names):
    return {name: 0 for name in class_names}


def _update_gt_class_counter_from_batch(batch_dict, class_names, class_counter):
    """D1: 统计当前 batch 的 GT 类别分布（按 gt_boxes 最后一列标签）。"""
    gt_boxes = batch_dict.get('gt_boxes', None)
    if gt_boxes is None:
        return

    if not isinstance(gt_boxes, torch.Tensor):
        gt_boxes = torch.as_tensor(gt_boxes)

    if gt_boxes.numel() == 0:
        return

    # [B, M, C] -> [B*M, C]
    if gt_boxes.dim() == 3:
        gt_boxes = gt_boxes.reshape(-1, gt_boxes.shape[-1])
    elif gt_boxes.dim() != 2:
        return

    cls = gt_boxes[:, -1].float()
    cls = cls[torch.isfinite(cls)]
    if cls.numel() == 0:
        return

    num_cls = len(class_names)
    # 兼容 0-based 标签
    if cls.min() >= 0 and cls.max() <= (num_cls - 1):
        cls = cls + 1

    cls = cls.round().long()
    valid = (cls >= 1) & (cls <= num_cls)
    cls = cls[valid]
    if cls.numel() == 0:
        return

    for i, name in enumerate(class_names, start=1):
        class_counter[name] += int((cls == i).sum().item())


def _update_pred_class_counter_from_annos(annos, class_counter):
    """D2: 统计当前 batch 的预测类别分布（按 generate_prediction_dicts 输出的 name）。"""
    for anno in annos:
        names = anno.get('name', [])
        for n in names:
            if n in class_counter:
                class_counter[n] += 1


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    gt_class_counter = _init_class_counter(class_names)
    pred_class_counter = _init_class_counter(class_names)

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        _update_gt_class_counter_from_batch(batch_dict, class_names, gt_class_counter)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        _update_pred_class_counter_from_annos(annos, pred_class_counter)
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    # ========== D1/D2 诊断日志：GT vs Pred 类别分布 ==========
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

    # 写入 ret_dict 便于后续程序化分析
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

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
