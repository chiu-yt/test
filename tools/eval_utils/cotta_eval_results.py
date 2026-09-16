import pickle
import time


def _finalize_results(cfg, dataset, result_dir, final_output_dir, logger,
                      metric, det_annos, gt_class_counter, pred_class_counter,
                      trainable_count, total_count, epoch_id, start_time):
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)
    if cfg.LOCAL_RANK != 0:
        return {}

    result = {}
    gt_num = metric['gt_num']
    for threshold in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        roi_recall = metric['recall_roi_%s' % threshold] / max(gt_num, 1)
        rcnn_recall = metric['recall_rcnn_%s' % threshold] / max(gt_num, 1)
        logger.info('recall_roi_%s: %f' % (threshold, roi_recall))
        logger.info('recall_rcnn_%s: %f' % (threshold, rcnn_recall))
        result['recall/roi_%s' % threshold] = roi_recall
        result['recall/rcnn_%s' % threshold] = rcnn_recall

    total_predictions = sum(len(annotation['name']) for annotation in det_annos)
    logger.info('Average predicted number of objects(%d samples): %.3f' % (
        len(det_annos), total_predictions / max(len(det_annos), 1)
    ))
    gt_total = max(sum(gt_class_counter.values()), 1)
    pred_total = max(sum(pred_class_counter.values()), 1)
    logger.info('-------- D1 GT Class Distribution (count / ratio) --------')
    for name in dataset.class_names:
        count = gt_class_counter[name]
        logger.info('GT[%s]: %d / %.4f' % (name, count, count / gt_total))
    logger.info('-------- D2 Pred Class Distribution (count / ratio) ------')
    for name in dataset.class_names:
        count = pred_class_counter[name]
        logger.info('Pred[%s]: %d / %.4f' % (name, count, count / pred_total))
        result['diag/gt_count_%s' % name] = gt_class_counter[name]
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
    result['cotta/trainable_params'] = trainable_count
    result['cotta/trainable_ratio'] = trainable_count / max(total_count, 1)
    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************CoTTA evaluation done.*****************')
    return result


def _gradient_norm(parameters):
    squared_norm = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        parameter_norm = parameter.grad.detach().norm(2).square()
        squared_norm = parameter_norm if squared_norm is None else squared_norm + parameter_norm
    return 0.0 if squared_norm is None else float(squared_norm.sqrt().item())


def _ema_parameter_delta(teacher, student, alpha):
    student_parameters = dict(student.named_parameters())
    delta_sum = 0.0
    parameter_count = 0
    for name, teacher_parameter in teacher.named_parameters():
        if not (teacher_parameter.is_floating_point() or teacher_parameter.is_complex()):
            continue
        difference = student_parameters[name].detach() - teacher_parameter.detach()
        delta_sum += float(difference.abs().sum().item())
        parameter_count += difference.numel()
    return (1.0 - alpha) * delta_sum / max(parameter_count, 1)
