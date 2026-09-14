import pickle
import time


def _finalize_sar_results(cfg, dataset, result_dir, final_output_dir, logger,
                          det_annos, pred_class_counter, sar_metadata,
                          epoch_id, start_time):
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / max(len(dataset), 1)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)
    if cfg.LOCAL_RANK != 0:
        return {}

    result = {}
    total_predictions = sum(len(annotation['name']) for annotation in det_annos)
    logger.info('Average predicted number of objects(%d samples): %.3f' % (
        len(det_annos), total_predictions / max(len(det_annos), 1)
    ))
    pred_total = max(sum(pred_class_counter.values()), 1)
    logger.info('-------- SAR First-Forward Pred Class Distribution --------')
    for name in dataset.class_names:
        count = pred_class_counter[name]
        logger.info('Pred[%s]: %d / %.4f' % (name, count, count / pred_total))
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
    result.update(sar_metadata)
    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************SAR evaluation done.*****************')
    return result
