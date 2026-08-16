import math

import numpy as np
import torch


def prediction_arrays(pred_dict):
    boxes = pred_dict.get('pred_boxes', None)
    labels = pred_dict.get('pred_labels', None)
    scores = pred_dict.get('pred_scores', None)
    if boxes is None or labels is None or scores is None:
        return np.zeros((0, 7), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32), None

    boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
    num_boxes = min(boxes_np.shape[0], labels_np.reshape(-1).shape[0], scores_np.reshape(-1).shape[0])
    return (
        np.asarray(boxes_np[:num_boxes, :7], dtype=np.float32),
        np.asarray(labels_np[:num_boxes], dtype=np.int64).reshape(-1),
        np.asarray(scores_np[:num_boxes], dtype=np.float32).reshape(-1),
        boxes,
    )


def prediction_valid_mask(boxes, labels, scores):
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    valid = np.isfinite(boxes[:, :7]).all(axis=1)
    if boxes.shape[1] >= 6:
        valid &= np.isfinite(boxes[:, 3:6]).all(axis=1)
        valid &= np.all(boxes[:, 3:6] > 0.0, axis=1)

    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    valid &= np.isfinite(labels)
    valid &= np.isfinite(scores)
    valid &= labels > 0
    return valid


def class_score_thresholds(labels, class_score_thresh):
    if class_score_thresh is None:
        return None

    thresholds = np.asarray(class_score_thresh, dtype=np.float32).reshape(-1)
    if thresholds.size == 0:
        return None

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    valid_labels = np.clip(labels, 1, thresholds.shape[0])
    return thresholds[valid_labels - 1]


def staged_filter_masks(
    boxes,
    labels,
    scores,
    target_class_ids,
    min_proposal_score,
    class_score_thresh,
    camera_support=None,
    camera_rescue_enabled=False,
    camera_low_score=0.05,
    camera_thresh=0.60,
):
    finite_mask = prediction_valid_mask(boxes, labels, scores)
    class_mask = finite_mask.copy()
    target_ids = None if target_class_ids is None else np.asarray(target_class_ids, dtype=np.int64).reshape(-1)
    if target_ids is not None and target_ids.size > 0:
        class_mask &= np.isin(np.abs(np.asarray(labels, dtype=np.int64).reshape(-1)), target_ids)

    min_score_mask = class_mask.copy()
    min_score_mask &= np.asarray(scores, dtype=np.float32).reshape(-1) >= float(min_proposal_score)

    score_mask = min_score_mask.copy()
    thresholds = class_score_thresholds(labels, class_score_thresh)
    if thresholds is not None:
        score_mask &= np.asarray(scores, dtype=np.float32).reshape(-1) >= thresholds

    rescue_mask = np.zeros_like(score_mask, dtype=bool)
    if camera_rescue_enabled and camera_support is not None and boxes.shape[0] > 0:
        support = np.asarray(camera_support, dtype=np.float32).reshape(-1)
        if support.shape[0] == boxes.shape[0]:
            score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
            low_score_mask = score_values >= float(camera_low_score)
            rescue_mask = class_mask & low_score_mask & (~score_mask) & (support >= float(camera_thresh))
            score_mask |= rescue_mask
    return finite_mask, class_mask, min_score_mask, score_mask, rescue_mask


def topk_per_class_mask(labels, scores, valid_mask, topk_per_class):
    if int(topk_per_class) <= 0 or not bool(np.any(valid_mask)):
        return valid_mask

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    selected_mask = np.zeros_like(valid_mask, dtype=bool)
    for class_id in np.unique(np.abs(labels[valid_mask])):
        class_mask = valid_mask & (np.abs(labels) == int(class_id))
        class_indices = np.flatnonzero(class_mask)
        if class_indices.size == 0:
            continue
        order = np.argsort(-scores[class_indices])[: int(topk_per_class)]
        selected_mask[class_indices[order]] = True
    return selected_mask


def reliability_summary(reliability_sum, num_matched, clean_valid_count, perturbed_valid_count, support_tau):
    n_ref = min(float(clean_valid_count), float(perturbed_valid_count))
    if n_ref < 2.0 or num_matched < 2:
        return 0.0, 0.0, 0.0, n_ref

    coverage = float(num_matched) / max(n_ref, 1.0)
    support = 1.0 - math.exp(-float(num_matched) / max(float(support_tau), 1e-6))
    instance_quality = float(reliability_sum) / max(float(num_matched), 1.0)
    return instance_quality * coverage * support, coverage, support, n_ref
