import math

import numpy as np
import torch

from pcdet.utils.tta_utils import _apply_lidar_aug_matrix_to_boxes_np


def apply_sparse_point_perturbation(batch_dict, drop_rate):
    """Drop a sparse subset of points while retaining at least one per frame."""
    points = batch_dict.get('points', None)
    if points is None or not torch.is_tensor(points) or points.shape[0] == 0:
        return batch_dict

    rate = float(np.clip(drop_rate, 0.0, 0.95))
    if rate <= 0.0:
        return batch_dict

    keep_mask = torch.rand(points.shape[0], device=points.device) >= rate
    batch_indices = points[:, 0].long()
    for batch_index in torch.unique(batch_indices).tolist():
        frame_indices = torch.where(batch_indices == batch_index)[0]
        if frame_indices.numel() > 0 and not bool(keep_mask[frame_indices].any()):
            keep_mask[frame_indices[0]] = True

    batch_dict['points'] = points[keep_mask]
    return batch_dict


def _prediction_arrays(pred_dict):
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


def _prediction_valid_mask(boxes, labels, scores):
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


def _class_score_thresholds(labels, class_score_thresh):
    if class_score_thresh is None:
        return None

    thresholds = np.asarray(class_score_thresh, dtype=np.float32).reshape(-1)
    if thresholds.size == 0:
        return None

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    valid_labels = np.clip(labels, 1, thresholds.shape[0])
    return thresholds[valid_labels - 1]


def _effective_valid_mask(boxes, labels, scores, target_class_ids, min_proposal_score, class_score_thresh):
    valid_mask = _prediction_valid_mask(boxes, labels, scores)
    if boxes.shape[0] == 0:
        return valid_mask

    target_ids = None if target_class_ids is None else np.asarray(target_class_ids, dtype=np.int64).reshape(-1)
    if target_ids is not None and target_ids.size > 0:
        valid_mask &= np.isin(np.abs(np.asarray(labels, dtype=np.int64).reshape(-1)), target_ids)

    valid_mask &= np.asarray(scores, dtype=np.float32).reshape(-1) >= float(min_proposal_score)

    class_thresholds = _class_score_thresholds(labels, class_score_thresh)
    if class_thresholds is not None:
        valid_mask &= np.asarray(scores, dtype=np.float32).reshape(-1) >= class_thresholds
    return valid_mask


def _staged_filter_masks(boxes, labels, scores, target_class_ids, min_proposal_score, class_score_thresh):
    finite_mask = _prediction_valid_mask(boxes, labels, scores)
    class_mask = finite_mask.copy()
    target_ids = None if target_class_ids is None else np.asarray(target_class_ids, dtype=np.int64).reshape(-1)
    if target_ids is not None and target_ids.size > 0:
        class_mask &= np.isin(np.abs(np.asarray(labels, dtype=np.int64).reshape(-1)), target_ids)

    min_score_mask = class_mask.copy()
    min_score_mask &= np.asarray(scores, dtype=np.float32).reshape(-1) >= float(min_proposal_score)

    score_mask = min_score_mask.copy()
    class_thresholds = _class_score_thresholds(labels, class_score_thresh)
    if class_thresholds is not None:
        score_mask &= np.asarray(scores, dtype=np.float32).reshape(-1) >= class_thresholds
    return finite_mask, class_mask, min_score_mask, score_mask


def _reliability_summary(reliability_sum, num_matched, clean_valid_count, perturbed_valid_count, support_tau):
    n_ref = min(float(clean_valid_count), float(perturbed_valid_count))
    if n_ref < 2.0 or num_matched < 2:
        return 0.0, 0.0, 0.0, n_ref

    coverage = float(num_matched) / max(n_ref, 1.0)
    support = 1.0 - math.exp(-float(num_matched) / max(float(support_tau), 1e-6))
    instance_quality = float(reliability_sum) / max(float(num_matched), 1.0)
    return instance_quality * coverage * support, coverage, support, n_ref


def _topk_per_class_mask(labels, scores, valid_mask, topk_per_class):
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


def compute_spcra_reliability(
    clean_pred_dicts,
    perturbed_pred_dicts,
    perturbed_lidar_aug_matrices=None,
    target_class_ids=None,
    min_proposal_score=0.0,
    topk_per_class=0,
    class_score_thresh=None,
    max_center_distance=1.0,
    reliability_floor=0.05,
    support_tau=3.0,
):
    """Compare clean and sparse-perturbed predictions and return per-box weights."""
    sidecars = []
    stats = {
        'clean_boxes': 0,
        'perturbed_boxes': 0,
        'clean_prefilter_valid_boxes': 0,
        'perturbed_prefilter_valid_boxes': 0,
        'clean_class_valid_boxes': 0,
        'perturbed_class_valid_boxes': 0,
        'clean_min_score_valid_boxes': 0,
        'perturbed_min_score_valid_boxes': 0,
        'clean_score_valid_boxes': 0,
        'perturbed_score_valid_boxes': 0,
        'clean_valid_boxes': 0,
        'perturbed_valid_boxes': 0,
        'clean_filtered_boxes': 0,
        'perturbed_filtered_boxes': 0,
        'matched_boxes': 0,
        'reliability_sum': 0.0,
        'reliability_raw_sum': 0.0,
        'coverage_sum': 0.0,
        'support_sum': 0.0,
        'effective_ref_boxes': 0.0,
        'effective_frames': 0,
        'reliability_min': 1.0,
    }
    distance_limit = max(float(max_center_distance), 1e-6)
    floor = float(np.clip(reliability_floor, 0.0, 1.0))

    for clean_pred, perturbed_pred in zip(clean_pred_dicts, perturbed_pred_dicts):
        clean_boxes, clean_labels, clean_scores, clean_box_tensor = _prediction_arrays(clean_pred)
        perturbed_boxes, perturbed_labels, perturbed_scores, _ = _prediction_arrays(perturbed_pred)
        clean_prefilter_valid_mask, clean_class_mask, clean_min_score_mask, clean_score_mask = _staged_filter_masks(
            clean_boxes,
            clean_labels,
            clean_scores,
            target_class_ids,
            min_proposal_score,
            class_score_thresh,
        )
        perturbed_prefilter_valid_mask, perturbed_class_mask, perturbed_min_score_mask, perturbed_score_mask = _staged_filter_masks(
            perturbed_boxes,
            perturbed_labels,
            perturbed_scores,
            target_class_ids,
            min_proposal_score,
            class_score_thresh,
        )
        clean_valid_mask = clean_score_mask.copy()
        perturbed_valid_mask = perturbed_score_mask.copy()
        clean_valid_mask = _topk_per_class_mask(clean_labels, clean_scores, clean_valid_mask, topk_per_class)
        perturbed_valid_mask = _topk_per_class_mask(perturbed_labels, perturbed_scores, perturbed_valid_mask, topk_per_class)
        if perturbed_lidar_aug_matrices is not None:
            perturbed_index = len(sidecars)
            if perturbed_index < len(perturbed_lidar_aug_matrices):
                perturbed_boxes = _apply_lidar_aug_matrix_to_boxes_np(
                    perturbed_boxes,
                    perturbed_lidar_aug_matrices[perturbed_index],
                    inverse=True,
                )

        reliability = np.zeros(clean_boxes.shape[0], dtype=np.float32)
        used_perturbed = set()

        stats['clean_boxes'] += int(clean_boxes.shape[0])
        stats['perturbed_boxes'] += int(perturbed_boxes.shape[0])
        stats['clean_prefilter_valid_boxes'] += int(clean_prefilter_valid_mask.sum())
        stats['perturbed_prefilter_valid_boxes'] += int(perturbed_prefilter_valid_mask.sum())
        stats['clean_class_valid_boxes'] += int(clean_class_mask.sum())
        stats['perturbed_class_valid_boxes'] += int(perturbed_class_mask.sum())
        stats['clean_min_score_valid_boxes'] += int(clean_min_score_mask.sum())
        stats['perturbed_min_score_valid_boxes'] += int(perturbed_min_score_mask.sum())
        stats['clean_score_valid_boxes'] += int(clean_score_mask.sum())
        stats['perturbed_score_valid_boxes'] += int(perturbed_score_mask.sum())
        stats['clean_valid_boxes'] += int(clean_valid_mask.sum())
        stats['perturbed_valid_boxes'] += int(perturbed_valid_mask.sum())
        stats['clean_filtered_boxes'] += int(clean_prefilter_valid_mask.sum() - clean_valid_mask.sum())
        stats['perturbed_filtered_boxes'] += int(perturbed_prefilter_valid_mask.sum() - perturbed_valid_mask.sum())

        clean_valid_indices = np.where(clean_valid_mask)[0]
        perturbed_valid_indices = np.where(perturbed_valid_mask)[0]
        if clean_valid_indices.size == 0 or perturbed_valid_indices.size == 0:
            if clean_box_tensor is not None and torch.is_tensor(clean_box_tensor):
                sidecars.append(torch.as_tensor(reliability, dtype=torch.float32, device=clean_box_tensor.device))
            else:
                sidecars.append(torch.as_tensor(reliability, dtype=torch.float32))
            continue

        clean_valid_boxes = clean_boxes[clean_valid_indices]
        clean_valid_labels = clean_labels[clean_valid_indices]
        perturbed_valid_boxes = perturbed_boxes[perturbed_valid_indices]
        perturbed_valid_labels = perturbed_labels[perturbed_valid_indices]
        perturbed_valid_scores = perturbed_scores[perturbed_valid_indices]

        frame_reliability_sum = 0.0
        frame_matched_boxes = 0
        for local_index, clean_index in enumerate(clean_valid_indices.tolist()):
            candidate_indices = np.where(clean_valid_labels[local_index] == perturbed_valid_labels)[0]
            candidate_indices = [idx for idx in candidate_indices.tolist() if idx not in used_perturbed]
            if not candidate_indices:
                continue

            center_delta = perturbed_valid_boxes[candidate_indices, :3] - clean_valid_boxes[local_index, :3]
            center_distances = np.linalg.norm(center_delta, axis=1)
            nearest_local = int(np.argmin(center_distances))
            perturbed_index = candidate_indices[nearest_local]
            center_distance = float(center_distances[nearest_local])
            if center_distance > distance_limit:
                continue

            used_perturbed.add(perturbed_index)
            dimension_scale = max(float(np.abs(clean_valid_boxes[local_index, 3:6]).mean()), 1e-3)
            dimension_cost = float(
                np.abs(clean_valid_boxes[local_index, 3:6] - perturbed_valid_boxes[perturbed_index, 3:6]).mean()
                / dimension_scale
            )
            score_cost = abs(float(clean_scores[clean_index] - perturbed_valid_scores[perturbed_index]))
            cost = center_distance / distance_limit + 0.5 * dimension_cost + 0.5 * score_cost
            reliability[clean_index] = max(floor, min(1.0, math.exp(-cost)))
            stats['matched_boxes'] += 1
            frame_matched_boxes += 1
            frame_reliability_sum += float(reliability[clean_index])

        reliability_used, coverage, support, n_ref = _reliability_summary(
            frame_reliability_sum,
            frame_matched_boxes,
            clean_valid_indices.size,
            perturbed_valid_indices.size,
            support_tau,
        )
        reliability *= float(coverage * support)
        stats['reliability_sum'] += reliability_used
        stats['reliability_raw_sum'] += frame_reliability_sum
        stats['coverage_sum'] += coverage
        stats['support_sum'] += support
        stats['effective_ref_boxes'] += n_ref
        stats['effective_frames'] += 1

        if clean_box_tensor is not None and torch.is_tensor(clean_box_tensor):
            sidecars.append(torch.as_tensor(reliability, dtype=torch.float32, device=clean_box_tensor.device))
        else:
            sidecars.append(torch.as_tensor(reliability, dtype=torch.float32))
        if reliability.size > 0:
            stats['reliability_min'] = min(stats['reliability_min'], float(reliability.min()))

    clean_count = max(float(stats['clean_boxes']), 1.0)
    valid_count = max(min(float(stats['clean_valid_boxes']), float(stats['perturbed_valid_boxes'])), 1.0)
    frame_count = max(float(stats['effective_frames']), 1.0)
    stats['match_rate'] = float(stats['matched_boxes']) / valid_count
    stats['reliability_raw_mean'] = stats['reliability_raw_sum'] / clean_count
    stats['reliability_mean'] = stats['reliability_sum'] / frame_count
    stats['coverage_mean'] = stats['coverage_sum'] / frame_count
    stats['support_mean'] = stats['support_sum'] / frame_count
    return sidecars, stats
