import math

import numpy as np
import torch

from pcdet.tta_methods.spcra_filter_utils import prediction_arrays, reliability_summary, staged_filter_masks, topk_per_class_mask
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


def compute_spcra_reliability(
    clean_pred_dicts,
    perturbed_pred_dicts,
    perturbed_lidar_aug_matrices=None,
    clean_camera_supports=None,
    perturbed_camera_supports=None,
    target_class_ids=None,
    min_proposal_score=0.0,
    topk_per_class=0,
    class_score_thresh=None,
    max_center_distance=1.0,
    reliability_floor=0.05,
    support_tau=3.0,
    reliability_clamp_max=1.0,
    camera_rescue_enabled=False,
    camera_low_score=0.05,
    camera_thresh=0.60,
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
        'clean_rescue_boxes': 0,
        'perturbed_rescue_boxes': 0,
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
    clamp_max = float(np.clip(reliability_clamp_max, 0.0, 1.0))

    for frame_index, (clean_pred, perturbed_pred) in enumerate(zip(clean_pred_dicts, perturbed_pred_dicts)):
        clean_boxes, clean_labels, clean_scores, clean_box_tensor = prediction_arrays(clean_pred)
        perturbed_boxes, perturbed_labels, perturbed_scores, _ = prediction_arrays(perturbed_pred)
        clean_camera_support = None
        perturbed_camera_support = None
        if clean_camera_supports is not None and frame_index < len(clean_camera_supports):
            clean_camera_support = clean_camera_supports[frame_index]
        if perturbed_camera_supports is not None and frame_index < len(perturbed_camera_supports):
            perturbed_camera_support = perturbed_camera_supports[frame_index]
        clean_prefilter_valid_mask, clean_class_mask, clean_min_score_mask, clean_score_mask, clean_rescue_mask = staged_filter_masks(
            clean_boxes,
            clean_labels,
            clean_scores,
            target_class_ids,
            min_proposal_score,
            class_score_thresh,
            camera_support=clean_camera_support,
            camera_rescue_enabled=camera_rescue_enabled,
            camera_low_score=camera_low_score,
            camera_thresh=camera_thresh,
        )
        perturbed_prefilter_valid_mask, perturbed_class_mask, perturbed_min_score_mask, perturbed_score_mask, perturbed_rescue_mask = staged_filter_masks(
            perturbed_boxes,
            perturbed_labels,
            perturbed_scores,
            target_class_ids,
            min_proposal_score,
            class_score_thresh,
            camera_support=perturbed_camera_support,
            camera_rescue_enabled=camera_rescue_enabled,
            camera_low_score=camera_low_score,
            camera_thresh=camera_thresh,
        )
        clean_valid_mask = clean_score_mask.copy()
        perturbed_valid_mask = perturbed_score_mask.copy()
        clean_valid_mask = topk_per_class_mask(clean_labels, clean_scores, clean_valid_mask, topk_per_class)
        perturbed_valid_mask = topk_per_class_mask(perturbed_labels, perturbed_scores, perturbed_valid_mask, topk_per_class)
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
        stats['clean_rescue_boxes'] += int(clean_rescue_mask.sum())
        stats['perturbed_rescue_boxes'] += int(perturbed_rescue_mask.sum())
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

        reliability_used, coverage, support, n_ref = reliability_summary(
            frame_reliability_sum,
            frame_matched_boxes,
            clean_valid_indices.size,
            perturbed_valid_indices.size,
            support_tau,
        )
        reliability_used = min(reliability_used, clamp_max)
        reliability *= float(coverage * support)
        reliability = np.minimum(reliability, clamp_max)
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
