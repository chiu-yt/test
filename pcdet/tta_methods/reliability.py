import math

import numpy as np
import torch


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


def compute_spcra_reliability(
    clean_pred_dicts,
    perturbed_pred_dicts,
    max_center_distance=1.0,
    reliability_floor=0.05,
):
    """Compare clean and sparse-perturbed predictions and return per-box weights."""
    sidecars = []
    stats = {
        'clean_boxes': 0,
        'perturbed_boxes': 0,
        'matched_boxes': 0,
        'reliability_sum': 0.0,
        'reliability_min': 1.0,
    }
    distance_limit = max(float(max_center_distance), 1e-6)
    floor = float(np.clip(reliability_floor, 0.0, 1.0))

    for clean_pred, perturbed_pred in zip(clean_pred_dicts, perturbed_pred_dicts):
        clean_boxes, clean_labels, clean_scores, clean_box_tensor = _prediction_arrays(clean_pred)
        perturbed_boxes, perturbed_labels, perturbed_scores, _ = _prediction_arrays(perturbed_pred)
        reliability = np.full(clean_boxes.shape[0], floor, dtype=np.float32)
        used_perturbed = set()

        stats['clean_boxes'] += int(clean_boxes.shape[0])
        stats['perturbed_boxes'] += int(perturbed_boxes.shape[0])
        for clean_index in range(clean_boxes.shape[0]):
            candidate_indices = np.where(clean_labels[clean_index] == perturbed_labels)[0]
            candidate_indices = [idx for idx in candidate_indices.tolist() if idx not in used_perturbed]
            if not candidate_indices:
                continue

            center_delta = perturbed_boxes[candidate_indices, :3] - clean_boxes[clean_index, :3]
            center_distances = np.linalg.norm(center_delta, axis=1)
            nearest_local = int(np.argmin(center_distances))
            perturbed_index = candidate_indices[nearest_local]
            center_distance = float(center_distances[nearest_local])
            if center_distance > distance_limit:
                continue

            used_perturbed.add(perturbed_index)
            dimension_scale = max(float(np.abs(clean_boxes[clean_index, 3:6]).mean()), 1e-3)
            dimension_cost = float(
                np.abs(clean_boxes[clean_index, 3:6] - perturbed_boxes[perturbed_index, 3:6]).mean()
                / dimension_scale
            )
            score_cost = abs(float(clean_scores[clean_index] - perturbed_scores[perturbed_index]))
            cost = center_distance / distance_limit + 0.5 * dimension_cost + 0.5 * score_cost
            reliability[clean_index] = max(floor, min(1.0, math.exp(-cost)))
            stats['matched_boxes'] += 1

        if clean_box_tensor is not None and torch.is_tensor(clean_box_tensor):
            sidecars.append(torch.as_tensor(reliability, dtype=torch.float32, device=clean_box_tensor.device))
        else:
            sidecars.append(torch.as_tensor(reliability, dtype=torch.float32))
        if reliability.size > 0:
            stats['reliability_sum'] += float(reliability.sum())
            stats['reliability_min'] = min(stats['reliability_min'], float(reliability.min()))

    clean_count = max(float(stats['clean_boxes']), 1.0)
    stats['match_rate'] = float(stats['matched_boxes']) / clean_count
    stats['reliability_mean'] = stats['reliability_sum'] / clean_count
    return sidecars, stats
