from enum import IntEnum

import torch
from scipy.optimize import linear_sum_assignment

from pcdet.ops.iou3d_nms import iou3d_nms_utils


class PseudoAction(IntEnum):
    LOW = 0
    MEDIUM = -1
    HIGH = 1


class DPOCostHistory:
    """Mutable stream-wide storage for finite DPO matching costs."""

    def __init__(self, alpha):
        self.alpha = float(alpha)
        self.values = []

    def append_batch(self, matched_costs):
        finite_costs = matched_costs[torch.isfinite(matched_costs)]
        if finite_costs.numel() > 0:
            self.values.append(finite_costs.detach().float().cpu())

    def thresholds(self, device):
        if not self.values:
            return None
        values = torch.cat(self.values).to(device)
        levels = values.new_tensor([self.alpha, 1.0 - self.alpha])
        return torch.quantile(values, levels)

    def __len__(self):
        return sum(int(values.numel()) for values in self.values)


class DPOCutoffState:
    """Mutable online EMA state that permanently latches an enabled cutoff."""

    def __init__(self, gamma=0.5, c_stop=None, enabled=False):
        self.gamma = float(gamma)
        self.c_stop = None if c_stop is None else float(c_stop)
        self.enabled = bool(enabled)
        self.ema = None
        self.stopped = False

    def update(self, matched_costs):
        finite_costs = matched_costs[torch.isfinite(matched_costs)]
        if finite_costs.numel() == 0:
            return self.stopped
        current_mean = float(finite_costs.detach().float().mean().item())
        if self.ema is None:
            self.ema = current_mean
        else:
            self.ema = self.gamma * self.ema + (1.0 - self.gamma) * current_mean
        if self.enabled and self.c_stop is not None and self.ema <= self.c_stop:
            self.stopped = True
        return self.stopped


def _valid_prediction_mask(prediction):
    boxes = prediction['pred_boxes']
    scores = prediction['pred_scores']
    labels = prediction['pred_labels']
    return (
        torch.isfinite(boxes[:, :7]).all(dim=1)
        & torch.isfinite(scores)
        & (boxes[:, 3:6] > 0).all(dim=1)
        & (labels > 0)
    )


def _thresholds_for_labels(value, labels):
    if isinstance(value, (float, int)):
        return labels.new_full(labels.shape, float(value), dtype=torch.float32)
    thresholds = labels.new_tensor(value, dtype=torch.float32)
    return thresholds[labels - 1]


def build_clean_pseudo_targets(pred_dicts, score_thresh, neg_thresh):
    targets = []
    for prediction in pred_dicts:
        valid = _valid_prediction_mask(prediction)
        boxes = prediction['pred_boxes'][valid].detach()
        scores = prediction['pred_scores'][valid].detach()
        labels = prediction['pred_labels'][valid].detach().long()
        positive_thresholds = _thresholds_for_labels(score_thresh, labels)
        negative_thresholds = _thresholds_for_labels(neg_thresh, labels)
        keep = []
        actions = []
        for index, score in enumerate(scores):
            if score >= positive_thresholds[index]:
                action = PseudoAction.HIGH
            elif score >= negative_thresholds[index]:
                action = PseudoAction.MEDIUM
            else:
                action = PseudoAction.LOW
            if action != PseudoAction.LOW:
                keep.append(index)
                actions.append(int(action))
        index_tensor = labels.new_tensor(keep, dtype=torch.long)
        targets.append({
            'boxes': boxes[index_tensor],
            'scores': scores[index_tensor],
            'labels': labels[index_tensor],
            'actions': labels.new_tensor(actions, dtype=torch.long),
            'source_count': int(scores.numel()),
        })
    return targets


def targets_to_training_tensors(targets):
    max_count = max((target['boxes'].shape[0] for target in targets), default=0)
    first_boxes = targets[0]['boxes']
    gt_boxes = first_boxes.new_zeros((len(targets), max_count, 10))
    actions = targets[0]['actions'].new_zeros((len(targets), max_count))
    for batch_index, target in enumerate(targets):
        count = target['boxes'].shape[0]
        if count == 0:
            continue
        boxes = target['boxes']
        box_values = boxes.new_zeros((count, 9))
        copied_dims = min(int(boxes.shape[1]), 9)
        box_values[:, :copied_dims] = boxes[:, :copied_dims]
        gt_boxes[batch_index, :count, :9] = box_values
        gt_boxes[batch_index, :count, 9] = target['labels'].to(gt_boxes.dtype)
        actions[batch_index, :count] = target['actions']
    return gt_boxes, actions


def _pairwise_cost(clean_boxes, disturbed_boxes):
    iou = iou3d_nms_utils.boxes_iou3d_gpu(
        clean_boxes[:, :7].contiguous(), disturbed_boxes[:, :7].contiguous()
    )
    l1 = torch.cdist(clean_boxes[:, :7], disturbed_boxes[:, :7], p=1)
    return -iou + 2.0 * l1


def _hungarian_same_class_matches(clean_boxes, clean_labels, disturbed_boxes, disturbed_labels):
    row_parts = []
    col_parts = []
    cost_parts = []
    for class_id in torch.unique(clean_labels):
        clean_indices = torch.where(clean_labels == class_id)[0]
        disturbed_indices = torch.where(disturbed_labels == class_id)[0]
        if clean_indices.numel() == 0 or disturbed_indices.numel() == 0:
            continue
        costs = _pairwise_cost(clean_boxes[clean_indices], disturbed_boxes[disturbed_indices])
        matched_rows, matched_cols = linear_sum_assignment(costs.detach().cpu().numpy())
        matched_rows = clean_indices.new_tensor(matched_rows, dtype=torch.long)
        matched_cols = disturbed_indices.new_tensor(matched_cols, dtype=torch.long)
        row_parts.append(clean_indices[matched_rows])
        col_parts.append(disturbed_indices[matched_cols])
        cost_parts.append(costs[matched_rows, matched_cols])
    if not row_parts:
        empty_index = clean_labels.new_empty((0,), dtype=torch.long)
        return empty_index, empty_index, clean_boxes.new_empty((0,))
    return torch.cat(row_parts), torch.cat(col_parts), torch.cat(cost_parts)


def match_refined_targets(clean_targets, disturbed_pred_dicts, history):
    batch_matches = []
    all_costs = []
    for clean, prediction in zip(clean_targets, disturbed_pred_dicts):
        valid = _valid_prediction_mask(prediction)
        disturbed_boxes = prediction['pred_boxes'][valid].detach()
        disturbed_labels = prediction['pred_labels'][valid].detach().long()
        rows, columns, matched_costs = _hungarian_same_class_matches(
            clean['boxes'], clean['labels'], disturbed_boxes, disturbed_labels
        )
        finite = torch.isfinite(matched_costs)
        batch_matches.append((rows[finite], columns[finite], matched_costs[finite]))
        all_costs.append(matched_costs[finite])
    nonempty = [costs for costs in all_costs if costs.numel() > 0]
    matched_costs = torch.cat(nonempty) if nonempty else clean_targets[0]['boxes'].new_empty((0,))
    history.append_batch(matched_costs)
    thresholds = history.thresholds(matched_costs.device)
    refined = []
    for clean, (rows, _, costs) in zip(clean_targets, batch_matches):
        actions = clean['actions'].new_full(clean['actions'].shape, int(PseudoAction.LOW))
        if thresholds is not None:
            actions[rows] = int(PseudoAction.MEDIUM)
            actions[rows[costs < thresholds[0]]] = int(PseudoAction.HIGH)
            actions[rows[costs > thresholds[1]]] = int(PseudoAction.LOW)
        keep = actions != int(PseudoAction.LOW)
        refined.append({
            'boxes': clean['boxes'][keep],
            'scores': clean['scores'][keep],
            'labels': clean['labels'][keep],
            'actions': actions[keep],
        })
    return refined, matched_costs, thresholds
