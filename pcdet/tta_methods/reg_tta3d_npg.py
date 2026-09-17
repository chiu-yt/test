from dataclasses import dataclass

import torch

from pcdet.ops.iou3d_nms import iou3d_nms_utils
from .reg_tta3d_utils import (
    NPGNoiseConfig,
    compute_npg_reliability,
    decoded_query_boxes,
    npg_deletion_mask,
    perturb_npg_query_boxes,
    query_scores,
)


@dataclass(frozen=True)  # noqa: SLOTS_OK - Required by the server's pre-3.10 dataclasses.
class NPGResult:
    pseudo_targets: torch.Tensor
    labels: torch.Tensor
    source_count: int
    deleted_count: int
    confidence_filtered_count: int
    kept_count: int
    inactive_threshold: bool


def _classwise_nms(boxes, scores, labels, threshold, pre_max, post_max):
    selected = []
    for class_label in torch.unique(labels):
        class_indices = torch.nonzero(labels == class_label, as_tuple=False).flatten()
        keep, _ = iou3d_nms_utils.nms_gpu(
            boxes[class_indices, :7], scores[class_indices], threshold,
            pre_maxsize=pre_max,
        )
        selected.append(class_indices[keep[:post_max]])
    if not selected:
        return labels.new_zeros((0,), dtype=torch.long)
    return torch.cat(selected)


def build_npg_pseudo_targets(raw_predictions, dense_head, method_cfg, generator=None):
    query_boxes = decoded_query_boxes(raw_predictions, dense_head=dense_head)
    query_confidence, query_labels = query_scores(
        raw_predictions, dense_head.query_labels, dense_head.num_classes
    )
    noise_config = NPGNoiseConfig(
        distribution=str(method_cfg.get('NPG_NOISE_DISTRIBUTION', 'normal')).lower(),
        dimension_magnitude=float(method_cfg.get('NPG_DIM_MAGNITUDE', 0.1)),
        yaw_magnitude=float(method_cfg.get('NPG_YAW_MAGNITUDE', 0.1)),
        eps=float(method_cfg.get('NPG_EPS', 1e-6)),
    )
    tau = float(method_cfg.get('NPG_TAU', 1.5))
    score_threshold = float(method_cfg.get('FINAL_SCORE_THRESH', 0.2))
    nms_threshold = float(method_cfg.get('NMS_THRESH', 0.1))
    nms_pre_max = int(method_cfg.get('NMS_PRE_MAXSIZE', 4096))
    nms_post_max = int(method_cfg.get('NMS_POST_MAXSIZE', 500))

    batch_targets = []
    kept_labels = []
    source_count = 0
    deleted_count = 0
    confidence_filtered_count = 0
    inactive_threshold = True
    for batch_index in range(query_boxes.shape[0]):
        clean_boxes = query_boxes[batch_index]
        clean_scores = query_confidence[batch_index]
        labels = query_labels[batch_index]
        noisy_boxes = perturb_npg_query_boxes(clean_boxes, noise_config, generator)
        noisy_scores = clean_scores
        aligned_iou = iou3d_nms_utils.boxes_aligned_iou3d_gpu(
            clean_boxes[:, :7], noisy_boxes[:, :7]
        ).reshape(-1)
        reliability = compute_npg_reliability(
            clean_scores, noisy_scores, aligned_iou, noise_config.eps
        )
        deleted = npg_deletion_mask(reliability, tau=tau)
        inactive_threshold = inactive_threshold and not bool(deleted.any().item())
        choose_noisy = noisy_scores > clean_scores
        merged_boxes = torch.where(choose_noisy[:, None], noisy_boxes, clean_boxes)
        merged_scores = torch.where(choose_noisy, noisy_scores, clean_scores)

        finite = (
            torch.isfinite(merged_boxes).all(dim=1)
            & torch.isfinite(merged_scores)
            & (merged_boxes[:, 3:6] > 0).all(dim=1)
        )
        confidence_keep = merged_scores >= score_threshold
        keep = finite & confidence_keep & ~deleted
        source_count += int(clean_boxes.shape[0])
        deleted_count += int(deleted.sum().item())
        confidence_filtered_count += int((finite & ~confidence_keep).sum().item())
        boxes = merged_boxes[keep]
        scores = merged_scores[keep]
        labels = labels[keep]
        selected = _classwise_nms(
            boxes, scores, labels, nms_threshold, nms_pre_max, nms_post_max
        )
        boxes = boxes[selected]
        labels = labels[selected]
        targets = torch.cat((boxes, (labels + 1).to(boxes.dtype)[:, None]), dim=1)
        batch_targets.append(targets)
        kept_labels.append(labels)

    target_width = query_boxes.shape[-1] + 1
    max_targets = max((targets.shape[0] for targets in batch_targets), default=0)
    padded = query_boxes.new_zeros((query_boxes.shape[0], max_targets, target_width))
    for batch_index, targets in enumerate(batch_targets):
        padded[batch_index, :targets.shape[0]] = targets
    joined_labels = (
        torch.cat(kept_labels) if kept_labels
        else query_labels.new_zeros((0,), dtype=torch.long)
    )
    return NPGResult(
        padded, joined_labels, source_count, deleted_count,
        confidence_filtered_count, int(joined_labels.numel()), inactive_threshold,
    )
