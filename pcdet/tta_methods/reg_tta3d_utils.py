from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


REGRESSION_BRANCHES = ('center', 'height', 'dim', 'rot', 'vel')
REGRESSION_PREFIXES = tuple(
    'dense_head.prediction_head.%s.' % branch for branch in REGRESSION_BRANCHES
)


class RegTTA3DMathError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NPGNoiseConfig:
    distribution: str
    dimension_magnitude: float
    yaw_magnitude: float
    eps: float = 1e-6


def compute_npg_reliability(score, score_noisy, iou3d, eps):
    denominator = torch.maximum(iou3d, torch.full_like(iou3d, eps))
    return 1.0 - torch.abs(score - score_noisy) / denominator


def npg_deletion_mask(reliability, tau=1.5):
    return reliability > tau


def _sample_noise(reference, distribution, magnitude, generator):
    shape = reference.shape
    options = {
        'device': reference.device,
        'dtype': reference.dtype,
        'generator': generator,
    }
    if distribution == 'normal':
        return torch.randn(shape, **options) * magnitude
    if distribution == 'uniform':
        return (torch.rand(shape, **options) * 2.0 - 1.0) * magnitude
    raise RegTTA3DMathError('NPG noise distribution must be normal or uniform')


def perturb_npg_query_boxes(boxes, config, generator=None):
    noisy_boxes = boxes.clone()
    dimension_noise = _sample_noise(
        boxes[:, 3:6], config.distribution, config.dimension_magnitude, generator
    )
    yaw_noise = _sample_noise(
        boxes[:, 6], config.distribution, config.yaw_magnitude, generator
    )
    noisy_boxes[:, 3:6] = torch.clamp(
        boxes[:, 3:6] * (1.0 + dimension_noise), min=config.eps
    )
    noisy_yaw = boxes[:, 6] + yaw_noise
    noisy_boxes[:, 6] = torch.atan2(torch.sin(noisy_yaw), torch.cos(noisy_yaw))
    return noisy_boxes


def compute_crr_dimension_loss(
        boxes, scores, labels, top_ratio=0.2, margin=0.1, weight=1.0):
    if boxes.numel() == 0:
        return boxes.sum() * 0.0

    class_losses = []
    for class_label in torch.unique(labels):
        class_mask = labels == class_label
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        sample_count = class_boxes.shape[0]
        top_count = max(1, int(math.ceil(sample_count * top_ratio)))
        order = torch.argsort(class_scores, descending=True)
        mean_size = class_boxes[order[:top_count], 3:6].mean(dim=0).detach()
        low_sizes = class_boxes[order[top_count:], 3:6]
        if low_sizes.numel() == 0:
            continue
        deviation = torch.abs(low_sizes - mean_size)
        class_losses.append(torch.where(
            deviation < margin, torch.zeros_like(deviation), deviation.square()
        ).sum(dim=1))

    if not class_losses:
        return boxes.sum() * 0.0
    return torch.cat(class_losses).mean() * weight


def update_cbu_regression_teacher(teacher, student, alpha):
    if not 0.99 <= alpha <= 0.999:
        raise RegTTA3DMathError('CBU alpha must be in [0.99, 0.999]')
    student_parameters = dict(student.named_parameters())
    total_delta = 0.0
    with torch.no_grad():
        for name, teacher_parameter in teacher.named_parameters():
            if not name.startswith(REGRESSION_PREFIXES):
                continue
            before = teacher_parameter.detach().clone()
            teacher_parameter.mul_(alpha).add_(
                student_parameters[name], alpha=1.0 - alpha
            )
            total_delta += float((teacher_parameter - before).abs().mean().item())
    return total_delta


def cbu_alpha_from_labels(labels, num_classes, alpha_min=0.99, alpha_max=0.999):
    if labels.numel() == 0:
        return alpha_max
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    proportions = counts / counts.sum()
    variance = proportions.var(unbiased=False)
    maximum_variance = (num_classes - 1.0) / (num_classes * num_classes)
    normalized = torch.clamp(variance / maximum_variance, min=0.0, max=1.0)
    return float((alpha_min + normalized * (alpha_max - alpha_min)).item())


def query_scores(raw_predictions, query_labels, num_classes):
    one_hot = F.one_hot(query_labels, num_classes=num_classes).permute(0, 2, 1)
    scores = raw_predictions['heatmap'].sigmoid()
    scores = scores * raw_predictions['query_heatmap_score'] * one_hot
    return scores.max(dim=1).values, scores.max(dim=1).indices


def decoded_query_boxes(raw_predictions, dense_head=None):
    center = raw_predictions['center'].clone()
    if dense_head is not None:
        center[:, 0] = (
            center[:, 0] * dense_head.feature_map_stride * dense_head.voxel_size[0]
            + dense_head.point_cloud_range[0]
        )
        center[:, 1] = (
            center[:, 1] * dense_head.feature_map_stride * dense_head.voxel_size[1]
            + dense_head.point_cloud_range[1]
        )
    dimensions = raw_predictions['dim'].exp()
    yaw = torch.atan2(
        raw_predictions['rot'][:, 0:1], raw_predictions['rot'][:, 1:2]
    )
    components = [
        center, raw_predictions['height'], dimensions, yaw
    ]
    if 'vel' in raw_predictions:
        components.append(raw_predictions['vel'])
    return torch.cat(components, dim=1).permute(0, 2, 1)
