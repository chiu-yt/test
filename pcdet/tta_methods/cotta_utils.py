import copy

import torch
from easydict import EasyDict

from pcdet.models.model_utils import model_nms_utils


def initialize_cotta_models(model):
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    deepcopy_memo = (
        {id(model.dataset): model.dataset} if hasattr(model, 'dataset') else {}
    )

    source_anchor = copy.deepcopy(model, deepcopy_memo.copy())
    teacher = copy.deepcopy(model, deepcopy_memo.copy())
    student = copy.deepcopy(model, deepcopy_memo.copy())

    source_anchor.eval()
    source_anchor.requires_grad_(False)
    teacher.eval()
    teacher.requires_grad_(False)

    student.train()
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name in trainable_names)

    return source_anchor, teacher, student


def update_ema_teacher(teacher, student, alpha):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('EMA alpha must be in [0, 1]')

    student_parameters = dict(student.named_parameters())
    student_buffers = dict(student.named_buffers())
    with torch.no_grad():
        for name, teacher_parameter in teacher.named_parameters():
            student_parameter = student_parameters[name]
            teacher_parameter.mul_(alpha).add_(student_parameter.detach(), alpha=1.0 - alpha)

        for name, teacher_buffer in teacher.named_buffers():
            student_buffer = student_buffers[name].detach()
            if teacher_buffer.is_floating_point() or teacher_buffer.is_complex():
                teacher_buffer.mul_(alpha).add_(student_buffer, alpha=1.0 - alpha)
            else:
                teacher_buffer.copy_(student_buffer)

    teacher.requires_grad_(False)
    teacher.eval()


def stochastic_restore(
        student, source_anchor, trainable_names, probability, generator=None):
    if not 0.0 <= probability <= 1.0:
        raise ValueError('Restoration probability must be in [0, 1]')

    eligible_names = set(trainable_names)
    source_parameters = dict(source_anchor.named_parameters())
    restored_count = 0
    eligible_count = 0

    with torch.no_grad():
        for name, student_parameter in student.named_parameters():
            if name not in eligible_names:
                continue

            source_parameter = source_parameters[name]
            source_value = source_parameter.detach().to(
                device=student_parameter.device,
                dtype=student_parameter.dtype,
                non_blocking=True,
            )
            restore_mask = torch.rand(
                student_parameter.shape,
                device=student_parameter.device,
                generator=generator,
            ) < probability
            student_parameter.copy_(torch.where(
                restore_mask,
                source_value,
                student_parameter,
            ))
            restored_count += int(restore_mask.sum().item())
            eligible_count += student_parameter.numel()

    return restored_count, eligible_count


def transform_boxes_between_views(boxes, source_matrix, target_matrix):
    transformed = boxes.clone()
    if transformed.shape[0] == 0:
        return transformed

    source_matrix = torch.as_tensor(
        source_matrix, dtype=transformed.dtype, device=transformed.device
    )
    target_matrix = torch.as_tensor(
        target_matrix, dtype=transformed.dtype, device=transformed.device
    )
    source_linear = source_matrix[:3, :3]
    target_linear = target_matrix[:3, :3]

    canonical_centers = torch.linalg.solve(
        source_linear,
        (transformed[:, :3] - source_matrix[:3, 3]).T,
    ).T
    transformed[:, :3] = canonical_centers @ target_linear.T + target_matrix[:3, 3]

    source_scale = torch.linalg.vector_norm(source_linear[:, 0])
    target_scale = torch.linalg.vector_norm(target_linear[:, 0])
    transformed[:, 3:6] *= target_scale / source_scale

    view_linear = target_linear @ torch.linalg.inv(source_linear)
    heading_vectors = torch.stack((
        torch.cos(boxes[:, 6]),
        torch.sin(boxes[:, 6]),
        torch.zeros_like(boxes[:, 6]),
    ), dim=1)
    transformed_headings = heading_vectors @ view_linear.T
    transformed[:, 6] = torch.atan2(
        transformed_headings[:, 1], transformed_headings[:, 0]
    )

    if transformed.shape[1] >= 9:
        velocity_vectors = torch.cat((
            boxes[:, 7:9],
            torch.zeros((boxes.shape[0], 1), dtype=boxes.dtype, device=boxes.device),
        ), dim=1)
        transformed[:, 7:9] = (velocity_vectors @ view_linear.T)[:, :2]

    return transformed


def filter_predictions_to_targets(
        pred_dicts, score_thresholds, nms_thresholds,
        nms_pre_maxsize, nms_post_maxsize):
    if len(score_thresholds) != len(nms_thresholds):
        raise ValueError('Score and NMS thresholds must define the same classes')

    num_classes = len(score_thresholds)
    filtered_batches = []
    class_counts = []

    for pred_dict in pred_dicts:
        boxes = pred_dict['pred_boxes']
        scores = pred_dict['pred_scores']
        labels = pred_dict['pred_labels']
        valid_mask = (
            torch.isfinite(boxes).all(dim=1)
            & torch.isfinite(scores)
            & (boxes[:, 3:6] > 0).all(dim=1)
            & (labels >= 1)
            & (labels <= num_classes)
        )

        batch_boxes = []
        batch_scores = []
        batch_labels = []
        batch_counts = torch.zeros(num_classes, dtype=torch.long, device=boxes.device)

        for class_index in range(num_classes):
            class_id = class_index + 1
            class_mask = (
                valid_mask
                & (labels == class_id)
                & (scores >= score_thresholds[class_index])
            )
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            if class_boxes.shape[0] == 0:
                continue

            nms_config = EasyDict({
                'NMS_TYPE': 'nms_gpu',
                'NMS_THRESH': nms_thresholds[class_index],
                'NMS_PRE_MAXSIZE': nms_pre_maxsize,
                'NMS_POST_MAXSIZE': nms_post_maxsize,
            })
            nms_boxes = class_boxes if class_boxes.is_cuda else class_boxes.cuda()
            nms_scores = class_scores if class_scores.is_cuda else class_scores.cuda()
            selected, _ = model_nms_utils.class_agnostic_nms(
                nms_scores, nms_boxes[:, :7], nms_config
            )
            selected = torch.as_tensor(
                selected, dtype=torch.long, device=class_boxes.device
            )

            selected_boxes = class_boxes[selected]
            selected_scores = class_scores[selected]
            batch_boxes.append(selected_boxes)
            batch_scores.append(selected_scores)
            batch_labels.append(torch.full(
                (selected.shape[0], 1), class_id,
                dtype=boxes.dtype, device=boxes.device,
            ))
            batch_counts[class_index] = selected.shape[0]

        if batch_boxes:
            target_rows = torch.cat((
                torch.cat(batch_boxes, dim=0)[:, :9],
                torch.cat(batch_labels, dim=0),
            ), dim=1)
            confidence_weights = torch.cat(batch_scores, dim=0)
        else:
            target_rows = boxes.new_zeros((0, 10))
            confidence_weights = scores.new_zeros((0,))

        filtered_batches.append((target_rows, confidence_weights))
        class_counts.append(batch_counts)

    max_count = max((rows.shape[0] for rows, _ in filtered_batches), default=0)
    first_boxes = pred_dicts[0]['pred_boxes']
    first_scores = pred_dicts[0]['pred_scores']
    targets = first_boxes.new_zeros((len(pred_dicts), max_count, 10))
    confidence_weights = first_scores.new_zeros((len(pred_dicts), max_count))
    for batch_index, (rows, weights) in enumerate(filtered_batches):
        targets[batch_index, :rows.shape[0]] = rows
        confidence_weights[batch_index, :weights.shape[0]] = weights

    return targets, confidence_weights, torch.stack(class_counts, dim=0)
