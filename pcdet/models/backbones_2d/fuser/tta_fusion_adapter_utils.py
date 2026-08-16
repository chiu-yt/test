import torch
import torch.nn.functional as F
from torch import nn


def pick_group_norm_groups(requested_groups, channels):
    max_groups = max(1, min(int(requested_groups), int(channels)))
    for groups in range(max_groups, 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def zero_last_affine(module):
    if isinstance(module, nn.Sequential):
        for submodule in reversed(list(module)):
            if isinstance(submodule, (nn.Linear, nn.Conv2d)):
                nn.init.zeros_(submodule.weight)
                nn.init.zeros_(submodule.bias)
                break


def world_to_grid(point_cloud_range, centers):
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range.tolist()
    x_norm = (centers[:, 0] - x_min) / max(x_max - x_min, 1e-6) * 2.0 - 1.0
    y_norm = (centers[:, 1] - y_min) / max(y_max - y_min, 1e-6) * 2.0 - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def sample_center_tokens(point_cloud_range, bev_feat, centers):
    height, width = bev_feat.shape[-2:]
    grid = world_to_grid(point_cloud_range, centers).view(1, -1, 1, 2)
    sampled = F.grid_sample(bev_feat, grid, mode='bilinear', align_corners=True)
    return sampled.squeeze(-1).squeeze(0).transpose(0, 1).contiguous()


def proposal_geometry(point_cloud_range, boxes):
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range.tolist()
    center = boxes[:, :3]
    dims = boxes[:, 3:6]
    yaw = boxes[:, 6]
    score = boxes[:, 8]

    x_norm = (center[:, 0] - x_min) / max(x_max - x_min, 1e-6)
    y_norm = (center[:, 1] - y_min) / max(y_max - y_min, 1e-6)
    z_norm = (center[:, 2] - z_min) / max(z_max - z_min, 1e-6)

    return torch.stack([
        x_norm,
        y_norm,
        z_norm,
        dims[:, 0] / max(x_max - x_min, 1e-6),
        dims[:, 1] / max(y_max - y_min, 1e-6),
        dims[:, 2] / max(z_max - z_min, 1e-6),
        torch.sin(yaw),
        score,
    ], dim=-1)


def stamp_residual(residual_map, center_xy, residual_vec, gate, pool_size, height, width):
    radius = max(pool_size, 1)
    cx = int(torch.round(center_xy[0]).item())
    cy = int(torch.round(center_xy[1]).item())
    x0 = max(0, cx - radius)
    x1 = min(width, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(height, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return

    yy = torch.arange(y0, y1, device=residual_map.device, dtype=residual_map.dtype)
    xx = torch.arange(x0, x1, device=residual_map.device, dtype=residual_map.dtype)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    dist2 = (yy - float(cy)) ** 2 + (xx - float(cx)) ** 2
    sigma = float(radius) / 2.0 if radius > 1 else 1.0
    kernel = torch.exp(-dist2 / max(2.0 * sigma * sigma, 1e-6))
    residual_map[:, y0:y1, x0:x1] += gate * residual_vec.view(-1, 1, 1) * kernel.view(1, y1 - y0, x1 - x0)


def proposal_residual_map(adapter, shared_context, proposal_boxes, proposal_mask):
    batch_size, _, height, width = shared_context.shape
    residual_map = shared_context.new_zeros((batch_size, adapter.channels, height, width))
    if proposal_boxes is None or proposal_mask is None:
        return residual_map

    for b_idx in range(batch_size):
        cur_mask = proposal_mask[b_idx]
        if cur_mask.numel() == 0 or not bool(cur_mask.any()):
            continue

        cur_boxes = proposal_boxes[b_idx][cur_mask]
        if cur_boxes.numel() == 0 or cur_boxes.shape[0] < adapter.min_active_proposals:
            continue

        cls_ids = cur_boxes[:, 7].round().long()
        scores = cur_boxes[:, 8]
        valid = (cls_ids > 0) & torch.isfinite(scores) & (scores >= adapter.min_proposal_score)
        if len(adapter.target_class_ids) > 0:
            class_valid = torch.zeros_like(valid)
            for class_id in adapter.target_class_ids:
                class_valid |= cls_ids == class_id
            valid &= class_valid
        if not bool(valid.any()):
            continue

        cur_boxes = cur_boxes[valid]
        cls_ids = cls_ids[valid]
        scores = scores[valid]
        cur_boxes, cls_ids, scores = filter_topk_per_class(adapter, cur_boxes, cls_ids, scores)
        if cur_boxes.numel() == 0:
            continue

        centers = cur_boxes[:, :3]
        center_tokens = sample_center_tokens(adapter.point_cloud_range, shared_context[b_idx:b_idx + 1], centers)
        geom_tokens = adapter.proposal_geom_proj(proposal_geometry(adapter.point_cloud_range, cur_boxes))
        class_tokens = adapter.class_embeddings(cls_ids.clamp(min=0, max=adapter.class_embeddings.num_embeddings - 1))
        proposal_tokens = center_tokens + geom_tokens + class_tokens

        for i in range(cur_boxes.shape[0]):
            stamp_proposal(adapter, residual_map[b_idx], proposal_tokens[i], cur_boxes[i], cls_ids[i], scores[i], height, width)

    return residual_map


def filter_topk_per_class(adapter, cur_boxes, cls_ids, scores):
    if adapter.topk_per_class <= 0:
        return cur_boxes, cls_ids, scores

    filtered_boxes = []
    filtered_cls = []
    filtered_scores = []
    for class_id in adapter.target_class_ids:
        class_mask = cls_ids == class_id
        if not bool(class_mask.any()):
            continue
        class_boxes = cur_boxes[class_mask]
        class_scores = scores[class_mask]
        order = torch.argsort(class_scores, descending=True)
        class_boxes = class_boxes[order[: adapter.topk_per_class]]
        class_scores = class_scores[order[: adapter.topk_per_class]]
        filtered_boxes.append(class_boxes)
        filtered_cls.append(torch.full((class_boxes.shape[0],), class_id, device=class_boxes.device, dtype=torch.long))
        filtered_scores.append(class_scores)
    if len(filtered_boxes) == 0:
        return cur_boxes.new_zeros((0, cur_boxes.shape[1])), cls_ids.new_zeros((0,)), scores.new_zeros((0,))
    return torch.cat(filtered_boxes, dim=0), torch.cat(filtered_cls, dim=0), torch.cat(filtered_scores, dim=0)


def stamp_proposal(adapter, residual_map, proposal_token, cur_box, cls_id_tensor, score_tensor, height, width):
    cls_id = int(cls_id_tensor.item())
    score = float(score_tensor.item())
    class_key = str(cls_id)
    shared_delta = adapter.proposal_shared_head(proposal_token)
    class_delta = adapter.class_shared_heads[class_key](proposal_token) if class_key in adapter.class_shared_heads else 0.0
    if adapter.router_mode == 'shared_only':
        delta_vec = shared_delta
    elif adapter.router_mode == 'class_specific':
        delta_vec = class_delta
    else:
        delta_vec = shared_delta + class_delta

    route = adapter.proposal_router(proposal_token)
    if class_key in adapter.class_gate_heads:
        route = route + adapter.class_gate_heads[class_key](proposal_token)
    gate = torch.sigmoid(route).view(1) * score

    center_xy = world_to_grid(adapter.point_cloud_range, cur_box.view(1, -1)[:, :3])[0]
    center_xy = torch.stack([
        (center_xy[0] + 1.0) * 0.5 * (width - 1),
        (center_xy[1] + 1.0) * 0.5 * (height - 1),
    ])
    stamp_residual(residual_map, center_xy, delta_vec, gate, adapter.pool_size, height, width)
