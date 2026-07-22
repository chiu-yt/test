from collections import OrderedDict

import torch

from pcdet.config import cfg
from pcdet.models import build_network


def snapshot_floating_state(model):
    model_ref = model.module if hasattr(model, 'module') else model
    return {
        k: v.detach().cpu().clone()
        for k, v in model_ref.state_dict().items()
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
    }


def compute_ridge_leverage_weights(feat_vec_list, topk=5, damping=1e-3, device=None):
    if len(feat_vec_list) == 0:
        return [], torch.zeros((0,), dtype=torch.float32)

    if device is None:
        device = feat_vec_list[0].device

    max_dim = max([f.numel() for f in feat_vec_list])
    aligned = []
    for feat in feat_vec_list:
        feat = feat.view(-1).float().to(device)
        if feat.numel() < max_dim:
            feat = torch.cat([feat, torch.zeros(max_dim - feat.numel(), dtype=feat.dtype, device=device)], dim=0)
        aligned.append(torch.nn.functional.normalize(feat, p=2, dim=0))

    fingerprints = torch.stack(aligned, dim=0)
    if fingerprints.shape[0] == 1:
        return [0], torch.ones((1,), dtype=torch.float32, device=device)

    fingerprints = fingerprints - fingerprints.mean(dim=0, keepdim=True)
    gram = torch.matmul(fingerprints, fingerprints.T) / float(fingerprints.shape[0])
    gram = gram + damping * torch.eye(gram.shape[0], dtype=gram.dtype, device=device)
    scores = (fingerprints * torch.linalg.solve(gram, fingerprints)).sum(dim=1)
    scores = torch.clamp(scores, min=0.0)

    select_num = min(max(int(topk), 1), len(feat_vec_list))
    _, indices = torch.topk(scores, k=select_num, largest=True, sorted=False)
    selected_scores = scores[indices]
    if (not torch.isfinite(selected_scores).all()) or float(selected_scores.sum().item()) <= 1e-8:
        weights = torch.ones(select_num, dtype=torch.float32, device=device) / float(select_num)
    else:
        weights = selected_scores.float() / selected_scores.float().sum()
    return [int(i.item()) for i in indices], weights


def _load_model_state(model_path, ram_cache):
    state = ram_cache.get(model_path)
    if state is not None:
        return state

    ckpt = torch.load(model_path, map_location='cpu')
    state = ckpt.get('model_state', ckpt)
    ram_cache[model_path] = state
    return state


def aggregate_model_via_codemerge(model_path_list, model_weights, source_state, dataset, ram_cache, main_model=None, logger=None, merge_scale=0.85):
    weights = [float(w.detach().cpu().item()) if isinstance(w, torch.Tensor) else float(w) for w in model_weights]
    model_ref = None
    if main_model is not None:
        model_ref = main_model.module if hasattr(main_model, 'module') else main_model
    param = next(model_ref.parameters(), None) if model_ref is not None else None
    device = param.device if param is not None else torch.device('cpu')

    selected_states = [_load_model_state(path, ram_cache) for path in model_path_list]
    merged_state = OrderedDict()

    for key, source_tensor in source_state.items():
        deltas, delta_weights = [], []
        for idx, state in enumerate(selected_states):
            value = state.get(key, None)
            if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
                continue
            if value.shape != source_tensor.shape:
                continue
            deltas.append(value.float() - source_tensor.float())
            delta_weights.append(weights[idx])

        if len(deltas) == 0:
            merged_state[key] = source_tensor.clone()
            continue

        stacked = torch.stack(deltas, dim=0)
        signs = torch.sign(stacked.sum(dim=0))
        sign_mask = (stacked * signs.unsqueeze(0)) >= 0
        weighted = torch.zeros_like(source_tensor.float())
        total_weight = torch.zeros_like(source_tensor.float())
        for delta_idx, weight in enumerate(delta_weights):
            mask = sign_mask[delta_idx].float()
            weighted += stacked[delta_idx] * mask * weight
            total_weight += mask * weight
        merged_delta = torch.where(total_weight > 0, weighted / torch.clamp(total_weight, min=1e-8), torch.zeros_like(weighted))
        merged_state[key] = source_tensor.float() + float(merge_scale) * merged_delta

    agg_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    missing, unexpected = agg_model.load_state_dict(merged_state, strict=False)
    if logger is not None:
        logger.info(
            f'[CodeMerge] loaded merged task-vector state | missing={len(missing)} | unexpected={len(unexpected)}'
        )

    if model_ref is not None:
        main_buf = dict(model_ref.named_buffers())
        for name, buf in agg_model.named_buffers():
            clean_name = name.replace('module.', '')
            for b_name, b_val in main_buf.items():
                if b_name.replace('module.', '') == clean_name:
                    buf.data.copy_(b_val.data)
                    break

    return agg_model.to(device).eval()
