import os

import torch

from pcdet.tta_methods.codemerge_utils import (
    aggregate_model_via_codemerge,
    compute_ridge_leverage_weights,
    snapshot_floating_state,
)
from pcdet.utils.inference_utils import forward_without_annotations
from pcdet.tta_methods.mos import MOS


class CodeMergeTTA(MOS):
    def __init__(self, model, tta_cfg, logger, dataset=None):
        super().__init__(model, tta_cfg, logger, dataset=dataset)
        self.codemerge_source_state = snapshot_floating_state(model)
        code_cfg = self.tta_cfg.get('CODEMERGE_SETTING', None)
        select_topk = int(code_cfg.get('SELECT_TOPK', 5)) if code_cfg is not None else 5
        self.max_ckpt_cache = max(self.max_ckpt_cache, select_topk + 1)
        if self.rank == 0:
            self.logger.info('[CodeMerge] 已缓存 source checkpoint 浮点参数，用于 task-vector merging baseline')

    def _perform_aggregation(self, model_path_list, batch_dict, current_preds):
        device = next(self.model.parameters()).device
        feat_vec_list, valid_paths = [], []
        fail_cnt = 0

        for path in model_path_list:
            try:
                if self.temp_model_shell is None:
                    return None
                temp_model_shell = self.temp_model_shell

                if path in self.ckpt_ram_cache:
                    state_dict = self.ckpt_ram_cache[path]
                    self.ckpt_ram_cache.move_to_end(path)
                else:
                    if len(self.ckpt_ram_cache) >= self.max_ckpt_cache:
                        self.ckpt_ram_cache.popitem(last=False)
                    state_dict = torch.load(path, map_location='cpu')['model_state']
                    self.ckpt_ram_cache[path] = state_dict

                temp_model_shell.load_state_dict(state_dict, strict=False)
                with torch.no_grad():
                    pred_dicts, _ = forward_without_annotations(temp_model_shell, batch_dict)

                feat_vec = self._extract_aggregation_feature(batch_dict, pred_dicts, device=device)
                if feat_vec is None:
                    fail_cnt += 1
                    continue
                feat_vec_list.append(feat_vec.detach().cpu())
                valid_paths.append(path)
            except (KeyError, OSError, RuntimeError) as err:
                fail_cnt += 1
                if self.rank == 0 and fail_cnt <= 3:
                    self.logger.warning(f'[CodeMerge] skip ckpt due to error: {os.path.basename(path)} | {str(err)}')

        if len(valid_paths) < 3:
            return None

        code_cfg = self.tta_cfg.get('CODEMERGE_SETTING', None)
        topk = int(code_cfg.get('SELECT_TOPK', 5)) if code_cfg is not None else 5
        damping = float(code_cfg.get('RIDGE_DAMPING', 1e-3)) if code_cfg is not None else 1e-3
        merge_scale = float(code_cfg.get('MERGE_SCALE', 0.85)) if code_cfg is not None else 0.85
        log_interval = int(code_cfg.get('LOG_INTERVAL', 50)) if code_cfg is not None else 50

        indices, weights = compute_ridge_leverage_weights(feat_vec_list, topk=topk, damping=damping, device=device)
        if len(indices) == 0:
            return None

        selected_paths = [valid_paths[i] for i in indices]
        if self.rank == 0 and (self.total_samples_seen % max(log_interval, 1) == 0):
            selected_names = [os.path.basename(path) for path in selected_paths]
            self.logger.info(
                f"[CodeMerge] samples_seen={self.total_samples_seen} | n={len(valid_paths)} | "
                f"selected={selected_names} | w={weights.detach().cpu().numpy().round(3).tolist()} | scale={merge_scale:.2f}"
            )

        return aggregate_model_via_codemerge(
            selected_paths,
            weights,
            self.codemerge_source_state,
            self.dataset,
            self.ckpt_ram_cache,
            main_model=self.model,
            logger=self.logger,
            merge_scale=merge_scale,
        )
