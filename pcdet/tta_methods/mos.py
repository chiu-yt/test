import copy
import time
import torch
import os
import glob
import numpy as np
import re
from collections import OrderedDict
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import linear_sum_assignment

from pcdet.config import cfg
from pcdet.models import load_data_to_gpu, build_network
from pcdet.utils import common_utils, commu_utils, memory_ensemble_utils, box_utils
from pcdet.utils.tta_utils import TTA_augmentation
from pcdet.datasets.augmentor.data_augmentor import DataAugmentor

# Global Cache (Module Level)
PSEUDO_LABELS = {}
NEW_PSEUDO_LABELS = {}

class MOS(object):
    def __init__(self, model, tta_cfg, logger, dataset=None): # 添加 dataset 参数
        self.model = model
        self.tta_cfg = tta_cfg
        self.logger = logger
        self.dataset = dataset # 保存 dataset
        self.alpha = tta_cfg.get('ALPHA', 0.5)
        self.rank = commu_utils.get_rank()
        
        # 【新增：动态初始化 TTA 增强器】
        if self.dataset is not None:
            if not hasattr(self.dataset, 'tta_data_augmentor'):
                # 尝试从 DATA_CONFIG_TAR 获取 TTA 增强配置
                tta_aug_cfg = cfg.get('DATA_CONFIG_TAR', cfg.DATA_CONFIG).get('TTA_DATA_AUGMENTOR', None)
                if tta_aug_cfg:
                    self.logger.info("理论复现：正在为 NuScenesDataset 挂载 TTA 数据增强器.")
                    self.dataset.tta_data_augmentor = DataAugmentor(
                        self.dataset.root_path,
                        tta_aug_cfg,
                        self.dataset.class_names,
                        logger=self.logger
                    )
                else:
                    # 如果没定义 TTA 专用增强，则回退使用标准增强器
                    self.logger.info("理论复现：未发现 TTA 专用增强配置，回退使用标准数据增强器")
                    self.dataset.tta_data_augmentor = getattr(self.dataset, 'data_augmentor', None)
        
        # Memory Bank Settings
        self.ckpt_ram_cache = OrderedDict()
        self.max_ckpt_cache = 4
        self.total_samples_seen = 0
        self.fallback_dim = 256
        
        # Initialize a temporary model shell for feature extraction (shared weights)
        # We only need the structure, weights will be loaded dynamically
        self.temp_model_shell = None

        # 当前 run 的 ckpt 目录（由 train_st_utils 显式注入）
        self.run_ckpt_dir = None

    def _init_temp_model(self, dataset):
        if self.temp_model_shell is None:
            self.temp_model_shell = build_network(
                model_cfg=cfg.MODEL, 
                num_class=len(cfg.CLASS_NAMES), 
                dataset=dataset
            )
            device = next(self.model.parameters()).device
            self.temp_model_shell.to(device)
            self.temp_model_shell.eval()

    def optimize(self, batch_dict):
        """
        Single iteration optimization
        """
        # 1. Setup Data & Model
        load_data_to_gpu(batch_dict)
        optimizer = batch_dict.get('optimizer') # Passed via hack or need to handle externally
        # Note: In our train_st_utils, we might not have put optimizer in batch_dict.
        # If optimizer is missing, we assume standard backward is handled here but step is outside?
        # Actually, let's look at standard TTA: usually optimizer is global.
        # But to match the function signature `loss, tb_dict, disp_dict = optimize(...)`, 
        # we need to do backward here.
        
        # *Self-Correction*: train_st_utils usually passes model/optim. 
        # But since MOS class wraps the model, we assume `self.model` is the one being trained.
        
        b_size = int(batch_dict.get('batch_size', 1))
        # 与 MOS-main 对齐：优先使用 train_st_utils 传入的 samples_seen 语义
        if 'samples_seen' in batch_dict:
            self.total_samples_seen = int(batch_dict['samples_seen'])
        else:
            self.total_samples_seen += b_size

        # 诊断：记录注入伪标签前的 GT 类别分布（若 batch 中存在真实标签）
        gt_hist = self._collect_hist_from_gt_boxes(batch_dict.get('gt_boxes', None))
        
        # Lazy init
        if self.temp_model_shell is None and self.dataset is not None:
             self._init_temp_model(self.dataset)

        # 2. Inference for Pseudo Labels (Current Model)
        self.model.eval()
        with torch.no_grad():
            pred_dicts, _ = self.model(batch_dict)
        
        # Save raw pseudo labels（A8: 恢复与原版 MOS 一致的 memory-ensemble 语义）
        need_update = self._should_memory_update()
        save_pseudo_label_batch(batch_dict, pred_dicts, need_update=need_update)
        
        # 3. MOS Aggregation Logic (The Core)
        # Check if we have enough checkpoints to perform synergy
        start_ckpt = self.tta_cfg.MOS_SETTING.AGGREGATE_START_CKPT
        # 优先使用 train_st_utils 显式注入的当前 run ckpt 目录。
        # 仅在未注入时才回退到自动查找（兼容旧流程）。
        ckpt_dir = self.run_ckpt_dir if self.run_ckpt_dir is not None else self._find_ckpt_dir()

        super_model = None
        if self.tta_cfg.METHOD == 'mos' and start_ckpt <= self.total_samples_seen and ckpt_dir:
            model_path_list = self._collect_aggregation_ckpts(ckpt_dir)

            if len(model_path_list) >= 3:
                super_model = self._perform_aggregation(model_path_list, batch_dict, pred_dicts)

        # 4. Refine Pseudo Labels with Super Model (if exists)
        if super_model is not None:
            super_model.eval()
            with torch.no_grad():
                # Inference again with aggregated model
                p_dicts, _ = super_model(batch_dict)
                save_pseudo_label_batch(batch_dict, p_dicts, need_update=need_update)
            del super_model

        # 5. Inject Pseudo Labels into Batch
        self._inject_pseudo_labels(batch_dict)

        # 诊断：记录注入后的 pseudo 类别分布
        pseudo_hist = self._collect_hist_from_gt_boxes(batch_dict.get('gt_boxes', None))

        # 6. Training Step (Self-Training)
        self.model.train()
        # Zero grad is usually done in loop, but we do it here to be safe
        if 'optimizer' in batch_dict: # If optimizer passed in batch_dict (hack)
             batch_dict['optimizer'].zero_grad()
        
        # Augmentation
        target_batch = TTA_augmentation(self.dataset, batch_dict)
        
        # Forward & Loss
        # 注意：OpenPCDet 不同 detector 的 train forward 返回格式可能不同
        model_out = self.model(target_batch)
        if isinstance(model_out, dict):
            loss = model_out['loss']
            tb_dict = model_out.get('tb_dict', {})
            disp_dict = model_out.get('disp_dict', {})
        elif isinstance(model_out, (tuple, list)):
            # 常见返回:
            # 1) (loss_tensor, tb_dict, disp_dict)
            # 2) (ret_dict_with_loss, tb_dict, disp_dict)
            first = model_out[0]
            if isinstance(first, dict):
                loss = first['loss']
                tb_dict = first.get('tb_dict', model_out[1] if len(model_out) > 1 else {})
                disp_dict = first.get('disp_dict', model_out[2] if len(model_out) > 2 else {})
            else:
                loss = first
                tb_dict = model_out[1] if len(model_out) > 1 else {}
                disp_dict = model_out[2] if len(model_out) > 2 else {}
        else:
            raise TypeError(f"[MM-MOS] Unexpected model forward return type: {type(model_out)}")

        if not isinstance(disp_dict, dict):
            disp_dict = {}

        # 与标准 model_func 语义对齐：确保 loss 为标量（mean）
        if isinstance(loss, torch.Tensor) and loss.ndim > 0:
            loss = loss.mean()

        # 与参考实现 train_st_utils 语义对齐：target loss 乘 TAR.LOSS_WEIGHT
        tar_loss_weight = float(cfg.SELF_TRAIN.TAR.get('LOSS_WEIGHT', 1.0))
        final_loss = loss * tar_loss_weight

        if isinstance(tb_dict, dict):
            tb_dict['loss_raw'] = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else float(loss)
            tb_dict['loss_tar_weight'] = float(tar_loss_weight)
            tb_dict['loss_tar_weighted'] = float(final_loss.detach().item()) if isinstance(final_loss, torch.Tensor) else float(final_loss)

        disp_dict['loss_raw'] = f"{float(loss.detach().item()):.4f}" if isinstance(loss, torch.Tensor) else f"{float(loss):.4f}"
        disp_dict['loss_w'] = f"{float(final_loss.detach().item()):.4f}" if isinstance(final_loss, torch.Tensor) else f"{float(final_loss):.4f}"
        disp_dict['tar_w'] = f"{tar_loss_weight:.2f}"

        # 仅展示长尾关键类（construction_vehicle, bus, trailer, traffic_cone）
        tail_names = ['construction_vehicle', 'bus', 'trailer', 'traffic_cone']
        gt_tail = '/'.join([str(gt_hist.get(n, 0)) for n in tail_names])
        ps_tail = '/'.join([str(pseudo_hist.get(n, 0)) for n in tail_names])
        disp_dict['tail_gt(cv/bus/tr/cone)'] = gt_tail
        disp_dict['tail_ps(cv/bus/tr/cone)'] = ps_tail

        if isinstance(tb_dict, dict):
            # 便于 tensorboard 检查每步长尾伪标签是否长期为0
            tb_dict['tail_ps/construction_vehicle'] = float(pseudo_hist.get('construction_vehicle', 0))
            tb_dict['tail_ps/bus'] = float(pseudo_hist.get('bus', 0))
            tb_dict['tail_ps/trailer'] = float(pseudo_hist.get('trailer', 0))
            tb_dict['tail_ps/traffic_cone'] = float(pseudo_hist.get('traffic_cone', 0))

        if self.rank == 0 and (self.total_samples_seen % 50 == 0):
            self.logger.info(
                f"[MM-MOS][TailDiag] GT(cv/bus/tr/cone)={gt_tail} | Pseudo(cv/bus/tr/cone)={ps_tail}"
            )
        
        # Backward is done here; optimizer.step() 由 train_st_utils 外层执行
        final_loss.backward()

        # 与标准 model_func 语义对齐：更新 global_step
        if hasattr(self.model, 'update_global_step'):
            self.model.update_global_step()
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'update_global_step'):
            self.model.module.update_global_step()
        
        # Return items for logging
        return final_loss.item(), tb_dict, disp_dict

    def _should_memory_update(self):
        """是否启用并执行 memory ensemble 更新（首个 batch 不更新，后续更新）。"""
        mem_cfg = cfg.SELF_TRAIN.get('MEMORY_ENSEMBLE', None)
        if mem_cfg is None or not mem_cfg.get('ENABLED', False):
            return False
        # 与原版语义对齐：cur_it > 0 时更新；这里用累计样本数近似首步判定
        return self.total_samples_seen > 0

    def _collect_hist_from_gt_boxes(self, gt_boxes):
        """统计 batch 内各类别框数量，兼容 [B, M, C] 或 [M, C]，类别列取最后一列。"""
        hist = {n: 0 for n in cfg.CLASS_NAMES}
        if gt_boxes is None:
            return hist

        try:
            gt = gt_boxes
            if not isinstance(gt, torch.Tensor):
                gt = torch.as_tensor(gt)

            if gt.numel() == 0:
                return hist

            # [B, M, C] -> [B*M, C]
            if gt.dim() == 3:
                gt = gt.reshape(-1, gt.shape[-1])
            elif gt.dim() != 2:
                return hist

            cls = gt[:, -1].float()
            cls = cls[torch.isfinite(cls)]
            if cls.numel() == 0:
                return hist

            num_cls = len(cfg.CLASS_NAMES)
            # 若是0-based标签，转成1-based
            if cls.min() >= 0 and cls.max() <= (num_cls - 1):
                cls = cls + 1

            cls = cls.round().long()
            valid = (cls >= 1) & (cls <= num_cls)
            cls = cls[valid]
            if cls.numel() == 0:
                return hist

            for i, n in enumerate(cfg.CLASS_NAMES, start=1):
                hist[n] = int((cls == i).sum().item())
        except Exception:
            # 诊断逻辑异常不应影响主训练流程
            pass

        return hist

    def _tensor_to_feature_vec(self, feat):
        """将任意中间特征张量池化为 1D 向量，优先保留通道维语义。"""
        if not isinstance(feat, torch.Tensor) or feat.numel() == 0:
            return None

        x = feat.float()
        if x.dim() >= 4:
            # 常见 BEV 特征: [B, C, H, W]
            reduce_dims = tuple(i for i in range(x.dim()) if i != 1)
            vec = x.mean(dim=reduce_dims)
        elif x.dim() == 3:
            # 兼容 [B, N, C] 或 [B, C, H]
            if x.shape[0] <= 8 and 16 <= x.shape[1] <= 2048:
                vec = x.mean(dim=(0, 2))   # 保留 C 维
            else:
                vec = x.mean(dim=(0, 1))   # 保留最后一维
        elif x.dim() == 2:
            vec = x.mean(dim=0)
        else:
            vec = x.view(-1)

        vec = vec.view(-1)
        if vec.numel() == 0 or not torch.isfinite(vec).all():
            return None
        return vec

    def _build_prediction_signature(self, pred_dicts, device):
        """在缺失中间特征时，使用预测分布构建轻量签名向量。"""
        num_cls = len(cfg.CLASS_NAMES)
        cls_hist = torch.zeros(num_cls, dtype=torch.float32, device=device)
        score_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
        num_boxes = 0

        for pred in pred_dicts:
            labels = pred.get('pred_labels', None)
            scores = pred.get('pred_scores', None)
            if labels is None:
                continue

            labels = labels.to(device).long().view(-1)
            if scores is None:
                scores = torch.ones_like(labels, dtype=torch.float32, device=device)
            else:
                scores = scores.to(device).float().view(-1)

            n = min(labels.numel(), scores.numel())
            if n == 0:
                continue
            labels = labels[:n]
            scores = scores[:n]

            valid = (labels >= 1) & (labels <= num_cls)
            if valid.sum().item() == 0:
                continue

            labels = labels[valid]
            scores = scores[valid]
            ones = torch.ones(labels.shape[0], dtype=torch.float32, device=device)
            cls_hist.scatter_add_(0, labels - 1, ones)
            score_sum += scores.sum()
            num_boxes += int(labels.shape[0])

        if num_boxes > 0:
            cls_hist = cls_hist / float(num_boxes)
            mean_score = score_sum / float(num_boxes)
        else:
            mean_score = torch.tensor(0.0, dtype=torch.float32, device=device)

        # 200 与 head proposal 数量一致，作为轻量归一化
        box_density = torch.tensor(float(num_boxes) / 200.0, dtype=torch.float32, device=device)
        sig = torch.cat([cls_hist, mean_score.view(1), box_density.view(1)], dim=0)

        norm = torch.norm(sig, p=2)
        if torch.isfinite(norm) and norm > 0:
            sig = sig / norm
        return sig

    def _extract_aggregation_feature(self, forward_batch, pred_dicts, device):
        """优先提取 BEV 融合后共享特征；若缺失则回退到预测签名。"""
        agg_cfg = self.tta_cfg.get('MOS_SETTING', None)
        default_keys = ['spatial_features_2d', 'spatial_features', 'shared_features', 'bev_features']
        feature_keys = list(agg_cfg.get('FEATURE_KEYS', default_keys)) if agg_cfg is not None else default_keys

        # 1) 先从 forward 后的 batch_dict 中取（更接近融合主干）
        for key in feature_keys:
            vec = self._tensor_to_feature_vec(forward_batch.get(key, None))
            if vec is not None:
                vec = vec.to(device)
                norm = torch.norm(vec, p=2)
                if torch.isfinite(norm) and norm > 0:
                    vec = vec / norm
                return vec

        # 2) 再尝试 pred_dict 附带特征
        for pred in pred_dicts:
            for key in feature_keys:
                vec = self._tensor_to_feature_vec(pred.get(key, None))
                if vec is not None:
                    vec = vec.to(device)
                    norm = torch.norm(vec, p=2)
                    if torch.isfinite(norm) and norm > 0:
                        vec = vec / norm
                    return vec

        # 3) 最后回退预测签名
        return self._build_prediction_signature(pred_dicts, device=device)

    def _perform_aggregation(self, model_path_list, batch_dict, current_preds):
        device = next(self.model.parameters()).device
        agg_cfg = self.tta_cfg.get('MOS_SETTING', None)
        box_topk = int(agg_cfg.get('BOX_TOPK', 50)) if agg_cfg is not None else 50
        box_cost_temp = float(agg_cfg.get('BOX_COST_TEMP', 25.0)) if agg_cfg is not None else 25.0
        sim_eps = float(agg_cfg.get('SIM_EPS', 1e-3)) if agg_cfg is not None else 1e-3
        log_interval = int(agg_cfg.get('LOG_INTERVAL', 50)) if agg_cfg is not None else 50

        feat_vec_list, pred_box_list, pred_score_list, valid_paths = [], [], [], []
        fail_cnt = 0

        for path in model_path_list:
            try:
                if self.temp_model_shell is None:
                    return None
                temp_model_shell = self.temp_model_shell

                # Load state dict (cached)
                if path in self.ckpt_ram_cache:
                    state_dict = self.ckpt_ram_cache[path]
                    self.ckpt_ram_cache.move_to_end(path)
                else:
                    if len(self.ckpt_ram_cache) >= self.max_ckpt_cache:
                        self.ckpt_ram_cache.popitem(last=False)
                    state_dict = torch.load(path, map_location='cpu')['model_state']
                    self.ckpt_ram_cache[path] = state_dict

                # Load into shell
                temp_model_shell.load_state_dict(state_dict, strict=False)

                # Inference current checkpoint model
                with torch.no_grad():
                    p_dicts, _ = temp_model_shell(batch_dict)

                feat_vec = self._extract_aggregation_feature(batch_dict, p_dicts, device=device)
                if feat_vec is None:
                    fail_cnt += 1
                    continue
                feat_vec_list.append(feat_vec.detach().cpu())

                cur_boxes, cur_scores = [], []
                for d in p_dicts:
                    boxes = d.get('pred_boxes', None)
                    if boxes is None:
                        boxes = torch.zeros((0, 7), dtype=torch.float32, device=device)
                    scores = d.get('pred_scores', None)
                    if scores is None:
                        scores = torch.zeros((boxes.shape[0],), dtype=torch.float32, device=boxes.device)
                    cur_boxes.append(boxes)
                    cur_scores.append(scores)

                pred_box_list.append(cur_boxes)
                pred_score_list.append(cur_scores)
                valid_paths.append(path)
            except Exception as e:
                fail_cnt += 1
                if self.rank == 0 and fail_cnt <= 3:
                    self.logger.warning(f"[MM-MOS][Agg] skip ckpt due to error: {os.path.basename(path)} | {str(e)}")

        if len(valid_paths) < 3:
            return None

        # 1) Feature similarity (multimodal BEV-aware)
        max_dim = max([f.numel() for f in feat_vec_list])
        aligned = []
        for f in feat_vec_list:
            if f.numel() < max_dim:
                f = torch.cat([f, torch.zeros(max_dim - f.numel(), dtype=f.dtype)], dim=0)
            f = F.normalize(f.to(device), p=2, dim=0)
            aligned.append(f)
        feat_matrix = torch.stack(aligned, dim=0)
        feat_sim = torch.matmul(feat_matrix, feat_matrix.T).clamp(min=0.0, max=1.0)

        # 2) Box consistency similarity
        n_model = len(valid_paths)
        box_sim = torch.eye(n_model, device=device)
        for i in range(n_model):
            for j in range(i + 1, n_model):
                costs = []
                batch_num = min(len(pred_box_list[i]), len(pred_box_list[j]))
                for b in range(batch_num):
                    b1, _ = topk_by_score(pred_box_list[i][b], pred_score_list[i][b], K=box_topk)
                    b2, _ = topk_by_score(pred_box_list[j][b], pred_score_list[j][b], K=box_topk)
                    costs.append(hungarian_match_diff(b1, b2))

                if len(costs) == 0:
                    pair_sim = 0.0
                else:
                    avg_cost = float(np.mean(costs))
                    pair_sim = float(np.exp(-avg_cost / max(box_cost_temp, 1e-6)))

                box_sim[i, j] = pair_sim
                box_sim[j, i] = pair_sim

        # 3) Build generalized Gram matrix and solve weights
        alpha = float(self.alpha)
        G = alpha * feat_sim + (1.0 - alpha) * box_sim
        G = 0.5 * (G + G.T)
        G = G + torch.eye(n_model, device=device) * sim_eps

        ones = torch.ones(n_model, device=device)
        try:
            c = torch.linalg.solve(G, ones)
        except Exception:
            c = torch.linalg.pinv(G) @ ones

        c = torch.relu(c)
        if (not torch.isfinite(c).all()) or float(c.sum().item()) <= 1e-8:
            c = torch.ones_like(ones) / float(n_model)
        else:
            c = c / c.sum()

        if self.rank == 0 and (self.total_samples_seen % max(log_interval, 1) == 0):
            self.logger.info(
                f"[MM-MOS][Agg] samples_seen={self.total_samples_seen} | n={n_model} | "
                f"feat_mean={float(feat_sim.mean().item()):.3f} | box_mean={float(box_sim.mean().item()):.3f} | "
                f"w={c.detach().cpu().numpy().round(3).tolist()}"
            )

        return aggregate_model_via_state_dict(valid_paths, c, self.dataset, self.ckpt_ram_cache, self.model, self.logger)

    def _inject_pseudo_labels(self, batch_dict):
        """
        将 pseudo labels 写入 batch_dict['gt_boxes']，并强制统一为 NuScenes/BEVFusion 常用的 10 列格式：
        [x, y, z, dx, dy, dz, yaw, vx, vy, cls]

        这样可以避免数据增强（global_rotation）在旋转速度 vx/vy 时出现维度不匹配：
        gt_boxes[:, 7:9] 必须是 2 列。
        """
        fids = batch_dict['frame_id']
        device = batch_dict['points'].device

        # 必须保证 batch 内每个 frame 都有 pseudo label，否则直接跳过本次注入
        if not all(fid in NEW_PSEUDO_LABELS for fid in fids):
            return

        max_box = max([len(NEW_PSEUDO_LABELS[fid]['gt_boxes']) for fid in fids])
        if max_box <= 0:
            return

        # 统一使用 10 列: [x,y,z,dx,dy,dz,yaw,vx,vy,cls]
        ps_batch = torch.zeros(len(fids), max_box, 10, device=device)

        for b_id, fid in enumerate(fids):
            ps_raw = NEW_PSEUDO_LABELS[fid]['gt_boxes']
            if ps_raw is None or len(ps_raw) == 0:
                continue

            ps_boxes_raw = torch.as_tensor(ps_raw, device=device).float()
            c = ps_boxes_raw.shape[1]

            if c == 10:
                ps10 = ps_boxes_raw[:, :10]

            elif c == 9:
                # raw pseudo 标准格式（来自 memory_ensemble_utils）：
                # [x, y, z, dx, dy, dz, yaw, cls, score]
                # 需转换为训练使用的 10 维：[x, y, z, dx, dy, dz, yaw, vx, vy, cls]
                ps10 = torch.zeros(ps_boxes_raw.shape[0], 10, device=device)
                ps10[:, :7] = ps_boxes_raw[:, :7]
                ps10[:, 7:9] = 0
                ps10[:, 9] = ps_boxes_raw[:, 7]

            elif c == 8:
                # 常见 pseudo: [x,y,z,dx,dy,dz,yaw,cls]，缺 vx,vy -> vx=vy=0
                ps10 = torch.zeros(ps_boxes_raw.shape[0], 10, device=device)
                ps10[:, :7] = ps_boxes_raw[:, :7]
                ps10[:, 7:9] = 0
                ps10[:, 9] = ps_boxes_raw[:, 7]

            elif c == 7:
                # 只有 7DoF box: [x,y,z,dx,dy,dz,yaw]
                ps10 = torch.zeros(ps_boxes_raw.shape[0], 10, device=device)
                ps10[:, :7] = ps_boxes_raw[:, :7]
                ps10[:, 7:9] = 0
                ps10[:, 9] = 0
            
            elif c == 11:
                ps10 = torch.zeros(ps_boxes_raw.shape[0], 10, device=device)
                ps10[:, :9] = ps_boxes_raw[:, :9]

                col9  = ps_boxes_raw[:, 9]   # 第10列
                col10 = ps_boxes_raw[:, 10]  # 第11列

    # 如果 col10 看起来像类别（接近整数且范围合理），用 col10 做 cls
                if torch.isfinite(col10).all() and ((col10 - col10.round()).abs().mean() < 1e-3) and (col10.min() >= -1) and (col10.max() <= 20):
                    ps10[:, 9] = col10
                else:
        # 否则认为 col9 是 cls
                    ps10[:, 9] = col9


            else:
                raise ValueError(f"[MM-MOS] Unexpected pseudo gt_boxes dim: {ps_boxes_raw.shape}")

            # 训练分支中 TransFusion 会执行: gt_labels_3d = gt_boxes[..., -1].long() - 1
            # 约定：
            # - 正标签:  1..num_classes
            # - 忽略标签: -1..-num_classes（需保留负号，由 head 内 valid_idx 过滤）
            # - 0: 保留为 0（padding/无效），不可强行改成 1
            num_cls = len(cfg.CLASS_NAMES)
            cls_col = ps10[:, 9].clone()
            if cls_col.numel() > 0:
                finite_mask = torch.isfinite(cls_col)
                # 非法值置 0（后续按无效处理）
                cls_col[~finite_mask] = 0

                # 仅在“无负标签且看起来是 0-based”时，对正标签做 +1。
                # 注意不对 0 做 +1，避免把 padding/无效框污染为 class-1。
                has_negative = bool((cls_col < 0).any())
                pos_mask_raw = cls_col > 0
                if (not has_negative) and pos_mask_raw.any() and cls_col.max() <= (num_cls - 1):
                    cls_col[pos_mask_raw] = cls_col[pos_mask_raw] + 1

                # 正标签裁剪到 [1, num_cls]
                pos_mask = cls_col > 0
                if pos_mask.any():
                    cls_col[pos_mask] = torch.clamp(cls_col[pos_mask].round(), min=1, max=num_cls)

                # 负标签保留符号，并把绝对值裁剪到 [1, num_cls]
                neg_mask = cls_col < 0
                if neg_mask.any():
                    cls_col[neg_mask] = -torch.clamp(cls_col[neg_mask].abs().round(), min=1, max=num_cls)

                # 0 保持为 0
                ps10[:, 9] = cls_col

            cur_num = ps10.shape[0]
            ps_batch[b_id, :cur_num, :] = ps10

        batch_dict['gt_boxes'] = ps_batch

    def _find_ckpt_dir(self):
        # Auto-detect checkpoint directory based on current working setup (fallback only)
        base = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG
        candidates = glob.glob(str(base / '*' / 'ckpt'))
        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            fallback = candidates[0]
            if self.rank == 0:
                self.logger.warning(f'[MM-MOS] run_ckpt_dir 未显式设置，回退自动目录: {fallback}')
            return fallback
        return None

    def _collect_aggregation_ckpts(self, ckpt_dir):
        agg_cfg = self.tta_cfg.get('MOS_SETTING', None)
        if agg_cfg is None:
            return []

        agg_num = int(agg_cfg.get('AGGREGATE_NUM', 3))
        if agg_num <= 0:
            return []

        raw_sources = agg_cfg.get('CKPT_SOURCES', ['iter'])
        if isinstance(raw_sources, str):
            ckpt_sources = [raw_sources]
        else:
            ckpt_sources = list(raw_sources)

        ckpt_sources = [str(x).lower() for x in ckpt_sources]
        reserve_epoch = int(agg_cfg.get('RESERVE_EPOCH_CKPTS', 0))

        iter_paths, epoch_paths = [], []
        if 'iter' in ckpt_sources:
            iter_paths = [
                p for p in glob.glob(os.path.join(ckpt_dir, '*checkpoint_iter_*.pth'))
                if extract_iter(p) >= 0
            ]
            iter_paths.sort(key=extract_iter)

        if 'epoch' in ckpt_sources:
            epoch_paths = [
                p for p in glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
                if extract_iter(p) >= 0
            ]
            epoch_paths.sort(key=extract_iter)

        selected = []
        if epoch_paths and reserve_epoch > 0:
            selected.extend(epoch_paths[-min(len(epoch_paths), reserve_epoch):])

        iter_budget = max(agg_num - len(selected), 0)
        if iter_paths and iter_budget > 0:
            selected.extend(iter_paths[-min(len(iter_paths), iter_budget):])

        if len(selected) < agg_num:
            merged_candidates = []
            if iter_paths:
                merged_candidates.extend(iter_paths)
            if epoch_paths:
                merged_candidates.extend(epoch_paths)

            merged_candidates.sort(key=os.path.getmtime)
            existing = set(selected)
            for path in reversed(merged_candidates):
                if path in existing:
                    continue
                selected.append(path)
                existing.add(path)
                if len(selected) >= agg_num:
                    break

        deduped = []
        seen = set()
        for path in selected:
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)

        deduped.sort(key=lambda p: (extract_iter(p), os.path.getmtime(p)))
        return deduped[-min(len(deduped), agg_num):]

# ================= Helper Functions =================

def extract_iter(path):
    m = re.search(r'(?:epoch_|iter_)(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1

def topk_by_score(boxes, scores, K=50):
    device = boxes.device if isinstance(boxes, torch.Tensor) else torch.device('cpu')
    if boxes is None or len(boxes) == 0:
        return torch.zeros((0, 7), device=device), torch.zeros((0,), device=device)
    
    if not isinstance(boxes, torch.Tensor): boxes = torch.as_tensor(boxes, device=device)
    if not isinstance(scores, torch.Tensor): scores = torch.as_tensor(scores, device=device)
    
    scores = scores.view(-1)
    K = min(K, boxes.shape[0])
    topk_scores, idx = torch.topk(scores, K, largest=True, sorted=False)
    return boxes[idx][:, :7], topk_scores

def hungarian_match_diff(bbox_pred_1, bbox_pred_2):
    n1, n2 = bbox_pred_1.shape[0], bbox_pred_2.shape[0]
    if n1 == 0 and n2 == 0: return 0.0
    if n1 == 0 or n2 == 0: return 100.0 
    
    b1 = bbox_pred_1[:, :7].detach().cpu().numpy()
    b2 = bbox_pred_2[:, :7].detach().cpu().numpy()
    
    diff = np.abs(b1[:, None, :] - b2[None, :, :])
    diff[..., 6] *= 0.1 # heading
    diff[..., 2] *= 0.5 # z
    cost = diff.sum(axis=-1)
    
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].sum()) if r.size > 0 else 100.0

def aggregate_model_via_state_dict(model_path_list, model_weights, dataset, ram_cache, main_model=None, logger=None):
    weights = [float(w.cpu().item()) if isinstance(w, torch.Tensor) else float(w) for w in model_weights]
    agg_state = {}
    p = next(main_model.parameters(), None) if main_model is not None else None
    device = p.device if p is not None else torch.device('cpu')
    
    for idx, model_path in enumerate(model_path_list):
        state = ram_cache.get(model_path)
        if state is None:
            ckpt = torch.load(model_path, map_location='cpu')
            state = ckpt.get('model_state', ckpt)
            ram_cache[model_path] = state
        
        for k, v in state.items():
            if not isinstance(v, torch.Tensor) or not v.dtype.is_floating_point: continue
            v_scaled = v.float() * weights[idx]
            if k in agg_state: agg_state[k] += v_scaled
            else: agg_state[k] = v_scaled.clone()
            
    if not agg_state:
        # Fallback
        last = ram_cache[model_path_list[-1]]
        agg_state = {k: v.clone() for k, v in last.items() if isinstance(v, torch.Tensor)}

    agg_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    agg_model.load_state_dict(agg_state, strict=False)
    
    # Sync BN stats from current main model (crucial for TTA stability)
    if main_model is not None:
        m_model = main_model.module if hasattr(main_model, 'module') else main_model
        main_buf = dict(m_model.named_buffers())
        for name, buf in agg_model.named_buffers():
            clean_name = name.replace('module.', '')
            for b_name, b_val in main_buf.items():
                if b_name.replace('module.', '') == clean_name:
                    buf.data.copy_(b_val.data)
                    break
                    
    return agg_model.to(device).eval()

def save_pseudo_label_batch(input_dict, pred_dicts, need_update=False):
    """
    兼容原版 MOS 语义：
    1) 先按阈值生成当前 batch 伪标签
    2) 若启用 MEMORY_ENSEMBLE 且 need_update=True，则与历史伪标签融合
    """
    global NEW_PSEUDO_LABELS, PSEUDO_LABELS

    # 先生成当前 batch 的原始伪标签
    NEW_PSEUDO_LABELS = memory_ensemble_utils.save_pseudo_label_batch(input_dict, NEW_PSEUDO_LABELS, pred_dicts)

    # D1: 基于深度估计不确定性的伪标签过滤（仅作用于高噪类）
    frame_ids = input_dict.get('frame_id', [])
    for b_idx, raw_fid in enumerate(frame_ids):
        fid = raw_fid.item() if hasattr(raw_fid, 'item') else raw_fid
        if fid in NEW_PSEUDO_LABELS:
            NEW_PSEUDO_LABELS[fid] = _apply_depth_uncertainty_filter(NEW_PSEUDO_LABELS[fid], input_dict, b_idx)
            NEW_PSEUDO_LABELS[fid] = _apply_multimodal_conflict_filter(NEW_PSEUDO_LABELS[fid], input_dict, b_idx)

    # A11: 前置质量过滤（按类别 Top-K 限流，重点约束长尾高噪类）
    frame_ids = input_dict.get('frame_id', [])
    for raw_fid in frame_ids:
        fid = raw_fid.item() if hasattr(raw_fid, 'item') else raw_fid
        if fid in NEW_PSEUDO_LABELS:
            NEW_PSEUDO_LABELS[fid] = _apply_class_topk_filter(NEW_PSEUDO_LABELS[fid])
            NEW_PSEUDO_LABELS[fid] = _apply_adaptive_noisy_class_cap(NEW_PSEUDO_LABELS[fid])

    mem_cfg = cfg.SELF_TRAIN.get('MEMORY_ENSEMBLE', {})
    use_mem = bool(mem_cfg is not None and mem_cfg.get('ENABLED', False) and need_update)
    if not use_mem:
        # 不做融合时，同步覆盖历史缓存
        PSEUDO_LABELS.update(NEW_PSEUDO_LABELS)
        return

    ens_name = mem_cfg.get('NAME', 'consistency_ensemble')
    ensemble_func = getattr(memory_ensemble_utils, ens_name, None)
    if ensemble_func is None:
        # 配置名无效时降级为不融合
        PSEUDO_LABELS.update(NEW_PSEUDO_LABELS)
        return

    frame_ids = input_dict.get('frame_id', [])
    for raw_fid in frame_ids:
        fid = raw_fid.item() if hasattr(raw_fid, 'item') else raw_fid
        if fid not in NEW_PSEUDO_LABELS:
            continue
        cur_infos = NEW_PSEUDO_LABELS[fid]
        if fid in PSEUDO_LABELS:
            merged = memory_ensemble_utils.memory_ensemble(
                PSEUDO_LABELS[fid], cur_infos, mem_cfg, ensemble_func
            )
            NEW_PSEUDO_LABELS[fid] = merged
            PSEUDO_LABELS[fid] = merged
        else:
            PSEUDO_LABELS[fid] = cur_infos


def _apply_class_topk_filter(gt_infos):
    """
    A11: 对单帧伪标签按类别执行 Top-K 过滤。
    - 默认每类保留 DEFAULT_TOPK_PER_CLASS
    - 可通过 CLASS_TOPK 对特定类别（如 pedestrian/traffic_cone）设置更小 K
    """
    ps_filter_cfg = cfg.SELF_TRAIN.get('PS_FILTER', None)
    if ps_filter_cfg is None or not ps_filter_cfg.get('ENABLED', False):
        return gt_infos

    gt_boxes = gt_infos.get('gt_boxes', None)
    if gt_boxes is None or len(gt_boxes) == 0:
        return gt_infos

    default_topk = int(ps_filter_cfg.get('DEFAULT_TOPK_PER_CLASS', 30))
    class_topk_cfg = ps_filter_cfg.get('CLASS_TOPK', {})

    labels = np.abs(gt_boxes[:, 7]).astype(np.int64)
    scores = gt_boxes[:, 8]

    keep_indices = []
    unique_cls = np.unique(labels)
    num_cls = len(cfg.CLASS_NAMES)

    for cls_id in unique_cls:
        if cls_id <= 0 or cls_id > num_cls:
            continue

        cls_name = cfg.CLASS_NAMES[cls_id - 1]
        k = int(class_topk_cfg.get(cls_name, default_topk))
        if k <= 0:
            continue

        cls_inds = np.where(labels == cls_id)[0]
        if cls_inds.size <= k:
            keep_indices.append(cls_inds)
        else:
            cls_scores = scores[cls_inds]
            top_local = np.argsort(-cls_scores)[:k]
            keep_indices.append(cls_inds[top_local])

    if len(keep_indices) == 0:
        empty_infos = {
            'gt_boxes': gt_boxes[:0],
            'cls_scores': None if gt_infos.get('cls_scores', None) is None else gt_infos['cls_scores'][:0],
            'iou_scores': None if gt_infos.get('iou_scores', None) is None else gt_infos['iou_scores'][:0],
            'memory_counter': gt_infos['memory_counter'][:0]
        }
        return empty_infos

    keep_indices = np.concatenate(keep_indices, axis=0)

    filtered_infos = {
        'gt_boxes': gt_boxes[keep_indices],
        'cls_scores': None if gt_infos.get('cls_scores', None) is None else gt_infos['cls_scores'][keep_indices],
        'iou_scores': None if gt_infos.get('iou_scores', None) is None else gt_infos['iou_scores'][keep_indices],
        'memory_counter': gt_infos['memory_counter'][keep_indices]
    }
    return filtered_infos


def _apply_depth_uncertainty_filter(gt_infos, batch_dict, batch_index):
    """
    D1: 仅对高噪类别（如 pedestrian / traffic_cone）做基于深度置信度的过滤。
    第一版使用框中心点在多相机中的最大深度置信度（max prob）作为过滤指标。
    """
    depth_filter_cfg = cfg.SELF_TRAIN.get('DEPTH_FILTER', None)
    if depth_filter_cfg is None or not depth_filter_cfg.get('ENABLED', False):
        return gt_infos

    gt_boxes = gt_infos.get('gt_boxes', None)
    if gt_boxes is None or len(gt_boxes) == 0:
        return gt_infos

    depth_conf_map = batch_dict.get('depth_conf_map', None)
    lidar2image = batch_dict.get('lidar2image', None)
    img_aug_matrix = batch_dict.get('img_aug_matrix', None)
    lidar_aug_matrix = batch_dict.get('lidar_aug_matrix', None)
    if depth_conf_map is None or lidar2image is None or img_aug_matrix is None or lidar_aug_matrix is None:
        return gt_infos

    target_names = set(depth_filter_cfg.get('TARGET_CLASSES', ['pedestrian', 'traffic_cone']))
    target_ids = [i + 1 for i, name in enumerate(cfg.CLASS_NAMES) if name in target_names]
    if len(target_ids) == 0:
        return gt_infos

    conf_thresh = float(depth_filter_cfg.get('CONF_THRESH', 0.25))
    class_conf_thresh_cfg = depth_filter_cfg.get('CLASS_CONF_THRESH', {})
    multi_cam_reduce = depth_filter_cfg.get('MULTI_CAM_REDUCE', 'max')

    labels = np.abs(gt_boxes[:, 7]).astype(np.int64)
    candidate_inds = np.where(np.isin(labels, np.array(target_ids, dtype=np.int64)))[0]
    if candidate_inds.size == 0:
        return gt_infos

    device = depth_conf_map.device if isinstance(depth_conf_map, torch.Tensor) else torch.device('cpu')
    centers = torch.as_tensor(gt_boxes[candidate_inds, :3], device=device, dtype=torch.float32)

    cur_depth_conf = depth_conf_map[batch_index]
    cur_lidar2image = lidar2image[batch_index].to(torch.float32)
    cur_img_aug_matrix = img_aug_matrix[batch_index].to(torch.float32)
    cur_lidar_aug_matrix = lidar_aug_matrix[batch_index].to(torch.float32)

    coords = centers.clone()
    coords -= cur_lidar_aug_matrix[:3, 3]
    coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(coords.transpose(1, 0))
    coords = cur_lidar2image[:, :3, :3].matmul(coords)
    coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)

    raw_depth = coords[:, 2, :].clone()
    coords[:, 2, :] = torch.clamp(coords[:, 2, :], 1e-5, 1e5)
    coords[:, :2, :] /= coords[:, 2:3, :]

    coords = cur_img_aug_matrix[:, :3, :3].matmul(coords)
    coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
    coords = coords[:, :2, :].transpose(1, 2)
    coords = coords[..., [1, 0]]

    if 'camera_imgs' in batch_dict:
        img_h, img_w = batch_dict['camera_imgs'].shape[-2:]
    else:
        img_h, img_w = cfg.MODEL.VTRANSFORM.IMAGE_SIZE
    feat_h, feat_w = cur_depth_conf.shape[-2:]

    on_img = (
        (raw_depth > 1e-5)
        & (coords[..., 0] >= 0) & (coords[..., 0] < img_h)
        & (coords[..., 1] >= 0) & (coords[..., 1] < img_w)
    )

    feat_y = torch.clamp((coords[..., 0] / max(float(img_h), 1.0) * feat_h).long(), min=0, max=feat_h - 1)
    feat_x = torch.clamp((coords[..., 1] / max(float(img_w), 1.0) * feat_w).long(), min=0, max=feat_w - 1)

    keep_mask = np.ones(gt_boxes.shape[0], dtype=bool)
    for local_idx, global_idx in enumerate(candidate_inds.tolist()):
        cls_id = int(labels[global_idx])
        cls_name = cfg.CLASS_NAMES[cls_id - 1] if 1 <= cls_id <= len(cfg.CLASS_NAMES) else None
        cur_conf_thresh = float(class_conf_thresh_cfg.get(cls_name, conf_thresh)) if cls_name is not None else conf_thresh

        cam_mask = on_img[:, local_idx]
        if cam_mask.sum().item() == 0:
            continue

        cam_conf = cur_depth_conf[cam_mask, feat_y[cam_mask, local_idx], feat_x[cam_mask, local_idx]]
        if cam_conf.numel() == 0:
            continue

        if multi_cam_reduce == 'max':
            depth_conf = cam_conf.max()
        else:
            depth_conf = cam_conf.mean()

        if float(depth_conf.item()) < cur_conf_thresh:
            keep_mask[global_idx] = False

    if keep_mask.all():
        return gt_infos

    filtered_infos = {
        'gt_boxes': gt_boxes[keep_mask],
        'cls_scores': None if gt_infos.get('cls_scores', None) is None else gt_infos['cls_scores'][keep_mask],
        'iou_scores': None if gt_infos.get('iou_scores', None) is None else gt_infos['iou_scores'][keep_mask],
        'memory_counter': gt_infos['memory_counter'][keep_mask]
    }
    return filtered_infos


def _apply_multimodal_conflict_filter(gt_infos, batch_dict, batch_index):
    """
    对高噪类别增加一个更保守的多模态冲突过滤：
    只有在“检测分数偏低”且“多点投影深度支持偏弱”同时成立时才移除框。
    这样比单点 center depth filter 更稳，也不会把相机暂时缺失支持的框一刀切删掉。
    """
    filter_cfg = cfg.SELF_TRAIN.get('CONFLICT_FILTER', None)
    if filter_cfg is None or not filter_cfg.get('ENABLED', False):
        return gt_infos

    gt_boxes = gt_infos.get('gt_boxes', None)
    if gt_boxes is None or len(gt_boxes) == 0:
        return gt_infos

    depth_conf_map = batch_dict.get('depth_conf_map', None)
    lidar2image = batch_dict.get('lidar2image', None)
    img_aug_matrix = batch_dict.get('img_aug_matrix', None)
    lidar_aug_matrix = batch_dict.get('lidar_aug_matrix', None)
    if depth_conf_map is None or lidar2image is None or img_aug_matrix is None or lidar_aug_matrix is None:
        return gt_infos

    target_names = set(filter_cfg.get('TARGET_CLASSES', ['pedestrian', 'traffic_cone']))
    target_ids = [i + 1 for i, name in enumerate(cfg.CLASS_NAMES) if name in target_names]
    if len(target_ids) == 0:
        return gt_infos

    labels = np.abs(gt_boxes[:, 7]).astype(np.int64)
    candidate_inds = np.where(np.isin(labels, np.array(target_ids, dtype=np.int64)))[0]
    if candidate_inds.size == 0:
        return gt_infos

    sample_mode = str(filter_cfg.get('SAMPLE_MODE', 'center_corners')).lower()
    min_visible_points = int(filter_cfg.get('MIN_VISIBLE_POINTS', 3))
    min_cam_coverage = float(filter_cfg.get('MIN_CAM_COVERAGE', 0.25))
    multi_cam_reduce = filter_cfg.get('MULTI_CAM_REDUCE', 'max')
    class_score_thresh_cfg = filter_cfg.get('CLASS_SCORE_THRESH', {})
    class_depth_thresh_cfg = filter_cfg.get('CLASS_DEPTH_CONF_THRESH', {})
    default_depth_thresh = float(filter_cfg.get('DEPTH_CONF_THRESH', 0.25))

    depth_filter_cfg = cfg.SELF_TRAIN.get('DEPTH_FILTER', None)
    if depth_filter_cfg is not None:
        default_depth_thresh = float(depth_filter_cfg.get('CONF_THRESH', default_depth_thresh))

    device = depth_conf_map.device if isinstance(depth_conf_map, torch.Tensor) else torch.device('cpu')
    boxes = torch.as_tensor(gt_boxes[candidate_inds, :7], device=device, dtype=torch.float32)
    centers = boxes[:, :3].unsqueeze(1)
    sample_points = centers
    if sample_mode in ['center_corners', 'corners', 'center_bottom_corners']:
        corners = box_utils.boxes_to_corners_3d(boxes)
        if sample_mode == 'center_bottom_corners':
            corners = corners[:, :4, :]
        sample_points = torch.cat([sample_points, corners], dim=1)

    num_boxes, num_samples = sample_points.shape[:2]
    flat_points = sample_points.reshape(-1, 3).transpose(1, 0)

    cur_depth_conf = depth_conf_map[batch_index]
    cur_lidar2image = lidar2image[batch_index].to(torch.float32)
    cur_img_aug_matrix = img_aug_matrix[batch_index].to(torch.float32)
    cur_lidar_aug_matrix = lidar_aug_matrix[batch_index].to(torch.float32)

    coords = flat_points.clone()
    coords -= cur_lidar_aug_matrix[:3, 3].reshape(3, 1)
    coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(coords)
    coords = cur_lidar2image[:, :3, :3].matmul(coords.unsqueeze(0))
    coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)

    raw_depth = coords[:, 2, :].clone()
    coords[:, 2, :] = torch.clamp(coords[:, 2, :], 1e-5, 1e5)
    coords[:, :2, :] /= coords[:, 2:3, :]
    coords = cur_img_aug_matrix[:, :3, :3].matmul(coords)
    coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
    coords = coords[:, :2, :].transpose(1, 2)
    coords = coords[..., [1, 0]].reshape(coords.shape[0], num_boxes, num_samples, 2)
    raw_depth = raw_depth.reshape(raw_depth.shape[0], num_boxes, num_samples)

    if 'camera_imgs' in batch_dict:
        img_h, img_w = batch_dict['camera_imgs'].shape[-2:]
    else:
        img_h, img_w = cfg.MODEL.VTRANSFORM.IMAGE_SIZE
    feat_h, feat_w = cur_depth_conf.shape[-2:]

    on_img = (
        (raw_depth > 1e-5)
        & (coords[..., 0] >= 0) & (coords[..., 0] < img_h)
        & (coords[..., 1] >= 0) & (coords[..., 1] < img_w)
    )

    feat_y = torch.clamp((coords[..., 0] / max(float(img_h), 1.0) * feat_h).long(), min=0, max=feat_h - 1)
    feat_x = torch.clamp((coords[..., 1] / max(float(img_w), 1.0) * feat_w).long(), min=0, max=feat_w - 1)

    keep_mask = np.ones(gt_boxes.shape[0], dtype=bool)
    default_score_thresh_arr = cfg.SELF_TRAIN.get('SCORE_THRESH', None)
    depth_class_cfg = depth_filter_cfg.get('CLASS_CONF_THRESH', {}) if depth_filter_cfg is not None else {}

    for local_idx, global_idx in enumerate(candidate_inds.tolist()):
        cls_id = int(labels[global_idx])
        cls_name = cfg.CLASS_NAMES[cls_id - 1] if 1 <= cls_id <= len(cfg.CLASS_NAMES) else None
        if cls_name is None:
            continue

        score_thresh = class_score_thresh_cfg.get(cls_name, None)
        if score_thresh is None and default_score_thresh_arr is not None and 1 <= cls_id <= len(default_score_thresh_arr):
            score_thresh = float(default_score_thresh_arr[cls_id - 1])
        if score_thresh is None:
            score_thresh = 0.25

        depth_thresh = class_depth_thresh_cfg.get(cls_name, depth_class_cfg.get(cls_name, default_depth_thresh))
        pred_score = float(gt_boxes[global_idx, 8]) if gt_boxes.shape[1] > 8 else 0.0

        cam_mask = on_img[:, local_idx, :]
        cam_visible_counts = cam_mask.sum(dim=1)
        if cam_visible_counts.max().item() <= 0:
            continue

        valid_cam_scores = []
        for cam_idx in range(cam_mask.shape[0]):
            visible_count = int(cam_visible_counts[cam_idx].item())
            if visible_count < min_visible_points:
                continue

            coverage = float(visible_count) / float(max(num_samples, 1))
            if coverage < min_cam_coverage:
                continue

            cur_mask = cam_mask[cam_idx]
            sampled_conf = cur_depth_conf[cam_idx, feat_y[cam_idx, local_idx, cur_mask], feat_x[cam_idx, local_idx, cur_mask]]
            if sampled_conf.numel() == 0:
                continue

            if multi_cam_reduce == 'mean':
                valid_cam_scores.append(sampled_conf.mean())
            else:
                valid_cam_scores.append(sampled_conf.max())

        if len(valid_cam_scores) == 0:
            continue

        valid_cam_scores = torch.stack(valid_cam_scores)
        if multi_cam_reduce == 'mean':
            depth_conf = valid_cam_scores.mean()
        else:
            depth_conf = valid_cam_scores.max()

        if pred_score < float(score_thresh) and float(depth_conf.item()) < float(depth_thresh):
            keep_mask[global_idx] = False

    if keep_mask.all():
        return gt_infos

    filtered_infos = {
        'gt_boxes': gt_boxes[keep_mask],
        'cls_scores': None if gt_infos.get('cls_scores', None) is None else gt_infos['cls_scores'][keep_mask],
        'iou_scores': None if gt_infos.get('iou_scores', None) is None else gt_infos['iou_scores'][keep_mask],
        'memory_counter': gt_infos['memory_counter'][keep_mask]
    }
    return filtered_infos


def _apply_adaptive_noisy_class_cap(gt_infos):
    """
    A12: 仅对高噪类别做自适应上限控制，减少对主类(car/truck等)的误伤。

    配置示例:
    SELF_TRAIN:
      ADAPTIVE_CAP:
        ENABLED: True
        TARGET_CLASSES: ['pedestrian', 'traffic_cone']
        BASE_TOPK: 20
        HIGH_SCORE_TOPK: 35
        HIGH_SCORE_THRESH: 0.45
    """
    cap_cfg = cfg.SELF_TRAIN.get('ADAPTIVE_CAP', None)
    if cap_cfg is None or not cap_cfg.get('ENABLED', False):
        return gt_infos

    gt_boxes = gt_infos.get('gt_boxes', None)
    if gt_boxes is None or len(gt_boxes) == 0:
        return gt_infos

    target_classes = set(cap_cfg.get('TARGET_CLASSES', ['pedestrian', 'traffic_cone']))
    base_topk = int(cap_cfg.get('BASE_TOPK', 20))
    high_topk = int(cap_cfg.get('HIGH_SCORE_TOPK', 35))
    high_thresh = float(cap_cfg.get('HIGH_SCORE_THRESH', 0.45))

    labels = np.abs(gt_boxes[:, 7]).astype(np.int64)
    scores = gt_boxes[:, 8]

    keep_parts = []
    num_cls = len(cfg.CLASS_NAMES)
    for cls_id in np.unique(labels):
        if cls_id <= 0 or cls_id > num_cls:
            continue

        cls_name = cfg.CLASS_NAMES[cls_id - 1]
        cls_inds = np.where(labels == cls_id)[0]

        if cls_name not in target_classes:
            keep_parts.append(cls_inds)
            continue

        if cls_inds.size <= base_topk:
            keep_parts.append(cls_inds)
            continue

        cls_scores = scores[cls_inds]
        high_local = np.where(cls_scores >= high_thresh)[0]
        if high_local.size >= base_topk:
            top_local = high_local[np.argsort(-cls_scores[high_local])[:high_topk]]
        else:
            top_local = np.argsort(-cls_scores)[:base_topk]

        keep_parts.append(cls_inds[top_local])

    if len(keep_parts) == 0:
        empty_infos = {
            'gt_boxes': gt_boxes[:0],
            'cls_scores': None if gt_infos.get('cls_scores', None) is None else gt_infos['cls_scores'][:0],
            'iou_scores': None if gt_infos.get('iou_scores', None) is None else gt_infos['iou_scores'][:0],
            'memory_counter': gt_infos['memory_counter'][:0]
        }
        return empty_infos

    keep_indices = np.concatenate(keep_parts, axis=0)
    filtered_infos = {
        'gt_boxes': gt_boxes[keep_indices],
        'cls_scores': None if gt_infos.get('cls_scores', None) is None else gt_infos['cls_scores'][keep_indices],
        'iou_scores': None if gt_infos.get('iou_scores', None) is None else gt_infos['iou_scores'][keep_indices],
        'memory_counter': gt_infos['memory_counter'][keep_indices]
    }
    return filtered_infos
