import copy
import time
import torch
import numpy as np
import torch.distributed as dist
from pcdet.config import cfg
from pcdet.utils import common_utils, box_utils
from torch.nn.utils import clip_grad_norm_

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    pass


def _ensure_gt_boxes_10_np(gt_boxes: np.ndarray):
    """
    将 gt_boxes 统一成 10 列: [x,y,z,dx,dy,dz,yaw,vx,vy,cls]
    兼容输入为 7/8/9/10 列的情况，避免增强阶段 global_rotation 对 vx/vy 旋转时报错。
    """
    if gt_boxes is None:
        return gt_boxes
    if not isinstance(gt_boxes, np.ndarray):
        gt_boxes = np.asarray(gt_boxes)
    if gt_boxes.ndim != 2:
        return gt_boxes

    c = gt_boxes.shape[1]
    if c == 10:
        return gt_boxes
    elif c == 9:
        # [x..vy] -> append cls=0
        pad = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        return np.concatenate([gt_boxes, pad], axis=1)
    elif c == 8:
        # 常见 pseudo: [x,y,z,dx,dy,dz,yaw,cls] -> 插入 vx,vy=0
        gt10 = np.zeros((gt_boxes.shape[0], 10), dtype=gt_boxes.dtype)
        gt10[:, :7] = gt_boxes[:, :7]
        gt10[:, 7:9] = 0
        gt10[:, 9] = gt_boxes[:, 7]
        return gt10
    elif c == 7:
        gt10 = np.zeros((gt_boxes.shape[0], 10), dtype=gt_boxes.dtype)
        gt10[:, :7] = gt_boxes[:, :7]
        gt10[:, 7:9] = 0
        gt10[:, 9] = 0
        return gt10
    else:
        raise ValueError(f"Unexpected gt_boxes shape: {gt_boxes.shape}")


def TTA_augmentation(dataset, target_batch, strength='mid'):
    if not hasattr(TTA_augmentation, "_printed"):
        print("[DEBUG] target_batch keys:", list(target_batch.keys()))
        TTA_augmentation._printed = True
    """
    针对 BEVFusion 和多 Batch 优化的 TTA 增强函数
    支持 imgaug：必须把 camera_imgs（以及可能的 image_shape 等meta）传入 single_dict
    """
    b_size = target_batch['batch_size']

    # 1) 准备容器
    new_points_list = []
    new_gt_boxes_list = []
    new_images_list = [] # ✅新增：保存增强后的单样本图像
    new_img_process_infos_list = []
    new_ori_shape_list = []
    new_img_aug_matrix_list = []

    point_cloud_range = getattr(dataset, 'point_cloud_range', None)
    if point_cloud_range is None and cfg.get('DATA_CONFIG', None) is not None:
        point_cloud_range = cfg.DATA_CONFIG.get('POINT_CLOUD_RANGE', None)

    # 2) 获取对应的增强器
    augmentor = dataset.tta_data_augmentor
    if strength == 'strong':
        augmentor = dataset.strong_tta_data_augmentor
    elif strength == 'weak':
        augmentor = dataset.weak_tta_data_augmentor
    # ✅TTA：如果 camera_imgs 是 Tensor，说明当前图像已经是 BEVFusion 预处理后的 Tensor
    # 标准增强器里的 imgaug 是 PIL 路线（img.rotate），会直接报错
    # 因此在 TTA 阶段过滤掉 imgaug，只保留点云增强
    def _get_aug_name(aug):
        if hasattr(aug, '__name__'):
            name = aug.__name__
        elif hasattr(aug, 'func') and hasattr(aug.func, '__name__'):
            # functools.partial(DataAugmentor.imgaug, ...)
            name = aug.func.__name__
        else:
            name = aug.__class__.__name__
        return name.lower()

    need_disable_imgaug = ('camera_imgs' in target_batch and torch.is_tensor(target_batch['camera_imgs'])) \
        or ('camera_imgs' not in target_batch or 'img_process_infos' not in target_batch)

    # 统一过滤增强器：imgaug（PIL 路线，与当前 tensor 图像不兼容）
    if hasattr(augmentor, 'data_augmentor_queue'):
        new_queue = []
        for aug in augmentor.data_augmentor_queue:
            aug_name = _get_aug_name(aug)

            if need_disable_imgaug and 'imgaug' in aug_name:
                continue

            new_queue.append(aug)
        augmentor.data_augmentor_queue = new_queue


    # 3) 逐样本处理
    for b_idx in range(b_size):
        # 点云 [N, 3+C]
        points_mask = (target_batch['points'][:, 0] == b_idx)
        cur_points = target_batch['points'][points_mask, 1:].cpu().numpy()

        # 框 [M, ?] -> 统一到 10 列
        cur_boxes = target_batch['gt_boxes'][b_idx].cpu().numpy()
        cur_boxes = _ensure_gt_boxes_10_np(cur_boxes)

        # 构造增强器需要的单样本输入
        single_dict = {
            'batch_size': 1,
            'points': cur_points,
            'gt_boxes': cur_boxes,
            'frame_id': target_batch['frame_id'][b_idx],
            'use_lead_xyz': True
        }

        # 传入 imgaug 所需字段（若存在），避免 data_augmentor.imgaug 中 KeyError
        if 'camera_imgs' in target_batch:
            single_dict['camera_imgs'] = target_batch['camera_imgs'][b_idx]
        if 'img_process_infos' in target_batch:
            single_dict['img_process_infos'] = target_batch['img_process_infos'][b_idx]

        

# 常见还会用到 ori_shape / img_aug_matrix / image_paths
        if 'ori_shape' in target_batch:
            single_dict['ori_shape'] = target_batch['ori_shape'][b_idx]
        if 'img_aug_matrix' in target_batch:
            single_dict['img_aug_matrix'] = target_batch['img_aug_matrix'][b_idx]
        if 'image_paths' in target_batch:
            single_dict['image_paths'] = target_batch['image_paths'][b_idx]

# 几何映射（有些 imgaug 或后续 image branch 会依赖）
        for k in ['lidar2camera', 'lidar2image', 'camera2ego', 'camera_intrinsics', 'camera2lidar']:
            if k in target_batch:
                single_dict[k] = target_batch[k][b_idx]


        # 4) 执行增强
        single_dict = augmentor.forward(data_dict=single_dict)

        # 4.1) 读取增强后的 boxes
        aug_boxes = _ensure_gt_boxes_10_np(single_dict['gt_boxes'])

        if aug_boxes is not None and aug_boxes.shape[0] > 0:
            try:
                finite_mask = np.isfinite(aug_boxes).all(axis=1)
                if aug_boxes.shape[1] >= 6:
                    finite_mask = finite_mask & (aug_boxes[:, 3] > 0) & (aug_boxes[:, 4] > 0) & (aug_boxes[:, 5] > 0)
                if isinstance(finite_mask, np.ndarray) and finite_mask.shape[0] == aug_boxes.shape[0]:
                    aug_boxes = aug_boxes.copy()
                    aug_boxes[~finite_mask] = 0
            except Exception:
                pass

        # 显式做一次 range mask（zero-out 方式）
        if point_cloud_range is not None and aug_boxes is not None and aug_boxes.shape[0] > 0:
            try:
                keep_mask = box_utils.mask_boxes_outside_range_numpy(
                    aug_boxes,
                    np.asarray(point_cloud_range),
                    min_num_corners=1,
                    use_center_to_filter=True
                )
                if isinstance(keep_mask, np.ndarray) and keep_mask.shape[0] == aug_boxes.shape[0]:
                    aug_boxes = aug_boxes.copy()
                    aug_boxes[~keep_mask] = 0
            except Exception:
                # 过滤异常不应阻断主流程
                pass

# 同步保存可能被 imgaug 更新的 meta（按样本存，最后组成 list/stack）
        if 'img_process_infos' in single_dict:
            new_img_process_infos_list.append(single_dict['img_process_infos'])
        if 'ori_shape' in single_dict:
            new_ori_shape_list.append(single_dict['ori_shape'])
        if 'img_aug_matrix' in single_dict:
            iam = single_dict['img_aug_matrix']
            if isinstance(iam, np.ndarray):
                iam = torch.from_numpy(iam)
            new_img_aug_matrix_list.append(iam.cuda())

        # 5) points 回写
        aug_points = torch.from_numpy(single_dict['points']).float().cuda()
        batch_idx_col = aug_points.new_ones(aug_points.shape[0], 1) * b_idx
        new_points_list.append(torch.cat([batch_idx_col, aug_points], dim=1))

        # 6) gt_boxes 回写（再兜底一次维度）
        new_gt_boxes_list.append(torch.from_numpy(aug_boxes).float().cuda())

        # ✅7) 图像回写：如果增强器返回了 camera_imgs，就把增强后的存起来
        if 'camera_imgs' in single_dict:
            imgs = single_dict['camera_imgs']
            if isinstance(imgs, np.ndarray):
                imgs = torch.from_numpy(imgs)
            # imgs 可能是 uint8 或 float；保持原 dtype 更安全
            if torch.is_tensor(imgs):
                new_images_list.append(imgs.cuda())
            else:
                # 极少数情况下不是 tensor/np，就直接用原图兜底
                new_images_list.append(target_batch['camera_imgs'][b_idx])

    # 8) 合并全 Batch 数据
    target_batch['points'] = torch.cat(new_points_list, dim=0)
    target_batch['gt_boxes'] = torch.stack(new_gt_boxes_list, dim=0)

    # ✅合并增强后的图像（如果存在）
    if len(new_images_list) > 0:
        target_batch['camera_imgs'] = torch.stack(new_images_list, dim=0)

    # 9) 重新生成 Voxel（如果 batch 里存在 voxel keys）
    if 'voxels' in target_batch:
        target_batch.pop('voxels')
        target_batch.pop('voxel_coords')
        target_batch.pop('voxel_num_points')

        # 仅复用 voxel 化处理器，避免走到 image_normalize/image_calibrate 造成图像字段依赖
        voxel_processor = None
        if hasattr(dataset, 'data_processor') and hasattr(dataset.data_processor, 'data_processor_queue'):
            for proc in dataset.data_processor.data_processor_queue:
                fn = proc.func if hasattr(proc, 'func') else proc
                fn_name = fn.__name__ if hasattr(fn, '__name__') else ''
                if fn_name == 'transform_points_to_voxels':
                    voxel_processor = proc
                    break

        # 注意：target_batch['points'] 是带 batch_idx 的 [bs_idx, x, y, z, ...]
        # data_processor.forward 期望单样本 points: [x, y, z, ...]，不能直接喂整批拼接点云
        voxels_list, voxel_coords_list, voxel_num_points_list = [], [], []
        for b_idx in range(b_size):
            pts_mask = (target_batch['points'][:, 0] == b_idx)
            cur_points = target_batch['points'][pts_mask, 1:].cpu().numpy().astype(np.float32)

            temp_dict = {
                'points': cur_points,
                'frame_id': target_batch['frame_id'][b_idx],
                'use_lead_xyz': True
            }
            if voxel_processor is not None:
                temp_dict = voxel_processor(data_dict=temp_dict)
            else:
                # 兜底：若未找到 voxel 处理器，退回原流程
                if 'img_process_infos' in target_batch:
                    temp_dict['img_process_infos'] = target_batch['img_process_infos'][b_idx]
                if 'camera_imgs' in target_batch:
                    temp_dict['camera_imgs'] = target_batch['camera_imgs'][b_idx]
                temp_dict = dataset.data_processor.forward(data_dict=temp_dict)

            cur_voxels = torch.from_numpy(temp_dict['voxels']).float().cuda()
            cur_coords = torch.from_numpy(temp_dict['voxel_coords']).int().cuda()
            cur_num_pts = torch.from_numpy(temp_dict['voxel_num_points']).int().cuda()

            # 为 voxel coords 补回 batch 维度
            batch_col = torch.full((cur_coords.shape[0], 1), b_idx, dtype=cur_coords.dtype, device=cur_coords.device)
            cur_coords = torch.cat([batch_col, cur_coords], dim=1)

            voxels_list.append(cur_voxels)
            voxel_coords_list.append(cur_coords)
            voxel_num_points_list.append(cur_num_pts)

        target_batch['voxels'] = torch.cat(voxels_list, dim=0)
        target_batch['voxel_coords'] = torch.cat(voxel_coords_list, dim=0)
        target_batch['voxel_num_points'] = torch.cat(voxel_num_points_list, dim=0)

    return target_batch


# --- 以下 rotate_points_along_z 必须确保设备一致 ---
def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C) 或 (B, N, 2) --> 适配速度旋转
        angle: (B), 旋转弧度
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    bz = angle.shape[0]
    
    # 获取输入数据的最后一个维度 (2 或 3+)
    input_dim = points.shape[-1]

    if input_dim == 2:
        # 【新增逻辑】专门处理 NuScenes 速度向量 (vx, vy) 的旋转
        # 创建 2x2 旋转矩阵
        rot_matrix = torch.stack((
            cosa,  sina,
            -sina, cosa
        ), dim=1).view(bz, 2, 2).float().to(points.device)
        
        points_rot = torch.matmul(points, rot_matrix)
    
    else:
        # 【原始逻辑】处理 3D 坐标 (x, y, z)
        zeros = angle.new_zeros(bz)
        ones = angle.new_ones(bz)
        rot_matrix = torch.stack((
            cosa,  sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(bz, 3, 3).float().to(points.device)
        
        # 确保 points 是 3D 的 [B, N, C]
        if points.dim() == 2:
            points = points.unsqueeze(0)
            
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        if input_dim > 3:
            # 拼接剩余的维度 (如强度、时间戳等)
            points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)

    return points_rot.numpy() if is_numpy else points_rot

# ... 其他函数 (update_ema_variables, save_pseudo_label_batch 等) 保持不变 ...
