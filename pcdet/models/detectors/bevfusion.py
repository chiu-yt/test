import torch

from .detector3d_template import Detector3DTemplate
from .. import backbones_image, view_transforms
from ..backbones_image import img_neck
from ..backbones_2d import fuser

class BevFusion(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'image_backbone','neck','vtransform','fuser',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()
       
    def build_neck(self,model_info_dict):
        if self.model_cfg.get('NECK', None) is None:
            return None, model_info_dict
        neck_module = img_neck.__all__[self.model_cfg.NECK.NAME](
            model_cfg=self.model_cfg.NECK
        )
        model_info_dict['module_list'].append(neck_module)

        return neck_module, model_info_dict
    
    def build_vtransform(self,model_info_dict):
        if self.model_cfg.get('VTRANSFORM', None) is None:
            return None, model_info_dict
        
        vtransform_module = view_transforms.__all__[self.model_cfg.VTRANSFORM.NAME](
            model_cfg=self.model_cfg.VTRANSFORM
        )
        model_info_dict['module_list'].append(vtransform_module)

        return vtransform_module, model_info_dict

    def build_image_backbone(self, model_info_dict):
        if self.model_cfg.get('IMAGE_BACKBONE', None) is None:
            return None, model_info_dict
        image_backbone_module = backbones_image.__all__[self.model_cfg.IMAGE_BACKBONE.NAME](
            model_cfg=self.model_cfg.IMAGE_BACKBONE
        )
        image_backbone_module.init_weights()
        model_info_dict['module_list'].append(image_backbone_module)

        return image_backbone_module, model_info_dict
    
    def build_fuser(self, model_info_dict):
        if self.model_cfg.get('FUSER', None) is None:
            return None, model_info_dict
    
        fuser_module = fuser.__all__[self.model_cfg.FUSER.NAME](
            model_cfg=self.model_cfg.FUSER
        )
        model_info_dict['module_list'].append(fuser_module)
        model_info_dict['num_bev_features'] = self.model_cfg.FUSER.OUT_CHANNEL
        return fuser_module, model_info_dict

    def forward(self, batch_dict):
        export_conflict_analysis = self._forward_only_conflict_analysis_enabled() and not self.training
        export_depth_entropy_analysis = self._depth_entropy_analysis_enabled() and not self.training
        lidar_bev = None
        image_bev = None

        for i,cur_module in enumerate(self.module_list):
            if export_conflict_analysis and cur_module is self.fuser:
                lidar_bev = batch_dict['spatial_features']
                image_bev = batch_dict['spatial_features_img']
            batch_dict = cur_module(batch_dict)
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if export_depth_entropy_analysis:
                self._attach_depth_entropy_analysis(batch_dict=batch_dict, pred_dicts=pred_dicts)
            if export_conflict_analysis:
                self._attach_forward_only_conflict_analysis(
                    batch_dict=batch_dict,
                    pred_dicts=pred_dicts,
                    lidar_bev=lidar_bev,
                    image_bev=image_bev
                )
            return pred_dicts, recall_dicts

    def _forward_only_conflict_analysis_enabled(self):
        analysis_cfg = self.model_cfg.get('FORWARD_ONLY_CONFLICT_ANALYSIS', None)
        return analysis_cfg is not None and analysis_cfg.get('ENABLED', False)

    def _depth_entropy_analysis_enabled(self):
        analysis_cfg = self.model_cfg.get('DEPTH_ENTROPY_ANALYSIS', None)
        return analysis_cfg is not None and analysis_cfg.get('ENABLED', False)

    def _depth_entropy_analysis_cfg(self):
        return self.model_cfg.get('DEPTH_ENTROPY_ANALYSIS', {})

    @staticmethod
    def _summarize_depth_entropy(depth_entropy):
        depth_entropy = depth_entropy.float()
        finite_mask = torch.isfinite(depth_entropy)
        if finite_mask.sum().item() == 0:
            return {
                'valid_pixels': 0,
                'mean': None,
                'p50': None,
                'p75': None,
                'p90': None,
                'per_camera_mean': []
            }

        valid_entropy = depth_entropy[finite_mask]
        quantiles = torch.quantile(valid_entropy, valid_entropy.new_tensor([0.5, 0.75, 0.9]))
        per_camera_mean = []
        for cam_idx in range(depth_entropy.shape[0]):
            cam_entropy = depth_entropy[cam_idx]
            cam_mask = torch.isfinite(cam_entropy)
            if cam_mask.sum().item() == 0:
                per_camera_mean.append(None)
            else:
                per_camera_mean.append(float(cam_entropy[cam_mask].mean().item()))

        return {
            'valid_pixels': int(finite_mask.sum().item()),
            'mean': float(valid_entropy.mean().item()),
            'p50': float(quantiles[0].item()),
            'p75': float(quantiles[1].item()),
            'p90': float(quantiles[2].item()),
            'per_camera_mean': per_camera_mean
        }

    def _attach_depth_entropy_analysis(self, batch_dict, pred_dicts):
        depth_entropy_map = batch_dict.get('depth_entropy_map', None)
        if depth_entropy_map is None:
            return

        for index, pred_dict in enumerate(pred_dicts):
            pred_dict['depth_entropy_analysis'] = self._summarize_depth_entropy(depth_entropy_map[index])
            pred_dict['depth_entropy_object_analysis'] = self._build_depth_entropy_object_analysis(
                batch_dict=batch_dict,
                pred_dict=pred_dict,
                batch_index=index
            )

    def _build_depth_entropy_object_analysis(self, batch_dict, pred_dict, batch_index):
        pred_boxes = pred_dict.get('pred_boxes', None)
        if pred_boxes is None or pred_boxes.shape[0] == 0:
            return {
                'depth_entropy': [],
                'num_visible_cams': [],
                'visible_cam_ids': []
            }

        depth_entropy_map = batch_dict.get('depth_entropy_map', None)
        lidar2image = batch_dict.get('lidar2image', None)
        img_aug_matrix = batch_dict.get('img_aug_matrix', None)
        lidar_aug_matrix = batch_dict.get('lidar_aug_matrix', None)
        if depth_entropy_map is None or lidar2image is None or img_aug_matrix is None or lidar_aug_matrix is None:
            return {
                'depth_entropy': [None for _ in range(pred_boxes.shape[0])],
                'num_visible_cams': [0 for _ in range(pred_boxes.shape[0])],
                'visible_cam_ids': [[] for _ in range(pred_boxes.shape[0])]
            }

        analysis_cfg = self._depth_entropy_analysis_cfg()
        multi_cam_reduce = str(analysis_cfg.get('OBJECT_MULTI_CAM_REDUCE', 'min')).lower()
        if multi_cam_reduce not in ['min', 'mean', 'max']:
            raise ValueError('Unsupported OBJECT_MULTI_CAM_REDUCE: %s' % multi_cam_reduce)

        centers = pred_boxes[:, :3].to(depth_entropy_map.device).float()
        cur_depth_entropy = depth_entropy_map[batch_index]
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
            img_h, img_w = self.model_cfg.VTRANSFORM.IMAGE_SIZE
        feat_h, feat_w = cur_depth_entropy.shape[-2:]

        on_img = (
            (raw_depth > 1e-5)
            & (coords[..., 0] >= 0) & (coords[..., 0] < img_h)
            & (coords[..., 1] >= 0) & (coords[..., 1] < img_w)
        )

        feat_y = torch.clamp((coords[..., 0] / max(float(img_h), 1.0) * feat_h).long(), min=0, max=feat_h - 1)
        feat_x = torch.clamp((coords[..., 1] / max(float(img_w), 1.0) * feat_w).long(), min=0, max=feat_w - 1)

        object_entropy = []
        num_visible_cams = []
        visible_cam_ids = []
        for box_idx in range(pred_boxes.shape[0]):
            cam_mask = on_img[:, box_idx]
            cam_ids = torch.where(cam_mask)[0]
            visible_cam_ids.append([int(cam_idx.item()) for cam_idx in cam_ids])
            num_visible_cams.append(int(cam_ids.numel()))
            if cam_ids.numel() == 0:
                object_entropy.append(None)
                continue

            cam_entropy = cur_depth_entropy[cam_mask, feat_y[cam_mask, box_idx], feat_x[cam_mask, box_idx]]
            cam_entropy = cam_entropy[torch.isfinite(cam_entropy)]
            if cam_entropy.numel() == 0:
                object_entropy.append(None)
                continue

            if multi_cam_reduce == 'mean':
                reduced_entropy = cam_entropy.mean()
            elif multi_cam_reduce == 'max':
                reduced_entropy = cam_entropy.max()
            else:
                reduced_entropy = cam_entropy.min()
            object_entropy.append(float(reduced_entropy.item()))

        return {
            'depth_entropy': object_entropy,
            'num_visible_cams': num_visible_cams,
            'visible_cam_ids': visible_cam_ids
        }

    @staticmethod
    def _copy_pred_box_dict(box_dict):
        return {
            'pred_boxes': box_dict['pred_boxes'],
            'pred_scores': box_dict['pred_scores'],
            'pred_labels': box_dict['pred_labels']
        }

    def _run_branch_post_fusion(self, lidar_bev, image_bev):
        branch_batch_dict = {
            'spatial_features': lidar_bev,
            'spatial_features_img': image_bev
        }
        branch_batch_dict = self.fuser(branch_batch_dict)
        branch_batch_dict = self.backbone_2d(branch_batch_dict)
        branch_preds = self.dense_head.predict(branch_batch_dict['spatial_features_2d'])
        return self.dense_head.get_bboxes(branch_preds)

    def _attach_forward_only_conflict_analysis(self, batch_dict, pred_dicts, lidar_bev, image_bev):
        if lidar_bev is None or image_bev is None:
            raise ValueError('Forward-only conflict analysis requires both lidar and image BEV features before fusion.')

        branch_pred_dicts = {
            'B_L': self._run_branch_post_fusion(lidar_bev, torch.zeros_like(image_bev)),
            'B_C': self._run_branch_post_fusion(torch.zeros_like(lidar_bev), image_bev)
        }

        for index, pred_dict in enumerate(pred_dicts):
            pred_dict['B_Fused'] = self._copy_pred_box_dict(pred_dict)
            pred_dict['B_L'] = self._copy_pred_box_dict(branch_pred_dicts['B_L'][index])
            pred_dict['B_C'] = self._copy_pred_box_dict(branch_pred_dicts['B_C'][index])

            if 'gt_boxes' in batch_dict:
                pred_dict['gt_boxes'] = batch_dict['gt_boxes'][index]

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
