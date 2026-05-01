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
