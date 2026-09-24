"""Server gate: actual CUDA TTA augmentation, not a substitute augmentation law."""

import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires Torch/CUDA OpenPCDet environment')
class TestK4StudentAugmentation(unittest.TestCase):
    def test_velocity_geometry_and_classes_when_real_student_is_augmented(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest('TTA_augmentation uses CUDA tensors')
        from easydict import EasyDict
        from pcdet.config import cfg
        from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
        from pcdet.tta_methods.spcra_k4_mos import augment_k4_student

        augmentor = DataAugmentor('.', EasyDict({
            'DISABLE_AUG_LIST': [], 'AUG_CONFIG_LIST': [
                EasyDict(NAME='random_world_flip', ALONG_AXIS_LIST=['x', 'y']),
                EasyDict(NAME='random_world_rotation', WORLD_ROT_ANGLE=[.3, .3]),
                EasyDict(NAME='random_world_scaling', WORLD_SCALE_RANGE=[1.1, 1.2]),
                EasyDict(NAME='random_world_translation', NOISE_TRANSLATE_STD=[.1, .1, .1]),
            ],
        }), ['car'])
        dataset = SimpleNamespace(tta_data_augmentor=augmentor,
                                  point_cloud_range=np.array([-20., -20., -10., 20., 20., 10.]))
        boxes = torch.tensor([[[1., 2., 0., 2., 3., 2., .1, 3., -4., 8.],
                               [100., 100., 0., 2., 2., 2., 0., 1., 2., 10.],
                               [0.] * 10]], device='cuda')
        base = torch.eye(4, device='cuda')[None]
        base[:, 0, 3] = 5.
        batch = {'batch_size': 1, 'frame_id': ['frame'],
                 'points': torch.tensor([[0., 1., 2., 0., .5]], device='cuda'),
                 'gt_boxes': boxes.clone(), 'lidar_aug_matrix': base.clone(),
                 'tta_pseudo_weights': torch.tensor([[.25, .8, 0.]], device='cuda'),
                 'tta_pseudo_reg_weights': torch.tensor([[.25, .8, 0.]], device='cuda')}
        previous_model = cfg.get('MODEL')
        rng = np.random.get_state()
        try:
            cfg.MODEL = EasyDict(TTA_FUSION_ADAPTER=EasyDict(SG_DFA=EasyDict(ENABLED=True)))
            np.random.seed(1024)
            target = augment_k4_student(dataset, batch)
        finally:
            np.random.set_state(rng)
            if previous_model is None:
                cfg.pop('MODEL')
            else:
                cfg.MODEL = previous_model
        delta = target['lidar_aug_matrix'][0] @ torch.linalg.inv(base[0])
        torch.testing.assert_close(target['gt_boxes'][0, 0, 7:9], delta[:2, :2] @ boxes[0, 0, 7:9])
        torch.testing.assert_close(target['gt_boxes'][0, 0, :3],
                                   delta[:3, :3] @ boxes[0, 0, :3] + delta[:3, 3])
        self.assertEqual(target['gt_boxes'][0, 0, 9].item(), 8.)
        torch.testing.assert_close(target['gt_boxes'][0, 1:], torch.zeros_like(boxes[0, 1:]))
        torch.testing.assert_close(target['tta_pseudo_weights'], torch.tensor([[.25, 0., 0.]], device='cuda'))
        torch.testing.assert_close(target['tta_pseudo_reg_weights'], target['tta_pseudo_weights'])
        self.assertGreater(target['tta_density_map'].sum().item(), 0.)
