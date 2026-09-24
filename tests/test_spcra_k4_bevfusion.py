import unittest
from types import SimpleNamespace

import numpy as np
import torch

from test_spcra_k4_runtime import load_contract_module


class FakeVoxelProcessor:
    func = SimpleNamespace(__name__='transform_points_to_voxels')

    def __call__(self, data_dict):
        points = data_dict['points']
        return {
            'voxels': points[:, None, :],
            'voxel_coords': np.zeros((len(points), 3), dtype=np.int32),
            'voxel_num_points': np.ones(len(points), dtype=np.int32),
        }


class TestK4BEVFusionContext(unittest.TestCase):
    def setUp(self):
        self.module = load_contract_module('spcra_k4_bevfusion')
        self.context = load_contract_module('spcra_k4_context')
        processor = FakeVoxelProcessor()
        self.dataset = SimpleNamespace(
            point_cloud_range=np.array([0., 0., -1., 4., 4., 1.]),
            data_processor=SimpleNamespace(data_processor_queue=[processor]),
        )

    def views(self, **config):
        defaults = {'CAMERA_RESCUE_ENABLED': False, 'DENSITY_GRID_SIZE': 4}
        defaults.update(config)
        return self.module.K4BEVFusionViews(self.dataset, defaults)

    def test_build_view_rebuilds_adapter_inputs_after_points_and_composes_calibration(self):
        # Given a reference frame, immutable proposals, and a non-identity calibration.
        reference_matrix = torch.tensor([[1., 0., 0., 10.], [0., 1., 0., 20.],
                                         [0., 0., 1., 30.], [0., 0., 0., 1.]])
        pristine = {
            'batch_size': 1,
            'points': torch.tensor([[0., .25, .25, 0.], [0., 3.25, 3.25, 0.]]),
            'lidar_aug_matrix': reference_matrix[None],
        }
        proposals = self.context.ReferenceProposalContext.from_inputs(
            torch.tensor([[[1., 1., 0., 1., 1., 1., 0., 4., .75]]]),
            torch.tensor([[True]]), batch_size=1,
        )
        delta = np.eye(4, dtype=np.float32)
        delta[0, 3] = 1.

        # When the view retains one point and applies its geometry delta.
        actual = self.views().build_view(
            pristine, np.array([0]), delta, proposal_context=proposals,
        )

        # Then density sees retained/transformed points, proposals share the delta,
        # and detector calibration composes view_delta @ reference.
        expected_calibration = torch.from_numpy(delta) @ reference_matrix
        torch.testing.assert_close(actual['lidar_aug_matrix'][0], expected_calibration)
        torch.testing.assert_close(actual['points'][0, 1:3], torch.tensor([1.25, .25]))
        self.assertGreater(actual['tta_density_map'][0, 0, 0, 1].item(), 0.)
        torch.testing.assert_close(actual['tta_proposal_boxes'][0, 0, 0], torch.tensor(2.))
        torch.testing.assert_close(actual['tta_proposal_mask'], torch.tensor([[True]]))

    def test_build_view_installs_explicit_empty_proposal_context_when_absent(self):
        # Given a valid view without any proposal source.
        pristine = {
            'batch_size': 1,
            'points': torch.tensor([[0., .25, .25, 0.]]),
            'lidar_aug_matrix': torch.eye(4)[None],
        }

        # When the view is built through the production builder.
        actual = self.views().build_view(pristine, np.array([0]), np.eye(4))

        # Then the adapter receives explicit, correctly shaped empty tensors.
        self.assertEqual(actual['tta_proposal_boxes'].shape, (1, 0, 9))
        self.assertEqual(actual['tta_proposal_mask'].shape, (1, 0))

    def test_camera_rescue_reads_each_views_own_image_features(self):
        # Given fixed predictions and opposite image-BEV energy layouts.
        views = self.views(CAMERA_RESCUE_ENABLED=True, CAMERA_SUPPORT_SCALE=1.)
        predictions = [{
            'pred_boxes': torch.tensor([[0., 1., 0., 1., 1., 1., 0.],
                                        [4., 1., 0., 1., 1., 1., 0.]]),
            'pred_labels': torch.tensor([1, 1]),
            'pred_scores': torch.tensor([.2, .2]),
        }]
        left = torch.tensor([[[[1., 3.], [1., 3.]]]])
        right = torch.tensor([[[[3., 1.], [3., 1.]]]])

        # When two separate view forwards expose their own image-BEV features.
        original_forward = self.module.forward_without_annotations
        setattr(self.module, 'forward_without_annotations', lambda model, batch: (predictions, {}))
        try:
            first = views.forward_view(None, {'spatial_features_img': left})
            second = views.forward_view(None, {'spatial_features_img': right})
        finally:
            setattr(self.module, 'forward_without_annotations', original_forward)

        # Then support follows each view's own feature tensor and no feature is retained.
        self.assertLess(first[0]['camera_support'][0], first[0]['camera_support'][1])
        self.assertGreater(second[0]['camera_support'][0], second[0]['camera_support'][1])
        self.assertNotIn('spatial_features_img', first[0])
        self.assertNotIn('lidar_aug_matrix', first[0])


if __name__ == '__main__':
    unittest.main()
