import copy

import numpy as np
import torch
from easydict import EasyDict

from pcdet.datasets.augmentor.data_augmentor import DataAugmentor


_VOXEL_KEYS = {'voxels', 'voxel_coords', 'voxel_num_points'}


class CottaEvaluationError(RuntimeError):
    pass


def _is_gt_field(key):
    key_lower = key.lower()
    return key_lower == 'gt' or key_lower.startswith('gt_') or 'ground_truth' in key_lower


class _ScaleViewBuilder:
    def __init__(self, dataset, scale_range):
        scale_cfg = EasyDict({
            'NAME': 'random_world_scaling',
            'WORLD_SCALE_RANGE': list(scale_range),
        })
        self.dataset = dataset
        self.augmentor = DataAugmentor(
            dataset.root_path, [scale_cfg], dataset.class_names, logger=None
        )
        self.voxel_processor = self._find_voxel_processor(dataset)

    @staticmethod
    def _find_voxel_processor(dataset):
        for processor in dataset.data_processor.data_processor_queue:
            function = processor.func if hasattr(processor, 'func') else processor
            if getattr(function, '__name__', '') == 'transform_points_to_voxels':
                return processor
        raise CottaEvaluationError(
            'CoTTA requires the native transform_points_to_voxels processor'
        )

    def build(self, original_batch):
        view = {
            key: copy.deepcopy(value)
            for key, value in original_batch.items()
            if not _is_gt_field(key) and key not in _VOXEL_KEYS and key != 'points'
        }
        points = original_batch['points']
        device = points.device
        batch_size = int(original_batch['batch_size'])
        point_batches = []
        voxel_batches = []
        coordinate_batches = []
        count_batches = []
        lidar_aug_matrices = []

        for batch_index in range(batch_size):
            point_mask = points[:, 0].long() == batch_index
            sample = {
                'points': points[point_mask, 1:].detach().cpu().numpy().copy(),
                'gt_boxes': np.zeros((0, 10), dtype=np.float32),
                'use_lead_xyz': True,
            }
            sample = self.augmentor.forward(data_dict=sample)
            sampled_matrix = self.dataset.set_lidar_aug_matrix({
                'noise_scale': sample['noise_scale']
            })['lidar_aug_matrix']
            sampled_matrix = torch.as_tensor(
                sampled_matrix, dtype=points.dtype, device=device
            )
            lidar_aug_matrices.append(
                sampled_matrix @ original_batch['lidar_aug_matrix'][batch_index]
            )

            sample = self.voxel_processor(data_dict=sample)
            sample_points = torch.from_numpy(sample['points']).to(device=device, dtype=points.dtype)
            batch_column = sample_points.new_full((sample_points.shape[0], 1), batch_index)
            point_batches.append(torch.cat((batch_column, sample_points), dim=1))
            voxel_batches.append(torch.from_numpy(sample['voxels']).to(device=device, dtype=points.dtype))
            coordinates = torch.from_numpy(sample['voxel_coords']).to(device=device, dtype=torch.int32)
            coordinate_batches.append(torch.cat((
                coordinates.new_full((coordinates.shape[0], 1), batch_index), coordinates
            ), dim=1))
            count_batches.append(
                torch.from_numpy(sample['voxel_num_points']).to(device=device, dtype=torch.int32)
            )

        view['points'] = torch.cat(point_batches, dim=0)
        view['voxels'] = torch.cat(voxel_batches, dim=0)
        view['voxel_coords'] = torch.cat(coordinate_batches, dim=0)
        view['voxel_num_points'] = torch.cat(count_batches, dim=0)
        view['lidar_aug_matrix'] = torch.stack(lidar_aug_matrices, dim=0)
        return view
