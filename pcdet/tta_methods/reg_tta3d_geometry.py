import copy

from easydict import EasyDict
import torch

from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
from pcdet.utils import common_utils


_VOXEL_FIELDS = ('voxels', 'voxel_coords', 'voxel_num_points')


class RegTTA3DGeometryError(RuntimeError):
    pass


def build_reg_tta3d_student_view(teacher_batch, augment_student_geometry, voxelize):
    student_batch = copy.deepcopy(teacher_batch)
    for voxel_field in ('voxels', 'voxel_coords', 'voxel_num_points'):
        student_batch.pop(voxel_field, None)

    student_batch = augment_student_geometry(student_batch)
    fresh_voxel_tensors = voxelize(student_batch['points'])
    student_batch.update(fresh_voxel_tensors)
    return student_batch


def _is_annotation_field(key):
    lowered = key.lower()
    return (
        lowered == 'gt' or lowered.startswith('gt_')
        or 'ground_truth' in lowered or lowered in {
            'annotations', 'annos', 'sample_annotation_tokens'
        }
    )


class RegTTA3DViewBuilder:
    def __init__(self, dataset, method_cfg):
        augmentations = [
            EasyDict({
                'NAME': 'random_world_flip',
                'ALONG_AXIS_LIST': list(method_cfg.get('FLIP_AXIS', ['x', 'y'])),
            }),
            EasyDict({
                'NAME': 'random_world_rotation',
                'WORLD_ROT_ANGLE': list(method_cfg.get('ROT_RANGE', [-0.78539816, 0.78539816])),
            }),
            EasyDict({
                'NAME': 'random_world_scaling',
                'WORLD_SCALE_RANGE': list(method_cfg.get('SCALE_RANGE', [0.95, 1.05])),
            }),
        ]
        self.dataset = dataset
        self.augmentor = DataAugmentor(
            dataset.root_path, augmentations, dataset.class_names, logger=None
        )
        self.voxel_processor = self._find_voxel_processor(dataset)

    @staticmethod
    def _find_voxel_processor(dataset):
        for processor in dataset.data_processor.data_processor_queue:
            function = processor.func if hasattr(processor, 'func') else processor
            if getattr(function, '__name__', '') == 'transform_points_to_voxels':
                return processor
        raise RegTTA3DGeometryError('Reg-TTA3D requires transform_points_to_voxels')

    def build(self, original_batch, pseudo_targets):
        view = {
            key: copy.deepcopy(value)
            for key, value in original_batch.items()
            if not _is_annotation_field(key)
            and key not in _VOXEL_FIELDS and key != 'points'
        }
        points = original_batch['points']
        device = points.device
        batch_size = int(original_batch['batch_size'])
        point_batches = []
        voxel_batches = []
        coordinate_batches = []
        count_batches = []
        transformed_targets = []
        lidar_aug_matrices = []

        for batch_index in range(batch_size):
            point_mask = points[:, 0].long() == batch_index
            valid_targets = pseudo_targets[batch_index, :, -1] > 0
            target_labels = pseudo_targets[batch_index, valid_targets, -1]
            target_geometry = pseudo_targets[batch_index, valid_targets, :-1]
            sample = {
                'points': points[point_mask, 1:].detach().cpu().numpy().copy(),
                'gt_boxes': target_geometry.detach().cpu().numpy().copy(),
                'use_lead_xyz': True,
            }
            sample = self.augmentor.forward(data_dict=sample)
            point_range_mask = common_utils.mask_points_by_range(
                sample['points'], self.dataset.point_cloud_range
            )
            sample['points'] = sample['points'][point_range_mask]
            sampled_matrix = self.dataset.set_lidar_aug_matrix(sample)['lidar_aug_matrix']
            sampled_matrix = torch.as_tensor(
                sampled_matrix, dtype=points.dtype, device=device
            )
            lidar_aug_matrices.append(
                sampled_matrix @ original_batch['lidar_aug_matrix'][batch_index]
            )
            transformed_geometry = torch.from_numpy(sample['gt_boxes']).to(
                device=device, dtype=pseudo_targets.dtype
            )
            transformed_targets.append(torch.cat((
                transformed_geometry, target_labels.to(device=device)[:, None]
            ), dim=1))

            sample = self.voxel_processor(data_dict=sample)
            sample_points = torch.from_numpy(sample['points']).to(
                device=device, dtype=points.dtype
            )
            batch_column = sample_points.new_full((sample_points.shape[0], 1), batch_index)
            point_batches.append(torch.cat((batch_column, sample_points), dim=1))
            voxel_batches.append(torch.from_numpy(sample['voxels']).to(
                device=device, dtype=points.dtype
            ))
            coordinates = torch.from_numpy(sample['voxel_coords']).to(
                device=device, dtype=torch.int32
            )
            coordinate_batches.append(torch.cat((
                coordinates.new_full((coordinates.shape[0], 1), batch_index), coordinates
            ), dim=1))
            count_batches.append(torch.from_numpy(sample['voxel_num_points']).to(
                device=device, dtype=torch.int32
            ))

        target_width = pseudo_targets.shape[-1]
        max_targets = max((targets.shape[0] for targets in transformed_targets), default=0)
        padded_targets = pseudo_targets.new_zeros((batch_size, max_targets, target_width))
        for batch_index, targets in enumerate(transformed_targets):
            padded_targets[batch_index, :targets.shape[0]] = targets

        view['points'] = torch.cat(point_batches, dim=0)
        view['voxels'] = torch.cat(voxel_batches, dim=0)
        view['voxel_coords'] = torch.cat(coordinate_batches, dim=0)
        view['voxel_num_points'] = torch.cat(count_batches, dim=0)
        view['lidar_aug_matrix'] = torch.stack(lidar_aug_matrices, dim=0)
        view['gt_boxes'] = padded_targets
        view['tta_proposal_boxes'] = padded_targets[..., :-1]
        return view
