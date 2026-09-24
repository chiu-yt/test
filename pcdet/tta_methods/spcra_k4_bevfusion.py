from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from .spcra_k4_augmentation import (
    AUGMENTATION_LAW_ID,
    LegacyAugmentationSampler,
    should_skip_imgaug,
    world_operators_from_queue,
)
from .spcra_k4_config import K4ConfigurationError
from .spcra_k4_context import (
    DensityMapSpec,
    ReferenceProposalContext,
    compose_view_calibration,
)
from .spcra_k4_core import K4Policy
from .spcra_k4_seeding import sample_unique_views
from pcdet.utils.inference_utils import forward_without_annotations


ViewSampler = LegacyAugmentationSampler


class K4BEVFusionViews:
    def __init__(self, dataset, spcra_config):
        if dataset is None:
            raise K4ConfigurationError('Formal K4 requires the existing BEVFusion dataset processor')
        processors = [processor for processor in dataset.data_processor.data_processor_queue
                      if processor.func.__name__ == 'transform_points_to_voxels']
        if len(processors) != 1:
            raise K4ConfigurationError('Formal K4 requires exactly one voxel processor')
        self.voxel_processor = processors[0]
        self.dataset = dataset
        self.point_cloud_range = np.asarray(dataset.point_cloud_range)
        range_values = [float(value) for value in self.point_cloud_range]
        self.density_spec = DensityMapSpec(
            point_cloud_range=(
                range_values[0], range_values[1], range_values[2],
                range_values[3], range_values[4], range_values[5],
            ),
            grid_size=int(spcra_config.get('DENSITY_GRID_SIZE', 32)),
        )
        self.config = spcra_config
        self.policy = K4Policy(
            max_center_distance=float(spcra_config.get('MAX_CENTER_DISTANCE', 1.)),
            min_proposal_score=float(spcra_config.get('MIN_PROPOSAL_SCORE', 0.)),
            support_tau=float(spcra_config.get('SUPPORT_TAU', 3.0)),
            camera_rescue_enabled=bool(spcra_config.get('CAMERA_RESCUE_ENABLED', False)),
            camera_low_score=float(spcra_config.get('CAMERA_LOW_SCORE', .05)),
            camera_thresh=float(spcra_config.get('CAMERA_THRESH', .6)),
        )
        self.drop_rate = float(spcra_config.get('DROP_RATE', .1))
        if not np.isfinite(self.drop_rate):
            raise K4ConfigurationError('K4 DROP_RATE must be finite')

    def sample(self, batch):
        samples = []
        points = batch['points'].detach().cpu().numpy()
        augmentor = getattr(self.dataset, 'tta_data_augmentor', None)
        if augmentor is None:
            augmentor = getattr(self.dataset, 'data_augmentor', None)
        if augmentor is None or not hasattr(augmentor, 'data_augmentor_queue'):
            raise K4ConfigurationError('Formal K4 requires an effective DataAugmentor queue')
        camera_imgs_present = 'camera_imgs' in batch
        skip_imgaug = should_skip_imgaug(
            camera_imgs_present=camera_imgs_present,
            camera_is_tensor=camera_imgs_present and torch.is_tensor(batch['camera_imgs']),
            img_process_infos_present='img_process_infos' in batch,
        )
        operators = world_operators_from_queue(
            augmentor.data_augmentor_queue, skip_imgaug=skip_imgaug,
        )
        for index in range(batch['batch_size']):
            reference = np.ascontiguousarray(points[points[:, 0] == index, 1:])
            digest = hashlib.sha256(reference.dtype.str.encode() + str(reference.shape).encode() + reference.tobytes()).hexdigest()
            sampler = ViewSampler(len(reference), self.drop_rate, operators)
            samples.append(sample_unique_views(
                batch['metadata'][index]['token'], base_seed=self.config.get('BASE_SEED', 1024),
                schema_version='k4_v1', reference_digest=digest,
                law_identifier=AUGMENTATION_LAW_ID, build_view=sampler,
            ))
        return tuple(samples)

    def build_context(self, reference_predictions):
        return ReferenceProposalContext.from_predictions(reference_predictions)

    def build_view(self, pristine, indices, transforms, proposal_context=None):
        batch_size = pristine['batch_size']
        if batch_size == 1:
            indices, transforms = (indices,), (transforms,)
        device = pristine['points'].device
        points, voxels, coordinates, counts, deltas = [], [], [], [], []
        for index in range(batch_size):
            reference = pristine['points'][pristine['points'][:, 0] == index, 1:]
            retained = torch.as_tensor(indices[index].copy(), device=device, dtype=torch.long)
            frame_points = reference[retained].clone()
            transform = torch.as_tensor(transforms[index].copy(), device=device, dtype=frame_points.dtype)
            frame_points[:, :3] = frame_points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
            deltas.append(transform)
            processed = self.voxel_processor(data_dict={
                'points': frame_points.detach().cpu().numpy(), 'use_lead_xyz': True,
            })
            frame_coords = torch.as_tensor(processed['voxel_coords'], device=device, dtype=torch.int32)
            coordinates.append(torch.cat((frame_coords.new_full((len(frame_coords), 1), index), frame_coords), dim=1))
            voxels.append(torch.as_tensor(processed['voxels'], device=device, dtype=frame_points.dtype))
            counts.append(torch.as_tensor(processed['voxel_num_points'], device=device, dtype=torch.int32))
            points.append(torch.cat((frame_points.new_full((len(frame_points), 1), index), frame_points), dim=1))
        view_deltas = torch.stack(deltas)
        pristine.update(
            points=torch.cat(points), voxels=torch.cat(voxels),
            voxel_coords=torch.cat(coordinates), voxel_num_points=torch.cat(counts),
            lidar_aug_matrix=compose_view_calibration(
                view_deltas, pristine['lidar_aug_matrix'],
            ),
        )
        context = proposal_context or ReferenceProposalContext.empty(batch_size)
        adapter_inputs = context.for_view(pristine['points'], view_deltas, self.density_spec)
        pristine.update(
            tta_density_map=adapter_inputs.density_map,
            tta_proposal_boxes=adapter_inputs.proposal_boxes,
            tta_proposal_mask=adapter_inputs.proposal_mask,
        )
        return pristine

    def predictions(self, pred_dicts, batch):
        result = []
        for index, prediction in enumerate(pred_dicts):
            compact = {key: prediction[key].detach().cpu().numpy().copy()
                       for key in ('pred_boxes', 'pred_labels', 'pred_scores')}
            if self.policy.camera_rescue_enabled:
                image_bev = batch['spatial_features_img'][index:index + 1]
                centers = prediction['pred_boxes'][:, :2].to(image_bev)
                extent = image_bev.new_tensor(self.point_cloud_range)
                normalized = (centers - extent[:2]) / (extent[3:5] - extent[:2]) * 2. - 1.
                if len(centers):
                    grid = normalized.view(1, -1, 1, 2)
                    sampled = F.grid_sample(image_bev, grid, mode='bilinear', align_corners=True)
                    energy = sampled[0, :, :, 0].abs().mean(dim=0)
                    centered = (energy - energy.mean()) / energy.std(unbiased=False).clamp_min(1e-6)
                    support = torch.sigmoid(float(self.config.get('CAMERA_SUPPORT_SCALE', 4.)) * centered)
                    compact['camera_support'] = support.detach().cpu().numpy().copy()
                else:
                    compact['camera_support'] = np.zeros(0)
            result.append(compact)
        return result

    def forward_view(self, model, batch):
        predictions, _ = forward_without_annotations(model, batch)
        return self.predictions(predictions, batch)
