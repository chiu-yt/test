"""Formal-only MOS composition; no loader, optimizer, or additional forward owner."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .spcra_k4_runtime import REFERENCE_FIELDS, run_k4_views
from .spcra_k4_training import K4TrainingError


def prepare_k4_reference(owner, batch) -> None:
    owner._prepare_sg_dfa_density_map(batch)
    batch_size = int(batch['batch_size'])
    batch['tta_proposal_boxes'] = batch['points'].new_zeros((batch_size, 0, 9))
    batch['tta_proposal_mask'] = batch['points'].new_zeros((batch_size, 0)).bool()


def run_current_k4(owner, batch, clean_predictions):
    reference = batch.pop('_spcra_k4_reference')
    predictions = owner.k4_views.predictions(clean_predictions, batch)
    context = owner.k4_views.build_context(predictions)
    capture = owner.figure7_capture
    if capture is not None:
        capture.reference(reference['points'].detach().cpu().numpy(),
                          reference['lidar_aug_matrix'].detach().cpu().numpy())
    training_fields = REFERENCE_FIELDS | {'voxels', 'voxel_coords', 'voxel_num_points',
                                         'optimizer', 'samples_seen'}
    for key in tuple(batch):
        if key not in training_fields:
            del batch[key]
    batch.update(reference)

    def build_view(pristine, indices, transforms):
        view = owner.k4_views.build_view(pristine, indices, transforms, proposal_context=context)
        if capture is not None:
            capture.view(view['points'].detach().cpu().numpy())
        return view

    return run_k4_views(
        owner.model, reference, predictions, view_samples=owner.k4_views.sample(reference),
        build_view=build_view, forward_view=owner.k4_views.forward_view,
        policy=owner.k4_views.policy,
        evidence_sink=None if capture is None else capture.evidence,
    )


def restore_student_rows(augmented: NDArray, classes: NDArray) -> NDArray:
    """Keep row-aligned classes, transformed geometry/velocity, and zero invalid rows."""
    if augmented.ndim != 3 or augmented.shape[-1] != 10 or augmented.shape[:2] != classes.shape:
        raise K4TrainingError('Student augmentation must preserve the BxNx10 pseudo row layout')
    result = augmented.copy()
    valid = ((classes > 0) & np.isfinite(result[..., :9]).all(axis=-1)
             & (result[..., 3:6] > 0).all(axis=-1))
    result[..., 9] = classes
    result[~valid] = 0
    return result


def augment_k4_student(dataset, batch, strength='mid'):
    from pcdet.utils.tta_utils import TTA_augmentation

    classes = batch['gt_boxes'][..., 9].detach().cpu().numpy().copy()
    base_transform = batch['lidar_aug_matrix'].clone()
    target = TTA_augmentation(dataset, batch, strength=strength)
    boxes = target['gt_boxes']
    restored = restore_student_rows(boxes.detach().cpu().numpy(), classes)
    target['gt_boxes'] = boxes.new_tensor(restored)
    valid = target['gt_boxes'][..., 9] > 0
    # Exact r applies only to positive-query cls/reg; dense heatmap targets stay native.
    for key in ('tta_pseudo_weights', 'tta_pseudo_reg_weights'):
        target[key] = target[key] * valid.to(target[key].dtype)
    target['lidar_aug_matrix'] = target['lidar_aug_matrix'] @ base_transform
    return target
