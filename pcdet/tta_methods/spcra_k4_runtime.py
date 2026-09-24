from __future__ import annotations

import copy
from contextlib import contextmanager, nullcontext
import random
import sys

import numpy as np

from .spcra_k4_config import validate_k4_config
from .spcra_k4_core import K4InputError, Predictions, compute_k4_reliability
from .spcra_k4_evidence import K4EvidenceInput, K4EvidenceSink, build_k4_evidence


REFERENCE_FIELDS = frozenset((
    'batch_size', 'points', 'frame_id', 'metadata', 'camera_imgs', 'images',
    'lidar_aug_matrix', 'img_aug_matrix', 'lidar2camera', 'lidar2image',
    'camera2ego', 'camera_intrinsics', 'camera2lidar', 'image_shape',
))


def snapshot_reference(batch):
    return {key: copy.deepcopy(value) for key, value in batch.items() if key in REFERENCE_FIELDS}


@contextmanager
def preserve_inference_state(model):
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch = sys.modules.get('torch')
    torch_state = torch.get_rng_state() if torch is not None else None
    cuda_state = torch.cuda.get_rng_state_all() if torch is not None and torch.cuda.is_initialized() else None
    try:
        model.eval()
        with torch.no_grad() if torch is not None else nullcontext():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None:
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


def _compact_prediction(prediction, transform=None):
    support = prediction.get('camera_support')
    return Predictions(
        np.array(prediction['pred_boxes'][:, :7], copy=True),
        np.array(prediction['pred_labels'], copy=True),
        np.array(prediction['pred_scores'], copy=True),
        camera_support=None if support is None else np.array(support, copy=True),
        lidar_aug_matrix=None if transform is None else np.array(transform, copy=True),
    )


def run_k4_views(
    model, batch, predictions, *, view_samples, build_view, forward_view, policy,
    evidence_sink: K4EvidenceSink | None = None,
):
    if len(predictions) != len(view_samples) or not predictions:
        raise K4InputError('K4 predictions and ViewSamples must match a nonempty batch')
    for samples in view_samples:
        if len(samples.indices) != 4 or len(samples.transforms) != 4 or len(set(samples.fingerprints)) != 4:
            raise K4InputError('K4 requires four distinct realized views per frame')
    references = tuple(_compact_prediction(prediction) for prediction in predictions)
    views = [[] for _ in references]
    with preserve_inference_state(model):
        for view_index in range(4):
            indices = tuple(samples.indices[view_index] for samples in view_samples)
            transforms = tuple(samples.transforms[view_index] for samples in view_samples)
            pristine = snapshot_reference(batch)
            view = None
            outputs = None
            try:
                view = build_view(pristine, indices[0] if len(indices) == 1 else indices,
                                  transforms[0] if len(transforms) == 1 else transforms)
                outputs = forward_view(model, view)
                if len(outputs) != len(references):
                    raise K4InputError('View predictions must retain the reference batch order')
                for frame_index in range(len(references)):
                    views[frame_index].append(_compact_prediction(outputs[frame_index], transforms[frame_index]))
            finally:
                if view is not None:
                    view.clear()
                pristine.clear()
                outputs = None
                view = None
        enriched = []
        for index, reference in enumerate(references):
            frame_views = tuple(views[index])
            result = compute_k4_reliability(reference, frame_views, policy)
            prediction = dict(predictions[index])
            prediction['spcra_reliability'] = result.reliability.copy()
            prediction['spcra_k4_accepted'] = result.reference_mask.copy()
            enriched.append(prediction)
            if evidence_sink is not None:
                source = K4EvidenceInput(reference, frame_views, view_samples[index])
                evidence_sink(build_k4_evidence(source, result))
        return enriched
