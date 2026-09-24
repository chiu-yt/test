from __future__ import annotations

import json
from typing import Mapping

import numpy as np

from pcdet.utils.figure7_artifacts import array_digest
from pcdet.utils.figure7_schema import CLASS_NAMES, Candidate

from .contracts import Figure7LoadError, LedgerEntry


def validate_arrays(arrays: Mapping[str, np.ndarray], candidate: Candidate) -> None:
    prefixes = ('reference', 'view_0', 'view_1', 'view_2', 'view_3')
    required = {'match_indices', 'view_quality', 'reliability'}
    required.update(prefix + '_' + suffix for prefix in prefixes for suffix in (
        'points', 'transform', 'boxes', 'labels', 'scores', 'mask', 'rescue_mask'))
    if required - set(arrays):
        raise Figure7LoadError('selected record is missing required arrays')
    if any(value.dtype.kind not in 'biuf' for value in arrays.values()):
        raise Figure7LoadError('arrays must contain real numeric or boolean values')
    for prefix in prefixes:
        points, boxes = arrays[prefix + '_points'], arrays[prefix + '_boxes']
        labels, scores = arrays[prefix + '_labels'], arrays[prefix + '_scores']
        mask, rescue = arrays[prefix + '_mask'], arrays[prefix + '_rescue_mask']
        transform = arrays[prefix + '_transform']
        if points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points).all():
            raise Figure7LoadError(prefix + ' points are invalid')
        if boxes.ndim != 2 or boxes.shape[1] != 7:
            raise Figure7LoadError(prefix + ' boxes are invalid')
        if any(value.shape != (len(boxes),) for value in (labels, scores, mask, rescue)):
            raise Figure7LoadError(prefix + ' prediction arrays are misaligned')
        if mask.dtype.kind != 'b' or rescue.dtype.kind != 'b' or np.any(rescue & ~mask):
            raise Figure7LoadError(prefix + ' masks are invalid')
        if labels.dtype.kind not in 'iuf' or not np.equal(labels[mask], np.floor(labels[mask])).all():
            raise Figure7LoadError(prefix + ' labels must be integers')
        if (not np.isfinite(boxes[mask]).all() or not np.isfinite(scores[mask]).all()
                or np.any((labels[mask] < 1) | (labels[mask] > len(CLASS_NAMES)))):
            raise Figure7LoadError(prefix + ' accepted predictions are invalid')
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.array_equal(transform[3], [0, 0, 0, 1])
                or np.linalg.det(transform[:3, :3]) == 0):
            raise Figure7LoadError(prefix + ' transform is invalid')
    count = len(arrays['reference_boxes'])
    matches, qualities = arrays['match_indices'], arrays['view_quality']
    if matches.shape != (count, 4) or qualities.shape != (count, 4):
        raise Figure7LoadError('reference correspondence arrays are misaligned')
    if matches.dtype.kind not in 'iu' or not np.isfinite(qualities).all():
        raise Figure7LoadError('correspondences must be integer indices and finite qualities')
    if arrays['reliability'].shape != (count,) or not np.isfinite(arrays['reliability']).all():
        raise Figure7LoadError('reliability is invalid')
    index = candidate.reference_index
    if not 0 <= index < count or not arrays['reference_mask'][index]:
        raise Figure7LoadError('selected proposal must be an accepted valid index')
    if (CLASS_NAMES[int(arrays['reference_labels'][index]) - 1] != candidate.class_name
            or not np.isfinite(candidate.score) or not np.isfinite(candidate.reliability)
            or abs(float(arrays['reference_scores'][index]) - candidate.score) > 1e-12
            or abs(float(arrays['reliability'][index]) - candidate.reliability) > 1e-12):
        raise Figure7LoadError('selected values disagree with captured arrays')
    for view_index in range(4):
        prefix = 'view_%d' % view_index
        column = matches[:, view_index]
        if np.any(column < -1) or np.any(column >= len(arrays[prefix + '_boxes'])):
            raise Figure7LoadError('correspondence index is invalid')
        matched = column >= 0
        indices = column[matched]
        if (not arrays['reference_mask'][matched].all()
                or not arrays[prefix + '_mask'][indices].all()
                or not np.array_equal(arrays[prefix + '_labels'][indices],
                                      arrays['reference_labels'][matched])):
            raise Figure7LoadError('matched proposals must be accepted and class-consistent')


def validate_observation(candidate: Candidate, arrays: Mapping[str, np.ndarray], entry: LedgerEntry) -> None:
    if (entry.identity != candidate.identity or entry.model_step != candidate.model_step
            or entry.record_id != candidate.record_id or entry.status != 'complete'):
        raise Figure7LoadError('candidate identity disagrees with checksummed observation')
    if len(entry.references) != len(arrays['reference_boxes']):
        raise Figure7LoadError('observation reference count disagrees with arrays')
    for index, reference in enumerate(entry.references):
        label = float(arrays['reference_labels'][index])
        class_name = CLASS_NAMES[int(label) - 1] if label.is_integer() and 1 <= label <= len(CLASS_NAMES) else 'invalid'
        score = float(arrays['reference_scores'][index])
        expected_score = score if np.isfinite(score) else None
        if (reference.index != index or reference.class_name != class_name
                or reference.score != expected_score
                or reference.accepted != bool(arrays['reference_mask'][index])
                or reference.rescued != bool(arrays['reference_rescue_mask'][index])
                or reference.matches != tuple(arrays['match_indices'][index])
                or reference.qualities != tuple(arrays['view_quality'][index])
                or reference.reliability != float(arrays['reliability'][index])):
            raise Figure7LoadError('observation references disagree with arrays')
    selected = entry.references[candidate.reference_index]
    key = selected.stable_key if candidate.pool == 'stable' else selected.variable_key
    if key != candidate.rank_key or candidate.pool not in entry.admitted_pools:
        raise Figure7LoadError('candidate ranking disagrees with checksummed observation')
    prefixes = ('reference', 'view_0', 'view_1', 'view_2', 'view_3')
    expected = {
        'point_counts': [len(arrays[prefix + '_points']) for prefix in prefixes],
        'prediction_counts': [len(arrays[prefix + '_boxes']) for prefix in prefixes],
        'accepted_counts': [int(np.count_nonzero(arrays[prefix + '_mask'])) for prefix in prefixes],
        'rescued_counts': [int(np.count_nonzero(arrays[prefix + '_rescue_mask'])) for prefix in prefixes],
        'point_checksums': [array_digest(arrays[prefix + '_points']) for prefix in prefixes],
        'transforms': [arrays[prefix + '_transform'].ravel().tolist() for prefix in prefixes],
        'completion': [True] * 6,
    }
    for name, value in expected.items():
        if json.dumps(entry.observation.get(name)) != json.dumps(value):
            raise Figure7LoadError('observation ' + name + ' disagrees with arrays')
    for name in ('coverage', 'support'):
        if name not in arrays or entry.observation.get(name) != arrays[name].tolist():
            raise Figure7LoadError('observation ' + name + ' disagrees with arrays')
