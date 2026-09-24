from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


class K4InputError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True)  # noqa: SLOTS_OK - Python 3.8 runtime.
class Predictions:
    """Caller-owned arrays are snapshotted and validated at evaluation, never mutated."""

    boxes: ArrayLike
    labels: ArrayLike
    scores: ArrayLike
    camera_support: ArrayLike | None = None
    lidar_aug_matrix: ArrayLike | None = None


@dataclass(frozen=True)  # noqa: SLOTS_OK - Python 3.8 runtime.
class K4Policy:
    max_center_distance: float = 1.0
    min_proposal_score: float = 0.0
    support_tau: float = 3.0
    camera_rescue_enabled: bool = False
    camera_low_score: float = 0.05
    camera_thresh: float = 0.60

    def __post_init__(self) -> None:
        positive = (self.max_center_distance, self.support_tau)
        probabilities = (self.min_proposal_score, self.camera_low_score, self.camera_thresh)
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise K4InputError('Distance limit and support_tau must be finite and positive')
        if not all(np.isfinite(value) and 0 <= value <= 1 for value in probabilities):
            raise K4InputError('Score and support thresholds must lie in [0, 1]')
        if type(self.camera_rescue_enabled) is not bool:
            raise K4InputError('camera_rescue_enabled must be boolean')


@dataclass(frozen=True)  # noqa: SLOTS_OK - Python 3.8 runtime.
class K4Result:
    reliability: NDArray[np.float64]
    view_quality: NDArray[np.float64]
    match_indices: NDArray[np.int64]
    reference_mask: NDArray[np.bool_]
    view_masks: tuple[NDArray[np.bool_], ...]
    reference_rescue_mask: NDArray[np.bool_]
    view_rescue_masks: tuple[NDArray[np.bool_], ...]
    coverage: NDArray[np.float64]
    support: NDArray[np.float64]


@dataclass(frozen=True)  # noqa: SLOTS_OK - Python 3.8 runtime.
class _Prepared:
    boxes: NDArray[np.float64]
    labels: NDArray[np.float64]
    scores: NDArray[np.float64]
    eligible: NDArray[np.bool_]
    rescued: NDArray[np.bool_]


def _real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf':
        raise K4InputError(f'{name} must contain real numeric values')
    return np.array(array, dtype=np.float64, copy=True)


def _prepare(prediction: Predictions, policy: K4Policy) -> _Prepared:
    boxes = _real_array(prediction.boxes, 'boxes')
    labels = _real_array(prediction.labels, 'labels')
    scores = _real_array(prediction.scores, 'scores')
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise K4InputError('boxes must have shape (N, 7)')
    if labels.shape != (len(boxes),) or scores.shape != (len(boxes),):
        raise K4InputError('labels and scores must have shape (N,) matching boxes')
    valid = np.isfinite(boxes).all(axis=1) & (boxes[:, 3:6] > 0).all(axis=1)
    valid &= np.isfinite(scores) & (scores >= 0) & (scores <= 1)
    valid &= np.isfinite(labels) & (labels >= 1) & (labels <= 10) & (labels == np.floor(labels))
    eligible = scores >= policy.min_proposal_score
    rescued = np.zeros(scores.shape, dtype=np.bool_)
    if prediction.camera_support is not None:
        support = _real_array(prediction.camera_support, 'camera_support')
        if support.shape != scores.shape or not np.isfinite(support).all():
            raise K4InputError('camera_support must be finite and aligned with scores')
        if np.any((support < 0) | (support > 1)):
            raise K4InputError('camera_support must lie in [0, 1]')
        if policy.camera_rescue_enabled:
            rescued = (~eligible & (scores >= policy.camera_low_score)
                       & (support >= policy.camera_thresh))
            eligible |= rescued
    if prediction.lidar_aug_matrix is not None:
        matrix = _real_array(prediction.lidar_aug_matrix, 'lidar_aug_matrix')
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise K4InputError('lidar_aug_matrix must be a finite 4x4 matrix')
        if not np.array_equal(matrix[3], [0., 0., 0., 1.]):
            raise K4InputError('lidar_aug_matrix must be affine')
        linear = matrix[:3, :3]
        scale = float(np.linalg.norm(linear[:, 0]))
        if not np.isfinite(scale) or scale <= 0:
            raise K4InputError('lidar_aug_matrix must have positive finite scale')
        rotation = linear / scale
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=1e-6)
                or not np.allclose(rotation[2, :2], 0., atol=1e-6)
                or not np.allclose(rotation[:2, 2], 0., atol=1e-6)):
            raise K4InputError('Only yaw-preserving uniform scale/rotation/reflection transforms are supported')
        inverse = np.linalg.inv(linear)
        boxes[valid, :3] = (boxes[valid, :3] - matrix[:3, 3]) @ inverse.T
        boxes[valid, 3:6] /= scale
        directions = np.column_stack((np.cos(boxes[valid, 6]), np.sin(boxes[valid, 6])))
        directions = directions @ inverse[:2, :2].T
        boxes[valid, 6] = np.arctan2(directions[:, 1], directions[:, 0])
        valid &= np.isfinite(boxes).all(axis=1) & (boxes[:, 3:6] > 0).all(axis=1)
    return _Prepared(boxes, labels, scores, eligible & valid, rescued & valid)


def compute_k4_reliability(
    reference: Predictions, views: tuple[Predictions, ...], policy: K4Policy,
) -> K4Result:
    """Use d/d_max + .5*mean(abs(delta_size))/max(mean(ref_size),1e-3) + .5*abs(delta_score).

    Each unmatched view contributes zero to the fixed four-view mean. Diagnostics
    are per-view coverage against min(eligible counts), and 1-exp(-matches/tau).
    """
    if len(views) != 4:
        raise K4InputError('Formal K4 requires exactly four views')
    clean = _prepare(reference, policy)
    prepared_views = tuple(_prepare(view, policy) for view in views)
    quality = np.zeros((len(clean.boxes), 4), dtype=np.float64)
    matches = np.full(quality.shape, -1, dtype=np.int64)
    coverage = np.zeros(4, dtype=np.float64)
    support = np.zeros(4, dtype=np.float64)
    reference_indices = np.flatnonzero(clean.eligible)
    traversal = reference_indices[np.lexsort((reference_indices, -clean.scores[reference_indices]))]
    for view_index, view in enumerate(prepared_views):
        available = view.eligible.copy()
        for reference_index in traversal:
            candidates = np.flatnonzero(available & (view.labels == clean.labels[reference_index]))
            if candidates.size == 0:
                continue
            distances = np.hypot.reduce(view.boxes[candidates, :3] - clean.boxes[reference_index, :3], axis=1)
            nearest = int(np.lexsort((candidates, distances))[0])
            distance = distances[nearest]
            if distance > policy.max_center_distance:
                continue
            matched = candidates[nearest]
            available[matched] = False
            matches[reference_index, view_index] = matched
            reference_size = clean.boxes[reference_index, 3:6]
            size_cost = np.abs(reference_size - view.boxes[matched, 3:6]).mean() / max(reference_size.mean(), 1e-3)
            score_cost = abs(clean.scores[reference_index] - view.scores[matched])
            cost = distance / policy.max_center_distance + 0.5 * size_cost + 0.5 * score_cost
            quality[reference_index, view_index] = np.exp(-cost)
        matched_count = np.count_nonzero(matches[:, view_index] >= 0)
        coverage[view_index] = matched_count / max(min(len(reference_indices), np.count_nonzero(view.eligible)), 1)
        support[view_index] = -np.expm1(-matched_count / policy.support_tau)
    reliability = np.clip(quality.mean(axis=1), 0., 1.)
    view_masks = tuple(view.eligible for view in prepared_views)
    view_rescue_masks = tuple(view.rescued for view in prepared_views)
    arrays = (reliability, quality, matches, clean.eligible, clean.rescued,
              coverage, support, *view_masks, *view_rescue_masks)
    for array in arrays:
        array.setflags(write=False)
    return K4Result(
        reliability, quality, matches, clean.eligible, view_masks,
        clean.rescued, view_rescue_masks, coverage, support,
    )
