from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class K4TrainingError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


def _weights(value, size: int) -> NDArray[np.float64]:
    weights = np.asarray(value)
    if weights.dtype.kind not in 'iuf' or weights.shape != (size,):
        raise K4TrainingError('Reliability must be a real vector aligned with predictions')
    if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise K4TrainingError('Reliability must lie in [0, 1] without clipping')
    return np.array(weights, dtype=np.float64, copy=True)


def build_k4_pseudo_info(prediction, thresholds):
    boxes = np.asarray(prediction['pred_boxes'])
    labels = np.asarray(prediction['pred_labels'])
    scores = np.asarray(prediction['pred_scores'])
    if boxes.ndim != 2 or boxes.shape[1] not in (7, 9):
        raise K4TrainingError('Detector boxes must have seven geometry or nine geometry/velocity columns')
    size = len(boxes)
    velocity_xy = (np.array(boxes[:, 7:9], dtype=np.float64, copy=True)
                   if boxes.shape[1] == 9 else np.zeros((size, 2), dtype=np.float64))
    reliability = _weights(prediction.get('spcra_reliability'), size)
    accepted = np.asarray(prediction.get('spcra_k4_accepted'))
    if accepted.dtype.kind != 'b' or accepted.shape != (size,):
        raise K4TrainingError('K4 acceptance must be an aligned boolean sidecar')
    if labels.shape != (size,) or scores.shape != (size,):
        raise K4TrainingError('Prediction labels/scores must align with boxes')
    if not np.isfinite(labels).all() or np.any((labels < 1) | (labels > 10) | (labels != np.floor(labels))):
        raise K4TrainingError('Prediction labels must remain one-based classes 1..10')
    negative = np.asarray(thresholds.get('NEG_THRESH', [0.]*10), dtype=np.float64)
    positive = np.asarray(thresholds['SCORE_THRESH'], dtype=np.float64)
    if negative.shape != (10,) or positive.shape != (10,):
        raise K4TrainingError('Pseudo thresholds must have ten class entries')
    valid = (np.isfinite(boxes[:, :7]).all(axis=1) & np.isfinite(velocity_xy).all(axis=1)
             & (boxes[:, 3:6] > 0).all(axis=1)
             & np.isfinite(scores) & (scores >= 0) & (scores <= 1))
    classes = labels.astype(np.int64) - 1
    keep = valid & ((scores >= negative[classes]) | accepted)
    signed = np.where(accepted, labels, -labels)
    gt_boxes = np.column_stack((boxes[keep, :7], signed[keep], scores[keep]))
    return {'gt_boxes': gt_boxes, 'reliability_weights': reliability[keep],
            'velocity_xy': velocity_xy[keep]}


def inject_k4_pseudo_labels(batch, infos) -> None:
    if len(infos) != batch['batch_size']:
        raise K4TrainingError('Every batch frame must have an explicit pseudo record')
    positives = []
    for info in infos:
        boxes = np.asarray(info['gt_boxes'])
        if boxes.ndim != 2 or boxes.shape[1] != 9 or not np.isfinite(boxes).all():
            raise K4TrainingError('Pseudo schema must be finite [box7,class,score]')
        weights = _weights(info.get('reliability_weights'), len(boxes))
        velocity_xy = np.asarray(info.get('velocity_xy'))
        if velocity_xy.dtype.kind not in 'iuf' or velocity_xy.shape != (len(boxes), 2):
            raise K4TrainingError('Velocity must be a real Nx2 sidecar aligned with pseudo rows')
        if not np.isfinite(velocity_xy).all():
            raise K4TrainingError('Persisted velocity must be finite')
        velocity_xy = np.array(velocity_xy, dtype=np.float64, copy=True)
        labels = boxes[:, 7]
        if np.any((np.abs(labels) < 1) | (np.abs(labels) > 10) | (labels != np.floor(labels))):
            raise K4TrainingError('Signed pseudo labels must be classes +/-1..10')
        if np.any(boxes[:, 3:6] <= 0) or np.any((boxes[:, 8] < 0) | (boxes[:, 8] > 1)):
            raise K4TrainingError('Pseudo geometry/scores must be valid')
        keep = labels > 0
        positives.append((boxes[keep], weights[keep], velocity_xy[keep]))
    max_boxes = max((len(boxes) for boxes, _, _ in positives), default=0)
    annotations = np.zeros((len(infos), max_boxes, 10), dtype=np.float64)
    reliability = np.zeros((len(infos), max_boxes), dtype=np.float64)
    for index, (boxes, weights, velocity_xy) in enumerate(positives):
        annotations[index, :len(boxes), :7] = boxes[:, :7]
        annotations[index, :len(boxes), 7:9] = velocity_xy
        annotations[index, :len(boxes), 9] = boxes[:, 7]
        reliability[index, :len(boxes)] = weights
    batch['gt_boxes'] = annotations
    batch['tta_pseudo_weights'] = reliability
    batch['tta_pseudo_reg_weights'] = reliability.copy()
    batch['spcra_k4'] = True
    batch.pop('tta_pseudo_actions', None)


def positive_query_weights(assigned_gt_inds, reliability, *, code_size=10):
    assigned = np.asarray(assigned_gt_inds)
    weights = _weights(reliability, len(reliability))
    if assigned.ndim != 1 or assigned.dtype.kind not in 'iu':
        raise K4TrainingError('Query assignments must be an integer vector')
    if np.any((assigned < -1) | (assigned > len(weights))):
        raise K4TrainingError('Query assignment is outside the pseudo table')
    if code_size <= 0:
        raise K4TrainingError('Regression code size must be positive')
    positive = assigned > 0
    classification = (assigned == 0).astype(np.float64)
    regression = np.zeros((len(assigned), code_size), dtype=np.float64)
    classification[positive] = weights[assigned[positive] - 1]
    regression[positive, :] = weights[assigned[positive] - 1, None]
    return classification, regression
