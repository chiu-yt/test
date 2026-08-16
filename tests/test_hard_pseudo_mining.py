import numpy as np

from pcdet.utils.hard_pseudo_mining import hard_pseudo_mining


def _boxes(labels, scores):
    boxes = np.zeros((len(labels), 9), dtype=np.float32)
    boxes[:, 3:6] = 1.0
    boxes[:, 7] = np.asarray(labels, dtype=np.float32)
    boxes[:, 8] = np.asarray(scores, dtype=np.float32)
    return boxes


def test_hard_pseudo_mining_promotes_supported_medium_target_class():
    gt_boxes = _boxes(labels=[-3], scores=[0.18])
    reliability = np.asarray([0.20], dtype=np.float32)

    mined_boxes, cls_weights, reg_weights = hard_pseudo_mining(
        gt_boxes=gt_boxes,
        reliability_weights=reliability,
        score_thresh=np.asarray([0.3, 0.3, 0.3], dtype=np.float32),
        neg_thresh=np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        target_class_ids={3},
        medium_weight=0.3,
        support_thresh=0.05,
    )

    assert mined_boxes[0, 7] == 3
    assert cls_weights[0] == np.float32(0.3)
    assert reg_weights[0] == np.float32(0.1)


def test_hard_pseudo_mining_ignores_unsupported_medium_target_class():
    gt_boxes = _boxes(labels=[-3], scores=[0.18])
    reliability = np.asarray([0.01], dtype=np.float32)

    mined_boxes, cls_weights, reg_weights = hard_pseudo_mining(
        gt_boxes=gt_boxes,
        reliability_weights=reliability,
        score_thresh=np.asarray([0.3, 0.3, 0.3], dtype=np.float32),
        neg_thresh=np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        target_class_ids={3},
        medium_weight=0.3,
        support_thresh=0.05,
    )

    assert mined_boxes[0, 7] == -3
    assert cls_weights[0] == 0.0
    assert reg_weights[0] == 0.0


def test_hard_pseudo_mining_keeps_high_score_positive_at_full_weight():
    gt_boxes = _boxes(labels=[3], scores=[0.35])
    reliability = np.asarray([0.0], dtype=np.float32)

    mined_boxes, cls_weights, reg_weights = hard_pseudo_mining(
        gt_boxes=gt_boxes,
        reliability_weights=reliability,
        score_thresh=np.asarray([0.3, 0.3, 0.3], dtype=np.float32),
        neg_thresh=np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        target_class_ids={3},
        medium_weight=0.3,
        support_thresh=0.05,
    )

    assert mined_boxes[0, 7] == 3
    assert cls_weights[0] == 1.0
    assert reg_weights[0] == 1.0


def test_hard_pseudo_mining_uses_class_name_threshold_overrides():
    gt_boxes = _boxes(labels=[-3], scores=[0.12])
    reliability = np.asarray([0.20], dtype=np.float32)

    mined_boxes, cls_weights, reg_weights = hard_pseudo_mining(
        gt_boxes=gt_boxes,
        reliability_weights=reliability,
        score_thresh=np.asarray([0.3, 0.3, 0.3], dtype=np.float32),
        neg_thresh=np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        target_class_ids={3},
        class_names=['car', 'truck', 'construction_vehicle'],
        medium_weight=0.3,
        support_thresh=0.05,
        low_thresh={'construction_vehicle': 0.15},
    )

    assert mined_boxes[0, 7] == -3
    assert cls_weights[0] == 0.0
    assert reg_weights[0] == 0.0
