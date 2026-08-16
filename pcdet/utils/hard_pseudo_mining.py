import numpy as np


def _threshold_array(thresh_cfg, default_thresh, class_names):
    thresh = np.asarray(default_thresh, dtype=np.float32).copy()
    if thresh_cfg is None:
        return thresh
    if isinstance(thresh_cfg, dict):
        for cls_idx, cls_name in enumerate(class_names):
            if cls_name in thresh_cfg:
                thresh[cls_idx] = float(thresh_cfg[cls_name])
        return thresh
    return np.asarray(thresh_cfg, dtype=np.float32)


def hard_pseudo_mining(gt_boxes, reliability_weights, score_thresh, neg_thresh,
                       target_class_ids, class_names=None, medium_weight=0.3,
                       medium_reg_weight=0.1, support_thresh=0.05,
                       high_thresh=None, low_thresh=None):
    """
    HINTED-style hard pseudo mining.

    Pseudo boxes arrive in the 9-column memory-bank layout
    ``[x, y, z, dx, dy, dz, yaw, cls, score]``. Boxes whose ``cls`` is already
    positive are "high-score" pseudo labels and keep full weight 1.0. Boxes with
    negative ``cls`` (the medium/low band that the self-training thresholds
    ignored) are promoted back to positive with fractional weight only when the
    class is a target class, the score is above the low threshold, and the
    SPCRA/reliability support is strong enough; everything else stays ignored
    with weight 0.0.

    Args:
        gt_boxes: (M, 9) float32 numpy array in memory-bank layout.
        reliability_weights: (M,) float32 support/reliability per box.
        score_thresh: (C,) float32 high-score thresholds per class (1-based).
        neg_thresh: (C,) float32 low/negative thresholds per class (1-based).
        target_class_ids: iterable of 1-based class ids eligible for rescue.
        class_names: optional class-name list for dict threshold overrides.
        medium_weight: loss weight assigned to rescued medium boxes.
        support_thresh: minimum reliability to rescue a medium box.
        high_thresh: optional (C,) override for ``score_thresh``.
        low_thresh: optional (C,) override for ``neg_thresh``.

    Returns:
        new_gt_boxes: (M, 9) boxes with rescued medium labels flipped positive.
        pseudo_cls_weights: (M,) float32 per-box classification weights.
        pseudo_reg_weights: (M,) float32 per-box box-regression weights.
    """
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
    if gt_boxes.ndim != 2 or gt_boxes.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return gt_boxes, empty, empty

    n = gt_boxes.shape[0]
    score_thresh = np.asarray(score_thresh, dtype=np.float32)
    neg_thresh = np.asarray(neg_thresh, dtype=np.float32)
    if class_names is None:
        class_names = []
    high_thresh = _threshold_array(high_thresh, score_thresh, class_names)
    low_thresh = _threshold_array(low_thresh, neg_thresh, class_names)

    reliability = np.asarray(reliability_weights, dtype=np.float32).reshape(-1)
    if reliability.shape[0] != n:
        reliability = np.ones(n, dtype=np.float32)

    target_ids = set(int(i) for i in target_class_ids)
    num_cls = score_thresh.shape[0]

    new_boxes = gt_boxes.copy()
    labels = new_boxes[:, 7].astype(np.int64)
    scores = new_boxes[:, 8].astype(np.float32)
    abs_labels = np.abs(labels)
    cls_weights = np.ones(n, dtype=np.float32)
    reg_weights = np.ones(n, dtype=np.float32)

    for i in range(n):
        cls = abs_labels[i]
        if cls < 1 or cls > num_cls:
            cls_weights[i] = 0.0
            reg_weights[i] = 0.0
            continue
        c = cls - 1
        if labels[i] > 0 and scores[i] >= high_thresh[c]:
            cls_weights[i] = 1.0
            reg_weights[i] = 1.0
            continue
        # Currently ignored: rescue only supported target-class medium boxes.
        if cls in target_ids and scores[i] >= low_thresh[c] and reliability[i] >= support_thresh:
            new_boxes[i, 7] = cls
            cls_weights[i] = float(medium_weight)
            reg_weights[i] = float(medium_reg_weight)
        else:
            new_boxes[i, 7] = -cls
            cls_weights[i] = 0.0
            reg_weights[i] = 0.0

    return new_boxes, cls_weights.astype(np.float32, copy=False), reg_weights.astype(np.float32, copy=False)
