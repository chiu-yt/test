import argparse
import csv
import json
import math
import pickle
from pathlib import Path


CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

GEOMETRY_KEYS = ['point_count', 'point_density', 'z_span', 'z_var']
TP_PROXY_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]


def _to_list(value):
    if value is None:
        return []
    if hasattr(value, 'tolist'):
        return value.tolist()
    return list(value)


def _finite(values):
    finite_values = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            finite_values.append(value)
    return finite_values


def _percentile(values, percentile):
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * percentile / 100.0
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _summary(values):
    values = sorted(_finite(values))
    if len(values) == 0:
        return {
            'count': 0,
            'mean': None,
            'p25': None,
            'p50': None,
            'p75': None,
            'p90': None,
        }
    return {
        'count': len(values),
        'mean': sum(values) / float(len(values)),
        'p25': _percentile(values, 25),
        'p50': _percentile(values, 50),
        'p75': _percentile(values, 75),
        'p90': _percentile(values, 90),
    }


def _class_name_from_gt_label(label):
    label = int(round(float(label)))
    if label <= 0 or label > len(CLASS_NAMES):
        return None
    return CLASS_NAMES[label - 1]


def _center_dist(box_a, box_b):
    return math.sqrt((float(box_a[0]) - float(box_b[0])) ** 2 + (float(box_a[1]) - float(box_b[1])) ** 2)


def _nearest_same_class_gt(pred_box, class_name, gt_boxes):
    best_dist = None
    for gt_box in gt_boxes:
        if len(gt_box) < 8:
            continue
        if _class_name_from_gt_label(gt_box[-1]) != class_name:
            continue
        if not all(math.isfinite(float(v)) for v in gt_box):
            continue
        dist = _center_dist(pred_box, gt_box)
        if best_dist is None or dist < best_dist:
            best_dist = dist
    return best_dist


def _safe_corr(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mean_x = sum(xs) / float(len(xs))
    mean_y = sum(ys) / float(len(ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    return cov / math.sqrt(var_x * var_y)


def _metric_summary(rows, metric_name):
    metric_rows = [row for row in rows if row[metric_name] is not None]
    by_threshold = {}
    for threshold in TP_PROXY_THRESHOLDS:
        tp_rows = [
            row for row in metric_rows
            if row['nearest_same_class_gt_dist'] is not None and row['nearest_same_class_gt_dist'] <= threshold
        ]
        fp_rows = [
            row for row in metric_rows
            if row['nearest_same_class_gt_dist'] is None or row['nearest_same_class_gt_dist'] > threshold
        ]
        by_threshold[str(threshold)] = {
            'tp_proxy': _summary([row[metric_name] for row in tp_rows]),
            'fp_proxy': _summary([row[metric_name] for row in fp_rows]),
        }

    both = [row for row in metric_rows if row['nearest_same_class_gt_dist'] is not None]
    return {
        'all': _summary([row[metric_name] for row in metric_rows]),
        'by_tp_proxy_center_dist': by_threshold,
        'gt_center_dist_corr': _safe_corr(
            [row[metric_name] for row in both],
            [row['nearest_same_class_gt_dist'] for row in both]
        ),
    }


def _summarize_rows(rows):
    summary = {
        'num_predictions': len(rows),
        'num_with_geometry': sum(row['has_geometry'] for row in rows),
        'num_with_gt_match': sum(row['nearest_same_class_gt_dist'] is not None for row in rows),
        'geometry': {},
        'classes': {},
    }

    for metric_name in GEOMETRY_KEYS:
        summary['geometry'][metric_name] = _metric_summary(rows, metric_name)

    for class_name in CLASS_NAMES:
        cls_rows = [row for row in rows if row['class_name'] == class_name]
        if len(cls_rows) == 0:
            continue

        cls_summary = {
            'num_predictions': len(cls_rows),
            'num_with_geometry': sum(row['has_geometry'] for row in cls_rows),
            'num_with_gt_match': sum(row['nearest_same_class_gt_dist'] is not None for row in cls_rows),
            'tp_proxy_2m_rate': float(sum(
                row['nearest_same_class_gt_dist'] is not None and row['nearest_same_class_gt_dist'] <= 2.0
                for row in cls_rows
            )) / float(max(len(cls_rows), 1)),
            'geometry': {},
        }
        for metric_name in GEOMETRY_KEYS:
            cls_summary['geometry'][metric_name] = _metric_summary(cls_rows, metric_name)
        summary['classes'][class_name] = cls_summary

    return summary


def analyze_objects(result_pkl):
    # OpenPCDet eval artifacts are trusted local pickles; never use this on third-party files.
    with Path(result_pkl).open('rb') as f:
        annos = pickle.load(f)

    rows = []
    missing_analysis = 0
    for sample_idx, anno in enumerate(annos):
        object_analysis = anno.get('geometry_object_analysis', None)
        if object_analysis is None:
            missing_analysis += 1
            continue

        names = _to_list(anno.get('name', []))
        scores = _to_list(anno.get('score', []))
        boxes = _to_list(anno.get('boxes_lidar', []))
        labels = _to_list(anno.get('pred_labels', []))
        gt_boxes = _to_list(object_analysis.get('gt_boxes', []))
        valid_geometry = _to_list(object_analysis.get('valid_geometry', []))
        geometry_values = {key: _to_list(object_analysis.get(key, [])) for key in GEOMETRY_KEYS}

        for pred_idx, box in enumerate(boxes):
            class_name = str(names[pred_idx]) if pred_idx < len(names) else ''
            gt_dist = _nearest_same_class_gt(box, class_name, gt_boxes)
            row = {
                'sample_idx': sample_idx,
                'frame_id': anno.get('frame_id', ''),
                'pred_idx': pred_idx,
                'class_name': class_name,
                'pred_label': labels[pred_idx] if pred_idx < len(labels) else None,
                'score': scores[pred_idx] if pred_idx < len(scores) else None,
                'has_geometry': bool(valid_geometry[pred_idx]) if pred_idx < len(valid_geometry) else False,
                'nearest_same_class_gt_dist': gt_dist,
            }
            for threshold in TP_PROXY_THRESHOLDS:
                key = str(threshold).replace('.', '_')
                row['tp_proxy_center_%sm' % key] = gt_dist is not None and gt_dist <= threshold
            for metric_name in GEOMETRY_KEYS:
                value = geometry_values[metric_name][pred_idx] if pred_idx < len(geometry_values[metric_name]) else None
                row[metric_name] = None if value is None else float(value)
            rows.append(row)

    summary = _summarize_rows(rows)
    summary['num_samples'] = len(annos)
    summary['num_samples_missing_geometry_analysis'] = missing_analysis
    summary['focus_classes'] = ['pedestrian', 'traffic_cone']
    return summary, rows


def _write_csv(rows, csv_path):
    fieldnames = [
        'sample_idx', 'frame_id', 'pred_idx', 'class_name', 'pred_label', 'score',
        'point_count', 'point_density', 'z_span', 'z_var', 'has_geometry',
        'nearest_same_class_gt_dist', 'tp_proxy_center_0_5m', 'tp_proxy_center_1_0m',
        'tp_proxy_center_2_0m', 'tp_proxy_center_4_0m'
    ]
    with Path(csv_path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze exported BEVFusion LiDAR geometry statistics from a trusted local eval result.pkl'
    )
    parser.add_argument('--trusted_result_pkl', required=True,
                        help='Trusted local OpenPCDet eval result.pkl; pickle files from third parties are unsafe')
    parser.add_argument('--out_json', default=None, help='Path to save geometry summary JSON')
    parser.add_argument('--out_csv', default=None, help='Path to save per-prediction geometry CSV')
    args = parser.parse_args()

    result_pkl = Path(args.trusted_result_pkl)
    out_json = Path(args.out_json) if args.out_json is not None else result_pkl.parent / 'geometry_probe_summary.json'
    out_csv = Path(args.out_csv) if args.out_csv is not None else result_pkl.parent / 'geometry_probe_per_prediction.csv'

    summary, rows = analyze_objects(result_pkl)
    with out_json.open('w') as f:
        json.dump(summary, f, indent=2)
    _write_csv(rows, out_csv)

    print('Saved geometry summary to %s' % out_json)
    print('Saved geometry rows to %s' % out_csv)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
