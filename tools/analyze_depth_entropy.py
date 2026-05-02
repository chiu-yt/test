import argparse
import json
import math
import pickle
from pathlib import Path


CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


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
    values = _finite(values)
    if len(values) == 0:
        return {
            'count': 0,
            'mean': None,
            'p50': None,
            'p75': None,
            'p90': None,
        }
    values = sorted(values)
    return {
        'count': len(values),
        'mean': sum(values) / float(len(values)),
        'p50': _percentile(values, 50),
        'p75': _percentile(values, 75),
        'p90': _percentile(values, 90),
    }


def analyze(result_pkl):
    # OpenPCDet eval artifacts are trusted local pickles; never use this on third-party files.
    with Path(result_pkl).open('rb') as f:
        annos = pickle.load(f)

    means, p50s, p75s, p90s = [], [], [], []
    camera_means = []
    missing = 0
    for anno in annos:
        stats = anno.get('depth_entropy_analysis', None)
        if stats is None:
            missing += 1
            continue

        for key, values in [('mean', means), ('p50', p50s), ('p75', p75s), ('p90', p90s)]:
            value = stats.get(key, None)
            if value is not None:
                values.append(value)

        for value in stats.get('per_camera_mean', []):
            if value is not None:
                camera_means.append(value)

    return {
        'num_samples': len(annos),
        'num_samples_with_depth_entropy': len(annos) - missing,
        'num_samples_missing_depth_entropy': missing,
        'sample_mean_entropy': _summary(means),
        'sample_p50_entropy': _summary(p50s),
        'sample_p75_entropy': _summary(p75s),
        'sample_p90_entropy': _summary(p90s),
        'camera_mean_entropy': _summary(camera_means),
    }


def _to_list(value):
    if value is None:
        return []
    if hasattr(value, 'tolist'):
        return value.tolist()
    return list(value)


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
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
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


def _summarize_object_rows(rows):
    summary = {
        'num_predictions': len(rows),
        'num_with_depth_entropy': sum(row['has_depth_entropy'] for row in rows),
        'num_with_gt_match': sum(row['nearest_same_class_gt_dist'] is not None for row in rows),
        'classes': {},
    }
    valid_entropy = [row['depth_entropy'] for row in rows if row['has_depth_entropy']]
    summary['depth_entropy'] = _summary(valid_entropy)

    for class_name in CLASS_NAMES:
        cls_rows = [row for row in rows if row['class_name'] == class_name]
        if len(cls_rows) == 0:
            continue

        cls_entropy = [row['depth_entropy'] for row in cls_rows if row['has_depth_entropy']]
        cls_dists = [row['nearest_same_class_gt_dist'] for row in cls_rows if row['nearest_same_class_gt_dist'] is not None]
        both = [row for row in cls_rows if row['has_depth_entropy'] and row['nearest_same_class_gt_dist'] is not None]
        tp_2m = [row for row in cls_rows if row['nearest_same_class_gt_dist'] is not None and row['nearest_same_class_gt_dist'] <= 2.0]
        fp_2m = [row for row in cls_rows if row['nearest_same_class_gt_dist'] is None or row['nearest_same_class_gt_dist'] > 2.0]

        cls_summary = {
            'num_predictions': len(cls_rows),
            'num_with_depth_entropy': len(cls_entropy),
            'missing_depth_entropy_rate': 1.0 - (float(len(cls_entropy)) / float(max(len(cls_rows), 1))),
            'num_with_gt_match': len(cls_dists),
            'depth_entropy': _summary(cls_entropy),
            'gt_center_dist': _summary(cls_dists),
            'entropy_gt_dist_corr': _safe_corr(
                [row['depth_entropy'] for row in both],
                [row['nearest_same_class_gt_dist'] for row in both]
            ),
            'tp_proxy_2m_rate': float(len(tp_2m)) / float(max(len(cls_rows), 1)),
            'tp_proxy_2m_entropy': _summary([row['depth_entropy'] for row in tp_2m if row['has_depth_entropy']]),
            'fp_proxy_2m_entropy': _summary([row['depth_entropy'] for row in fp_2m if row['has_depth_entropy']]),
            'visible_cams': _summary([row['num_visible_cams'] for row in cls_rows]),
        }

        if len(cls_entropy) >= 4:
            entropy_sorted = sorted(cls_entropy)
            low_cut = _percentile(entropy_sorted, 25)
            high_cut = _percentile(entropy_sorted, 75)
            low_rows = [row for row in cls_rows if row['has_depth_entropy'] and row['depth_entropy'] <= low_cut]
            high_rows = [row for row in cls_rows if row['has_depth_entropy'] and row['depth_entropy'] >= high_cut]
            cls_summary.update({
                'low_entropy_q25': low_cut,
                'high_entropy_q75': high_cut,
                'low_entropy_tp_proxy_2m_rate': float(sum(row['nearest_same_class_gt_dist'] is not None and row['nearest_same_class_gt_dist'] <= 2.0 for row in low_rows)) / float(max(len(low_rows), 1)),
                'high_entropy_tp_proxy_2m_rate': float(sum(row['nearest_same_class_gt_dist'] is not None and row['nearest_same_class_gt_dist'] <= 2.0 for row in high_rows)) / float(max(len(high_rows), 1)),
            })
        summary['classes'][class_name] = cls_summary
    return summary


def analyze_objects(result_pkl):
    with Path(result_pkl).open('rb') as f:
        annos = pickle.load(f)

    rows = []
    missing_analysis = 0
    for sample_idx, anno in enumerate(annos):
        object_analysis = anno.get('depth_entropy_object_analysis', None)
        if object_analysis is None:
            missing_analysis += 1
            continue

        names = _to_list(anno.get('name', []))
        scores = _to_list(anno.get('score', []))
        boxes = _to_list(anno.get('boxes_lidar', []))
        labels = _to_list(anno.get('pred_labels', []))
        entropies = _to_list(object_analysis.get('depth_entropy', []))
        num_visible_cams = _to_list(object_analysis.get('num_visible_cams', []))
        visible_cam_ids = _to_list(object_analysis.get('visible_cam_ids', []))
        gt_boxes = _to_list(object_analysis.get('gt_boxes', []))

        for pred_idx, box in enumerate(boxes):
            class_name = str(names[pred_idx]) if pred_idx < len(names) else ''
            entropy = entropies[pred_idx] if pred_idx < len(entropies) else None
            entropy = None if entropy is None else float(entropy)
            gt_dist = _nearest_same_class_gt(box, class_name, gt_boxes)
            rows.append({
                'sample_idx': sample_idx,
                'frame_id': anno.get('frame_id', ''),
                'pred_idx': pred_idx,
                'class_name': class_name,
                'pred_label': labels[pred_idx] if pred_idx < len(labels) else None,
                'score': scores[pred_idx] if pred_idx < len(scores) else None,
                'depth_entropy': entropy,
                'has_depth_entropy': entropy is not None and math.isfinite(entropy),
                'num_visible_cams': int(num_visible_cams[pred_idx]) if pred_idx < len(num_visible_cams) else 0,
                'visible_cam_ids': visible_cam_ids[pred_idx] if pred_idx < len(visible_cam_ids) else [],
                'nearest_same_class_gt_dist': gt_dist,
                'tp_proxy_center_0_5m': gt_dist is not None and gt_dist <= 0.5,
                'tp_proxy_center_1m': gt_dist is not None and gt_dist <= 1.0,
                'tp_proxy_center_2m': gt_dist is not None and gt_dist <= 2.0,
                'tp_proxy_center_4m': gt_dist is not None and gt_dist <= 4.0,
            })

    summary = _summarize_object_rows(rows)
    summary['num_samples'] = len(annos)
    summary['num_samples_missing_object_analysis'] = missing_analysis
    return summary, rows


def _write_object_csv(rows, csv_path):
    import csv

    fieldnames = [
        'sample_idx', 'frame_id', 'pred_idx', 'class_name', 'pred_label', 'score',
        'depth_entropy', 'has_depth_entropy', 'num_visible_cams', 'visible_cam_ids',
        'nearest_same_class_gt_dist', 'tp_proxy_center_0_5m', 'tp_proxy_center_1m',
        'tp_proxy_center_2m', 'tp_proxy_center_4m'
    ]
    with Path(csv_path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row['visible_cam_ids'] = '|'.join(str(v) for v in row['visible_cam_ids'])
            writer.writerow(out_row)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze exported BEVFusion LSS depth entropy statistics from a trusted local eval result.pkl'
    )
    parser.add_argument('--trusted_result_pkl', required=True,
                        help='Trusted local OpenPCDet eval result.pkl; pickle files from third parties are unsafe')
    parser.add_argument('--out_json', default=None, help='Path to save summary JSON')
    parser.add_argument('--object_out_json', default=None, help='Path to save object-level depth entropy summary JSON')
    parser.add_argument('--object_out_csv', default=None, help='Path to save object-level per-prediction CSV')
    args = parser.parse_args()

    result_pkl = Path(args.trusted_result_pkl)
    out_json = Path(args.out_json) if args.out_json is not None else result_pkl.parent / 'depth_entropy_summary.json'
    summary = analyze(result_pkl)

    with out_json.open('w') as f:
        json.dump(summary, f, indent=2)

    print('Saved summary to %s' % out_json)
    print(json.dumps(summary, indent=2))

    object_summary, object_rows = analyze_objects(result_pkl)
    if object_summary['num_samples_missing_object_analysis'] < object_summary['num_samples']:
        object_out_json = Path(args.object_out_json) if args.object_out_json is not None else result_pkl.parent / 'depth_entropy_object_summary.json'
        object_out_csv = Path(args.object_out_csv) if args.object_out_csv is not None else result_pkl.parent / 'depth_entropy_object_per_prediction.csv'
        with object_out_json.open('w') as f:
            json.dump(object_summary, f, indent=2)
        _write_object_csv(object_rows, object_out_csv)
        print('Saved object summary to %s' % object_out_json)
        print('Saved object rows to %s' % object_out_csv)


if __name__ == '__main__':
    main()
