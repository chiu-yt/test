import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def _empty_branch():
    return {
        'name': np.array([]),
        'score': np.array([], dtype=np.float32),
        'boxes_lidar': np.zeros((0, 9), dtype=np.float32),
        'pred_labels': np.array([], dtype=np.int64),
    }


def _as_array(value):
    if value is None:
        return np.asarray([])
    return np.asarray(value)


def _class_name_from_gt_label(label):
    label = int(round(float(label)))
    if label <= 0 or label > len(CLASS_NAMES):
        return None
    return CLASS_NAMES[label - 1]


def _center_distance(boxes_a, boxes_b):
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    diff = boxes_a[:, None, :2] - boxes_b[None, :, :2]
    return np.linalg.norm(diff, axis=-1)


def _nearest_same_class(query_box, query_name, branch, max_dist=None):
    names = _as_array(branch.get('name', []))
    boxes = _as_array(branch.get('boxes_lidar', np.zeros((0, 9))))
    scores = _as_array(branch.get('score', np.zeros((0,), dtype=np.float32)))
    if boxes.shape[0] == 0:
        return None

    mask = names == query_name
    if not np.any(mask):
        return None

    cand_boxes = boxes[mask]
    cand_scores = scores[mask]
    dists = _center_distance(query_box[None, :], cand_boxes)[0]
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    if max_dist is not None and best_dist > max_dist:
        return None
    return {
        'box': cand_boxes[best_idx],
        'score': float(cand_scores[best_idx]) if cand_scores.shape[0] > 0 else 0.0,
        'dist': best_dist,
    }


def _nearest_gt(query_box, query_name, gt_boxes):
    if gt_boxes is None or gt_boxes.shape[0] == 0:
        return None

    names = []
    valid_boxes = []
    for box in gt_boxes:
        if not np.isfinite(box).all() or box.shape[0] < 8:
            continue
        name = _class_name_from_gt_label(box[-1])
        if name is None:
            continue
        names.append(name)
        valid_boxes.append(box)

    if len(valid_boxes) == 0:
        return None

    valid_boxes = np.asarray(valid_boxes)
    names = np.asarray(names)
    mask = names == query_name
    if not np.any(mask):
        return None
    cand_boxes = valid_boxes[mask]
    dists = _center_distance(query_box[None, :], cand_boxes)[0]
    best_idx = int(np.argmin(dists))
    return {
        'box': cand_boxes[best_idx],
        'dist': float(dists[best_idx]),
    }


def _safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.shape[0] < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _summarize_rows(rows):
    summary = {
        'num_fused_predictions': len(rows),
        'num_with_both_modalities': int(sum(r['has_lidar_match'] and r['has_camera_match'] for r in rows)),
        'num_with_gt_match': int(sum(np.isfinite(r['fused_gt_center_dist']) for r in rows)),
        'classes': {},
    }

    for class_name in CLASS_NAMES:
        cls_rows = [r for r in rows if r['class_name'] == class_name]
        if len(cls_rows) == 0:
            continue
        conflicts = np.asarray([r['lc_center_dist'] for r in cls_rows], dtype=np.float64)
        gt_dists = np.asarray([r['fused_gt_center_dist'] for r in cls_rows], dtype=np.float64)
        valid_conflict = np.isfinite(conflicts)
        valid_gt = np.isfinite(gt_dists)
        both = valid_conflict & valid_gt

        cls_summary = {
            'num_fused_predictions': len(cls_rows),
            'num_with_both_modalities': int(valid_conflict.sum()),
            'num_with_gt_match': int(valid_gt.sum()),
            'conflict_gt_error_corr': _safe_corr(conflicts[both], gt_dists[both]),
        }
        if valid_conflict.any():
            cls_summary.update({
                'lc_center_dist_mean': float(np.mean(conflicts[valid_conflict])),
                'lc_center_dist_p50': float(np.percentile(conflicts[valid_conflict], 50)),
                'lc_center_dist_p75': float(np.percentile(conflicts[valid_conflict], 75)),
                'lc_center_dist_p90': float(np.percentile(conflicts[valid_conflict], 90)),
            })
        if valid_gt.any():
            cls_summary.update({
                'fused_gt_center_dist_mean': float(np.mean(gt_dists[valid_gt])),
                'fused_recall_center_0_5m': float(np.mean(gt_dists[valid_gt] <= 0.5)),
                'fused_recall_center_1m': float(np.mean(gt_dists[valid_gt] <= 1.0)),
                'fused_recall_center_2m': float(np.mean(gt_dists[valid_gt] <= 2.0)),
                'fused_recall_center_4m': float(np.mean(gt_dists[valid_gt] <= 4.0)),
            })
        if both.sum() >= 4:
            valid_rows = [r for r in cls_rows if np.isfinite(r['lc_center_dist']) and np.isfinite(r['fused_gt_center_dist'])]
            q25 = float(np.percentile([r['lc_center_dist'] for r in valid_rows], 25))
            q75 = float(np.percentile([r['lc_center_dist'] for r in valid_rows], 75))
            low = [r for r in valid_rows if r['lc_center_dist'] <= q25]
            high = [r for r in valid_rows if r['lc_center_dist'] >= q75]
            cls_summary.update({
                'low_conflict_mean_gt_dist': float(np.mean([r['fused_gt_center_dist'] for r in low])) if low else None,
                'high_conflict_mean_gt_dist': float(np.mean([r['fused_gt_center_dist'] for r in high])) if high else None,
                'low_conflict_recall_2m': float(np.mean([r['fused_gt_center_dist'] <= 2.0 for r in low])) if low else None,
                'high_conflict_recall_2m': float(np.mean([r['fused_gt_center_dist'] <= 2.0 for r in high])) if high else None,
            })
        summary['classes'][class_name] = cls_summary

    conflicts = np.asarray([r['lc_center_dist'] for r in rows], dtype=np.float64)
    gt_dists = np.asarray([r['fused_gt_center_dist'] for r in rows], dtype=np.float64)
    both = np.isfinite(conflicts) & np.isfinite(gt_dists)
    summary['overall_conflict_gt_error_corr'] = _safe_corr(conflicts[both], gt_dists[both])
    return summary


def analyze(result_pkl, max_branch_match_dist):
    with Path(result_pkl).open('rb') as f:
        annos = pickle.load(f)

    rows = []
    missing_analysis = 0
    for sample_idx, anno in enumerate(annos):
        analysis = anno.get('forward_only_conflict_analysis')
        if analysis is None:
            missing_analysis += 1
            continue

        fused = analysis.get('B_Fused', anno)
        lidar = analysis.get('B_L', _empty_branch())
        camera = analysis.get('B_C', _empty_branch())
        gt_boxes = _as_array(analysis.get('gt_boxes', np.zeros((0, 8), dtype=np.float32)))

        fused_names = _as_array(fused.get('name', []))
        fused_scores = _as_array(fused.get('score', []))
        fused_boxes = _as_array(fused.get('boxes_lidar', np.zeros((0, 9), dtype=np.float32)))

        for pred_idx in range(fused_boxes.shape[0]):
            class_name = str(fused_names[pred_idx])
            fused_box = fused_boxes[pred_idx]
            lidar_match = _nearest_same_class(fused_box, class_name, lidar, max_dist=max_branch_match_dist)
            camera_match = _nearest_same_class(fused_box, class_name, camera, max_dist=max_branch_match_dist)
            gt_match = _nearest_gt(fused_box, class_name, gt_boxes)

            lc_center_dist = np.nan
            lidar_fused_dist = np.nan
            camera_fused_dist = np.nan
            score_gap = np.nan
            if lidar_match is not None:
                lidar_fused_dist = lidar_match['dist']
            if camera_match is not None:
                camera_fused_dist = camera_match['dist']
            if lidar_match is not None and camera_match is not None:
                lc_center_dist = float(np.linalg.norm(lidar_match['box'][:2] - camera_match['box'][:2]))
                score_gap = abs(lidar_match['score'] - camera_match['score'])

            rows.append({
                'sample_idx': sample_idx,
                'frame_id': anno.get('frame_id', ''),
                'pred_idx': pred_idx,
                'class_name': class_name,
                'fused_score': float(fused_scores[pred_idx]) if fused_scores.shape[0] > pred_idx else np.nan,
                'has_lidar_match': lidar_match is not None,
                'has_camera_match': camera_match is not None,
                'lidar_fused_center_dist': lidar_fused_dist,
                'camera_fused_center_dist': camera_fused_dist,
                'lc_center_dist': lc_center_dist,
                'lc_score_gap': score_gap,
                'fused_gt_center_dist': gt_match['dist'] if gt_match is not None else np.nan,
            })

    summary = _summarize_rows(rows)
    summary['num_samples'] = len(annos)
    summary['num_samples_missing_analysis'] = missing_analysis
    summary['max_branch_match_dist'] = max_branch_match_dist
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description='Analyze BEVFusion forward-only modality conflict probe result.pkl')
    parser.add_argument('--result_pkl', required=True, help='Path to eval result.pkl with forward_only_conflict_analysis')
    parser.add_argument('--out_dir', default=None, help='Directory to save summary JSON and per-prediction CSV')
    parser.add_argument('--max_branch_match_dist', type=float, default=4.0,
                        help='Max center distance in meters to associate B_L/B_C predictions to a fused prediction')
    args = parser.parse_args()

    result_pkl = Path(args.result_pkl)
    out_dir = Path(args.out_dir) if args.out_dir is not None else result_pkl.parent / 'conflict_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, rows = analyze(result_pkl, args.max_branch_match_dist)

    summary_path = out_dir / 'summary.json'
    rows_path = out_dir / 'per_prediction.csv'
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2)

    fieldnames = [
        'sample_idx', 'frame_id', 'pred_idx', 'class_name', 'fused_score',
        'has_lidar_match', 'has_camera_match', 'lidar_fused_center_dist',
        'camera_fused_center_dist', 'lc_center_dist', 'lc_score_gap', 'fused_gt_center_dist'
    ]
    with rows_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print('Saved summary to %s' % summary_path)
    print('Saved per-prediction rows to %s' % rows_path)
    print(json.dumps({
        'num_samples': summary['num_samples'],
        'num_fused_predictions': summary['num_fused_predictions'],
        'num_with_both_modalities': summary['num_with_both_modalities'],
        'overall_conflict_gt_error_corr': summary['overall_conflict_gt_error_corr'],
    }, indent=2))


if __name__ == '__main__':
    main()
