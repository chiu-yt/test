import argparse
import json
import math
import pickle
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(
        description='Analyze exported BEVFusion LSS depth entropy statistics from a trusted local eval result.pkl'
    )
    parser.add_argument('--trusted_result_pkl', required=True,
                        help='Trusted local OpenPCDet eval result.pkl; pickle files from third parties are unsafe')
    parser.add_argument('--out_json', default=None, help='Path to save summary JSON')
    args = parser.parse_args()

    result_pkl = Path(args.trusted_result_pkl)
    out_json = Path(args.out_json) if args.out_json is not None else result_pkl.parent / 'depth_entropy_summary.json'
    summary = analyze(result_pkl)

    with out_json.open('w') as f:
        json.dump(summary, f, indent=2)

    print('Saved summary to %s' % out_json)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
