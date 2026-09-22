import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

from tools.figure5_utils.final_selection import FINAL_SAMPLE_TOKENS


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'tools/generate_figure5_final.py'


def _boxes(center_x):
    return np.array([[center_x, 0.0, 0.0, 2.0, 1.0, 1.7, 0.0]], dtype=float)


def _prediction(token, center_x, class_name='pedestrian'):
    if center_x is None:
        return dict(
            metadata=dict(token=token), name=np.array([]), score=np.array([]),
            boxes_lidar=np.empty((0, 9)),
        )
    return dict(
        metadata=dict(token=token), name=np.array([class_name]), score=np.array([0.9]),
        boxes_lidar=_boxes(center_x),
    )


def _write_pickle(path, rows):
    with path.open('wb') as stream:
        pickle.dump(rows, stream)
    return path


def test_fixed_token_cli_exports_only_declared_offline_artifacts(tmp_path: Path):
    # Given native infos/results with one evidence-backed purpose per fixed row.
    tokens = tuple(str(token) for token in FINAL_SAMPLE_TOKENS)
    classes = ('pedestrian', 'pedestrian', 'car')
    gt_centers = (35.0, 10.0, 0.0)
    infos = [dict(
        token=token, gt_names=np.array([class_name]), gt_boxes=_boxes(center),
        lidar_path='unused.bin', sweeps=[],
    ) for token, class_name, center in zip(tokens, classes, gt_centers)]
    source = [
        _prediction(tokens[0], None),
        _prediction(tokens[1], None),
        _prediction(tokens[2], 1.0, 'car'),
    ]
    codemerge = [
        _prediction(tokens[0], None),
        _prediction(tokens[1], None),
        _prediction(tokens[2], 1.2, 'car'),
    ]
    refuse = [
        _prediction(tokens[0], 35.0),
        _prediction(tokens[1], 10.0),
        _prediction(tokens[2], 0.0, 'car'),
    ]
    paths = tuple(_write_pickle(tmp_path / name, rows) for name, rows in (
        ('infos.pkl', infos), ('source.pkl', source),
        ('codemerge.pkl', codemerge), ('refuse.pkl', refuse),
    ))
    for token, center in zip(tokens, gt_centers):
        np.save(tmp_path / (token + '.npy'), np.array([[center, 0, 0, 1, 0]], dtype=float))
    output = tmp_path / 'final'
    command = [
        sys.executable, str(CLI), '--infos', str(paths[0]), '--data-root', str(tmp_path),
        '--source-result', str(paths[1]), '--codemerge-result', str(paths[2]),
        '--refuse-result', str(paths[3]), '--output-dir', str(output),
        '--sparse-points-dir', str(tmp_path),
    ]

    # When the server CLI runs through strict loading, matching, selection, and rendering.
    result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)

    # Then exact final names exist while runtime-only intermediate maps do not.
    assert result.returncode == 0, result.stderr
    expected = {
        'figure5_final_clean.png', 'figure5_final_clean.pdf',
        'figure5_final_callout.png', 'figure5_final_callout.pdf',
        'figure5_horizontal_clean.png', 'figure5_horizontal_callout.png',
        'figure5_horizontal_clean.pdf', 'figure5_horizontal_callout.pdf',
        'figure5_row1.png', 'figure5_row2.png', 'figure5_row3.png',
        'figure5_refine_summary.md', 'figure6_status_manifest.json',
    }
    expected.update('density_%s.png' % token for token in tokens)
    expected.update('finaldet_%s.png' % token for token in tokens)
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads((output / 'figure6_status_manifest.json').read_text())
    assert manifest['point_provenance']['mode'] == 'supplied_sparse_npy_unverified'
    assert manifest['point_provenance']['exact_dataloader_replay'] is False
    assert all(not tuple(output.glob('%s_*.png' % stem))
               for stem in ('reliability', 'rgplm', 'sgdfa'))
    summary = (output / 'figure5_refine_summary.md').read_text()
    assert 'callout versions' in summary
    assert 'far-range recovery' in summary
    assert 'small-object recovery' in summary
    assert 'localization / false-positive improvement' in summary
    assert 'distance_m=' in summary
    assert 'roi=' in summary
    assert summary.count('horizontal_crop=') == 3
    for token in tokens:
        assert 'density_%s.png' % token in summary
        assert 'reliability_%s.png' % token in summary
        assert 'rgplm_%s.png' % token in summary
        assert 'sgdfa_%s.png' % token in summary
        assert 'finaldet_%s.png' % token in summary
