"""CLI contract tests use native-format synthetic inputs, never research data."""
import csv
import json
from pathlib import Path
import pickle
import subprocess
import sys
from typing import List

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'tools/generate_figure5_candidates.py'


@pytest.fixture
def command(tmp_path: Path) -> List[str]:
    infos = []
    runs = [[], [], []]
    for token in ('b', 'a'):
        boxes = np.array([[35, 0, 0, 1, 1, 2, 0]], dtype=float)
        infos.append(dict(token=token, gt_names=np.array(['pedestrian']),
                          gt_boxes=boxes, lidar_path='key.bin', sweeps=[]))
        for index, score in enumerate((0.09, 0.1, 0.8)):
            runs[index].append(dict(metadata=dict(token=token), boxes_lidar=boxes,
                                    name=np.array(['pedestrian']), score=np.array([score])))
        np.save(tmp_path / (token + '.npy'), np.array([[35, 0, 0, 1, 0]], dtype=float))
    paths = [tmp_path / name for name in ('infos.pkl', 'source.pkl', 'merge.pkl', 'refuse.pkl')]
    for path, rows in zip(paths, [infos, runs[0], list(reversed(runs[1])), runs[2]]):
        with path.open('wb') as stream:
            pickle.dump(rows, stream)
    return [sys.executable, str(CLI), '--infos', str(paths[0]), '--data-root', str(tmp_path),
            '--source-result', str(paths[1]), '--codemerge-result', str(paths[2]),
            '--refuse-result', str(paths[3]), '--output-dir', str(tmp_path / 'out'),
            '--sparse-points-dir', str(tmp_path)]


def test_cli_emits_exact_package_when_native_results_are_supplied(command: List[str], tmp_path: Path) -> None:
    """Given reordered native runs, when CLI runs, then package and filtered evidence agree."""
    result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    output = tmp_path / 'out'
    expected = {'figure5_candidates_top100.csv', 'figure5_candidates_by_type.csv',
                'figure5_callouts.json', 'figure5_candidate_summary.md', 'contact_sheet_page1.png'}
    for rank, token in enumerate(('a', 'b'), 1):
        stem = '%d_%s' % (rank, token)
        expected.update(('qual_panels_top60/' + stem + '.png',
                         'qual_panels_top30_hd/' + stem + '.png',
                         'qual_panels_top30_hd/' + stem + '_callout.png'))
    assert {str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()} == expected
    with (output / 'figure5_candidates_top100.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row['sample_token'] for row in rows] == ['a', 'b']
    assert [(row['num_gt_objects'], row['num_source_preds'], row['num_codemerge_preds'],
             row['num_refuse_preds'], row['recovered_from_source'], row['recovered_from_codemerge'])
            for row in rows] == [('1', '0', '1', '1', '1', '0')] * 2
    assert rows[0]['primary_distances'] == '35.0'
    with (output / 'figure5_candidates_by_type.csv').open() as stream:
        bucket_rows = list(csv.DictReader(stream))
    assert {(row['theme'], row['type_rank']) for row in bucket_rows} == {
        (theme, rank) for theme in ('far_range_recovery', 'small_object_recovery') for rank in ('1', '2')}
    callouts = json.loads((output / 'figure5_callouts.json').read_text())
    assert set(callouts) == {'a', 'b'}
    assert callouts['a'][0] == dict(type='far_range_recovery', **{
        'class': 'pedestrian', 'distance': 35.0, 'roi': [32.5, -2.5, 37.5, 2.5]})


def test_cli_refuses_existing_output_when_overwrite_is_absent(command: List[str], tmp_path: Path) -> None:
    """Given occupied output, when CLI runs, then existing bytes survive unchanged."""
    output = tmp_path / 'out'
    output.mkdir()
    sentinel = output / 'figure5_candidates_top100.csv'
    sentinel.write_text('keep me')
    result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0
    assert '--overwrite' in result.stderr
    assert sentinel.read_text() == 'keep me'


def test_cli_limit_and_overwrite_when_previous_package_exists(command: List[str], tmp_path: Path) -> None:
    """Given stale package images, when explicit overwrite limits input, then stale images disappear."""
    output = tmp_path / 'out'
    (output / 'qual_panels_top60').mkdir(parents=True)
    (output / 'qual_panels_top60/2_old.png').write_bytes(b'stale')
    (output / 'contact_sheet_page9.png').write_bytes(b'stale')
    sentinel = output / 'contact_sheet_page1_notes.png'
    sentinel.write_bytes(b'preserve')
    result = subprocess.run(command + ['--limit', '1', '--overwrite'], capture_output=True,
                            text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in (output / 'qual_panels_top60').iterdir()) == ['1_a.png']
    assert sorted(path.name for path in output.glob('contact_sheet_page*.png')) == [
        'contact_sheet_page1.png', 'contact_sheet_page1_notes.png']
    assert sentinel.read_bytes() == b'preserve'


def test_renderer_uses_corner_roi_when_callouts_come_from_candidates() -> None:
    """Given min/max coordinates, when rendered, then the rectangle encloses the intended ROI."""
    from tools.figure5_utils.domain import Callout, CalloutKind, DistanceMeters, FrameRecord, SampleToken
    from tools.figure5_utils.palette import DetectionClass
    from tools.figure5_utils.rendering import PanelSpec, render_comparison_panel
    import matplotlib.pyplot as plt

    callout = Callout((32.5, -2.5, 37.5, 2.5), DetectionClass.PEDESTRIAN,
                      DistanceMeters(35), CalloutKind.FAR_RANGE_RECOVERY)
    frame = FrameRecord(SampleToken('a'), (), (), (), ())
    figure = render_comparison_panel(PanelSpec(frame, np.empty((0, 5)), 1, (callout,)), True)
    try:
        patch = figure.axes[0].patches[0]
        bounds = patch.get_path().get_extents(patch.get_patch_transform()).bounds
        assert tuple(bounds) == (-2.5, 32.5, 5, 5)
    finally:
        plt.close(figure)


def test_bucket_exports_retain_twenty_when_theme_is_below_overall_top100(tmp_path: Path) -> None:
    """Given rare themes below rank 100, when exported, then each retains twenty independently."""
    from tools.figure5_utils.artifacts import export_tables
    from tools.figure5_utils.candidates import CandidateEvaluation
    from tools.figure5_utils.domain import CalloutKind, CandidateEvidence, FrameRecord, SampleToken

    candidates = tuple(CandidateEvaluation(
        FrameRecord(SampleToken('token-%03d' % index), (), (), (), ()), CandidateEvidence(),
        float(200 - index), tuple(CalloutKind) if index >= 100 else (), ()) for index in range(125))
    export_tables(candidates, tmp_path)
    with (tmp_path / 'figure5_candidates_top100.csv').open() as stream:
        overall = list(csv.DictReader(stream))
    with (tmp_path / 'figure5_candidates_by_type.csv').open() as stream:
        by_type = list(csv.DictReader(stream))
    assert len(overall) == 100
    for theme in CalloutKind:
        selected = [row for row in by_type if row['theme'] == theme.value]
        assert [int(row['type_rank']) for row in selected] == list(range(1, 21))
        assert [int(row['rank']) for row in selected] == list(range(101, 121))
    assert len(json.loads((tmp_path / 'figure5_callouts.json').read_text())) == 120


def test_cli_reconstructs_points_when_saved_arrays_are_not_requested(command: List[str], tmp_path: Path) -> None:
    """Given five-sweep metadata, when smoke runs without NPY inputs, then reconstruction renders."""
    with (tmp_path / 'infos.pkl').open('rb') as stream:
        infos = pickle.load(stream)
    for info in infos:
        info['sweeps'] = [dict(lidar_path='key.bin', transform_matrix=None, time_lag=0.1)] * 4
    with (tmp_path / 'infos.pkl').open('wb') as stream:
        pickle.dump(infos, stream)
    np.array([[35, 0, 0, 1, 0]], dtype=np.float32).tofile(tmp_path / 'key.bin')
    result = subprocess.run(command[:-2] + ['--limit', '1'], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'out/qual_panels_top60/1_a.png').is_file()
