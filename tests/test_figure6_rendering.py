import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pcdet.utils.figure6_artifacts import write_record
from pcdet.utils.figure6_schema import (
    CaptureRecord, FIXED_TOKENS, Occurrence, StageCapture, StageState, StageStatus,
)
from tools.figure6_utils.contracts import load_crop_manifest
from tools.figure6_utils.export import artifact_names, export_figure6
from tools.figure6_utils.loading import load_evidence
from tools.figure6_utils.rendering import render_plate


def _complete(owner, **arrays):
    return StageCapture(StageStatus(StageState.COMPLETE, owner), arrays)


def _empty(owner, name, shape):
    return StageCapture(StageStatus(StageState.OBSERVED_EMPTY, owner), {
        name: np.empty(shape, dtype=np.float32),
    })


def _record(token, iteration=0, reliability_state=StageState.COMPLETE):
    center_x = 12.0 + 10.0 * FIXED_TOKENS.index(token)
    boxes = np.array([[center_x, -2.0, 0.5, 4.0, 1.8, 1.5, 0.1]], dtype=np.float32)
    prediction = {
        'pred_boxes': boxes,
        'pred_scores': np.array([0.75], dtype=np.float32),
        'pred_labels': np.array([1], dtype=np.int64),
    }
    reliability = {
        StageState.COMPLETE: _complete(
            'aggregated_pseudo_source', **prediction,
            spcra_reliability=np.array([0.6], dtype=np.float32),
        ),
        StageState.OBSERVED_EMPTY: _empty(
            'aggregated_pseudo_source', 'pred_boxes', (0, 7),
        ),
        StageState.FAILED: StageCapture(
            StageStatus(StageState.FAILED, 'aggregated_pseudo_source', 'nonfinite'), {},
        ),
        StageState.MISSING: StageCapture(
            StageStatus(StageState.MISSING, 'aggregated_pseudo_source', 'not observed'), {},
        ),
    }[reliability_state]
    pseudo = np.array([[center_x, -2.0, 0.5, 4.0, 1.8, 1.5, 0.1, 1.0, 0.72]], dtype=np.float32)
    identity = Occurrence(token, 'frame-' + token[:4], 0, iteration, iteration, 0, 1, 0, 1)
    return CaptureRecord(identity, {
        'method': 'mos', 'final_pseudo_source': 'aggregated_pseudo_source',
        'point_cloud_range': '[0.0, -20.0, -5.0, 40.0, 20.0, 3.0]',
    }, {
        'input': _complete('current_pre_update', points=np.array([
            [0.0, center_x - 1.0, -2.0, 0.0, 0.5],
            [0.0, center_x, -1.5, 0.0, 0.7],
        ], dtype=np.float32)),
        'spcra.aggregated_pseudo_source': reliability,
        'effective_pseudo.aggregated_pseudo_source': _complete(
            'aggregated_pseudo_source', gt_boxes=pseudo,
        ),
        'sg_dfa': _complete(
            'current_pre_update', delta=np.arange(16, dtype=np.float32).reshape(1, 4, 4),
        ),
        'final_detection': _complete('current_pre_update', **prediction),
    })


def _manifest(path):
    crops = (
        (10.1234567890123, -8.9876543210987, 22.1111111111111, 9.2222222222222),
        (20.1234567890123, -7.9876543210987, 32.1111111111111, 10.2222222222222),
        (30.1234567890123, -6.9876543210987, 42.1111111111111, 11.2222222222222),
    )
    payload = {
        'schema_version': 1,
        'coordinate_conventions': {
            'canonical_frame': 'lidar', 'units': 'm', 'horizontal_axis': 'y',
            'vertical_axis': 'x',
            'crop_tuple_order': ['x_min', 'y_min', 'x_max', 'y_max'],
            'roi_tuple_order': ['x_min', 'y_min', 'x_max', 'y_max'],
        },
        'layouts': {'square': 'figure5_final', 'horizontal': 'figure5_horizontal'},
        'rows': [
            {'row_number': index + 1, 'sample_token': token, 'purpose': 'row-%d' % (index + 1),
             'square_crop': list(crop), 'horizontal_crop': list(crop), 'callouts': []}
            for index, (token, crop) in enumerate(zip(FIXED_TOKENS, crops))
        ],
    }
    path.write_text(json.dumps(payload), encoding='utf-8')
    return crops


def test_crop_manifest_when_loaded_preserves_full_precision_and_order(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    expected = _manifest(manifest)

    rows = load_crop_manifest(manifest)

    assert tuple(row.token for row in rows) == FIXED_TOKENS
    assert tuple(row.crop for row in rows) == expected


def test_occurrences_when_loaded_are_deterministic_and_disclose_source(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    _manifest(manifest)
    capture = tmp_path / 'capture'
    write_record(capture, _record(FIXED_TOKENS[0], iteration=9))
    earliest = write_record(capture, _record(FIXED_TOKENS[0], iteration=2))

    evidence = load_evidence(capture, load_crop_manifest(manifest))

    row = evidence.rows[0]
    assert row.occurrence_path == earliest
    assert row.alternative_count == 1
    assert row.stages[1].selected_source == 'aggregated_pseudo_source'
    assert row.stages[2].selected_source == 'aggregated_pseudo_source'
    assert tuple(item.token for item in evidence.rows) == FIXED_TOKENS


@pytest.mark.parametrize('state', [
    StageState.MISSING, StageState.FAILED, StageState.OBSERVED_EMPTY,
])
def test_runtime_state_when_rendered_remains_honest(tmp_path, state):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    _manifest(manifest)
    capture = tmp_path / 'capture'
    write_record(capture, _record(FIXED_TOKENS[0], reliability_state=state))

    evidence = load_evidence(capture, load_crop_manifest(manifest))

    assert evidence.rows[0].stages[1].runtime_state is state
    arrays = evidence.rows[0].stages[1].arrays
    assert bool(arrays) is (state is StageState.OBSERVED_EMPTY)
    assert all(values.size == 0 for values in arrays.values())
    figure = render_plate((evidence.rows[0],))
    rendered = {text.get_text() for text in figure.axes[1].texts}
    plt.close(figure)
    assert {
        StageState.MISSING: 'MISSING\nnot observed',
        StageState.FAILED: 'FAILED\nnonfinite',
        StageState.OBSERVED_EMPTY: 'OBSERVED EMPTY',
    }[state] in rendered


def test_sgdfa_when_rendered_uses_calibrated_bev_crop_and_axis_order(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    _manifest(manifest)
    capture = tmp_path / 'capture'
    write_record(capture, _record(FIXED_TOKENS[0]))

    evidence = load_evidence(capture, load_crop_manifest(manifest))
    figure = render_plate((evidence.rows[0],))

    image = figure.axes[3].images[0]
    np.testing.assert_array_equal(image.get_array(), [[5.0, 9.0], [6.0, 10.0]])
    assert tuple(image.get_extent()) == (-10.0, 10.0, 10.0, 30.0)
    plt.close(figure)


def test_export_when_records_are_synthetic_publishes_exact_bundle(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    crops = _manifest(manifest)
    capture = tmp_path / 'capture'
    for token in FIXED_TOKENS:
        write_record(capture, _record(token))
    corrupt = capture / 'rank_00000' / 'records' / 'corrupt'
    corrupt.mkdir(parents=True)
    (corrupt / 'metadata.json').write_text('{', encoding='utf-8')
    output = tmp_path / 'figure6'

    export_figure6(capture, manifest, output, overwrite=False)

    assert {path.name for path in output.iterdir()} == set(artifact_names(FIXED_TOKENS))
    status = json.loads((output / 'figure6_status_manifest.json').read_text(encoding='utf-8'))
    assert [row['token'] for row in status['rows']] == list(FIXED_TOKENS)
    assert [row['crop'] for row in status['rows']] == [list(crop) for crop in crops]
    assert len(status['record_issues']) == 1
    assert status['drafts']['2row']['tokens'] == list(FIXED_TOKENS[:2])
    assert status['drafts']['3row']['tokens'] == list(FIXED_TOKENS)
    for name in ('figure6_draft_2row.png', 'figure6_draft_3row.png'):
        assert (output / name).read_bytes().startswith(b'\x89PNG')
    for name in ('figure6_draft_2row.pdf', 'figure6_draft_3row.pdf'):
        assert (output / name).read_bytes().startswith(b'%PDF')
    with pytest.raises(FileExistsError):
        export_figure6(capture, manifest, output, overwrite=False)
