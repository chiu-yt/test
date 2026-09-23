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
from tools.figure6_utils.loading import load_evidence
from tools.figure6_utils.rendering import RenderDataError, render_plate


def _complete(owner, **arrays):
    return StageCapture(StageStatus(StageState.COMPLETE, owner), arrays)


def _empty(owner, name, shape):
    return StageCapture(StageStatus(StageState.OBSERVED_EMPTY, owner), {
        name: np.empty(shape, dtype=np.float32),
    })


def _record(token, iteration=0, reliability_state=StageState.COMPLETE,
            reliability_values=None, pseudo_rows=None, sgdfa_scale=1.0):
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
            spcra_reliability=(
                np.array([0.6], dtype=np.float32)
                if reliability_values is None else reliability_values
            ),
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
    pseudo = (
        np.array([[center_x, -2.0, 0.5, 4.0, 1.8, 1.5, 0.1, 1.0, 0.72]], dtype=np.float32)
        if pseudo_rows is None else pseudo_rows
    )
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
            'current_pre_update',
            delta=np.arange(16, dtype=np.float32).reshape(1, 4, 4) * sgdfa_scale,
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
             'square_crop': list(crop), 'horizontal_crop': list(crop), 'callouts': [
                 {'roi': [crop[0] + 1.0, crop[1] + 1.0, crop[0] + 3.0, crop[1] + 3.0],
                  'kind': 'recovery', 'class_name': 'car', 'distance_m': 18.5},
             ]}
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
    assert rows[0].callouts[0].roi == (
        expected[0][0] + 1.0, expected[0][1] + 1.0,
        expected[0][0] + 3.0, expected[0][1] + 3.0,
    )
    assert rows[0].callouts[0].kind == 'recovery'


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
    assert row.callouts == load_crop_manifest(manifest)[0].callouts


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


def test_sgdfa_when_rendered_uses_nearest_signed_map_and_row_local_scale(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    _manifest(manifest)
    capture = tmp_path / 'capture'
    write_record(capture, _record(FIXED_TOKENS[0]))
    write_record(capture, _record(FIXED_TOKENS[1], sgdfa_scale=-10.0))

    evidence = load_evidence(capture, load_crop_manifest(manifest))
    figure = render_plate(evidence.rows[:2])

    images = (figure.axes[3].images[0], figure.axes[8].images[0])
    np.testing.assert_array_equal(images[0].get_array(), [[5.0, 9.0], [6.0, 10.0]])
    assert tuple(images[0].get_extent()) == (-10.0, 10.0, 10.0, 30.0)
    assert all(image.get_interpolation() == 'nearest' for image in images)
    expected_limits = (
        float(np.percentile(np.abs([[5.0, 9.0], [6.0, 10.0]]), 98.0)),
        float(np.percentile(
            np.abs([[-60.0, -100.0, -140.0], [-70.0, -110.0, -150.0]]), 98.0,
        )),
    )
    actual_limits = tuple(np.asarray(image.norm.vmax).item() for image in images)
    np.testing.assert_allclose(actual_limits, expected_limits)
    np.testing.assert_allclose(
        tuple(np.asarray(image.norm.vmin).item() for image in images),
        tuple(-limit for limit in expected_limits),
    )
    plt.close(figure)


def test_reliability_when_lengths_differ_rejects_instead_of_truncating(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    _manifest(manifest)
    capture = tmp_path / 'capture'
    write_record(capture, _record(
        FIXED_TOKENS[0], reliability_values=np.array([0.2, 0.8], dtype=np.float32),
    ))
    row = load_evidence(capture, load_crop_manifest(manifest)).rows[0]

    with pytest.raises(RenderDataError, match='equal lengths'):
        render_plate((row,))


def test_plate_when_rendered_has_fixed_publication_labels_and_no_debug_text(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    _manifest(manifest)
    capture = tmp_path / 'capture'
    for token in FIXED_TOKENS[:2]:
        write_record(capture, _record(token))
    rows = load_evidence(capture, load_crop_manifest(manifest)).rows[:2]

    figure = render_plate(rows)

    assert tuple(axis.get_title() for axis in figure.axes[:5]) == (
        'LiDAR Density', 'SPCRA Reliability', 'RG-PLM Retained',
        'SG-DFA Response', 'Final Detection',
    )
    figure_text = tuple(text.get_text() for text in figure.texts)
    assert figure_text == ('(a) Far-range recovery', '(b) Small-object recovery')
    panel_text = ' '.join(text.get_text() for axis in figure.axes for text in axis.texts)
    assert 'token' not in panel_text.lower()
    assert 'rank' not in panel_text.lower()
    assert 'car' not in panel_text.lower()
    assert sum(text.startswith('r=') for text in panel_text.split()) <= 4
    assert tuple(round(value, 2) for value in figure.get_size_inches()) == (7.2, 2.75)
    plt.close(figure)
