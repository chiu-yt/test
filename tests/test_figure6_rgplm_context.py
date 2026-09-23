from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pytest

from pcdet.utils.figure6_artifacts import write_record
from pcdet.utils.figure6_schema import (
    CaptureRecord, FIXED_TOKENS, StageCapture, StageState, StageStatus,
)
from test_figure6_rendering import _complete, _manifest, _record
from tools.figure6_utils.contracts import load_crop_manifest
from tools.figure6_utils.loading import load_evidence
from tools.figure6_utils.rendering import RenderDataError, draw_stage, row_sgdfa_limit


CANDIDATE_COLOR = '#B0B0B0'


def _load_row(tmp_path, record=None):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    _manifest(manifest)
    capture = tmp_path / 'capture'
    write_record(capture, _record(FIXED_TOKENS[0]) if record is None else record)
    return load_evidence(capture, load_crop_manifest(manifest)).rows[0]


def _render_stage(row, stage_index):
    figure, axis = plt.subplots()
    try:
        draw_stage(axis, row, row.stages[stage_index], False, row_sgdfa_limit(row))
        return tuple(axis.lines)
    finally:
        plt.close(figure)


def _record_with_candidate(candidate):
    source = _record(FIXED_TOKENS[0])
    stages = dict(source.stages)
    stages['spcra.aggregated_pseudo_source'] = candidate
    return CaptureRecord(source.identity, source.protocol, stages)


def test_rgplm_draws_candidate_context_behind_retained_foreground(tmp_path):
    candidate_boxes = np.array([
        [12.0, -2.0, 0.5, 4.0, 2.0, 1.5, 0.0],
        [15.0, 1.0, 0.5, 2.0, 1.0, 1.5, 0.0],
        [18.0, 4.0, 0.5, 3.0, 1.0, 1.5, 0.0],
    ], dtype=np.float32)
    candidate = _complete(
        'aggregated_pseudo_source', pred_boxes=candidate_boxes,
        pred_scores=np.array([np.nan, -1.0, 99.0], dtype=np.float32),
        pred_labels=np.array([-9, 0, 99], dtype=np.int64),
        spcra_reliability=np.array([np.nan, -4.0, 8.0], dtype=np.float32),
    )
    source = _record_with_candidate(candidate)
    stages = dict(source.stages)
    stages['effective_pseudo.aggregated_pseudo_source'] = _complete(
        'aggregated_pseudo_source',
        gt_boxes=np.array([
            [16.0, 5.0, 0.5, 4.0, 2.0, 1.5, 0.0, 1.0, 0.72],
        ], dtype=np.float32),
    )
    row = _load_row(tmp_path, CaptureRecord(source.identity, source.protocol, stages))
    lines = _render_stage(row, 2)

    assert len(lines) == 4
    candidate_lines, retained = lines[:3], lines[3]
    assert tuple(to_hex(line.get_color()).upper() for line in candidate_lines) == (
        CANDIDATE_COLOR, CANDIDATE_COLOR, CANDIDATE_COLOR,
    )
    np.testing.assert_allclose(
        np.asarray(candidate_lines[0].get_xydata()),
        [[-1.0, 14.0], [-3.0, 14.0], [-3.0, 10.0], [-1.0, 10.0], [-1.0, 14.0]],
    )
    assert all(line.get_linewidth() == 0.5 for line in candidate_lines)
    assert all(line.get_zorder() == 2 for line in candidate_lines)
    assert to_hex(retained.get_color()).upper() == '#FF9E00'
    assert retained.get_linewidth() == 1.6
    assert retained.get_zorder() == 3
    np.testing.assert_allclose(
        np.asarray(retained.get_xydata()),
        [[6.0, 18.0], [4.0, 18.0], [4.0, 14.0], [6.0, 14.0], [6.0, 18.0]],
    )


def test_rgplm_candidate_geometry_ignores_auxiliary_values_and_count(tmp_path):
    candidate_boxes = np.array([
        [11.0, -3.0, 0.5, 2.0, 1.0, 1.0, 0.0],
        [17.0, 4.0, 0.5, 4.0, 2.0, 1.0, 0.0],
    ], dtype=np.float32)
    candidate = _complete(
        'aggregated_pseudo_source', pred_boxes=candidate_boxes,
        pred_scores=np.array([np.nan], dtype=np.float32),
        pred_labels=np.array([], dtype=np.int64),
        spcra_reliability=np.array([np.inf, -np.inf, np.nan], dtype=np.float32),
    )
    lines = _render_stage(_load_row(tmp_path, _record_with_candidate(candidate)), 2)

    context = tuple(line for line in lines if line.get_linewidth() == 0.5)
    retained = tuple(line for line in lines if line.get_linewidth() == 1.6)
    assert len(context) == 2
    assert len(retained) == 1
    np.testing.assert_allclose(
        np.asarray(context[1].get_xydata()),
        [[5.0, 19.0], [3.0, 19.0], [3.0, 15.0], [5.0, 15.0], [5.0, 19.0]],
    )


def test_rgplm_skips_only_invalid_individual_candidate_geometry(tmp_path):
    candidate = _complete(
        'aggregated_pseudo_source',
        pred_boxes=np.array([
            [11.0, -3.0, 0.5, 2.0, 1.0, 1.0, 0.0],
            [14.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.0],
            [np.nan, 2.0, 0.5, 2.0, 1.0, 1.0, 0.0],
        ], dtype=np.float32),
        pred_scores=np.ones(3, dtype=np.float32),
        pred_labels=np.ones(3, dtype=np.int64),
        spcra_reliability=np.ones(3, dtype=np.float32),
    )
    lines = _render_stage(_load_row(tmp_path, _record_with_candidate(candidate)), 2)

    assert tuple(line.get_linewidth() for line in lines) == (0.5, 1.6)


@pytest.mark.parametrize('arrays', [
    {'pred_scores': np.ones(1, dtype=np.float32)},
    {'pred_boxes': np.ones((1, 6), dtype=np.float32)},
])
def test_rgplm_rejects_malformed_pred_boxes_on_complete_candidate_stage(tmp_path, arrays):
    candidate = StageCapture(
        StageStatus(StageState.COMPLETE, 'aggregated_pseudo_source'), arrays,
    )
    row = _load_row(tmp_path, _record_with_candidate(candidate))

    with pytest.raises(RenderDataError, match='pred_boxes'):
        _render_stage(row, 2)


@pytest.mark.parametrize('state, arrays', [
    (StageState.MISSING, {}),
    (StageState.FAILED, {}),
    (StageState.OBSERVED_EMPTY, {'pred_boxes': np.empty((0, 7), dtype=np.float32)}),
])
def test_rgplm_omits_unavailable_candidate_context_but_keeps_retained(
        tmp_path, state, arrays):
    row = _load_row(tmp_path)
    candidate = replace(
        row.stages[1], runtime_state=state, arrays=arrays,
        detail='capture failed' if state is StageState.FAILED else '',
    )
    row = replace(row, stages=(row.stages[0], candidate) + row.stages[2:])

    lines = _render_stage(row, 2)

    assert len(lines) == 1
    assert lines[0].get_linewidth() == 1.6
    assert to_hex(lines[0].get_color()).upper() == '#FF9E00'


def test_rgplm_current_source_fallback_uses_resolved_current_geometry(tmp_path):
    source = _record(FIXED_TOKENS[0])
    stages = dict(source.stages)
    del stages['spcra.aggregated_pseudo_source']
    stages['spcra.current_pre_update'] = _complete(
        'current_pre_update',
        pred_boxes=np.array([[19.0, 5.0, 0.5, 2.0, 4.0, 1.0, 0.0]], dtype=np.float32),
        pred_scores=np.array([0.1], dtype=np.float32),
        pred_labels=np.array([10], dtype=np.int64),
        spcra_reliability=np.array([0.0], dtype=np.float32),
    )
    row = _load_row(tmp_path, CaptureRecord(source.identity, source.protocol, stages))

    lines = _render_stage(row, 2)

    context = tuple(line for line in lines if line.get_linewidth() == 0.5)
    assert row.stages[1].selected_stage == 'spcra.current_pre_update'
    assert len(context) == 1
    np.testing.assert_allclose(
        np.asarray(context[0].get_xydata()),
        [[7.0, 20.0], [3.0, 20.0], [3.0, 18.0], [7.0, 18.0], [7.0, 20.0]],
    )


def test_final_detection_preserves_geometry_color_order_at_published_width(tmp_path):
    row = _load_row(tmp_path)
    final = replace(row.stages[4], arrays={
        'pred_boxes': np.array([
            [12.0, -2.0, 0.5, 4.0, 2.0, 1.0, 0.0],
            [15.0, 1.0, 0.5, 2.0, 1.0, 1.0, 0.0],
            [18.0, 4.0, 0.5, 2.0, 4.0, 1.0, 0.0],
        ], dtype=np.float32),
        'pred_scores': np.array([0.9, 0.8, 0.7], dtype=np.float32),
        'pred_labels': np.array([1, 9, 2], dtype=np.int64),
    })
    row = replace(row, stages=row.stages[:4] + (final,))
    lines = _render_stage(row, 4)

    assert tuple(to_hex(line.get_color()).upper() for line in lines) == (
        '#FF9E00', '#0000E6', '#FF6347',
    )
    assert tuple(line.get_linewidth() for line in lines) == (1.4, 1.4, 1.4)
    np.testing.assert_allclose(
        np.asarray(lines[0].get_xydata()),
        [[-1.0, 14.0], [-3.0, 14.0], [-3.0, 10.0], [-1.0, 10.0], [-1.0, 14.0]],
    )
