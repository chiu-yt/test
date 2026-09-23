import json
from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
import numpy as np
import pytest

from pcdet.utils.figure6_artifacts import write_record
from pcdet.utils.figure6_schema import FIXED_TOKENS
from test_figure6_rendering import _manifest, _record
from tools.figure6_utils.contracts import Callout, load_crop_manifest
from tools.figure6_utils.export import artifact_names, export_figure6
from tools.figure6_utils.loading import load_evidence
from tools.figure6_utils.rendering import render_plate, render_row, row_sgdfa_limit


def _rows(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    _manifest(manifest)
    capture = tmp_path / 'capture'
    for token in FIXED_TOKENS[:2]:
        write_record(capture, _record(token))
    return load_evidence(capture, load_crop_manifest(manifest)).rows[:2]


@pytest.mark.parametrize('alternate', [False, True])
def test_plate_when_rendered_includes_three_band_reliability_key(tmp_path, alternate):
    figure = render_plate(_rows(tmp_path), alternate=alternate)
    figure.canvas.draw()

    color_axis = figure.axes[-1]
    labels = tuple(label.get_text() for label in color_axis.get_xticklabels())
    color_mesh = color_axis.collections[-1]
    assert isinstance(color_mesh, QuadMesh)
    assert labels == ('Low', 'Mid', 'High')
    assert np.asarray(color_mesh.get_array()).size == 3
    tight_bounds = color_axis.get_tightbbox()
    assert tight_bounds is not None
    key_bounds = (color_axis.get_window_extent(), tight_bounds)
    key_bounds += tuple(label.get_window_extent() for label in color_axis.get_xticklabels())
    assert min(bounds.y0 for bounds in key_bounds) > figure.bbox.height * 0.015
    assert max(bounds.y1 for bounds in key_bounds) < figure.bbox.height
    assert len(figure.axes) == 11
    plt.close(figure)


def test_plate_when_manifest_has_three_rois_displays_only_enlarged_first_roi(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    crops = _manifest(manifest)
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    x_min, y_min, _, _ = crops[0]
    payload['rows'][0]['callouts'] = [
        {'roi': [x_min, y_min, x_min + 2.0, y_min + 2.0],
         'kind': 'first', 'class_name': 'car', 'distance_m': 10.0},
        {'roi': [x_min + 3.0, y_min + 3.0, x_min + 4.0, y_min + 4.0],
         'kind': 'second', 'class_name': 'car', 'distance_m': 20.0},
        {'roi': [x_min + 5.0, y_min + 5.0, x_min + 6.0, y_min + 6.0],
         'kind': 'third', 'class_name': 'car', 'distance_m': 30.0},
    ]
    manifest.write_text(json.dumps(payload), encoding='utf-8')
    capture = tmp_path / 'capture'
    write_record(capture, _record(FIXED_TOKENS[0]))
    row = load_evidence(capture, load_crop_manifest(manifest)).rows[0]

    figure = render_plate((row,))

    expected_geometry = (y_min, x_min, 2.2, 2.2)
    for axis in figure.axes[:5]:
        rectangles = tuple(
            patch for patch in axis.patches if isinstance(patch, Rectangle)
        )
        assert len(rectangles) == 1
        patch = rectangles[0]
        np.testing.assert_allclose(
            (patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()),
            expected_geometry,
        )
        assert patch.get_edgecolor() == to_rgba('#CC0000', alpha=0.72)
    plt.close(figure)


def test_reliability_annotations_choose_nearest_unique_proposal_per_roi(tmp_path):
    row = _rows(tmp_path)[0]
    boxes = np.array([
        [13.0, -1.0, 0.5, 2.0, 1.0, 1.0, 0.0],
        [12.0, -2.0, 0.5, 2.0, 1.0, 1.0, 0.0],
        [15.0, 3.0, 0.5, 2.0, 1.0, 1.0, 0.0],
    ], dtype=np.float32)
    reliability = replace(row.stages[1], arrays={
        'pred_boxes': boxes,
        'pred_scores': np.array([0.7, 0.8, 0.9], dtype=np.float32),
        'pred_labels': np.array([1, 1, 1], dtype=np.int64),
        'spcra_reliability': np.array([0.9, 0.1, 0.5], dtype=np.float32),
    })
    stages = row.stages[:1] + (reliability,) + row.stages[2:]
    callouts = (
        Callout((10.0, -4.0, 14.0, 0.0), 'first', 'car', 10.0),
        Callout((10.0, -3.8, 14.4, 0.2), 'overlap', 'car', 11.0),
        Callout((14.0, 2.0, 16.0, 4.0), 'third', 'car', 12.0),
    )

    rendered_row = replace(row, stages=stages, callouts=callouts)
    figures = (render_plate((rendered_row,)), render_row(rendered_row))

    for figure in figures:
        figure.canvas.draw()
        annotations = figure.axes[1].texts
        assert tuple(text.get_text() for text in annotations) == ('r=0.10',)
        roi = next(patch for patch in figure.axes[1].patches if isinstance(patch, Rectangle))
        label_x, label_y = annotations[0].get_position()
        assert roi.get_y() + roi.get_height() < label_y < row.crop[2]
        assert row.crop[1] < label_x < row.crop[3]
        assert annotations[0].get_verticalalignment() == 'bottom'
        background = annotations[0].get_bbox_patch()
        assert background is not None
        assert background.get_facecolor() == to_rgba('#FFFFFF', alpha=0.82)
        assert annotations[0].get_window_extent().y0 > roi.get_window_extent().y1

    alternate = render_plate((rendered_row,), alternate=True)
    assert not alternate.axes[1].texts
    assert len(alternate.axes[1].lines) == len(figures[0].axes[1].lines)
    assert len(alternate.axes[1].patches) == len(figures[0].axes[1].patches) == 1
    for figure in figures:
        plt.close(figure)
    plt.close(alternate)


def test_reliability_annotation_when_roi_touches_crop_top_falls_below(tmp_path):
    row = _rows(tmp_path)[0]
    crop_top = row.crop[2]
    reliability = replace(row.stages[1], arrays={
        'pred_boxes': np.array([
            [crop_top - 0.75, -2.0, 0.5, 2.0, 1.0, 1.0, 0.0],
        ], dtype=np.float32),
        'pred_scores': np.array([0.8], dtype=np.float32),
        'pred_labels': np.array([1], dtype=np.int64),
        'spcra_reliability': np.array([0.5], dtype=np.float32),
    })
    stages = row.stages[:1] + (reliability,) + row.stages[2:]
    callout = Callout((crop_top - 1.5, -4.0, crop_top, 0.0), 'edge', 'car', 10.0)

    figure = render_plate((replace(row, stages=stages, callouts=(callout,)),))
    figure.canvas.draw()

    annotation = figure.axes[1].texts[0]
    roi = next(patch for patch in figure.axes[1].patches if isinstance(patch, Rectangle))
    label_x, label_y = annotation.get_position()
    assert row.crop[0] < label_y < roi.get_y()
    assert row.crop[1] < label_x < row.crop[3]
    assert annotation.get_verticalalignment() == 'top'
    assert annotation.get_bbox_patch() is not None
    assert annotation.get_window_extent().y1 < roi.get_window_extent().y0
    plt.close(figure)


def test_row_labels_when_rendered_stay_left_of_data_panels(tmp_path):
    figure = render_plate(_rows(tmp_path))
    figure.canvas.draw()

    for row_index, label in enumerate(figure.texts):
        panel = figure.axes[row_index * 5].get_window_extent()
        assert label.get_window_extent().x1 < panel.x0
    plt.close(figure)


def test_plate_and_standalone_rows_use_independent_sgdfa_norms(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    _manifest(manifest)
    capture = tmp_path / 'capture'
    write_record(capture, _record(FIXED_TOKENS[0], sgdfa_scale=1.0))
    write_record(capture, _record(FIXED_TOKENS[1], sgdfa_scale=-10.0))
    rows = load_evidence(capture, load_crop_manifest(manifest)).rows[:2]
    row_limits = tuple(row_sgdfa_limit(row) for row in rows)

    plate = render_plate(rows)
    row_figures = tuple(
        render_row(row) for row in rows
    )

    plate_norms = tuple(
        (plate.axes[index].images[0].norm.vmin, plate.axes[index].images[0].norm.vmax)
        for index in (3, 8)
    )
    row_norms = tuple(
        (figure.axes[3].images[0].norm.vmin, figure.axes[3].images[0].norm.vmax)
        for figure in row_figures
    )
    assert row_norms == plate_norms
    assert tuple(norm[1] for norm in plate_norms) == row_limits
    assert plate_norms[0] != plate_norms[1]
    plt.close(plate)
    for figure in row_figures:
        plt.close(figure)


@pytest.mark.parametrize('include_alt, expected', [
    (False, {
        'figure6_refined_2row.png', 'figure6_refined_2row.pdf',
        'figure6_row_a.png', 'figure6_row_b.png', 'figure6_refine_summary.md',
    }),
    (True, {
        'figure6_refined_2row.png', 'figure6_refined_2row.pdf',
        'figure6_row_a.png', 'figure6_row_b.png', 'figure6_refine_summary.md',
        'figure6_refined_2row_alt.png', 'figure6_refined_2row_alt.pdf',
    }),
])
def test_export_when_records_are_synthetic_publishes_exact_bundle(
        tmp_path, include_alt, expected):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    crops = _manifest(manifest)
    capture = tmp_path / 'capture'
    for token in FIXED_TOKENS:
        write_record(capture, _record(token))
    corrupt = capture / 'rank_00000' / 'records' / 'corrupt'
    corrupt.mkdir(parents=True)
    (corrupt / 'metadata.json').write_text('{', encoding='utf-8')
    output = tmp_path / 'figure6'

    export_figure6(capture, manifest, output, overwrite=False, include_alt=include_alt)

    assert {path.name for path in output.iterdir()} == expected
    assert set(artifact_names(include_alt)) == expected
    for name in tuple(item for item in expected if item.endswith('.png')):
        assert (output / name).read_bytes().startswith(b'\x89PNG')
    for name in tuple(item for item in expected if item.endswith('.pdf')):
        assert (output / name).read_bytes().startswith(b'%PDF')
    summary = (output / 'figure6_refine_summary.md').read_text(encoding='utf-8')
    assert all(token in summary for token in FIXED_TOKENS[:2])
    assert FIXED_TOKENS[2] not in summary
    assert 'pre-update' in summary
    assert 'horizontal-y / vertical-x' in summary
    assert 'row-local 98th-percentile SG-DFA scale: `true`' in summary
    assert ('alternate plate included: `%s`' % str(include_alt).lower()) in summary
    assert len(crops) == 3
    with pytest.raises(FileExistsError):
        export_figure6(capture, manifest, output, overwrite=False, include_alt=include_alt)
