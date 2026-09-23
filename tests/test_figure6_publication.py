from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pytest

from pcdet.utils.figure6_artifacts import write_record
from pcdet.utils.figure6_schema import FIXED_TOKENS
from test_figure6_rendering import _manifest, _record
from tools.figure6_utils.contracts import Callout, load_crop_manifest
from tools.figure6_utils.export import artifact_names, export_figure6
from tools.figure6_utils.loading import load_evidence
from tools.figure6_utils.rendering import pooled_sgdfa_limit, render_plate, render_row


def _rows(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    _manifest(manifest)
    capture = tmp_path / 'capture'
    for token in FIXED_TOKENS[:2]:
        write_record(capture, _record(token))
    return load_evidence(capture, load_crop_manifest(manifest)).rows[:2]


@pytest.mark.parametrize('alternate', [False, True])
def test_plate_when_rendered_includes_fixed_reliability_key(tmp_path, alternate):
    figure = render_plate(_rows(tmp_path), alternate=alternate)

    color_axis = figure.axes[-1]
    np.testing.assert_array_equal(color_axis.get_xticks(), (0.0, 1.0))
    assert len(figure.axes) == 11
    plt.close(figure)


def test_plate_when_manifest_has_three_rois_displays_first_two_in_order(tmp_path):
    row = _rows(tmp_path)[0]
    callouts = (
        Callout((11.0, -8.0, 12.0, -7.0), 'first', 'car', 10.0),
        Callout((13.0, -6.0, 14.0, -5.0), 'second', 'car', 20.0),
        Callout((15.0, -4.0, 16.0, -3.0), 'third', 'car', 30.0),
    )

    figure = render_plate((replace(row, callouts=callouts),))

    for axis in figure.axes[:5]:
        assert len(axis.patches) == 2
        assert tuple(
            (patch.get_x(), patch.get_y())
            for patch in axis.patches if isinstance(patch, Rectangle)
        ) == ((-8.0, 11.0), (-6.0, 13.0))
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

    figure = render_plate((replace(row, stages=stages, callouts=callouts),))

    annotations = figure.axes[1].texts
    assert tuple(text.get_text() for text in annotations) == ('r=0.10', 'r=0.90')
    assert tuple(text.get_position() for text in annotations) == ((-2.0, 12.0), (-1.0, 13.0))

    alternate = render_plate((replace(row, stages=stages, callouts=callouts),), alternate=True)
    assert not alternate.axes[1].texts
    assert len(alternate.axes[1].lines) == len(figure.axes[1].lines)
    assert len(alternate.axes[1].patches) == len(figure.axes[1].patches) == 2
    plt.close(figure)
    plt.close(alternate)


def test_row_labels_when_rendered_stay_left_of_data_panels(tmp_path):
    figure = render_plate(_rows(tmp_path))
    figure.canvas.draw()

    for row_index, label in enumerate(figure.texts):
        panel = figure.axes[row_index * 5].get_window_extent()
        assert label.get_window_extent().x1 < panel.x0
    plt.close(figure)


def test_standalone_rows_use_same_pooled_sgdfa_norm_as_main_plate(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    _manifest(manifest)
    capture = tmp_path / 'capture'
    write_record(capture, _record(FIXED_TOKENS[0], sgdfa_scale=1.0))
    write_record(capture, _record(FIXED_TOKENS[1], sgdfa_scale=-10.0))
    rows = load_evidence(capture, load_crop_manifest(manifest)).rows[:2]
    limit = pooled_sgdfa_limit(rows)

    plate = render_plate(rows, sgdfa_limit=limit)
    row_figures = tuple(render_row(row, limit) for row in rows)

    plate_norms = tuple(
        (plate.axes[index].images[0].norm.vmin, plate.axes[index].images[0].norm.vmax)
        for index in (3, 8)
    )
    row_norms = tuple(
        (figure.axes[3].images[0].norm.vmin, figure.axes[3].images[0].norm.vmax)
        for figure in row_figures
    )
    assert row_norms == plate_norms
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
    assert 'pooled two-row 98th-percentile scale: `true`' in summary
    assert ('alternate plate included: `%s`' % str(include_alt).lower()) in summary
    assert len(crops) == 3
    with pytest.raises(FileExistsError):
        export_figure6(capture, manifest, output, overwrite=False, include_alt=include_alt)
