import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np

from pcdet.utils.figure6_artifacts import write_record
from pcdet.utils.figure6_schema import CaptureRecord, FIXED_TOKENS
from test_figure6_rendering import _complete, _manifest, _record
from tools.figure6_utils.contracts import load_crop_manifest
from tools.figure6_utils.loading import load_evidence
from tools.figure6_utils.rendering import render_plate


def test_reliability_when_rendered_uses_discrete_boundary_colors_and_clipping(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    _manifest(manifest)
    capture = tmp_path / 'capture'
    source = _record(FIXED_TOKENS[0])
    low_boundary = 1.0 / 3.0
    high_boundary = 2.0 / 3.0
    reliability_values = np.array([
        -0.25, 0.0, np.nextafter(low_boundary, 0.0), low_boundary,
        np.nextafter(high_boundary, 0.0), high_boundary, 1.0, 1.25,
    ])
    boxes = np.tile(
        np.array([[12.0, -2.0, 0.5, 4.0, 1.8, 1.5, 0.1]], dtype=np.float32),
        (len(reliability_values), 1),
    )
    stages = dict(source.stages)
    stages['spcra.aggregated_pseudo_source'] = _complete(
        'aggregated_pseudo_source',
        pred_boxes=boxes,
        pred_scores=np.full(len(reliability_values), 0.75, dtype=np.float32),
        pred_labels=np.ones(len(reliability_values), dtype=np.int64),
        spcra_reliability=reliability_values,
    )
    write_record(capture, CaptureRecord(source.identity, source.protocol, stages))

    figure = render_plate((load_evidence(capture, load_crop_manifest(manifest)).rows[0],))

    colors = tuple(to_hex(line.get_color()).upper() for line in figure.axes[1].lines)
    np.testing.assert_array_equal(colors, (
        '#C62828', '#C62828', '#C62828',
        '#FDD835', '#FDD835',
        '#2E7D32', '#2E7D32', '#2E7D32',
    ))
    plt.close(figure)


def test_rgplm_when_rendered_keeps_only_valid_effective_labels_at_published_width(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    _manifest(manifest)
    capture = tmp_path / 'capture'
    pseudo = np.array([
        [12.0, -2.0, 0.5, 4.0, 1.8, 1.5, 0.1, 1.0, 0.72],
        [13.0, 0.0, 0.5, 2.0, 1.0, 1.0, 0.0, -9.0, 0.99],
        [14.0, 0.0, 0.5, 2.0, 1.0, 1.0, 0.0, 0.0, 0.99],
        [15.0, 0.0, 0.5, 2.0, 1.0, 1.0, 0.0, 1.5, 0.99],
        [16.0, 0.0, 0.5, 2.0, 1.0, 1.0, 0.0, np.nan, 0.99],
        [17.0, 0.0, 0.5, 2.0, 1.0, 1.0, 0.0, 99.0, 0.99],
    ], dtype=np.float32)
    write_record(capture, _record(FIXED_TOKENS[0], pseudo_rows=pseudo))

    figure = render_plate((load_evidence(capture, load_crop_manifest(manifest)).rows[0],))

    assert len(figure.axes[2].lines) == 1
    np.testing.assert_allclose(figure.axes[2].lines[0].get_linewidth(), 1.6)
    assert not figure.axes[2].texts
    plt.close(figure)


def test_rgplm_when_injection_is_selected_uses_class_column_nine(tmp_path):
    manifest = tmp_path / 'figure5_crop_manifest.json'
    _manifest(manifest)
    capture = tmp_path / 'capture'
    source = _record(FIXED_TOKENS[0])
    stages = dict(source.stages)
    del stages['effective_pseudo.aggregated_pseudo_source']
    stages['injection'] = _complete(
        'current_pre_update',
        gt_boxes=np.array([
            [12.0, -2.0, 0.5, 4.0, 1.8, 1.5, 0.1, 1.0, 0.72, -9.0],
            [13.0, -1.0, 0.5, 2.0, 1.0, 1.0, 0.0, -7.0, 0.01, 2.0],
        ], dtype=np.float32),
    )
    write_record(capture, CaptureRecord(source.identity, source.protocol, stages))

    figure = render_plate((load_evidence(capture, load_crop_manifest(manifest)).rows[0],))

    assert len(figure.axes[2].lines) == 1
    np.testing.assert_allclose(figure.axes[2].lines[0].get_linewidth(), 1.6)
    plt.close(figure)
