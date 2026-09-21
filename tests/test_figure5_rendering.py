from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tools.figure5_utils import (
    Box3D,
    Callout,
    CalloutKind,
    Detection,
    DetectionClass,
    DetectionScore,
    DistanceMeters,
    FrameRecord,
    SampleToken,
    YawRadians,
)
from tools.figure5_utils.rendering import (
    CONTACT_SHEET_PAGE_SIZE,
    HD_DPI,
    PANEL_SIZE_INCHES,
    STANDARD_DPI,
    PanelOutputs,
    PanelSpec,
    render_comparison_panel,
    save_contact_sheets,
    save_panel_pngs,
)


def _detection(token, class_name, center):
    return Detection(
        sample_token=token,
        class_name=class_name,
        score=DetectionScore(0.9),
        box=Box3D(
            center=center,
            size=(4.0, 2.0, 1.5),
            yaw=YawRadians(0.25),
        ),
    )


def _panel(rank=1, callouts=()):
    token = SampleToken('sample-token-%03d' % rank)
    car = _detection(token, DetectionClass.CAR, (8.0, 2.0, 0.0))
    pedestrian = _detection(token, DetectionClass.PEDESTRIAN, (20.0, -5.0, 0.0))
    frame = FrameRecord(
        sample_token=token,
        ground_truth=(car, pedestrian),
        source_only=(car,),
        codemerge=(car, pedestrian),
        refuse_tta=(car, pedestrian),
    )
    points = np.array([
        [-10.0, -8.0, 0.0],
        [0.0, 0.0, 0.0],
        [12.0, 7.0, 0.0],
    ])
    return PanelSpec(frame=frame, lidar_points=points, rank=rank, callouts=callouts)


def _pixel_size(size_inches, dpi):
    return tuple(round(dimension * dpi) for dimension in size_inches)


def test_render_comparison_panel_uses_fixed_titles_and_axes_geometry():
    # Given one ranked frame with the shared LiDAR points.
    panel = _panel(rank=3)

    # When the clean comparison panel is rendered.
    figure = render_comparison_panel(panel)

    # Then all methods use the same square BEV geometry and fixed titles.
    assert [axis.get_title() for axis in figure.axes] == [
        'GT', 'Source-only', 'CodeMerge', 'ReFuse-TTA',
    ]
    assert [axis.get_xlim() for axis in figure.axes] == [(-50.0, 50.0)] * 4
    assert [axis.get_ylim() for axis in figure.axes] == [(-50.0, 50.0)] * 4
    assert all(axis.get_aspect() == 1.0 for axis in figure.axes)
    assert len({tuple(axis.get_position().bounds) for axis in figure.axes}) == 4
    assert any('Rank 03' in text.get_text() for text in figure.texts)
    assert any(str(panel.frame.sample_token) in text.get_text() for text in figure.texts)
    plt.close(figure)


def test_callout_changes_only_artists_not_canvas_or_axes_geometry():
    # Given a physical callout shared by the four method views.
    callout = Callout(
        roi=(6.0, -8.0, 8.0, 6.0),
        class_name=DetectionClass.CAR,
        distance_m=DistanceMeters(32.0),
        kind=CalloutKind.FAR_RANGE_RECOVERY,
    )
    panel = _panel(callouts=(callout,))

    # When clean and annotated variants are rendered.
    clean = render_comparison_panel(panel, show_callouts=False)
    annotated = render_comparison_panel(panel, show_callouts=True)

    # Then geometry is identical and red rectangles exist only when supplied.
    assert tuple(clean.get_size_inches()) == tuple(annotated.get_size_inches())
    assert [axis.get_position().bounds for axis in clean.axes] == [
        axis.get_position().bounds for axis in annotated.axes
    ]
    assert sum(len(axis.patches) for axis in clean.axes) == 0
    assert sum(len(axis.patches) for axis in annotated.axes) == 4
    assert all(patch.get_edgecolor()[:3] == (0.8, 0.0, 0.0)
               for axis in annotated.axes for patch in axis.patches)
    plt.close(clean)
    plt.close(annotated)


def test_save_panel_pngs_writes_standard_and_geometry_matched_hd_variants(tmp_path):
    # Given clean and callout output paths for one panel.
    outputs = PanelOutputs(
        standard=tmp_path / 'standard.png',
        hd_clean=tmp_path / 'hd-clean.png',
        hd_callout=tmp_path / 'hd-callout.png',
    )

    # When all publication variants are exported.
    saved = save_panel_pngs(_panel(), outputs)

    # Then files exist at the requested standard/HD dimensions and figures close.
    assert saved == (outputs.standard, outputs.hd_clean, outputs.hd_callout)
    assert all(path.exists() for path in saved)
    with Image.open(outputs.standard) as image:
        assert image.size == _pixel_size(PANEL_SIZE_INCHES, STANDARD_DPI)
    with Image.open(outputs.hd_clean) as clean, Image.open(outputs.hd_callout) as callout:
        assert clean.size == callout.size == _pixel_size(PANEL_SIZE_INCHES, HD_DPI)
    assert plt.get_fignums() == []


def test_contact_sheets_paginate_twelve_candidates_and_write_outputs(tmp_path):
    # Given thirteen ranked panels and a contact-sheet output stem.
    panels = tuple(_panel(rank=index + 1) for index in range(13))
    output_stem = tmp_path / 'candidates'

    # When contact sheets are composed and saved.
    saved = save_contact_sheets(panels, output_stem)

    # Then twelve candidates fill page one and the remainder starts page two.
    assert saved == (
        tmp_path / 'candidates-page-01.png',
        tmp_path / 'candidates-page-02.png',
    )
    assert all(path.exists() for path in saved)
    with Image.open(saved[0]) as first_page:
        assert first_page.size == _pixel_size(CONTACT_SHEET_PAGE_SIZE, STANDARD_DPI)
    assert plt.get_fignums() == []
