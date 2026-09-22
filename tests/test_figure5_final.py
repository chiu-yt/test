import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tools.figure5_utils.candidates import CandidateEvaluation
from tools.figure5_utils.domain import (
    Box3D, Callout, CalloutKind, CandidateEvidence, Detection, DetectionScore,
    DistanceMeters, FrameRecord, SampleToken, YawRadians,
)
from tools.figure5_utils.final_artifacts import (
    artifact_names, publish_final_artifacts, write_status_manifest,
)
from tools.figure5_utils import final_artifacts, final_rendering
from tools.figure5_utils.final_rendering import FinalRenderRow, render_final_plate, render_final_row
from tools.figure5_utils.final_selection import (
    FINAL_SAMPLE_TOKENS, FinalSelectionError, select_final_rows,
)
from tools.figure5_utils.palette import DetectionClass


def _detection(token, center):
    return Detection(
        SampleToken(token), DetectionClass.PEDESTRIAN, DetectionScore(0.9),
        Box3D(center, (2.0, 1.0, 1.7), YawRadians(0.0)),
    )


def _candidate(token, callouts):
    detection = _detection(token, (35.0, 0.0, 0.0))
    frame = FrameRecord(SampleToken(token), (detection,), (), (), (detection,))
    return CandidateEvaluation(frame, CandidateEvidence(), 0.0,
                               tuple(item.kind for item in callouts), tuple(callouts))


def _callout(kind, roi, distance=35.0):
    return Callout(roi, DetectionClass.PEDESTRIAN, DistanceMeters(distance), kind)


def _valid_candidates(row3_kind=CalloutKind.BETTER_LOCALIZATION):
    return {
        FINAL_SAMPLE_TOKENS[0]: _candidate(FINAL_SAMPLE_TOKENS[0], (
            _callout(CalloutKind.FAR_RANGE_RECOVERY, (30.0, -4.0, 38.0, 4.0)),
        )),
        FINAL_SAMPLE_TOKENS[1]: _candidate(FINAL_SAMPLE_TOKENS[1], (
            _callout(CalloutKind.SMALL_OBJECT_RECOVERY, (8.0, 12.0, 12.0, 16.0)),
        )),
        FINAL_SAMPLE_TOKENS[2]: _candidate(FINAL_SAMPLE_TOKENS[2], (
            _callout(row3_kind, (-12.0, -10.0, -5.0, -3.0)),
        )),
    }


def _render_rows():
    points = np.array([[35.0, 0.0, 0.0, 1.0, 0.0]])
    return tuple(FinalRenderRow(selection, points)
                 for selection in select_final_rows(_valid_candidates()))


def test_final_rows_keep_fixed_order_categories_and_square_crops():
    # Given all fixed tokens with evidence for their declared row purpose.
    candidates = _valid_candidates()

    # When final rows are selected without ranking.
    rows = select_final_rows(candidates)

    # Then order, evidence category, and bounded adaptive crop are fixed.
    assert tuple(row.frame.sample_token for row in rows) == FINAL_SAMPLE_TOKENS
    assert tuple(row.callouts[0].kind for row in rows) == (
        CalloutKind.FAR_RANGE_RECOVERY,
        CalloutKind.SMALL_OBJECT_RECOVERY,
        CalloutKind.BETTER_LOCALIZATION,
    )
    for index, row in enumerate(rows):
        x_min, y_min, x_max, y_max = row.crop
        assert x_max - x_min == pytest.approx(y_max - y_min)
        assert x_max - x_min >= (40.0 if index == 0 else 30.0)
        assert -50.0 <= x_min < x_max <= 50.0
        assert -50.0 <= y_min < y_max <= 50.0


def test_final_rows_limit_visible_callouts_and_fallback_to_fp_reduction():
    # Given nearer visible evidence, excess visible evidence, and an off-canvas annotation.
    candidates = _valid_candidates(CalloutKind.FALSE_POSITIVE_REDUCTION)
    first = candidates[FINAL_SAMPLE_TOKENS[0]]
    candidates[FINAL_SAMPLE_TOKENS[0]] = CandidateEvaluation(
        first.frame, first.evidence, first.score, first.memberships,
        first.available_callouts + (
            _callout(CalloutKind.FAR_RANGE_RECOVERY, (40.0, 1.0, 45.0, 5.0), 45.0),
            _callout(CalloutKind.FAR_RANGE_RECOVERY, (20.0, 1.0, 25.0, 5.0), 25.0),
            _callout(CalloutKind.FAR_RANGE_RECOVERY, (60.0, 1.0, 65.0, 5.0), 65.0),
        ),
    )

    # When rows are selected.
    rows = select_final_rows(candidates)

    # Then only the two farthest visible row-one ROIs survive and row three uses honest fallback.
    assert [item.distance_m for item in rows[0].callouts] == [45.0, 35.0]
    assert rows[2].callouts[0].kind is CalloutKind.FALSE_POSITIVE_REDUCTION
    assert rows[1].callouts[0].roi == pytest.approx((7.25, 11.25, 12.75, 16.75))


@pytest.mark.parametrize('roi', [(55.0, 0.0, 60.0, 5.0), (49.0, 0.0, 55.0, 5.0)])
def test_final_rows_fail_when_required_visible_evidence_is_absent(roi):
    # Given the fixed far-range frame with only an off-canvas far-range ROI.
    candidates = _valid_candidates()
    first = candidates[FINAL_SAMPLE_TOKENS[0]]
    candidates[FINAL_SAMPLE_TOKENS[0]] = CandidateEvaluation(
        first.frame, first.evidence, first.score, first.memberships,
        (_callout(CalloutKind.FAR_RANGE_RECOVERY, roi),),
    )

    # When final selection is attempted, then unsupported callouts are not invented.
    with pytest.raises(FinalSelectionError):
        select_final_rows(candidates)


def test_final_plate_has_top_only_titles_shared_row_geometry_and_callout_artists():
    # Given three selected rows using one point array and crop per row.
    rows = _render_rows()
    selections = tuple(row.selection for row in rows)

    # When clean and callout final plates are rendered.
    clean = render_final_plate(rows, show_callouts=False)
    callout = render_final_plate(rows, show_callouts=True)

    # Then the 3x4 artist, title, and crop contracts remain publication-clean.
    assert len(clean.axes) == len(callout.axes) == 12
    assert [axis.get_title() for axis in clean.axes] == [
        'GT', 'Source-only', 'CodeMerge', 'ReFuse-TTA',
    ] + [''] * 8
    assert all('Rank' not in text.get_text() for text in clean.texts)
    assert all(str(token) not in text.get_text()
               for token in FINAL_SAMPLE_TOKENS for text in clean.texts)
    for row_index, selection in enumerate(selections):
        axes = clean.axes[row_index * 4:(row_index + 1) * 4]
        expected_xlim = (selection.crop[1], selection.crop[3])
        expected_ylim = (selection.crop[0], selection.crop[2])
        assert [axis.get_xlim() for axis in axes] == [expected_xlim] * 4
        assert [axis.get_ylim() for axis in axes] == [expected_ylim] * 4
    assert sum(len(axis.patches) for axis in clean.axes) == 0
    assert sum(len(axis.patches) for axis in callout.axes) == 12
    row_figure = render_final_row(rows[0])
    assert sum(len(axis.patches) for axis in row_figure.axes) == 4
    plt.close(clean)
    plt.close(callout)
    plt.close(row_figure)


def test_horizontal_crop_excludes_unrelated_box_and_includes_boundary_crossing_box():
    # Given an inside box, an oriented boundary-crossing box, and a fully unrelated box.
    selection = select_final_rows(_valid_candidates())[0]
    token = str(selection.frame.sample_token)
    inside = _detection(token, (35.0, 0.0, 0.0))
    unrelated = _detection(token, (-35.0, 40.0, 0.0))
    boundary = replace(
        inside,
        box=Box3D(
            (selection.crop[0] - 1.5, 0.0, 0.0),
            (4.0, 2.0, 1.7), YawRadians(np.pi / 12.0),
        ),
    )
    frame_with_unrelated = replace(
        selection.frame,
        ground_truth=(inside,), source_only=(unrelated,),
        codemerge=(boundary,), refuse_tta=(inside,),
    )
    frame_without_unrelated = replace(frame_with_unrelated, source_only=())
    row = FinalRenderRow(replace(selection, frame=frame_with_unrelated),
                         np.zeros((0, 5), dtype=float))
    reference_row = replace(row, selection=replace(selection, frame=frame_without_unrelated))
    row_without_boundary = replace(
        row, selection=replace(selection, frame=replace(frame_without_unrelated, codemerge=())),
    )

    # When the rendering-only crops are computed.
    crop = final_rendering.horizontal_crop(row)
    reference_crop = final_rendering.horizontal_crop(reference_row)

    # Then the unrelated box changes nothing while the crossing box contributes with 4 m context.
    boundary_corners = final_rendering._box_bev_corners(boundary)
    x_min, y_min, x_max, y_max = crop
    assert crop == pytest.approx(reference_crop)
    assert crop != pytest.approx(final_rendering.horizontal_crop(row_without_boundary))
    assert x_min <= float(boundary_corners[:, 0].min()) - 4.0
    assert (y_max - y_min) / (x_max - x_min) == pytest.approx(1.8)
    assert row.selection.crop == selection.crop
    assert row.selection.callouts == selection.callouts


def test_horizontal_crop_derives_from_legacy_crop_when_row_has_no_content():
    # Given an otherwise valid row with no boxes or callout ROI.
    selection = select_final_rows(_valid_candidates())[0]
    empty_frame = replace(
        selection.frame, ground_truth=(), source_only=(), codemerge=(), refuse_tta=(),
    )
    empty_selection = replace(selection, frame=empty_frame, callouts=())
    row = FinalRenderRow(empty_selection, np.zeros((0, 5), dtype=float))

    # When the horizontal crop falls back to the existing selection crop.
    crop = final_rendering.horizontal_crop(row)

    # Then it contains the legacy crop and expands, rather than shrinking, to the target ratio.
    assert crop[0] <= empty_selection.crop[0]
    assert crop[1] <= empty_selection.crop[1]
    assert crop[2] >= empty_selection.crop[2]
    assert crop[3] >= empty_selection.crop[3]
    assert (crop[3] - crop[1]) / (crop[2] - crop[0]) == pytest.approx(1.8)


def test_horizontal_plate_has_exact_canvas_shared_crops_titles_and_callout_behavior():
    # Given the same fixed three rows used by the legacy renderer.
    rows = _render_rows()
    selections = tuple(row.selection for row in rows)

    # When clean and callout horizontal plates are rendered.
    clean = final_rendering.render_horizontal_plate(rows, show_callouts=False)
    callout = final_rendering.render_horizontal_plate(rows, show_callouts=True)

    # Then both are compact 3x4 publication plates with one crop per row and top-only titles.
    assert tuple(clean.get_size_inches()) == pytest.approx((7.2, 3.6))
    assert len(clean.axes) == len(callout.axes) == 12
    assert [axis.get_title() for axis in clean.axes] == [
        'GT', 'Source-only', 'CodeMerge', 'ReFuse-TTA',
    ] + [''] * 8
    clean.canvas.draw()
    assert all(axis.get_window_extent().width / axis.get_window_extent().height
               == pytest.approx(1.8, rel=0.01) for axis in clean.axes)
    for row_index, row in enumerate(rows):
        crop = final_rendering.horizontal_crop(row)
        axes = clean.axes[row_index * 4:(row_index + 1) * 4]
        callout_axes = callout.axes[row_index * 4:(row_index + 1) * 4]
        assert [axis.get_xlim() for axis in axes] == [(crop[1], crop[3])] * 4
        assert [axis.get_ylim() for axis in axes] == [(crop[0], crop[2])] * 4
        assert [axis.get_xlim() for axis in callout_axes] == [(crop[1], crop[3])] * 4
        assert [axis.get_ylim() for axis in callout_axes] == [(crop[0], crop[2])] * 4
    assert sum(len(axis.patches) for axis in clean.axes) == 0
    assert sum(len(axis.patches) for axis in callout.axes) == 12
    assert all(row.selection is selection for row, selection in zip(rows, selections))
    plt.close(clean)
    plt.close(callout)


def test_summary_records_computed_horizontal_crop_for_each_row(tmp_path: Path):
    # Given the fixed final rows.
    rows = _render_rows()

    # When the reproducibility summary is written.
    final_artifacts._write_summary(rows, tmp_path, supplied_points=False)
    summary = (tmp_path / final_artifacts.FIGURE5_FILES[-1]).read_text(encoding='utf-8')

    # Then every row discloses its computed horizontal crop coordinates.
    for row in rows:
        expected = tuple(round(value, 3) for value in final_rendering.horizontal_crop(row))
        assert 'horizontal_crop=%s' % (expected,) in summary


def test_final_artifact_names_and_manifest_are_exact_and_honest(tmp_path: Path):
    # Given the fixed three-frame output contract.
    names = artifact_names(FINAL_SAMPLE_TOKENS)

    # When the Figure 6 status manifest is written for reconstructed points.
    manifest_path = write_status_manifest(tmp_path, FINAL_SAMPLE_TOKENS, supplied_points=False)
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))

    # Then all exact Figure 5/6 names exist in the contract and runtime maps stay declarative.
    assert names[:8] == (
        'figure5_final_clean.png', 'figure5_final_clean.pdf',
        'figure5_final_callout.png', 'figure5_final_callout.pdf',
        'figure5_row1.png', 'figure5_row2.png', 'figure5_row3.png',
        'figure5_refine_summary.md',
    )
    assert final_artifacts.HORIZONTAL_FIGURE5_FILES == (
        'figure5_horizontal_clean.png', 'figure5_horizontal_callout.png',
        'figure5_horizontal_clean.pdf', 'figure5_horizontal_callout.pdf',
    )
    assert all(name in names for name in final_artifacts.HORIZONTAL_FIGURE5_FILES)
    assert manifest_path.name == 'figure6_status_manifest.json'
    assert payload['point_provenance']['exact_dataloader_replay'] is False
    for token in FINAL_SAMPLE_TOKENS:
        assert payload['artifacts']['density_%s.png' % token]['status'] == 'generated'
        assert payload['artifacts']['finaldet_%s.png' % token]['status'] == 'generated'
        for stem in ('reliability', 'rgplm', 'sgdfa'):
            filename = '%s_%s.png' % (stem, token)
            assert filename in names
            assert payload['artifacts'][filename]['status'] == 'requires_runtime_capture'
            assert not (tmp_path / filename).exists()


def test_supplied_sparse_arrays_do_not_certify_exact_dataloader_replay(tmp_path: Path):
    # Given token-named arrays without an independently verified run manifest.
    manifest_path = write_status_manifest(tmp_path, FINAL_SAMPLE_TOKENS, supplied_points=True)

    # When their provenance is serialized.
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))

    # Then file supply is recorded without claiming exact evaluation replay.
    assert payload['point_provenance']['mode'] == 'supplied_sparse_npy_unverified'
    assert payload['point_provenance']['exact_dataloader_replay'] is False


def test_final_publication_requires_overwrite_and_replaces_owned_files_atomically(tmp_path: Path):
    # Given one fully staged file and an existing same-name output.
    staging = tmp_path / 'staging'
    output = tmp_path / 'output'
    staging.mkdir()
    output.mkdir()
    (staging / 'figure5_final_clean.png').write_bytes(b'new-complete-file')
    destination = output / 'figure5_final_clean.png'
    destination.write_bytes(b'old-complete-file')

    # When publication is attempted without and then with explicit overwrite.
    with pytest.raises(FileExistsError):
        publish_final_artifacts(staging, output, overwrite=False)
    assert destination.read_bytes() == b'old-complete-file'
    publish_final_artifacts(staging, output, overwrite=True)

    # Then replacement exposes only the complete staged bytes.
    assert destination.read_bytes() == b'new-complete-file'


def test_final_publication_rejects_existing_runtime_capture_even_with_overwrite(tmp_path: Path):
    # Given a staged final figure and an existing runtime-only artifact.
    staging = tmp_path / 'staging'
    output = tmp_path / 'output'
    staging.mkdir()
    output.mkdir()
    (staging / 'figure5_final_clean.png').write_bytes(b'new-figure')
    runtime = output / ('reliability_%s.png' % FINAL_SAMPLE_TOKENS[0])
    runtime.write_bytes(b'captured-runtime-evidence')

    # When publication is requested with overwrite enabled, then runtime evidence blocks reuse.
    with pytest.raises(FileExistsError):
        publish_final_artifacts(staging, output, overwrite=True)
    assert runtime.read_bytes() == b'captured-runtime-evidence'
    assert not (output / 'figure5_final_clean.png').exists()
