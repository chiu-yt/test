import json
from dataclasses import replace
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from tools.figure5_utils import final_artifacts
from tools.figure5_utils.candidates import CandidateEvaluation
from tools.figure5_utils.domain import (
    Callout, CalloutKind, CandidateEvidence, DistanceMeters, FrameRecord,
)
from tools.figure5_utils.final_rendering import FinalRenderRow, horizontal_crop
from tools.figure5_utils.final_selection import FINAL_SAMPLE_TOKENS, select_final_rows
from tools.figure5_utils.palette import DetectionClass


@pytest.fixture
def selected_rows() -> Tuple[FinalRenderRow, ...]:
    candidates = {}
    kinds = (CalloutKind.FAR_RANGE_RECOVERY, CalloutKind.SMALL_OBJECT_RECOVERY,
             CalloutKind.BETTER_LOCALIZATION)
    for token, kind in zip(reversed(FINAL_SAMPLE_TOKENS), reversed(kinds)):
        callouts = tuple(Callout(
            (10.123456789012345 + offset, -3.987654321098765,
             13.234567890123456 + offset, 1.876543210987654),
            DetectionClass.PEDESTRIAN, DistanceMeters(35.123456789012345 + offset), kind,
        ) for offset in (0.0, 2.0))
        candidates[token] = CandidateEvaluation(
            FrameRecord(token, (), (), (), ()), CandidateEvidence(), 0.0,
            (kind,), callouts,
        )
    return tuple(FinalRenderRow(selection, np.empty((0, 5)))
                 for selection in select_final_rows(candidates))


def test_manifest_preserves_selected_geometry_when_written(
        tmp_path: Path, selected_rows: Tuple[FinalRenderRow, ...]) -> None:
    """Given fractional selected geometry, when written, then floats round-trip exactly."""
    path = final_artifacts.write_crop_manifest(selected_rows, tmp_path, final_artifacts.ROW_PURPOSES)
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert [row['sample_token'] for row in payload['rows']] == list(FINAL_SAMPLE_TOKENS)
    for saved, row in zip(payload['rows'], selected_rows):
        selection = row.selection
        assert saved['row_number'] == selection.row_number
        assert saved['purpose'] == final_artifacts.ROW_PURPOSES[selection.row_number - 1]
        assert saved['square_crop'] == list(selection.crop)
        assert saved['horizontal_crop'] == list(horizontal_crop(row))
        assert saved['square_crop'] != [round(value, 3) for value in selection.crop]
        assert saved['horizontal_crop'] != [round(value, 3) for value in horizontal_crop(row)]
        assert saved['callouts'] == [dict(
            roi=list(callout.roi), kind=callout.kind.value,
            class_name=callout.class_name.value, distance_m=callout.distance_m,
        ) for callout in selection.callouts]


def test_manifest_declares_coordinates_and_layouts_when_written(
        tmp_path: Path, selected_rows: Tuple[FinalRenderRow, ...]) -> None:
    """Given selected rows, when serialized, then version and layout conventions are explicit."""
    path = final_artifacts.write_crop_manifest(selected_rows, tmp_path, final_artifacts.ROW_PURPOSES)
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert path.name == 'figure5_crop_manifest.json'
    assert payload['schema_version'] == 1
    assert payload['coordinate_conventions'] == dict(
        canonical_frame='lidar', units='m', horizontal_axis='y', vertical_axis='x',
        crop_tuple_order=['x_min', 'y_min', 'x_max', 'y_max'],
        roi_tuple_order=['x_min', 'y_min', 'x_max', 'y_max'],
    )
    assert payload['layouts'] == dict(square='figure5_final', horizontal='figure5_horizontal')
    assert path.name in final_artifacts.artifact_names(FINAL_SAMPLE_TOKENS)


def test_manifest_replaces_complete_same_directory_file_when_destination_exists(
        tmp_path: Path, selected_rows: Tuple[FinalRenderRow, ...],
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Given an old manifest, when replacement occurs, then only complete JSON is exposed."""
    destination = tmp_path / 'figure5_crop_manifest.json'
    destination.write_text('old-complete-manifest', encoding='utf-8')
    replace_path = Path.replace
    replacements = []

    def observe_replace(source: Path, target: Path) -> Path:
        assert source.parent == target.parent == tmp_path
        assert source != target == destination
        assert destination.read_text(encoding='utf-8') == 'old-complete-manifest'
        assert len(json.loads(source.read_text(encoding='utf-8'))['rows']) == 3
        replacements.append(target)
        return replace_path(source, target)

    monkeypatch.setattr(Path, 'replace', observe_replace)
    final_artifacts.write_crop_manifest(selected_rows, tmp_path, final_artifacts.ROW_PURPOSES)
    assert replacements == [destination]
    assert tuple(tmp_path.iterdir()) == (destination,)


def test_manifest_preserves_destination_when_geometry_is_nonfinite(
        tmp_path: Path, selected_rows: Tuple[FinalRenderRow, ...]) -> None:
    """Given invalid geometry, when serialized, then no partial manifest replaces the old one."""
    destination = tmp_path / 'figure5_crop_manifest.json'
    destination.write_text('old-complete-manifest', encoding='utf-8')
    selection = selected_rows[0].selection
    callout = replace(selection.callouts[0], distance_m=DistanceMeters(float('nan')))
    row = replace(selected_rows[0], selection=replace(selection, callouts=(callout,)))

    with pytest.raises(ValueError):
        final_artifacts.write_crop_manifest(
            (row,) + selected_rows[1:], tmp_path, final_artifacts.ROW_PURPOSES,
        )

    assert destination.read_text(encoding='utf-8') == 'old-complete-manifest'
    assert tuple(tmp_path.iterdir()) == (destination,)


def test_manifest_preserves_destination_when_replacement_fails(
        tmp_path: Path, selected_rows: Tuple[FinalRenderRow, ...],
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Given an existing manifest, when replacement fails, then it survives without temp debris."""
    destination = tmp_path / 'figure5_crop_manifest.json'
    destination.write_text('old-complete-manifest', encoding='utf-8')

    def fail_replace(source: Path, target: Path) -> Path:
        raise PermissionError(target)

    monkeypatch.setattr(Path, 'replace', fail_replace)
    with pytest.raises(PermissionError):
        final_artifacts.write_crop_manifest(selected_rows, tmp_path, final_artifacts.ROW_PURPOSES)
    assert destination.read_text(encoding='utf-8') == 'old-complete-manifest'
    assert tuple(tmp_path.iterdir()) == (destination,)
