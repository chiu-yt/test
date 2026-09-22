"""Integrity-checked occurrence selection and runtime-stage resolution."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np

from pcdet.utils.figure6_artifacts import scan_records, select_occurrences
from pcdet.utils.figure6_schema import ArtifactError, CaptureRecord, StageCapture, StageState

from .contracts import Crop, CropRow


COLUMN_TITLES = (
    'LiDAR Density', 'SPCRA Reliability', 'RG-PLM Retained',
    'SG-DFA Response', 'Final Detection',
)
COLUMN_SLUGS = ('density', 'reliability', 'rgplm', 'sgdfa', 'finaldet')


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class StageEvidence:
    title: str
    slug: str
    selected_stage: str
    selected_source: str
    runtime_state: StageState
    owner: str
    detail: str
    arrays: Mapping[str, np.ndarray]
    spatial_extent: Optional[Crop]


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class TokenEvidence:
    token: str
    row_number: int
    purpose: str
    crop: Crop
    occurrence_path: Optional[Path]
    selection_key: Optional[Tuple[int, int, int, int, str]]
    alternative_count: int
    protocol: Mapping[str, str]
    stages: Tuple[StageEvidence, ...]


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class RecordProblem:
    path: Path
    detail: str


@dataclass(frozen=True)  # noqa: SLOTS_OK - package supports Python 3.8.
class EvidenceBundle:
    rows: Tuple[TokenEvidence, ...]
    issues: Tuple[RecordProblem, ...]


def _missing(title: str, slug: str, detail: str) -> StageEvidence:
    return StageEvidence(
        title, slug, 'unavailable', 'unavailable', StageState.MISSING,
        'renderer', detail, {}, None,
    )


def _from_capture(title: str, slug: str, name: str, source: str,
                  capture: Optional[StageCapture], spatial_extent: Optional[Crop]) -> StageEvidence:
    if capture is None:
        return _missing(title, slug, 'runtime stage %s was not captured' % name)
    return StageEvidence(
        title, slug, name, source, capture.status.state, capture.status.owner,
        capture.status.detail, capture.arrays, spatial_extent,
    )


def _source(record: CaptureRecord) -> str:
    source = record.protocol.get('final_pseudo_source', 'unavailable')
    if source == 'unavailable':
        return 'current_pre_update'
    return source


def _spatial_extent(record: CaptureRecord) -> Optional[Crop]:
    encoded = record.protocol.get('point_cloud_range')
    if encoded is None or encoded == 'unavailable':
        return None
    try:
        values = json.loads(encoded)
        if not isinstance(values, list) or len(values) != 6:
            raise ArtifactError('point_cloud_range protocol must contain six values')
        numeric = tuple(float(value) for value in values)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ArtifactError('invalid point_cloud_range protocol') from error
    if not all(math.isfinite(value) for value in numeric):
        raise ArtifactError('point_cloud_range protocol must be finite')
    extent = numeric[0], numeric[1], numeric[3], numeric[4]
    if extent[0] >= extent[2] or extent[1] >= extent[3]:
        raise ArtifactError('point_cloud_range protocol must be nondegenerate')
    return extent


def _resolve(record: CaptureRecord) -> Tuple[StageEvidence, ...]:
    stages = record.stages
    source = _source(record)
    spatial_extent = _spatial_extent(record)
    input_name = 'input'
    input_stage = stages.get(input_name)
    density = stages.get('input_density')
    if density is not None and density.status.state is StageState.COMPLETE:
        input_name, input_stage = 'input_density', density
    spcra_name = 'spcra.' + source
    spcra = stages.get(spcra_name)
    if spcra is None and source != 'current_pre_update':
        spcra_name = 'spcra.current_pre_update'
        spcra = stages.get(spcra_name)
    pseudo_name = 'effective_pseudo.' + source
    pseudo = stages.get(pseudo_name)
    if pseudo is None:
        pseudo_name = 'injection'
        pseudo = stages.get(pseudo_name)
    return (
        _from_capture(COLUMN_TITLES[0], COLUMN_SLUGS[0], input_name,
                      'current_pre_update', input_stage, spatial_extent),
        _from_capture(COLUMN_TITLES[1], COLUMN_SLUGS[1], spcra_name,
                      spcra.status.owner if spcra is not None else source, spcra, spatial_extent),
        _from_capture(COLUMN_TITLES[2], COLUMN_SLUGS[2], pseudo_name, source, pseudo,
                      spatial_extent),
        _from_capture(COLUMN_TITLES[3], COLUMN_SLUGS[3], 'sg_dfa',
                      'current_pre_update', stages.get('sg_dfa'), spatial_extent),
        _from_capture(COLUMN_TITLES[4], COLUMN_SLUGS[4], 'final_detection',
                      'current_pre_update', stages.get('final_detection'), spatial_extent),
    )


def load_evidence(capture_dir: Path, crops: Tuple[CropRow, ...]) -> EvidenceBundle:
    """Scan validated records and align deterministic selections to canonical crops."""
    catalog = scan_records(capture_dir)
    selections = select_occurrences(catalog.records)
    rows = []
    for crop in crops:
        selection = selections.get(crop.token)
        if selection is None:
            stages = tuple(_missing(title, slug, 'no valid occurrence selected')
                           for title, slug in zip(COLUMN_TITLES, COLUMN_SLUGS))
            rows.append(TokenEvidence(
                crop.token, crop.row_number, crop.purpose, crop.crop, None, None,
                0, {}, stages,
            ))
            continue
        located = selection.selected
        rows.append(TokenEvidence(
            crop.token, crop.row_number, crop.purpose, crop.crop, located.path,
            located.record.identity.selection_key, len(selection.alternatives),
            located.record.protocol, _resolve(located.record),
        ))
    issues = tuple(RecordProblem(issue.path, issue.detail) for issue in catalog.issues)
    return EvidenceBundle(tuple(rows), issues)
